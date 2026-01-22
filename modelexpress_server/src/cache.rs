// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::time::{Duration as TokioDuration, interval};
use tracing::{debug, error, info, warn};

use crate::database::{ModelDatabase, ModelRecord};
use modelexpress_common::config::DurationConfig;
use modelexpress_common::models::ModelStatus;

/// Configuration for cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEvictionConfig {
    /// Whether cache eviction is enabled
    pub enabled: bool,
    /// The eviction policy to use
    pub policy: EvictionPolicyType,
    /// How often to run the eviction process (accepts duration strings like "2h", "30m", "45s")
    pub check_interval: DurationConfig,
}

impl Default for CacheEvictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            policy: EvictionPolicyType::Lru(LruConfig::default()),
            check_interval: DurationConfig::hours(1), // Default: check every hour
        }
    }
}

/// Available cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EvictionPolicyType {
    /// Least Recently Used policy
    Lru(LruConfig),
}

/// Configuration for LRU eviction policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LruConfig {
    /// Time threshold before an unused model is eligible for removal
    pub unused_threshold: DurationConfig,
    /// Maximum number of models to keep (None = no limit based on count)
    pub max_models: Option<u32>,
    /// Minimum free disk space to maintain (in bytes, None = no disk space checks)
    pub min_free_space_bytes: Option<u64>,
}

impl Default for LruConfig {
    fn default() -> Self {
        Self {
            unused_threshold: DurationConfig::new(Duration::days(7)), // Default: 7 days
            max_models: None,
            min_free_space_bytes: None,
        }
    }
}

/// Result of a cache eviction operation
#[derive(Debug, Clone)]
pub struct EvictionResult {
    /// Number of models that were evicted
    pub evicted_count: u32,
    /// List of model names that were evicted
    pub evicted_models: Vec<String>,
    /// Total size freed (if available)
    pub bytes_freed: Option<u64>,
    /// Reason for eviction
    pub reason: EvictionReason,
}

/// Reason for cache eviction
#[derive(Debug, Clone)]
pub enum EvictionReason {
    /// Models exceeded unused time threshold
    TimeThreshold,
    /// Too many models (count limit)
    CountLimit,
    /// Insufficient disk space
    DiskSpace,
    /// Manual eviction requested
    Manual,
}

/// Trait for implementing different eviction policies
#[async_trait::async_trait]
pub trait EvictionPolicyTrait {
    /// Determine which models should be evicted based on the policy
    async fn select_for_eviction(
        &self,
        models: &[ModelRecord],
        config: &CacheEvictionConfig,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;
}

/// LRU (Least Recently Used) eviction policy implementation
pub struct LruEvictionPolicy;

impl LruEvictionPolicy {
    /// Check if a model should be evicted based on time threshold
    fn is_time_expired(model: &ModelRecord, threshold: &DurationConfig) -> bool {
        let threshold_duration = threshold.as_chrono_duration();
        let cutoff_time = match Utc::now().checked_sub_signed(threshold_duration) {
            Some(time) => time,
            None => Utc::now(),
        };
        model.last_used_at < cutoff_time
    }

    /// Get disk space information for the models directory
    async fn get_disk_space_info() -> Option<(u64, u64)> {
        // This is a placeholder - in a real implementation you would:
        // 1. Check the actual models directory path
        // 2. Use statvfs or similar to get actual disk space
        // For now, we'll return None to indicate disk space checking is not implemented
        None
    }
}

#[async_trait::async_trait]
impl EvictionPolicyTrait for LruEvictionPolicy {
    async fn select_for_eviction(
        &self,
        models: &[ModelRecord],
        config: &CacheEvictionConfig,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let EvictionPolicyType::Lru(lru_config) = &config.policy;

        let mut candidates_for_eviction = Vec::new();

        // Filter models that are eligible for eviction (only DOWNLOADED models)
        let downloaded_models: Vec<&ModelRecord> = models
            .iter()
            .filter(|model| model.status == ModelStatus::DOWNLOADED)
            .collect();

        debug!(
            "Evaluating {downloaded_count} downloaded models for eviction",
            downloaded_count = downloaded_models.len()
        );

        // 1. Check time-based eviction
        for model in &downloaded_models {
            if Self::is_time_expired(model, &lru_config.unused_threshold) {
                debug!(
                    "Model '{model_name}' is expired (last used: {last_used_at})",
                    model_name = model.model_name,
                    last_used_at = model.last_used_at
                );
                candidates_for_eviction.push(model.model_name.clone());
            }
        }

        // 2. Check count-based eviction
        if let Some(max_models) = lru_config.max_models {
            let models_to_remove_by_count =
                downloaded_models.len().saturating_sub(max_models as usize);
            if models_to_remove_by_count > 0 {
                debug!(
                    "Need to remove {models_to_remove_by_count} models due to count limit (have: {downloaded_count}, max: {max_models})",
                    models_to_remove_by_count = models_to_remove_by_count,
                    downloaded_count = downloaded_models.len(),
                    max_models = max_models
                );

                // Sort by last_used_at (oldest first) and take the oldest models
                let mut sorted_models = downloaded_models.clone();
                sorted_models.sort_by_key(|model| model.last_used_at);

                for model in sorted_models.iter().take(models_to_remove_by_count) {
                    if !candidates_for_eviction.contains(&model.model_name) {
                        candidates_for_eviction.push(model.model_name.clone());
                    }
                }
            }
        }

        // 3. Check disk space-based eviction (if configured and implemented)
        if let Some(_min_free_space) = lru_config.min_free_space_bytes
            && let Some((_total_space, _free_space)) = Self::get_disk_space_info().await
        {
            // This is where we would implement disk space checking
            // For now, we'll log that it's not implemented
            debug!("Disk space checking is not yet implemented");
        }

        debug!(
            "Selected {evicted_count} models for eviction: {candidates:?}",
            evicted_count = candidates_for_eviction.len(),
            candidates = candidates_for_eviction
        );

        Ok(candidates_for_eviction)
    }
}

/// Background service that manages cache eviction
pub struct CacheEvictionService {
    database: ModelDatabase,
    config: CacheEvictionConfig,
}

impl CacheEvictionService {
    /// Create a new cache eviction service
    pub fn new(database: ModelDatabase, config: CacheEvictionConfig) -> Self {
        Self { database, config }
    }

    /// Start the background eviction service
    pub async fn start(
        self,
        mut shutdown_receiver: tokio::sync::oneshot::Receiver<()>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.config.enabled {
            info!("Cache eviction service is disabled");
            return Ok(());
        }

        info!(
            "Starting cache eviction service with policy: {policy:?}, check interval: {interval}s",
            policy = self.config.policy,
            interval = self.config.check_interval.num_seconds()
        );

        let mut interval_timer = interval(TokioDuration::from_secs(
            self.config.check_interval.num_seconds() as u64,
        ));

        loop {
            tokio::select! {
                _ = interval_timer.tick() => {
                    if let Err(e) = self.run_eviction_cycle().await {
                        error!("Error during cache eviction cycle: {e}", e = e);
                    }
                }
                _ = &mut shutdown_receiver => {
                    info!("Cache eviction service received shutdown signal");
                    break;
                }
            }
        }

        info!("Cache eviction service stopped");
        Ok(())
    }

    /// Run a single eviction cycle
    async fn run_eviction_cycle(
        &self,
    ) -> Result<EvictionResult, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Starting cache eviction cycle");

        // Get all models from the database
        let models = self.database.get_models_by_last_used(None)?;
        debug!(
            "Found {total_models} total models in database",
            total_models = models.len()
        );

        // Select models for eviction based on the configured policy
        let models_to_evict = match &self.config.policy {
            EvictionPolicyType::Lru(_) => {
                let lru_policy = LruEvictionPolicy;
                lru_policy
                    .select_for_eviction(&models, &self.config)
                    .await?
            }
        };

        let evicted_count = models_to_evict.len() as u32;

        if evicted_count == 0 {
            debug!("No models selected for eviction");
            return Ok(EvictionResult {
                evicted_count: 0,
                evicted_models: Vec::new(),
                bytes_freed: None,
                reason: EvictionReason::TimeThreshold,
            });
        }

        info!(
            "Evicting {evicted_count} models: {models:?}",
            evicted_count = evicted_count,
            models = models_to_evict
        );

        // Remove models from the database and filesystem
        let mut successfully_evicted = Vec::new();
        for model_name in &models_to_evict {
            match self.evict_model(model_name).await {
                Ok(()) => {
                    successfully_evicted.push(model_name.clone());
                    info!(
                        "Successfully evicted model: {model_name}",
                        model_name = model_name
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to evict model '{model_name}': {e}",
                        model_name = model_name,
                        e = e
                    );
                }
            }
        }

        let result = EvictionResult {
            evicted_count: successfully_evicted.len() as u32,
            evicted_models: successfully_evicted,
            bytes_freed: None, // Could be implemented with actual file size tracking
            reason: EvictionReason::TimeThreshold,
        };

        if result.evicted_count > 0 {
            info!(
                "Cache eviction cycle completed: {evicted_count} models evicted",
                evicted_count = result.evicted_count
            );
        } else {
            debug!("Cache eviction cycle completed: no models evicted");
        }

        Ok(result)
    }

    /// Evict a single model (remove from database and filesystem)
    async fn evict_model(
        &self,
        model_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Remove from database first
        self.database.delete_model(model_name)?;

        // Remove from filesystem
        // This is where you would implement actual file removal
        // For now, we'll just log the action since the download module
        // would need to be consulted for the actual file paths
        debug!(
            "Would remove model files for: {model_name}",
            model_name = model_name
        );

        // In a real implementation, you would:
        // 1. Get the model file path from the download module
        // 2. Remove the model files from disk
        // 3. Update any in-memory caches

        Ok(())
    }

    /// Manually trigger eviction for specific models
    pub async fn manual_evict(
        &self,
        model_names: &[String],
    ) -> Result<EvictionResult, Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Manual eviction requested for models: {models:?}",
            models = model_names
        );

        let mut successfully_evicted = Vec::new();
        for model_name in model_names {
            match self.evict_model(model_name).await {
                Ok(()) => {
                    successfully_evicted.push(model_name.clone());
                    info!(
                        "Successfully evicted model: {model_name}",
                        model_name = model_name
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to evict model '{model_name}': {e}",
                        model_name = model_name,
                        e = e
                    );
                }
            }
        }

        Ok(EvictionResult {
            evicted_count: successfully_evicted.len() as u32,
            evicted_models: successfully_evicted,
            bytes_freed: None,
            reason: EvictionReason::Manual,
        })
    }

    /// Get statistics about the current cache state
    pub async fn get_cache_stats(
        &self,
    ) -> Result<CacheStats, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.database.get_models_by_last_used(None)?;
        let (downloading, downloaded, error) = self.database.get_status_counts()?;

        let _now = Utc::now();
        let mut oldest_model: Option<DateTime<Utc>> = None;
        let mut newest_model: Option<DateTime<Utc>> = None;

        for model in &models {
            if model.status == ModelStatus::DOWNLOADED {
                if oldest_model.is_none_or(|oldest| model.last_used_at < oldest) {
                    oldest_model = Some(model.last_used_at);
                }
                if newest_model.is_none_or(|newest| model.last_used_at > newest) {
                    newest_model = Some(model.last_used_at);
                }
            }
        }

        Ok(CacheStats {
            total_models: models.len() as u32,
            downloading_models: downloading,
            downloaded_models: downloaded,
            error_models: error,
            oldest_model_last_used: oldest_model,
            newest_model_last_used: newest_model,
        })
    }
}

/// Statistics about the current cache state
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub total_models: u32,
    pub downloading_models: u32,
    pub downloaded_models: u32,
    pub error_models: u32,
    pub oldest_model_last_used: Option<DateTime<Utc>>,
    pub newest_model_last_used: Option<DateTime<Utc>>,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::database::ModelDatabase;
    use modelexpress_common::models::ModelProvider;
    use tempfile::TempDir;

    fn create_test_database() -> (ModelDatabase, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let db_path = temp_dir.path().join("test_models.db");
        let db = ModelDatabase::new(db_path.to_str().expect("Invalid path"))
            .expect("Failed to create test database");
        (db, temp_dir)
    }

    #[test]
    fn test_default_config() {
        let config = CacheEvictionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.check_interval.num_seconds(), 3600);
        assert!(matches!(config.policy, EvictionPolicyType::Lru(_)));
    }

    #[test]
    fn test_lru_config_defaults() {
        let lru_config = LruConfig::default();
        assert_eq!(lru_config.unused_threshold.num_seconds(), 7 * 24 * 3600);
        assert!(lru_config.max_models.is_none());
        assert!(lru_config.min_free_space_bytes.is_none());
    }

    #[test]
    fn test_duration_config_parsing() {
        use modelexpress_common::config::parse_duration_string;

        // Test string parsing
        let json = r#"{"enabled": true, "policy": {"type": "lru", "unused_threshold": "7d"}, "check_interval": "2h"}"#;
        let config: CacheEvictionConfig =
            serde_json::from_str(json).expect("Failed to parse config");
        assert_eq!(config.check_interval.num_seconds(), 2 * 3600); // 2 hours

        // Test number parsing (seconds)
        let json = r#"{"enabled": true, "policy": {"type": "lru", "unused_threshold": 604800}, "check_interval": 1800}"#;
        let config: CacheEvictionConfig =
            serde_json::from_str(json).expect("Failed to parse config");
        assert_eq!(config.check_interval.num_seconds(), 1800); // 30 minutes

        // Test various duration formats
        assert_eq!(
            parse_duration_string("30m")
                .expect("Failed to parse 30m")
                .num_seconds(),
            30 * 60
        );
        assert_eq!(
            parse_duration_string("45s")
                .expect("Failed to parse 45s")
                .num_seconds(),
            45
        );
        assert_eq!(
            parse_duration_string("1d")
                .expect("Failed to parse 1d")
                .num_seconds(),
            24 * 3600
        );
        assert_eq!(
            parse_duration_string("2h30m")
                .expect("Failed to parse 2h30m")
                .num_seconds(),
            2 * 3600 + 30 * 60
        );
    }

    #[test]
    fn test_is_time_expired() {
        let now = Utc::now();

        // Create a model that was last used 8 days ago
        let old_model = ModelRecord {
            model_name: "old-model".to_string(),
            provider: ModelProvider::HuggingFace,
            status: ModelStatus::DOWNLOADED,
            created_at: now - Duration::days(10),
            last_used_at: now - Duration::days(8),
            message: None,
        };

        // Create a model that was last used 5 days ago
        let recent_model = ModelRecord {
            model_name: "recent-model".to_string(),
            provider: ModelProvider::HuggingFace,
            status: ModelStatus::DOWNLOADED,
            created_at: now - Duration::days(6),
            last_used_at: now - Duration::days(5),
            message: None,
        };

        let threshold = DurationConfig::new(Duration::days(7)); // 7 days

        assert!(LruEvictionPolicy::is_time_expired(&old_model, &threshold));
        assert!(!LruEvictionPolicy::is_time_expired(
            &recent_model,
            &threshold
        ));
    }

    #[tokio::test]
    async fn test_lru_eviction_policy_time_based() {
        let now = Utc::now();

        let models = vec![
            ModelRecord {
                model_name: "old-model".to_string(),
                provider: ModelProvider::HuggingFace,
                status: ModelStatus::DOWNLOADED,
                created_at: now - Duration::days(10),
                last_used_at: now - Duration::days(8),
                message: None,
            },
            ModelRecord {
                model_name: "recent-model".to_string(),
                provider: ModelProvider::HuggingFace,
                status: ModelStatus::DOWNLOADED,
                created_at: now - Duration::days(6),
                last_used_at: now - Duration::days(5),
                message: None,
            },
            ModelRecord {
                model_name: "downloading-model".to_string(),
                provider: ModelProvider::HuggingFace,
                status: ModelStatus::DOWNLOADING,
                created_at: now - Duration::days(10),
                last_used_at: now - Duration::days(8),
                message: None,
            },
        ];

        let config = CacheEvictionConfig {
            enabled: true,
            policy: EvictionPolicyType::Lru(LruConfig {
                unused_threshold: DurationConfig::new(Duration::days(7)), // 7 days
                max_models: None,
                min_free_space_bytes: None,
            }),
            check_interval: DurationConfig::hours(1),
        };

        let policy = LruEvictionPolicy;
        let evicted = policy
            .select_for_eviction(&models, &config)
            .await
            .expect("Failed to select models for eviction");

        // Should only evict the old downloaded model, not the downloading one
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], "old-model");
    }

    #[tokio::test]
    async fn test_lru_eviction_policy_count_based() {
        let now = Utc::now();

        let models = vec![
            ModelRecord {
                model_name: "model1".to_string(),
                provider: ModelProvider::HuggingFace,
                status: ModelStatus::DOWNLOADED,
                created_at: now - Duration::days(3),
                last_used_at: now - Duration::days(3),
                message: None,
            },
            ModelRecord {
                model_name: "model2".to_string(),
                provider: ModelProvider::HuggingFace,
                status: ModelStatus::DOWNLOADED,
                created_at: now - Duration::days(2),
                last_used_at: now - Duration::days(2),
                message: None,
            },
            ModelRecord {
                model_name: "model3".to_string(),
                provider: ModelProvider::HuggingFace,
                status: ModelStatus::DOWNLOADED,
                created_at: now - Duration::days(1),
                last_used_at: now - Duration::days(1),
                message: None,
            },
        ];

        let config = CacheEvictionConfig {
            enabled: true,
            policy: EvictionPolicyType::Lru(LruConfig {
                unused_threshold: DurationConfig::new(Duration::days(30)), // 30 days (none should be expired)
                max_models: Some(2),                                       // Limit to 2 models
                min_free_space_bytes: None,
            }),
            check_interval: DurationConfig::hours(1),
        };

        let policy = LruEvictionPolicy;
        let evicted = policy
            .select_for_eviction(&models, &config)
            .await
            .expect("Failed to select models for eviction");

        // Should evict the oldest model to stay within the limit of 2
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], "model1");
    }

    #[tokio::test]
    async fn test_cache_eviction_service_creation() {
        let (db, _temp_dir) = create_test_database();
        let config = CacheEvictionConfig::default();

        let service = CacheEvictionService::new(db, config.clone());
        assert!(service.config.enabled);
    }

    #[tokio::test]
    async fn test_manual_evict() {
        let (db, _temp_dir) = create_test_database();
        let config = CacheEvictionConfig::default();

        // Add a test model
        db.set_status(
            "test-model",
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
            None,
        )
        .expect("Failed to set model status");

        let service = CacheEvictionService::new(db.clone(), config);

        let models_to_evict = vec!["test-model".to_string()];
        let result = service
            .manual_evict(&models_to_evict)
            .await
            .expect("Failed to manually evict models");

        assert_eq!(result.evicted_count, 1);
        assert_eq!(result.evicted_models[0], "test-model");
        assert!(matches!(result.reason, EvictionReason::Manual));

        // Verify model was removed from database
        assert!(
            db.get_status("test-model")
                .expect("Failed to get model status")
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_get_cache_stats() {
        let (db, _temp_dir) = create_test_database();
        let config = CacheEvictionConfig::default();

        // Add test models with different statuses
        db.set_status(
            "model1",
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
            None,
        )
        .expect("Failed to set model1 status");
        db.set_status(
            "model2",
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADING,
            None,
        )
        .expect("Failed to set model2 status");
        db.set_status(
            "model3",
            ModelProvider::HuggingFace,
            ModelStatus::ERROR,
            None,
        )
        .expect("Failed to set model3 status");

        let service = CacheEvictionService::new(db, config);
        let stats = service
            .get_cache_stats()
            .await
            .expect("Failed to get cache stats");

        assert_eq!(stats.total_models, 3);
        assert_eq!(stats.downloaded_models, 1);
        assert_eq!(stats.downloading_models, 1);
        assert_eq!(stats.error_models, 1);
    }
}
