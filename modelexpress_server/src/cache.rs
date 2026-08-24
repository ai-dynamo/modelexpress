// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::time::{Duration as TokioDuration, interval};
use tracing::{debug, error, info, warn};

use std::path::PathBuf;
use std::sync::Arc;

use crate::registry::backend::ModelRecord;
use crate::registry::entry_key::EntryKey;
use crate::registry::state::RegistryManager;
use modelexpress_common::config::DurationConfig;
use modelexpress_common::download::get_provider;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// One model selected for eviction, with the rule that picked it.
///
/// The reason travels with the model rather than with the cycle: a single pass
/// can evict some models for age and others for the count limit, and a
/// cycle-level reason has to pick one and misreport the rest.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    /// Registry key of the model to evict.
    pub model_name: String,
    /// Which rule selected it.
    pub reason: EvictionReason,
}

/// Trait for implementing different eviction policies
#[async_trait::async_trait]
pub trait EvictionPolicyTrait {
    /// Determine which models should be evicted based on the policy
    async fn select_for_eviction(
        &self,
        models: &[ModelRecord],
        config: &CacheEvictionConfig,
    ) -> Result<Vec<EvictionCandidate>, Box<dyn std::error::Error + Send + Sync>>;
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
    ) -> Result<Vec<EvictionCandidate>, Box<dyn std::error::Error + Send + Sync>> {
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
                candidates_for_eviction.push(EvictionCandidate {
                    model_name: model.model_name.clone(),
                    reason: EvictionReason::TimeThreshold,
                });
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
                    if !candidates_for_eviction
                        .iter()
                        .any(|candidate| candidate.model_name == model.model_name)
                    {
                        candidates_for_eviction.push(EvictionCandidate {
                            model_name: model.model_name.clone(),
                            reason: EvictionReason::CountLimit,
                        });
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
    registry: Arc<RegistryManager>,
    config: CacheEvictionConfig,
    cache_directory: PathBuf,
    metrics: crate::metrics::cache::CacheMetrics,
}

impl CacheEvictionService {
    /// Create a new cache eviction service
    pub fn new(
        registry: Arc<RegistryManager>,
        config: CacheEvictionConfig,
        cache_directory: PathBuf,
        metrics: crate::metrics::cache::CacheMetrics,
    ) -> Self {
        Self {
            metrics,
            registry,
            config,
            cache_directory,
        }
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
        let models = self.registry.get_models_by_last_used(None).await?;
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
        for candidate in &models_to_evict {
            let model_name = &candidate.model_name;
            match self.evict_model(model_name).await {
                Ok(()) => {
                    successfully_evicted.push(model_name.clone());
                    // Counted per model, with the rule that actually selected it:
                    // one cycle can evict some models for age and others for the
                    // count limit.
                    self.metrics.record_eviction(candidate.reason);
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
            // A summary of a cycle that may have evicted for more than one
            // reason; the per-reason breakdown is in mx_cache_evictions_total.
            reason: models_to_evict
                .first()
                .map_or(EvictionReason::TimeThreshold, |candidate| candidate.reason),
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

    /// Evict a single entry (remove from filesystem, then from the registry).
    ///
    /// `entry_key` is a registry key, so it has to be parsed back into the model name
    /// and revision the provider needs in order to delete the right snapshot.
    async fn evict_model(
        &self,
        entry_key: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Look up the model record to determine the provider
        let record = self
            .registry
            .get_model_record(entry_key)
            .await?
            .ok_or_else(|| format!("model '{entry_key}' not found in registry"))?;

        let key = EntryKey::parse(entry_key);

        // Entries that outlive this one and belong to the same model. Which files we can
        // delete depends entirely on what they still reference.
        let siblings = self.registry.get_models_by_last_used(None).await?;
        let survivors: Vec<EntryKey> = siblings
            .iter()
            .filter(|sibling| sibling.model_name != entry_key)
            .map(|sibling| EntryKey::parse(&sibling.model_name))
            .filter(|other| other.model_name == key.model_name)
            .collect();

        // An entry with no revision covers every snapshot of its model, so it shares
        // files with all of them. That is how a record written before revisions existed
        // behaves: evicting it must not delete a snapshot a revision-scoped entry uses,
        // and a revision-scoped evict must not delete files the legacy entry covers.
        let snapshot_still_referenced = survivors.iter().any(|other| {
            other.revision == key.revision || other.revision.is_none() || key.revision.is_none()
        });

        // Delete files from disk first. If this fails, we keep the registry record
        // so the next eviction cycle can retry.
        let delete_revision = if survivors.is_empty() {
            // Nothing else references this model. Drop the whole cache entry rather than
            // just this revision, so snapshots left behind by an earlier shared-file
            // decision cannot outlive the last record pointing at them.
            Some(None)
        } else if snapshot_still_referenced {
            debug!(
                "Keeping files for '{entry_key}': another registry entry still references \
                 the same snapshot"
            );
            None
        } else {
            Some(key.revision.as_deref())
        };

        if let Some(revision) = delete_revision {
            get_provider(record.provider)
                .delete_model_revision(&key.model_name, self.cache_directory.clone(), revision)
                .await
                .map_err(|e| format!("failed to delete model files for '{entry_key}': {e}"))?;
        }

        // Only remove from registry after successful filesystem deletion
        self.registry.delete_model(entry_key).await?;

        Ok(())
    }

    /// Get statistics about the current cache state
    pub async fn get_cache_stats(
        &self,
    ) -> Result<CacheStats, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.registry.get_models_by_last_used(None).await?;
        let (downloading, downloaded, error) = self.registry.get_status_counts().await?;

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
    use crate::registry::backend::MockRegistryBackend;
    use modelexpress_common::models::ModelProvider;
    use tempfile::TempDir;

    fn service_with_mock(
        mock: MockRegistryBackend,
        config: CacheEvictionConfig,
    ) -> (CacheEvictionService, TempDir) {
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(mock)));
        let cache_dir = TempDir::new().expect("Failed to create cache directory");
        let service = CacheEvictionService::new(
            registry,
            config,
            cache_dir.path().to_path_buf(),
            crate::metrics::cache::CacheMetrics::register(&mut crate::metrics::new_registry()),
        );
        (service, cache_dir)
    }

    /// As [`service_with_mock`], but hands back the Prometheus registry the
    /// eviction counter was registered into.
    ///
    /// [`service_with_mock`] registers into a temporary registry that is dropped on
    /// the spot, so nothing recorded through it can ever be encoded and any metric
    /// assertion against a service built that way would be vacuous.
    fn service_with_mock_and_registry(
        mock: MockRegistryBackend,
        config: CacheEvictionConfig,
    ) -> (
        CacheEvictionService,
        TempDir,
        prometheus_client::registry::Registry,
    ) {
        let mut metrics_registry = crate::metrics::new_registry();
        let metrics = crate::metrics::cache::CacheMetrics::register(&mut metrics_registry);
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(mock)));
        let cache_dir = TempDir::new().expect("Failed to create cache directory");
        let service =
            CacheEvictionService::new(registry, config, cache_dir.path().to_path_buf(), metrics);
        (service, cache_dir, metrics_registry)
    }

    /// Value of `mx_cache_evictions_total{reason=...}`, or 0 when the series was
    /// never created. A reason that is never emitted is absent rather than zero,
    /// which is exactly the case a `contains` assertion cannot tell apart from a
    /// wrong count.
    fn evictions_for(encoded: &str, reason: &str) -> i64 {
        let prefix = format!(r#"mx_cache_evictions_total{{reason="{reason}"}} "#);
        encoded
            .lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_default()
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
        assert_eq!(evicted[0].model_name, "old-model");
        assert_eq!(evicted[0].reason, EvictionReason::TimeThreshold);
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
        assert_eq!(evicted[0].model_name, "model1");
        // Selected by the count limit, not by age. This assertion is the whole
        // point of carrying the reason per candidate: before, this path was
        // reported as a time-threshold eviction.
        assert_eq!(evicted[0].reason, EvictionReason::CountLimit);
    }

    #[tokio::test]
    async fn test_cache_eviction_service_creation() {
        let mock = MockRegistryBackend::new();
        let (service, _cache_dir) = service_with_mock(mock, CacheEvictionConfig::default());
        assert!(service.config.enabled);
    }

    #[tokio::test]
    async fn test_get_cache_stats_uses_registry() {
        let now = Utc::now();
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .once()
            .returning(move |_| {
                Ok(vec![ModelRecord {
                    model_name: "model1".to_string(),
                    provider: ModelProvider::HuggingFace,
                    status: ModelStatus::DOWNLOADED,
                    created_at: now,
                    last_used_at: now,
                    message: None,
                }])
            });
        mock.expect_get_status_counts()
            .once()
            .returning(|| Ok((1, 1, 1)));
        let (service, _cache_dir) = service_with_mock(mock, CacheEvictionConfig::default());
        let stats = service.get_cache_stats().await.expect("stats");
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.downloaded_models, 1);
        assert_eq!(stats.downloading_models, 1);
        assert_eq!(stats.error_models, 1);
    }

    fn downloaded_record(model_name: &str) -> ModelRecord {
        let now = Utc::now();
        ModelRecord {
            model_name: model_name.to_string(),
            provider: ModelProvider::HuggingFace,
            status: ModelStatus::DOWNLOADED,
            created_at: now,
            last_used_at: now,
            message: None,
        }
    }

    fn key(revision: Option<&str>, metadata_only: bool) -> String {
        EntryKey::new("test/model", revision.map(String::from), metadata_only).to_string()
    }

    /// Which snapshots of `test/model` survive evicting `entry_key` while `siblings`
    /// remain in the registry. Two snapshots exist on disk: `abc123` and `def456`.
    async fn evict_and_report_surviving_snapshots(
        entry_key: &str,
        siblings: Vec<String>,
    ) -> Vec<String> {
        let evicted = entry_key.to_string();
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_model_record()
            .once()
            .returning(move |name| Ok(Some(downloaded_record(name))));
        mock.expect_get_models_by_last_used()
            .once()
            .returning(move |_| {
                let mut records = vec![downloaded_record(&evicted)];
                records.extend(siblings.iter().map(|name| downloaded_record(name)));
                Ok(records)
            });
        mock.expect_delete_model().once().returning(|_| Ok(()));

        let (service, cache_dir) = service_with_mock(mock, CacheEvictionConfig::default());
        let snapshots_dir = cache_dir.path().join("models--test--model/snapshots");
        for commit in ["abc123", "def456"] {
            let snapshot = snapshots_dir.join(commit);
            std::fs::create_dir_all(&snapshot).expect("Failed to create snapshot");
            std::fs::write(snapshot.join("config.json"), b"{}").expect("Failed to write config");
        }

        service.evict_model(entry_key).await.expect("evict");

        let mut surviving: Vec<String> = std::fs::read_dir(&snapshots_dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();
        surviving.sort();
        surviving
    }

    #[tokio::test]
    async fn test_evict_removes_only_its_own_snapshot() {
        assert_eq!(
            evict_and_report_surviving_snapshots(
                &key(Some("abc123"), false),
                vec![key(Some("def456"), false)]
            )
            .await,
            vec!["def456".to_string()]
        );
    }

    #[tokio::test]
    async fn test_evict_keeps_files_shared_with_the_metadata_only_entry() {
        assert_eq!(
            evict_and_report_surviving_snapshots(
                &key(Some("abc123"), false),
                vec![key(Some("abc123"), true), key(Some("def456"), false)]
            )
            .await,
            vec!["abc123".to_string(), "def456".to_string()]
        );
    }

    #[tokio::test]
    async fn test_evict_keeps_files_a_revision_scoped_entry_still_uses() {
        // A pre-revision record covers every snapshot of its model, so evicting it must
        // not delete the snapshot a revision-scoped entry points at.
        assert_eq!(
            evict_and_report_surviving_snapshots("test/model", vec![key(Some("abc123"), false)])
                .await,
            vec!["abc123".to_string(), "def456".to_string()]
        );
    }

    #[tokio::test]
    async fn test_evict_of_the_last_entry_removes_every_snapshot() {
        // Otherwise a snapshot kept alive by an earlier shared-file decision would
        // outlive the last record that pointed at it, and never be reclaimed.
        assert!(
            evict_and_report_surviving_snapshots(&key(Some("abc123"), false), vec![])
                .await
                .is_empty()
        );
    }

    fn record_used_days_ago(model_name: &str, days: i64) -> ModelRecord {
        let then = Utc::now()
            .checked_sub_signed(Duration::days(days))
            .expect("timestamp in range");
        ModelRecord {
            model_name: model_name.to_string(),
            provider: ModelProvider::HuggingFace,
            status: ModelStatus::DOWNLOADED,
            created_at: then,
            last_used_at: then,
            message: None,
        }
    }

    fn lru_config(unused_threshold_days: i64, max_models: Option<u32>) -> CacheEvictionConfig {
        CacheEvictionConfig {
            enabled: true,
            policy: EvictionPolicyType::Lru(LruConfig {
                unused_threshold: DurationConfig::new(Duration::days(unused_threshold_days)),
                max_models,
                min_free_space_bytes: None,
            }),
            check_interval: DurationConfig::hours(1),
        }
    }

    /// A mock that lets a whole cycle run against `records`: the cycle lists them
    /// once and `evict_model` lists them again per victim, then reads and deletes
    /// each victim's record. The cache directory is empty, which the Hugging Face
    /// provider treats as a no-op delete, so nothing here touches the network.
    fn mock_over(records: Vec<ModelRecord>) -> MockRegistryBackend {
        let mut mock = MockRegistryBackend::new();
        let listed = records.clone();
        mock.expect_get_models_by_last_used()
            .returning(move |_| Ok(listed.clone()));
        mock.expect_get_model_record()
            .returning(move |name| Ok(Some(downloaded_record(name))));
        mock.expect_delete_model().returning(|_| Ok(()));
        mock
    }

    /// The shipped-a-wrong-number bug, one layer below where it was fixed. The
    /// policy now carries the rule that selected each model, but nothing checked
    /// that the reason survives the trip to the counter: an operator tuning
    /// `max_models` would see `reason="count_limit"` sit at zero forever while
    /// every count-limit eviction was attributed to the time threshold.
    ///
    /// `test_lru_eviction_policy_count_based` asserts the candidate struct, which
    /// is a different claim -- it stays green when the service mislabels the
    /// counter.
    #[tokio::test]
    async fn a_count_limit_eviction_is_counted_as_a_count_limit() {
        let records = vec![
            record_used_days_ago("model1", 3),
            record_used_days_ago("model2", 2),
            record_used_days_ago("model3", 1),
        ];
        // Nothing is old enough for the time rule; only the count limit fires.
        let (service, _cache_dir, metrics_registry) =
            service_with_mock_and_registry(mock_over(records), lru_config(30, Some(2)));

        let result = service.run_eviction_cycle().await.expect("eviction cycle");
        assert_eq!(result.evicted_count, 1);
        assert_eq!(result.evicted_models, vec!["model1".to_string()]);
        // The cycle summary reports the rule that actually fired, not a default.
        assert_eq!(result.reason, EvictionReason::CountLimit);

        let encoded = crate::metrics::encode_text(&metrics_registry)
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(evictions_for(&encoded, "count_limit"), 1, "{encoded}");
        assert_eq!(
            evictions_for(&encoded, "time_threshold"),
            0,
            "the count rule must not be reported as an age eviction: {encoded}"
        );
    }

    /// One pass can evict some models for age and others for the count limit, which
    /// is why the reason travels per candidate rather than per cycle. A single
    /// cycle-level reason has to pick one and misreport the rest, and no
    /// single-reason test can tell the two designs apart.
    #[tokio::test]
    async fn a_mixed_cycle_counts_each_model_against_its_own_rule() {
        let records = vec![
            record_used_days_ago("stale-model", 8),
            record_used_days_ago("mid-model", 3),
            record_used_days_ago("fresh-model", 1),
        ];
        // stale-model is past the 7-day threshold; the limit of 1 then also takes
        // the next-oldest, and that one is selected by the count rule.
        let (service, _cache_dir, metrics_registry) =
            service_with_mock_and_registry(mock_over(records), lru_config(7, Some(1)));

        let result = service.run_eviction_cycle().await.expect("eviction cycle");
        assert_eq!(result.evicted_count, 2);

        let encoded = crate::metrics::encode_text(&metrics_registry)
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(
            evictions_for(&encoded, "time_threshold"),
            1,
            "the aged model is one time-threshold eviction: {encoded}"
        );
        assert_eq!(
            evictions_for(&encoded, "count_limit"),
            1,
            "the model taken to satisfy max_models is a count-limit eviction: {encoded}"
        );
    }

    /// The counter is booked inside the success arm, so it stays a count of models
    /// that actually left the cache. Counting selections instead would report disk
    /// as reclaimed while the entries are still there and the next cycle retries
    /// them, double-counting the same model on every pass.
    #[tokio::test]
    async fn an_eviction_that_fails_is_not_counted() {
        let records = vec![
            record_used_days_ago("model1", 3),
            record_used_days_ago("model2", 2),
            record_used_days_ago("model3", 1),
        ];
        let listed = records.clone();
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .returning(move |_| Ok(listed.clone()));
        // The record vanished between selection and eviction, so evict_model errors
        // before it can delete anything.
        mock.expect_get_model_record().returning(|_| Ok(None));

        let (service, _cache_dir, metrics_registry) =
            service_with_mock_and_registry(mock, lru_config(30, Some(2)));

        let result = service.run_eviction_cycle().await.expect("eviction cycle");
        assert_eq!(result.evicted_count, 0);

        let encoded = crate::metrics::encode_text(&metrics_registry)
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(
            evictions_for(&encoded, "count_limit"),
            0,
            "a failed eviction freed nothing and must not be counted: {encoded}"
        );
    }
}
