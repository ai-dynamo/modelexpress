// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Utils, constants,
    models::ModelProvider,
    providers::{
        gcs::GcsProviderCache, huggingface::HuggingFaceProviderCache, ngc::NgcProviderCache,
    },
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Configuration for model cache management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Local path where models are cached
    pub local_path: PathBuf,
    /// Server endpoint for model downloads
    pub server_endpoint: String,
    /// Timeout for cache operations
    pub timeout_secs: Option<u64>,
    /// Whether to use shared storage mode (client and server share a network drive)
    /// When false, files will be streamed from server to client
    #[serde(default = "default_shared_storage")]
    pub shared_storage: bool,
    /// Chunk size in bytes for file transfer streaming when shared_storage is false
    #[serde(default = "default_transfer_chunk_size")]
    pub transfer_chunk_size: usize,
}

fn default_shared_storage() -> bool {
    constants::DEFAULT_SHARED_STORAGE
}

fn default_transfer_chunk_size() -> usize {
    constants::DEFAULT_TRANSFER_CHUNK_SIZE
}

impl Default for CacheConfig {
    fn default() -> Self {
        let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
        Self {
            local_path: PathBuf::from(home).join(constants::DEFAULT_CACHE_PATH),
            server_endpoint: format!("http://localhost:{}", constants::DEFAULT_GRPC_PORT),
            timeout_secs: None,
            shared_storage: constants::DEFAULT_SHARED_STORAGE,
            transfer_chunk_size: constants::DEFAULT_TRANSFER_CHUNK_SIZE,
        }
    }
}

impl CacheConfig {
    /// Discover cache configuration
    pub fn discover() -> Result<Self> {
        // Priority order:
        // 1. Command line argument (--cache-path)
        // 2. Environment variable (MODEL_EXPRESS_CACHE_DIRECTORY)
        // 3. Config file (~/.model-express/config.yaml)
        // 4. Auto-detection (common paths)
        // 5. Default fallback

        // Try command line args first
        if let Some(path) = Self::get_cache_path_from_args() {
            return Self::from_path(path);
        }

        // Try environment variable
        if let Ok(path) = env::var("MODEL_EXPRESS_CACHE_DIRECTORY") {
            return Self::from_path(path);
        }

        // Try config file
        if let Ok(config) = Self::from_config_file() {
            return Ok(config);
        }

        // Try auto-detection
        if let Ok(config) = Self::auto_detect() {
            return Ok(config);
        }

        // Use default configuration as fallback
        debug!("Using default cache configuration");
        Ok(Self::default())
    }

    /// Create a cache configuration with explicit parameters
    pub fn new(local_path: PathBuf, server_endpoint: Option<String>) -> Result<Self> {
        // Ensure the directory exists
        fs::create_dir_all(&local_path)
            .with_context(|| format!("Failed to create cache directory: {local_path:?}"))?;

        Ok(Self {
            local_path,
            server_endpoint: server_endpoint.unwrap_or_else(Self::get_default_server_endpoint),
            timeout_secs: None,
            shared_storage: constants::DEFAULT_SHARED_STORAGE,
            transfer_chunk_size: constants::DEFAULT_TRANSFER_CHUNK_SIZE,
        })
    }

    /// Create config from a specific path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let local_path = path.as_ref().to_path_buf();

        // Ensure the directory exists
        fs::create_dir_all(&local_path)
            .with_context(|| format!("Failed to create cache directory: {local_path:?}"))?;

        Ok(Self {
            local_path,
            server_endpoint: Self::get_default_server_endpoint(),
            timeout_secs: None,
            shared_storage: constants::DEFAULT_SHARED_STORAGE,
            transfer_chunk_size: constants::DEFAULT_TRANSFER_CHUNK_SIZE,
        })
    }

    /// Load configuration from file
    pub fn from_config_file() -> Result<Self> {
        let config_path = Self::get_config_path()?;

        if !config_path.exists() {
            return Err(anyhow::anyhow!("Config file not found: {:?}", config_path));
        }

        let content = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {config_path:?}"))?;

        let config: Self = serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {config_path:?}"))?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_config_file(&self) -> Result<()> {
        let config_path = Self::get_config_path()?;

        // Ensure config directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {parent:?}"))?;
        }

        let content = serde_yaml::to_string(self).context("Failed to serialize config")?;

        fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config file: {config_path:?}"))?;

        Ok(())
    }

    /// Auto-detect cache configuration
    pub fn auto_detect() -> Result<Self> {
        let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
        let common_paths = vec![
            PathBuf::from(&home).join(constants::DEFAULT_CACHE_PATH),
            PathBuf::from(&home).join(constants::DEFAULT_HF_CACHE_PATH),
            PathBuf::from("/cache"),
            PathBuf::from("/app/models"),
            PathBuf::from("./cache"),
            PathBuf::from("./models"),
        ];

        for path in common_paths {
            if path.exists() && path.is_dir() {
                return Ok(Self {
                    local_path: path,
                    server_endpoint: Self::get_default_server_endpoint(),
                    timeout_secs: None,
                    shared_storage: constants::DEFAULT_SHARED_STORAGE,
                    transfer_chunk_size: constants::DEFAULT_TRANSFER_CHUNK_SIZE,
                });
            }
        }

        Err(anyhow::anyhow!(
            "No cache directory found in common locations"
        ))
    }

    /// Query server for cache information
    pub fn from_server() -> Result<Self> {
        // This would typically make an HTTP request to the server
        // For now, we'll return an error to indicate server is not available
        Err(anyhow::anyhow!("Server not available for cache discovery"))
    }

    /// Get cache path from command line arguments
    fn get_cache_path_from_args() -> Option<String> {
        let args: Vec<String> = env::args().collect();

        for (i, arg) in args.iter().enumerate() {
            if arg == "--cache-path"
                && let Some(next_arg) = args.get(i.saturating_add(1))
            {
                return Some(next_arg.clone());
            }
        }

        None
    }

    /// Get default server endpoint
    fn get_default_server_endpoint() -> String {
        env::var("MODEL_EXPRESS_SERVER_ENDPOINT")
            .unwrap_or_else(|_| format!("http://localhost:{}", constants::DEFAULT_GRPC_PORT))
    }

    /// Get configuration file path
    fn get_config_path() -> Result<PathBuf> {
        let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());

        Ok(PathBuf::from(home).join(constants::DEFAULT_CONFIG_PATH))
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<CacheStats> {
        let mut models = Vec::new();

        if !self.local_path.exists() {
            return Ok(CacheStats {
                total_models: 0,
                total_size: 0,
                models,
            });
        }

        for provider in [
            ModelProvider::HuggingFace,
            ModelProvider::Ngc,
            ModelProvider::Gcs,
        ] {
            models.extend(cache_for_provider(provider).list_models(&self.local_path)?);
        }

        models.sort_by(|left, right| {
            provider_sort_key(left.provider)
                .cmp(&provider_sort_key(right.provider))
                .then_with(|| left.name.cmp(&right.name))
        });

        let total_size = models.iter().map(|model| model.size).sum();

        Ok(CacheStats {
            total_models: models.len(),
            total_size,
            models,
        })
    }

    /// Clear specific model from cache for a given provider.
    pub fn clear_model(&self, model_name: &str, provider: ModelProvider) -> Result<()> {
        cache_for_provider(provider).clear_model(&self.local_path, model_name)
    }

    /// Clear entire cache
    pub fn clear_all(&self) -> Result<()> {
        if self.local_path.exists() {
            for entry in fs::read_dir(&self.local_path)
                .with_context(|| format!("Failed to read cache directory: {:?}", self.local_path))?
            {
                let entry = entry
                    .with_context(|| format!("Failed to read entry in: {:?}", self.local_path))?;
                let path = entry.path();
                if path.is_dir() {
                    fs::remove_dir_all(&path)
                        .with_context(|| format!("Failed to remove directory: {:?}", path))?;
                } else {
                    fs::remove_file(&path)
                        .with_context(|| format!("Failed to remove file: {:?}", path))?;
                }
            }
            info!("Cleared entire cache");
        } else {
            warn!("Cache directory does not exist");
        }

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_models: usize,
    pub total_size: u64,
    pub models: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub provider: ModelProvider,
    pub name: String,
    pub size: u64,
    pub path: PathBuf,
}

impl CacheStats {
    /// Format bytes as human readable string
    fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        match bytes {
            size if size >= GB => format!("{:.2} GB", size as f64 / GB as f64),
            size if size >= MB => format!("{:.2} MB", size as f64 / MB as f64),
            size if size >= KB => format!("{:.2} KB", size as f64 / KB as f64),
            size => format!("{size} bytes"),
        }
    }

    /// Format total size as human readable string
    pub fn format_total_size(&self) -> String {
        Self::format_bytes(self.total_size)
    }

    /// Format individual model size as human readable string
    pub fn format_model_size(&self, model: &ModelInfo) -> String {
        Self::format_bytes(model.size)
    }
}

pub(crate) trait ProviderCache: Send + Sync {
    fn clear_model(&self, cache_root: &Path, model_name: &str) -> Result<()>;
    fn resolve_model_path(
        &self,
        cache_root: &Path,
        model_name: &str,
        revision: Option<&str>,
    ) -> Result<PathBuf>;
    fn list_models(&self, cache_root: &Path) -> Result<Vec<ModelInfo>>;
}

pub(crate) fn cache_for_provider(provider: ModelProvider) -> &'static dyn ProviderCache {
    match provider {
        ModelProvider::HuggingFace => &HuggingFaceProviderCache,
        ModelProvider::Ngc => &NgcProviderCache,
        ModelProvider::Gcs => &GcsProviderCache,
    }
}

pub fn resolve_model_path(
    cache_root: &Path,
    provider: ModelProvider,
    model_name: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    cache_for_provider(provider).resolve_model_path(cache_root, model_name, revision)
}

pub(crate) fn directory_size(path: &Path) -> Result<u64> {
    let mut size: u64 = 0;

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            size = size.saturating_add(fs::metadata(&path)?.len());
        } else if path.is_dir() {
            size = size.saturating_add(directory_size(&path)?);
        }
    }

    Ok(size)
}

fn provider_sort_key(provider: ModelProvider) -> u8 {
    match provider {
        ModelProvider::HuggingFace => 0,
        ModelProvider::Ngc => 1,
        ModelProvider::Gcs => 2,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::Utils;
    use tempfile::TempDir;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_cache_config_from_path() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let config =
            CacheConfig::from_path(temp_dir.path()).expect("Failed to create config from path");

        assert_eq!(config.local_path, temp_dir.path());
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_cache_config_save_and_load() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let original_config = CacheConfig {
            local_path: temp_dir.path().join("cache"),
            server_endpoint: "http://localhost:8001".to_string(),
            timeout_secs: Some(30),
            shared_storage: false,
            transfer_chunk_size: 64 * 1024,
        };

        // Save config
        original_config
            .save_to_config_file()
            .expect("Failed to save config");

        // Load config
        let loaded_config = CacheConfig::from_config_file().expect("Failed to load config");

        assert_eq!(loaded_config.local_path, original_config.local_path);
        assert_eq!(
            loaded_config.server_endpoint,
            original_config.server_endpoint
        );
        assert_eq!(loaded_config.timeout_secs, original_config.timeout_secs);
        assert_eq!(loaded_config.shared_storage, original_config.shared_storage);
        assert_eq!(
            loaded_config.transfer_chunk_size,
            original_config.transfer_chunk_size
        );
    }

    #[test]
    fn test_cache_stats_formatting() {
        let stats = CacheStats {
            total_models: 2,
            total_size: 1024 * 1024 * 5, // 5 MB
            models: vec![
                ModelInfo {
                    provider: ModelProvider::HuggingFace,
                    name: "model1".to_string(),
                    size: 1024 * 1024 * 2, // 2 MB
                    path: PathBuf::from("/test/model1"),
                },
                ModelInfo {
                    provider: ModelProvider::Gcs,
                    name: "gs://bucket/model2".to_string(),
                    size: 1024 * 1024 * 3, // 3 MB
                    path: PathBuf::from("/test/model2"),
                },
            ],
        };

        assert_eq!(stats.format_total_size(), "5.00 MB");
        assert_eq!(stats.format_model_size(&stats.models[0]), "2.00 MB");
        assert_eq!(stats.format_model_size(&stats.models[1]), "3.00 MB");
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();

        let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
        assert_eq!(
            config.local_path,
            PathBuf::from(&home).join(constants::DEFAULT_CACHE_PATH)
        );
        assert_eq!(
            config.server_endpoint,
            String::from("http://localhost:8001")
        );
        assert_eq!(config.timeout_secs, None);
        assert!(config.shared_storage);
        assert_eq!(
            config.transfer_chunk_size,
            constants::DEFAULT_TRANSFER_CHUNK_SIZE
        );
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_get_config_path() {
        let config_path = CacheConfig::get_config_path().expect("Failed to get config path");

        let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
        assert_eq!(
            config_path,
            PathBuf::from(&home).join(constants::DEFAULT_CONFIG_PATH)
        );
    }

    #[test]
    fn test_resolve_model_path_huggingface_uses_snapshot_layout() {
        let cache_root = Path::new("/tmp/cache");

        assert_eq!(
            resolve_model_path(
                cache_root,
                ModelProvider::HuggingFace,
                "google/t5-small",
                Some("abc123"),
            )
            .expect("Expected HF model path"),
            PathBuf::from("/tmp/cache/models--google--t5-small/snapshots/abc123")
        );
    }

    #[test]
    fn test_resolve_model_path_gcs_uses_full_url_layout() {
        let cache_root = Path::new("/tmp/cache");

        assert_eq!(
            resolve_model_path(
                cache_root,
                ModelProvider::Gcs,
                "gs://envbucket/dev/bake/qwen/rev123",
                None,
            )
            .expect("Expected GCS model path"),
            PathBuf::from("/tmp/cache/gcs/envbucket/dev/bake/qwen/rev123")
        );
    }

    #[test]
    fn test_resolve_model_path_gcs_full_url_trailing_slash_normalizes() {
        let cache_root = Path::new("/tmp/cache");

        assert_eq!(
            resolve_model_path(
                cache_root,
                ModelProvider::Gcs,
                "gs://sourcebucket/dev/bake/qwen/rev123/",
                None,
            )
            .expect("Expected GCS model path"),
            PathBuf::from("/tmp/cache/gcs/sourcebucket/dev/bake/qwen/rev123")
        );
    }

    fn create_test_cache_config(local_path: PathBuf) -> CacheConfig {
        CacheConfig {
            local_path,
            server_endpoint: "http://localhost:8001".to_string(),
            timeout_secs: None,
            shared_storage: false,
            transfer_chunk_size: 64 * 1024,
        }
    }

    #[test]
    fn test_get_cache_stats_supports_hf_and_gcs_layouts() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_path).expect("Failed to create cache directory");

        let hf_model_dir = cache_path.join("models--google--t5-small");
        fs::create_dir_all(&hf_model_dir).expect("Failed to create HF model directory");
        fs::write(hf_model_dir.join("config.json"), b"{}").expect("Failed to write HF file");

        let gcs_model_dir = resolve_model_path(
            &cache_path,
            ModelProvider::Gcs,
            "gs://envbucket/dev/bake/qwen/rev123",
            None,
        )
        .expect("Failed to resolve GCS path");
        fs::create_dir_all(gcs_model_dir.join("weights"))
            .expect("Failed to create GCS model directory");
        fs::write(gcs_model_dir.join("tokenizer.json"), b"{}")
            .expect("Failed to write GCS tokenizer");
        fs::write(gcs_model_dir.join("weights/model.bin"), b"abcd")
            .expect("Failed to write GCS weights");
        let gcs_metadata_dir = gcs_model_dir.join(".mx");
        fs::create_dir_all(&gcs_metadata_dir).expect("Failed to create GCS metadata directory");
        fs::write(
            gcs_metadata_dir.join("manifest.json"),
            r#"{"version":1,"model":"gs://envbucket/dev/bake/qwen/rev123","files":[{"path":"tokenizer.json","size":2,"crc32c":"00000000","generation":null},{"path":"weights/model.bin","size":4,"crc32c":"00000000","generation":null}]}
"#,
        )
        .expect("Failed to write GCS manifest");

        let ignored_dir = cache_path.join("tmp");
        fs::create_dir_all(&ignored_dir).expect("Failed to create ignored directory");
        fs::write(ignored_dir.join("scratch.txt"), b"ignore")
            .expect("Failed to write ignored file");

        let stats = create_test_cache_config(cache_path)
            .get_cache_stats()
            .expect("Failed to get cache stats");

        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.total_size, 8);
        assert_eq!(stats.models.len(), 2);

        assert_eq!(stats.models[0].provider, ModelProvider::HuggingFace);
        assert_eq!(stats.models[0].name, "google/t5-small");
        assert_eq!(stats.models[0].size, 2);
        assert_eq!(stats.models[0].path, hf_model_dir);
        assert_eq!(stats.models[1].provider, ModelProvider::Gcs);
        assert_eq!(stats.models[1].name, "gs://envbucket/dev/bake/qwen/rev123");
        assert_eq!(stats.models[1].size, 6);
        assert_eq!(stats.models[1].path, gcs_model_dir);
        assert!(stats.models.iter().all(|model| model.name != "tmp"));
    }

    #[test]
    fn test_clear_model_removes_only_requested_layout() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_path).expect("Failed to create cache directory");

        let hf_model_dir = cache_path.join("models--google--t5-small");
        fs::create_dir_all(&hf_model_dir).expect("Failed to create HF model directory");
        fs::write(hf_model_dir.join("config.json"), b"{}").expect("Failed to write HF file");

        let gcs_model_dir = resolve_model_path(
            &cache_path,
            ModelProvider::Gcs,
            "gs://envbucket/org/model/rev1",
            None,
        )
        .expect("Failed to resolve GCS path");
        fs::create_dir_all(&gcs_model_dir).expect("Failed to create GCS model directory");
        fs::write(gcs_model_dir.join("tokenizer.json"), b"{}").expect("Failed to write GCS file");

        let config = create_test_cache_config(cache_path);

        config
            .clear_model("gs://envbucket/org/model/rev1", ModelProvider::Gcs)
            .expect("Failed to clear GCS model");
        assert!(hf_model_dir.exists(), "HF model should remain");
        assert!(!gcs_model_dir.exists(), "GCS model should be removed");

        config
            .clear_model("google/t5-small", ModelProvider::HuggingFace)
            .expect("Failed to clear HF model");
        assert!(!hf_model_dir.exists(), "HF model should be removed");
    }

    #[test]
    fn test_clear_all_removes_contents_but_keeps_directory() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_path).expect("Failed to create cache directory");

        // Create some test content
        let model_dir = cache_path.join("models--test--model");
        fs::create_dir_all(&model_dir).expect("Failed to create model directory");
        fs::write(model_dir.join("config.json"), "{}").expect("Failed to write file");
        fs::write(cache_path.join("test_file.txt"), "test").expect("Failed to write file");

        let config = create_test_cache_config(cache_path.clone());

        // Clear cache
        config.clear_all().expect("Failed to clear cache");

        // Directory should still exist but be empty
        assert!(cache_path.exists(), "Cache directory should still exist");
        assert!(
            fs::read_dir(&cache_path)
                .expect("Failed to read dir")
                .next()
                .is_none(),
            "Cache directory should be empty"
        );
    }

    #[test]
    fn test_clear_all_handles_nonexistent_directory() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("nonexistent_cache");

        let config = create_test_cache_config(cache_path.clone());

        // Should succeed without error even if directory doesn't exist
        config
            .clear_all()
            .with_context(|| format!("Failed to clear cache: {cache_path:?}"))
            .expect("Failed to clear cache");
        assert!(!cache_path.exists());
    }

    #[test]
    fn test_clear_all_removes_nested_directories() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_path).expect("Failed to create cache directory");

        // Create nested structure
        let deep_path = cache_path.join("a").join("b").join("c");
        fs::create_dir_all(&deep_path).expect("Failed to create nested directories");
        fs::write(deep_path.join("deep_file.txt"), "deep").expect("Failed to write file");

        let config = create_test_cache_config(cache_path.clone());

        config.clear_all().expect("Failed to clear cache");

        assert!(cache_path.exists(), "Cache directory should still exist");
        assert!(
            fs::read_dir(&cache_path)
                .expect("Failed to read dir")
                .next()
                .is_none(),
            "Cache directory should be empty after clearing nested content"
        );
    }
}
