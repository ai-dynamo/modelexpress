// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Utils, constants};
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

    /// Convert a Hugging Face folder name back to the original model ID
    /// Examples:
    /// - "models--google-t5--t5-small" -> "google-t5/t5-small"
    pub fn folder_name_to_model_id(folder_name: &str) -> String {
        // Handle models
        if let Some(stripped) = folder_name.strip_prefix("models--") {
            // Convert models--owner--repo to owner/repo
            stripped.replace("--", "/")
        } else if folder_name.starts_with("datasets--") {
            // TODO: Handle datasets names conversion
            folder_name.to_string()
        } else if folder_name.starts_with("spaces--") {
            // TODO: Handle spaces names conversion
            folder_name.to_string()
        } else {
            // If it doesn't match the expected pattern, return as-is
            folder_name.to_string()
        }
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<CacheStats> {
        let mut stats = CacheStats {
            total_models: 0,
            total_size: 0,
            models: Vec::new(),
        };

        if !self.local_path.exists() {
            return Ok(stats);
        }

        for entry in fs::read_dir(&self.local_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let size = Self::get_directory_size(&path)?;
                let folder_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                info!("Folder name: {}", folder_name);
                // Convert folder name back to human-readable model ID
                let model_name = Self::folder_name_to_model_id(&folder_name);

                stats.total_models = stats.total_models.saturating_add(1);
                stats.total_size = stats.total_size.saturating_add(size);
                stats.models.push(ModelInfo {
                    name: model_name,
                    size,
                    path: path.to_path_buf(),
                });
            }
        }

        Ok(stats)
    }

    /// Get directory size recursively
    fn get_directory_size(path: &Path) -> Result<u64> {
        let mut size: u64 = 0;

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                size = size.saturating_add(fs::metadata(&path)?.len());
            } else if path.is_dir() {
                size = size.saturating_add(Self::get_directory_size(&path)?);
            }
        }

        Ok(size)
    }

    /// Clear specific model from cache
    pub fn clear_model(&self, model_name: &str) -> Result<()> {
        let model_path = self.local_path.join(model_name);

        if model_path.exists() {
            fs::remove_dir_all(&model_path)
                .with_context(|| format!("Failed to remove model: {model_path:?}"))?;
            info!("Cleared model: {}", model_name);
        } else {
            warn!("Model not found in cache: {}", model_name);
        }

        Ok(())
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

#[cfg(test)]
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
                    name: "model1".to_string(),
                    size: 1024 * 1024 * 2, // 2 MB
                    path: PathBuf::from("/test/model1"),
                },
                ModelInfo {
                    name: "model2".to_string(),
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
    fn test_folder_name_to_model_id() {
        // Test models conversion
        assert_eq!(
            CacheConfig::folder_name_to_model_id("models--google-t5--t5-small"),
            "google-t5/t5-small"
        );
        assert_eq!(
            CacheConfig::folder_name_to_model_id("models--microsoft--DialoGPT-medium"),
            "microsoft/DialoGPT-medium"
        );
        assert_eq!(
            CacheConfig::folder_name_to_model_id("models--huggingface--CodeBERTa-small-v1"),
            "huggingface/CodeBERTa-small-v1"
        );

        // Test single name models (no organization)
        assert_eq!(
            CacheConfig::folder_name_to_model_id("models--bert-base-uncased"),
            "bert-base-uncased"
        );

        // Test datasets (TODO - should return as-is for now)
        assert_eq!(
            CacheConfig::folder_name_to_model_id("datasets--squad"),
            "datasets--squad"
        );
        assert_eq!(
            CacheConfig::folder_name_to_model_id("datasets--huggingface--squad"),
            "datasets--huggingface--squad"
        );

        // Test spaces (TODO - should return as-is for now)
        assert_eq!(
            CacheConfig::folder_name_to_model_id("spaces--gradio--hello-world"),
            "spaces--gradio--hello-world"
        );

        // Test unrecognized patterns (should return as-is)
        assert_eq!(
            CacheConfig::folder_name_to_model_id("random-folder-name"),
            "random-folder-name"
        );
        assert_eq!(
            CacheConfig::folder_name_to_model_id("some--other--format"),
            "some--other--format"
        );

        // Test edge cases
        assert_eq!(CacheConfig::folder_name_to_model_id("models--"), "");
        assert_eq!(
            CacheConfig::folder_name_to_model_id("models--single"),
            "single"
        );
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_clear_all_removes_contents_but_keeps_directory() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_path).expect("Failed to create cache directory");

        // Create some test content
        let model_dir = cache_path.join("models--test--model");
        fs::create_dir_all(&model_dir).expect("Failed to create model directory");
        fs::write(model_dir.join("config.json"), "{}").expect("Failed to write file");
        fs::write(cache_path.join("test_file.txt"), "test").expect("Failed to write file");

        let config = CacheConfig {
            local_path: cache_path.clone(),
            server_endpoint: "http://localhost:8001".to_string(),
            timeout_secs: None,
            shared_storage: false,
            transfer_chunk_size: 64 * 1024,
        };

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
    #[allow(clippy::expect_used)]
    fn test_clear_all_handles_nonexistent_directory() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("nonexistent_cache");

        let config = CacheConfig {
            local_path: cache_path.clone(),
            server_endpoint: "http://localhost:8001".to_string(),
            timeout_secs: None,
            shared_storage: false,
            transfer_chunk_size: 64 * 1024,
        };

        // Should succeed without error even if directory doesn't exist
        config.clear_all().expect("clear_all should handle nonexistent directory");
        assert!(!cache_path.exists());
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_clear_all_removes_nested_directories() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let cache_path = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_path).expect("Failed to create cache directory");

        // Create nested structure
        let deep_path = cache_path.join("a").join("b").join("c");
        fs::create_dir_all(&deep_path).expect("Failed to create nested directories");
        fs::write(deep_path.join("deep_file.txt"), "deep").expect("Failed to write file");

        let config = CacheConfig {
            local_path: cache_path.clone(),
            server_endpoint: "http://localhost:8001".to_string(),
            timeout_secs: None,
            shared_storage: false,
            transfer_chunk_size: 64 * 1024,
        };

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
