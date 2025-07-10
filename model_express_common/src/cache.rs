use crate::constants;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

/// Configuration for model cache management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Local path where models are cached
    pub local_path: PathBuf,
    /// Server endpoint for model downloads
    pub server_endpoint: String,
    /// Whether to auto-mount cache on startup
    pub auto_mount: bool,
    /// Timeout for cache operations
    pub timeout_secs: Option<u64>,
}

impl CacheConfig {
    /// Discover cache configuration using the hybrid approach
    pub fn discover() -> Result<Self> {
        // Priority order:
        // 1. Command line argument (--cache-path)
        // 2. Environment variable (MODEL_EXPRESS_CACHE_PATH)
        // 3. Config file (~/.model-express/config.yaml)
        // 4. Auto-detection (common paths)
        // 5. Server query (if server is reachable)
        // 6. User prompt (fallback)

        // Try command line args first
        if let Some(path) = Self::get_cache_path_from_args() {
            return Self::from_path(path);
        }

        // Try environment variable
        if let Ok(path) = env::var("MODEL_EXPRESS_CACHE_PATH") {
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

        // Try server query (non-blocking)
        if let Ok(config) = Self::from_server() {
            return Ok(config);
        }

        // Prompt user as fallback
        Self::prompt_user()
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
            auto_mount: true,
            timeout_secs: None,
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
        let common_paths = vec![
            PathBuf::from("~/.model-express/cache"),
            PathBuf::from("~/.cache/huggingface/hub"),
            PathBuf::from("/cache"),
            PathBuf::from("/app/models"),
            PathBuf::from("./cache"),
            PathBuf::from("./models"),
        ];

        for path in common_paths {
            let expanded_path = Self::expand_path(&path)?;
            if expanded_path.exists() && expanded_path.is_dir() {
                return Ok(Self {
                    local_path: expanded_path,
                    server_endpoint: Self::get_default_server_endpoint(),
                    auto_mount: true,
                    timeout_secs: None,
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

    /// Interactive user prompt for configuration
    pub fn prompt_user() -> Result<Self> {
        info!("ModelExpress Cache Configuration");
        info!("================================");

        // Get cache path
        let cache_path = Self::prompt_cache_path()?;

        // Get server endpoint
        let server_endpoint = Self::prompt_server_endpoint()?;

        // Get auto-mount preference
        let auto_mount = Self::prompt_auto_mount()?;

        let config = Self {
            local_path: cache_path,
            server_endpoint,
            auto_mount,
            timeout_secs: None,
        };

        // Save configuration
        if Self::prompt_save_config()? {
            config.save_to_config_file()?;
            info!("Configuration saved to {:?}", Self::get_config_path()?);
        }

        Ok(config)
    }

    /// Get cache path from command line arguments
    fn get_cache_path_from_args() -> Option<String> {
        let args: Vec<String> = env::args().collect();

        for (i, arg) in args.iter().enumerate() {
            if arg == "--cache-path" {
                if let Some(next_arg) = args.get(i.saturating_add(1)) {
                    return Some(next_arg.clone());
                }
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
        let home = env::var("HOME")
            .or_else(|_| env::var("USERPROFILE"))
            .context("Could not determine home directory")?;

        Ok(PathBuf::from(home)
            .join(".model-express")
            .join("config.yaml"))
    }

    /// Expand path with tilde and environment variables
    fn expand_path(path: &Path) -> Result<PathBuf> {
        let path_str = path.to_string_lossy();

        if let Some(stripped) = path_str.strip_prefix("~/") {
            let home = env::var("HOME")
                .or_else(|_| env::var("USERPROFILE"))
                .context("Could not determine home directory")?;
            Ok(PathBuf::from(home).join(stripped))
        } else {
            Ok(path.to_path_buf())
        }
    }

    /// Prompt for cache path
    fn prompt_cache_path() -> Result<PathBuf> {
        loop {
            print!("Enter your local cache mount path [~/.model-express/cache]: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            let path = if input.is_empty() {
                PathBuf::from("~/.model-express/cache")
            } else {
                PathBuf::from(input)
            };

            let expanded_path = Self::expand_path(&path)?;

            // Create directory if it doesn't exist
            if !expanded_path.exists() {
                print!("Directory does not exist. Create it? [Y/n]: ");
                io::stdout().flush()?;

                let mut create_input = String::new();
                io::stdin().read_line(&mut create_input)?;

                if create_input.trim().to_lowercase() != "n" {
                    fs::create_dir_all(&expanded_path).with_context(|| {
                        format!("Failed to create directory: {expanded_path:?}")
                    })?;
                    return Ok(expanded_path);
                }
            } else if expanded_path.is_dir() {
                return Ok(expanded_path);
            } else {
                error!("Path exists but is not a directory");
                continue;
            }
        }
    }

    /// Prompt for server endpoint
    fn prompt_server_endpoint() -> Result<String> {
        print!(
            "Enter your server endpoint [{}]: ",
            Self::get_default_server_endpoint()
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        Ok(if input.is_empty() {
            Self::get_default_server_endpoint()
        } else {
            input.to_string()
        })
    }

    /// Prompt for auto-mount preference
    fn prompt_auto_mount() -> Result<bool> {
        print!("Auto-mount cache on startup? [Y/n]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        Ok(input.trim().to_lowercase() != "n")
    }

    /// Prompt to save configuration
    fn prompt_save_config() -> Result<bool> {
        print!("Save this configuration? [Y/n]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        Ok(input.trim().to_lowercase() != "n")
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
                let model_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

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
            fs::remove_dir_all(&self.local_path)
                .with_context(|| format!("Failed to clear cache: {:?}", self.local_path))?;
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
    use tempfile::TempDir;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_cache_config_from_path() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let config = CacheConfig::from_path(temp_dir.path()).expect("Failed to create config from path");

        assert_eq!(config.local_path, temp_dir.path());
        assert!(config.auto_mount);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_cache_config_save_and_load() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let original_config = CacheConfig {
            local_path: temp_dir.path().join("cache"),
            server_endpoint: "http://localhost:8001".to_string(),
            auto_mount: true,
            timeout_secs: Some(30),
        };

        // Save config
        original_config.save_to_config_file().expect("Failed to save config");

        // Load config
        let loaded_config = CacheConfig::from_config_file().expect("Failed to load config");

        assert_eq!(loaded_config.local_path, original_config.local_path);
        assert_eq!(
            loaded_config.server_endpoint,
            original_config.server_endpoint
        );
        assert_eq!(loaded_config.auto_mount, original_config.auto_mount);
        assert_eq!(loaded_config.timeout_secs, original_config.timeout_secs);
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
}
