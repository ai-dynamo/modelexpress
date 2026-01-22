// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cache::CacheConfig;
use crate::config::{ConnectionConfig, LogFormat, LogLevel, load_layered_config};
use anyhow::Result;
use clap::Parser;
use config::ConfigError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Shared command line arguments for the ModelExpress client.
///
/// # Adding New Arguments
///
/// This struct is the **single source of truth** for client CLI arguments and environment
/// variables. It is shared between:
/// - The `modelexpress-cli` binary (via `#[command(flatten)]` in the `Cli` struct)
/// - Any other client binaries that need these arguments
/// - The `ClientConfig::load()` function which applies these values
///
/// When adding a new argument:
/// 1. Add the field here with appropriate `#[arg(...)]` attributes
/// 2. Include `env = "MODEL_EXPRESS_..."` for environment variable support
/// 3. Update `ClientConfig::load()` to apply the new argument to the config
/// 4. Add tests in the `tests` module below
/// 5. Update CLI.md documentation if applicable
///
/// # Short Flags
///
/// Avoid using `-v` as a short flag here - it's reserved for the CLI's `--verbose` flag
/// which uses `-v`, `-vv`, `-vvv` counting. The CLI embeds this struct via flatten,
/// so short flag conflicts will cause runtime panics.
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct ClientArgs {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Server endpoint
    #[arg(short, long, env = "MODEL_EXPRESS_ENDPOINT")]
    pub endpoint: Option<String>,

    /// Request timeout in seconds
    #[arg(short, long, env = "MODEL_EXPRESS_TIMEOUT")]
    pub timeout: Option<u64>,

    /// Cache path override
    #[arg(long, env = "MODEL_EXPRESS_CACHE_PATH")]
    pub cache_path: Option<PathBuf>,

    /// Log level (no short flag to avoid conflict with CLI's -v/--verbose)
    #[arg(long, env = "MODEL_EXPRESS_LOG_LEVEL", value_enum)]
    pub log_level: Option<LogLevel>,

    /// Log format
    #[arg(long, env = "MODEL_EXPRESS_LOG_FORMAT", value_enum)]
    pub log_format: Option<LogFormat>,

    /// Quiet mode (suppress all output except errors)
    #[arg(long, short = 'q')]
    pub quiet: bool,

    /// Maximum number of retries
    #[arg(long, env = "MODEL_EXPRESS_MAX_RETRIES")]
    pub max_retries: Option<u32>,

    /// Retry delay in seconds
    #[arg(long, env = "MODEL_EXPRESS_RETRY_DELAY")]
    pub retry_delay: Option<u64>,

    /// Disable shared storage mode (will transfer files from server to client)
    #[arg(long, env = "MODEL_EXPRESS_NO_SHARED_STORAGE")]
    pub no_shared_storage: bool,

    /// Chunk size in bytes for file transfer when shared storage is disabled
    #[arg(long, env = "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE")]
    pub transfer_chunk_size: Option<usize>,
}

/// Complete client configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientConfig {
    /// Connection settings
    pub connection: ConnectionConfig,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Logging configuration for the client
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default)]
    pub level: LogLevel,
    /// Log format
    #[serde(default)]
    pub format: LogFormat,
    /// Quiet mode
    pub quiet: bool,
}

impl ClientConfig {
    /// Load configuration from multiple sources in order of precedence:
    /// 1. Command line arguments (highest priority)
    /// 2. Environment variables (handled by clap's `env` attribute on `ClientArgs`)
    /// 3. Configuration file
    /// 4. Default values (lowest priority)
    ///
    /// # Adding New Arguments
    ///
    /// When you add a new field to `ClientArgs`:
    /// 1. Add the corresponding override logic below in the "Apply CLI argument overrides" section
    /// 2. Map the `ClientArgs` field to the appropriate `ClientConfig` field
    /// 3. Add a test in the `tests` module to verify the override works
    pub fn load(args: ClientArgs) -> Result<Self, ConfigError> {
        // Start with layered config loading (file + env + defaults)
        let mut config =
            load_layered_config(args.config.clone(), "MODEL_EXPRESS", Self::default())?;

        // ==================== APPLY CLI ARGUMENT OVERRIDES ====================
        // When adding a new field to ClientArgs, add the override logic here.
        // These overrides apply CLI arguments (which include env vars via clap)
        // on top of the config file values.

        // Connection settings
        if let Some(endpoint) = args.endpoint {
            config.connection.endpoint = endpoint;
        }

        if let Some(timeout) = args.timeout {
            config.connection.timeout_secs = Some(timeout);
        }

        if let Some(max_retries) = args.max_retries {
            config.connection.max_retries = Some(max_retries);
        }

        if let Some(retry_delay) = args.retry_delay {
            config.connection.retry_delay_secs = Some(retry_delay);
        }

        // Cache settings
        if let Some(cache_path) = args.cache_path {
            config.cache.local_path = cache_path;
        }

        if args.no_shared_storage {
            config.cache.shared_storage = false;
        }

        if let Some(chunk_size) = args.transfer_chunk_size {
            config.cache.transfer_chunk_size = chunk_size;
        }

        // Logging settings
        if let Some(log_level) = args.log_level {
            config.logging.level = log_level;
        }

        if let Some(log_format) = args.log_format {
            config.logging.format = log_format;
        }

        if args.quiet {
            config.logging.quiet = true;
        }

        // ==================== END CLI ARGUMENT OVERRIDES ====================

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate endpoint
        if self.connection.endpoint.is_empty() {
            return Err(ConfigError::Message(
                "Server endpoint cannot be empty".to_string(),
            ));
        }

        // Validate timeout
        if let Some(timeout) = self.connection.timeout_secs
            && timeout == 0
        {
            return Err(ConfigError::Message(
                "Timeout must be greater than 0".to_string(),
            ));
        }

        // Validate cache path exists or can be created
        if !self.cache.local_path.exists()
            && let Err(e) = std::fs::create_dir_all(&self.cache.local_path)
        {
            return Err(ConfigError::Message(format!(
                "Cannot create cache directory {:?}: {}",
                self.cache.local_path, e
            )));
        }

        Ok(())
    }

    /// Get the gRPC endpoint for backward compatibility
    pub fn grpc_endpoint(&self) -> &str {
        &self.connection.endpoint
    }

    /// Get the timeout in seconds for backward compatibility
    pub fn timeout_secs(&self) -> Option<u64> {
        self.connection.timeout_secs
    }

    /// Create a simple client config for testing
    pub fn for_testing(endpoint: impl Into<String>) -> Self {
        Self {
            connection: ConnectionConfig::new(endpoint),
            cache: CacheConfig::default(),
            logging: LoggingConfig::default(),
        }
    }

    /// Apply cache path override if provided
    pub fn with_cache_path(mut self, cache_path: Option<PathBuf>) -> Self {
        if let Some(path) = cache_path {
            self.cache.local_path = path;
        }
        self
    }

    /// Set timeout for the connection
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.connection.timeout_secs = Some(timeout_secs);
        self
    }

    /// Set the server endpoint for both connection and cache
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.connection.endpoint = endpoint.clone();
        self.cache.server_endpoint = endpoint;
        self
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::constants;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert!(config.connection.endpoint.contains("8001"));
        assert_eq!(config.connection.timeout_secs, Some(30));
        assert!(!config.logging.quiet);
    }

    #[test]
    fn test_client_config_for_testing() {
        let config = ClientConfig::for_testing("http://test.example.com:1234");
        assert_eq!(config.connection.endpoint, "http://test.example.com:1234");
    }

    #[test]
    fn test_client_config_with_endpoint() {
        let config =
            ClientConfig::default().with_endpoint("http://custom.example.com:5678".to_string());

        assert_eq!(config.connection.endpoint, "http://custom.example.com:5678");
        assert_eq!(
            config.cache.server_endpoint,
            "http://custom.example.com:5678"
        );
    }

    #[test]
    fn test_client_config_validation() {
        let mut config = ClientConfig::default();
        assert!(config.validate().is_ok());

        config.connection.endpoint = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_client_config_backward_compatibility() {
        let config = ClientConfig::for_testing("http://test.com:8080");
        assert_eq!(config.grpc_endpoint(), "http://test.com:8080");
        assert_eq!(config.timeout_secs(), Some(30));
    }

    #[test]
    fn test_client_config_shared_storage_defaults() {
        let config = ClientConfig::default();
        assert!(config.cache.shared_storage);
        assert_eq!(
            config.cache.transfer_chunk_size,
            constants::DEFAULT_TRANSFER_CHUNK_SIZE
        );
    }

    #[test]
    fn test_client_config_shared_storage_override() {
        let mut config = ClientConfig::default();
        config.cache.shared_storage = false;
        config.cache.transfer_chunk_size = 64 * 1024;

        assert!(!config.cache.shared_storage);
        assert_eq!(config.cache.transfer_chunk_size, 64 * 1024);
    }

    #[test]
    fn test_client_args_parse_defaults() {
        // Test that ClientArgs can be parsed with no arguments (uses defaults)
        let args = ClientArgs::try_parse_from(["test"]).expect("Failed to parse empty args");

        assert!(args.endpoint.is_none());
        assert!(args.timeout.is_none());
        assert!(args.cache_path.is_none());
        assert!(!args.quiet);
        assert!(!args.no_shared_storage);
        assert!(args.transfer_chunk_size.is_none());
    }

    #[test]
    fn test_client_args_parse_cli_flags() {
        // Test parsing various CLI flags
        let args = ClientArgs::try_parse_from([
            "test",
            "--endpoint",
            "http://custom:9000",
            "--timeout",
            "60",
            "--quiet",
            "--no-shared-storage",
            "--transfer-chunk-size",
            "1048576",
        ])
        .expect("Failed to parse CLI args");

        assert_eq!(args.endpoint, Some("http://custom:9000".to_string()));
        assert_eq!(args.timeout, Some(60));
        assert!(args.quiet);
        assert!(args.no_shared_storage);
        assert_eq!(args.transfer_chunk_size, Some(1048576));
    }

    #[test]
    fn test_client_args_short_flags() {
        // Test short flag variants (-e for endpoint, -t for timeout, -q for quiet)
        let args =
            ClientArgs::try_parse_from(["test", "-e", "http://short:8000", "-t", "45", "-q"])
                .expect("Failed to parse short flags");

        assert_eq!(args.endpoint, Some("http://short:8000".to_string()));
        assert_eq!(args.timeout, Some(45));
        assert!(args.quiet);
    }

    #[test]
    fn test_client_args_log_level() {
        // Test --log-level flag (no short flag to avoid conflict with CLI's -v)
        let args = ClientArgs::try_parse_from(["test", "--log-level", "debug"])
            .expect("Failed to parse log level");

        assert_eq!(args.log_level, Some(LogLevel::Debug));
    }

    #[test]
    fn test_client_config_load_applies_cli_args() {
        // Test that ClientConfig::load() properly applies CLI arguments
        let args = ClientArgs {
            config: None,
            endpoint: Some("http://cli-override:7777".to_string()),
            timeout: Some(120),
            cache_path: None,
            log_level: None,
            log_format: None,
            quiet: true,
            max_retries: Some(5),
            retry_delay: Some(10),
            no_shared_storage: true,
            transfer_chunk_size: Some(2097152),
        };

        let config = ClientConfig::load(args).expect("Failed to load config");

        assert_eq!(config.connection.endpoint, "http://cli-override:7777");
        assert_eq!(config.connection.timeout_secs, Some(120));
        assert!(config.logging.quiet);
        assert_eq!(config.connection.max_retries, Some(5));
        assert_eq!(config.connection.retry_delay_secs, Some(10));
        assert!(!config.cache.shared_storage);
        assert_eq!(config.cache.transfer_chunk_size, 2097152);
    }
}
