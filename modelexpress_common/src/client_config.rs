// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cache::CacheConfig;
use crate::config::{ConnectionConfig, LogFormat, LogLevel, load_layered_config};
use anyhow::Result;
use clap::Parser;
use config::ConfigError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Command line arguments for the client
#[derive(Parser, Debug)]
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

    /// Log level
    #[arg(short = 'v', long, env = "MODEL_EXPRESS_LOG_LEVEL", value_enum)]
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
    /// 2. Environment variables
    /// 3. Configuration file
    /// 4. Default values (lowest priority)
    pub fn load(args: ClientArgs) -> Result<Self, ConfigError> {
        // Start with layered config loading (file + env + defaults)
        let mut config =
            load_layered_config(args.config.clone(), "MODEL_EXPRESS", Self::default())?;

        // Override with command line arguments
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

        if let Some(cache_path) = args.cache_path {
            config.cache.local_path = cache_path;
        }

        if let Some(log_level) = args.log_level {
            config.logging.level = log_level;
        }

        if let Some(log_format) = args.log_format {
            config.logging.format = log_format;
        }

        if args.quiet {
            config.logging.quiet = true;
        }

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    #[allow(clippy::collapsible_if)]
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate endpoint
        if self.connection.endpoint.is_empty() {
            return Err(ConfigError::Message(
                "Server endpoint cannot be empty".to_string(),
            ));
        }

        // Validate timeout
        if let Some(timeout) = self.connection.timeout_secs {
            if timeout == 0 {
                return Err(ConfigError::Message(
                    "Timeout must be greater than 0".to_string(),
                ));
            }
        }

        // Validate cache path exists or can be created
        if !self.cache.local_path.exists() {
            if let Err(e) = std::fs::create_dir_all(&self.cache.local_path) {
                return Err(ConfigError::Message(format!(
                    "Cannot create cache directory {:?}: {}",
                    self.cache.local_path, e
                )));
            }
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
mod tests {
    use super::*;

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
}
