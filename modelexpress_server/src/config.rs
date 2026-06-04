// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use config::ConfigError;
use modelexpress_common::config::{LogFormat, LogLevel, load_layered_config};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::num::NonZeroU16;
use std::path::PathBuf;
use tracing::Level;

use crate::cache::CacheEvictionConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum AuthMode {
    #[default]
    Off,
    Enforce,
}

/// A `<namespace>:<serviceaccount>` pair, parsed from `ns:sa`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServiceAccountRef {
    pub namespace: String,
    pub service_account: String,
}

impl std::fmt::Display for ServiceAccountRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.namespace, self.service_account)
    }
}

impl std::str::FromStr for ServiceAccountRef {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (namespace, service_account) = s
            .split_once(':')
            .ok_or_else(|| format!("expected '<namespace>:<serviceaccount>', got '{s}'"))?;
        let namespace = namespace.trim();
        let service_account = service_account.trim();
        if namespace.is_empty() || service_account.is_empty() {
            return Err(format!(
                "namespace and service account must both be non-empty in '{s}'"
            ));
        }
        Ok(Self {
            namespace: namespace.to_string(),
            service_account: service_account.to_string(),
        })
    }
}

/// Comma-separated CLI/env list of `T`, ignoring blank entries.
#[derive(Debug, Clone)]
pub struct CommaList<T>(Vec<T>);

impl<T> CommaList<T> {
    #[must_use]
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T: std::str::FromStr> std::str::FromStr for CommaList<T> {
    type Err = T::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let items = s
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(T::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(items))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    pub mode: Option<AuthMode>,
    pub token_audiences: Vec<String>,
    pub allowed_service_accounts: Vec<ServiceAccountRef>,
    pub cache_ttl_secs: u64,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            mode: None,
            token_audiences: Vec::new(),
            allowed_service_accounts: Vec::new(),
            cache_ttl_secs: 60,
        }
    }
}

impl SecurityConfig {
    #[must_use]
    pub fn resolve_mode(&self) -> AuthMode {
        self.mode.unwrap_or(AuthMode::Off)
    }

    pub fn validate_resolved(&self, mode: AuthMode) -> Result<(), String> {
        if mode == AuthMode::Enforce {
            if self.token_audiences.is_empty() {
                return Err(
                    "security mode 'enforce' requires at least one token audience \
                    (security.token_audiences)"
                        .to_string(),
                );
            }
            if self.allowed_service_accounts.is_empty() {
                return Err(
                    "security mode 'enforce' requires a non-empty service account \
                    allowlist (security.allowed_service_accounts)"
                        .to_string(),
                );
            }
        }
        Ok(())
    }
}

#[derive(clap::Args, Debug, Default)]
pub struct SecurityArgs {
    /// ServiceAccount auth mode (off, enforce). Off by default.
    #[arg(
        long = "security-mode",
        env = "MODEL_EXPRESS_SECURITY_MODE",
        value_enum
    )]
    pub mode: Option<AuthMode>,

    /// Comma-separated SA token audiences the caller's token must carry.
    #[arg(
        long = "security-token-audiences",
        env = "MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES"
    )]
    pub token_audiences: Option<CommaList<String>>,

    /// Comma-separated allowed callers as `<namespace>:<serviceaccount>`.
    #[arg(
        long = "security-allowed-service-accounts",
        env = "MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS"
    )]
    pub allowed_service_accounts: Option<CommaList<ServiceAccountRef>>,

    /// TTL for the verified-token and rejection caches, in seconds.
    #[arg(
        long = "security-cache-ttl-secs",
        env = "MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS"
    )]
    pub cache_ttl_secs: Option<u64>,
}

/// Command line arguments for the server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ServerArgs {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Server port
    #[arg(short, long, env = modelexpress_common::envs::MODEL_EXPRESS_SERVER_PORT)]
    pub port: Option<NonZeroU16>,

    /// Server host address
    #[arg(long, env = modelexpress_common::envs::MODEL_EXPRESS_SERVER_HOST)]
    pub host: Option<String>,

    /// Log level
    #[arg(short, long, env = modelexpress_common::envs::MODEL_EXPRESS_LOG_LEVEL, value_enum)]
    pub log_level: Option<LogLevel>,

    /// Log format
    #[arg(long, env = modelexpress_common::envs::MODEL_EXPRESS_LOG_FORMAT, value_enum)]
    pub log_format: Option<LogFormat>,

    /// Cache directory path
    #[arg(long, env = modelexpress_common::envs::MODEL_EXPRESS_CACHE_DIRECTORY)]
    pub cache_directory: Option<PathBuf>,

    /// Enable cache eviction
    #[arg(long, env = modelexpress_common::envs::MODEL_EXPRESS_CACHE_EVICTION_ENABLED)]
    pub cache_eviction_enabled: Option<bool>,

    #[command(flatten)]
    pub security: SecurityArgs,

    /// Validate configuration and exit
    #[arg(long)]
    pub validate_config: bool,
}

/// Complete server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    /// Server settings
    pub server: ServerSettings,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

/// Server-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: NonZeroU16,
}

/// Cache configuration wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache eviction settings
    pub eviction: CacheEvictionConfig,
    /// Cache directory path
    pub directory: PathBuf,
    /// Maximum cache size in bytes
    pub max_size_bytes: Option<u64>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default)]
    pub level: LogLevel,
    /// Log format (json, pretty, compact)
    #[serde(default)]
    pub format: LogFormat,
    /// Log to file
    pub file: Option<PathBuf>,
    /// Enable structured logging
    pub structured: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: modelexpress_common::constants::DEFAULT_GRPC_PORT,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            eviction: CacheEvictionConfig::default(),
            directory: PathBuf::from("./cache"),
            max_size_bytes: None,
        }
    }
}

impl ServerConfig {
    /// Load configuration from multiple sources in order of precedence:
    /// 1. Command line arguments (highest priority)
    /// 2. Environment variables
    /// 3. Configuration file
    /// 4. Default values (lowest priority)
    pub fn load(args: ServerArgs) -> Result<Self, ConfigError> {
        Self::load_internal(args, false)
    }

    /// Load and validate configuration file strictly without fallbacks.
    /// This method should be used when validating configuration files.
    /// It will return an error if the file has invalid syntax or values.
    pub fn load_and_validate_strict(args: ServerArgs) -> Result<Self, ConfigError> {
        Self::load_internal(args, true)
    }

    /// Internal method to load configuration with optional strict mode
    fn load_internal(args: ServerArgs, strict_mode: bool) -> Result<Self, ConfigError> {
        let mut config = if strict_mode {
            // Use strict loading - fail on any configuration errors
            if let Some(ref config_file) = args.config {
                // Load file strictly without fallbacks
                modelexpress_common::config::validate_config_file(config_file)?
            } else {
                // No config file specified, use defaults
                Self::default()
            }
        } else {
            // Use layered config loading with fallbacks to defaults
            load_layered_config(
                args.config.clone(),
                modelexpress_common::envs::MODEL_EXPRESS_PREFIX,
                Self::default(),
            )?
        };

        // Apply command line overrides (same for both modes)
        if let Some(port) = args.port {
            config.server.port = port;
        }

        if let Some(host) = args.host {
            config.server.host = host;
        }

        if let Some(log_level) = args.log_level {
            config.logging.level = log_level;
        }

        if let Some(log_format) = args.log_format {
            config.logging.format = log_format;
        }

        // Apply cache overrides
        if let Some(cache_directory) = args.cache_directory {
            config.cache.directory = cache_directory;
        }

        if let Some(cache_eviction_enabled) = args.cache_eviction_enabled {
            config.cache.eviction.enabled = cache_eviction_enabled;
        }

        // Apply security overrides
        if let Some(mode) = args.security.mode {
            config.security.mode = Some(mode);
        }
        if let Some(token_audiences) = args.security.token_audiences {
            config.security.token_audiences = token_audiences.into_inner();
        }
        if let Some(allowed) = args.security.allowed_service_accounts {
            config.security.allowed_service_accounts = allowed.into_inner();
        }
        if let Some(cache_ttl_secs) = args.security.cache_ttl_secs {
            config.security.cache_ttl_secs = cache_ttl_secs;
        }

        // Validate the final configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate cache directory
        if let Some(parent) = self.cache.directory.parent()
            && !parent.exists()
        {
            return Err(ConfigError::Message(format!(
                "Cache directory parent does not exist: {}",
                parent.display()
            )));
        }

        let mode = self.security.resolve_mode();
        self.security
            .validate_resolved(mode)
            .map_err(ConfigError::Message)?;

        Ok(())
    }

    /// Get the server socket address
    pub fn socket_addr(&self) -> Result<SocketAddr, ConfigError> {
        let addr = format!("{}:{}", self.server.host, self.server.port);
        addr.parse()
            .map_err(|e| ConfigError::Message(format!("Invalid server address {addr}: {e}")))
    }

    /// Get the logging level as a tracing Level
    pub fn log_level(&self) -> Level {
        self.logging.level.into()
    }

    /// Print the configuration for debugging
    pub fn print_config(&self) {
        use tracing::info;

        info!("Server Configuration:");
        info!("  Host: {}", self.server.host);
        info!("  Port: {}", self.server.port);

        info!("Cache Configuration:");
        info!("  Directory: {}", self.cache.directory.display());
        info!("  Max Size: {:?}", self.cache.max_size_bytes);
        info!("  Eviction Enabled: {}", self.cache.eviction.enabled);
        info!(
            "  Eviction Check Interval: {}s",
            self.cache.eviction.check_interval.num_seconds()
        );

        info!("Logging Configuration:");
        info!("  Level: {}", self.logging.level);
        info!("  Format: {}", self.logging.format);
        info!("  File: {:?}", self.logging.file);
        info!("  Structured: {}", self.logging.structured);

        info!("Security Configuration:");
        info!("  Mode: {:?}", self.security.resolve_mode());
        info!("  Token audiences: {:?}", self.security.token_audiences);
        info!(
            "  Allowed service accounts: {}",
            self.security
                .allowed_service_accounts
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        info!("  Cache TTL: {}s", self.security.cache_ttl_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use clap::Parser;
    use modelexpress_common::config::{DurationConfig, parse_duration_string};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_log_level_enum_parsing() {
        // Test valid log levels
        let valid_levels = [
            ("trace", LogLevel::Trace),
            ("debug", LogLevel::Debug),
            ("info", LogLevel::Info),
            ("warn", LogLevel::Warn),
            ("error", LogLevel::Error),
        ];

        for (level_str, expected_level) in &valid_levels {
            let args = vec!["test", "--log-level", level_str];
            if let Ok(parsed_args) = ServerArgs::try_parse_from(args) {
                assert_eq!(parsed_args.log_level, Some(*expected_level));

                // Test conversion to string
                assert_eq!(expected_level.to_string(), *level_str);

                // Test conversion to tracing::Level
                let tracing_level: Level = (*expected_level).into();
                // Just verify the conversion works - the exact debug format may vary
                match expected_level {
                    LogLevel::Trace => assert_eq!(tracing_level, Level::TRACE),
                    LogLevel::Debug => assert_eq!(tracing_level, Level::DEBUG),
                    LogLevel::Info => assert_eq!(tracing_level, Level::INFO),
                    LogLevel::Warn => assert_eq!(tracing_level, Level::WARN),
                    LogLevel::Error => assert_eq!(tracing_level, Level::ERROR),
                }
            } else {
                panic!("Failed to parse valid log level: {level_str}");
            }
        }
    }

    #[test]
    fn test_log_level_enum_invalid() {
        // Test invalid log level
        let args = vec!["test", "--log-level", "invalid"];
        let result = ServerArgs::try_parse_from(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(LogLevel::Trace.to_string(), "trace");
        assert_eq!(LogLevel::Debug.to_string(), "debug");
        assert_eq!(LogLevel::Info.to_string(), "info");
        assert_eq!(LogLevel::Warn.to_string(), "warn");
        assert_eq!(LogLevel::Error.to_string(), "error");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_log_level_from_str() {
        assert_eq!(
            "trace"
                .parse::<LogLevel>()
                .expect("Failed to parse 'trace'"),
            LogLevel::Trace
        );
        assert_eq!(
            "debug"
                .parse::<LogLevel>()
                .expect("Failed to parse 'debug'"),
            LogLevel::Debug
        );
        assert_eq!(
            "info".parse::<LogLevel>().expect("Failed to parse 'info'"),
            LogLevel::Info
        );
        assert_eq!(
            "warn".parse::<LogLevel>().expect("Failed to parse 'warn'"),
            LogLevel::Warn
        );
        assert_eq!(
            "error"
                .parse::<LogLevel>()
                .expect("Failed to parse 'error'"),
            LogLevel::Error
        );

        // Test case insensitive
        assert_eq!(
            "TRACE"
                .parse::<LogLevel>()
                .expect("Failed to parse 'TRACE'"),
            LogLevel::Trace
        );
        assert_eq!(
            "Info".parse::<LogLevel>().expect("Failed to parse 'Info'"),
            LogLevel::Info
        );

        // Test invalid
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[test]
    fn test_log_format_display() {
        assert_eq!(LogFormat::Json.to_string(), "json");
        assert_eq!(LogFormat::Pretty.to_string(), "pretty");
        assert_eq!(LogFormat::Compact.to_string(), "compact");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_log_format_from_str() {
        assert_eq!(
            "json".parse::<LogFormat>().expect("Failed to parse 'json'"),
            LogFormat::Json
        );
        assert_eq!(
            "pretty"
                .parse::<LogFormat>()
                .expect("Failed to parse 'pretty'"),
            LogFormat::Pretty
        );
        assert_eq!(
            "compact"
                .parse::<LogFormat>()
                .expect("Failed to parse 'compact'"),
            LogFormat::Compact
        );

        // Test case insensitive
        assert_eq!(
            "JSON".parse::<LogFormat>().expect("Failed to parse 'JSON'"),
            LogFormat::Json
        );
        assert_eq!(
            "Pretty"
                .parse::<LogFormat>()
                .expect("Failed to parse 'Pretty'"),
            LogFormat::Pretty
        );

        // Test invalid
        assert!("invalid".parse::<LogFormat>().is_err());
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_parse_duration_string() {
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
            parse_duration_string("2h")
                .expect("Failed to parse 2h")
                .num_seconds(),
            2 * 3600
        );
        assert_eq!(
            parse_duration_string("2h30m")
                .expect("Failed to parse 2h30m")
                .num_seconds(),
            2 * 3600 + 30 * 60
        );

        // Test invalid format
        assert!(parse_duration_string("invalid").is_err());
    }

    #[test]
    fn test_duration_config() {
        // Test creation
        let duration_config = DurationConfig::new(Duration::hours(2));
        assert_eq!(duration_config.num_seconds(), 2 * 3600);
        assert_eq!(duration_config.as_chrono_duration(), Duration::hours(2));

        // Test hours constructor
        let duration_config = DurationConfig::hours(3);
        assert_eq!(duration_config.num_seconds(), 3 * 3600);

        // Test display
        assert_eq!(duration_config.to_string(), "10800s");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_duration_config_serde() {
        // Test deserializing from string
        let json = r#""2h""#;
        let duration_config: DurationConfig = serde_json::from_str(json).expect("Failed to parse");
        assert_eq!(duration_config.num_seconds(), 2 * 3600);

        // Test deserializing from number (seconds)
        let json = r#"3600"#;
        let duration_config: DurationConfig = serde_json::from_str(json).expect("Failed to parse");
        assert_eq!(duration_config.num_seconds(), 3600);

        // Test serializing (it should serialize as just the number)
        let duration_config = DurationConfig::hours(1);
        let serialized = serde_json::to_string(&duration_config).expect("Failed to serialize");
        assert_eq!(serialized, r#"3600"#);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_server_config_load_and_validate_strict_valid_config() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config_file = temp_dir.path().join("valid_server_config.yaml");

        let valid_config = r#"
            server:
              host: "127.0.0.1"
              port: 8002
            cache:
              eviction:
                enabled: false
                policy:
                  type: lru
                  unused_threshold: "3d"
                  max_models: 10
                  min_free_space_bytes: 1000000
                check_interval: "30m"
              directory: "./test_cache"
              max_size_bytes: 5000000
            logging:
              level: Debug
              format: Json
              file: null
              structured: true
        "#;

        fs::write(&config_file, valid_config).expect("Failed to write config file");

        let args = ServerArgs {
            config: Some(config_file),
            port: None,
            host: None,
            log_level: None,
            log_format: None,
            cache_directory: None,
            cache_eviction_enabled: None,
            security: SecurityArgs::default(),
            validate_config: false,
        };

        let result = ServerConfig::load_and_validate_strict(args);
        assert!(result.is_ok());

        let config = result.expect("Expected successful config parsing");
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port.get(), 8002);
        assert!(!config.cache.eviction.enabled);
        assert_eq!(config.logging.level, LogLevel::Debug);
        assert_eq!(config.logging.format, LogFormat::Json);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_server_config_load_and_validate_strict_invalid_config() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config_file = temp_dir.path().join("invalid_server_config.yaml");

        let invalid_config = r#"
            server:
              host: "127.0.0.1"
              port: 8002
            cache:
              eviction:
                enabled: "not_a_boolean"  # Invalid type
        "#;

        fs::write(&config_file, invalid_config).expect("Failed to write config file");

        let args = ServerArgs {
            config: Some(config_file),
            port: None,
            host: None,
            log_level: None,
            log_format: None,
            cache_directory: None,
            cache_eviction_enabled: None,
            security: SecurityArgs::default(),
            validate_config: false,
        };

        let result = ServerConfig::load_and_validate_strict(args);
        assert!(result.is_err());
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_server_config_load_and_validate_strict_with_cli_overrides() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config_file = temp_dir.path().join("override_test_config.yaml");

        let base_config = r#"
            server:
              host: "127.0.0.1"
              port: 8002
            cache:
              eviction:
                enabled: true
                policy:
                  type: lru
                  unused_threshold: "1d"
                  max_models: null
                  min_free_space_bytes: null
                check_interval: "1h"
              directory: "./cache"
              max_size_bytes: null
            logging:
              level: Info
              format: Pretty
              file: null
              structured: false
        "#;

        fs::write(&config_file, base_config).expect("Failed to write config file");

        let args = ServerArgs {
            config: Some(config_file),
            port: Some(NonZeroU16::new(9000).expect("9000 is non-zero")),
            host: Some("0.0.0.0".to_string()),
            log_level: Some(LogLevel::Error),
            log_format: Some(LogFormat::Json),
            cache_directory: Some(PathBuf::from("/tmp/override_cache")),
            cache_eviction_enabled: Some(false),
            security: SecurityArgs::default(),
            validate_config: false,
        };

        let result = ServerConfig::load_and_validate_strict(args);
        assert!(result.is_ok());

        let config = result.expect("Expected successful config parsing");
        // CLI overrides should be applied
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port.get(), 9000);
        assert_eq!(config.logging.level, LogLevel::Error);
        assert_eq!(config.logging.format, LogFormat::Json);
        assert_eq!(config.cache.directory, PathBuf::from("/tmp/override_cache"));
        assert!(!config.cache.eviction.enabled);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_server_config_load_and_validate_strict_no_config_file() {
        let args = ServerArgs {
            config: None,
            port: Some(NonZeroU16::new(9001).expect("9001 is non-zero")),
            host: Some("localhost".to_string()),
            log_level: Some(LogLevel::Warn),
            log_format: None,
            cache_directory: None,
            cache_eviction_enabled: None,
            security: SecurityArgs::default(),
            validate_config: false,
        };

        // When no config file is specified, it should fall back to normal loading
        let result = ServerConfig::load_and_validate_strict(args);
        assert!(result.is_ok());

        let config = result.expect("Expected successful config parsing");
        assert_eq!(config.server.host, "localhost");
        assert_eq!(config.server.port.get(), 9001);
        assert_eq!(config.logging.level, LogLevel::Warn);
    }

    #[test]
    fn security_defaults_to_off() {
        let security = SecurityConfig::default();
        assert_eq!(security.resolve_mode(), AuthMode::Off);
        assert!(security.token_audiences.is_empty());
        assert!(security.allowed_service_accounts.is_empty());
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn service_account_ref_parses_namespace_and_sa() {
        let parsed: ServiceAccountRef = "vllm:worker".parse().expect("valid ns:sa");
        assert_eq!(parsed.namespace, "vllm");
        assert_eq!(parsed.service_account, "worker");
        assert_eq!(parsed.to_string(), "vllm:worker");
        let trimmed: ServiceAccountRef = "  ns : sa ".parse().expect("trims");
        assert_eq!(trimmed, "ns:sa".parse().expect("valid"));
    }

    #[test]
    fn service_account_ref_rejects_malformed() {
        assert!("noseparator".parse::<ServiceAccountRef>().is_err());
        assert!(":sa".parse::<ServiceAccountRef>().is_err());
        assert!("ns:".parse::<ServiceAccountRef>().is_err());
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn comma_list_parses_and_skips_blanks() {
        let list: CommaList<ServiceAccountRef> =
            "a:one, b:two ,, c:three".parse().expect("valid list");
        let items = list.into_inner();
        assert_eq!(items.len(), 3);
        assert_eq!(items[1].namespace, "b");
        assert_eq!(items[2].service_account, "three");

        let empty: CommaList<String> = "  ".parse().expect("empty");
        assert!(empty.into_inner().is_empty());
    }

    #[test]
    fn enforce_requires_audience_and_allowlist() {
        let mut security = SecurityConfig {
            mode: Some(AuthMode::Enforce),
            ..SecurityConfig::default()
        };
        assert!(security.validate_resolved(AuthMode::Enforce).is_err());

        security.token_audiences = vec!["modelexpress".to_string()];
        assert!(security.validate_resolved(AuthMode::Enforce).is_err());

        security.allowed_service_accounts = vec![ServiceAccountRef {
            namespace: "vllm".to_string(),
            service_account: "worker".to_string(),
        }];
        assert!(security.validate_resolved(AuthMode::Enforce).is_ok());
    }
}
