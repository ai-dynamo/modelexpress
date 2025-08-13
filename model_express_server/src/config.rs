use chrono::Duration;
use clap::{Parser, ValueEnum};
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::net::SocketAddr;
use std::num::NonZeroU16;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::Level;

use crate::cache::CacheEvictionConfig;

/// Parse a duration string into a `chrono::Duration`.
/// Supports formats like "2h", "30m", "45s", "1d", etc.
pub fn parse_duration_string(value: &str) -> Result<Duration, String> {
    use jiff::{Span, SpanRelativeTo};
    let span = Span::from_str(value).map_err(|err| format!("Invalid duration: {err}"))?;

    // Convert jiff::Span to chrono::Duration
    // For spans with days, we need to specify that days are 24 hours
    let signed_duration = span
        .to_duration(SpanRelativeTo::days_are_24_hours())
        .map_err(|err| format!("Invalid duration: {err}"))?;

    let std_duration = std::time::Duration::try_from(signed_duration)
        .map_err(|err| format!("Invalid duration: {err}"))?;

    Duration::from_std(std_duration).map_err(|err| format!("Duration out of range: {err}"))
}

/// A wrapper around chrono::Duration that can be deserialized from string or seconds
#[derive(Debug, Clone, Serialize)]
pub struct DurationConfig {
    #[serde(with = "duration_serde")]
    duration: Duration,
}

impl DurationConfig {
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }

    pub fn hours(hours: i64) -> Self {
        Self {
            duration: Duration::hours(hours),
        }
    }

    pub fn as_chrono_duration(&self) -> Duration {
        self.duration
    }

    pub fn num_seconds(&self) -> i64 {
        self.duration.num_seconds()
    }
}

impl fmt::Display for DurationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}s", self.duration.num_seconds())
    }
}

impl<'de> Deserialize<'de> for DurationConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct DurationVisitor;

        impl<'de> Visitor<'de> for DurationVisitor {
            type Value = DurationConfig;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter
                    .write_str("a duration string like '2h', '30m', '45s' or number of seconds")
            }

            fn visit_str<E>(self, value: &str) -> Result<DurationConfig, E>
            where
                E: de::Error,
            {
                parse_duration_string(value)
                    .map(DurationConfig::new)
                    .map_err(de::Error::custom)
            }

            fn visit_i64<E>(self, value: i64) -> Result<DurationConfig, E>
            where
                E: de::Error,
            {
                Ok(DurationConfig::new(Duration::seconds(value)))
            }

            fn visit_u64<E>(self, value: u64) -> Result<DurationConfig, E>
            where
                E: de::Error,
            {
                Ok(DurationConfig::new(Duration::seconds(value as i64)))
            }
        }

        deserializer.deserialize_any(DurationVisitor)
    }
}

// Helper module for serializing Duration
mod duration_serde {
    use chrono::Duration;
    use serde::{Serialize, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.num_seconds().serialize(serializer)
    }
}

/// Log level wrapper for clap ValueEnum
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Serialize, Deserialize, Default)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "trace"),
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
        }
    }
}

impl From<LogLevel> for Level {
    fn from(log_level: LogLevel) -> Self {
        match log_level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(LogLevel::Trace),
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warn" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            _ => Err(format!("Invalid log level: {s}")),
        }
    }
}

/// Log format wrapper for clap ValueEnum
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Serialize, Deserialize, Default)]
pub enum LogFormat {
    Json,
    #[default]
    Pretty,
    Compact,
}

impl fmt::Display for LogFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogFormat::Json => write!(f, "json"),
            LogFormat::Pretty => write!(f, "pretty"),
            LogFormat::Compact => write!(f, "compact"),
        }
    }
}

impl FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(LogFormat::Json),
            "pretty" => Ok(LogFormat::Pretty),
            "compact" => Ok(LogFormat::Compact),
            _ => Err(format!("Invalid log format: {s}")),
        }
    }
}

/// Command line arguments for the server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ServerArgs {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Server port
    #[arg(short, long, env = "SERVER_PORT")]
    pub port: Option<NonZeroU16>,

    /// Server host address
    #[arg(long, env = "SERVER_HOST")]
    pub host: Option<String>,

    /// Log level
    #[arg(short, long, env = "LOG_LEVEL", value_enum)]
    pub log_level: Option<LogLevel>,

    /// Log format
    #[arg(long, env = "LOG_FORMAT", value_enum)]
    pub log_format: Option<LogFormat>,

    /// Database file path
    #[arg(short, long, env = "DATABASE_PATH")]
    pub database_path: Option<PathBuf>,

    /// Validate configuration and exit
    #[arg(long)]
    pub validate_config: bool,
}

/// Complete server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    /// Server settings
    pub server: ServerSettings,
    /// Database settings
    pub database: DatabaseSettings,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Server-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: NonZeroU16,
    /// Enable graceful shutdown
    pub graceful_shutdown: bool,
    /// Shutdown timeout in seconds
    pub shutdown_timeout_seconds: u64,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSettings {
    /// Database file path
    pub path: PathBuf,
    /// Enable WAL mode
    pub wal_mode: bool,
    /// Connection pool size
    pub pool_size: u32,
    /// Connection timeout in seconds
    pub connection_timeout_seconds: u64,
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
            port: model_express_common::constants::DEFAULT_GRPC_PORT,
            graceful_shutdown: true,
            shutdown_timeout_seconds: 30,
        }
    }
}

impl Default for DatabaseSettings {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./models.db"),
            wal_mode: true,
            pool_size: 10,
            connection_timeout_seconds: 30,
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
        let mut builder = Config::builder();

        // Add configuration file if specified or if default exists
        if let Some(config_path) = &args.config {
            if config_path.exists() {
                builder = builder.add_source(File::from(config_path.clone()));
            } else {
                return Err(ConfigError::Message(format!(
                    "Configuration file not found: {}",
                    config_path.display()
                )));
            }
        } else {
            // Try to load from default locations
            let default_configs = [
                "model-express.yaml",
                "model-express.yml",
                "/etc/model-express/config.yaml",
                "/etc/model-express/config.yml",
            ];

            for config_path in &default_configs {
                if PathBuf::from(config_path).exists() {
                    builder = builder.add_source(File::with_name(config_path).required(false));
                    break;
                }
            }
        }

        // Add environment variables (with MODEL_EXPRESS_ prefix)
        // Enable debug logging for environment variable parsing
        if std::env::var("RUST_LOG")
            .unwrap_or_default()
            .contains("debug")
        {
            eprintln!("Debug: Parsing environment variables with MODEL_EXPRESS_ prefix...");
            for (key, value) in std::env::vars() {
                if key.starts_with("MODEL_EXPRESS_") {
                    eprintln!("  Found env var: {key}={value}");
                }
            }
        }

        builder = builder.add_source(
            Environment::with_prefix("MODEL_EXPRESS")
                .try_parsing(true)
                .separator("_"),
        );

        // Build the config and merge with defaults
        let config_result = builder.build();
        let mut config = match config_result {
            Ok(c) => {
                // Clone the config before trying to deserialize since try_deserialize consumes it
                match c.clone().try_deserialize::<ServerConfig>() {
                    Ok(full_config) => full_config,
                    Err(deserialize_err) => {
                        // Provide detailed error information about what went wrong
                        eprintln!(
                            "Warning: Configuration deserialization failed with error: {deserialize_err}"
                        );

                        // Check for common issues and provide better guidance
                        if deserialize_err.to_string().contains("missing field") {
                            eprintln!(
                                "  This is likely due to incomplete environment configuration."
                            );
                            eprintln!(
                                "  The configuration system expects all fields to be present or uses defaults."
                            );
                            eprintln!(
                                "  Falling back to default configuration with manual value extraction..."
                            );
                        } else if deserialize_err.to_string().contains("sequence") {
                            eprintln!(
                                "  This was likely caused by the previous list separator configuration."
                            );
                            eprintln!(
                                "  This should be fixed now, but if you still see this error,"
                            );
                            eprintln!("  please check your environment variable values.");
                        } else {
                            eprintln!(
                                "  Unexpected deserialization error. Falling back to manual configuration."
                            );
                        }

                        // Start with default and manually merge what we can
                        let mut default_config = ServerConfig::default();

                        // Apply values from the config if they exist
                        if let Ok(server_port) = c.get::<u16>("server.port") {
                            default_config.server.port = NonZeroU16::new(server_port)
                                .unwrap_or(model_express_common::constants::DEFAULT_GRPC_PORT);
                        }
                        if let Ok(server_host) = c.get::<String>("server.host") {
                            default_config.server.host = server_host;
                        }
                        if let Ok(logging_level) = c.get::<String>("logging.level") {
                            if let Ok(level) = logging_level.parse::<LogLevel>() {
                                default_config.logging.level = level;
                            }
                        }
                        if let Ok(logging_format) = c.get::<String>("logging.format") {
                            if let Ok(format) = logging_format.parse::<LogFormat>() {
                                default_config.logging.format = format;
                            }
                        }
                        if let Ok(database_path) = c.get::<String>("database.path") {
                            default_config.database.path = PathBuf::from(database_path);
                        }
                        if let Ok(cache_directory) = c.get::<String>("cache.directory") {
                            default_config.cache.directory = PathBuf::from(cache_directory);
                        }

                        default_config
                    }
                }
            }
            Err(build_err) => {
                // If building the config fails entirely, show the error and use defaults
                eprintln!("Error: Configuration building failed: {build_err}");
                eprintln!("  This could be due to:");
                eprintln!("  - Invalid environment variable values");
                eprintln!("  - Malformed configuration file");
                eprintln!("  - Environment variables with unexpected types");
                eprintln!("  Falling back to default configuration...");
                ServerConfig::default()
            }
        };

        // Override with command line arguments
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

        if let Some(database_path) = args.database_path {
            config.database.path = database_path;
        }

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate database path parent directory exists
        if let Some(parent) = self.database.path.parent() {
            if !parent.exists() {
                return Err(ConfigError::Message(format!(
                    "Database directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // Validate cache directory
        if let Some(parent) = self.cache.directory.parent() {
            if !parent.exists() {
                return Err(ConfigError::Message(format!(
                    "Cache directory parent does not exist: {}",
                    parent.display()
                )));
            }
        }

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
        info!("  Graceful Shutdown: {}", self.server.graceful_shutdown);
        info!(
            "  Shutdown Timeout: {}s",
            self.server.shutdown_timeout_seconds
        );

        info!("Database Configuration:");
        info!("  Path: {}", self.database.path.display());
        info!("  WAL Mode: {}", self.database.wal_mode);
        info!("  Pool Size: {}", self.database.pool_size);
        info!(
            "  Connection Timeout: {}s",
            self.database.connection_timeout_seconds
        );

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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

        // Test serializing (it serializes as an object with the duration field)
        let duration_config = DurationConfig::hours(1);
        let serialized = serde_json::to_string(&duration_config).expect("Failed to serialize");
        assert_eq!(serialized, r#"{"duration":3600}"#);
    }
}
