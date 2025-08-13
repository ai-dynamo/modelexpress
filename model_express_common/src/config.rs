use chrono::Duration;
use clap::ValueEnum;
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::Level;

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

/// Base trait for configuration loading with layered approach
pub trait ConfigLoader<T> {
    /// Load configuration from multiple sources in order of precedence:
    /// 1. Command line arguments (highest priority)
    /// 2. Environment variables
    /// 3. Configuration file
    /// 4. Default values (lowest priority)
    fn load_layered(
        config_file: Option<PathBuf>,
        env_prefix: &str,
        defaults: T,
    ) -> Result<T, ConfigError>
    where
        T: serde::de::DeserializeOwned + Default;
}

/// Default implementation of layered configuration loading
pub fn load_layered_config<T>(
    config_file: Option<PathBuf>,
    env_prefix: &str,
    defaults: T,
) -> Result<T, ConfigError>
where
    T: serde::de::DeserializeOwned + Default,
{
    let mut builder = Config::builder();

    // Add configuration file if specified or if default exists
    if let Some(config_path) = &config_file {
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

    // Add environment variables
    builder = builder.add_source(
        Environment::with_prefix(env_prefix)
            .try_parsing(true)
            .separator("_"),
    );

    // Build the config and merge with defaults
    let config_result = builder.build();

    match config_result {
        Ok(c) => {
            // Try to deserialize the full config
            match c.try_deserialize::<T>() {
                Ok(full_config) => Ok(full_config),
                Err(_) => {
                    // If deserialization fails, fall back to defaults
                    // This is a safe fallback for partial configurations
                    Ok(defaults)
                }
            }
        }
        Err(_) => {
            // If building the config fails entirely, use defaults
            Ok(defaults)
        }
    }
}

/// Common configuration for client connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// The endpoint to connect to
    pub endpoint: String,

    /// Timeout in seconds for requests
    pub timeout_secs: Option<u64>,

    /// Maximum retries for failed requests
    pub max_retries: Option<u32>,

    /// Retry delay in seconds
    pub retry_delay_secs: Option<u64>,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            endpoint: format!("http://localhost:{}", crate::constants::DEFAULT_GRPC_PORT),
            timeout_secs: Some(crate::constants::DEFAULT_TIMEOUT_SECS),
            max_retries: Some(3),
            retry_delay_secs: Some(1),
        }
    }
}

impl ConnectionConfig {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout_secs: Some(crate::constants::DEFAULT_TIMEOUT_SECS),
            max_retries: Some(3),
            retry_delay_secs: Some(1),
        }
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    pub fn with_retries(mut self, max_retries: u32, delay_secs: u64) -> Self {
        self.max_retries = Some(max_retries);
        self.retry_delay_secs = Some(delay_secs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_config_from_string() {
        match parse_duration_string("2h") {
            Ok(duration) => assert_eq!(duration.num_hours(), 2),
            Err(e) => panic!("Failed to parse duration '2h': {e}"),
        }
    }

    #[test]
    fn test_log_level_from_string() {
        match "info".parse::<LogLevel>() {
            Ok(level) => assert_eq!(level, LogLevel::Info),
            Err(e) => panic!("Failed to parse 'info' as LogLevel: {e}"),
        }
        match "debug".parse::<LogLevel>() {
            Ok(level) => assert_eq!(level, LogLevel::Debug),
            Err(e) => panic!("Failed to parse 'debug' as LogLevel: {e}"),
        }
    }

    #[test]
    fn test_log_format_from_string() {
        match "json".parse::<LogFormat>() {
            Ok(format) => assert_eq!(format, LogFormat::Json),
            Err(e) => panic!("Failed to parse 'json' as LogFormat: {e}"),
        }
        match "pretty".parse::<LogFormat>() {
            Ok(format) => assert_eq!(format, LogFormat::Pretty),
            Err(e) => panic!("Failed to parse 'pretty' as LogFormat: {e}"),
        }
    }

    #[test]
    fn test_connection_config_default() {
        let config = ConnectionConfig::default();
        assert!(config.endpoint.contains("8001"));
        assert_eq!(config.timeout_secs, Some(30));
    }

    #[test]
    fn test_connection_config_builder() {
        let config = ConnectionConfig::new("http://test.com:8080")
            .with_timeout(60)
            .with_retries(5, 2);

        assert_eq!(config.endpoint, "http://test.com:8080");
        assert_eq!(config.timeout_secs, Some(60));
        assert_eq!(config.max_retries, Some(5));
        assert_eq!(config.retry_delay_secs, Some(2));
    }
}
