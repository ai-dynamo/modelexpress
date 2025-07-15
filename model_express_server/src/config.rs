use clap::Parser;
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::Level;

use crate::cache::CacheEvictionConfig;

/// Command line arguments for the server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ServerArgs {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Server port
    #[arg(short, long, env = "SERVER_PORT")]
    pub port: Option<u16>,

    /// Server host address
    #[arg(long, env = "SERVER_HOST")]
    pub host: Option<String>,

    /// Log level
    #[arg(short, long, env = "LOG_LEVEL")]
    pub log_level: Option<String>,

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
    pub port: u16,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    /// Log format (json, pretty, compact)
    pub format: String,
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

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            file: None,
            structured: false,
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

        // Start with default values
        builder = builder.add_source(Config::try_from(&ServerConfig::default())?);

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
        builder = builder.add_source(
            Environment::with_prefix("MODEL_EXPRESS")
                .try_parsing(true)
                .separator("_")
                .list_separator(","),
        );

        let mut config: ServerConfig = builder.build()?.try_deserialize()?;

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

        if let Some(database_path) = args.database_path {
            config.database.path = database_path;
        }

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate log level
        match self.logging.level.to_lowercase().as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => {
                return Err(ConfigError::Message(format!(
                    "Invalid log level: {}. Must be one of: trace, debug, info, warn, error",
                    self.logging.level
                )));
            }
        }

        // Validate log format
        match self.logging.format.to_lowercase().as_str() {
            "json" | "pretty" | "compact" => {}
            _ => {
                return Err(ConfigError::Message(format!(
                    "Invalid log format: {}. Must be one of: json, pretty, compact",
                    self.logging.format
                )));
            }
        }

        // Validate port range
        if self.server.port == 0 {
            return Err(ConfigError::Message("Server port cannot be 0".to_string()));
        }

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
        match self.logging.level.to_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "info" => Level::INFO,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO, // Default fallback
        }
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
            self.cache.eviction.check_interval_seconds
        );

        info!("Logging Configuration:");
        info!("  Level: {}", self.logging.level);
        info!("  Format: {}", self.logging.format);
        info!("  File: {:?}", self.logging.file);
        info!("  Structured: {}", self.logging.structured);
    }
}
