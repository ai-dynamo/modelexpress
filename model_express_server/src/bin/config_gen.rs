use clap::Parser;
use model_express_server::config::ServerConfig;
use std::fs;
use std::num::NonZeroU16;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Generate or migrate ModelExpress configuration file"
)]
struct ConfigGenArgs {
    /// Output file path
    #[arg(short, long, default_value = "model-express.yaml")]
    output: PathBuf,

    /// Input configuration file to migrate/upgrade
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output format (yaml only for now)
    #[arg(short, long, default_value = "yaml")]
    format: String,

    /// Overwrite existing file
    #[arg(long)]
    overwrite: bool,

    /// Show differences between input and output configurations
    #[arg(long)]
    show_diff: bool,

    /// Dry run - show what would be generated without writing the file
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ConfigGenArgs::parse();

    // Check if output file exists and overwrite is not set (unless dry run)
    if !args.dry_run && args.output.exists() && !args.overwrite {
        eprintln!(
            "Error: File {} already exists. Use --overwrite to replace it.",
            args.output.display()
        );
        std::process::exit(1);
    }

    // Load configuration - either from input file or default
    let config = if let Some(input_path) = &args.input {
        load_existing_config(input_path)?
    } else {
        ServerConfig::default()
    };

    // Generate the output content
    let content = match args.format.to_lowercase().as_str() {
        "yaml" | "yml" => serde_yaml::to_string(&config)?,
        _ => {
            eprintln!(
                "Error: Unsupported format '{}'. Use yaml only.",
                args.format
            );
            std::process::exit(1);
        }
    };

    // Show differences if requested
    if args.show_diff {
        if let Some(input_path) = args.input.as_ref() {
            show_configuration_diff(input_path, &content)?;
        }
    }

    // If dry run, just print the content
    if args.dry_run {
        println!("Generated configuration (dry run):");
        println!("{}", "=".repeat(50));
        println!("{content}");
        println!("{}", "=".repeat(50));

        if let Some(input_path) = &args.input {
            println!("Migrated from: {}", input_path.display());
        } else {
            println!("Generated from default configuration");
        }
        return Ok(());
    }

    // Create parent directory if it doesn't exist
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write the configuration file
    fs::write(&args.output, content)?;

    println!("Configuration file generated: {}", args.output.display());

    if let Some(input_path) = &args.input {
        println!("Migrated from: {}", input_path.display());
        println!("Review the generated file for any new configuration options.");
    } else {
        println!("Edit this file to customize your ModelExpress server settings.");
    }

    Ok(())
}

/// Load an existing configuration file and merge it with defaults
fn load_existing_config(input_path: &PathBuf) -> Result<ServerConfig, Box<dyn std::error::Error>> {
    if !input_path.exists() {
        return Err(format!(
            "Input configuration file not found: {}",
            input_path.display()
        )
        .into());
    }

    println!(
        "Loading existing configuration from: {}",
        input_path.display()
    );

    // Read the existing file content
    let content = fs::read_to_string(input_path)?;

    // Try to parse as the current config format first
    match serde_yaml::from_str::<ServerConfig>(&content) {
        Ok(config) => {
            println!("Successfully loaded existing configuration");
            Ok(config)
        }
        Err(e) => {
            eprintln!("Warning: Could not parse existing config as current format: {e}");
            eprintln!("Attempting to migrate from older format...");

            // Try to parse as a generic YAML value and merge with defaults
            match serde_yaml::from_str::<serde_yaml::Value>(&content) {
                Ok(yaml_value) => {
                    let mut default_config = ServerConfig::default();
                    merge_yaml_into_config(&mut default_config, &yaml_value)?;
                    println!("Successfully migrated configuration from older format");
                    Ok(default_config)
                }
                Err(parse_err) => {
                    Err(format!("Could not parse configuration file: {parse_err}").into())
                }
            }
        }
    }
}

/// Merge YAML values into the default configuration
fn merge_yaml_into_config(
    config: &mut ServerConfig,
    yaml: &serde_yaml::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let serde_yaml::Value::Mapping(map) = yaml {
        for (key, value) in map {
            if let serde_yaml::Value::String(key_str) = key {
                match key_str.as_str() {
                    "server" => merge_server_config(&mut config.server, value)?,
                    "database" => merge_database_config(&mut config.database, value)?,
                    "cache" => merge_cache_config(&mut config.cache, value)?,
                    "logging" => merge_logging_config(&mut config.logging, value)?,
                    _ => {
                        eprintln!("Warning: Unknown configuration key '{key_str}', ignoring");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Helper functions to merge specific config sections
fn merge_server_config(
    server: &mut model_express_server::config::ServerSettings,
    value: &serde_yaml::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let serde_yaml::Value::Mapping(map) = value {
        for (key, val) in map {
            if let serde_yaml::Value::String(key_str) = key {
                match key_str.as_str() {
                    "host" => {
                        if let serde_yaml::Value::String(host) = val {
                            server.host = host.clone();
                        }
                    }
                    "port" => {
                        if let serde_yaml::Value::Number(n) = val
                            && let Some(port_u64) = n.as_u64()
                            && let Ok(port_u16) = u16::try_from(port_u64)
                            && let Some(port) = NonZeroU16::new(port_u16)
                        {
                            server.port = port;
                        }
                    }
                    "graceful_shutdown" => {
                        if let serde_yaml::Value::Bool(b) = val {
                            server.graceful_shutdown = *b;
                        }
                    }
                    "shutdown_timeout_seconds" => {
                        if let serde_yaml::Value::Number(n) = val
                            && let Some(timeout) = n.as_u64()
                        {
                            server.shutdown_timeout_seconds = timeout;
                        }
                    }
                    _ => {
                        eprintln!("Warning: Unknown configuration key '{key_str}', ignoring");
                    }
                }
            }
        }
    }
    Ok(())
}

fn merge_database_config(
    database: &mut model_express_server::config::DatabaseSettings,
    value: &serde_yaml::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let serde_yaml::Value::Mapping(map) = value {
        for (key, val) in map {
            if let serde_yaml::Value::String(key_str) = key {
                match key_str.as_str() {
                    "path" => {
                        if let serde_yaml::Value::String(path) = val {
                            database.path = PathBuf::from(path);
                        }
                    }
                    "wal_mode" => {
                        if let serde_yaml::Value::Bool(b) = val {
                            database.wal_mode = *b;
                        }
                    }
                    "pool_size" => {
                        if let serde_yaml::Value::Number(n) = val
                            && let Some(size) = n.as_u64()
                        {
                            database.pool_size = size as u32;
                        }
                    }
                    "connection_timeout_seconds" => {
                        if let serde_yaml::Value::Number(n) = val
                            && let Some(timeout) = n.as_u64()
                        {
                            database.connection_timeout_seconds = timeout;
                        }
                    }
                    _ => {
                        eprintln!("Warning: Unknown configuration key '{key_str}', ignoring");
                    }
                }
            }
        }
    }
    Ok(())
}

fn merge_cache_config(
    cache: &mut model_express_server::config::CacheConfig,
    value: &serde_yaml::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let serde_yaml::Value::Mapping(map) = value {
        for (key, val) in map {
            if let serde_yaml::Value::String(key_str) = key {
                match key_str.as_str() {
                    "directory" => {
                        if let serde_yaml::Value::String(dir) = val {
                            cache.directory = PathBuf::from(dir);
                        }
                    }
                    "max_size_bytes" => {
                        if let serde_yaml::Value::Number(n) = val {
                            cache.max_size_bytes = n.as_u64();
                        } else if val.is_null() {
                            cache.max_size_bytes = None;
                        }
                    }
                    "eviction" => {
                        // This would need more complex merging for nested structures
                        // For now, try to deserialize the whole eviction section
                        if let Ok(eviction) = serde_yaml::from_value(val.clone()) {
                            cache.eviction = eviction;
                        }
                    }
                    _ => {
                        eprintln!("Warning: Unknown configuration key '{key_str}', ignoring");
                    }
                }
            }
        }
    }
    Ok(())
}

fn merge_logging_config(
    logging: &mut model_express_server::config::LoggingConfig,
    value: &serde_yaml::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let serde_yaml::Value::Mapping(map) = value {
        for (key, val) in map {
            if let serde_yaml::Value::String(key_str) = key {
                match key_str.as_str() {
                    "level" => {
                        if let serde_yaml::Value::String(level) = val {
                            match level.parse() {
                                Ok(parsed_level) => logging.level = parsed_level,
                                Err(_) => eprintln!(
                                    "Warning: Invalid log level '{level}', keeping default"
                                ),
                            }
                        }
                    }
                    "format" => {
                        if let serde_yaml::Value::String(format) = val {
                            match format.parse() {
                                Ok(parsed_format) => logging.format = parsed_format,
                                Err(_) => eprintln!(
                                    "Warning: Invalid log format '{format}', keeping default"
                                ),
                            }
                        }
                    }
                    "file" => {
                        if let serde_yaml::Value::String(file) = val {
                            logging.file = Some(PathBuf::from(file));
                        } else if val.is_null() {
                            logging.file = None;
                        }
                    }
                    "structured" => {
                        if let serde_yaml::Value::Bool(b) = val {
                            logging.structured = *b;
                        }
                    }
                    _ => {
                        eprintln!("Warning: Unknown configuration key '{key_str}', ignoring");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Show a diff between input and output configurations
fn show_configuration_diff(
    input_path: &PathBuf,
    output_content: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_content = fs::read_to_string(input_path)?;

    println!("Configuration differences:");
    println!("{}", "=".repeat(50));
    println!("Input file: {}", input_path.display());
    println!("--- Original");
    println!("+++ Generated");

    // Simple line-by-line diff
    let input_lines: Vec<&str> = input_content.lines().collect();
    let output_lines: Vec<&str> = output_content.lines().collect();

    let max_lines = input_lines.len().max(output_lines.len());

    for i in 0..max_lines {
        let input_line = input_lines.get(i).unwrap_or(&"");
        let output_line = output_lines.get(i).unwrap_or(&"");

        if input_line != output_line {
            if !input_line.is_empty() {
                println!("- {input_line}");
            }
            if !output_line.is_empty() {
                println!("+ {output_line}");
            }
        }
    }

    println!("{}", "=".repeat(50));
    Ok(())
}
