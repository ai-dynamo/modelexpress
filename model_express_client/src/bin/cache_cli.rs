use clap::{Parser, Subcommand};
use model_express_client::{Client, ClientConfig, ModelProvider};
use model_express_common::cache::{CacheConfig, CacheStats};
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, warn, error, debug};
use std::io::Write;

#[derive(Parser)]
#[command(name = "model-express")]
#[command(about = "ModelExpress Cache Management CLI")]
#[command(version = "1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Cache path override
    #[arg(long, value_name = "PATH")]
    cache_path: Option<PathBuf>,

    /// Server endpoint
    #[arg(long, value_name = "ENDPOINT")]
    server_endpoint: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize cache configuration
    Init {
        /// Cache path
        #[arg(long, value_name = "PATH")]
        cache_path: Option<PathBuf>,
        
        /// Server endpoint
        #[arg(long, value_name = "ENDPOINT")]
        server_endpoint: Option<String>,
        
        /// Auto-mount on startup
        #[arg(long)]
        auto_mount: Option<bool>,
    },

    /// List cached models
    List {
        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },

    /// Show cache status and usage
    Status,

    /// Clear specific model from cache
    Clear {
        /// Model name to clear
        model_name: String,
    },

    /// Clear entire cache
    ClearAll {
        /// Confirm without prompting
        #[arg(long)]
        yes: bool,
    },

    /// Pre-download model to cache
    Preload {
        /// Model name to preload
        model_name: String,
        
        /// Model provider
        #[arg(long, default_value = "huggingface")]
        provider: String,
    },

    /// Validate cache integrity
    Validate {
        /// Model name to validate (optional)
        model_name: Option<String>,
    },

    /// Show cache statistics
    Stats {
        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { cache_path, server_endpoint, auto_mount } => {
            init_cache(cache_path, server_endpoint, auto_mount).await?;
        }
        Commands::List { detailed } => {
            list_cached_models(detailed).await?;
        }
        Commands::Status => {
            show_cache_status().await?;
        }
        Commands::Clear { model_name } => {
            clear_cached_model(&model_name).await?;
        }
        Commands::ClearAll { yes } => {
            clear_all_cached_models(yes).await?;
        }
        Commands::Preload { model_name, provider } => {
            preload_model(&model_name, &provider).await?;
        }
        Commands::Validate { model_name } => {
            validate_cache(model_name).await?;
        }
        Commands::Stats { detailed } => {
            show_cache_stats(detailed).await?;
        }
    }

    Ok(())
}

async fn init_cache(
    cache_path: Option<PathBuf>,
    server_endpoint: Option<String>,
    auto_mount: Option<bool>,
) -> Result<()> {
    info!("ModelExpress Cache Configuration");
    info!("================================");

    let config = if let Some(path) = cache_path {
        CacheConfig::from_path(path)?
    } else {
        CacheConfig::prompt_user()?
    };

    // Override with command line options if provided
    let mut config = config;
    if let Some(endpoint) = server_endpoint {
        config.server_endpoint = endpoint;
    }
    if let Some(mount) = auto_mount {
        config.auto_mount = mount;
    }

    // Save configuration
    config.save_to_config_file()?;
    info!("Configuration saved successfully!");
    info!("Cache path: {:?}", config.local_path);
    info!("Server endpoint: {}", config.server_endpoint);
    info!("Auto-mount: {}", config.auto_mount);

    Ok(())
}

async fn list_cached_models(detailed: bool) -> Result<()> {
    let cache_config = get_cache_config()?;
    let stats = cache_config.get_cache_stats()?;

    info!("Cached Models");
    info!("=============");
    info!("Total models: {}", stats.total_models);
    info!("Total size: {}", stats.format_total_size());

    if stats.models.is_empty() {
        info!("No models found in cache.");
        return Ok(());
    }

    info!("Models:");
    for model in &stats.models {
        if detailed {
            info!("  {} ({}) - {:?}", 
                model.name, 
                stats.format_model_size(model), 
                model.path
            );
        } else {
            info!("  {} ({})", 
                model.name, 
                stats.format_model_size(model)
            );
        }
    }

    Ok(())
}

async fn show_cache_status() -> Result<()> {
    let cache_config = get_cache_config()?;
    let stats = cache_config.get_cache_stats()?;

    info!("Cache Status");
    info!("============");
    info!("Cache path: {:?}", cache_config.local_path);
    info!("Server endpoint: {}", cache_config.server_endpoint);
    info!("Auto-mount: {}", cache_config.auto_mount);
    info!("Total models: {}", stats.total_models);
    info!("Total size: {}", stats.format_total_size());

    // Check if cache directory exists and is accessible
    if cache_config.local_path.exists() {
        info!("Cache directory: ✅ Accessible");
    } else {
        warn!("Cache directory: ❌ Not found");
    }

    // Try to connect to server
    match Client::new(ClientConfig::new(&cache_config.server_endpoint)).await {
        Ok(_) => info!("Server connection: ✅ Available"),
        Err(_) => warn!("Server connection: ❌ Unavailable"),
    }

    Ok(())
}

async fn clear_cached_model(model_name: &str) -> Result<()> {
    let cache_config = get_cache_config()?;
    
    info!("Clearing model: {}", model_name);
    cache_config.clear_model(model_name)?;
    
    Ok(())
}

async fn clear_all_cached_models(yes: bool) -> Result<()> {
    let cache_config = get_cache_config()?;
    
    if !yes {
        print!("Are you sure you want to clear all cached models? [y/N]: ");
        std::io::stdout().flush()?;
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        if input.trim().to_lowercase() != "y" {
            info!("Operation cancelled.");
            return Ok(());
        }
    }
    
    info!("Clearing all cached models...");
    cache_config.clear_all()?;
    
    Ok(())
}

async fn preload_model(model_name: &str, provider_str: &str) -> Result<()> {
    let cache_config = get_cache_config()?;
    let provider = parse_provider(provider_str)?;
    
    info!("Pre-loading model: {} (provider: {:?})", model_name, provider);
    
    let client_config = ClientConfig::new(&cache_config.server_endpoint);
    let mut client = Client::new_with_cache(client_config, cache_config).await?;
    
    client.preload_model_to_cache(model_name, provider).await?;
    info!("Model pre-loaded successfully!");
    
    Ok(())
}

async fn validate_cache(model_name: Option<String>) -> Result<()> {
    let cache_config = get_cache_config()?;
    
    info!("Validating cache...");
    
    if let Some(name) = model_name {
        // Validate specific model
        let model_path = cache_config.local_path.join(&name);
        if model_path.exists() {
            info!("✅ Model '{}' found in cache", name);
            
            // Check for common model files
            let required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"];
            for file in &required_files {
                let file_path = model_path.join(file);
                if file_path.exists() {
                    debug!("  ✅ {} found", file);
                } else {
                    warn!("  ⚠️  {} missing", file);
                }
            }
        } else {
            error!("❌ Model '{}' not found in cache", name);
        }
    } else {
        // Validate entire cache
        let stats = cache_config.get_cache_stats()?;
        info!("Found {} models in cache", stats.total_models);
        
        for model in &stats.models {
            info!("  {} ({})", model.name, stats.format_model_size(model));
        }
    }
    
    Ok(())
}

async fn show_cache_stats(detailed: bool) -> Result<()> {
    let cache_config = get_cache_config()?;
    let stats = cache_config.get_cache_stats()?;
    
    info!("Cache Statistics");
    info!("================");
    info!("Total models: {}", stats.total_models);
    info!("Total size: {}", stats.format_total_size());
    
    if detailed && !stats.models.is_empty() {
        info!("Detailed Statistics:");
        for model in &stats.models {
            info!("  {}: {} bytes ({})", 
                model.name, 
                model.size, 
                stats.format_model_size(model)
            );
        }
    }
    
    Ok(())
}

fn get_cache_config() -> Result<CacheConfig> {
    CacheConfig::discover()
        .map_err(|e| anyhow::anyhow!("Failed to discover cache configuration: {}", e))
}

fn parse_provider(provider_str: &str) -> Result<ModelProvider> {
    match provider_str.to_lowercase().as_str() {
        "huggingface" | "hf" => Ok(ModelProvider::HuggingFace),
        _ => Err(anyhow::anyhow!("Unknown provider: {}", provider_str)),
    }
} 