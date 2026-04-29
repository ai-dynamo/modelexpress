// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::args::{DownloadStrategy, ModelCommands, OutputFormat};
use super::output::{print_human_readable, print_output};
use super::payload::read_payload;
use colored::*;
use modelexpress_client::{Client, ClientConfig, ModelProvider};
use modelexpress_common::{
    cache::{CacheConfig, CacheStats, ModelInfo, resolve_model_path},
    download,
};
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use tracing::{debug, error, info};

fn format_model_line(stats: &CacheStats, model: &ModelInfo, detailed: bool) -> String {
    if detailed {
        format!(
            "  [{}] {} ({}) - {:?}",
            model.provider,
            model.name,
            stats.format_model_size(model),
            model.path
        )
    } else {
        format!(
            "  [{}] {} ({})",
            model.provider,
            model.name,
            stats.format_model_size(model)
        )
    }
}

fn model_json(stats: &CacheStats, model: &ModelInfo, detailed: bool) -> serde_json::Value {
    if detailed {
        serde_json::json!({
            "provider": model.provider.to_string(),
            "name": model.name,
            "size": model.size,
            "formatted_size": stats.format_model_size(model),
            "path": model.path
        })
    } else {
        serde_json::json!({
            "provider": model.provider.to_string(),
            "name": model.name,
            "size": model.size,
            "formatted_size": stats.format_model_size(model)
        })
    }
}

/// Handle the health check command
pub async fn handle_health_command(
    config: ClientConfig,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    debug!(
        "Initiating health check to server: {}",
        config.connection.endpoint
    );

    let mut client = Client::new(config).await?;
    let status = client.health_check().await?;

    info!("Health check completed successfully");

    match format {
        OutputFormat::Human => {
            println!("{}", "Server Health Status".green().bold());
            println!(
                "  {}: {}",
                "Status".cyan().bold(),
                if status.status == "healthy" || status.status == "ok" {
                    status.status.green()
                } else {
                    status.status.red()
                }
            );
            println!("  {}: {}", "Version".cyan().bold(), status.version);
            println!("  {}: {} seconds", "Uptime".cyan().bold(), status.uptime);
        }
        _ => print_output(&status, format),
    }

    Ok(())
}

/// Handle model commands (unified model management)
pub async fn handle_model_command(
    command: ModelCommands,
    storage_path_override: Option<PathBuf>,
    server_config: ClientConfig,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ModelCommands::Download {
            model_name,
            provider,
            strategy,
        } => {
            download_model(
                storage_path_override,
                model_name,
                provider,
                strategy,
                server_config,
                format,
            )
            .await
        }
        ModelCommands::Init {
            storage_path,
            server_endpoint,
        } => init_model_storage(storage_path, server_endpoint, format).await,
        ModelCommands::List { detailed } => {
            list_models(storage_path_override, detailed, format).await
        }
        ModelCommands::Status => show_model_status(storage_path_override, format).await,
        ModelCommands::Clear {
            provider,
            model_name,
        } => {
            clear_model(storage_path_override, provider, &model_name, &server_config, format).await
        }
        ModelCommands::ClearAll { yes } => {
            clear_all_models(storage_path_override, yes, &server_config, format).await
        }
        ModelCommands::Validate { model_name } => {
            validate_models(storage_path_override, model_name, format).await
        }
        ModelCommands::Stats { detailed } => {
            show_model_stats(storage_path_override, detailed, format).await
        }
    }
}

async fn download_model(
    storage_path_override: Option<PathBuf>,
    model_name: String,
    provider: ModelProvider,
    strategy: DownloadStrategy,
    config: ClientConfig,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    debug!(
        "Starting model download: {} with provider {:?} and strategy {:?}",
        model_name, provider, strategy
    );

    if let OutputFormat::Human = format {
        println!("{}", "Model Download".green().bold());
        println!("  {}: {}", "Model".cyan().bold(), model_name);
        println!("  {}: {}", "Provider".cyan().bold(), provider);
        println!("  {}: {:?}", "Strategy".cyan().bold(), strategy);
        println!();
    }

    info!("Downloading model: {}", model_name);

    // Get cache config if available, applying settings from ClientConfig
    let cache_config = if let Some(path) = storage_path_override {
        Some(CacheConfig::from_path(path)?)
    } else {
        CacheConfig::discover().ok()
    };

    // Apply shared_storage and transfer_chunk_size settings from ClientConfig
    let cache_config = cache_config.map(|mut c| {
        c.shared_storage = config.cache.shared_storage;
        c.transfer_chunk_size = config.cache.transfer_chunk_size;
        c.server_endpoint = config.connection.endpoint.clone();
        c
    });

    let result = match strategy {
        DownloadStrategy::SmartFallback => {
            debug!("Using smart fallback strategy");
            let mut config = config.clone();
            if let Some(cache_config) = cache_config {
                config.cache = cache_config;
            }
            Client::request_model_with_smart_fallback(model_name.clone(), provider, config, false)
                .await
                .map(|_| ())
        }
        DownloadStrategy::ServerOnly => {
            debug!("Using server-only strategy");
            let mut client = if let Some(cache_config) = cache_config {
                Client::new_with_cache(config.clone(), cache_config).await?
            } else {
                Client::new(config.clone()).await?
            };
            client
                .request_model(&model_name, provider, false)
                .await
                .map(|_| ())
        }
        DownloadStrategy::Direct => {
            debug!("Using direct download strategy");
            download::download_model(
                &model_name,
                provider,
                cache_config.map(|config| config.local_path),
                false,
            )
            .await
            .map(|_| ())
            .map_err(|e| {
                modelexpress_common::Error::Generic(format!("Direct download failed: {e}")).into()
            })
        }
    };

    match result {
        Ok(()) => {
            info!("Model download completed successfully: {}", model_name);
            let success_msg = format!("Model '{model_name}' downloaded successfully");
            match format {
                OutputFormat::Human => {
                    println!("{}", "✅ SUCCESS".green().bold());
                    println!("  {success_msg}");
                }
                _ => {
                    let output = serde_json::json!({
                        "success": true,
                        "message": success_msg,
                        "model_name": model_name,
                        "provider": provider.to_string(),
                        "strategy": format!("{:?}", strategy)
                    });
                    print_output(&output, format);
                }
            }
        }
        Err(e) => {
            error!("Model download failed for {}: {}", model_name, e);
            let error_msg = format!("Failed to download model '{model_name}': {e}");
            match format {
                OutputFormat::Human => {
                    println!("{}", "❌ FAILED".red().bold());
                    println!("  {error_msg}");
                }
                _ => {
                    let output = serde_json::json!({
                        "success": false,
                        "error": error_msg,
                        "model_name": model_name,
                        "provider": provider.to_string(),
                        "strategy": format!("{:?}", strategy)
                    });
                    print_output(&output, format);
                }
            }
            return Err(e);
        }
    }

    Ok(())
}

/// Handle API send command
pub async fn handle_api_send(
    action: String,
    payload: Option<String>,
    payload_file: Option<String>,
    config: ClientConfig,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    debug!("Preparing API request for action: {}", action);

    let mut client = Client::new(config).await?;
    let payload_data = read_payload(payload, payload_file)?;

    if payload_data.is_some() {
        debug!("API request includes payload data");
    }

    if let OutputFormat::Human = format {
        println!("{}", "API Request".green().bold());
        println!("  {}: {}", "Action".cyan().bold(), action);
        if payload_data.is_some() {
            println!("  {}: Yes", "Payload".cyan().bold());
        }
        println!();
    }

    info!("Sending API request: {}", action);

    let response: Value = client.send_request(&action, payload_data).await?;

    info!("API request completed successfully");

    match format {
        OutputFormat::Human => {
            println!("{}", "Response:".green().bold());
            print_human_readable(&response);
        }
        _ => print_output(&response, format),
    }

    Ok(())
}

async fn init_model_storage(
    storage_path: Option<PathBuf>,
    server_endpoint: Option<String>,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = if let Some(path) = storage_path {
        CacheConfig::from_path(path)?
    } else {
        // Use default configuration instead of prompting
        CacheConfig::default()
    };

    // Override with command line options if provided
    let mut config = config;
    if let Some(endpoint) = server_endpoint {
        config.server_endpoint = endpoint;
    }

    // Save configuration
    config.save_to_config_file()?;

    match format {
        OutputFormat::Human => {
            println!("{}", "ModelExpress Storage Configuration".green().bold());
            println!("{}", "===================================".green().bold());
            println!("Configuration saved successfully!");
            println!("Storage path: {:?}", config.local_path);
            println!("Server endpoint: {}", config.server_endpoint);
        }
        _ => {
            let output = serde_json::json!({
                "success": true,
                "storage_path": config.local_path,
                "server_endpoint": config.server_endpoint,
            });
            print_output(&output, format);
        }
    }

    Ok(())
}

async fn list_models(
    storage_path_override: Option<PathBuf>,
    detailed: bool,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage_config = get_storage_config(storage_path_override)?;
    let stats = storage_config.get_cache_stats()?;

    match format {
        OutputFormat::Human => {
            println!("{}", "Downloaded Models".green().bold());
            println!("{}", "=================".green().bold());
            println!("Total models: {}", stats.total_models);
            println!("Total size: {}", stats.format_total_size());

            if stats.models.is_empty() {
                println!("No models found in storage.");
                return Ok(());
            }

            println!("Models:");
            for model in &stats.models {
                println!("{}", format_model_line(&stats, model, detailed));
            }
        }
        _ => {
            let models_json: Vec<serde_json::Value> = stats
                .models
                .iter()
                .map(|model| model_json(&stats, model, detailed))
                .collect();

            let output = serde_json::json!({
                "total_models": stats.total_models,
                "total_size": stats.format_total_size(),
                "models": models_json
            });
            print_output(&output, format);
        }
    }

    Ok(())
}

async fn show_model_status(
    storage_path_override: Option<PathBuf>,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage_config = get_storage_config(storage_path_override)?;
    let stats = storage_config.get_cache_stats()?;

    let storage_accessible = storage_config.local_path.exists();
    let server_available = Client::new(ClientConfig::for_testing(&storage_config.server_endpoint))
        .await
        .is_ok();

    match format {
        OutputFormat::Human => {
            println!("{}", "Model Storage Status".green().bold());
            println!("{}", "====================".green().bold());
            println!("Storage path: {:?}", storage_config.local_path);
            println!("Server endpoint: {}", storage_config.server_endpoint);
            println!("Total models: {}", stats.total_models);
            println!("Total size: {}", stats.format_total_size());

            // Check if storage directory exists and is accessible
            if storage_accessible {
                println!("Storage directory: ✅ Accessible");
            } else {
                println!("Storage directory: ❌ Not found");
            }

            // Try to connect to server
            if server_available {
                println!("Server connection: ✅ Available");
            } else {
                println!("Server connection: ❌ Unavailable");
            }
        }
        _ => {
            let output = serde_json::json!({
                "storage_path": storage_config.local_path,
                "server_endpoint": storage_config.server_endpoint,
                "total_models": stats.total_models,
                "total_size": stats.format_total_size(),
                "storage_accessible": storage_accessible,
                "server_available": server_available
            });
            print_output(&output, format);
        }
    }

    Ok(())
}

/// Check whether clearing local model files is safe.
///
/// Refuses if shared storage mode is enabled (the client's local path IS the server's
/// storage, so deleting files would corrupt the server's database). Also refuses if
/// the server reports a cache directory that matches the client's local path, even
/// when shared_storage is not explicitly set (belt and suspenders).
async fn check_shared_storage_guard(
    storage_config: &CacheConfig,
    server_config: &ClientConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if server_config.cache.shared_storage {
        return Err(
            "model clear is not available in shared storage mode (client and server share the same filesystem)"
                .into(),
        );
    }

    // Belt and suspenders: even if shared_storage is false, check whether the server's
    // cache directory actually matches our local path.
    if let Ok(mut client) = Client::new(server_config.clone()).await
        && let Ok(status) = client.health_check().await
        && let Some(server_dir) = status.cache_directory
    {
        let server_path =
            std::fs::canonicalize(&server_dir).unwrap_or_else(|_| PathBuf::from(&server_dir));
        let local_path = std::fs::canonicalize(&storage_config.local_path)
            .unwrap_or_else(|_| storage_config.local_path.clone());
        if server_path == local_path {
            return Err(
                "model clear is not available: local cache path matches the server's cache directory"
                    .into(),
            );
        }
    }

    Ok(())
}

async fn clear_model(
    storage_path_override: Option<PathBuf>,
    provider: ModelProvider,
    model_name: &str,
    server_config: &ClientConfig,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage_config = get_storage_config(storage_path_override)?;

    check_shared_storage_guard(&storage_config, server_config).await?;

    storage_config.clear_model(model_name, provider)?;

    match format {
        OutputFormat::Human => {
            println!(
                "✅ Model '{model_name}' cleared from storage for provider {}",
                provider
            );
        }
        _ => {
            let output = serde_json::json!({
                "success": true,
                "message": format!("Model '{}' cleared from storage", model_name),
                "model_name": model_name,
                "provider": provider.to_string()
            });
            print_output(&output, format);
        }
    }

    Ok(())
}

async fn clear_all_models(
    storage_path_override: Option<PathBuf>,
    yes: bool,
    server_config: &ClientConfig,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage_config = get_storage_config(storage_path_override)?;

    check_shared_storage_guard(&storage_config, server_config).await?;

    if !yes && matches!(format, OutputFormat::Human) {
        print!("Are you sure you want to clear all models from storage? [y/N]: ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() != "y" {
            println!("Operation cancelled.");
            return Ok(());
        }
    }

    storage_config.clear_all()?;

    match format {
        OutputFormat::Human => {
            println!("✅ All models cleared from storage");
        }
        _ => {
            let output = serde_json::json!({
                "success": true,
                "message": "All models cleared from storage"
            });
            print_output(&output, format);
        }
    }

    Ok(())
}

async fn validate_models(
    storage_path_override: Option<PathBuf>,
    model_name: Option<String>,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage_config = get_storage_config(storage_path_override)?;

    if let Some(name) = model_name {
        // Validate specific model.
        // Try the HuggingFace cache layout first (models--org--name/snapshots/...),
        // then fall back to a plain path join for other providers.
        let model_path = resolve_model_path(
            &storage_config.local_path,
            ModelProvider::HuggingFace,
            &name,
            None,
        )
        .unwrap_or_else(|_| storage_config.local_path.join(&name));
        let exists = model_path.exists();

        match format {
            OutputFormat::Human => {
                println!("{}", "Model Validation".green().bold());
                if exists {
                    println!("✅ Model '{name}' found in storage");

                    // Check for common model files
                    let required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"];
                    for file in &required_files {
                        let file_path = model_path.join(file);
                        if file_path.exists() {
                            debug!("  ✅ {} found", file);
                        } else {
                            println!("  ⚠️  {file} missing");
                        }
                    }
                } else {
                    println!("❌ Model '{name}' not found in storage");
                }
            }
            _ => {
                let output = serde_json::json!({
                    "model_name": name,
                    "exists": exists,
                    "path": model_path
                });
                print_output(&output, format);
            }
        }
    } else {
        // Validate entire storage
        let stats = storage_config.get_cache_stats()?;

        match format {
            OutputFormat::Human => {
                println!("{}", "Model Validation".green().bold());
                println!("Found {} models in storage", stats.total_models);

                for model in &stats.models {
                    println!("{}", format_model_line(&stats, model, false));
                }
            }
            _ => {
                let output = serde_json::json!({
                    "total_models": stats.total_models,
                    "models": stats.models.iter().map(|model| {
                        model_json(&stats, model, false)
                    }).collect::<Vec<_>>()
                });
                print_output(&output, format);
            }
        }
    }

    Ok(())
}

async fn show_model_stats(
    storage_path_override: Option<PathBuf>,
    detailed: bool,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage_config = get_storage_config(storage_path_override)?;
    let stats = storage_config.get_cache_stats()?;

    match format {
        OutputFormat::Human => {
            println!("{}", "Model Storage Statistics".green().bold());
            println!("{}", "========================".green().bold());
            println!("Total models: {}", stats.total_models);
            println!("Total size: {}", stats.format_total_size());

            if detailed && !stats.models.is_empty() {
                println!("Detailed Statistics:");
                for model in &stats.models {
                    println!(
                        "  [{}] {}: {} bytes ({})",
                        model.provider,
                        model.name,
                        model.size,
                        stats.format_model_size(model)
                    );
                }
            }
        }
        _ => {
            let models_data = if detailed {
                Some(
                    stats
                        .models
                        .iter()
                        .map(|model| model_json(&stats, model, false))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };

            let mut output = serde_json::json!({
                "total_models": stats.total_models,
                "total_size": stats.format_total_size()
            });

            if let Some(models) = models_data {
                output["detailed_models"] = serde_json::Value::Array(models);
            }

            print_output(&output, format);
        }
    }

    Ok(())
}

fn get_storage_config(
    storage_path_override: Option<PathBuf>,
) -> Result<CacheConfig, Box<dyn std::error::Error>> {
    // If storage path is provided via CLI, use it
    if let Some(path) = storage_path_override {
        return Ok(CacheConfig::from_path(path)?);
    }

    // Otherwise, try to discover configuration
    CacheConfig::discover()
        .map_err(|e| format!("Failed to discover storage configuration: {e}").into())
}
