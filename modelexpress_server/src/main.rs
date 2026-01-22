// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use modelexpress_common::grpc::{
    api::api_service_server::ApiServiceServer, health::health_service_server::HealthServiceServer,
    model::model_service_server::ModelServiceServer,
};
use modelexpress_server::{
    cache::CacheEvictionService,
    config::{ServerArgs, ServerConfig},
    database::ModelDatabase,
    services::{ApiServiceImpl, HealthServiceImpl, ModelServiceImpl},
};
use tonic::transport::Server;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = ServerArgs::parse();

    // Check if we should validate config and exit
    if args.validate_config {
        match ServerConfig::load_and_validate_strict(args) {
            Ok(config) => {
                println!("Configuration is valid ✓");
                config.print_config();
                return Ok(());
            }
            Err(e) => {
                eprintln!("Configuration validation failed: {e}");
                std::process::exit(1);
            }
        }
    }

    // Load configuration from multiple sources
    let config = ServerConfig::load(args)?;

    // Initialize tracing with the configured log level
    let log_level = config.log_level();

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(log_level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting ModelExpress server...");
    config.print_config();

    // Get server address
    let addr = config.socket_addr().map_err(|e| {
        error!("Invalid server address: {e}");
        e
    })?;

    // Initialize database
    let database = match ModelDatabase::new(&config.database.path.to_string_lossy()) {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            return Err(e);
        }
    };

    // Create cache eviction service
    let cache_service = CacheEvictionService::new(database.clone(), config.cache.eviction.clone());

    // Create shutdown channels
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    // Start cache eviction service in background
    let cache_handle = if config.cache.eviction.enabled {
        info!("Starting cache eviction service...");
        Some(tokio::spawn(async move {
            if let Err(e) = cache_service.start(shutdown_rx).await {
                error!("Cache eviction service error: {e}");
            }
        }))
    } else {
        info!("Cache eviction service is disabled");
        None
    };

    // Create service implementations
    let health_service = HealthServiceImpl;
    let api_service = ApiServiceImpl;
    let model_service = ModelServiceImpl;

    // Setup graceful shutdown handler
    let shutdown_signal = async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!("Failed to install CTRL+C signal handler: {e}");
            return;
        }
        info!("Received CTRL+C, shutting down gracefully...");

        // Signal cache eviction service to shutdown
        if shutdown_tx.send(()).is_err() {
            error!("Failed to send shutdown signal to cache eviction service");
        }
    };

    // Start the gRPC server
    info!("Starting gRPC server on: {addr}");
    let server_result = Server::builder()
        .add_service(HealthServiceServer::new(health_service))
        .add_service(ApiServiceServer::new(api_service))
        .add_service(ModelServiceServer::new(model_service))
        .serve_with_shutdown(addr, shutdown_signal)
        .await;

    // Wait for cache service to complete if it was started
    if let Some(handle) = cache_handle
        && let Err(e) = handle.await
    {
        error!("Cache eviction service join error: {e}");
    }

    server_result?;
    info!("Server shutdown complete");
    Ok(())
}
