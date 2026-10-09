// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use modelexpress_server::{
    backend_config::BackendConfig,
    config::{ServerArgs, ServerConfig},
    run_server,
};
use tokio::signal::unix::{SignalKind, signal};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

    // Shut down gracefully on CTRL+C (SIGINT) or SIGTERM. SIGTERM is what
    // Kubernetes and container runtimes send to stop a container; as PID 1 the
    // server would otherwise ignore it and be SIGKILLed after the grace period.
    let mut sigterm = signal(SignalKind::terminate())?;
    let shutdown = async move {
        tokio::select! {
            result = tokio::signal::ctrl_c() => match result {
                Ok(()) => info!("Received CTRL+C, shutting down gracefully..."),
                Err(e) => error!("Failed to install CTRL+C signal handler: {e}"),
            },
            _ = sigterm.recv() => info!("Received SIGTERM, shutting down gracefully..."),
        }
    };

    let backend = BackendConfig::from_env()?;

    run_server(config, backend, shutdown).await
}
