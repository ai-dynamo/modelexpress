// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use modelexpress_server::{
    backend_config::BackendConfig,
    config::{ServerArgs, ServerConfig},
    run_server,
};
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
    let env_filter = EnvFilter::from_default_env().add_directive(log_level.into());
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(env_filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Translate a CTRL+C into a graceful shutdown of the embedded server.
    let shutdown = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!("Failed to install CTRL+C signal handler: {e}");
            return;
        }
        info!("Received CTRL+C, shutting down gracefully...");
    };

    let backend = BackendConfig::from_env()?;

    run_server(config, backend, shutdown).await
}
