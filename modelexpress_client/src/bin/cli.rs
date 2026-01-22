// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ModelExpress CLI binary.
//!
//! # Argument Architecture
//!
//! This CLI uses a layered argument structure:
//!
//! - **`ClientArgs`** (in `modelexpress_common/src/client_config.rs`):
//!   Shared arguments like `--endpoint`, `--timeout`, `--cache-path`, etc.
//!   These support environment variables (e.g., `MODEL_EXPRESS_ENDPOINT`).
//!   Add new shared arguments there.
//!
//! - **`Cli`** (in `modules/args.rs`):
//!   CLI-specific arguments like `--format`, `--verbose`.
//!   Embeds `ClientArgs` via `#[command(flatten)]`.
//!   Add CLI-only arguments there.
//!
//! - **`ClientConfig::load()`**:
//!   Takes `ClientArgs` and applies values to the configuration.
//!   Update this when adding new arguments to `ClientArgs`.

mod modules {
    pub mod args;
    pub mod handlers;
    pub mod output;
    pub mod payload;
}

use clap::Parser;
use colored::*;
use modelexpress_client::ClientConfig;
use modules::args::{ApiCommands, Cli, Commands};
use modules::handlers::{handle_api_send, handle_health_command, handle_model_command};
use modules::output::{print_output, setup_logging};
use std::process;
use tracing::{debug, error};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_logging(cli.verbose, cli.client_args.quiet);

    // Load configuration from the flattened ClientArgs.
    // ClientArgs contains all shared arguments (endpoint, timeout, cache settings, etc.)
    // which are parsed by clap with environment variable support.
    let config = match ClientConfig::load(cli.client_args.clone()) {
        Ok(config) => config,
        Err(e) => {
            error!("Configuration error: {e}");
            if !cli.client_args.quiet {
                eprintln!("{}: {e}", "Configuration Error".red().bold());
            }
            process::exit(1);
        }
    };

    debug!(
        "CLI initialized with config: endpoint={}, timeout={:?}, format={:?}",
        config.connection.endpoint, config.connection.timeout_secs, cli.format
    );

    let result = match cli.command {
        Commands::Health => {
            debug!("Executing health command");
            handle_health_command(config, &cli.format).await
        }
        Commands::Model { command } => {
            debug!("Executing model command");
            handle_model_command(
                command,
                cli.client_args.cache_path.clone(),
                config,
                &cli.format,
            )
            .await
        }
        Commands::Api { command } => match command {
            ApiCommands::Send {
                action,
                payload,
                payload_file,
            } => {
                debug!("Executing API send command");
                handle_api_send(action, payload, payload_file, config, &cli.format).await
            }
        },
    };

    if let Err(e) = result {
        error!("Command execution failed: {}", e);
        if !cli.client_args.quiet {
            match cli.format {
                modules::args::OutputFormat::Human => {
                    eprintln!("{}: {}", "Error".red().bold(), e);
                }
                _ => {
                    let error_output = serde_json::json!({
                        "success": false,
                        "error": e.to_string()
                    });
                    print_output(&error_output, &cli.format);
                }
            }
        }
        process::exit(1);
    } else {
        debug!("Command executed successfully");
    }
}

#[cfg(test)]
mod tests {
    use super::modules::args::{Cli, CliModelProvider};
    use clap::Parser;
    use modelexpress_client::ModelProvider;

    #[test]
    fn test_cli_model_provider_conversion() {
        let provider = CliModelProvider::HuggingFace;
        let converted: ModelProvider = provider.into();
        assert_eq!(converted, ModelProvider::HuggingFace);
    }

    #[test]
    fn test_cli_flattened_client_args_parsing() {
        // Test that the flattened ClientArgs fields are accessible through Cli
        let cli = Cli::try_parse_from([
            "modelexpress-cli",
            "--endpoint",
            "http://test:9999",
            "--timeout",
            "90",
            "--no-shared-storage",
            "--transfer-chunk-size",
            "524288",
            "health",
        ])
        .expect("Failed to parse CLI");

        // Verify flattened ClientArgs fields are accessible
        assert_eq!(
            cli.client_args.endpoint,
            Some("http://test:9999".to_string())
        );
        assert_eq!(cli.client_args.timeout, Some(90));
        assert!(cli.client_args.no_shared_storage);
        assert_eq!(cli.client_args.transfer_chunk_size, Some(524288));
    }

    #[test]
    fn test_cli_with_cache_path() {
        let cli = Cli::try_parse_from([
            "modelexpress-cli",
            "--cache-path",
            "/custom/cache/path",
            "health",
        ])
        .expect("Failed to parse CLI with cache path");

        assert_eq!(
            cli.client_args.cache_path,
            Some(std::path::PathBuf::from("/custom/cache/path"))
        );
    }

    #[test]
    fn test_cli_quiet_mode() {
        let cli = Cli::try_parse_from(["modelexpress-cli", "--quiet", "health"])
            .expect("Failed to parse");

        assert!(cli.client_args.quiet);
    }

    #[test]
    fn test_cli_format_and_verbose_are_cli_specific() {
        // These fields are CLI-specific, not in ClientArgs
        let cli = Cli::try_parse_from(["modelexpress-cli", "--format", "json", "-vvv", "health"])
            .expect("Failed to parse");

        assert_eq!(cli.format, super::modules::args::OutputFormat::Json);
        assert_eq!(cli.verbose, 3);
    }
}
