// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod modules {
    pub mod args;
    pub mod handlers;
    pub mod output;
    pub mod payload;
}

use clap::Parser;
use colored::*;
use model_express_client::{ClientArgs, ClientConfig};
use modules::args::{ApiCommands, Cli, Commands};
use modules::handlers::{handle_api_send, handle_health_command, handle_model_command};
use modules::output::{print_output, setup_logging};
use std::process;
use tracing::{debug, error};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_logging(cli.verbose, cli.quiet);

    let client_args = ClientArgs {
        config: None, // CLI doesn't have a config file option yet
        endpoint: Some(cli.endpoint),
        timeout: Some(cli.timeout),
        cache_path: cli.cache_path.clone(),
        log_level: None, // Could map from verbose levels if needed
        log_format: None,
        quiet: cli.quiet,
        max_retries: None,
        retry_delay: None,
    };

    // Load the configuration
    let config = match ClientConfig::load(client_args) {
        Ok(config) => config,
        Err(e) => {
            error!("Configuration error: {e}");
            if !cli.quiet {
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
            handle_model_command(command, cli.cache_path, config, &cli.format).await
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
        if !cli.quiet {
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
    use super::modules::args::CliModelProvider;
    use model_express_client::ModelProvider;

    #[test]
    fn test_cli_model_provider_conversion() {
        let provider = CliModelProvider::HuggingFace;
        let converted: ModelProvider = provider.into();
        assert_eq!(converted, ModelProvider::HuggingFace);
    }
}
