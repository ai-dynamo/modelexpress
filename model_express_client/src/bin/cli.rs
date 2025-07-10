mod modules {
    pub mod args;
    pub mod handlers;
    pub mod output;
    pub mod payload;
}

use modules::args::{ApiCommands, Cli, Commands};
use clap::Parser;
use colored::*;
use modules::handlers::{handle_api_send, handle_health_command, handle_model_command};
use model_express_client::ClientConfig;
use modules::output::{print_output, setup_logging};
use std::process;
use tracing::{debug, error};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_logging(cli.verbose, cli.quiet);

    debug!(
        "CLI initialized with config: endpoint={}, timeout={}, format={:?}",
        cli.endpoint, cli.timeout, cli.format
    );

    let config = ClientConfig::new(cli.endpoint).with_timeout(cli.timeout);

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
    use super::modules::args::{CliModelProvider};
    use model_express_client::ModelProvider;

    #[test]
    fn test_cli_model_provider_conversion() {
        let provider = CliModelProvider::HuggingFace;
        let converted: ModelProvider = provider.into();
        assert_eq!(converted, ModelProvider::HuggingFace);
    }
}
