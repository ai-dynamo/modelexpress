use clap::Parser;
use model_express_common::grpc::{
    api::api_service_server::ApiServiceServer, health::health_service_server::HealthServiceServer,
    model::model_service_server::ModelServiceServer,
};
use model_express_server::{
    config::{ServerArgs, ServerConfig},
    services::{ApiServiceImpl, HealthServiceImpl, ModelServiceImpl},
};
use std::net::SocketAddr;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = ServerArgs::parse();

    // Check if we should validate config and exit
    if args.validate_config {
        match ServerConfig::load(args) {
            Ok(_) => {
                println!("Configuration is valid");
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

    info!("Starting model_express_server with gRPC...");

    // Use configured port instead of environment variable or default
    let port = config.server.port;
    let addr = SocketAddr::from(([0, 0, 0, 0], port.get()));

    // Create service implementations
    let health_service = HealthServiceImpl;
    let api_service = ApiServiceImpl;
    let model_service = ModelServiceImpl;

    // Start the gRPC server
    info!("Listening on gRPC endpoint: {}", addr);
    Server::builder()
        .add_service(HealthServiceServer::new(health_service))
        .add_service(ApiServiceServer::new(api_service))
        .add_service(ModelServiceServer::new(model_service))
        .serve(addr)
        .await?;

    Ok(())
}
