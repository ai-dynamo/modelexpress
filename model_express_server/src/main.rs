use model_express_common::{
    constants,
    grpc::{
        api::api_service_server::ApiServiceServer,
        health::health_service_server::HealthServiceServer,
        model::model_service_server::ModelServiceServer,
    },
};
use model_express_server::services::{ApiServiceImpl, HealthServiceImpl, ModelServiceImpl};
use std::net::SocketAddr;
use tonic::transport::Server;
use tracing::{Level, info};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting model_express_server with gRPC...");

    let addr = SocketAddr::from(([0, 0, 0, 0], constants::DEFAULT_GRPC_PORT));

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
