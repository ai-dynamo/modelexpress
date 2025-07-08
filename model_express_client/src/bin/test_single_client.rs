use model_express_client::{Client, ClientConfig};
use model_express_common::models::ModelProvider;
use std::env;
use std::time::Instant;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let test_mode = args.iter().any(|arg| arg == "--test-model");
    let test_model = if test_mode {
        // Get the model name from the next argument
        if let Some(model_index) = args.iter().position(|arg| arg == "--test-model") {
            // Safely check if there's a next argument
            args.get(model_index.saturating_add(1))
                .cloned()
                .or_else(|| {
                    error!("Error: --test-model requires a model name");
                    None
                })
        } else {
            error!("Error: --test-model flag not found");
            None
        }
    } else {
        Some("Qwen/Qwen2.5-3B-Instruct".to_string()) // Default model name - big enough for testing
    };

    // Check if we have a valid model name
    let model_name = match test_model {
        Some(name) => name,
        None => {
            return Err("No valid model name provided".into());
        }
    };

    // Initialize a gRPC client with default configuration
    let mut client = Client::new(ClientConfig::default()).await?;

    // Check server health
    info!("Checking server health...");
    let health = client.health_check().await?;
    info!("Server status: {}", health.status);
    info!("Server version: {}", health.version);
    info!("Server uptime: {} seconds", health.uptime);

    // Run the model download test
    info!("\nRunning model download test");
    info!("Testing with model: {model_name}");

    run_model_test(&model_name).await?;

    // Test provider selection with fallback
    info!("\nTesting provider selection with fallback...");
    run_fallback_test(&model_name).await?;

    Ok(())
}

// Function to run the model download test
async fn run_model_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a timestamp for the start
    let start_time = Instant::now();

    let mut client = Client::new(ClientConfig::default()).await?;
    info!("Client: Requesting model {model_name}");
    let start = Instant::now();

    match client.request_model(model_name.to_string()).await {
        Ok(()) => {
            info!("Client: Model downloaded in {:?}", start.elapsed());
            info!("Client completed in {:?}", start_time.elapsed());
            info!("TEST PASSED: Model was downloaded successfully");
            Ok(())
        }
        Err(e) => {
            error!("Client: Model download failed: {e}");
            Err(format!("Client failed to download model: {e}").into())
        }
    }
}

// Function to test fallback functionality
async fn run_fallback_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing fallback functionality (assuming server is running)...");
    let mut client = Client::new(ClientConfig::default()).await?;

    let start = Instant::now();

    // This should work via server since it's running
    match client
        .request_model_with_provider_and_fallback(model_name, ModelProvider::HuggingFace)
        .await
    {
        Ok(()) => {
            info!(
                "Model downloaded with fallback capability in {:?}",
                start.elapsed()
            );
        }
        Err(e) => {
            return Err(format!("Failed to download model with fallback enabled: {e}").into());
        }
    }

    // Test direct download functionality
    info!("Testing direct download (bypassing server)...");
    let start_direct = Instant::now();

    match Client::download_model_directly(model_name, ModelProvider::HuggingFace).await {
        Ok(()) => {
            info!("Model downloaded directly in {:?}", start_direct.elapsed());
        }
        Err(e) => {
            return Err(format!("Failed to download model directly: {e}").into());
        }
    }

    // Test smart fallback (will use server if available, direct download if not)
    info!("Testing smart fallback...");
    let start_smart = Instant::now();

    match Client::request_model_with_smart_fallback(
        model_name,
        ModelProvider::HuggingFace,
        ClientConfig::default(),
    )
    .await
    {
        Ok(()) => {
            info!(
                "Model downloaded with smart fallback in {:?}",
                start_smart.elapsed()
            );
            info!(
                "FALLBACK TEST PASSED: Server-with-fallback, direct download, and smart fallback all work"
            );
            Ok(())
        }
        Err(e) => Err(format!("Failed to download model with smart fallback: {e}").into()),
    }
}
