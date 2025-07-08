#![allow(clippy::expect_used, clippy::unwrap_used)]

use model_express_client::{Client, ClientConfig};
use model_express_common::models::ModelProvider;
use std::env;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let test_mode = args.iter().any(|arg| arg == "--test-model");
    let test_model = if test_mode {
        // Get the model name from the next argument
        let model_index = args
            .iter()
            .position(|arg| arg == "--test-model")
            .expect("--test-model should be present when test_mode is true");
        if let Some(next_arg) = args.get(model_index.saturating_add(1)) {
            Some(next_arg.clone())
        } else {
            println!("Error: --test-model requires a model name");
            return Err("Missing model name".into());
        }
    } else {
        Some("Qwen/Qwen2.5-3B-Instruct".to_string()) // Default model name - big enough for testing
    };

    // Initialize a gRPC client with default configuration
    let mut client = Client::new(ClientConfig::default()).await?;

    // Check server health
    println!("Checking server health...");
    let health = client.health_check().await?;
    println!("Server status: {}", health.status);
    println!("Server version: {}", health.version);
    println!("Server uptime: {} seconds", health.uptime);

    // Run the concurrent model download test
    println!("\nRunning integration test for concurrent model downloads");
    let model_name = test_model.expect("Model name should be present");
    println!("Testing with model: {model_name}");

    run_concurrent_model_test(&model_name).await?;

    // Test provider selection with fallback
    println!("\nTesting provider selection with fallback...");
    run_fallback_test(&model_name).await?;

    Ok(())
}

// Function to run the concurrent model download test
async fn run_concurrent_model_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::task;

    // Create a timestamp for the start
    let start_time = Instant::now();

    // Clone the model name for the tasks
    let model_name1 = model_name.to_string();
    let model_name2 = model_name.to_string();

    // Spawn two tasks to download the same model concurrently
    let client1_task = task::spawn(async move {
        let mut client1 = Client::new(ClientConfig::default())
            .await
            .expect("Failed to create client 1");
        println!("Client 1: Requesting model {model_name1}");
        let start = Instant::now();
        client1
            .request_model(model_name1)
            .await
            .expect("Client 1 failed to download model");
        println!("Client 1: Model downloaded in {:?}", start.elapsed());
    });

    // Wait a short time so the first client starts first
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client2_task = task::spawn(async move {
        let mut client2 = Client::new(ClientConfig::default())
            .await
            .expect("Failed to create client 2");
        println!("Client 2: Requesting model {model_name2}");
        let start = Instant::now();
        client2
            .request_model(model_name2)
            .await
            .expect("Client 2 failed to download model");
        println!("Client 2: Model downloaded in {:?}", start.elapsed());
    });

    // Wait for both clients to complete
    client1_task.await?;
    client2_task.await?;

    println!("Both clients completed in {:?}", start_time.elapsed());

    println!("INTEGRATION TEST PASSED: Model was downloaded and both clients received it");

    Ok(())
}

// Function to test provider selection
#[allow(dead_code)]
async fn run_provider_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = Client::new(ClientConfig::default()).await?;

    println!("Testing explicit Hugging Face provider selection...");
    let start = Instant::now();

    client
        .request_model_with_provider(model_name, ModelProvider::HuggingFace)
        .await
        .expect("Failed to download model with explicit Hugging Face provider");

    println!(
        "Model downloaded with explicit provider in {:?}",
        start.elapsed()
    );
    println!("PROVIDER TEST PASSED: Model was downloaded using explicit provider selection");

    Ok(())
}

// Function to test fallback functionality
async fn run_fallback_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing fallback functionality (assuming server is running)...");
    let mut client = Client::new(ClientConfig::default()).await?;

    let start = Instant::now();

    // This should work via server since it's running
    client
        .request_model_with_provider_and_fallback(model_name, ModelProvider::HuggingFace)
        .await
        .expect("Failed to download model with fallback enabled");

    println!(
        "Model downloaded with fallback capability in {:?}",
        start.elapsed()
    );

    // Test direct download functionality
    println!("Testing direct download (bypassing server)...");
    let start_direct = Instant::now();

    Client::download_model_directly(model_name, ModelProvider::HuggingFace)
        .await
        .expect("Failed to download model directly");

    println!("Model downloaded directly in {:?}", start_direct.elapsed());

    // Test smart fallback (will use server if available, direct download if not)
    println!("Testing smart fallback...");
    let start_smart = Instant::now();

    Client::request_model_with_smart_fallback(
        model_name,
        ModelProvider::HuggingFace,
        ClientConfig::default(),
    )
    .await
    .expect("Failed to download model with smart fallback");

    println!(
        "Model downloaded with smart fallback in {:?}",
        start_smart.elapsed()
    );
    println!(
        "FALLBACK TEST PASSED: Server-with-fallback, direct download, and smart fallback all work"
    );

    Ok(())
}
