// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This binary supports multiple test modes:
//! - `--concurrent`: Test concurrent model downloads (default)
//! - `--single`: Test single model download
//! - `--test-model <name>`: Specify which model to use for testing

#![allow(clippy::expect_used)]

use modelexpress_client::{Client, ClientConfig};
use modelexpress_common::models::ModelProvider;
use std::env;
use std::time::{Duration, Instant};
use tracing::{error, info};

/// Test mode selection
#[derive(Debug, Clone, Copy, PartialEq)]
enum TestMode {
    /// Test concurrent model downloads from multiple clients
    Concurrent,
    /// Test single model download
    Single,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Determine test mode
    let test_mode = if args.iter().any(|arg| arg == "--single") {
        TestMode::Single
    } else {
        TestMode::Concurrent // Default to concurrent testing
    };

    // Get model name from args or use default
    let model_name = if let Some(model_index) = args.iter().position(|arg| arg == "--test-model") {
        if let Some(next_arg) = args.get(model_index.saturating_add(1)) {
            next_arg.clone()
        } else {
            error!("Error: --test-model requires a model name");
            return Err("Missing model name".into());
        }
    } else {
        "Qwen/Qwen2.5-3B-Instruct".to_string() // Default model name - big enough for testing
    };

    info!("Test mode: {:?}", test_mode);
    info!("Testing with model: {}", model_name);

    // Initialize a gRPC client with default configuration
    let mut client = Client::new(ClientConfig::default()).await?;

    // Check server health
    info!("Checking server health...");
    let health = client.health_check().await?;
    info!("Server status: {}", health.status);
    info!("Server version: {}", health.version);
    info!("Server uptime: {} seconds", health.uptime);

    // Run the appropriate test based on mode
    match test_mode {
        TestMode::Concurrent => {
            info!("\nRunning integration test for concurrent model downloads");
            run_concurrent_model_test(&model_name).await?;
        }
        TestMode::Single => {
            info!("\nRunning single model download test");
            run_single_model_test(&model_name).await?;
        }
    }

    // Test provider selection with fallback (common to both modes)
    info!("\nTesting provider selection with fallback...");
    run_fallback_test(&model_name).await?;

    Ok(())
}

/// Run the concurrent model download test with two clients
async fn run_concurrent_model_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::task;

    let start_time = Instant::now();

    // Clone the model name for the tasks
    let model_name1 = model_name.to_string();
    let model_name2 = model_name.to_string();

    // Spawn two tasks to download the same model concurrently
    let client1_task = task::spawn(async move {
        let mut client1 = Client::new(ClientConfig::default())
            .await
            .expect("Failed to create client 1");
        info!("Client 1: Requesting model {model_name1}");
        let start = Instant::now();
        client1
            .request_model(model_name1, false)
            .await
            .expect("Client 1 failed to download model");
        info!("Client 1: Model downloaded in {:?}", start.elapsed());
    });

    // Wait a short time so the first client starts first
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client2_task = task::spawn(async move {
        let mut client2 = Client::new(ClientConfig::default())
            .await
            .expect("Failed to create client 2");
        info!("Client 2: Requesting model {model_name2}");
        let start = Instant::now();
        client2
            .request_model(model_name2, false)
            .await
            .expect("Client 2 failed to download model");
        info!("Client 2: Model downloaded in {:?}", start.elapsed());
    });

    // Wait for both clients to complete
    client1_task.await?;
    client2_task.await?;

    info!("Both clients completed in {:?}", start_time.elapsed());
    info!("CONCURRENT TEST PASSED: Model was downloaded and both clients received it");

    Ok(())
}

/// Run a single model download test
async fn run_single_model_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut client = Client::new(ClientConfig::default()).await?;
    info!("Client: Requesting model {model_name}");
    let start = Instant::now();

    match client.request_model(model_name.to_string(), false).await {
        Ok(()) => {
            info!("Client: Model downloaded in {:?}", start.elapsed());
            info!("Client completed in {:?}", start_time.elapsed());
            info!("SINGLE TEST PASSED: Model was downloaded successfully");
            Ok(())
        }
        Err(e) => {
            error!("Client: Model download failed: {e}");
            Err(format!("Client failed to download model: {e}").into())
        }
    }
}

/// Test fallback functionality including server fallback, direct download, and smart fallback
async fn run_fallback_test(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing fallback functionality (assuming server is running)...");
    let mut client = Client::new(ClientConfig::default()).await?;

    let start = Instant::now();

    // This should work via server since it's running
    match client
        .request_model_with_provider_and_fallback(model_name, ModelProvider::HuggingFace, false)
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

    match Client::download_model_directly(model_name, ModelProvider::HuggingFace, false).await {
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
        false,
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
