// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

use modelexpress_client::{Client, ClientConfig};
use modelexpress_common::{constants, models::ModelProvider};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn};

#[tokio::test]
#[ignore = "Ignore by default since it requires a running server"]
async fn test_integration_full_workflow() {
    // Initialize logging for tests
    let _ = tracing_subscriber::fmt::try_init();

    // This test requires the server to be running
    let config =
        ClientConfig::for_testing(format!("http://127.0.0.1:{}", constants::DEFAULT_GRPC_PORT));

    // Try to connect to the server
    let mut client =
        if let Ok(Ok(client)) = timeout(Duration::from_secs(5), Client::new(config)).await {
            client
        } else {
            info!("Server not available, skipping integration test");
            return;
        };

    // Test health check
    let health_result = client.health_check().await;
    assert!(
        health_result.is_ok(),
        "Health check failed: {health_result:?}"
    );

    let status = health_result.expect("Expected health check to succeed");
    assert!(!status.version.is_empty());
    assert_eq!(status.status, "ok");

    // Test ping request
    let ping_result: Result<serde_json::Value, _> = client.send_request("ping", None).await;
    assert!(ping_result.is_ok(), "Ping request failed: {ping_result:?}");

    let ping_response = ping_result.expect("Expected ping request to succeed");
    assert_eq!(ping_response["message"], "pong");

    // Test unknown action
    let unknown_result: Result<serde_json::Value, _> = client.send_request("unknown", None).await;
    assert!(unknown_result.is_err(), "Unknown action should fail");
}

#[tokio::test]
#[ignore = "Ignore by default since it may require network access"]
async fn test_integration_model_download_fallback() {
    let config = ClientConfig::for_testing("http://127.0.0.1:99999"); // Invalid port to force fallback

    // This should fallback to direct download since server is not available
    let result = Client::request_model_with_smart_fallback(
        "invalid-model-name-12345", // Use invalid model to test error handling
        ModelProvider::HuggingFace,
        config,
        false,
    )
    .await;

    // Should fail because model doesn't exist, but we should get a meaningful error
    assert!(result.is_err());
    let error_msg = result.expect_err("Expected error result").to_string();
    assert!(error_msg.contains("Direct download failed") || error_msg.contains("Failed to fetch"));
}

#[tokio::test]
async fn test_integration_direct_download_invalid_model() {
    // Test direct download with an invalid model name
    let result = Client::download_model_directly(
        "definitely-not-a-real-model-name-12345",
        ModelProvider::HuggingFace,
        false,
    )
    .await;

    // Should fail with a meaningful error
    assert!(result.is_err());
    let error_msg = result.expect_err("Expected error result").to_string();
    assert!(error_msg.contains("Direct download failed"));
}

#[tokio::test]
#[ignore = "Ignore by default since it requires network access and takes time"]
async fn test_integration_small_model_download() {
    // Test with a very small, real model (only run this in CI or when explicitly requested)
    // Note: This test requires internet access and may take some time

    let result = Client::download_model_directly(
        "prajjwal1/bert-tiny", // A very small BERT model for testing
        ModelProvider::HuggingFace,
        false,
    )
    .await;

    match result {
        Ok(()) => info!("Small model download successful"),
        Err(e) => {
            // In CI environments, this might fail due to network restrictions
            warn!("Model download failed (may be expected in test env): {e}");
        }
    }
}

#[tokio::test]
async fn test_integration_client_config_validation() {
    // Test various client configurations

    // Valid configuration
    let valid_config = ClientConfig::for_testing("http://localhost:8001");
    assert_eq!(valid_config.connection.endpoint, "http://localhost:8001");
    assert!(valid_config.connection.timeout_secs.is_some());

    // Default configuration (use for_testing with default endpoint)
    let default_config =
        ClientConfig::for_testing(format!("http://localhost:{}", constants::DEFAULT_GRPC_PORT));
    assert!(default_config.connection.endpoint.contains("localhost"));
    assert!(
        default_config
            .connection
            .endpoint
            .contains(&constants::DEFAULT_GRPC_PORT.to_string())
    );

    // Configuration with invalid endpoint should still create but fail on connection
    let invalid_config = ClientConfig::for_testing("invalid-url");
    let client_result = Client::new(invalid_config).await;
    assert!(client_result.is_err());
}

#[tokio::test]
async fn test_integration_offline_mode_without_cache() {
    // Test that HF_HUB_OFFLINE mode properly fails when model is not cached
    unsafe {
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }

    let result = Client::download_model_directly(
        "nonexistent-model-for-offline-test",
        ModelProvider::HuggingFace,
        false,
    )
    .await;

    unsafe {
        std::env::remove_var("HF_HUB_OFFLINE");
    }

    // Should fail because model is not in cache
    assert!(result.is_err());
    let error_msg = result.expect_err("Expected error result").to_string();
    assert!(
        error_msg.contains("not found in cache"),
        "Error should mention model not found: {error_msg}"
    );
}

#[tokio::test]
async fn test_integration_offline_mode_with_cached_model() {
    // Test that HF_HUB_OFFLINE mode returns cached models correctly
    // This test creates a mock cache structure and verifies offline mode uses it
    use modelexpress_common::download;

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let cache_path = temp_dir.path().to_path_buf();

    // Create a mock cache structure
    let model_name = "test/offline-model";
    let normalized_name = model_name.replace("/", "--");
    let snapshot_path = cache_path
        .join(format!("models--{normalized_name}"))
        .join("snapshots")
        .join("abc1234");
    std::fs::create_dir_all(&snapshot_path).expect("Failed to create mock cache");

    // Create a dummy config file in the snapshot
    std::fs::write(snapshot_path.join("config.json"), "{}").expect("Failed to create config file");

    unsafe {
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }

    // Use download::download_model directly with explicit cache_dir to avoid CacheConfig::discover
    let result = download::download_model(
        model_name,
        ModelProvider::HuggingFace,
        Some(cache_path),
        false,
    )
    .await;

    unsafe {
        std::env::remove_var("HF_HUB_OFFLINE");
    }

    // Should succeed with cached model
    assert!(
        result.is_ok(),
        "Expected offline mode to succeed with cached model: {result:?}"
    );
}
