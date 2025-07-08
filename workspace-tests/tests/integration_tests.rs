#![allow(clippy::expect_used, clippy::unwrap_used)]

use model_express_client::{Client, ClientConfig};
use model_express_common::{constants, models::ModelProvider};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
#[ignore = "Ignore by default since it requires a running server"]
async fn test_integration_full_workflow() {
    // This test requires the server to be running
    let config = ClientConfig {
        grpc_endpoint: format!("http://127.0.0.1:{}", constants::DEFAULT_GRPC_PORT),
        timeout_secs: Some(30),
    };

    // Try to connect to the server
    let mut client =
        if let Ok(Ok(client)) = timeout(Duration::from_secs(5), Client::new(config)).await {
            client
        } else {
            println!("Server not available, skipping integration test");
            return;
        };

    // Test health check
    let health_result = client.health_check().await;
    assert!(
        health_result.is_ok(),
        "Health check failed: {health_result:?}"
    );

    let status = health_result.unwrap();
    assert!(!status.version.is_empty());
    assert_eq!(status.status, "ok");

    // Test ping request
    let ping_result: Result<serde_json::Value, _> = client.send_request("ping", None).await;
    assert!(ping_result.is_ok(), "Ping request failed: {ping_result:?}");

    let ping_response = ping_result.unwrap();
    assert_eq!(ping_response["message"], "pong");

    // Test unknown action
    let unknown_result: Result<serde_json::Value, _> = client.send_request("unknown", None).await;
    assert!(unknown_result.is_err(), "Unknown action should fail");
}

#[tokio::test]
#[ignore = "Ignore by default since it may require network access"]
async fn test_integration_model_download_fallback() {
    let config = ClientConfig {
        grpc_endpoint: "http://127.0.0.1:99999".to_string(), // Invalid port to force fallback
        timeout_secs: Some(1),
    };

    // This should fallback to direct download since server is not available
    let result = Client::request_model_with_smart_fallback(
        "invalid-model-name-12345", // Use invalid model to test error handling
        ModelProvider::HuggingFace,
        config,
    )
    .await;

    // Should fail because model doesn't exist, but we should get a meaningful error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Direct download failed") || error_msg.contains("Failed to fetch"));
}

#[tokio::test]
async fn test_integration_direct_download_invalid_model() {
    // Test direct download with an invalid model name
    let result = Client::download_model_directly(
        "definitely-not-a-real-model-name-12345",
        ModelProvider::HuggingFace,
    )
    .await;

    // Should fail with a meaningful error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
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
    )
    .await;

    match result {
        Ok(()) => println!("Small model download successful"),
        Err(e) => {
            // In CI environments, this might fail due to network restrictions
            println!("Model download failed (may be expected in test env): {e}");
        }
    }
}

#[tokio::test]
async fn test_integration_client_config_validation() {
    // Test various client configurations

    // Valid configuration
    let valid_config = ClientConfig::new("http://localhost:8001").with_timeout(30);
    assert_eq!(valid_config.grpc_endpoint, "http://localhost:8001");
    assert_eq!(valid_config.timeout_secs, Some(30));

    // Default configuration
    let default_config = ClientConfig::default();
    assert!(default_config.grpc_endpoint.contains("localhost"));
    assert!(
        default_config
            .grpc_endpoint
            .contains(&constants::DEFAULT_GRPC_PORT.to_string())
    );

    // Configuration with invalid endpoint should still create but fail on connection
    let invalid_config = ClientConfig::new("invalid-url");
    let client_result = Client::new(invalid_config).await;
    assert!(client_result.is_err());
}
