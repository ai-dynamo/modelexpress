#![allow(clippy::expect_used, clippy::unwrap_used)]

use model_express_client::{Client, ClientConfig, ModelProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    println!("Testing model download with server fallback...");

    let model_name = "google-t5/t5-small";

    // Test smart fallback - this should work whether server is running or not
    println!("Attempting to download model with smart fallback...");

    match Client::request_model_with_smart_fallback(
        model_name,
        ModelProvider::HuggingFace,
        ClientConfig::default(),
    )
    .await
    {
        Ok(()) => {
            println!("✅ SUCCESS: Model '{model_name}' downloaded successfully!");
            println!(
                "The download worked either via server (if running) or direct download (if server unavailable)"
            );
        }
        Err(e) => {
            println!("❌ FAILED: Could not download model '{model_name}': {e}");
            return Err(e.into());
        }
    }

    Ok(())
}
