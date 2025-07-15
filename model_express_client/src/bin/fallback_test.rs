#![allow(clippy::expect_used)]

use model_express_client::{Client, ClientConfig, ModelProvider};
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Testing model download with server fallback...");

    let model_name = "google-t5/t5-small";

    // Test smart fallback - this should work whether server is running or not
    info!("Attempting to download model with smart fallback...");

    match Client::request_model_with_smart_fallback(
        model_name,
        ModelProvider::HuggingFace,
        ClientConfig::default(),
    )
    .await
    {
        Ok(()) => {
            info!("✅ SUCCESS: Model '{model_name}' downloaded successfully!");
            info!(
                "The download worked either via server (if running) or direct download (if server unavailable)"
            );
        }
        Err(e) => {
            error!("❌ FAILED: Could not download model '{model_name}': {e}");
            return Err(e.into());
        }
    }

    Ok(())
}
