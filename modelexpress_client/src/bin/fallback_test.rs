// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

use modelexpress_client::{Client, ClientConfig, ModelProvider};
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    info!("Testing smart fallback with unavailable server...");

    Client::request_model_with_smart_fallback(
        "google-t5/t5-small",
        ModelProvider::HuggingFace,
        ClientConfig::for_testing("http://127.0.0.1:54321"),
        false,
    )
    .await?;

    Ok(())
}
