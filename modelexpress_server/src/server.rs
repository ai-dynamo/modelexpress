// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable server entrypoint. `main` is a thin shell over [`run_server`] so the
//! whole startup path (registry, P2P state, health, reaper, graceful shutdown) can
//! be embedded by a downstream binary that provides its own configuration or services.

use std::future::Future;
use std::sync::Arc;

use modelexpress_common::grpc::{
    api::api_service_server::ApiServiceServer, health::health_service_server::HealthServiceServer,
    model::model_service_server::ModelServiceServer, p2p::p2p_service_server::P2pServiceServer,
};
use tonic::transport::Server;
use tracing::{error, info};

use crate::cache::CacheEvictionService;
use crate::config::ServerConfig;
use crate::p2p::{service::P2pServiceImpl, state::P2pStateManager};
use crate::registry::state::RegistryManager;
use crate::services::{ApiServiceImpl, HealthServiceImpl, ModelDownloadTracker, ModelServiceImpl};

/// Maximum gRPC message size (100MB) for large models like DeepSeek-V3.
/// Each worker can have thousands of tensor descriptors with NIXL metadata.
const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024;

/// Run the ModelExpress gRPC server to completion.
///
/// Connects the registry and P2P metadata backends (failing fast if either is
/// unreachable), starts the cache-eviction and reaper background tasks, serves all
/// gRPC services, and tears everything down once `shutdown` resolves. Logging is the
/// caller's responsibility: install a subscriber before calling this.
///
/// All server state (registry, download tracker, P2P) is instance-scoped, so this
/// can be called multiple times in one process, including concurrently.
pub async fn run_server(
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting ModelExpress server...");
    config.print_config();

    // Get server address
    let addr = config.socket_addr().map_err(|e| {
        error!("Invalid server address: {e}");
        e
    })?;

    // Initialize the model registry manager (Redis or Kubernetes CRDs). Shares the
    // MX_METADATA_BACKEND selector with the P2P state manager below.
    let registry = Arc::new(RegistryManager::new());
    match tokio::time::timeout(std::time::Duration::from_secs(10), registry.connect()).await {
        Ok(Ok(backend)) => info!("Model registry connected (backend: {backend})"),
        Ok(Err(e)) => {
            error!("Failed to connect to model registry backend: {}", e);
            return Err(e.to_string().into());
        }
        Err(_) => {
            error!("Timed out connecting to model registry backend");
            return Err("model registry backend connection timed out".into());
        }
    }

    // Initialize the download tracker, injected with the registry.
    let tracker = Arc::new(ModelDownloadTracker::new(registry.clone()));

    // Create cache eviction service
    let cache_service = CacheEvictionService::new(
        registry.clone(),
        config.cache.eviction.clone(),
        config.cache.directory.clone(),
    );

    // Create shutdown channels
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    // Start cache eviction service in background
    let cache_handle = if config.cache.eviction.enabled {
        info!("Starting cache eviction service...");
        Some(tokio::spawn(async move {
            if let Err(e) = cache_service.start(shutdown_rx).await {
                error!("Cache eviction service error: {e}");
            }
        }))
    } else {
        info!("Cache eviction service is disabled");
        None
    };

    // Create service implementations
    let health_service = HealthServiceImpl;
    let api_service = ApiServiceImpl;
    let model_service = ModelServiceImpl::new(tracker);

    // Create standard gRPC health service (grpc.health.v1.Health)
    let (health_reporter, health_service_v1) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<HealthServiceServer<HealthServiceImpl>>()
        .await;
    health_reporter
        .set_serving::<ApiServiceServer<ApiServiceImpl>>()
        .await;
    health_reporter
        .set_serving::<ModelServiceServer<ModelServiceImpl>>()
        .await;
    health_reporter
        .set_serving::<P2pServiceServer<P2pServiceImpl>>()
        .await;

    // Initialize P2P state manager — fails fast if backend is misconfigured or unreachable
    let p2p_state = Arc::new(P2pStateManager::new());

    match tokio::time::timeout(std::time::Duration::from_secs(10), p2p_state.connect()).await {
        Ok(Ok(backend)) => info!("P2P state manager connected (backend: {backend})"),
        Ok(Err(e)) => {
            error!("Failed to connect to P2P metadata backend: {}", e);
            return Err(e);
        }
        Err(_) => {
            error!("Timed out connecting to P2P metadata backend");
            return Err("P2P metadata backend connection timed out".into());
        }
    }

    let p2p_service = P2pServiceImpl::new(p2p_state.clone());

    // Start reaper for stale source detection
    let (reaper_shutdown_tx, reaper_shutdown_rx) = tokio::sync::oneshot::channel();
    let reaper_state = p2p_state.clone();
    let reaper_handle = tokio::spawn(async move {
        crate::p2p::reaper::run_reaper(reaper_state, reaper_shutdown_rx).await;
    });

    // Fan the caller's shutdown trigger out to the background tasks, then let
    // serve_with_shutdown observe the same trigger to stop accepting connections.
    let shutdown_signal = async move {
        shutdown.await;

        // Signal cache eviction service to shutdown
        if shutdown_tx.send(()).is_err() {
            error!("Failed to send shutdown signal to cache eviction service");
        }

        // Signal reaper to shutdown
        if reaper_shutdown_tx.send(()).is_err() {
            error!("Failed to send shutdown signal to reaper");
        }
    };

    // Start the gRPC server
    info!("Starting gRPC server on: {addr}");
    let server_result = Server::builder()
        .add_service(health_service_v1)
        .add_service(HealthServiceServer::new(health_service))
        .add_service(ApiServiceServer::new(api_service))
        .add_service(ModelServiceServer::new(model_service))
        .add_service(
            P2pServiceServer::new(p2p_service)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE),
        )
        .serve_with_shutdown(addr, shutdown_signal)
        .await;

    // Wait for background services to complete
    if let Some(handle) = cache_handle
        && let Err(e) = handle.await
    {
        error!("Cache eviction service join error: {e}");
    }
    if let Err(e) = reaper_handle.await {
        error!("Reaper join error: {e}");
    }

    server_result?;
    info!("Server shutdown complete");
    Ok(())
}
