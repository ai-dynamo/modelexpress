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
    refit::refit_service_server::RefitServiceServer,
};
use tonic::transport::Server;
use tower::Layer;
use tracing::{error, info, warn};

use crate::auth::{AuthLayer, AuthState};
use crate::backend_config::BackendConfig;
use crate::cache::CacheEvictionService;
use crate::config::{AuthMode, ServerConfig};
use crate::metrics;
use crate::p2p::{service::P2pServiceImpl, state::P2pStateManager};
use crate::refit::{
    backend::create_backend as create_refit_backend,
    backend::instrumented::InstrumentedRefitBackend, service::RefitServiceImpl,
};
use crate::registry::state::RegistryManager;
use crate::services::{ApiServiceImpl, HealthServiceImpl, ModelDownloadTracker, ModelServiceImpl};

/// Maximum gRPC message size (100MB) for large models like DeepSeek-V3.
/// Each worker can have thousands of tensor descriptors with NIXL metadata.
const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024;

/// How long to wait for the metrics listener to drain before abandoning it.
///
/// Short on purpose. A scrape is a sub-second GET, so anything still open after
/// this is a stalled client rather than work worth waiting for — and this is the
/// last await in [`run_server`], so waiting on it delays nothing else and blocks
/// everything.
const METRICS_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Owns the metrics listener task and stops it on **every** exit path.
///
/// [`run_server`] has several fallible steps between starting this listener and
/// reaching its normal shutdown -- registry connect, refit backend, P2P connect,
/// auth setup -- and it is documented as callable more than once in a process.
/// Without a guard, an early `?` return left the task *detached*: still draining,
/// owned by nobody, for as long as a client held a connection open.
///
/// The port itself was already released on that path, and it is worth being
/// precise about why, because the reason is easy to break. Dropping the guard
/// drops its oneshot sender, which closes the channel, resolves the shutdown
/// future and makes axum drop the listener socket. Measured: suppressing the
/// abort alone still frees the port; only suppressing the abort *and* the
/// shutdown signal holds it. So the drop-chain does the freeing and this guard
/// bounds the task -- and makes a chain that used to be implicit explicit, so
/// moving the sender somewhere longer-lived cannot silently undo it.
struct MetricsListener {
    handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MetricsListener {
    /// Start the listener. Returns `None` when metrics are disabled.
    ///
    /// The registry is built and fully populated by the caller. It has to be:
    /// `Registry::register` takes `&mut self` and there is no interior
    /// mutability, so once the registry is behind the `Arc` this task shares, no
    /// further family can be added. Building it here would make the listener the
    /// only thing that could ever own a metric.
    fn spawn(
        metrics_addr: Option<std::net::SocketAddr>,
        registry: Arc<prometheus_client::registry::Registry>,
    ) -> Option<Self> {
        let metrics_addr = metrics_addr?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(metrics::serve(metrics_addr, registry, async move {
            let _ = shutdown_rx.await;
        }));
        Some(Self {
            handle: Some(handle),
            shutdown_tx: Some(shutdown_tx),
        })
    }

    /// Signal, drain with a deadline, then abort if it overruns.
    ///
    /// Bounded because the listener is unauthenticated and reachable from the pod
    /// network, and hyper's graceful shutdown holds a connection that wrote a
    /// partial request head open indefinitely. Without a deadline any client
    /// killed mid-write -- a scraper, a probe, a port scan -- would wedge the last
    /// await in `run_server` and burn the whole termination grace period.
    async fn shutdown(mut self) {
        // A send failure just means the task already finished -- the normal case
        // when the bind failed, which is a supported degraded state. The join
        // below is the real synchronization point, so this must not be an error.
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        // `take` leaves `Drop` with nothing to do, so the abort below is the only
        // one that can run.
        let Some(mut handle) = self.handle.take() else {
            return;
        };
        match tokio::time::timeout(METRICS_DRAIN_TIMEOUT, &mut handle).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!("Metrics listener join error: {e}"),
            Err(_) => {
                warn!(
                    "Metrics listener did not drain within {}s; aborting it to finish shutdown",
                    METRICS_DRAIN_TIMEOUT.as_secs()
                );
                handle.abort();
                let _ = handle.await;
            }
        }
    }
}

impl Drop for MetricsListener {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Reached only when `run_server` returned early. There is no async
            // context to drain in here, so stop the task outright rather than
            // leaving it detached and draining on its own.
            handle.abort();
        }
    }
}

/// Run the ModelExpress gRPC server to completion.
///
/// Connects the registry and P2P metadata backends (failing fast if either is
/// unreachable), starts the cache-eviction and reaper background tasks, serves all
/// gRPC services, and tears everything down once `shutdown` resolves. Logging is the
/// caller's responsibility: install a subscriber before calling this.
///
/// All server state (registry, download tracker, P2P) is instance-scoped, so this
/// can be called multiple times in one process, including concurrently. The metadata
/// `backend` is injected by the caller, so this never reads process env itself.
pub async fn run_server(
    config: ServerConfig,
    backend: BackendConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting ModelExpress server...");
    config.print_config();

    // Get server address
    let addr = config.socket_addr().map_err(|e| {
        error!("Invalid server address: {e}");
        e
    })?;
    let metrics_addr = config.metrics_socket_addr().map_err(|e| {
        error!("Invalid metrics address: {e}");
        e
    })?;

    // Start the Prometheus listener before anything else can fail, so a scrape
    // proves the exporter came up even on a server that never reaches a healthy
    // state. It is a separate HTTP/1.1 listener on its own port because tonic
    // serves HTTP/2 only — a scrape aimed at the gRPC port can never succeed.
    // Its shutdown is signalled at the very end of this function, after the gRPC
    // server has drained, so the drain window stays scrapeable.
    //
    // Held in a guard so every return path stops it: the steps below can fail,
    // and a detached listener would keep the port for the next call.
    //
    // The families are registered unconditionally, even when the listener is
    // disabled. The alternative -- an `Option` at every instrumentation site --
    // buys four `Arc` allocations and a second, untested code path.
    let mut metrics_registry = metrics::new_registry();
    metrics::register_build_info(&mut metrics_registry, &backend);
    let grpc_metrics = metrics::grpc::GrpcMetrics::register(&mut metrics_registry);
    let backend_metrics = metrics::backend::BackendMetrics::register(&mut metrics_registry);
    let registry_metrics = metrics::registry::RegistryMetrics::register(&mut metrics_registry);
    let download_metrics = metrics::registry::DownloadMetrics::register(&mut metrics_registry);
    let cache_metrics = metrics::cache::CacheMetrics::register(&mut metrics_registry);
    let metrics_registry = Arc::new(metrics_registry);

    let metrics_listener = MetricsListener::spawn(metrics_addr, Arc::clone(&metrics_registry));
    if metrics_listener.is_none() {
        info!("Metrics endpoint is disabled");
    }

    // Initialize the model registry manager (Redis or Kubernetes CRDs). Shares the
    // injected backend with the P2P state manager below.
    let registry = Arc::new(
        RegistryManager::with_config(backend.clone())
            .with_metrics(backend_metrics.clone(), registry_metrics.clone()),
    );
    match tokio::time::timeout(std::time::Duration::from_secs(10), registry.connect()).await {
        Ok(Ok(backend_name)) => info!("Model registry connected (backend: {backend_name})"),
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
    let tracker = Arc::new(ModelDownloadTracker::new(
        registry.clone(),
        download_metrics,
        registry_metrics.clone(),
    ));

    // Create cache eviction service
    let cache_service = CacheEvictionService::new(
        registry.clone(),
        config.cache.eviction.clone(),
        config.cache.directory.clone(),
        cache_metrics.clone(),
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
    let stats_tracker = tracker.clone();
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

    // Timed inside the timeout, matching the P2P and registry managers: the
    // factory connects the concrete backend before it can be wrapped, so the
    // decorator's own `connect` forward is never reached.
    //
    // A connect that exceeds the timeout is recorded too: `BackendMetrics::time`
    // installs its guard before the await, so the dropped future lands as
    // `result="cancelled"` rather than as silence.
    let refit_backend = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        backend_metrics.time(
            metrics::backend::Store::Refit,
            "connect",
            create_refit_backend(&backend),
        ),
    )
    .await
    {
        Ok(Ok(refit_backend)) => refit_backend,
        Ok(Err(error)) => {
            error!("Failed to connect Refit metadata backend: {error}");
            return Err(error.to_string().into());
        }
        Err(_) => {
            error!("Timed out connecting to Refit metadata backend");
            return Err("Refit metadata backend connection timed out".into());
        }
    };
    let refit_service = refit_backend
        .map(|inner| InstrumentedRefitBackend::wrap(inner, backend_metrics.clone()))
        .map(RefitServiceImpl::new);
    if refit_service.is_some() {
        health_reporter
            .set_serving::<RefitServiceServer<RefitServiceImpl>>()
            .await;
    } else {
        info!("Refit service is unavailable: this metadata backend is not implemented yet");
    }

    // Initialize P2P state manager — fails fast if backend is misconfigured or unreachable
    let p2p_state =
        Arc::new(P2pStateManager::with_config(backend).with_metrics(backend_metrics.clone()));

    match tokio::time::timeout(std::time::Duration::from_secs(10), p2p_state.connect()).await {
        Ok(Ok(backend_name)) => info!("P2P state manager connected (backend: {backend_name})"),
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

    // Registry statistics refresh. Independent of the cache-eviction service on
    // purpose: that one ticks hourly and is skipped entirely when eviction is
    // disabled, which would leave these gauges permanently absent rather than
    // merely stale.
    let (stats_shutdown_tx, stats_shutdown_rx) = tokio::sync::oneshot::channel();
    let stats_registry = registry.clone();
    let stats_metrics = cache_metrics.clone();
    let stats_handle = tokio::spawn(async move {
        crate::registry::stats_refresh::run_stats_refresh(
            stats_registry,
            stats_metrics,
            std::sync::Arc::new(move || stats_tracker.waiting_count()),
            stats_shutdown_rx,
        )
        .await;
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

        // Signal registry stats refresh to shutdown
        if stats_shutdown_tx.send(()).is_err() {
            error!("Failed to send shutdown signal to registry stats refresh");
        }
    };

    let mode = config.security.resolve_mode();
    let auth_layer = if mode == AuthMode::Off {
        info!("ServiceAccount auth: disabled");
        None
    } else {
        config
            .security
            .validate_resolved(mode)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
        let client = kube::Client::try_default().await.map_err(|e| {
            error!("ServiceAccount auth enabled but kube client init failed: {e}");
            e
        })?;
        info!(
            "ServiceAccount auth: enforce ({} allowed service account(s), {} audience(s))",
            config.security.allowed_service_accounts.len(),
            config.security.token_audiences.len()
        );
        Some(AuthLayer::new(Arc::new(AuthState::new(
            client,
            &config.security,
        ))))
    };

    let api = ApiServiceServer::new(api_service);
    let model = ModelServiceServer::new(model_service);
    let p2p = P2pServiceServer::new(p2p_service)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE);
    let refit = refit_service.map(|service| {
        RefitServiceServer::new(service)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
    });

    info!("Starting gRPC server on: {addr}");
    // One layer for the whole router rather than one per service: it covers the
    // two health services as well, and a service added later is instrumented
    // without a second edit. It sits outside `AuthLayer`, so a rejected call is
    // counted as `outcome="unauthenticated"` instead of vanishing.
    let router = Server::builder()
        .layer(metrics::grpc::GrpcMetricsLayer::new(grpc_metrics))
        .add_service(health_service_v1)
        .add_service(HealthServiceServer::new(health_service));
    let router = match &auth_layer {
        Some(layer) => router
            .add_service(layer.layer(api))
            .add_service(layer.layer(model))
            .add_service(layer.layer(p2p))
            .add_optional_service(refit.map(|service| layer.layer(service))),
        None => router
            .add_service(api)
            .add_service(model)
            .add_service(p2p)
            .add_optional_service(refit),
    };
    let server_result = router.serve_with_shutdown(addr, shutdown_signal).await;

    // Wait for background services to complete
    if let Some(handle) = cache_handle
        && let Err(e) = handle.await
    {
        error!("Cache eviction service join error: {e}");
    }
    if let Err(e) = stats_handle.await {
        error!("Registry stats refresh join error: {e}");
    }
    if let Err(e) = reaper_handle.await {
        error!("Reaper join error: {e}");
    }

    // The metrics listener stops last, once the gRPC server has drained and the
    // background tasks have joined. Signalling it alongside them would make the
    // shutdown window unscrapeable during exactly the window these metrics exist
    // to explain.
    if let Some(metrics_listener) = metrics_listener {
        metrics_listener.shutdown().await;
    }

    server_result?;
    info!("Server shutdown complete");
    Ok(())
}
