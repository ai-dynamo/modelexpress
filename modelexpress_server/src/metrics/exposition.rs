// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `/metrics` HTTP listener.
//!
//! A second listener on its own port, separate from the tonic gRPC server. It
//! has to be separate: tonic serves HTTP/2 only, and Prometheus scrapes with an
//! HTTP/1.1 `GET`, so the two cannot share a port.
//!
//! `axum` already reaches the lock file as a transitive dependency of tonic, so
//! the workspace pins it with `default-features = false, features = ["http1",
//! "tokio"]`. A GET-only route needs none of the form/json/query extractors the
//! default feature set enables, and turning them on would pull
//! `serde_path_to_error` and seven more dependency edges into the server binary.
//!
//! Two behaviours are load-bearing:
//!
//! - **A bind failure logs and returns.** The model cache service must not fail
//!   to start because something else holds the metrics port.
//! - **The listener shuts down last.** [`crate::server::run_server`] signals it
//!   only after the gRPC server has drained and the background tasks have
//!   joined, so the drain window — exactly the window these metrics exist to
//!   explain — stays scrapeable while it is happening.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::{error, info};

use prometheus_client::registry::Registry;

use super::OPENMETRICS_CONTENT_TYPE;

/// Serve `/metrics` on `addr` until `shutdown` resolves.
///
/// Never returns an error to the caller: a metrics listener that cannot bind is
/// a degraded deployment, not a failed one, so every failure is logged and
/// swallowed here rather than propagated into server startup.
pub async fn serve(
    addr: SocketAddr,
    registry: Arc<Registry>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let app = Router::new()
        .route("/metrics", get(handler))
        .with_state(registry);

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            error!(
                "Metrics listener failed to bind {addr}: {e}. \
                 Continuing without a metrics endpoint."
            );
            return;
        }
    };

    info!("Metrics endpoint listening on http://{addr}/metrics");
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
    {
        error!("Metrics listener error: {e}");
    }
    info!("Metrics endpoint stopped");
}

/// Encode the registry on demand.
///
/// This is intentionally the whole handler: everything it encodes is already in
/// memory. Anything that would need a Redis `SCAN` or another keyspace walk to
/// compute must be refreshed by a background task into a plain gauge instead, or
/// every scrape interval would put that walk on the backend.
async fn handler(State(registry): State<Arc<Registry>>) -> Response {
    match super::encode_text(&registry) {
        Ok(body) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, OPENMETRICS_CONTENT_TYPE)],
            body,
        )
            .into_response(),
        Err(e) => {
            error!("Failed to encode metrics: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to encode metrics\n",
            )
                .into_response()
        }
    }
}
