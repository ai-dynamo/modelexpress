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

use super::{MetricsRegistry, OPENMETRICS_CONTENT_TYPE};

/// Serve `/metrics` on `addr` until `shutdown` resolves.
///
/// Never returns an error to the caller: a metrics listener that cannot bind is
/// a degraded deployment, not a failed one, so every failure is logged and
/// swallowed here rather than propagated into server startup.
pub async fn serve(
    addr: SocketAddr,
    registry: Arc<MetricsRegistry>,
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
async fn handler(State(registry): State<Arc<MetricsRegistry>>) -> Response {
    match registry.encode_text() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_config::BackendConfig;
    use crate::metrics::BuildInfo;
    use std::net::{Ipv4Addr, SocketAddrV4};

    /// Bind an ephemeral port, serve, and scrape it over real HTTP/1.1 — the
    /// protocol Prometheus actually uses, and the one the old gRPC-port
    /// annotation could never satisfy.
    #[tokio::test]
    async fn serves_openmetrics_over_http1() {
        let mut registry = MetricsRegistry::new();
        let _build_info = BuildInfo::register(
            &mut registry,
            &BackendConfig::Redis {
                url: "redis://localhost:6379".to_string(),
            },
        );

        // Bind :0 first to learn a free port, then release it for `serve`.
        let probe = match tokio::net::TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::LOCALHOST,
            0,
        )))
        .await
        {
            Ok(probe) => probe,
            Err(e) => panic!("failed to reserve an ephemeral port: {e}"),
        };
        let addr = match probe.local_addr() {
            Ok(addr) => addr,
            Err(e) => panic!("failed to read the reserved port: {e}"),
        };
        drop(probe);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(serve(addr, Arc::new(registry), async move {
            let _ = shutdown_rx.await;
        }));

        let body = scrape_with_retry(addr).await;
        assert!(
            body.contains("mx_build_info"),
            "expected mx_build_info in the scrape, got: {body}"
        );
        assert!(
            body.contains("# EOF"),
            "expected an OpenMetrics EOF marker, got: {body}"
        );

        let _ = shutdown_tx.send(());
        let _ = handle.await;
    }

    /// A raw HTTP/1.1 scrape, retried while the listener comes up.
    async fn scrape_with_retry(addr: SocketAddr) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        for _ in 0..50_i32 {
            let Ok(mut stream) = tokio::net::TcpStream::connect(addr).await else {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                continue;
            };
            let request =
                format!("GET /metrics HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
            if stream.write_all(request.as_bytes()).await.is_err() {
                continue;
            }
            let mut body = String::new();
            if stream.read_to_string(&mut body).await.is_ok() {
                return body;
            }
        }
        String::from("<no response>")
    }
}
