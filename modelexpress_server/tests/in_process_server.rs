// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boot the real server via `run_server` (in-memory backend) and drive it with the real
//! client over loopback. The two tests run in parallel, so two servers share the process
//! at once.
//!
//! These boot a server, so they're gated behind the `integration-tests` feature and skipped
//! by default: `cargo test -p modelexpress-server --features integration-tests`.

#![allow(clippy::expect_used)]

use std::num::NonZeroU16;
use std::time::Duration;

use modelexpress_client::Client;
use modelexpress_common::client_config::ClientConfig;
use modelexpress_common::config::ConnectionConfig;
use modelexpress_server::backend_config::BackendConfig;
use modelexpress_server::config::ServerConfig;
use modelexpress_server::run_server;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn start_server(port: u16) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    start_server_with_metrics(port, free_port())
}

fn start_server_with_metrics(
    port: u16,
    metrics_port: u16,
) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let mut config = ServerConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = NonZeroU16::new(port).expect("port is non-zero");
    // An ephemeral metrics port per server. These tests deliberately run two
    // servers in one process, and the default 9401 would have the second lose
    // the bind — which the server tolerates by design, but which would leave the
    // tests silently exercising the degraded path.
    config.server.metrics_port = metrics_port;
    config.cache.eviction.enabled = false;

    let (tx, rx) = oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };
    let handle = tokio::spawn(run_server(config, BackendConfig::Memory, shutdown));
    (tx, handle)
}

async fn connect_client(port: u16) -> Client {
    let config = ClientConfig {
        connection: ConnectionConfig::new(format!("http://127.0.0.1:{port}")),
        ..Default::default()
    };
    for _ in 0..100 {
        if let Ok(client) = Client::new(config.clone()).await {
            return client;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server on port {port} never became reachable");
}

async fn stop_and_join(shutdown: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    let _ = shutdown.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server task did not exit in time")
        .expect("server task panicked")
        .expect("run_server returned an error");
}

async fn assert_boots_and_serves() {
    let port = free_port();
    let (shutdown, handle) = start_server(port);

    let mut client = connect_client(port).await;
    client
        .health_check()
        .await
        .expect("health_check round-trip should succeed");

    stop_and_join(shutdown, handle).await;
}

#[tokio::test]
async fn server_boots_and_serves_a_client() {
    assert_boots_and_serves().await;
}

// A second, independent server. cargo runs the tests in parallel, so this one and the
// one above stand up two `run_server` instances in the same process at the same time.
#[tokio::test]
async fn another_server_boots_and_serves_a_client() {
    assert_boots_and_serves().await;
}

/// A real scrape of a real server boot, over the protocol Prometheus uses.
///
/// This is the exit criterion for the metrics work: `up == 1` on a server that
/// has served no traffic at all. The chart's scrape annotation used to point at
/// the tonic gRPC listener, which speaks HTTP/2 only, so an HTTP/1.1
/// `GET /metrics` could never complete and every server pod reported `up == 0`
/// permanently — indistinguishable from a crashed pod.
#[tokio::test]
async fn server_serves_metrics_over_http1() {
    let port = free_port();
    let metrics_port = free_port();
    let (shutdown, handle) = start_server_with_metrics(port, metrics_port);

    // Wait for the gRPC side, so the metrics assertion is about a server that
    // is genuinely up rather than about a race.
    let mut client = connect_client(port).await;
    client
        .health_check()
        .await
        .expect("health_check round-trip should succeed");

    let body = scrape(metrics_port).await;
    assert!(
        body.starts_with("HTTP/1.1 200"),
        "expected an HTTP/1.1 200, got: {body}"
    );
    assert!(
        body.contains("mx_build_info"),
        "expected mx_build_info in the scrape, got: {body}"
    );
    assert!(
        body.contains(r#"component="server""#),
        "expected the server component label, got: {body}"
    );
    assert!(
        body.contains(r#"backend="memory""#),
        "expected the metadata backend on mx_build_info, got: {body}"
    );

    // The gRPC port must NOT answer an HTTP/1.1 GET. This is the defect stated
    // as an assertion: if this ever starts passing, someone has pointed the two
    // listeners at one port and the scrape target will go down.
    let grpc_body = scrape(port).await;
    assert!(
        !grpc_body.starts_with("HTTP/1.1 200"),
        "the gRPC port answered an HTTP/1.1 scrape; the ports have been merged: {grpc_body}"
    );

    stop_and_join(shutdown, handle).await;
}

/// A raw HTTP/1.1 `GET /metrics`, retried while the listener comes up.
async fn scrape(port: u16) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    for _ in 0..100 {
        let Ok(mut stream) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
            tokio::time::sleep(Duration::from_millis(50)).await;
            continue;
        };
        let request =
            format!("GET /metrics HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
        if stream.write_all(request.as_bytes()).await.is_err() {
            continue;
        }
        let mut body = String::new();
        match tokio::time::timeout(Duration::from_secs(5), stream.read_to_string(&mut body)).await {
            Ok(Ok(_)) => return body,
            _ => return String::new(),
        }
    }
    String::new()
}
