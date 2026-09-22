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

/// Reserve `N` distinct ephemeral ports.
///
/// All `N` sockets are held open until every port has been read, then released
/// together. Calling a one-port helper twice can hand back the same port -- the
/// first socket is already closed by then, so the OS is free to reuse it -- and
/// the two listeners would race for it, failing whichever bound second.
fn free_ports<const N: usize>() -> [u16; N] {
    let sockets: Vec<std::net::TcpListener> = (0..N)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port"))
        .collect();
    let mut ports = [0_u16; N];
    for (slot, socket) in ports.iter_mut().zip(&sockets) {
        *slot = socket.local_addr().expect("local addr").port();
    }
    ports
}

fn start_server(port: u16) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let [metrics_port] = free_ports::<1>();
    start_server_with_metrics(port, metrics_port)
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
    let [port] = free_ports::<1>();
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
    let [port, metrics_port] = free_ports::<2>();
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

/// An early failure inside `run_server` must leave the metrics port bindable.
///
/// The listener starts before the registry, refit and P2P backends connect, and
/// any of those can fail. This pins the contract that such a failure leaves the
/// port free for the next `run_server` in the same process, which is a
/// documented use.
///
/// Scope, measured rather than assumed: the release comes from the guard's
/// oneshot sender dropping, which fires the graceful shutdown and makes axum drop
/// the listener socket. This test therefore does *not* fail if only
/// `MetricsListener::drop`'s abort is removed -- it fails when the release path
/// itself breaks, e.g. if the sender is moved somewhere that outlives the
/// function. The abort is defence in depth for the detached task, and is not
/// what this test covers.
///
/// The stalled connection keeps a request in flight across the failure, so a
/// listener that only stopped accepting would still be holding the socket.
#[tokio::test]
async fn an_early_failure_releases_the_metrics_port() {
    use tokio::io::AsyncWriteExt;

    let [port, metrics_port, dead_port] = free_ports::<3>();

    let mut config = ServerConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = NonZeroU16::new(port).expect("port is non-zero");
    config.server.metrics_port = metrics_port;
    config.cache.eviction.enabled = false;

    // Redis on a port nothing is listening on: the registry connect fails and
    // run_server returns Err long before its normal shutdown path.
    let backend = BackendConfig::Redis {
        url: format!("redis://127.0.0.1:{dead_port}"),
    };

    let (_tx, rx) = oneshot::channel::<()>();
    let server = tokio::spawn(run_server(config, backend, async move {
        let _ = rx.await;
    }));

    // Hold a connection that has written a partial request head and stalled.
    // Kept alive for the rest of the test.
    let mut stalled = None;
    for _ in 0..200 {
        match tokio::net::TcpStream::connect(("127.0.0.1", metrics_port)).await {
            Ok(mut stream) => {
                let partial =
                    format!("GET /metrics HTTP/1.1\r\nHost: 127.0.0.1:{metrics_port}\r\n");
                stream
                    .write_all(partial.as_bytes())
                    .await
                    .expect("write partial request head");
                stream.flush().await.expect("flush");
                stalled = Some(stream);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
        }
    }
    assert!(stalled.is_some(), "metrics listener never came up");

    let result = tokio::time::timeout(Duration::from_secs(60), server)
        .await
        .expect("run_server did not return")
        .expect("run_server task panicked");
    assert!(result.is_err(), "expected the backend connect to fail");

    // The port must be free again even though the stalled connection is still
    // open. Retry briefly: the abort is asynchronous.
    let mut bound = None;
    for _ in 0..100 {
        match std::net::TcpListener::bind(("127.0.0.1", metrics_port)) {
            Ok(listener) => {
                bound = Some(listener);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
    drop(stalled);
    assert!(
        bound.is_some(),
        "metrics port {metrics_port} is still held after run_server returned early; \
         the listener was detached rather than stopped"
    );
}

/// The Phase 2 exit criterion, end to end: a handler that fails **in band** is
/// reported as a failure by the metrics layer.
///
/// `PublishMetadata` with no identity returns `Ok` with `success: false`. The
/// gRPC status is therefore `Ok`, so a status-code-derived metric would record
/// this as a success -- and would likewise record a total Redis outage as a
/// success, since `ListSources` reports that failure the same way.
///
/// This is also the only test that can exercise the *propagation* half of the
/// mechanism. The handler inserts an `RpcOutcome` into its response extensions
/// and the tower layer reads it back off the encoded `http::Response`; the step
/// between them is `tonic::Response::into_http`, which is `pub(crate)`, so no
/// unit test outside tonic can drive it.
#[tokio::test]
async fn an_in_band_failure_is_not_recorded_as_a_success() {
    use modelexpress_common::grpc::p2p::PublishMetadataRequest;
    use modelexpress_common::grpc::p2p::p2p_service_client::P2pServiceClient;

    let [port, metrics_port] = free_ports::<2>();
    let (shutdown, handle) = start_server_with_metrics(port, metrics_port);

    // Wait for the gRPC side before driving it.
    let mut client = connect_client(port).await;
    client
        .health_check()
        .await
        .expect("health_check round-trip should succeed");

    let mut p2p = P2pServiceClient::connect(format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect to the p2p service");

    let response = p2p
        .publish_metadata(PublishMetadataRequest {
            identity: None,
            ..Default::default()
        })
        .await
        .expect("the RPC itself succeeds -- that is the point");
    assert!(
        !response.into_inner().success,
        "expected an in-band failure, not a transport failure"
    );

    let body = scrape(metrics_port).await;

    // The outcome came from the handler, not from the status code.
    assert!(
        body.contains(
            r#"mx_grpc_requests_total{method="P2pService/PublishMetadata",outcome="invalid_argument"} 1"#
        ),
        "expected the handler's own outcome in the scrape, got: {body}"
    );
    // The same call must not also be counted as a success.
    assert!(
        !body.contains(
            r#"mx_grpc_requests_total{method="P2pService/PublishMetadata",outcome="ok"}"#
        ),
        "the in-band failure was also counted as a success: {body}"
    );
    // Latency is observed under the same label set.
    assert!(
        body.contains(r#"mx_grpc_request_seconds_count{method="P2pService/PublishMetadata""#),
        "expected a latency observation, got: {body}"
    );
    // The health probe that got us here is counted too, proving one layer covers
    // services it was never explicitly attached to.
    assert!(
        body.contains(r#"method="HealthService/GetHealth""#),
        "expected the health service to be covered by the same layer, got: {body}"
    );

    stop_and_join(shutdown, handle).await;
}
