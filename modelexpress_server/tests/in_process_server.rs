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
    let mut config = ServerConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = NonZeroU16::new(port).expect("port is non-zero");
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
