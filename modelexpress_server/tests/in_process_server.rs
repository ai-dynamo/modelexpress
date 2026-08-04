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
use modelexpress_common::grpc::revision::revision_catalog_service_client::RevisionCatalogServiceClient;
use modelexpress_common::grpc::revision::{
    ChangeState, CommitVersionRequest, DeltaLocation, DeltaTransferMethod, GetRevisionRequest,
    PublishRevisionRequest, RankDelta, RevisionLifecycleState, RevisionManifest, RevisionRank,
    S3Location, delta_location,
};
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

#[tokio::test]
async fn revision_catalog_runs_through_the_real_server_router() {
    let port = free_port();
    let (shutdown, handle) = start_server(port);
    let endpoint = format!("http://127.0.0.1:{port}");
    let mut client = loop {
        match RevisionCatalogServiceClient::connect(endpoint.clone()).await {
            Ok(client) => break client,
            Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    };
    let manifest = RevisionManifest {
        model_id: "model".to_string(),
        version: "v1".to_string(),
        base_version: Some("v0".to_string()),
        transfer_method: DeltaTransferMethod::Canonical as i32,
        delta_method: Some("xor".to_string()),
        compression_algorithm: Some("zstd".to_string()),
        format_digest: "format".to_string(),
        base_digest: Some("base".to_string()),
        target_digest: "target".to_string(),
        ranks: vec![RevisionRank {
            trainer_rank: 0,
            producer_id: "producer".to_string(),
            source_layout_digest: "layout".to_string(),
            delta: Some(RankDelta {
                change_state: ChangeState::Dirty as i32,
                checksum: Some("deadbeef".to_string()),
                location: Some(DeltaLocation {
                    transport: Some(delta_location::Transport::S3(S3Location {
                        bucket: "bucket".to_string(),
                        key: "model/v1/root.json".to_string(),
                        object_version: Some("object-v1".to_string()),
                    })),
                }),
                delta_descriptor: None,
            }),
            shards: vec![],
        }],
    };
    let published = client
        .publish_revision(PublishRevisionRequest {
            manifest: Some(manifest),
            publisher_id: "trainer".to_string(),
            publication_mode: None,
        })
        .await
        .expect("publish over network")
        .into_inner();
    assert!(published.created);

    let committed = client
        .commit_version(CommitVersionRequest {
            model_id: "model".to_string(),
            version: "v1".to_string(),
        })
        .await
        .expect("commit over network")
        .into_inner();
    assert_eq!(
        committed.revision.as_ref().map(|record| record.state),
        Some(RevisionLifecycleState::Committed as i32)
    );

    let fetched = client
        .get_revision(GetRevisionRequest {
            model_id: "model".to_string(),
            version: "v1".to_string(),
        })
        .await
        .expect("get over network")
        .into_inner();
    assert_eq!(fetched.revision, committed.revision);
    stop_and_join(shutdown, handle).await;
}
