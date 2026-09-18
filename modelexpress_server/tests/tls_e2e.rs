// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The real server terminating TLS, driven by the real client over loopback.
//! Gated behind `integration-tests`:
//! `cargo test -p modelexpress-server --features integration-tests --test tls_e2e` (rustls), or
//! `cargo test -p modelexpress-server --no-default-features --features openssl,integration-tests --test tls_e2e`.

#![allow(clippy::expect_used)]

use std::num::NonZeroU16;
use std::path::PathBuf;
use std::time::Duration;

use modelexpress_client::Client;
use modelexpress_common::client_config::ClientConfig;
use modelexpress_common::config::ConnectionConfig;
use modelexpress_common::tls::TlsVersion;
use modelexpress_server::backend_config::BackendConfig;
use modelexpress_server::config::{ServerConfig, TlsConfig};
use modelexpress_server::run_server;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair,
    PKCS_ECDSA_P256_SHA256,
};
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

/// A CA and the leaf it signed for 127.0.0.1 and ::1.
struct Chain {
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
}

fn chain(dir: &TempDir, stem: &str) -> Chain {
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("ca params");
    ca_params
        .distinguished_name
        .push(DnType::CommonName, format!("{stem} CA"));
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).expect("key");
    let ca = CertifiedIssuer::self_signed(ca_params, ca_key).expect("ca cert");

    let mut leaf_params = CertificateParams::new(vec!["127.0.0.1".to_string(), "::1".to_string()])
        .expect("leaf params");
    leaf_params
        .distinguished_name
        .push(DnType::CommonName, stem);
    let leaf_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).expect("key");
    let leaf = leaf_params.signed_by(&leaf_key, &ca).expect("leaf cert");

    let pair = Chain {
        ca: dir.path().join(format!("{stem}-ca.crt")),
        cert: dir.path().join(format!("{stem}.crt")),
        key: dir.path().join(format!("{stem}.key")),
    };
    std::fs::write(&pair.ca, ca.pem()).expect("write ca");
    std::fs::write(&pair.cert, leaf.pem()).expect("write cert");
    std::fs::write(&pair.key, leaf_key.serialize_pem()).expect("write key");
    pair
}

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

fn start_server(port: u16, tls: TlsConfig) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    start_server_on("127.0.0.1", port, tls)
}

fn start_server_on(
    host: &str,
    port: u16,
    tls: TlsConfig,
) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let [metrics_port] = free_ports::<1>();
    let mut config = ServerConfig::default();
    config.server.host = host.to_string();
    config.server.port = NonZeroU16::new(port).expect("port is non-zero");
    config.server.metrics_port = metrics_port;
    config.cache.eviction.enabled = false;
    config.tls = tls;

    let (tx, rx) = oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };
    let handle = tokio::spawn(run_server(config, BackendConfig::Memory, shutdown));
    (tx, handle)
}

fn client_config(scheme: &str, port: u16, ca: Option<&PathBuf>) -> ClientConfig {
    client_config_for(&format!("{scheme}://127.0.0.1:{port}"), ca)
}

fn client_config_for(endpoint: &str, ca: Option<&PathBuf>) -> ClientConfig {
    ClientConfig {
        connection: ConnectionConfig {
            tls_ca_file: ca.cloned(),
            ..ConnectionConfig::new(endpoint.to_string())
        },
        ..Default::default()
    }
}

/// Retry until the listener is up, then return the last attempt's outcome.
async fn try_connect(config: &ClientConfig) -> Result<Client, Box<modelexpress_common::Error>> {
    let mut last = Err(Box::new(modelexpress_common::Error::Transport(
        "never attempted".to_string(),
    )));
    for _ in 0..100 {
        last = Client::new(config.clone()).await;
        if last.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    last
}

async fn wait_for_listener(port: u16) {
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server never listened on {port}");
}

async fn stop(tx: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    tx.send(()).expect("server still running");
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server shut down in time")
        .expect("server task")
        .expect("server exited cleanly");
}

#[tokio::test]
async fn client_with_the_right_ca_talks_to_a_tls_server() {
    let dir = TempDir::new().expect("tempdir");
    let server_pair = chain(&dir, "server");
    let [port] = free_ports::<1>();
    let (tx, handle) = start_server(
        port,
        TlsConfig {
            cert_file: Some(server_pair.cert.clone()),
            key_file: Some(server_pair.key.clone()),
            min_version: Some(TlsVersion::Tls12),
            ..TlsConfig::default()
        },
    );

    let mut client = try_connect(&client_config("https", port, Some(&server_pair.ca)))
        .await
        .expect("TLS client connects with the server cert as CA");
    client.health_check().await.expect("health check over TLS");

    stop(tx, handle).await;
}

#[tokio::test]
async fn plaintext_client_is_refused_by_a_tls_server() {
    let dir = TempDir::new().expect("tempdir");
    let server_pair = chain(&dir, "server");
    let [port] = free_ports::<1>();
    let (tx, handle) = start_server(
        port,
        TlsConfig {
            cert_file: Some(server_pair.cert.clone()),
            key_file: Some(server_pair.key.clone()),
            ..TlsConfig::default()
        },
    );
    wait_for_listener(port).await;

    // The TCP connect succeeds, so the client only learns on the first RPC that
    // the server never speaks h2c back.
    let outcome = match Client::new(client_config("http", port, None)).await {
        Ok(mut client) => client.health_check().await.map(|_| ()),
        Err(e) => Err(e),
    };
    assert!(
        outcome.is_err(),
        "plaintext client must not reach a TLS listener"
    );

    stop(tx, handle).await;
}

#[tokio::test]
async fn client_with_the_wrong_ca_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let server_pair = chain(&dir, "server");
    let other_pair = chain(&dir, "other");
    let [port] = free_ports::<1>();
    let (tx, handle) = start_server(
        port,
        TlsConfig {
            cert_file: Some(server_pair.cert.clone()),
            key_file: Some(server_pair.key.clone()),
            ..TlsConfig::default()
        },
    );
    wait_for_listener(port).await;

    let outcome = Client::new(client_config("https", port, Some(&other_pair.ca))).await;
    assert!(
        outcome.is_err(),
        "a CA that did not issue the server cert must fail verification"
    );

    stop(tx, handle).await;
}

#[tokio::test]
async fn modern_profile_is_honored_end_to_end() {
    let dir = TempDir::new().expect("tempdir");
    let server_pair = chain(&dir, "server");
    let [port] = free_ports::<1>();
    let (tx, handle) = start_server(
        port,
        TlsConfig {
            cert_file: Some(server_pair.cert.clone()),
            key_file: Some(server_pair.key.clone()),
            min_version: Some(TlsVersion::Tls13),
            cipher_suites: vec![
                "TLS_AES_128_GCM_SHA256".to_string(),
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            groups: vec!["X25519MLKEM768".to_string(), "X25519".to_string()],
        },
    );

    let mut client = try_connect(&client_config("https", port, Some(&server_pair.ca)))
        .await
        .expect("TLS 1.3 client connects under the Modern profile");
    client
        .health_check()
        .await
        .expect("health check under Modern profile");

    stop(tx, handle).await;
}

#[tokio::test]
async fn plaintext_server_still_serves_plaintext_clients() {
    let [port] = free_ports::<1>();
    let (tx, handle) = start_server(port, TlsConfig::default());

    let mut client = try_connect(&client_config("http", port, None))
        .await
        .expect("plaintext client connects to a plaintext server");
    client.health_check().await.expect("plaintext health check");

    stop(tx, handle).await;
}

#[tokio::test]
async fn bad_certificate_paths_fail_startup() {
    let [port] = free_ports::<1>();
    let (_tx, handle) = start_server(
        port,
        TlsConfig {
            cert_file: Some(PathBuf::from("/nonexistent/tls.crt")),
            key_file: Some(PathBuf::from("/nonexistent/tls.key")),
            ..TlsConfig::default()
        },
    );
    let result = tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server task ends")
        .expect("server task joins");
    assert!(result.is_err(), "unreadable cert must fail run_server");
}

#[tokio::test]
async fn ipv6_endpoint_verifies_against_the_ip_san() {
    let Ok(probe) = std::net::TcpListener::bind("[::1]:0") else {
        eprintln!("skipping: no IPv6 loopback on this host");
        return;
    };
    let port = probe.local_addr().expect("local addr").port();
    drop(probe);

    let dir = TempDir::new().expect("tempdir");
    let server_pair = chain(&dir, "server");
    let (tx, handle) = start_server_on(
        "[::1]",
        port,
        TlsConfig {
            cert_file: Some(server_pair.cert.clone()),
            key_file: Some(server_pair.key.clone()),
            ..TlsConfig::default()
        },
    );

    let mut client = try_connect(&client_config_for(
        &format!("https://[::1]:{port}"),
        Some(&server_pair.ca),
    ))
    .await
    .expect("TLS client connects to a bracketed IPv6 endpoint");
    client.health_check().await.expect("health check over IPv6");

    stop(tx, handle).await;
}

#[tokio::test]
async fn old_profile_with_legacy_ciphers_still_serves() {
    let dir = TempDir::new().expect("tempdir");
    let server_pair = chain(&dir, "server");
    let [port] = free_ports::<1>();
    let ciphers = [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305",
        "ECDHE-ECDSA-AES128-SHA256",
        "ECDHE-RSA-AES128-SHA256",
        "ECDHE-ECDSA-AES128-SHA",
        "ECDHE-RSA-AES128-SHA",
        "ECDHE-ECDSA-AES256-SHA384",
        "ECDHE-RSA-AES256-SHA384",
        "ECDHE-ECDSA-AES256-SHA",
        "ECDHE-RSA-AES256-SHA",
        "AES128-GCM-SHA256",
        "AES256-GCM-SHA384",
        "AES128-SHA256",
        "AES256-SHA256",
        "AES128-SHA",
        "AES256-SHA",
        "DES-CBC3-SHA",
    ];
    let (tx, handle) = start_server(
        port,
        TlsConfig {
            cert_file: Some(server_pair.cert.clone()),
            key_file: Some(server_pair.key.clone()),
            min_version: Some(TlsVersion::Tls10),
            cipher_suites: ciphers.iter().map(|c| (*c).to_string()).collect(),
            groups: vec![
                "X25519MLKEM768".to_string(),
                "X25519".to_string(),
                "secp256r1".to_string(),
                "secp384r1".to_string(),
            ],
        },
    );

    let mut client = try_connect(&client_config("https", port, Some(&server_pair.ca)))
        .await
        .expect("TLS client connects under the Old profile");
    client
        .health_check()
        .await
        .expect("health check under Old profile");

    stop(tx, handle).await;
}
