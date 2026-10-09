// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TLS termination for the gRPC listener: the listener is wrapped and handed
//! to tonic as a stream of already negotiated connections.

#[cfg(not(any(feature = "tls-openssl", feature = "tls-rustls")))]
compile_error!("modelexpress-server needs a TLS backend: enable `tls-rustls` or `tls-openssl`");

#[cfg(feature = "tls-openssl")]
mod backend_openssl;
#[cfg(all(feature = "tls-rustls", not(feature = "tls-openssl")))]
mod backend_rustls;

#[cfg(feature = "tls-openssl")]
use crate::tls::backend_openssl as backend;
#[cfg(all(feature = "tls-rustls", not(feature = "tls-openssl")))]
use crate::tls::backend_rustls as backend;

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::server::{Connected, TcpConnectInfo};
use tracing::{debug, warn};

use crate::config::TlsConfig;
use crate::tls::backend::Acceptor;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

const ACCEPT_BACKLOG: usize = 64;

const MAX_CONCURRENT_HANDSHAKES: usize = 256;

const HANDSHAKE_FAILURE_LOG_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error(transparent)]
    Config(#[from] crate::config::TlsConfigError),
    #[error("tls config: {0}")]
    Unsupported(String),
    #[cfg(feature = "tls-openssl")]
    #[error("openssl: {0}")]
    OpenSsl(#[from] openssl::error::ErrorStack),
    #[cfg(all(feature = "tls-rustls", not(feature = "tls-openssl")))]
    #[error("rustls: {0}")]
    Rustls(#[from] rustls::Error),
    #[cfg(all(feature = "tls-rustls", not(feature = "tls-openssl")))]
    #[error("reading {path}: {source}")]
    Pem {
        path: std::path::PathBuf,
        source: rustls::pki_types::pem::Error,
    },
    #[error("bind {addr}: {source}")]
    Bind {
        addr: SocketAddr,
        source: std::io::Error,
    },
}

/// A handshake engine configured from `TlsConfig`, shared by every connection.
pub struct TlsAcceptor(Acceptor);

/// Build the acceptor from the resolved config, or `None` when TLS is off.
pub fn build_acceptor(config: &TlsConfig) -> Result<Option<TlsAcceptor>, TlsError> {
    Ok(backend::build(config)?.map(TlsAcceptor))
}

/// A negotiated server-side TLS connection tonic can serve HTTP/2 over.
pub struct TlsConn {
    stream: backend::Stream,
    info: TcpConnectInfo,
}

impl Connected for TlsConn {
    type ConnectInfo = TcpConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        self.info.clone()
    }
}

#[cfg(test)]
impl TlsConn {
    fn tcp(&self) -> &TcpStream {
        backend::tcp(&self.stream)
    }
}

impl AsyncRead for TlsConn {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsConn {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

/// Wrap `listener` in a stream of negotiated connections for
/// `serve_with_incoming_shutdown`. A failed handshake is logged and dropped.
pub fn incoming(
    listener: TcpListener,
    acceptor: TlsAcceptor,
) -> ReceiverStream<Result<TlsConn, std::io::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel(ACCEPT_BACKLOG);
    let acceptor = Arc::new(acceptor);
    let handshakes = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_HANDSHAKES));
    let failures = Arc::new(HandshakeFailures::default());
    tokio::spawn(async move {
        loop {
            let accepted = tokio::select! {
                () = tx.closed() => break,
                accepted = listener.accept() => accepted,
            };
            let (tcp, peer) = match accepted {
                Ok(accepted) => accepted,
                Err(e) => {
                    warn!("TLS listener accept failed: {e}");
                    continue;
                }
            };
            if let Err(e) = tcp.set_nodelay(true) {
                debug!("setting TCP_NODELAY for {peer} failed: {e}");
            }
            let Ok(permit) = Arc::clone(&handshakes).acquire_owned().await else {
                break;
            };
            let conn_tx = tx.clone();
            let acceptor = Arc::clone(&acceptor);
            let failures = Arc::clone(&failures);
            tokio::spawn(async move {
                let _permit = permit;
                match tokio::time::timeout(HANDSHAKE_TIMEOUT, handshake(&acceptor, tcp)).await {
                    Ok(Ok(Some(conn))) => {
                        let _ = conn_tx.send(Ok(conn)).await;
                    }
                    Ok(Ok(None)) => debug!("{peer} closed the connection before a TLS ClientHello"),
                    Ok(Err(e)) => failures.record(peer, &e),
                    Err(_) => failures.record(
                        peer,
                        &format!("timed out after {}s", HANDSHAKE_TIMEOUT.as_secs()),
                    ),
                }
            });
        }
    });
    ReceiverStream::new(rx)
}

/// Logs failed handshakes at warn, at most once per
/// `HANDSHAKE_FAILURE_LOG_INTERVAL`, with a count of the ones in between.
#[derive(Default)]
struct HandshakeFailures {
    state: Mutex<FailureWindow>,
}

#[derive(Default)]
struct FailureWindow {
    last_logged: Option<Instant>,
    suppressed: u64,
}

impl HandshakeFailures {
    fn record(&self, peer: SocketAddr, reason: &dyn std::fmt::Display) {
        match self.admit(Instant::now()) {
            Some(0) => warn!("TLS handshake with {peer} failed: {reason}"),
            Some(suppressed) => warn!(
                "TLS handshake with {peer} failed: {reason} \
                 ({suppressed} more failed since the last report)"
            ),
            None => debug!("TLS handshake with {peer} failed: {reason}"),
        }
    }

    /// The number of failures suppressed since the last report when this one
    /// should be logged, or `None` when it falls inside the interval.
    fn admit(&self, now: Instant) -> Option<u64> {
        let mut window = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match window.last_logged {
            Some(last) if now.saturating_duration_since(last) < HANDSHAKE_FAILURE_LOG_INTERVAL => {
                window.suppressed = window.suppressed.saturating_add(1);
                None
            }
            _ => {
                window.last_logged = Some(now);
                Some(std::mem::take(&mut window.suppressed))
            }
        }
    }
}

/// Negotiate TLS on `tcp`, or `None` when the peer closes or resets the
/// connection before sending anything, which is what a TCP probe or a port
/// scan does.
async fn handshake(
    acceptor: &TlsAcceptor,
    tcp: TcpStream,
) -> Result<Option<TlsConn>, Box<dyn std::error::Error + Send + Sync>> {
    if matches!(tcp.peek(&mut [0_u8; 1]).await, Ok(0) | Err(_)) {
        return Ok(None);
    }
    let info = TcpConnectInfo {
        local_addr: tcp.local_addr().ok(),
        remote_addr: tcp.peer_addr().ok(),
    };
    let stream = acceptor.0.accept(tcp).await?;
    Ok(Some(TlsConn { stream, info }))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use rcgen::{
        BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair,
        PKCS_ECDSA_P256_SHA256,
    };
    use tempfile::TempDir;
    use tokio::io::AsyncWriteExt;
    use tokio::net::{TcpListener, TcpStream};
    use tokio_stream::StreamExt;
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::transport::server::Connected;

    use crate::config::TlsConfig;
    use crate::tls::{
        HANDSHAKE_FAILURE_LOG_INTERVAL, HandshakeFailures, TlsAcceptor, TlsConn, build_acceptor,
        handshake, incoming,
    };

    struct Fixture {
        acceptor: TlsAcceptor,
        ca: PathBuf,
        _dir: TempDir,
    }

    /// An acceptor serving a `localhost` leaf signed by a fresh CA.
    fn fixture() -> Fixture {
        let dir = TempDir::new().expect("tempdir");
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("ca params");
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "incoming test CA");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).expect("ca key");
        let ca = CertifiedIssuer::self_signed(ca_params, ca_key).expect("ca cert");
        let leaf_params = CertificateParams::new(vec!["localhost".to_string()]).expect("leaf");
        let leaf_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).expect("leaf key");
        let leaf = leaf_params.signed_by(&leaf_key, &ca).expect("leaf cert");

        let ca_path = dir.path().join("ca.crt");
        let cert_path = dir.path().join("tls.crt");
        let key_path = dir.path().join("tls.key");
        std::fs::write(&ca_path, ca.pem()).expect("write ca");
        std::fs::write(&cert_path, leaf.pem()).expect("write cert");
        std::fs::write(&key_path, leaf_key.serialize_pem()).expect("write key");
        let acceptor = build_acceptor(&TlsConfig {
            cert_file: Some(cert_path),
            key_file: Some(key_path),
            ..TlsConfig::default()
        })
        .expect("build")
        .expect("enabled");
        Fixture {
            acceptor,
            ca: ca_path,
            _dir: dir,
        }
    }

    #[cfg(all(feature = "tls-rustls", not(feature = "tls-openssl")))]
    async fn tls_client(addr: SocketAddr, ca: &Path) -> Result<impl Send, String> {
        use std::sync::Arc;

        use rustls::pki_types::pem::PemObject;
        use rustls::pki_types::{CertificateDer, ServerName};

        let mut roots = rustls::RootCertStore::empty();
        for cert in CertificateDer::pem_file_iter(ca).expect("open ca") {
            roots.add(cert.expect("parse ca")).expect("add ca");
        }
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("client versions")
            .with_root_certificates(roots)
            .with_no_client_auth();
        config.alpn_protocols = vec![b"h2".to_vec()];
        let tcp = TcpStream::connect(addr).await.expect("connect");
        let name = ServerName::try_from("localhost").expect("server name");
        tokio_rustls::TlsConnector::from(Arc::new(config))
            .connect(name, tcp)
            .await
            .map_err(|e| e.to_string())
    }

    #[cfg(feature = "tls-openssl")]
    async fn tls_client(addr: SocketAddr, ca: &Path) -> Result<impl Send, String> {
        use std::pin::Pin;

        use openssl::ssl::{SslConnector, SslMethod};

        let mut builder = SslConnector::builder(SslMethod::tls_client()).expect("connector");
        builder.set_ca_file(ca).expect("ca");
        builder.set_alpn_protos(b"\x02h2").expect("alpn");
        let ssl = builder
            .build()
            .configure()
            .expect("configure")
            .into_ssl("localhost")
            .expect("ssl");
        let tcp = TcpStream::connect(addr).await.expect("connect");
        let mut stream = tokio_openssl::SslStream::new(ssl, tcp).expect("stream");
        Pin::new(&mut stream)
            .connect()
            .await
            .map_err(|e| e.to_string())?;
        Ok(stream)
    }

    async fn listen(
        acceptor: TlsAcceptor,
    ) -> (SocketAddr, ReceiverStream<std::io::Result<TlsConn>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, incoming(listener, acceptor))
    }

    async fn next_conn(conns: &mut ReceiverStream<std::io::Result<TlsConn>>) -> TlsConn {
        tokio::time::timeout(Duration::from_secs(10), conns.next())
            .await
            .expect("a connection within 10s")
            .expect("stream open")
            .expect("connection")
    }

    #[tokio::test]
    async fn incoming_yields_negotiated_connections_with_nodelay() {
        let fixture = fixture();
        let (addr, mut conns) = listen(fixture.acceptor).await;
        let _client = tls_client(addr, &fixture.ca).await.expect("handshake");

        let conn = next_conn(&mut conns).await;
        assert!(conn.tcp().nodelay().expect("nodelay"));
        let info = conn.connect_info();
        assert_eq!(info.local_addr, Some(addr));
        assert!(
            info.remote_addr
                .is_some_and(|remote| remote.ip().is_loopback() && remote.port() != addr.port())
        );
    }

    #[tokio::test]
    async fn silent_and_non_tls_peers_do_not_stop_the_listener() {
        let fixture = fixture();
        let (addr, mut conns) = listen(fixture.acceptor).await;
        drop(TcpStream::connect(addr).await.expect("probe"));
        let mut plaintext = TcpStream::connect(addr).await.expect("plaintext");
        plaintext
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .expect("write");

        let _client = tls_client(addr, &fixture.ca).await.expect("handshake");
        let conn = next_conn(&mut conns).await;
        assert!(conn.tcp().nodelay().expect("nodelay"));
    }

    #[tokio::test]
    async fn peer_closing_before_client_hello_is_not_a_handshake() {
        let fixture = fixture();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(TcpStream::connect(addr).await.expect("probe"));
        let (tcp, _) = listener.accept().await.expect("accept");
        let outcome = handshake(&fixture.acceptor, tcp).await.expect("no error");
        assert!(outcome.is_none());
    }

    #[tokio::test]
    async fn non_tls_bytes_are_a_handshake_error() {
        let fixture = fixture();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut plaintext = TcpStream::connect(addr).await.expect("connect");
        plaintext
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .expect("write");
        let (tcp, _) = listener.accept().await.expect("accept");
        assert!(handshake(&fixture.acceptor, tcp).await.is_err());
    }

    #[test]
    fn failure_logs_are_rate_limited_with_a_suppressed_count() {
        let after = |start: Instant, offset: Duration| start.checked_add(offset).expect("instant");
        let failures = HandshakeFailures::default();
        let start = Instant::now();
        let interval = HANDSHAKE_FAILURE_LOG_INTERVAL;
        let just_inside = interval
            .checked_sub(Duration::from_millis(1))
            .expect("interval");

        assert_eq!(failures.admit(start), Some(0));
        assert_eq!(failures.admit(after(start, Duration::from_secs(1))), None);
        assert_eq!(failures.admit(after(start, just_inside)), None);
        assert_eq!(failures.admit(after(start, interval)), Some(2));
        assert_eq!(
            failures.admit(after(
                start,
                interval.saturating_add(Duration::from_secs(1))
            )),
            None
        );
        assert_eq!(
            failures.admit(after(start, interval.saturating_mul(2))),
            Some(1)
        );
    }
}
