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
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

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

/// Bind `addr` and return a stream of negotiated connections for
/// `serve_with_incoming_shutdown`. A failed handshake is logged and dropped.
pub async fn incoming(
    addr: SocketAddr,
    acceptor: TlsAcceptor,
) -> Result<ReceiverStream<Result<TlsConn, std::io::Error>>, TlsError> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|source| TlsError::Bind { addr, source })?;
    let (tx, rx) = tokio::sync::mpsc::channel(ACCEPT_BACKLOG);
    let acceptor = Arc::new(acceptor);
    let handshakes = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_HANDSHAKES));
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
            let Ok(permit) = Arc::clone(&handshakes).acquire_owned().await else {
                break;
            };
            let conn_tx = tx.clone();
            let acceptor = Arc::clone(&acceptor);
            tokio::spawn(async move {
                let _permit = permit;
                match tokio::time::timeout(HANDSHAKE_TIMEOUT, handshake(&acceptor, tcp)).await {
                    Ok(Ok(conn)) => {
                        let _ = conn_tx.send(Ok(conn)).await;
                    }
                    Ok(Err(e)) => debug!("TLS handshake with {peer} failed: {e}"),
                    Err(_) => debug!(
                        "TLS handshake with {peer} timed out after {}s",
                        HANDSHAKE_TIMEOUT.as_secs()
                    ),
                }
            });
        }
    });
    Ok(ReceiverStream::new(rx))
}

async fn handshake(
    acceptor: &TlsAcceptor,
    tcp: TcpStream,
) -> Result<TlsConn, Box<dyn std::error::Error + Send + Sync>> {
    let info = TcpConnectInfo {
        local_addr: tcp.local_addr().ok(),
        remote_addr: tcp.peer_addr().ok(),
    };
    let stream = acceptor.0.accept(tcp).await?;
    Ok(TlsConn { stream, info })
}
