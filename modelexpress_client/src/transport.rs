// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Channel setup. An `https://` endpoint is negotiated by the build's TLS
//! backend: OpenSSL (`tls-native`) or rustls (`tls-rustls`).

use std::path::Path;

use modelexpress_common::{Error, Result};
use tonic::transport::{Channel, Endpoint};

#[cfg(any(feature = "tls-native", feature = "tls-rustls"))]
const DEFAULT_HTTPS_PORT: u16 = 443;

/// Open a channel to `endpoint`, over TLS when its scheme is `https`.
/// `tls_ca_file` is a PEM bundle trusted in addition to the system store.
pub async fn connect(endpoint: Endpoint, tls_ca_file: Option<&Path>) -> Result<Channel> {
    if endpoint.uri().scheme_str() != Some("https") {
        return Ok(endpoint.connect().await?);
    }
    connect_tls(endpoint, tls_ca_file).await
}

/// The host to dial and verify, without IPv6 brackets, and the port.
#[cfg(any(feature = "tls-native", feature = "tls-rustls"))]
fn host_and_port(uri: &http::Uri) -> std::result::Result<(&str, u16), String> {
    let host = uri
        .host()
        .ok_or_else(|| format!("endpoint {uri} has no host"))?;
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    Ok((host, uri.port_u16().unwrap_or(DEFAULT_HTTPS_PORT)))
}

#[cfg(feature = "tls-native")]
async fn connect_tls(endpoint: Endpoint, tls_ca_file: Option<&Path>) -> Result<Channel> {
    use std::pin::Pin;

    use hyper_util::rt::TokioIo;
    use openssl::ssl::{SslConnector, SslMethod};
    use tokio_openssl::SslStream;

    const ALPN_H2: &[u8] = b"\x02h2";

    let transport = |message: String| Box::new(Error::Transport(message));
    let mut builder = SslConnector::builder(SslMethod::tls_client())
        .map_err(|e| transport(format!("openssl connector: {e}")))?;
    if let Some(ca_file) = tls_ca_file {
        builder
            .set_ca_file(ca_file)
            .map_err(|e| transport(format!("loading TLS CA bundle {}: {e}", ca_file.display())))?;
    }
    builder
        .set_alpn_protos(ALPN_H2)
        .map_err(|e| transport(format!("openssl alpn: {e}")))?;
    let connector = builder.build();

    let channel = endpoint
        .connect_with_connector(tower::service_fn(move |uri: http::Uri| {
            let connector = connector.clone();
            async move {
                let (host, port) = host_and_port(&uri)?;
                let tcp = tokio::net::TcpStream::connect((host, port)).await?;
                let ssl = connector.configure()?.into_ssl(host)?;
                let mut stream = SslStream::new(ssl, tcp)?;
                Pin::new(&mut stream).connect().await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(TokioIo::new(stream))
            }
        }))
        .await?;
    Ok(channel)
}

#[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
async fn connect_tls(endpoint: Endpoint, tls_ca_file: Option<&Path>) -> Result<Channel> {
    use std::sync::Arc;

    use hyper_util::rt::TokioIo;
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, ServerName};
    use rustls::{ClientConfig, RootCertStore};
    use tokio_rustls::TlsConnector;

    let transport = |message: String| Box::new(Error::Transport(message));
    let mut roots = RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    for error in &native.errors {
        tracing::debug!("skipping part of the system trust store: {error}");
    }
    roots.add_parsable_certificates(native.certs);
    if let Some(ca_file) = tls_ca_file {
        let loading =
            |e: String| transport(format!("loading TLS CA bundle {}: {e}", ca_file.display()));
        let certs = CertificateDer::pem_file_iter(ca_file)
            .and_then(|certs| certs.collect::<std::result::Result<Vec<_>, _>>())
            .map_err(|e| loading(e.to_string()))?;
        if certs.is_empty() {
            return Err(loading("no certificates found".to_string()));
        }
        for cert in certs {
            roots.add(cert).map_err(|e| loading(e.to_string()))?;
        }
    }
    let mut config =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| transport(format!("rustls client config: {e}")))?
            .with_root_certificates(roots)
            .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec()];
    let connector = TlsConnector::from(Arc::new(config));

    let channel = endpoint
        .connect_with_connector(tower::service_fn(move |uri: http::Uri| {
            let connector = connector.clone();
            async move {
                let (host, port) = host_and_port(&uri)?;
                let name = ServerName::try_from(host.to_string())?;
                let tcp = tokio::net::TcpStream::connect((host, port)).await?;
                let stream = connector.connect(name, tcp).await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(TokioIo::new(stream))
            }
        }))
        .await?;
    Ok(channel)
}

#[cfg(not(any(feature = "tls-native", feature = "tls-rustls")))]
async fn connect_tls(endpoint: Endpoint, _tls_ca_file: Option<&Path>) -> Result<Channel> {
    Err(Box::new(Error::Transport(format!(
        "{} is an https endpoint, but this client was built without a TLS backend",
        endpoint.uri()
    ))))
}

#[cfg(all(test, any(feature = "tls-native", feature = "tls-rustls")))]
#[allow(clippy::expect_used)]
mod tests {
    use crate::transport::host_and_port;

    fn parts(uri: &str) -> Result<(String, u16), String> {
        let uri: http::Uri = uri.parse().expect("uri");
        host_and_port(&uri).map(|(host, port)| (host.to_string(), port))
    }

    #[test]
    fn dns_host_with_explicit_port() {
        assert_eq!(
            parts("https://mx.ns.svc.cluster.local:8001"),
            Ok(("mx.ns.svc.cluster.local".to_string(), 8001))
        );
    }

    #[test]
    fn missing_port_defaults_to_443() {
        assert_eq!(
            parts("https://example.com"),
            Ok(("example.com".to_string(), 443))
        );
    }

    #[test]
    fn ipv6_brackets_are_stripped() {
        assert_eq!(parts("https://[::1]:8001"), Ok(("::1".to_string(), 8001)));
        assert_eq!(parts("https://[fd00::1]"), Ok(("fd00::1".to_string(), 443)));
    }

    #[test]
    fn ipv4_host_is_kept() {
        assert_eq!(
            parts("https://127.0.0.1:1"),
            Ok(("127.0.0.1".to_string(), 1))
        );
    }
}
