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
/// `tls_ca_file` is a PEM bundle trusted instead of the system store.
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
        builder.set_cert_store(ca_store(ca_file)?);
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
    use rustls::ClientConfig;
    use rustls::pki_types::ServerName;
    use tokio_rustls::TlsConnector;

    let transport = |message: String| Box::new(Error::Transport(message));
    let roots = match tls_ca_file {
        Some(ca_file) => ca_roots(ca_file)?,
        None => {
            let native = rustls_native_certs::load_native_certs();
            for error in &native.errors {
                tracing::debug!("skipping part of the system trust store: {error}");
            }
            system_roots(native.certs)?
        }
    };
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

#[cfg(any(feature = "tls-native", feature = "tls-rustls"))]
fn ca_bundle_error(ca_file: &Path, reason: impl std::fmt::Display) -> Box<Error> {
    Box::new(Error::Transport(format!(
        "loading TLS CA bundle {}: {reason}",
        ca_file.display()
    )))
}

/// The certificates in the PEM bundle at `ca_file`, as the only trust anchors.
#[cfg(feature = "tls-native")]
fn ca_store(ca_file: &Path) -> Result<openssl::x509::store::X509Store> {
    use openssl::x509::X509;
    use openssl::x509::store::X509StoreBuilder;

    let pem = std::fs::read(ca_file).map_err(|e| ca_bundle_error(ca_file, e))?;
    let certs = X509::stack_from_pem(&pem).map_err(|e| ca_bundle_error(ca_file, e))?;
    if certs.is_empty() {
        return Err(ca_bundle_error(ca_file, "no certificates found"));
    }
    let mut store = X509StoreBuilder::new().map_err(|e| ca_bundle_error(ca_file, e))?;
    for cert in certs {
        store
            .add_cert(cert)
            .map_err(|e| ca_bundle_error(ca_file, e))?;
    }
    Ok(store.build())
}

/// The certificates in the PEM bundle at `ca_file`, as the only trust anchors.
#[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
fn ca_roots(ca_file: &Path) -> Result<rustls::RootCertStore> {
    use rustls::pki_types::CertificateDer;
    use rustls::pki_types::pem::PemObject;

    let certs = CertificateDer::pem_file_iter(ca_file)
        .and_then(|certs| certs.collect::<std::result::Result<Vec<_>, _>>())
        .map_err(|e| ca_bundle_error(ca_file, e))?;
    if certs.is_empty() {
        return Err(ca_bundle_error(ca_file, "no certificates found"));
    }
    let mut roots = rustls::RootCertStore::empty();
    for cert in certs {
        roots.add(cert).map_err(|e| ca_bundle_error(ca_file, e))?;
    }
    Ok(roots)
}

/// The system store's certificates as trust anchors. An empty store is an
/// error: every handshake would fail with an unknown issuer.
#[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
fn system_roots(
    certs: Vec<rustls::pki_types::CertificateDer<'static>>,
) -> Result<rustls::RootCertStore> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add_parsable_certificates(certs);
    if roots.is_empty() {
        return Err(Box::new(Error::Transport(
            "no trusted CA certificates: the system trust store is empty or unreadable; \
             set MODEL_EXPRESS_TLS_CA_FILE to the CA that issued the server certificate"
                .to_string(),
        )));
    }
    Ok(roots)
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

    fn write_ca(dir: &tempfile::TempDir, count: usize) -> std::path::PathBuf {
        let pem: String = (0..count)
            .map(|_| {
                rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
                    .expect("self-signed cert")
                    .cert
                    .pem()
            })
            .collect();
        let ca_file = dir.path().join("ca.crt");
        std::fs::write(&ca_file, pem).expect("write ca");
        ca_file
    }

    #[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
    #[test]
    fn an_empty_system_store_is_an_error() {
        let Err(error) = crate::transport::system_roots(Vec::new()) else {
            panic!("expected an error for an empty trust store");
        };
        assert!(
            error.to_string().contains("no trusted CA certificates"),
            "{error}"
        );
    }

    #[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
    #[test]
    fn a_ca_file_is_the_whole_trust_store() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let roots = crate::transport::ca_roots(&write_ca(&dir, 2)).expect("CA roots");
        assert_eq!(roots.len(), 2);
    }

    #[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
    #[test]
    fn a_ca_file_without_certificates_is_an_error() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let Err(error) = crate::transport::ca_roots(&write_ca(&dir, 0)) else {
            panic!("expected an error for an empty CA file");
        };
        assert!(
            error.to_string().contains("no certificates found"),
            "{error}"
        );
    }

    #[cfg(feature = "tls-native")]
    #[test]
    fn a_ca_file_is_the_whole_openssl_store() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let store = crate::transport::ca_store(&write_ca(&dir, 2)).expect("CA store");
        assert_eq!(store.all_certificates().len(), 2);
    }

    #[cfg(feature = "tls-native")]
    #[test]
    fn a_ca_file_without_certificates_is_an_openssl_error() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let Err(error) = crate::transport::ca_store(&write_ca(&dir, 0)) else {
            panic!("expected an error for an empty CA file");
        };
        assert!(
            error.to_string().contains("no certificates found"),
            "{error}"
        );
    }

    #[test]
    fn a_missing_ca_file_names_the_path() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let missing = dir.path().join("absent.crt");
        #[cfg(feature = "tls-native")]
        let result = crate::transport::ca_store(&missing).map(drop);
        #[cfg(all(feature = "tls-rustls", not(feature = "tls-native")))]
        let result = crate::transport::ca_roots(&missing).map(drop);
        let Err(error) = result else {
            panic!("expected an error for a missing CA file");
        };
        assert!(error.to_string().contains("absent.crt"), "{error}");
    }

    #[test]
    fn ipv4_host_is_kept() {
        assert_eq!(
            parts("https://127.0.0.1:1"),
            Ok(("127.0.0.1".to_string(), 1))
        );
    }
}
