// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! rustls handshakes for the gRPC listener, with the ring provider. Settings
//! arrive in OpenSSL spelling; names rustls does not implement are dropped
//! with a warning.

use std::sync::Arc;

use modelexpress_common::tls::{TlsVersion, split_cipher_suites};
use rustls::crypto::{CryptoProvider, SupportedKxGroup, ring};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{
    CipherSuite, NamedGroup, ServerConfig, SupportedCipherSuite, SupportedProtocolVersion,
};
use tokio::net::TcpStream;
use tracing::warn;

use crate::config::TlsConfig;
use crate::tls::TlsError;

pub type Stream = tokio_rustls::server::TlsStream<TcpStream>;

static TLS13_ONLY: &[&SupportedProtocolVersion] = &[&rustls::version::TLS13];

pub struct Acceptor {
    inner: tokio_rustls::TlsAcceptor,
}

impl Acceptor {
    pub async fn accept(
        &self,
        tcp: TcpStream,
    ) -> Result<Stream, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.inner.accept(tcp).await?)
    }
}

/// OpenSSL cipher names rustls implements, with the suite each one names.
const CIPHERS: [(&str, CipherSuite); 9] = [
    (
        "TLS_AES_128_GCM_SHA256",
        CipherSuite::TLS13_AES_128_GCM_SHA256,
    ),
    (
        "TLS_AES_256_GCM_SHA384",
        CipherSuite::TLS13_AES_256_GCM_SHA384,
    ),
    (
        "TLS_CHACHA20_POLY1305_SHA256",
        CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
    ),
    (
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    ),
    (
        "ECDHE-RSA-AES128-GCM-SHA256",
        CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
    ),
    (
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    ),
    (
        "ECDHE-RSA-AES256-GCM-SHA384",
        CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    ),
    (
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
    ),
    (
        "ECDHE-RSA-CHACHA20-POLY1305",
        CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
    ),
];

/// OpenSSL group names and aliases ring implements. OpenSSL matches these
/// case-insensitively.
const GROUPS: [(&str, NamedGroup); 6] = [
    ("X25519", NamedGroup::X25519),
    ("secp256r1", NamedGroup::secp256r1),
    ("prime256v1", NamedGroup::secp256r1),
    ("P-256", NamedGroup::secp256r1),
    ("secp384r1", NamedGroup::secp384r1),
    ("P-384", NamedGroup::secp384r1),
];

/// Build the rustls acceptor from the resolved config, or `None` when TLS is off.
pub fn build(config: &TlsConfig) -> Result<Option<Acceptor>, TlsError> {
    let Some((cert, key)) = config.key_pair()? else {
        return Ok(None);
    };
    let chain = CertificateDer::pem_file_iter(cert)
        .and_then(|certs| certs.collect::<Result<Vec<_>, _>>())
        .map_err(|source| TlsError::Pem {
            path: cert.clone(),
            source,
        })?;
    let key = PrivateKeyDer::from_pem_file(key).map_err(|source| TlsError::Pem {
        path: key.clone(),
        source,
    })?;
    let mut server = ServerConfig::builder_with_provider(Arc::new(provider(config)?))
        .with_protocol_versions(protocol_versions(config.min_version))?
        .with_no_client_auth()
        .with_single_cert(chain, key)?;
    server.alpn_protocols = vec![b"h2".to_vec()];
    Ok(Some(Acceptor {
        inner: tokio_rustls::TlsAcceptor::from(Arc::new(server)),
    }))
}

/// ring's provider narrowed to the configured ciphers and groups. The TLS 1.2
/// and 1.3 lists are independent, as in OpenSSL, and the TLS 1.2 list is only
/// read when TLS 1.2 can be negotiated.
fn provider(config: &TlsConfig) -> Result<CryptoProvider, TlsError> {
    let base = ring::default_provider();
    let (tls12, tls13) = split_cipher_suites(&config.cipher_suites);
    let mut cipher_suites = select_cipher_suites(&base, &tls13, &rustls::version::TLS13)?;
    if config.min_version != Some(TlsVersion::Tls13) {
        cipher_suites.extend(select_cipher_suites(
            &base,
            &tls12,
            &rustls::version::TLS12,
        )?);
    }
    let kx_groups = select_kx_groups(&base, &config.groups)?;
    Ok(CryptoProvider {
        cipher_suites,
        kx_groups,
        ..base
    })
}

/// The provider's suites for `version`, in the order `names` gives them, or
/// all of them when `names` is empty. A list naming none of them is an error.
fn select_cipher_suites(
    provider: &CryptoProvider,
    names: &[String],
    version: &SupportedProtocolVersion,
) -> Result<Vec<SupportedCipherSuite>, TlsError> {
    let available: Vec<SupportedCipherSuite> = provider
        .cipher_suites
        .iter()
        .copied()
        .filter(|suite| suite.version() == version)
        .collect();
    if names.is_empty() {
        return Ok(available);
    }
    let mut selected = Vec::with_capacity(names.len());
    for name in names {
        let suite = CIPHERS
            .iter()
            .find(|(openssl, _)| openssl == name)
            .and_then(|(_, id)| available.iter().find(|suite| suite.suite() == *id));
        match suite {
            Some(suite) if !selected.contains(suite) => selected.push(*suite),
            Some(_) => {}
            None => warn!("TLS cipher {name} is not supported by rustls; dropping it"),
        }
    }
    if selected.is_empty() {
        return Err(TlsError::Unsupported(format!(
            "none of the {:?} ciphers {} are supported by rustls",
            version.version,
            names.join(":")
        )));
    }
    Ok(selected)
}

/// The provider's groups named in `names`, in order, or all of them when
/// `names` is empty. Unknown names are dropped. A list naming none of them is
/// an error, as for ciphers: falling back to the defaults would widen the key
/// exchange policy the list asked for.
fn select_kx_groups(
    provider: &CryptoProvider,
    names: &[String],
) -> Result<Vec<&'static dyn SupportedKxGroup>, TlsError> {
    let names: Vec<&str> = names
        .iter()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .collect();
    if names.is_empty() {
        return Ok(provider.kx_groups.clone());
    }
    let mut selected: Vec<&'static dyn SupportedKxGroup> = Vec::with_capacity(names.len());
    for name in names.iter().copied() {
        let group = GROUPS
            .iter()
            .find(|(openssl, _)| openssl.eq_ignore_ascii_case(name))
            .and_then(|(_, id)| {
                provider
                    .kx_groups
                    .iter()
                    .copied()
                    .find(|group| group.name() == *id)
            });
        match group {
            Some(group) if !selected.iter().any(|seen| seen.name() == group.name()) => {
                selected.push(group);
            }
            Some(_) => {}
            None => warn!("TLS group {name} is not supported by rustls; dropping it"),
        }
    }
    if selected.is_empty() {
        return Err(TlsError::Unsupported(format!(
            "none of the groups {} are supported by rustls",
            names.join(":")
        )));
    }
    Ok(selected)
}

fn protocol_versions(min: Option<TlsVersion>) -> &'static [&'static SupportedProtocolVersion] {
    match min {
        Some(TlsVersion::Tls13) => TLS13_ONLY,
        Some(version @ (TlsVersion::Tls10 | TlsVersion::Tls11)) => {
            warn!(
                "rustls implements TLS1.2 and TLS1.3 only; minimum {version} is raised to TLS1.2"
            );
            rustls::ALL_VERSIONS
        }
        Some(TlsVersion::Tls12) | None => rustls::ALL_VERSIONS,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use modelexpress_common::tls::TlsVersion;
    use rcgen::{
        BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair,
        PKCS_ECDSA_P256_SHA256,
    };
    use rustls::crypto::{CryptoProvider, ring};
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, ServerName};
    use rustls::{
        CipherSuite, ClientConfig, NamedGroup, ProtocolVersion, RootCertStore,
        SupportedProtocolVersion,
    };
    use tempfile::TempDir;
    use tokio::net::{TcpListener, TcpStream};
    use tokio_rustls::TlsConnector;

    use crate::config::{TlsConfig, TlsConfigError};
    use crate::tls::TlsError;
    use crate::tls::backend_rustls::{Acceptor, build, select_kx_groups};

    const INTERMEDIATE_CIPHERS: [&str; 9] = [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305",
    ];

    static TLS12: &[&SupportedProtocolVersion] = &[&rustls::version::TLS12];
    static TLS13: &[&SupportedProtocolVersion] = &[&rustls::version::TLS13];

    const PROFILE_GROUPS: [&str; 4] = ["X25519MLKEM768", "X25519", "secp256r1", "secp384r1"];

    struct Chain {
        ca: PathBuf,
        cert: PathBuf,
        key: PathBuf,
    }

    /// A fresh ECDSA P-256 CA and the `localhost` leaf it signed.
    fn chain(dir: &TempDir, stem: &str) -> Chain {
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("ca params");
        ca_params
            .distinguished_name
            .push(DnType::CommonName, format!("{stem} CA"));
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).expect("ca key");
        let ca = CertifiedIssuer::self_signed(ca_params, ca_key).expect("ca cert");

        let mut leaf_params = CertificateParams::new(vec!["localhost".to_string()]).expect("leaf");
        leaf_params
            .distinguished_name
            .push(DnType::CommonName, stem);
        let leaf_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).expect("leaf key");
        let leaf = leaf_params.signed_by(&leaf_key, &ca).expect("leaf cert");

        let chain = Chain {
            ca: dir.path().join(format!("{stem}-ca.crt")),
            cert: dir.path().join(format!("{stem}.crt")),
            key: dir.path().join(format!("{stem}.key")),
        };
        std::fs::write(&chain.ca, ca.pem()).expect("write ca");
        std::fs::write(&chain.cert, leaf.pem()).expect("write cert");
        std::fs::write(&chain.key, leaf_key.serialize_pem()).expect("write key");
        chain
    }

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    /// What a test client offers. Empty lists mean ring's defaults.
    struct Offer {
        versions: &'static [&'static SupportedProtocolVersion],
        suites: Vec<CipherSuite>,
        groups: Vec<NamedGroup>,
        alpn: Vec<Vec<u8>>,
    }

    impl Default for Offer {
        fn default() -> Self {
            Self {
                versions: rustls::ALL_VERSIONS,
                suites: Vec::new(),
                groups: Vec::new(),
                alpn: vec![b"h2".to_vec()],
            }
        }
    }

    #[derive(Debug)]
    struct Negotiated {
        version: ProtocolVersion,
        suite: CipherSuite,
        group: Option<NamedGroup>,
        alpn: Option<Vec<u8>>,
    }

    struct Server {
        acceptor: Acceptor,
        ca: PathBuf,
        _dir: TempDir,
    }

    fn server(adjust: impl FnOnce(&mut TlsConfig)) -> Server {
        let dir = TempDir::new().expect("tempdir");
        let chain = chain(&dir, "server");
        let mut config = TlsConfig {
            cert_file: Some(chain.cert),
            key_file: Some(chain.key),
            ..TlsConfig::default()
        };
        adjust(&mut config);
        let acceptor = build(&config).expect("build").expect("enabled");
        Server {
            acceptor,
            ca: chain.ca,
            _dir: dir,
        }
    }

    fn client_config(ca: &Path, offer: Offer) -> ClientConfig {
        let base = ring::default_provider();
        let provider = CryptoProvider {
            cipher_suites: base
                .cipher_suites
                .iter()
                .copied()
                .filter(|suite| offer.suites.is_empty() || offer.suites.contains(&suite.suite()))
                .collect(),
            kx_groups: base
                .kx_groups
                .iter()
                .copied()
                .filter(|group| offer.groups.is_empty() || offer.groups.contains(&group.name()))
                .collect(),
            ..base
        };
        let mut roots = RootCertStore::empty();
        for cert in CertificateDer::pem_file_iter(ca).expect("open ca") {
            roots.add(cert.expect("parse ca")).expect("add ca");
        }
        let mut config = ClientConfig::builder_with_provider(Arc::new(provider))
            .with_protocol_versions(offer.versions)
            .expect("client versions")
            .with_root_certificates(roots)
            .with_no_client_auth();
        config.alpn_protocols = offer.alpn;
        config
    }

    /// One handshake over loopback between `server` and a rustls client that
    /// trusts the server's CA and offers `offer`.
    async fn handshake(server: Server, offer: Offer) -> Result<Negotiated, String> {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let config = client_config(&server.ca, offer);
        let acceptor = server.acceptor;
        let accept = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.expect("accept");
            acceptor
                .accept(tcp)
                .await
                .map(|stream| stream.get_ref().1.alpn_protocol().map(<[u8]>::to_vec))
                .map_err(|e| e.to_string())
        });

        let tcp = TcpStream::connect(addr).await.expect("connect");
        let name = ServerName::try_from("localhost").expect("server name");
        let connected = TlsConnector::from(Arc::new(config))
            .connect(name, tcp)
            .await;
        let server_side = tokio::time::timeout(Duration::from_secs(10), accept)
            .await
            .expect("server handshake finished")
            .expect("server task");
        let stream = connected.map_err(|e| format!("client: {e}"))?;
        let alpn = server_side.map_err(|e| format!("server: {e}"))?;
        let conn = stream.get_ref().1;
        Ok(Negotiated {
            version: conn.protocol_version().expect("version"),
            suite: conn.negotiated_cipher_suite().expect("suite").suite(),
            group: conn
                .negotiated_key_exchange_group()
                .map(|group| group.name()),
            alpn,
        })
    }

    fn tls12_only(suites: &[CipherSuite]) -> Offer {
        Offer {
            versions: TLS12,
            suites: suites.to_vec(),
            ..Offer::default()
        }
    }

    #[test]
    fn disabled_config_builds_no_acceptor() {
        assert!(build(&TlsConfig::default()).expect("build").is_none());
    }

    #[test]
    fn half_configured_key_pair_is_a_config_error() {
        let config = TlsConfig {
            key_file: Some(PathBuf::from("/nonexistent/tls.key")),
            ..TlsConfig::default()
        };
        assert!(matches!(
            build(&config),
            Err(TlsError::Config(TlsConfigError::KeyWithoutCert))
        ));
    }

    #[test]
    fn missing_files_are_pem_errors_naming_the_path() {
        let config = TlsConfig {
            cert_file: Some(PathBuf::from("/nonexistent/tls.crt")),
            key_file: Some(PathBuf::from("/nonexistent/tls.key")),
            ..TlsConfig::default()
        };
        match build(&config) {
            Err(TlsError::Pem { path, .. }) => {
                assert_eq!(path, PathBuf::from("/nonexistent/tls.crt"))
            }
            other => panic!("expected a PEM error, got {:?}", other.map(|a| a.is_some())),
        }
    }

    #[test]
    fn mismatched_key_is_rejected() {
        let dir = TempDir::new().expect("tempdir");
        let a = chain(&dir, "a");
        let b = chain(&dir, "b");
        let config = TlsConfig {
            cert_file: Some(a.cert),
            key_file: Some(b.key),
            ..TlsConfig::default()
        };
        assert!(matches!(build(&config), Err(TlsError::Rustls(_))));
    }

    #[test]
    fn cert_file_without_certificates_is_rejected() {
        let dir = TempDir::new().expect("tempdir");
        let pair = chain(&dir, "server");
        let empty = dir.path().join("empty.crt");
        std::fs::write(&empty, "").expect("write empty cert");
        let config = TlsConfig {
            cert_file: Some(empty),
            key_file: Some(pair.key),
            ..TlsConfig::default()
        };
        assert!(build(&config).is_err());
    }

    #[test]
    fn key_file_without_a_key_is_a_pem_error() {
        let dir = TempDir::new().expect("tempdir");
        let pair = chain(&dir, "server");
        let config = TlsConfig {
            cert_file: Some(pair.cert.clone()),
            key_file: Some(pair.cert),
            ..TlsConfig::default()
        };
        assert!(matches!(build(&config), Err(TlsError::Pem { .. })));
    }

    #[tokio::test]
    async fn negotiates_h2_over_tls13_by_default() {
        let negotiated = handshake(server(|_| {}), Offer::default())
            .await
            .expect("handshake");
        assert_eq!(negotiated.version, ProtocolVersion::TLSv1_3);
        assert_eq!(negotiated.alpn.as_deref(), Some(&b"h2"[..]));
    }

    #[tokio::test]
    async fn client_without_alpn_still_connects() {
        let negotiated = handshake(
            server(|_| {}),
            Offer {
                alpn: Vec::new(),
                ..Offer::default()
            },
        )
        .await
        .expect("handshake");
        assert_eq!(negotiated.alpn, None);
    }

    #[tokio::test]
    async fn client_offering_only_http1_is_refused() {
        let outcome = handshake(
            server(|_| {}),
            Offer {
                alpn: vec![b"http/1.1".to_vec()],
                ..Offer::default()
            },
        )
        .await;
        assert!(outcome.is_err(), "{outcome:?}");
    }

    #[tokio::test]
    async fn client_trusting_another_ca_is_refused() {
        let dir = TempDir::new().expect("tempdir");
        let other = chain(&dir, "other");
        let mut server = server(|_| {});
        server.ca = other.ca;
        assert!(handshake(server, Offer::default()).await.is_err());
    }

    #[tokio::test]
    async fn min_version_tls13_rejects_tls12_clients() {
        let min13 = |config: &mut TlsConfig| config.min_version = Some(TlsVersion::Tls13);
        assert!(handshake(server(min13), tls12_only(&[])).await.is_err());
        let negotiated = handshake(server(min13), Offer::default())
            .await
            .expect("tls13 client");
        assert_eq!(negotiated.version, ProtocolVersion::TLSv1_3);
    }

    #[tokio::test]
    async fn min_version_tls12_accepts_tls12_clients() {
        let negotiated = handshake(
            server(|config| config.min_version = Some(TlsVersion::Tls12)),
            tls12_only(&[]),
        )
        .await
        .expect("tls12 client");
        assert_eq!(negotiated.version, ProtocolVersion::TLSv1_2);
    }

    #[tokio::test]
    async fn legacy_minimums_are_raised_to_tls12() {
        for version in [TlsVersion::Tls10, TlsVersion::Tls11] {
            let negotiated = handshake(
                server(|config| config.min_version = Some(version)),
                tls12_only(&[]),
            )
            .await
            .expect("tls12 client");
            assert_eq!(negotiated.version, ProtocolVersion::TLSv1_2, "{version}");
        }
    }

    #[tokio::test]
    async fn cipher_suites_restrict_tls12_negotiation() {
        let restrict = |config: &mut TlsConfig| {
            config.cipher_suites = strings(&["ECDHE-ECDSA-AES256-GCM-SHA384"]);
        };
        let negotiated = handshake(
            server(restrict),
            tls12_only(&[CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384]),
        )
        .await
        .expect("listed suite");
        assert_eq!(
            negotiated.suite,
            CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
        );
        assert!(
            handshake(
                server(restrict),
                tls12_only(&[CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256]),
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn cipher_suites_restrict_tls13_negotiation() {
        let restrict = |config: &mut TlsConfig| {
            config.cipher_suites = strings(&["TLS_CHACHA20_POLY1305_SHA256"]);
        };
        let offer = |suite| Offer {
            versions: TLS13,
            suites: vec![suite],
            ..Offer::default()
        };
        let negotiated = handshake(
            server(restrict),
            offer(CipherSuite::TLS13_CHACHA20_POLY1305_SHA256),
        )
        .await
        .expect("listed suite");
        assert_eq!(
            negotiated.suite,
            CipherSuite::TLS13_CHACHA20_POLY1305_SHA256
        );
        assert!(
            handshake(
                server(restrict),
                offer(CipherSuite::TLS13_AES_128_GCM_SHA256)
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn tls12_cipher_list_leaves_tls13_suites_enabled() {
        let restrict = |config: &mut TlsConfig| {
            config.cipher_suites = strings(&["ECDHE-ECDSA-AES256-GCM-SHA384"]);
        };
        let negotiated = handshake(
            server(restrict),
            Offer {
                versions: TLS13,
                suites: vec![CipherSuite::TLS13_AES_128_GCM_SHA256],
                ..Offer::default()
            },
        )
        .await
        .expect("tls13 default suite");
        assert_eq!(negotiated.suite, CipherSuite::TLS13_AES_128_GCM_SHA256);
    }

    #[tokio::test]
    async fn unsupported_ciphers_are_dropped_not_fatal() {
        let restrict = |config: &mut TlsConfig| {
            config.min_version = Some(TlsVersion::Tls10);
            config.cipher_suites = strings(&[
                "AES128-SHA",
                "DES-CBC3-SHA",
                "ECDHE-ECDSA-AES256-GCM-SHA384",
            ]);
        };
        assert!(
            handshake(
                server(restrict),
                tls12_only(&[CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384]),
            )
            .await
            .is_ok()
        );
        assert!(
            handshake(
                server(restrict),
                tls12_only(&[CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256]),
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn a_cipher_list_rustls_cannot_honor_is_rejected_at_build() {
        for names in [
            vec!["NOT-A-CIPHER"],
            vec!["TLS_NOT_A_SUITE"],
            vec!["AES128-SHA", "DES-CBC3-SHA"],
        ] {
            let dir = TempDir::new().expect("tempdir");
            let pair = chain(&dir, "server");
            let config = TlsConfig {
                cert_file: Some(pair.cert),
                key_file: Some(pair.key),
                cipher_suites: strings(&names),
                ..TlsConfig::default()
            };
            assert!(
                matches!(build(&config), Err(TlsError::Unsupported(_))),
                "{names:?}"
            );
        }
    }

    #[tokio::test]
    async fn groups_restrict_key_exchange() {
        let x25519 = |config: &mut TlsConfig| config.groups = strings(&["X25519"]);
        let negotiated = handshake(
            server(x25519),
            Offer {
                groups: vec![NamedGroup::X25519],
                ..Offer::default()
            },
        )
        .await
        .expect("listed group");
        assert_eq!(negotiated.group, Some(NamedGroup::X25519));
        assert!(
            handshake(
                server(x25519),
                Offer {
                    groups: vec![NamedGroup::secp384r1],
                    ..Offer::default()
                },
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn profile_group_list_drops_post_quantum_under_ring() {
        let names: Vec<NamedGroup> =
            select_kx_groups(&ring::default_provider(), &strings(&PROFILE_GROUPS))
                .expect("groups")
                .iter()
                .map(|group| group.name())
                .collect();
        assert_eq!(
            names,
            [
                NamedGroup::X25519,
                NamedGroup::secp256r1,
                NamedGroup::secp384r1
            ]
        );
    }

    #[test]
    fn openssl_group_aliases_resolve_and_dedupe() {
        let names: Vec<NamedGroup> = select_kx_groups(
            &ring::default_provider(),
            &strings(&["P-256", " prime256v1 ", "x25519", "", "P-384"]),
        )
        .expect("groups")
        .iter()
        .map(|group| group.name())
        .collect();
        assert_eq!(
            names,
            [
                NamedGroup::secp256r1,
                NamedGroup::X25519,
                NamedGroup::secp384r1
            ]
        );
    }

    #[test]
    fn all_unknown_groups_are_a_config_error() {
        let result = select_kx_groups(
            &ring::default_provider(),
            &strings(&["NOT_A_GROUP", "X25519MLKEM768"]),
        );
        let Err(TlsError::Unsupported(message)) = result else {
            panic!("expected an unsupported-groups error");
        };
        assert!(message.contains("NOT_A_GROUP:X25519MLKEM768"), "{message}");

        let dir = TempDir::new().expect("tempdir");
        let pair = chain(&dir, "server");
        let config = TlsConfig {
            cert_file: Some(pair.cert),
            key_file: Some(pair.key),
            groups: strings(&["NOT_A_GROUP"]),
            ..TlsConfig::default()
        };
        assert!(matches!(build(&config), Err(TlsError::Unsupported(_))));
    }

    #[tokio::test]
    async fn blank_group_names_leave_provider_defaults() {
        let negotiated = handshake(
            server(|config| config.groups = strings(&["", "  "])),
            Offer {
                groups: vec![NamedGroup::secp384r1],
                ..Offer::default()
            },
        )
        .await
        .expect("default groups");
        assert_eq!(negotiated.group, Some(NamedGroup::secp384r1));
    }

    #[tokio::test]
    async fn tls12_ciphers_are_not_read_when_only_tls13_is_served() {
        let tls13_only = |config: &mut TlsConfig| {
            config.min_version = Some(TlsVersion::Tls13);
            config.cipher_suites = strings(&["DHE-RSA-AES128-GCM-SHA256"]);
        };
        let negotiated = handshake(server(tls13_only), Offer::default())
            .await
            .expect("tls13 client");
        assert_eq!(negotiated.version, ProtocolVersion::TLSv1_3);

        let dir = TempDir::new().expect("tempdir");
        let pair = chain(&dir, "server");
        let tls12_served = TlsConfig {
            cert_file: Some(pair.cert),
            key_file: Some(pair.key),
            min_version: Some(TlsVersion::Tls12),
            cipher_suites: strings(&["DHE-RSA-AES128-GCM-SHA256"]),
            ..TlsConfig::default()
        };
        assert!(matches!(
            build(&tls12_served),
            Err(TlsError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn intermediate_profile_serves_tls12_and_tls13() {
        let intermediate = |config: &mut TlsConfig| {
            config.min_version = Some(TlsVersion::Tls12);
            config.cipher_suites = strings(&INTERMEDIATE_CIPHERS);
            config.groups = strings(&PROFILE_GROUPS);
        };
        let tls12 = handshake(server(intermediate), tls12_only(&[]))
            .await
            .expect("tls12 under Intermediate");
        assert_eq!(tls12.version, ProtocolVersion::TLSv1_2);
        let tls13 = handshake(server(intermediate), Offer::default())
            .await
            .expect("tls13 under Intermediate");
        assert_eq!(tls13.version, ProtocolVersion::TLSv1_3);
        assert_eq!(tls13.group, Some(NamedGroup::X25519));
    }

    #[tokio::test]
    async fn modern_profile_refuses_tls12() {
        let modern = |config: &mut TlsConfig| {
            config.min_version = Some(TlsVersion::Tls13);
            config.cipher_suites = strings(&INTERMEDIATE_CIPHERS[..3]);
            config.groups = strings(&PROFILE_GROUPS);
        };
        assert!(handshake(server(modern), tls12_only(&[])).await.is_err());
        assert!(handshake(server(modern), Offer::default()).await.is_ok());
    }
}
