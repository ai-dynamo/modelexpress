// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenSSL handshakes for the gRPC listener.

use std::pin::Pin;

use modelexpress_common::tls::{TlsVersion, split_cipher_suites};
use openssl::error::ErrorStack;
use openssl::ssl::{
    AlpnError, Ssl, SslContext, SslContextBuilder, SslFiletype, SslMethod, SslVersion,
    select_next_proto,
};
use tokio::net::TcpStream;
use tokio_openssl::SslStream;
use tracing::warn;

use crate::config::TlsConfig;
use crate::tls::TlsError;

/// ALPN protocol list in OpenSSL wire format: one length-prefixed entry.
const ALPN_H2: &[u8] = b"\x02h2";

pub type Stream = SslStream<TcpStream>;

pub struct Acceptor {
    context: SslContext,
}

impl Acceptor {
    pub async fn accept(
        &self,
        tcp: TcpStream,
    ) -> Result<Stream, Box<dyn std::error::Error + Send + Sync>> {
        let ssl = Ssl::new(&self.context)?;
        let mut stream = SslStream::new(ssl, tcp)?;
        Pin::new(&mut stream).accept().await?;
        Ok(stream)
    }
}

/// Build the OpenSSL acceptor from the resolved config, or `None` when TLS is off.
pub fn build(config: &TlsConfig) -> Result<Option<Acceptor>, TlsError> {
    let Some((cert, key)) = config.key_pair()? else {
        return Ok(None);
    };
    let mut builder = SslContextBuilder::new(SslMethod::tls_server())?;
    builder.set_certificate_chain_file(cert)?;
    builder.set_private_key_file(key, SslFiletype::PEM)?;
    builder.check_private_key()?;
    builder.set_min_proto_version(Some(ssl_version(min_version(config.min_version))))?;
    let (tls12, tls13) = split_cipher_suites(&config.cipher_suites);
    if !tls12.is_empty() {
        builder.set_cipher_list(&tls12.join(":"))?;
    }
    if !tls13.is_empty() {
        builder.set_ciphersuites(&tls13.join(":"))?;
    }
    let groups = supported_groups(&config.groups)?;
    if !groups.is_empty() {
        builder.set_groups_list(&groups.join(":"))?;
    }
    builder.set_alpn_select_callback(|_ssl, client| {
        select_next_proto(ALPN_H2, client).ok_or(AlpnError::ALERT_FATAL)
    });
    Ok(Some(Acceptor {
        context: builder.build(),
    }))
}

/// The configured minimum, floored at TLS 1.2. rustls cannot go below it, and
/// an unset minimum otherwise leaves the OpenSSL default in charge.
fn min_version(configured: Option<TlsVersion>) -> TlsVersion {
    match configured {
        Some(version) if version >= TlsVersion::Tls12 => version,
        Some(version) => {
            warn!("TLS minimum {version} is below TLS1.2, using TLS1.2");
            TlsVersion::Tls12
        }
        None => TlsVersion::Tls12,
    }
}

/// The subset of `groups` this OpenSSL can negotiate, in the order given.
/// Each name is probed separately, since `set_groups_list` rejects a whole
/// list containing one unknown name.
fn supported_groups(groups: &[String]) -> Result<Vec<String>, ErrorStack> {
    let mut probe = SslContextBuilder::new(SslMethod::tls_server())?;
    let mut supported = Vec::with_capacity(groups.len());
    for group in groups.iter().map(|g| g.trim()).filter(|g| !g.is_empty()) {
        if probe.set_groups_list(group).is_ok() {
            supported.push(group.to_string());
        } else {
            warn!("TLS group {group} is not supported by the linked OpenSSL; dropping it");
        }
    }
    Ok(supported)
}

fn ssl_version(version: TlsVersion) -> SslVersion {
    match version {
        TlsVersion::Tls10 => SslVersion::TLS1,
        TlsVersion::Tls11 => SslVersion::TLS1_1,
        TlsVersion::Tls12 => SslVersion::TLS1_2,
        TlsVersion::Tls13 => SslVersion::TLS1_3,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use modelexpress_common::tls::TlsVersion;
    use openssl::ssl::{SslConnector, SslMethod, SslVerifyMode, SslVersion};
    use tempfile::TempDir;

    use crate::config::{TlsConfig, TlsConfigError};
    use crate::tls::TlsError;
    use crate::tls::backend_openssl::{ALPN_H2, build, min_version};

    /// A throwaway CA-signed leaf for `localhost` written into `dir`.
    pub(crate) fn self_signed(dir: &TempDir) -> (PathBuf, PathBuf) {
        use openssl::asn1::Asn1Time;
        use openssl::hash::MessageDigest;
        use openssl::pkey::PKey;
        use openssl::rsa::Rsa;
        use openssl::x509::extension::SubjectAlternativeName;
        use openssl::x509::{X509, X509NameBuilder};

        let rsa = Rsa::generate(2048).expect("rsa");
        let key = PKey::from_rsa(rsa).expect("pkey");
        let mut name = X509NameBuilder::new().expect("name");
        name.append_entry_by_text("CN", "localhost").expect("cn");
        let name = name.build();
        let mut builder = X509::builder().expect("x509");
        builder.set_version(2).expect("version");
        builder.set_subject_name(&name).expect("subject");
        builder.set_issuer_name(&name).expect("issuer");
        builder.set_pubkey(&key).expect("pubkey");
        builder
            .set_not_before(&Asn1Time::days_from_now(0).expect("now"))
            .expect("not before");
        builder
            .set_not_after(&Asn1Time::days_from_now(1).expect("tomorrow"))
            .expect("not after");
        let san = SubjectAlternativeName::new()
            .dns("localhost")
            .ip("127.0.0.1")
            .build(&builder.x509v3_context(None, None))
            .expect("san");
        builder.append_extension(san).expect("append san");
        builder.sign(&key, MessageDigest::sha256()).expect("sign");
        let cert = builder.build();

        let cert_path = dir.path().join("tls.crt");
        let key_path = dir.path().join("tls.key");
        std::fs::write(&cert_path, cert.to_pem().expect("cert pem")).expect("write cert");
        std::fs::write(&key_path, key.private_key_to_pem_pkcs8().expect("key pem"))
            .expect("write key");
        (cert_path, key_path)
    }

    fn context(config: &TlsConfig) -> Result<Option<openssl::ssl::SslContext>, TlsError> {
        Ok(build(config)?.map(|acceptor| acceptor.context))
    }

    fn config(dir: &TempDir) -> TlsConfig {
        let (cert_file, key_file) = self_signed(dir);
        TlsConfig {
            cert_file: Some(cert_file),
            key_file: Some(key_file),
            ..TlsConfig::default()
        }
    }

    /// Run one handshake over loopback with a client pinned to `max` and report
    /// whether it negotiated, plus the ALPN protocol it got.
    fn handshake_with(
        context: &openssl::ssl::SslContext,
        max: SslVersion,
        cipher: Option<&str>,
    ) -> Result<Option<Vec<u8>>, String> {
        handshake_with_groups(context, max, cipher, None)
    }

    fn handshake_with_alpn(
        context: &openssl::ssl::SslContext,
        alpn: &[u8],
    ) -> Result<Option<Vec<u8>>, String> {
        handshake_with_opts(context, SslVersion::TLS1_3, None, None, alpn)
    }

    fn handshake_with_groups(
        context: &openssl::ssl::SslContext,
        max: SslVersion,
        cipher: Option<&str>,
        groups: Option<&str>,
    ) -> Result<Option<Vec<u8>>, String> {
        handshake_with_opts(context, max, cipher, groups, ALPN_H2)
    }

    fn handshake_with_opts(
        context: &openssl::ssl::SslContext,
        max: SslVersion,
        cipher: Option<&str>,
        groups: Option<&str>,
        alpn: &[u8],
    ) -> Result<Option<Vec<u8>>, String> {
        use openssl::ssl::Ssl;
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let context = context.clone();
        let server = std::thread::spawn(move || {
            let (tcp, _) = listener.accept().expect("accept");
            let ssl = Ssl::new(&context).expect("ssl");
            let mut stream = match ssl.accept(tcp) {
                Ok(stream) => stream,
                Err(_) => return None,
            };
            let mut buf = [0_u8; 4];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(b"pong");
            Some(stream.ssl().selected_alpn_protocol().map(<[u8]>::to_vec))
        });

        let mut builder = SslConnector::builder(SslMethod::tls_client()).expect("connector");
        builder.set_verify(SslVerifyMode::NONE);
        builder.set_max_proto_version(Some(max)).expect("max");
        builder.set_alpn_protos(alpn).expect("alpn");
        if let Some(cipher) = cipher {
            builder.set_cipher_list(cipher).expect("cipher");
        }
        if let Some(groups) = groups {
            builder.set_groups_list(groups).expect("groups");
        }
        let connector = builder.build();
        let tcp = std::net::TcpStream::connect(addr).expect("connect");
        let result = connector
            .connect("localhost", tcp)
            .map_err(|e| e.to_string());
        match result {
            Ok(mut stream) => {
                let _ = stream.write_all(b"ping");
                let mut buf = [0_u8; 4];
                let _ = stream.read(&mut buf);
                Ok(server.join().expect("server thread").flatten())
            }
            Err(e) => {
                let _ = server.join();
                Err(e)
            }
        }
    }

    #[test]
    fn disabled_config_builds_no_context() {
        let ctx = context(&TlsConfig::default()).expect("build");
        assert!(ctx.is_none());
    }

    #[test]
    fn half_configured_key_pair_is_a_config_error() {
        let config = TlsConfig {
            cert_file: Some(PathBuf::from("/nonexistent/tls.crt")),
            ..TlsConfig::default()
        };
        assert!(matches!(
            context(&config),
            Err(TlsError::Config(TlsConfigError::CertWithoutKey))
        ));
    }

    #[test]
    fn missing_files_are_openssl_errors() {
        let config = TlsConfig {
            cert_file: Some(PathBuf::from("/nonexistent/tls.crt")),
            key_file: Some(PathBuf::from("/nonexistent/tls.key")),
            ..TlsConfig::default()
        };
        assert!(matches!(context(&config), Err(TlsError::OpenSsl(_))));
    }

    #[test]
    fn mismatched_key_is_rejected() {
        let a = TempDir::new().expect("tempdir");
        let b = TempDir::new().expect("tempdir");
        let (cert_file, _) = self_signed(&a);
        let (_, key_file) = self_signed(&b);
        let config = TlsConfig {
            cert_file: Some(cert_file),
            key_file: Some(key_file),
            ..TlsConfig::default()
        };
        assert!(matches!(context(&config), Err(TlsError::OpenSsl(_))));
    }

    #[test]
    fn negotiates_h2_over_tls13_by_default() {
        let dir = TempDir::new().expect("tempdir");
        let ctx = context(&config(&dir)).expect("build").expect("enabled");
        let alpn = handshake_with(&ctx, SslVersion::TLS1_3, None).expect("handshake");
        assert_eq!(alpn.as_deref(), Some(&b"h2"[..]));
    }

    #[test]
    fn min_version_tls13_rejects_tls12_clients() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.min_version = Some(TlsVersion::Tls13);
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with(&ctx, SslVersion::TLS1_2, None).is_err());
        assert!(handshake_with(&ctx, SslVersion::TLS1_3, None).is_ok());
    }

    #[test]
    fn client_offering_only_http1_is_refused() {
        let dir = TempDir::new().expect("tempdir");
        let config = config(&dir);
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with_alpn(&ctx, b"\x08http/1.1").is_err());
        assert!(handshake_with_alpn(&ctx, ALPN_H2).is_ok());
    }

    #[test]
    fn unset_min_version_refuses_tls11_clients() {
        let dir = TempDir::new().expect("tempdir");
        let config = config(&dir);
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with(&ctx, SslVersion::TLS1_1, None).is_err());
        assert!(handshake_with(&ctx, SslVersion::TLS1_2, None).is_ok());
    }

    #[test]
    fn min_version_below_tls12_is_raised() {
        assert_eq!(min_version(None), TlsVersion::Tls12);
        assert_eq!(min_version(Some(TlsVersion::Tls10)), TlsVersion::Tls12);
        assert_eq!(min_version(Some(TlsVersion::Tls13)), TlsVersion::Tls13);
    }

    #[test]
    fn min_version_tls12_accepts_tls12_clients() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.min_version = Some(TlsVersion::Tls12);
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with(&ctx, SslVersion::TLS1_2, None).is_ok());
    }

    #[test]
    fn cipher_suites_restrict_tls12_negotiation() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.min_version = Some(TlsVersion::Tls12);
        config.cipher_suites = vec![
            "ECDHE-RSA-AES256-GCM-SHA384".to_string(),
            "TLS_AES_256_GCM_SHA384".to_string(),
        ];
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(
            handshake_with(
                &ctx,
                SslVersion::TLS1_2,
                Some("ECDHE-RSA-AES256-GCM-SHA384")
            )
            .is_ok()
        );
        assert!(
            handshake_with(
                &ctx,
                SslVersion::TLS1_2,
                Some("ECDHE-RSA-AES128-GCM-SHA256")
            )
            .is_err()
        );
    }

    #[test]
    fn groups_restrict_key_exchange() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.groups = vec!["X25519".to_string()];
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with_groups(&ctx, SslVersion::TLS1_3, None, Some("X25519")).is_ok());
        assert!(handshake_with_groups(&ctx, SslVersion::TLS1_3, None, Some("secp384r1")).is_err());
    }

    #[test]
    fn unknown_groups_are_dropped_not_fatal() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.groups = vec![
            "NOT_A_GROUP".to_string(),
            "X25519".to_string(),
            " secp256r1 ".to_string(),
        ];
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with_groups(&ctx, SslVersion::TLS1_3, None, Some("secp256r1")).is_ok());
        assert!(handshake_with_groups(&ctx, SslVersion::TLS1_3, None, Some("secp384r1")).is_err());
    }

    #[test]
    fn all_unknown_groups_leave_openssl_defaults() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.groups = vec!["NOT_A_GROUP".to_string()];
        let ctx = context(&config).expect("build").expect("enabled");
        assert!(handshake_with_groups(&ctx, SslVersion::TLS1_3, None, Some("secp384r1")).is_ok());
    }

    #[test]
    fn unknown_cipher_name_is_rejected_at_build() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.cipher_suites = vec!["NOT-A-CIPHER".to_string()];
        assert!(matches!(context(&config), Err(TlsError::OpenSsl(_))));
    }

    #[test]
    fn unknown_tls13_suite_is_rejected_at_build() {
        let dir = TempDir::new().expect("tempdir");
        let mut config = config(&dir);
        config.cipher_suites = vec!["TLS_NOT_A_SUITE".to_string()];
        assert!(matches!(context(&config), Err(TlsError::OpenSsl(_))));
    }
}
