// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use object_store::client::{
    CryptoProvider, DigestAlgorithm, DigestContext, HmacContext, Signer, SigningAlgorithm,
};
use openssl::error::ErrorStack;
use openssl::hash::MessageDigest;
use openssl::pkey::{PKey, Private};
use openssl::sign::Signer as OpensslSigner;
use thiserror::Error;

const STORE: &str = "OpensslCryptoProvider";

#[derive(Debug, Error)]
enum OpensslCryptoError {
    #[error("Invalid RSA key: {source}")]
    InvalidKey { source: ErrorStack },

    #[error("Error computing digest: {source}")]
    Digest { source: ErrorStack },

    #[error("Error signing: {source}")]
    Sign { source: ErrorStack },

    #[error("Digest input was not fully consumed")]
    Incomplete,
}

impl From<OpensslCryptoError> for object_store::Error {
    fn from(value: OpensslCryptoError) -> Self {
        Self::Generic {
            store: STORE,
            source: Box::new(value),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct OpensslCryptoProvider;

impl CryptoProvider for OpensslCryptoProvider {
    fn digest(&self, algorithm: DigestAlgorithm) -> object_store::Result<Box<dyn DigestContext>> {
        let digest = message_digest(&algorithm)?;
        let hasher = openssl::hash::Hasher::new(digest)
            .map_err(|source| OpensslCryptoError::Digest { source })?;
        Ok(Box::new(OpensslDigestContext {
            hasher,
            out: None,
            failed: false,
        }))
    }

    fn hmac(
        &self,
        algorithm: DigestAlgorithm,
        secret: &[u8],
    ) -> object_store::Result<Box<dyn HmacContext>> {
        let key = PKey::hmac(secret).map_err(|source| OpensslCryptoError::InvalidKey { source })?;
        Ok(Box::new(OpensslHmacContext {
            key,
            digest: message_digest(&algorithm)?,
            buffer: Vec::new(),
            out: None,
        }))
    }

    fn sign(
        &self,
        algorithm: SigningAlgorithm,
        pem: &[u8],
    ) -> object_store::Result<Box<dyn Signer>> {
        let digest = match algorithm {
            SigningAlgorithm::RS256 => MessageDigest::sha256(),
            _ => return Err(unsupported(&algorithm)),
        };
        let key = PKey::private_key_from_pem(pem)
            .map_err(|source| OpensslCryptoError::InvalidKey { source })?;
        Ok(Box::new(OpensslRsaKey { key, digest }))
    }
}

fn message_digest(algorithm: &DigestAlgorithm) -> object_store::Result<MessageDigest> {
    match algorithm {
        DigestAlgorithm::Sha256 => Ok(MessageDigest::sha256()),
        _ => Err(unsupported(algorithm)),
    }
}

fn unsupported(algorithm: &dyn std::fmt::Debug) -> object_store::Error {
    object_store::Error::NotSupported {
        source: format!("Unsupported algorithm {algorithm:?}").into(),
    }
}

struct OpensslDigestContext {
    hasher: openssl::hash::Hasher,
    out: Option<Vec<u8>>,
    failed: bool,
}

impl DigestContext for OpensslDigestContext {
    fn update(&mut self, data: &[u8]) {
        if self.hasher.update(data).is_err() {
            self.failed = true;
        }
    }

    fn finish(&mut self) -> object_store::Result<&[u8]> {
        if self.failed {
            return Err(OpensslCryptoError::Incomplete.into());
        }
        let digest = self
            .hasher
            .finish()
            .map_err(|source| OpensslCryptoError::Digest { source })?;
        Ok(self.out.insert(digest.to_vec()).as_slice())
    }
}

struct OpensslHmacContext {
    key: PKey<Private>,
    digest: MessageDigest,
    buffer: Vec<u8>,
    out: Option<Vec<u8>>,
}

impl HmacContext for OpensslHmacContext {
    fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    fn finish(&mut self) -> object_store::Result<&[u8]> {
        let mut signer = OpensslSigner::new(self.digest, &self.key)
            .map_err(|source| OpensslCryptoError::Sign { source })?;
        signer
            .update(&self.buffer)
            .map_err(|source| OpensslCryptoError::Sign { source })?;
        let tag = signer
            .sign_to_vec()
            .map_err(|source| OpensslCryptoError::Sign { source })?;
        Ok(self.out.insert(tag).as_slice())
    }
}

struct OpensslRsaKey {
    key: PKey<Private>,
    digest: MessageDigest,
}

impl Signer for OpensslRsaKey {
    fn sign(&self, string_to_sign: &[u8]) -> object_store::Result<Vec<u8>> {
        let mut signer = OpensslSigner::new(self.digest, &self.key)
            .map_err(|source| OpensslCryptoError::Sign { source })?;
        signer
            .update(string_to_sign)
            .map_err(|source| OpensslCryptoError::Sign { source })?;
        let signature = signer
            .sign_to_vec()
            .map_err(|source| OpensslCryptoError::Sign { source })?;
        Ok(signature)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use openssl::hash::MessageDigest;
    use openssl::rsa::Rsa;
    use openssl::sign::Verifier;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn digest_matches_nist_sha256_vector() {
        let mut ctx = OpensslCryptoProvider
            .digest(DigestAlgorithm::Sha256)
            .expect("digest context");
        ctx.update(b"abc");
        assert_eq!(
            hex(ctx.finish().expect("digest")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn digest_is_incremental() {
        let mut split = OpensslCryptoProvider
            .digest(DigestAlgorithm::Sha256)
            .expect("digest context");
        split.update(b"a");
        split.update(b"b");
        split.update(b"c");
        let mut whole = OpensslCryptoProvider
            .digest(DigestAlgorithm::Sha256)
            .expect("digest context");
        whole.update(b"abc");
        assert_eq!(
            hex(split.finish().expect("digest")),
            hex(whole.finish().expect("digest"))
        );
    }

    #[test]
    fn hmac_matches_rfc4231_case_2() {
        let mut ctx = OpensslCryptoProvider
            .hmac(DigestAlgorithm::Sha256, b"Jefe")
            .expect("hmac context");
        ctx.update(b"what do ya want ");
        ctx.update(b"for nothing?");
        assert_eq!(
            hex(ctx.finish().expect("hmac")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn rs256_signature_verifies() {
        let rsa = Rsa::generate(2048).expect("generate key");
        let key = PKey::from_rsa(rsa).expect("pkey");
        let pem = key.private_key_to_pem_pkcs8().expect("pem");

        let signer = OpensslCryptoProvider
            .sign(SigningAlgorithm::RS256, &pem)
            .expect("signer");
        let signature = signer.sign(b"string to sign").expect("signature");

        let mut verifier = Verifier::new(MessageDigest::sha256(), &key).expect("verifier");
        verifier.update(b"string to sign").expect("update");
        assert!(verifier.verify(&signature).expect("verify"));
    }

    #[test]
    fn digest_reports_a_failed_update_on_finish() {
        let mut ctx = OpensslDigestContext {
            hasher: openssl::hash::Hasher::new(MessageDigest::sha256()).expect("hasher"),
            out: None,
            failed: true,
        };
        let err = ctx
            .finish()
            .expect_err("a failed update must not produce a digest");
        assert!(
            matches!(err, object_store::Error::Generic { store: STORE, .. }),
            "unexpected error: {err:?}"
        );
        assert!(err.to_string().contains("not fully consumed"), "{err}");
    }

    #[test]
    fn unsupported_algorithm_is_not_supported_error() {
        let err = unsupported(&SigningAlgorithm::RS256);
        assert!(
            matches!(err, object_store::Error::NotSupported { .. }),
            "unexpected error: {err:?}"
        );
        assert!(err.to_string().contains("RS256"), "{err}");
    }

    #[test]
    fn sign_rejects_a_pem_that_is_not_a_key() {
        let err = OpensslCryptoProvider.sign(SigningAlgorithm::RS256, b"not a pem");
        assert!(err.is_err(), "expected an error for a non-PEM input");
    }

    const SUITE_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    const SUITE_DATE: &str = "20150830";
    const SUITE_REGION: &str = "us-east-1";
    const SUITE_SERVICE: &str = "service";
    const SUITE_CANONICAL_REQUEST: &str = concat!(
        "GET\n",
        "/\n",
        "\n",
        "host:example.amazonaws.com\n",
        "x-amz-date:20150830T123600Z\n",
        "\n",
        "host;x-amz-date\n",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    const SUITE_CANONICAL_REQUEST_HASH: &str =
        "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63";
    const SUITE_STRING_TO_SIGN: &str = concat!(
        "AWS4-HMAC-SHA256\n",
        "20150830T123600Z\n",
        "20150830/us-east-1/service/aws4_request\n",
        "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63"
    );
    const SUITE_SIGNATURE: &str =
        "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31";

    fn hmac_sha256(secret: &[u8], data: &[u8]) -> Vec<u8> {
        let mut ctx = OpensslCryptoProvider
            .hmac(DigestAlgorithm::Sha256, secret)
            .expect("hmac context");
        ctx.update(data);
        ctx.finish().expect("hmac").to_vec()
    }

    #[test]
    fn hashes_the_sigv4_suite_canonical_request() {
        let mut ctx = OpensslCryptoProvider
            .digest(DigestAlgorithm::Sha256)
            .expect("digest context");
        ctx.update(SUITE_CANONICAL_REQUEST.as_bytes());
        assert_eq!(
            hex(ctx.finish().expect("digest")),
            SUITE_CANONICAL_REQUEST_HASH
        );
    }

    #[test]
    fn reproduces_the_sigv4_suite_signature() {
        let date_key = hmac_sha256(
            format!("AWS4{SUITE_SECRET_KEY}").as_bytes(),
            SUITE_DATE.as_bytes(),
        );
        let region_key = hmac_sha256(&date_key, SUITE_REGION.as_bytes());
        let service_key = hmac_sha256(&region_key, SUITE_SERVICE.as_bytes());
        let signing_key = hmac_sha256(&service_key, b"aws4_request");
        let signature = hmac_sha256(&signing_key, SUITE_STRING_TO_SIGN.as_bytes());
        assert_eq!(hex(&signature), SUITE_SIGNATURE);
    }

    #[tokio::test]
    #[ignore = "requires an S3 endpoint in MX_TEST_S3_ENDPOINT"]
    async fn signs_a_request_a_real_s3_server_accepts() {
        use object_store::{ObjectStoreExt, aws::AmazonS3Builder, path::Path};

        let Ok(endpoint) = std::env::var("MX_TEST_S3_ENDPOINT") else {
            return;
        };
        let bucket = std::env::var("MX_TEST_S3_BUCKET").unwrap_or_else(|_| "test-bucket".into());
        let key = std::env::var("MX_TEST_S3_KEY").unwrap_or_else(|_| "model.bin".into());
        let mut builder = AmazonS3Builder::new()
            .with_region(std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into()))
            .with_bucket_name(bucket)
            .with_endpoint(endpoint)
            .with_access_key_id(std::env::var("AWS_ACCESS_KEY_ID").expect("access key"))
            .with_secret_access_key(std::env::var("AWS_SECRET_ACCESS_KEY").expect("secret key"))
            .with_virtual_hosted_style_request(false)
            .with_allow_http(true)
            .with_crypto_provider(std::sync::Arc::new(OpensslCryptoProvider));
        if let Ok(token) = std::env::var("AWS_SESSION_TOKEN") {
            builder = builder.with_token(token);
        }
        let store = builder.build().expect("build store");

        let got = store
            .get(&Path::from(key))
            .await
            .expect("signed GET was rejected")
            .bytes()
            .await
            .expect("read body");
        assert!(!got.is_empty(), "signed GET returned an empty object");
    }
}
