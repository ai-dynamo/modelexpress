// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use oci_client::Reference;
use std::fmt;

const OCI_SCHEME: &str = "oci://";

#[derive(Debug, Clone)]
pub struct OciReference {
    inner: Reference,
}

impl OciReference {
    pub fn parse(model_name: &str) -> Result<Self> {
        let reference = model_name.strip_prefix(OCI_SCHEME).unwrap_or(model_name);

        if !Self::has_explicit_registry(reference) {
            anyhow::bail!(
                "OCI reference '{model_name}' must be registry-qualified, for example registry.example.com/repo/model:tag"
            );
        }

        if !Self::has_explicit_tag_or_digest(reference) {
            anyhow::bail!("OCI reference '{model_name}' must include an explicit tag or digest");
        }

        let inner = reference
            .parse::<Reference>()
            .with_context(|| format!("Failed to parse OCI reference '{model_name}'"))?;
        Ok(Self { inner })
    }

    pub fn as_client_reference(&self) -> &Reference {
        &self.inner
    }

    pub fn registry(&self) -> &str {
        self.inner.registry()
    }

    pub fn repository(&self) -> &str {
        self.inner.repository()
    }

    pub fn tag(&self) -> Option<&str> {
        self.inner.tag()
    }

    pub fn digest(&self) -> Option<&str> {
        self.inner.digest()
    }

    pub fn registry_endpoint(&self) -> &str {
        self.inner.resolve_registry()
    }

    pub fn canonical_name(&self) -> String {
        match self.digest() {
            Some(digest) => format!("{}/{}@{}", self.registry(), self.repository(), digest),
            None => self.to_string(),
        }
    }

    fn has_explicit_tag_or_digest(reference: &str) -> bool {
        if reference.contains('@') {
            return true;
        }

        let last_slash = reference.rfind('/');
        let last_colon = reference.rfind(':');

        match (last_slash, last_colon) {
            (Some(slash), Some(colon)) => colon > slash,
            (None, Some(_)) => true,
            _ => false,
        }
    }

    fn has_explicit_registry(reference: &str) -> bool {
        let Some((registry, _)) = reference.split_once('/') else {
            return false;
        };

        registry.contains('.') || registry.contains(':') || registry == "localhost"
    }
}

impl fmt::Display for OciReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.inner)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_oci_reference_accepts_scheme_tag_and_digest() {
        let tagged = OciReference::parse("oci://registry.example.com/team/model:v1")
            .expect("tagged reference should parse");
        assert_eq!(tagged.registry(), "registry.example.com");
        assert_eq!(tagged.repository(), "team/model");
        assert_eq!(tagged.tag(), Some("v1"));
        assert_eq!(tagged.to_string(), "registry.example.com/team/model:v1");

        let digest = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let by_digest = OciReference::parse(&format!("registry.example.com/team/model@{digest}"))
            .expect("digest reference should parse");
        assert_eq!(by_digest.digest(), Some(digest));

        let tagged_digest =
            OciReference::parse(&format!("registry.example.com/team/model:v1@{digest}"))
                .expect("tagged digest reference should parse");
        assert_eq!(
            tagged_digest.canonical_name(),
            format!("registry.example.com/team/model@{digest}")
        );
    }

    #[test]
    fn test_parse_oci_reference_rejects_missing_explicit_ref_or_registry() {
        let missing_ref = OciReference::parse("registry.example.com/team/model")
            .expect_err("missing tag or digest should fail");
        assert!(
            missing_ref
                .to_string()
                .contains("must include an explicit tag or digest")
        );

        let missing_registry =
            OciReference::parse("team/model:v1").expect_err("missing registry should fail");
        assert!(
            missing_registry
                .to_string()
                .contains("must be registry-qualified")
        );
    }
}
