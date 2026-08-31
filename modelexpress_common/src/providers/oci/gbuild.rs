// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::layer_download::TITLE_ANNOTATION;
use anyhow::{Context, Result};
use oci_client::manifest::{OciDescriptor, OciImageManifest};
use serde::Deserialize;
use std::collections::HashSet;

pub const ARTIFACT_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.v1";
pub const RUNTIME_MANIFEST_JSON_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+json";
pub const RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+capnp";
pub const PRESET_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.preset.v1+json";
pub const PAYLOAD_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
pub(super) const MANIFEST_CAPNP_FILE_NAME: &str = "manifest.v2.capnp.bin";

const OCI_EMPTY_CONFIG_MEDIA_TYPE: &str = "application/vnd.oci.empty.v1+json";
const OCI_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const RUNTIME_MANIFEST_REVISION: u8 = 2;

pub fn is_gbuild_artifact(manifest: &OciImageManifest) -> bool {
    manifest.artifact_type.as_deref() == Some(ARTIFACT_MEDIA_TYPE)
}

pub fn validate_gbuild_manifest(manifest: &OciImageManifest) -> Result<()> {
    if manifest.schema_version != 2
        || manifest.media_type.as_deref() != Some(OCI_MANIFEST_MEDIA_TYPE)
        || !is_gbuild_artifact(manifest)
    {
        anyhow::bail!("OCI manifest does not use the GBuild full-compile artifact contract");
    }
    if manifest.subject.is_some() {
        anyhow::bail!("GBuild OCI manifest must not contain a subject");
    }
    if manifest.config.media_type != OCI_EMPTY_CONFIG_MEDIA_TYPE {
        anyhow::bail!("GBuild OCI artifact must use the standard empty config");
    }
    validate_descriptor(&manifest.config, "GBuild OCI config")?;

    let mut titles = HashSet::new();
    let mut manifest_json = false;
    let mut manifest_capnp = false;
    let mut preset = false;
    let mut payload_count = 0usize;

    for descriptor in &manifest.layers {
        validate_descriptor(descriptor, "GBuild OCI layer")?;
        let title = descriptor
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(TITLE_ANNOTATION))
            .context("GBuild OCI layer is missing its title annotation")?;
        if !titles.insert(title) {
            anyhow::bail!("GBuild OCI layer titles must be unique");
        }

        match descriptor.media_type.as_str() {
            RUNTIME_MANIFEST_JSON_MEDIA_TYPE => {
                if manifest_json || title != "manifest.json" {
                    anyhow::bail!("GBuild OCI artifact must contain one manifest.json layer");
                }
                manifest_json = true;
            }
            RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE => {
                if manifest_capnp || title != MANIFEST_CAPNP_FILE_NAME {
                    anyhow::bail!(
                        "GBuild OCI artifact must contain one {MANIFEST_CAPNP_FILE_NAME} layer"
                    );
                }
                manifest_capnp = true;
            }
            PRESET_MEDIA_TYPE => {
                if preset
                    || title.contains('/')
                    || !title.ends_with("-original.json")
                    || title == "-original.json"
                {
                    anyhow::bail!(
                        "GBuild OCI artifact must contain one top-level *-original.json layer"
                    );
                }
                preset = true;
            }
            PAYLOAD_MEDIA_TYPE => payload_count = payload_count.saturating_add(1),
            media_type => anyhow::bail!(
                "GBuild OCI artifact contains unsupported layer media type '{media_type}'"
            ),
        }
    }

    if !manifest_json || !manifest_capnp || !preset || payload_count == 0 {
        anyhow::bail!("GBuild OCI artifact is missing required metadata or payload layers");
    }
    Ok(())
}

fn validate_descriptor(descriptor: &OciDescriptor, description: &str) -> Result<()> {
    let digest = descriptor.digest.strip_prefix("sha256:");
    if descriptor.size <= 0
        || digest.is_none_or(|hash| {
            hash.len() != 64
                || !hash
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        || descriptor.urls.is_some()
    {
        anyhow::bail!(
            "{description} descriptor must have a positive size, SHA-256 digest, and no alternate URLs"
        );
    }
    Ok(())
}

pub(super) fn validate_runtime_manifest(manifest: &[u8]) -> Result<()> {
    let header: RuntimeManifestHeader =
        serde_json::from_slice(manifest).context("Failed to parse the GBuild runtime manifest")?;
    if header.contract_revision != RUNTIME_MANIFEST_REVISION {
        anyhow::bail!(
            "GBuild runtime manifest revision {} is not supported; expected {RUNTIME_MANIFEST_REVISION}",
            header.contract_revision
        );
    }
    Ok(())
}

#[derive(Deserialize)]
struct RuntimeManifestHeader {
    contract_revision: u8,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn descriptor(media_type: &str, digest_byte: char, title: Option<&str>) -> serde_json::Value {
        let mut descriptor = json!({
            "mediaType": media_type,
            "digest": format!("sha256:{}", digest_byte.to_string().repeat(64)),
            "size": 10,
        });
        if let Some(title) = title {
            descriptor["annotations"] = json!({TITLE_ANNOTATION: title});
        }
        descriptor
    }

    fn manifest() -> OciImageManifest {
        serde_json::from_value(json!({
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "artifactType": ARTIFACT_MEDIA_TYPE,
            "config": descriptor(OCI_EMPTY_CONFIG_MEDIA_TYPE, 'a', None),
            "layers": [
                descriptor(RUNTIME_MANIFEST_JSON_MEDIA_TYPE, 'b', Some("manifest.json")),
                descriptor(
                    RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE,
                    'c',
                    Some(MANIFEST_CAPNP_FILE_NAME),
                ),
                descriptor(PRESET_MEDIA_TYPE, 'd', Some("llama-original.json")),
                descriptor(PAYLOAD_MEDIA_TYPE, 'e', Some("payload.tar")),
            ],
        }))
        .expect("valid OCI manifest fixture")
    }

    #[test]
    fn test_gbuild_artifact_accepts_required_layers() {
        validate_gbuild_manifest(&manifest()).expect("valid GBuild artifact");
    }

    #[test]
    fn test_gbuild_artifact_rejects_missing_or_unknown_layers() {
        let mut missing = manifest();
        missing.layers.pop();
        assert!(validate_gbuild_manifest(&missing).is_err());

        let mut unknown = manifest();
        unknown.layers[3].media_type = "application/octet-stream".to_string();
        assert!(validate_gbuild_manifest(&unknown).is_err());
    }

    #[test]
    fn test_gbuild_artifact_rejects_runtime_manifest_v1_media_type() {
        let mut manifest = manifest();
        manifest.layers[0].media_type =
            "application/vnd.groq.gbuild.runtime-manifest.v1+json".to_string();
        assert!(validate_gbuild_manifest(&manifest).is_err());
    }

    #[test]
    fn test_gbuild_artifact_accepts_runtime_manifest_revision_2() {
        validate_runtime_manifest(br#"{"contract_revision":2,"model":{}}"#)
            .expect("revision 2 must be accepted");
    }

    #[test]
    fn test_gbuild_artifact_rejects_other_runtime_manifest_revisions() {
        for manifest in [
            br#"{"contract_revision":1}"#.as_slice(),
            br#"{"contract_revision":3}"#.as_slice(),
            br#"{"contract_revision":true}"#.as_slice(),
            br#"{"model":{}}"#.as_slice(),
        ] {
            assert!(validate_runtime_manifest(manifest).is_err());
        }
    }
}
