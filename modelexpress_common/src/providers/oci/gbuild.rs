// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::layer_download::TITLE_ANNOTATION;
use anyhow::{Context, Result};
use oci_client::manifest::{OciDescriptor, OciImageManifest};
use std::collections::HashSet;

pub const ARTIFACT_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.v1";
pub const RUNTIME_MANIFEST_JSON_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+json";
pub const RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+capnp";
pub const COMPILE_METADATA_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.compile-metadata.v1.tar+zstd";
pub const PAYLOAD_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+zstd";
pub(super) const MANIFEST_CAPNP_FILE_NAME: &str = "manifest.v2.capnp.bin";
pub(super) const COMPILE_METADATA_FILE_NAME: &str = "compile-metadata.tar.zst";
pub(super) const PAYLOAD_FILE_NAME: &str = "payload.tar.zst";

const OCI_EMPTY_CONFIG_MEDIA_TYPE: &str = "application/vnd.oci.empty.v1+json";
const OCI_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";

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
    let mut compile_metadata = false;
    let mut payload = false;

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
            COMPILE_METADATA_MEDIA_TYPE => {
                if compile_metadata || title != COMPILE_METADATA_FILE_NAME {
                    anyhow::bail!(
                        "GBuild OCI artifact must contain one {COMPILE_METADATA_FILE_NAME} layer"
                    );
                }
                compile_metadata = true;
            }
            PAYLOAD_MEDIA_TYPE => {
                if payload || title != PAYLOAD_FILE_NAME {
                    anyhow::bail!("GBuild OCI artifact must contain one {PAYLOAD_FILE_NAME} layer");
                }
                payload = true;
            }
            media_type => anyhow::bail!(
                "GBuild OCI artifact contains unsupported layer media type '{media_type}'"
            ),
        }
    }

    if !manifest_json || !manifest_capnp || !compile_metadata || !payload {
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
                descriptor(
                    COMPILE_METADATA_MEDIA_TYPE,
                    'd',
                    Some("compile-metadata.tar.zst"),
                ),
                descriptor(PAYLOAD_MEDIA_TYPE, 'e', Some("payload.tar.zst")),
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
}
