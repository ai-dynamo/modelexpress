// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{layer_download::TITLE_ANNOTATION, path::ArtifactPath};
use anyhow::{Context, Result};
use oci_client::manifest::{OciDescriptor, OciImageManifest};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

pub const ARTIFACT_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.v1";
pub const CONFIG_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.config.v1+json";
pub const RUNTIME_MANIFEST_JSON_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+json";
pub const RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+capnp";
pub const PRESET_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.preset.v1+json";
pub const PAYLOAD_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+zstd";
pub(super) const MANIFEST_CAPNP_FILE_NAME: &str = "manifest.v2.capnp.bin";

const OCI_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const CREATED_ANNOTATION: &str = "org.opencontainers.image.created";
const REPRODUCIBLE_CREATED_AT: &str = "1970-01-01T00:00:00Z";
const RUNTIME_MANIFEST_REVISION: u8 = 2;

#[derive(Debug)]
pub struct GbuildArtifact {
    pub metadata: [GbuildMetadataLayer; 3],
    pub payloads: Vec<GbuildPayloadLayer>,
}

#[derive(Debug)]
pub struct GbuildMetadataLayer {
    pub descriptor: OciDescriptor,
    pub path: ArtifactPath,
}

#[derive(Debug)]
pub struct GbuildPayloadLayer {
    pub descriptor: OciDescriptor,
    pub members: Vec<ArtifactPath>,
    pub uncompressed_size_bytes: u64,
}

impl GbuildArtifact {
    pub fn from_manifest_and_config(manifest: &OciImageManifest, config: &[u8]) -> Result<Self> {
        Self::validate_outer_manifest(manifest)?;
        let config: GbuildConfig =
            serde_json::from_slice(config).context("Failed to parse the GBuild OCI config")?;
        if config.version != 1 {
            anyhow::bail!(
                "GBuild OCI config version {} is not supported",
                config.version
            );
        }
        if config.partitions.is_empty() {
            anyhow::bail!("GBuild OCI config must contain at least one partition");
        }
        let partition_ids: Vec<u32> = config
            .partitions
            .iter()
            .map(|partition| partition.partition_id)
            .collect();
        if partition_ids.windows(2).any(|ids| ids[0] >= ids[1]) {
            anyhow::bail!("GBuild OCI config partitions must have unique IDs in canonical order");
        }

        let payload_count = usize::from(config.tokenizer.is_some())
            .checked_add(config.partitions.len())
            .and_then(|count| count.checked_add(usize::from(config.runtime_assets.is_some())))
            .context("GBuild OCI payload count overflowed")?;
        let expected_layer_count = payload_count
            .checked_add(3)
            .context("GBuild OCI layer count overflowed")?;
        if manifest.layers.len() != expected_layer_count {
            anyhow::bail!(
                "GBuild OCI manifest has {} layers but its config describes {}",
                manifest.layers.len(),
                expected_layer_count
            );
        }

        let mut layers = Self::index_layers(manifest)?;
        let manifest_json = Self::take_metadata_layer(
            &mut layers,
            "manifest.json",
            RUNTIME_MANIFEST_JSON_MEDIA_TYPE,
        )?;
        let manifest_capnp = Self::take_metadata_layer(
            &mut layers,
            MANIFEST_CAPNP_FILE_NAME,
            RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE,
        )?;
        let preset_path = Self::preset_path(&layers)?;
        let preset = Self::take_metadata_layer(&mut layers, &preset_path, PRESET_MEDIA_TYPE)?;
        let metadata = [manifest_json, manifest_capnp, preset];

        let mut owned_paths = HashSet::from([
            metadata[0].path.as_str().to_string(),
            metadata[1].path.as_str().to_string(),
            metadata[2].path.as_str().to_string(),
        ]);
        let mut payloads = Vec::with_capacity(payload_count);
        if let Some(tokenizer) = config.tokenizer {
            payloads.push(Self::take_payload_layer(
                &mut layers,
                tokenizer.layer,
                tokenizer.members,
                tokenizer.uncompressed_size_bytes,
                &mut owned_paths,
            )?);
        }
        for partition in config.partitions {
            payloads.push(Self::take_payload_layer(
                &mut layers,
                partition.layer,
                partition.members,
                partition.uncompressed_size_bytes,
                &mut owned_paths,
            )?);
        }
        if let Some(runtime_assets) = config.runtime_assets {
            payloads.push(Self::take_payload_layer(
                &mut layers,
                runtime_assets.layer,
                runtime_assets.members,
                runtime_assets.uncompressed_size_bytes,
                &mut owned_paths,
            )?);
        }
        if !layers.is_empty() {
            anyhow::bail!("GBuild OCI manifest contains layers that its config does not own");
        }

        Ok(Self { metadata, payloads })
    }

    fn validate_outer_manifest(manifest: &OciImageManifest) -> Result<()> {
        if manifest.schema_version != 2
            || manifest.media_type.as_deref() != Some(OCI_MANIFEST_MEDIA_TYPE)
            || !is_gbuild_artifact(manifest)
        {
            anyhow::bail!("OCI manifest does not use the GBuild full-compile artifact contract");
        }
        if manifest.subject.is_some() {
            anyhow::bail!("GBuild OCI manifest must not contain a subject");
        }
        let expected_annotations = BTreeMap::from([(
            CREATED_ANNOTATION.to_string(),
            REPRODUCIBLE_CREATED_AT.to_string(),
        )]);
        if manifest.annotations.as_ref() != Some(&expected_annotations) {
            anyhow::bail!("GBuild OCI manifest must use the reproducible creation annotation");
        }
        if manifest.config.media_type != CONFIG_MEDIA_TYPE {
            anyhow::bail!("GBuild OCI config must use media type '{CONFIG_MEDIA_TYPE}'");
        }
        Self::validate_descriptor(&manifest.config, "GBuild OCI config")?;
        if manifest.config.annotations.is_some() {
            anyhow::bail!("GBuild OCI config descriptor must not contain annotations");
        }
        if manifest.layers.len() < 4 {
            anyhow::bail!("GBuild OCI manifest must contain metadata and payload layers");
        }
        Ok(())
    }

    fn index_layers(manifest: &OciImageManifest) -> Result<BTreeMap<String, OciDescriptor>> {
        let mut layers = BTreeMap::new();
        for descriptor in &manifest.layers {
            Self::validate_descriptor(descriptor, "GBuild OCI layer")?;
            let Some(annotations) = descriptor.annotations.as_ref() else {
                anyhow::bail!("GBuild OCI layer descriptor must contain a title annotation");
            };
            let Some(title) = annotations.get(TITLE_ANNOTATION) else {
                anyhow::bail!("GBuild OCI layer descriptor must contain a title annotation");
            };
            if annotations.len() != 1 {
                anyhow::bail!("GBuild OCI layer descriptor must contain only the title annotation");
            }
            if layers.insert(title.clone(), descriptor.clone()).is_some() {
                anyhow::bail!("GBuild OCI layer titles must be unique");
            }
        }
        Ok(layers)
    }

    fn take_metadata_layer(
        layers: &mut BTreeMap<String, OciDescriptor>,
        path: &str,
        media_type: &str,
    ) -> Result<GbuildMetadataLayer> {
        let descriptor = layers
            .remove(path)
            .with_context(|| format!("GBuild OCI metadata layer '{path}' is missing"))?;
        if descriptor.media_type != media_type {
            anyhow::bail!("GBuild OCI metadata layer '{path}' must use media type '{media_type}'");
        }
        let path = ArtifactPath::from_relative_path(
            Path::new(path),
            &format!("GBuild OCI metadata path '{path}'"),
        )?;
        Ok(GbuildMetadataLayer { descriptor, path })
    }

    fn preset_path(layers: &BTreeMap<String, OciDescriptor>) -> Result<String> {
        let mut presets = layers
            .iter()
            .filter(|(_, descriptor)| descriptor.media_type == PRESET_MEDIA_TYPE);
        let Some((path, _)) = presets.next() else {
            anyhow::bail!("GBuild OCI preset metadata layer is missing");
        };
        if presets.next().is_some()
            || path.contains('/')
            || !path.ends_with("-original.json")
            || path == "-original.json"
        {
            anyhow::bail!("GBuild OCI preset must be one top-level *-original.json layer");
        }
        Ok(path.clone())
    }

    fn take_payload_layer(
        layers: &mut BTreeMap<String, OciDescriptor>,
        layer: String,
        member_names: Vec<String>,
        uncompressed_size_bytes: u64,
        owned_paths: &mut HashSet<String>,
    ) -> Result<GbuildPayloadLayer> {
        ArtifactPath::from_relative_path(
            Path::new(&layer),
            &format!("GBuild OCI payload layer '{layer}'"),
        )?;
        if !layer.starts_with("payloads/") || !layer.ends_with(".tar.zst") {
            anyhow::bail!("GBuild OCI payload layer must use a payloads/*.tar.zst title");
        }
        let descriptor = layers
            .remove(&layer)
            .with_context(|| format!("GBuild OCI payload layer '{layer}' is missing"))?;
        if descriptor.media_type != PAYLOAD_MEDIA_TYPE || uncompressed_size_bytes == 0 {
            anyhow::bail!(
                "GBuild OCI payload layer must use '{PAYLOAD_MEDIA_TYPE}' and a positive uncompressed size"
            );
        }
        if member_names.is_empty()
            || member_names
                .windows(2)
                .any(|members| members[0] >= members[1])
        {
            anyhow::bail!(
                "GBuild OCI config members must be non-empty, unique, and in canonical order"
            );
        }

        let mut members = Vec::with_capacity(member_names.len());
        for member in member_names {
            let path = ArtifactPath::from_relative_path(
                Path::new(&member),
                &format!("GBuild OCI config member '{member}'"),
            )?;
            if !owned_paths.insert(member) {
                anyhow::bail!("GBuild OCI config members cannot overlap another artifact path");
            }
            members.push(path);
        }

        Ok(GbuildPayloadLayer {
            descriptor,
            members,
            uncompressed_size_bytes,
        })
    }

    fn validate_descriptor(descriptor: &OciDescriptor, description: &str) -> Result<()> {
        if descriptor.size <= 0
            || !Self::is_sha256_digest(&descriptor.digest)
            || descriptor.urls.is_some()
        {
            anyhow::bail!(
                "{description} descriptor must have a positive size, SHA-256 digest, and no alternate URLs"
            );
        }
        Ok(())
    }

    fn is_sha256_digest(digest: &str) -> bool {
        let Some(hash) = digest.strip_prefix("sha256:") else {
            return false;
        };
        hash.len() == 64
            && hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }
}

pub fn is_gbuild_artifact(manifest: &OciImageManifest) -> bool {
    manifest.artifact_type.as_deref() == Some(ARTIFACT_MEDIA_TYPE)
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GbuildConfig {
    version: u8,
    tokenizer: Option<ConfigPayload>,
    partitions: Vec<ConfigPartition>,
    runtime_assets: Option<ConfigPayload>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigPayload {
    layer: String,
    members: Vec<String>,
    uncompressed_size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigPartition {
    partition_id: u32,
    layer: String,
    members: Vec<String>,
    uncompressed_size_bytes: u64,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use oci_client::manifest::OciImageManifest;
    use serde_json::{Value, json};

    const DIGEST_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const DIGEST_D: &str =
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DIGEST_E: &str =
        "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn config() -> Value {
        json!({
            "version": 1,
            "tokenizer": {
                "layer": "payloads/0000.tar.zst",
                "members": ["tokenizer/config.json"],
                "uncompressed_size_bytes": 400,
            },
            "partitions": [{
                "partition_id": 0,
                "layer": "payloads/0001.tar.zst",
                "members": ["program.0.gas", "program.0.weight"],
                "uncompressed_size_bytes": 500,
            }],
        })
    }

    fn outer_descriptor(media_type: &str, digest: &str, size: u64, title: Option<&str>) -> Value {
        let mut value = json!({
            "mediaType": media_type,
            "digest": digest,
            "size": size,
        });
        if let Some(title) = title {
            value["annotations"] = json!({TITLE_ANNOTATION: title});
        }
        value
    }

    fn manifest() -> OciImageManifest {
        serde_json::from_value(json!({
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "artifactType": ARTIFACT_MEDIA_TYPE,
            "config": outer_descriptor(CONFIG_MEDIA_TYPE, DIGEST_A, 100, None),
            "layers": [
                outer_descriptor(
                    RUNTIME_MANIFEST_JSON_MEDIA_TYPE,
                    DIGEST_A,
                    10,
                    Some("manifest.json"),
                ),
                outer_descriptor(
                    RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE,
                    DIGEST_B,
                    20,
                    Some(MANIFEST_CAPNP_FILE_NAME),
                ),
                outer_descriptor(PRESET_MEDIA_TYPE, DIGEST_C, 30, Some("llama-original.json")),
                outer_descriptor(PAYLOAD_MEDIA_TYPE, DIGEST_D, 40, Some("payloads/0000.tar.zst")),
                outer_descriptor(PAYLOAD_MEDIA_TYPE, DIGEST_E, 50, Some("payloads/0001.tar.zst")),
            ],
            "annotations": {CREATED_ANNOTATION: REPRODUCIBLE_CREATED_AT},
        }))
        .expect("valid OCI manifest fixture")
    }

    fn parse_contract(config: &Value, manifest: &OciImageManifest) -> Result<GbuildArtifact> {
        GbuildArtifact::from_manifest_and_config(
            manifest,
            &serde_json::to_vec(config).expect("serialize config fixture"),
        )
    }

    #[test]
    fn test_gbuild_artifact_maps_config_to_oci_layers() {
        let artifact = parse_contract(&config(), &manifest()).expect("valid contract");

        assert_eq!(artifact.metadata[0].path.as_str(), "manifest.json");
        assert_eq!(artifact.metadata[1].path.as_str(), MANIFEST_CAPNP_FILE_NAME);
        assert_eq!(artifact.metadata[2].path.as_str(), "llama-original.json");
        assert_eq!(artifact.payloads.len(), 2);
        assert_eq!(
            artifact.payloads[0].descriptor.digest, DIGEST_D,
            "the OCI manifest owns the layer descriptor"
        );
        assert_eq!(artifact.payloads[1].members[1].as_str(), "program.0.weight");
        assert_eq!(artifact.payloads[1].uncompressed_size_bytes, 500);
    }

    #[test]
    fn test_gbuild_artifact_rejects_unowned_or_missing_layers() {
        let mut bad_config = config();
        bad_config["partitions"][0]["layer"] = json!("payloads/missing.tar.zst");

        let error = parse_contract(&bad_config, &manifest()).expect_err("missing layer must fail");
        assert!(error.to_string().contains("missing"));
    }

    #[test]
    fn test_gbuild_artifact_rejects_noncanonical_members_and_ownership() {
        let mut bad_config = config();
        bad_config["partitions"][0]["members"] =
            json!(["program.0.weight", "program.0.gas", "tokenizer/config.json"]);

        let error = parse_contract(&bad_config, &manifest()).expect_err("member order must fail");
        assert!(error.to_string().contains("members"));

        bad_config["partitions"][0]["members"] = json!(["tokenizer/config.json"]);
        let error = parse_contract(&bad_config, &manifest()).expect_err("member overlap must fail");
        assert!(error.to_string().contains("overlap"));
    }

    #[test]
    fn test_gbuild_artifact_rejects_runtime_manifest_v1_media_type() {
        let mut bad_manifest = manifest();
        bad_manifest.layers[0].media_type =
            "application/vnd.groq.gbuild.runtime-manifest.v1+json".to_string();

        let error = parse_contract(&config(), &bad_manifest).expect_err("v1 media type must fail");
        assert!(error.to_string().contains(RUNTIME_MANIFEST_JSON_MEDIA_TYPE));
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
            let error = validate_runtime_manifest(manifest).expect_err("non-v2 manifest must fail");
            assert!(
                error.to_string().contains("runtime manifest")
                    || error.to_string().contains("expected 2")
            );
        }
    }
}
