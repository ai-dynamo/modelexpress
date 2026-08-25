// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{layer_download::TITLE_ANNOTATION, path::ArtifactPath};
use anyhow::{Context, Result};
use oci_client::manifest::{OciDescriptor, OciImageManifest};
use serde::Deserialize;
use std::{collections::HashSet, path::Path};

pub const ARTIFACT_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.v1";
pub const CONFIG_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.config.v1+json";
pub const RUNTIME_MANIFEST_JSON_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+json";
pub const RUNTIME_MANIFEST_CAPNP_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.runtime-manifest.v2+capnp";
pub const TRANSPORT_INDEX_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.full-compile.transport.v1+json";
pub const MANIFEST_JSON_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.manifest.v1+json";
pub const MANIFEST_CAPNP_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.manifest.v1+capnp";
pub const PRESET_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.preset.v1+json";
pub const PAYLOAD_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+zstd";
pub(super) const MANIFEST_CAPNP_FILE_NAME: &str = "manifest.v2.capnp.bin";
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
    pub fn from_manifest_and_config(_manifest: &OciImageManifest, _config: &[u8]) -> Result<Self> {
        anyhow::bail!("GBuild OCI config contract is not implemented")
    }

    pub fn from_manifest_and_transport_index(
        manifest: &OciImageManifest,
        transport_index: &[u8],
    ) -> Result<Self> {
        Self::validate_outer_manifest(manifest)?;
        let transport: TransportIndex = serde_json::from_slice(transport_index)
            .context("Failed to parse the GBuild OCI transport index")?;
        if transport.version != 1 {
            anyhow::bail!(
                "GBuild OCI transport-index version {} is not supported",
                transport.version
            );
        }

        let TransportMetadata {
            manifest_json,
            manifest_capnp,
            preset,
        } = transport.metadata;
        let metadata = [
            Self::metadata_layer(
                &manifest.layers[0],
                manifest_json,
                "manifest.json",
                MANIFEST_JSON_MEDIA_TYPE,
            )?,
            Self::metadata_layer(
                &manifest.layers[1],
                manifest_capnp,
                MANIFEST_CAPNP_FILE_NAME,
                MANIFEST_CAPNP_MEDIA_TYPE,
            )?,
            Self::preset_layer(&manifest.layers[2], preset)?,
        ];

        if transport.partitions.is_empty() {
            anyhow::bail!("GBuild OCI transport index must contain at least one partition");
        }
        let partition_ids: Vec<u32> = transport
            .partitions
            .iter()
            .map(|partition| partition.partition_id)
            .collect();
        if partition_ids.windows(2).any(|ids| ids[0] >= ids[1]) {
            anyhow::bail!(
                "GBuild OCI transport-index partitions must have unique IDs in canonical order"
            );
        }

        let payload_count = usize::from(transport.tokenizer.is_some())
            .checked_add(transport.partitions.len())
            .and_then(|count| count.checked_add(usize::from(transport.runtime_assets.is_some())))
            .context("GBuild OCI payload count overflowed")?;
        let expected_layer_count = payload_count
            .checked_add(3)
            .context("GBuild OCI layer count overflowed")?;
        if manifest.layers.len() != expected_layer_count {
            anyhow::bail!(
                "GBuild OCI outer manifest has {} layers but the transport index describes {}",
                manifest.layers.len(),
                expected_layer_count
            );
        }

        let mut owned_paths = HashSet::from([
            metadata[0].path.as_str().to_string(),
            metadata[1].path.as_str().to_string(),
            metadata[2].path.as_str().to_string(),
        ]);
        let mut payloads = Vec::with_capacity(payload_count);
        let mut outer_payloads = manifest.layers.iter().skip(3);

        if let Some(tokenizer) = transport.tokenizer {
            payloads.push(Self::payload_layer(
                outer_payloads
                    .next()
                    .context("GBuild OCI tokenizer layer is missing")?,
                tokenizer.descriptor,
                tokenizer.members,
                &mut owned_paths,
            )?);
        }

        for partition in transport.partitions {
            payloads.push(Self::payload_layer(
                outer_payloads
                    .next()
                    .context("GBuild OCI partition layer is missing")?,
                partition.descriptor,
                partition.members,
                &mut owned_paths,
            )?);
        }

        if let Some(runtime_assets) = transport.runtime_assets {
            payloads.push(Self::payload_layer(
                outer_payloads
                    .next()
                    .context("GBuild OCI runtime-assets layer is missing")?,
                runtime_assets.descriptor,
                runtime_assets.members,
                &mut owned_paths,
            )?);
        }

        Ok(Self { metadata, payloads })
    }

    fn validate_outer_manifest(manifest: &OciImageManifest) -> Result<()> {
        if manifest.schema_version != 2
            || manifest.media_type.as_deref() != Some("application/vnd.oci.image.manifest.v1+json")
            || !is_gbuild_artifact(manifest)
        {
            anyhow::bail!("OCI manifest does not use the GBuild full-compile artifact contract");
        }
        if manifest.subject.is_some() || manifest.annotations.is_some() {
            anyhow::bail!("GBuild OCI outer manifest must not contain subject or annotations");
        }
        if manifest.config.media_type != TRANSPORT_INDEX_MEDIA_TYPE {
            anyhow::bail!("GBuild OCI config must use media type '{TRANSPORT_INDEX_MEDIA_TYPE}'");
        }
        Self::validate_outer_descriptor(&manifest.config, "GBuild OCI config")?;
        if manifest.config.annotations.is_some() {
            anyhow::bail!("GBuild OCI config descriptor must not contain annotations");
        }
        if manifest.layers.len() < 4 {
            anyhow::bail!("GBuild OCI outer manifest must contain metadata and payload layers");
        }
        Ok(())
    }

    fn metadata_layer(
        outer: &OciDescriptor,
        transport: TransportMetadataFile,
        expected_path: &str,
        expected_media_type: &str,
    ) -> Result<GbuildMetadataLayer> {
        if transport.path != expected_path || transport.descriptor.media_type != expected_media_type
        {
            anyhow::bail!(
                "GBuild OCI metadata must use path '{expected_path}' and media type '{expected_media_type}'"
            );
        }
        Self::validate_transport_descriptor(&transport.descriptor, "metadata descriptor")?;
        Self::require_matching_descriptor(outer, &transport.descriptor, "metadata descriptor")?;
        let expected_annotations = std::collections::BTreeMap::from([(
            TITLE_ANNOTATION.to_string(),
            expected_path.into(),
        )]);
        if outer.annotations.as_ref() != Some(&expected_annotations) {
            anyhow::bail!(
                "GBuild OCI metadata descriptor for '{expected_path}' must contain only the title annotation"
            );
        }
        let path = ArtifactPath::from_relative_path(
            Path::new(&transport.path),
            &format!("GBuild OCI metadata path '{}'", transport.path),
        )?;
        Ok(GbuildMetadataLayer {
            descriptor: outer.clone(),
            path,
        })
    }

    fn preset_layer(
        outer: &OciDescriptor,
        transport: TransportMetadataFile,
    ) -> Result<GbuildMetadataLayer> {
        let path = transport.path.clone();
        if path.contains('/')
            || !path.ends_with("-original.json")
            || path == "-original.json"
            || transport.descriptor.media_type != PRESET_MEDIA_TYPE
        {
            anyhow::bail!("GBuild OCI preset must be a top-level *-original.json metadata layer");
        }
        Self::metadata_layer(outer, transport, &path, PRESET_MEDIA_TYPE)
    }

    fn payload_layer(
        outer: &OciDescriptor,
        transport: TransportPayloadDescriptor,
        member_names: Vec<String>,
        owned_paths: &mut HashSet<String>,
    ) -> Result<GbuildPayloadLayer> {
        if transport.media_type != PAYLOAD_MEDIA_TYPE || transport.uncompressed_size_bytes == 0 {
            anyhow::bail!(
                "GBuild OCI payload descriptor must use '{PAYLOAD_MEDIA_TYPE}' and positive sizes"
            );
        }
        Self::validate_transport_descriptor(&transport.as_descriptor(), "payload descriptor")?;
        Self::require_matching_descriptor(outer, &transport.as_descriptor(), "payload descriptor")?;
        if outer.annotations.is_some() {
            anyhow::bail!("GBuild OCI payload descriptors must not contain annotations");
        }
        if member_names.is_empty()
            || member_names
                .windows(2)
                .any(|members| members[0] >= members[1])
        {
            anyhow::bail!(
                "GBuild OCI transport-index members must be non-empty, unique, and in canonical order"
            );
        }

        let mut members = Vec::with_capacity(member_names.len());
        for member in member_names {
            let path = ArtifactPath::from_relative_path(
                Path::new(&member),
                &format!("GBuild OCI transport-index member '{member}'"),
            )?;
            if !owned_paths.insert(member) {
                anyhow::bail!(
                    "GBuild OCI transport-index members cannot overlap metadata or another payload"
                );
            }
            members.push(path);
        }

        Ok(GbuildPayloadLayer {
            descriptor: outer.clone(),
            members,
            uncompressed_size_bytes: transport.uncompressed_size_bytes,
        })
    }

    fn validate_transport_descriptor(
        descriptor: &TransportDescriptor,
        description: &str,
    ) -> Result<()> {
        if descriptor.size_bytes == 0 || !Self::is_sha256_digest(&descriptor.digest) {
            anyhow::bail!("GBuild OCI {description} must have a positive size and SHA-256 digest");
        }
        Ok(())
    }

    fn validate_outer_descriptor(descriptor: &OciDescriptor, description: &str) -> Result<()> {
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

    fn require_matching_descriptor(
        outer: &OciDescriptor,
        transport: &TransportDescriptor,
        description: &str,
    ) -> Result<()> {
        Self::validate_outer_descriptor(outer, description)?;
        let outer_size = u64::try_from(outer.size).context("OCI descriptor size is negative")?;
        if outer.media_type != transport.media_type
            || outer.digest != transport.digest
            || outer_size != transport.size_bytes
        {
            anyhow::bail!("GBuild OCI {description} does not match the outer manifest");
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
struct TransportIndex {
    version: u8,
    metadata: TransportMetadata,
    tokenizer: Option<TransportPayload>,
    partitions: Vec<TransportPartition>,
    runtime_assets: Option<TransportPayload>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportMetadata {
    manifest_json: TransportMetadataFile,
    manifest_capnp: TransportMetadataFile,
    preset: TransportMetadataFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportMetadataFile {
    path: String,
    descriptor: TransportDescriptor,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportDescriptor {
    media_type: String,
    digest: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportPayload {
    descriptor: TransportPayloadDescriptor,
    members: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportPartition {
    partition_id: u32,
    descriptor: TransportPayloadDescriptor,
    members: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportPayloadDescriptor {
    media_type: String,
    digest: String,
    size_bytes: u64,
    uncompressed_size_bytes: u64,
}

impl TransportPayloadDescriptor {
    fn as_descriptor(&self) -> TransportDescriptor {
        TransportDescriptor {
            media_type: self.media_type.clone(),
            digest: self.digest.clone(),
            size_bytes: self.size_bytes,
        }
    }
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

    fn descriptor(media_type: &str, digest: &str, size: u64) -> Value {
        json!({
            "media_type": media_type,
            "digest": digest,
            "size_bytes": size,
        })
    }

    fn payload_descriptor(digest: &str, size: u64, uncompressed_size: u64) -> Value {
        json!({
            "media_type": PAYLOAD_MEDIA_TYPE,
            "digest": digest,
            "size_bytes": size,
            "uncompressed_size_bytes": uncompressed_size,
        })
    }

    fn transport_index() -> Value {
        json!({
            "version": 1,
            "metadata": {
                "manifest_json": {
                    "path": "manifest.json",
                    "descriptor": descriptor(MANIFEST_JSON_MEDIA_TYPE, DIGEST_A, 10),
                },
                "manifest_capnp": {
                    "path": MANIFEST_CAPNP_FILE_NAME,
                    "descriptor": descriptor(MANIFEST_CAPNP_MEDIA_TYPE, DIGEST_B, 20),
                },
                "preset": {
                    "path": "llama-original.json",
                    "descriptor": descriptor(PRESET_MEDIA_TYPE, DIGEST_C, 30),
                },
            },
            "tokenizer": {
                "descriptor": payload_descriptor(DIGEST_D, 40, 400),
                "members": ["tokenizer/config.json"],
            },
            "partitions": [{
                "partition_id": 0,
                "descriptor": payload_descriptor(DIGEST_E, 50, 500),
                "members": ["program.0.gas", "program.0.weight"],
            }],
            "runtime_assets": null,
        })
    }

    fn outer_descriptor(media_type: &str, digest: &str, size: u64, title: Option<&str>) -> Value {
        let mut value = json!({
            "mediaType": media_type,
            "digest": digest,
            "size": size,
        });
        if let Some(title) = title {
            value["annotations"] = json!({"org.opencontainers.image.title": title});
        }
        value
    }

    fn manifest() -> OciImageManifest {
        serde_json::from_value(json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "artifactType": ARTIFACT_MEDIA_TYPE,
            "config": outer_descriptor(TRANSPORT_INDEX_MEDIA_TYPE, DIGEST_A, 100, None),
            "layers": [
                outer_descriptor(MANIFEST_JSON_MEDIA_TYPE, DIGEST_A, 10, Some("manifest.json")),
                outer_descriptor(MANIFEST_CAPNP_MEDIA_TYPE, DIGEST_B, 20, Some(MANIFEST_CAPNP_FILE_NAME)),
                outer_descriptor(PRESET_MEDIA_TYPE, DIGEST_C, 30, Some("llama-original.json")),
                outer_descriptor(PAYLOAD_MEDIA_TYPE, DIGEST_D, 40, None),
                outer_descriptor(PAYLOAD_MEDIA_TYPE, DIGEST_E, 50, None),
            ],
        }))
        .expect("valid OCI manifest fixture")
    }

    fn thin_config() -> Value {
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
            "runtime_assets": null,
        })
    }

    fn oras_manifest() -> OciImageManifest {
        serde_json::from_value(json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
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
            "annotations": {"org.opencontainers.image.created": "1970-01-01T00:00:00Z"},
        }))
        .expect("valid ORAS manifest fixture")
    }

    fn parse_contract(index: &Value, manifest: &OciImageManifest) -> Result<GbuildArtifact> {
        GbuildArtifact::from_manifest_and_transport_index(
            manifest,
            &serde_json::to_vec(index).expect("serialize transport-index fixture"),
        )
    }

    #[test]
    fn test_gbuild_artifact_maps_transport_index_to_outer_layers() {
        let artifact = parse_contract(&transport_index(), &manifest()).expect("valid contract");

        assert_eq!(artifact.metadata[0].path.as_str(), "manifest.json");
        assert_eq!(artifact.metadata[1].path.as_str(), MANIFEST_CAPNP_FILE_NAME);
        assert_eq!(artifact.metadata[2].path.as_str(), "llama-original.json");
        assert_eq!(artifact.payloads.len(), 2);
        assert_eq!(
            artifact.payloads[0].members[0].as_str(),
            "tokenizer/config.json"
        );
        assert_eq!(artifact.payloads[1].uncompressed_size_bytes, 500);
    }

    #[test]
    fn test_gbuild_artifact_maps_thin_config_to_oras_layers() {
        let artifact = GbuildArtifact::from_manifest_and_config(
            &oras_manifest(),
            &serde_json::to_vec(&thin_config()).expect("serialize config fixture"),
        )
        .expect("valid contract");

        assert_eq!(artifact.metadata[0].path.as_str(), "manifest.json");
        assert_eq!(artifact.metadata[1].path.as_str(), MANIFEST_CAPNP_FILE_NAME);
        assert_eq!(artifact.metadata[2].path.as_str(), "llama-original.json");
        assert_eq!(artifact.payloads.len(), 2);
        assert_eq!(
            artifact.payloads[0].descriptor.digest, DIGEST_D,
            "the OCI manifest owns the layer descriptor"
        );
        assert_eq!(artifact.payloads[1].members[1].as_str(), "program.0.weight");
    }

    #[test]
    fn test_gbuild_artifact_rejects_descriptor_drift() {
        let mut index = transport_index();
        index["partitions"][0]["descriptor"]["digest"] = json!(DIGEST_A);

        let error = parse_contract(&index, &manifest()).expect_err("descriptor drift must fail");
        assert!(error.to_string().contains("descriptor"));
    }

    #[test]
    fn test_gbuild_artifact_rejects_noncanonical_members_and_ownership() {
        let mut index = transport_index();
        index["partitions"][0]["members"] =
            json!(["program.0.weight", "program.0.gas", "tokenizer/config.json"]);

        let error = parse_contract(&index, &manifest()).expect_err("member drift must fail");
        assert!(error.to_string().contains("members"));

        index["partitions"][0]["members"] = json!(["tokenizer/config.json"]);
        let error = parse_contract(&index, &manifest()).expect_err("member overlap must fail");
        assert!(error.to_string().contains("overlap"));
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
