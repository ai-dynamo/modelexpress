// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::path::ArtifactPath;
use anyhow::Result;
use oci_client::manifest::{OciDescriptor, OciImageManifest};
use serde::Deserialize;

pub const ARTIFACT_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.full-compile.v1";
pub const TRANSPORT_INDEX_MEDIA_TYPE: &str =
    "application/vnd.groq.gbuild.full-compile.transport.v1+json";
pub const MANIFEST_JSON_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.manifest.v1+json";
pub const MANIFEST_CAPNP_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.manifest.v1+capnp";
pub const PRESET_MEDIA_TYPE: &str = "application/vnd.groq.gbuild.preset.v1+json";
pub const PAYLOAD_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+zstd";

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
    pub fn from_manifest_and_transport_index(
        _manifest: &OciImageManifest,
        _transport_index: &[u8],
    ) -> Result<Self> {
        todo!("validate the GBuild OCI transport contract")
    }
}

pub fn is_gbuild_artifact(manifest: &OciImageManifest) -> bool {
    manifest.artifact_type.as_deref() == Some(ARTIFACT_MEDIA_TYPE)
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
                    "path": "manifest.capnp.bin",
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
                outer_descriptor(MANIFEST_CAPNP_MEDIA_TYPE, DIGEST_B, 20, Some("manifest.capnp.bin")),
                outer_descriptor(PRESET_MEDIA_TYPE, DIGEST_C, 30, Some("llama-original.json")),
                outer_descriptor(PAYLOAD_MEDIA_TYPE, DIGEST_D, 40, None),
                outer_descriptor(PAYLOAD_MEDIA_TYPE, DIGEST_E, 50, None),
            ],
        }))
        .expect("valid OCI manifest fixture")
    }

    fn parse_contract(index: &Value, manifest: &OciImageManifest) -> Result<GbuildArtifact> {
        GbuildArtifact::from_manifest_and_transport_index(
            manifest,
            &serde_json::to_vec(index).expect("serialize transport-index fixture"),
        )
    }

    #[test]
    fn test_gbuild_artifact_maps_transport_index_to_outer_layers() {
        // TODO: implement
        let artifact = parse_contract(&transport_index(), &manifest()).expect("valid contract");

        assert_eq!(artifact.metadata[0].path.as_str(), "manifest.json");
        assert_eq!(artifact.metadata[1].path.as_str(), "manifest.capnp.bin");
        assert_eq!(artifact.metadata[2].path.as_str(), "llama-original.json");
        assert_eq!(artifact.payloads.len(), 2);
        assert_eq!(
            artifact.payloads[0].members[0].as_str(),
            "tokenizer/config.json"
        );
        assert_eq!(artifact.payloads[1].uncompressed_size_bytes, 500);
    }

    #[test]
    fn test_gbuild_artifact_rejects_descriptor_drift() {
        // TODO: implement
        let mut index = transport_index();
        index["partitions"][0]["descriptor"]["digest"] = json!(DIGEST_A);

        let error = parse_contract(&index, &manifest()).expect_err("descriptor drift must fail");
        assert!(error.to_string().contains("descriptor"));
    }

    #[test]
    fn test_gbuild_artifact_rejects_noncanonical_members_and_ownership() {
        // TODO: implement
        let mut index = transport_index();
        index["partitions"][0]["members"] =
            json!(["program.0.weight", "program.0.gas", "tokenizer/config.json"]);

        let error = parse_contract(&index, &manifest()).expect_err("member drift must fail");
        assert!(error.to_string().contains("members"));
    }
}
