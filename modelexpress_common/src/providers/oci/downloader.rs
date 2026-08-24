// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    archive_format::ArchiveFormat,
    cache_entry::StagingCacheEntry,
    gbuild::{GbuildArtifact, is_gbuild_artifact},
    layer_download::{LayerDownload, LayerDownloadKind, LayerDownloads},
    path::ArtifactPath,
    reference::OciReference,
    registry_auth,
};
use anyhow::{Context, Result};
use oci_client::{
    Client,
    client::{ClientConfig, ClientProtocol},
    manifest::{OciDescriptor, OciImageManifest, OciManifest},
    secrets::RegistryAuth,
};
use std::path::{Path, PathBuf};
use tracing::info;

const MANIFEST_FILE_NAME: &str = "manifest.json";

pub struct Downloader<'a> {
    original_ref: &'a str,
    reference: &'a OciReference,
    client: Client,
}

impl<'a> Downloader<'a> {
    pub fn new(original_ref: &'a str, reference: &'a OciReference) -> Self {
        Self {
            original_ref,
            reference,
            client: Self::client_for_reference(reference),
        }
    }

    pub async fn download_to_staging(
        &self,
        staging_entry: &StagingCacheEntry,
        ignore_weights: bool,
    ) -> Result<()> {
        let staging_files = staging_entry.files_dir();
        tokio::fs::create_dir_all(&staging_files)
            .await
            .with_context(|| format!("Failed to create OCI staging directory {staging_files:?}"))?;

        let auth = registry_auth::resolve(self.reference.registry_endpoint()).await?;
        let manifest = self.pull_image_manifest(&auth).await?;
        if manifest.layers.is_empty() {
            anyhow::bail!(
                "OCI artifact '{}' contains no layer descriptors",
                self.original_ref
            );
        }

        if is_gbuild_artifact(&manifest) {
            if self.reference.digest().is_none() {
                anyhow::bail!(
                    "GBuild OCI artifact '{}' must use an immutable digest reference",
                    self.original_ref
                );
            }
            if ignore_weights {
                anyhow::bail!(
                    "GBuild OCI artifacts must be materialized completely; ignore_weights is not supported"
                );
            }
            return self
                .download_gbuild_artifact(staging_entry, &staging_files, &manifest)
                .await;
        }

        let downloads = LayerDownloads::from_layers(&manifest.layers, ignore_weights)?;
        self.download_layers(staging_entry, &staging_files, downloads.as_slice())
            .await?;
        self.download_manifest_json(&manifest, &staging_files)
            .await?;

        Ok(())
    }

    async fn pull_image_manifest(&self, auth: &RegistryAuth) -> Result<OciImageManifest> {
        let (manifest, _) = self
            .client
            .pull_manifest(self.reference.as_client_reference(), auth)
            .await
            .with_context(|| format!("Failed to pull OCI manifest for '{}'", self.original_ref))?;
        Self::image_manifest(manifest)
    }

    async fn download_gbuild_artifact(
        &self,
        staging_entry: &StagingCacheEntry,
        staging_files: &Path,
        manifest: &OciImageManifest,
    ) -> Result<()> {
        let blob_root = staging_entry.blob_root();
        let transport_index_path = blob_root.join("transport-index.json");
        self.pull_blob_to_file(
            &manifest.config,
            &transport_index_path,
            "GBuild OCI transport index",
        )
        .await?;
        let transport_index = tokio::fs::read(&transport_index_path)
            .await
            .with_context(|| {
                format!("Failed to read GBuild OCI transport index {transport_index_path:?}")
            })?;
        tokio::fs::remove_file(&transport_index_path)
            .await
            .with_context(|| {
                format!("Failed to remove GBuild OCI transport index {transport_index_path:?}")
            })?;
        let artifact =
            GbuildArtifact::from_manifest_and_transport_index(manifest, &transport_index)?;

        for metadata in artifact.metadata {
            self.pull_blob_to_file(
                &metadata.descriptor,
                &staging_files.join(metadata.path.as_path()),
                "GBuild OCI metadata file",
            )
            .await?;
        }

        for payload in artifact.payloads {
            let blob_path = blob_root.join(payload.descriptor.digest.replace(':', "-"));
            self.pull_blob_to_file(&payload.descriptor, &blob_path, "GBuild OCI payload blob")
                .await?;
            ArchiveFormat::TarZstd
                .extract_gbuild_payload(
                    &blob_path,
                    staging_files,
                    &payload.members,
                    payload.uncompressed_size_bytes,
                )
                .with_context(|| {
                    format!(
                        "Failed to extract GBuild OCI payload {}",
                        payload.descriptor.digest
                    )
                })?;
            tokio::fs::remove_file(&blob_path).await.with_context(|| {
                format!("Failed to remove GBuild OCI payload blob {blob_path:?}")
            })?;
        }

        Self::remove_blob_root(&blob_root).await
    }

    async fn download_manifest_json(
        &self,
        manifest: &OciImageManifest,
        staging_files: &Path,
    ) -> Result<()> {
        let output_path = staging_files.join(MANIFEST_FILE_NAME);
        // The model artifact wins if it already provided manifest.json as a
        // layer file or archive member; otherwise expose the OCI config blob as
        // manifest.json so gbuild-produced models can carry model config there.
        if tokio::fs::try_exists(&output_path)
            .await
            .with_context(|| format!("Failed to inspect OCI manifest.json {output_path:?}"))?
        {
            return Ok(());
        }

        self.pull_blob_to_file(&manifest.config, &output_path, "OCI manifest.json")
            .await
            .with_context(|| {
                format!(
                    "Failed to download OCI config blob {} as manifest.json",
                    manifest.config.digest
                )
            })
    }

    async fn download_layers(
        &self,
        staging_entry: &StagingCacheEntry,
        staging_files: &Path,
        downloads: &[LayerDownload],
    ) -> Result<usize> {
        let mut file_count = 0usize;
        let blob_root = staging_entry.blob_root();

        for download in downloads {
            match &download.kind {
                LayerDownloadKind::Raw { path } => {
                    self.download_raw_blob(download, staging_files, path)
                        .await?;
                    info!(
                        "Downloaded OCI blob {} for file '{}'",
                        download.descriptor.digest, path
                    );
                    file_count = file_count.saturating_add(1);
                }
                LayerDownloadKind::Archive { format } => {
                    let path = self.download_archive_blob(download, &blob_root).await?;
                    // Archive member paths define the artifact layout. Layer title
                    // annotations are labels/debug metadata unless a manifest schema
                    // explicitly assigns placement semantics.
                    let extracted_files =
                        format.extract_blob(&path, staging_files).with_context(|| {
                            format!(
                                "Failed to extract OCI archive blob {}",
                                download.descriptor.digest
                            )
                        })?;

                    tokio::fs::remove_file(&path).await.with_context(|| {
                        format!("Failed to remove OCI temporary blob file {path:?}")
                    })?;

                    file_count = file_count.saturating_add(extracted_files.len());
                }
            }
        }

        Self::remove_blob_root(&blob_root).await?;

        Ok(file_count)
    }

    async fn download_raw_blob(
        &self,
        download: &LayerDownload,
        staging_files: &Path,
        relative_path: &ArtifactPath,
    ) -> Result<()> {
        let output_path = staging_files.join(relative_path.as_path());
        self.pull_blob_to_file(&download.descriptor, &output_path, "OCI output file")
            .await
            .with_context(|| {
                format!(
                    "Failed to download OCI blob {} for file '{}'",
                    download.descriptor.digest, relative_path
                )
            })?;

        Ok(())
    }

    async fn download_archive_blob(
        &self,
        download: &LayerDownload,
        blob_root: &Path,
    ) -> Result<PathBuf> {
        let path = blob_root.join(download.descriptor.digest.replace(':', "-"));
        self.pull_blob_to_file(&download.descriptor, &path, "OCI archive blob")
            .await?;

        Ok(path)
    }

    async fn pull_blob_to_file(
        &self,
        descriptor: &OciDescriptor,
        output_path: &Path,
        description: &str,
    ) -> Result<()> {
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create {description} directory {parent:?}"))?;
        }

        let mut output = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)
            .await
            .with_context(|| format!("Failed to create {description} {output_path:?}"))?;

        info!(
            "Downloading OCI blob {} to {}",
            descriptor.digest,
            output_path.display()
        );

        self.client
            .pull_blob(
                self.reference.as_client_reference(),
                descriptor,
                &mut output,
            )
            .await
            .with_context(|| {
                format!(
                    "Failed to download OCI blob {} to {}",
                    descriptor.digest,
                    output_path.display()
                )
            })?;
        output
            .sync_all()
            .await
            .with_context(|| format!("Failed to sync {description} {output_path:?}"))?;

        let actual_size = output
            .metadata()
            .await
            .with_context(|| format!("Failed to inspect {description} {output_path:?}"))?
            .len();
        let expected_size = u64::try_from(descriptor.size)
            .with_context(|| format!("{description} has a negative OCI descriptor size"))?;
        if actual_size != expected_size {
            anyhow::bail!(
                "Downloaded {description} has {actual_size} bytes; expected {expected_size}"
            );
        }

        Ok(())
    }

    async fn remove_blob_root(blob_root: &Path) -> Result<()> {
        if !tokio::fs::try_exists(blob_root).await.with_context(|| {
            format!("Failed to inspect OCI temporary blob directory {blob_root:?}")
        })? {
            return Ok(());
        }
        tokio::fs::remove_dir_all(blob_root)
            .await
            .with_context(|| format!("Failed to remove OCI temporary blob directory {blob_root:?}"))
    }

    fn client_for_reference(reference: &OciReference) -> Client {
        let mut config = ClientConfig::default();
        let registry = reference.registry_endpoint();

        if Self::is_loopback_registry(registry) {
            config.protocol = ClientProtocol::HttpsExcept(vec![registry.to_string()]);
        }

        Client::new(config)
    }

    fn is_loopback_registry(registry: &str) -> bool {
        let host = registry
            .split_once(':')
            .map_or(registry, |(host, _)| host)
            .trim_matches(['[', ']']);

        host == "localhost" || host == "127.0.0.1" || host == "::1"
    }

    fn image_manifest(manifest: OciManifest) -> Result<OciImageManifest> {
        match manifest {
            OciManifest::Image(manifest) => Ok(manifest),
            OciManifest::ImageIndex(_) => {
                anyhow::bail!(
                    "OCI image index manifests are not supported for model artifacts; use an OCI image manifest"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::{
        OciProvider,
        cache_entry::{CACHE_ROOT_DIR_NAME, TMP_DIR_NAME},
        layer_download::TITLE_ANNOTATION,
    };
    use super::MANIFEST_FILE_NAME;
    use crate::providers::ModelProviderTrait;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::fs;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn digest_bytes(bytes: &[u8]) -> String {
        format!("sha256:{:x}", Sha256::digest(bytes))
    }

    fn tar_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut bytes);
            for (path, contents) in entries {
                let mut header = tar::Header::new_gnu();
                header.set_size(contents.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder
                    .append_data(&mut header, path, *contents)
                    .expect("append tar entry");
            }
            builder.finish().expect("finish tar");
        }
        bytes
    }

    #[tokio::test]
    async fn test_mock_registry_download_publishes_final_cache_entry() {
        let cache_dir = TempDir::new().expect("temp cache");
        let server = MockServer::start().await;
        let registry = server
            .uri()
            .strip_prefix("http://")
            .expect("wiremock should use http")
            .to_string();
        let repo = "team/model";
        let config = b"{}";
        let artifact_manifest = br#"{"artifact":true}"#;
        let tokenizer = b"{\"tokenizer\":true}";
        let weights = b"weights";
        let config_digest = digest_bytes(config);
        let artifact_manifest_digest = digest_bytes(artifact_manifest);
        let tokenizer_digest = digest_bytes(tokenizer);
        let weights_digest = digest_bytes(weights);

        let manifest = json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": 2,
                "digest": digest_bytes(b"{}")
            },
            "layers": [
                {
                    "mediaType": "application/octet-stream",
                    "size": config.len(),
                    "digest": config_digest,
                    "annotations": { TITLE_ANNOTATION: "config.json" }
                },
                {
                    "mediaType": "application/octet-stream",
                    "size": tokenizer.len(),
                    "digest": tokenizer_digest,
                    "annotations": { TITLE_ANNOTATION: "tokenizer.json" }
                },
                {
                    "mediaType": "application/octet-stream",
                    "size": artifact_manifest.len(),
                    "digest": artifact_manifest_digest,
                    "annotations": { TITLE_ANNOTATION: "manifest.json" }
                },
                {
                    "mediaType": "application/octet-stream",
                    "size": weights.len(),
                    "digest": weights_digest,
                    "annotations": { TITLE_ANNOTATION: "model.safetensors" }
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/manifests/v1")))
            .respond_with(ResponseTemplate::new(200).set_body_json(manifest))
            .mount(&server)
            .await;

        for (digest, body) in [
            (config_digest.as_str(), config.as_slice()),
            (
                artifact_manifest_digest.as_str(),
                artifact_manifest.as_slice(),
            ),
            (tokenizer_digest.as_str(), tokenizer.as_slice()),
            (weights_digest.as_str(), weights.as_slice()),
        ] {
            Mock::given(method("GET"))
                .and(path(format!("/v2/{repo}/blobs/{digest}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(body.to_vec()))
                .mount(&server)
                .await;
        }

        let model_ref = format!("{registry}/{repo}:v1");
        let path = OciProvider
            .download_model(&model_ref, Some(cache_dir.path().to_path_buf()), true)
            .await
            .expect("download should succeed");

        assert!(path.join("config.json").is_file());
        assert!(path.join("tokenizer.json").is_file());
        assert_eq!(
            fs::read(path.join(MANIFEST_FILE_NAME)).expect("read artifact manifest.json"),
            artifact_manifest
        );
        assert!(!path.join("model.safetensors").exists());
        assert!(
            !path
                .parent()
                .expect("files directory has a cache entry parent")
                .join("metadata")
                .exists()
        );

        let oci_root = cache_dir.path().join(CACHE_ROOT_DIR_NAME);
        let tmp_root = oci_root.join(TMP_DIR_NAME);
        assert!(!tmp_root.exists() || fs::read_dir(&tmp_root).expect("read tmp").next().is_none());
    }

    #[tokio::test]
    async fn test_mock_registry_download_extracts_archive_layer() {
        let cache_dir = TempDir::new().expect("temp cache");
        let server = MockServer::start().await;
        let registry = server
            .uri()
            .strip_prefix("http://")
            .expect("wiremock should use http")
            .to_string();
        let repo = "team/archive-model";
        let manifest_json = br#"{"build":{"id":"archive-model"}}"#;
        let archive = tar_bytes(&[
            ("config.json", b"{}"),
            ("model.safetensors", b"weights"),
            ("README.md", b"readme"),
        ]);
        let manifest_digest = digest_bytes(manifest_json);
        let archive_digest = digest_bytes(&archive);

        let manifest = json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.kitops.modelkit.config.v1+json",
                "size": manifest_json.len(),
                "digest": manifest_digest
            },
            "layers": [
                {
                    "mediaType": "application/vnd.kitops.modelkit.model.v1.tar",
                    "size": archive.len(),
                    "digest": archive_digest,
                    "annotations": { TITLE_ANNOTATION: "part-0" }
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/manifests/v1")))
            .respond_with(ResponseTemplate::new(200).set_body_json(manifest))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/blobs/{manifest_digest}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(manifest_json))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/blobs/{archive_digest}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(archive))
            .mount(&server)
            .await;

        let model_ref = format!("{registry}/{repo}:v1");
        let path = OciProvider
            .download_model(&model_ref, Some(cache_dir.path().to_path_buf()), false)
            .await
            .expect("download should succeed");

        assert_eq!(
            fs::read(path.join(MANIFEST_FILE_NAME)).expect("read artifact manifest.json"),
            manifest_json
        );
        assert!(path.join("config.json").is_file());
        assert!(path.join("model.safetensors").is_file());
        assert!(!path.join("part-0/config.json").exists());
        assert!(!path.join("README.md").exists());
    }

    #[tokio::test]
    async fn test_mock_registry_materializes_digest_pinned_gbuild_artifact() {
        let cache_dir = TempDir::new().expect("temp cache");
        let server = MockServer::start().await;
        let registry = server
            .uri()
            .strip_prefix("http://")
            .expect("wiremock should use http")
            .to_string();
        let repo = "team/gbuild";
        let manifest_json = br#"{"build":{"id":"gbuild"}}"#;
        let manifest_capnp = b"capnp";
        let preset = br#"{"model":"llama"}"#;
        let tar = tar_bytes(&[("README.md", b"readme"), ("program.0.gas", b"gas")]);
        let payload = zstd::stream::encode_all(tar.as_slice(), 3).expect("compress payload");
        let manifest_json_digest = digest_bytes(manifest_json);
        let manifest_capnp_digest = digest_bytes(manifest_capnp);
        let preset_digest = digest_bytes(preset);
        let payload_digest = digest_bytes(&payload);
        let transport_index = serde_json::to_vec(&json!({
            "version": 1,
            "metadata": {
                "manifest_json": {
                    "path": "manifest.json",
                    "descriptor": {
                        "media_type": "application/vnd.groq.gbuild.manifest.v1+json",
                        "digest": manifest_json_digest,
                        "size_bytes": manifest_json.len(),
                    },
                },
                "manifest_capnp": {
                    "path": "manifest.capnp.bin",
                    "descriptor": {
                        "media_type": "application/vnd.groq.gbuild.manifest.v1+capnp",
                        "digest": manifest_capnp_digest,
                        "size_bytes": manifest_capnp.len(),
                    },
                },
                "preset": {
                    "path": "llama-original.json",
                    "descriptor": {
                        "media_type": "application/vnd.groq.gbuild.preset.v1+json",
                        "digest": preset_digest,
                        "size_bytes": preset.len(),
                    },
                },
            },
            "tokenizer": null,
            "partitions": [{
                "partition_id": 0,
                "descriptor": {
                    "media_type": "application/vnd.oci.image.layer.v1.tar+zstd",
                    "digest": payload_digest,
                    "size_bytes": payload.len(),
                    "uncompressed_size_bytes": tar.len(),
                },
                "members": ["README.md", "program.0.gas"],
            }],
            "runtime_assets": null,
        }))
        .expect("serialize transport index");
        let transport_index_digest = digest_bytes(&transport_index);
        let outer_manifest = serde_json::to_vec(&json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "artifactType": "application/vnd.groq.gbuild.full-compile.v1",
            "config": {
                "mediaType": "application/vnd.groq.gbuild.full-compile.transport.v1+json",
                "size": transport_index.len(),
                "digest": transport_index_digest,
            },
            "layers": [
                {
                    "mediaType": "application/vnd.groq.gbuild.manifest.v1+json",
                    "size": manifest_json.len(),
                    "digest": manifest_json_digest,
                    "annotations": { TITLE_ANNOTATION: "manifest.json" },
                },
                {
                    "mediaType": "application/vnd.groq.gbuild.manifest.v1+capnp",
                    "size": manifest_capnp.len(),
                    "digest": manifest_capnp_digest,
                    "annotations": { TITLE_ANNOTATION: "manifest.capnp.bin" },
                },
                {
                    "mediaType": "application/vnd.groq.gbuild.preset.v1+json",
                    "size": preset.len(),
                    "digest": preset_digest,
                    "annotations": { TITLE_ANNOTATION: "llama-original.json" },
                },
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+zstd",
                    "size": payload.len(),
                    "digest": payload_digest,
                },
            ],
        }))
        .expect("serialize outer manifest");
        let outer_manifest_digest = digest_bytes(&outer_manifest);

        Mock::given(method("GET"))
            .and(path(format!(
                "/v2/{repo}/manifests/{outer_manifest_digest}"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(outer_manifest.clone()))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/manifests/latest")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(outer_manifest))
            .mount(&server)
            .await;

        for (digest, body) in [
            (transport_index_digest.as_str(), transport_index.as_slice()),
            (manifest_json_digest.as_str(), manifest_json.as_slice()),
            (manifest_capnp_digest.as_str(), manifest_capnp.as_slice()),
            (preset_digest.as_str(), preset.as_slice()),
            (payload_digest.as_str(), payload.as_slice()),
        ] {
            Mock::given(method("GET"))
                .and(path(format!("/v2/{repo}/blobs/{digest}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(body.to_vec()))
                .mount(&server)
                .await;
        }

        let model_ref = format!("{registry}/{repo}@{outer_manifest_digest}");
        let path = OciProvider
            .download_model(&model_ref, Some(cache_dir.path().to_path_buf()), false)
            .await
            .expect("GBuild artifact should materialize");

        assert_eq!(
            fs::read(path.join(MANIFEST_FILE_NAME)).expect("read runtime manifest"),
            manifest_json
        );
        assert_eq!(
            fs::read(path.join("manifest.capnp.bin")).expect("read Cap'n Proto manifest"),
            manifest_capnp
        );
        assert_eq!(
            fs::read(path.join("llama-original.json")).expect("read preset"),
            preset
        );
        assert_eq!(
            fs::read(path.join("README.md")).expect("read filtered filename"),
            b"readme"
        );
        assert_eq!(
            fs::read(path.join("program.0.gas")).expect("read program"),
            b"gas"
        );
        assert!(!path.join("transport-index.json").exists());

        let tag_error = OciProvider
            .download_model(
                &format!("{registry}/{repo}:latest"),
                Some(cache_dir.path().to_path_buf()),
                false,
            )
            .await
            .expect_err("GBuild tag reference must fail");
        assert!(tag_error.to_string().contains("immutable digest reference"));
    }

    #[tokio::test]
    async fn test_mock_registry_downloads_manifest_after_filtering_layers() {
        let cache_dir = TempDir::new().expect("temp cache");
        let server = MockServer::start().await;
        let registry = server
            .uri()
            .strip_prefix("http://")
            .expect("wiremock should use http")
            .to_string();
        let repo = "team/archive-model";
        let manifest_json = br#"{"build":{"id":"manifest-only"}}"#;
        let manifest_digest = digest_bytes(manifest_json);

        let manifest = json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.kitops.modelkit.config.v1+json",
                "size": manifest_json.len(),
                "digest": manifest_digest
            },
            "layers": [
                {
                    "mediaType": "application/vnd.kitops.modelkit.model.v1.tar",
                    "size": 7,
                    "digest": digest_bytes(b"archive")
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/manifests/v1")))
            .respond_with(ResponseTemplate::new(200).set_body_json(manifest))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path(format!("/v2/{repo}/blobs/{manifest_digest}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(manifest_json))
            .mount(&server)
            .await;

        let model_ref = format!("{registry}/{repo}:v1");
        let path = OciProvider
            .download_model(&model_ref, Some(cache_dir.path().to_path_buf()), true)
            .await
            .expect("manifest-only download should publish manifest");

        assert_eq!(
            fs::read(path.join(MANIFEST_FILE_NAME)).expect("read artifact manifest.json"),
            manifest_json
        );
    }
}
