// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    cache_entry::StagingCacheEntry,
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
    auth: RegistryAuth,
    client: Client,
}

impl<'a> Downloader<'a> {
    pub fn new(original_ref: &'a str, reference: &'a OciReference) -> Self {
        Self {
            original_ref,
            reference,
            auth: registry_auth::from_env(),
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

        let manifest = self.pull_image_manifest().await?;
        if manifest.layers.is_empty() {
            anyhow::bail!(
                "OCI artifact '{}' contains no layer descriptors",
                self.original_ref
            );
        }

        let downloads = LayerDownloads::from_layers(&manifest.layers, ignore_weights)?;
        self.download_layers(staging_entry, &staging_files, downloads.as_slice())
            .await?;
        self.download_manifest_json(&manifest, &staging_files)
            .await?;

        Ok(())
    }

    async fn pull_image_manifest(&self) -> Result<OciImageManifest> {
        let (manifest, _) = self
            .client
            .pull_manifest(self.reference.as_client_reference(), &self.auth)
            .await
            .with_context(|| format!("Failed to pull OCI manifest for '{}'", self.original_ref))?;
        Self::image_manifest(manifest)
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

        if tokio::fs::try_exists(&blob_root).await.with_context(|| {
            format!("Failed to inspect OCI temporary blob directory {blob_root:?}")
        })? {
            tokio::fs::remove_dir_all(&blob_root)
                .await
                .with_context(|| {
                    format!("Failed to remove OCI temporary blob directory {blob_root:?}")
                })?;
        }

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

        Ok(())
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
