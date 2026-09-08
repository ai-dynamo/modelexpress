// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{archive_format::ArchiveFormat, path::ArtifactPath};
use anyhow::{Context, Result};
use oci_client::manifest::OciDescriptor;
use std::collections::HashSet;

pub const TITLE_ANNOTATION: &str = "org.opencontainers.image.title";
const CNCF_FILEPATH_ANNOTATION: &str = "org.cncf.model.filepath";

#[derive(Debug, Clone)]
pub struct LayerDownload {
    pub descriptor: OciDescriptor,
    pub kind: LayerDownloadKind,
}

#[derive(Debug, Clone)]
pub enum LayerDownloadKind {
    Raw { path: ArtifactPath },
    Archive { format: ArchiveFormat },
}

pub struct LayerDownloads {
    downloads: Vec<LayerDownload>,
}

impl LayerDownloads {
    pub fn from_layers(layers: &[OciDescriptor], ignore_weights: bool) -> Result<Self> {
        let mut seen_paths = HashSet::new();
        let mut downloads = Vec::new();

        for layer in layers {
            if ignore_weights && ArchiveFormat::is_archive_media_type(&layer.media_type) {
                tracing::debug!(
                    "Skipping OCI archive layer {} because ignore_weights=true",
                    layer.digest
                );
                continue;
            }

            if let Some(format) = ArchiveFormat::from_media_type(&layer.media_type)? {
                downloads.push(LayerDownload::archive(layer, format));
                continue;
            }

            let Some(path) = Self::raw_layer_path(layer, ignore_weights)? else {
                continue;
            };

            if !seen_paths.insert(path.clone()) {
                anyhow::bail!("Duplicate OCI artifact file path '{path}'");
            }

            downloads.push(LayerDownload::raw(layer, path));
        }

        Ok(Self { downloads })
    }

    pub fn as_slice(&self) -> &[LayerDownload] {
        &self.downloads
    }

    fn raw_layer_path(layer: &OciDescriptor, ignore_weights: bool) -> Result<Option<ArtifactPath>> {
        let title = LayerDownload::output_path_annotation(layer).with_context(|| {
            format!(
                "OCI layer {} is missing required '{TITLE_ANNOTATION}' or '{CNCF_FILEPATH_ANNOTATION}' annotation",
                layer.digest
            )
        })?;
        let path = ArtifactPath::from_title(title)?;

        if path.is_skipped(ignore_weights) {
            tracing::debug!("Skipping OCI artifact file: {path}");
            return Ok(None);
        }

        Ok(Some(path))
    }
}

impl LayerDownload {
    fn archive(layer: &OciDescriptor, format: ArchiveFormat) -> Self {
        Self {
            descriptor: layer.clone(),
            kind: LayerDownloadKind::Archive { format },
        }
    }

    fn raw(layer: &OciDescriptor, path: ArtifactPath) -> Self {
        Self {
            descriptor: layer.clone(),
            kind: LayerDownloadKind::Raw { path },
        }
    }

    fn output_path_annotation(layer: &OciDescriptor) -> Option<&str> {
        layer.annotations.as_ref().and_then(|annotations| {
            annotations
                .get(TITLE_ANNOTATION)
                .or_else(|| annotations.get(CNCF_FILEPATH_ANNOTATION))
                .map(String::as_str)
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::archive_format::ArchiveFormat;
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    fn digest_bytes(bytes: &[u8]) -> String {
        format!("sha256:{:x}", Sha256::digest(bytes))
    }

    fn descriptor(title: Option<&str>, bytes: &[u8]) -> OciDescriptor {
        descriptor_with_media_type("application/octet-stream", title, bytes)
    }

    fn descriptor_with_media_type(
        media_type: &str,
        title: Option<&str>,
        bytes: &[u8],
    ) -> OciDescriptor {
        let annotations =
            title.map(|title| BTreeMap::from([(TITLE_ANNOTATION.to_string(), title.to_string())]));

        OciDescriptor {
            media_type: media_type.to_string(),
            digest: digest_bytes(bytes),
            size: bytes.len() as i64,
            urls: None,
            annotations,
        }
    }

    fn raw_download_path(download: &LayerDownload) -> &str {
        match &download.kind {
            LayerDownloadKind::Raw { path } => path.as_str(),
            LayerDownloadKind::Archive { .. } => panic!("expected raw download"),
        }
    }

    #[test]
    fn test_prepare_layers_requires_title_and_rejects_duplicates() {
        let missing_title = vec![descriptor(None, b"config")];
        let Err(err) = LayerDownloads::from_layers(&missing_title, false) else {
            panic!("missing title should fail");
        };
        assert!(err.to_string().contains(TITLE_ANNOTATION));

        let duplicate = vec![
            descriptor(Some("config.json"), b"one"),
            descriptor(Some("config.json"), b"two"),
        ];
        let Err(err) = LayerDownloads::from_layers(&duplicate, false) else {
            panic!("duplicate path should fail");
        };
        assert!(err.to_string().contains("Duplicate OCI artifact file path"));
    }

    #[test]
    fn test_prepare_layers_applies_ignore_rules() {
        let layers = vec![
            descriptor(Some("README.md"), b"readme"),
            descriptor(Some("README.md"), b"duplicate ignored readme"),
            descriptor(Some(".gitattributes"), b"dotfile"),
            descriptor(Some("diagram.png"), b"image"),
            descriptor(Some("model.safetensors"), b"weights"),
            descriptor(Some("config.json"), b"config"),
        ];

        let without_weights =
            LayerDownloads::from_layers(&layers, true).expect("ignore_weights should succeed");
        assert_eq!(without_weights.as_slice().len(), 1);
        assert_eq!(
            raw_download_path(&without_weights.as_slice()[0]),
            "config.json"
        );

        let with_weights =
            LayerDownloads::from_layers(&layers, false).expect("download selection should succeed");
        assert_eq!(with_weights.as_slice().len(), 2);
        assert_eq!(
            raw_download_path(&with_weights.as_slice()[0]),
            "model.safetensors"
        );
        assert_eq!(
            raw_download_path(&with_weights.as_slice()[1]),
            "config.json"
        );
    }

    #[test]
    fn test_prepare_layers_accepts_archive_layers_without_title() {
        let archive = descriptor_with_media_type(
            "application/vnd.kitops.modelkit.model.v1.tar",
            None,
            b"archive",
        );
        let downloads =
            LayerDownloads::from_layers(&[archive], false).expect("archive should select");
        assert_eq!(downloads.as_slice().len(), 1);
        match &downloads.as_slice()[0].kind {
            LayerDownloadKind::Archive { format } => {
                assert_eq!(*format, ArchiveFormat::Tar);
            }
            LayerDownloadKind::Raw { .. } => panic!("expected archive download"),
        }

        let archive = descriptor_with_media_type(
            "application/vnd.oci.image.layer.v1.tar+zstd",
            Some("part-0"),
            b"archive",
        );
        let downloads =
            LayerDownloads::from_layers(&[archive], false).expect("archive should select");
        match &downloads.as_slice()[0].kind {
            LayerDownloadKind::Archive { format } => {
                assert_eq!(*format, ArchiveFormat::TarZstd);
            }
            LayerDownloadKind::Raw { .. } => panic!("expected archive download"),
        }
    }

    #[test]
    fn test_prepare_layers_skips_archives_when_ignoring_weights() {
        let layers = vec![
            descriptor_with_media_type(
                "application/vnd.kitops.modelkit.model.v1.tar",
                None,
                b"archive",
            ),
            descriptor_with_media_type(
                "application/vnd.oci.image.layer.v1.tar+gzip",
                None,
                b"unsupported archive",
            ),
            descriptor(Some("config.json"), b"config"),
        ];

        let downloads =
            LayerDownloads::from_layers(&layers, true).expect("download selection should succeed");

        assert_eq!(downloads.as_slice().len(), 1);
        assert_eq!(raw_download_path(&downloads.as_slice()[0]), "config.json");

        let unsupported = descriptor_with_media_type(
            "application/vnd.oci.image.layer.v1.tar+gzip",
            None,
            b"unsupported archive",
        );
        let Err(err) = LayerDownloads::from_layers(&[unsupported], false) else {
            panic!("unsupported archive should fail without ignore_weights");
        };
        assert!(err.to_string().contains("not supported"));
    }
}
