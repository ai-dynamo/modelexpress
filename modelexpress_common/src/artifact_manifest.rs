// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic local manifests for file-backed artifact sources.

use crate::grpc::p2p::{
    ArtifactManifest as ProtoArtifactManifest, ArtifactManifestChunk as ProtoArtifactManifestChunk,
    ArtifactManifestFile as ProtoArtifactManifestFile, ArtifactSourceMetadata,
    GetArtifactManifestChunksResponse, GetArtifactManifestHeaderResponse,
};
use anyhow::{Context, Result, anyhow, bail};
use crc32c::{crc32c, crc32c_append};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::{Component, Path, PathBuf},
};

pub const ARTIFACT_MANIFEST_VERSION: u32 = 1;
pub const MAX_ARTIFACT_TRANSFER_CHUNK_SIZE: u64 = 4 * 1024 * 1024 * 1024;
// Number of chunk metadata records per GetArtifactManifestChunks response.
// This is not the artifact byte chunk size; 1024 keeps metadata responses
// bounded while avoiding one RPC per transfer chunk.
const ARTIFACT_CHUNK_METADATA_PAGE_SIZE: u32 = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub manifest_version: u32,
    pub mx_source_type: i32,
    pub chunk_size: u64,
    pub files: Vec<ArtifactManifestFile>,
    pub chunks: Vec<ArtifactManifestChunk>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifestFile {
    pub file_index: u32,
    pub path: String,
    pub size: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifestChunk {
    pub chunk_index: u32,
    pub file_index: u32,
    pub file_offset: u64,
    pub length: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedArtifactManifest {
    pub artifact_id: String,
    pub manifest: ArtifactManifest,
}

impl ArtifactManifest {
    pub fn from_directory(
        root: impl AsRef<Path>,
        chunk_size: u64,
        mx_source_type: i32,
    ) -> Result<Self> {
        if chunk_size == 0 {
            bail!("artifact manifest chunk_size must be greater than zero");
        }
        if chunk_size > MAX_ARTIFACT_TRANSFER_CHUNK_SIZE {
            bail!(
                "artifact manifest chunk_size {} exceeds maximum {}",
                chunk_size,
                MAX_ARTIFACT_TRANSFER_CHUNK_SIZE
            );
        }

        let root = root
            .as_ref()
            .canonicalize()
            .with_context(|| format!("failed to canonicalize artifact root {:?}", root.as_ref()))?;
        if !root.is_dir() {
            bail!("artifact root is not a directory: {}", root.display());
        }

        let mut paths = Vec::new();
        collect_regular_files(&root, &root, &mut paths)?;
        let mut manifest_files = paths
            .into_iter()
            .map(|path| manifest_path(&path).map(|manifest_path| (manifest_path, path)))
            .collect::<Result<Vec<_>>>()?;
        manifest_files.sort_by(|left, right| left.0.cmp(&right.0));

        let mut files = Vec::with_capacity(manifest_files.len());
        let mut chunks = Vec::new();
        for (file_index, (manifest_path, path)) in manifest_files.into_iter().enumerate() {
            let metadata = fs::metadata(&path)
                .with_context(|| format!("failed to stat artifact file {}", path.display()))?;
            let size = metadata.len();
            let file_index =
                u32::try_from(file_index).context("artifact manifest file index exceeds u32")?;
            files.push(ArtifactManifestFile {
                file_index,
                path: manifest_path,
                size,
                checksum: file_checksum(&path)?,
            });
            chunks.extend(chunks_for_file(
                &path,
                file_index,
                chunks.len(),
                chunk_size,
            )?);
        }

        Ok(Self {
            manifest_version: ARTIFACT_MANIFEST_VERSION,
            mx_source_type,
            chunk_size,
            files,
            chunks,
        })
    }

    pub fn seal(&self) -> Result<SealedArtifactManifest> {
        let canonical =
            serde_json::to_vec(self).context("failed to serialize artifact manifest")?;
        let digest = Sha256::digest(&canonical);
        Ok(SealedArtifactManifest {
            artifact_id: format!("{digest:x}"),
            manifest: self.clone(),
        })
    }

    pub fn total_size(&self) -> Result<u64> {
        self.files.iter().try_fold(0_u64, |total, file| {
            total
                .checked_add(file.size)
                .ok_or_else(|| anyhow!("artifact manifest total size overflow"))
        })
    }

    pub fn chunk_count(&self) -> Result<u32> {
        u32::try_from(self.chunks.len()).context("artifact manifest chunk count exceeds u32")
    }

    pub fn to_proto(&self) -> ProtoArtifactManifest {
        ProtoArtifactManifest {
            manifest_version: self.manifest_version,
            mx_source_type: self.mx_source_type,
            chunk_size: self.chunk_size,
            files: self
                .files
                .iter()
                .map(ArtifactManifestFile::to_proto)
                .collect(),
            chunks: self
                .chunks
                .iter()
                .map(ArtifactManifestChunk::to_proto)
                .collect(),
        }
    }
}

impl SealedArtifactManifest {
    pub fn source_metadata(&self) -> Result<ArtifactSourceMetadata> {
        Ok(ArtifactSourceMetadata {
            artifact_id: self.artifact_id.clone(),
            total_size: self.manifest.total_size()?,
            file_count: u32::try_from(self.manifest.files.len())
                .context("artifact manifest file count exceeds u32")?,
            chunk_count: self.manifest.chunk_count()?,
        })
    }

    pub fn to_header_response(
        &self,
        mx_source_id: impl Into<String>,
        metadata_endpoint: impl Into<String>,
        agent_name: impl Into<String>,
        worker_rank: u32,
    ) -> Result<GetArtifactManifestHeaderResponse> {
        Ok(GetArtifactManifestHeaderResponse {
            mx_source_id: mx_source_id.into(),
            artifact_id: self.artifact_id.clone(),
            manifest_version: self.manifest.manifest_version,
            mx_source_type: self.manifest.mx_source_type,
            total_size: self.manifest.total_size()?,
            file_count: u32::try_from(self.manifest.files.len())
                .context("artifact manifest file count exceeds u32")?,
            chunk_count: self.manifest.chunk_count()?,
            chunk_size: self.manifest.chunk_size,
            metadata_endpoint: metadata_endpoint.into(),
            agent_name: agent_name.into(),
            worker_rank,
            files: self
                .manifest
                .files
                .iter()
                .map(ArtifactManifestFile::to_proto)
                .collect(),
        })
    }

    pub fn to_chunks_response(
        &self,
        mx_source_id: impl Into<String>,
        start_chunk_index: u32,
        max_chunks: u32,
    ) -> Result<GetArtifactManifestChunksResponse> {
        let start = usize::try_from(start_chunk_index)
            .context("artifact manifest start chunk index exceeds usize")?;
        if start > self.manifest.chunks.len() {
            bail!(
                "artifact manifest start_chunk_index {} exceeds chunk_count {}",
                start_chunk_index,
                self.manifest.chunks.len()
            );
        }
        let max_chunks = if max_chunks == 0 {
            ARTIFACT_CHUNK_METADATA_PAGE_SIZE
        } else {
            max_chunks.min(ARTIFACT_CHUNK_METADATA_PAGE_SIZE)
        };
        let max =
            usize::try_from(max_chunks).context("artifact manifest page size exceeds usize")?;
        let end = start.saturating_add(max).min(self.manifest.chunks.len());
        let next_page_token = if end < self.manifest.chunks.len() {
            end.to_string()
        } else {
            String::new()
        };
        Ok(GetArtifactManifestChunksResponse {
            mx_source_id: mx_source_id.into(),
            artifact_id: self.artifact_id.clone(),
            start_chunk_index,
            chunks: self.manifest.chunks[start..end]
                .iter()
                .map(ArtifactManifestChunk::to_proto)
                .collect(),
            next_page_token,
        })
    }
}

impl ArtifactManifestFile {
    fn to_proto(&self) -> ProtoArtifactManifestFile {
        ProtoArtifactManifestFile {
            file_index: self.file_index,
            path: self.path.clone(),
            size: self.size,
            checksum: self.checksum.clone(),
        }
    }
}

impl ArtifactManifestChunk {
    fn to_proto(&self) -> ProtoArtifactManifestChunk {
        ProtoArtifactManifestChunk {
            chunk_index: self.chunk_index,
            file_index: self.file_index,
            file_offset: self.file_offset,
            length: self.length,
            checksum: self.checksum.clone(),
        }
    }
}

fn collect_regular_files(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read artifact directory {}", dir.display()))?
        .collect::<std::result::Result<Vec<_>, io::Error>>()
        .with_context(|| {
            format!(
                "failed to read artifact directory entry in {}",
                dir.display()
            )
        })?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect artifact path {}", path.display()))?;
        if file_type.is_symlink() {
            bail!(
                "artifact manifest does not support symlink {}",
                path.display()
            );
        }
        if file_type.is_dir() {
            collect_regular_files(root, &path, files)?;
        } else if file_type.is_file() {
            let canonical = path.canonicalize().with_context(|| {
                format!("failed to canonicalize artifact file {}", path.display())
            })?;
            if !canonical.starts_with(root) {
                bail!(
                    "artifact file resolves outside artifact root: {}",
                    path.display()
                );
            }
            files.push(canonical);
        }
    }

    Ok(())
}

fn manifest_path(path: &Path) -> Result<String> {
    if !path.is_absolute() {
        bail!(
            "artifact manifest path must be absolute: {}",
            path.display()
        );
    }
    let mut parts = Vec::new();
    for component in path.components() {
        let part = match component {
            Component::RootDir => continue,
            Component::Normal(part) => part,
            _ => bail!("unsafe artifact absolute path {}", path.display()),
        };
        let part = part
            .to_str()
            .ok_or_else(|| anyhow!("artifact path is not valid UTF-8: {}", path.display()))?;
        parts.push(part);
    }
    if parts.is_empty() {
        bail!("empty artifact absolute path");
    }
    Ok(format!("/{}", parts.join("/")))
}

fn chunks_for_file(
    path: &Path,
    file_index: u32,
    first_chunk_index: usize,
    chunk_size: u64,
) -> Result<Vec<ArtifactManifestChunk>> {
    let size = fs::metadata(path)
        .with_context(|| format!("failed to stat artifact file {}", path.display()))?
        .len();
    if size == 0 {
        return Ok(Vec::new());
    }

    let file = fs::File::open(path)
        .with_context(|| format!("failed to open artifact file {}", path.display()))?;
    let mut reader = io::BufReader::new(file);
    let mut chunks = Vec::new();
    let mut offset = 0_u64;
    while offset < size {
        let remaining = size
            .checked_sub(offset)
            .ok_or_else(|| anyhow!("artifact chunk offset exceeded file size"))?;
        let chunk_len = remaining.min(chunk_size);
        let mut buffer =
            vec![0_u8; usize::try_from(chunk_len).context("artifact chunk length exceeds usize")?];
        reader
            .read_exact(&mut buffer)
            .with_context(|| format!("failed to read artifact chunk from {}", path.display()))?;
        let chunk_index = first_chunk_index
            .checked_add(chunks.len())
            .ok_or_else(|| anyhow!("artifact manifest chunk index overflow"))?;
        chunks.push(ArtifactManifestChunk {
            chunk_index: u32::try_from(chunk_index)
                .context("artifact manifest chunk index exceeds u32")?,
            file_index,
            file_offset: offset,
            length: chunk_len,
            checksum: format!("{:08x}", crc32c(&buffer)),
        });
        offset = offset
            .checked_add(chunk_len)
            .ok_or_else(|| anyhow!("artifact chunk offset overflow"))?;
    }
    Ok(chunks)
}

fn file_checksum(path: &Path) -> Result<String> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open artifact file {}", path.display()))?;
    let mut reader = io::BufReader::new(file);
    let mut buffer = [0_u8; 64 * 1024];
    let mut checksum = 0;

    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to read artifact file {}", path.display()))?;
        if read == 0 {
            break;
        }
        checksum = crc32c_append(checksum, &buffer[..read]);
    }

    Ok(format!("{checksum:08x}"))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::grpc::p2p::MxSourceType;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn manifest_from_directory_sorts_hashes_and_chunks_files() {
        let temp_dir = TempDir::new().expect("create temp dir");
        fs::create_dir(temp_dir.path().join("nested")).expect("create nested dir");
        fs::write(temp_dir.path().join("nested/b.bin"), b"abcdef").expect("write b");
        fs::write(temp_dir.path().join("a.txt"), b"xyz").expect("write a");

        let manifest = ArtifactManifest::from_directory(
            temp_dir.path(),
            4,
            MxSourceType::TorchCompileCache as i32,
        )
        .expect("manifest");

        assert_eq!(manifest.manifest_version, ARTIFACT_MANIFEST_VERSION);
        assert_eq!(
            manifest.mx_source_type,
            MxSourceType::TorchCompileCache as i32
        );
        assert_eq!(manifest.chunk_size, 4);
        assert_eq!(
            manifest
                .files
                .iter()
                .map(|file| file.path.clone())
                .collect::<Vec<_>>(),
            vec![
                temp_dir
                    .path()
                    .join("a.txt")
                    .canonicalize()
                    .expect("canonicalize a")
                    .to_str()
                    .expect("a path utf8")
                    .to_string(),
                temp_dir
                    .path()
                    .join("nested/b.bin")
                    .canonicalize()
                    .expect("canonicalize b")
                    .to_str()
                    .expect("b path utf8")
                    .to_string(),
            ]
        );
        assert_eq!(
            manifest.chunks,
            vec![
                chunk(0, 0, 0, 3, "25236885"),
                chunk(1, 1, 0, 4, "92c80a31"),
                chunk(2, 1, 4, 2, "6bb2dff5"),
            ]
        );
        assert_eq!(manifest.total_size().expect("total size"), 9);
        assert_eq!(manifest.chunk_count().expect("chunk count"), 3);
    }

    #[test]
    fn manifest_from_directory_sorts_by_manifest_path_with_prefix_collision() {
        let temp_dir = TempDir::new().expect("create temp dir");
        fs::create_dir(temp_dir.path().join("sub")).expect("create sub dir");
        fs::write(temp_dir.path().join("sub/inner.bin"), b"inner").expect("write inner");
        fs::write(temp_dir.path().join("sub.txt"), b"text").expect("write text");

        let manifest = ArtifactManifest::from_directory(
            temp_dir.path(),
            8,
            MxSourceType::TorchCompileCache as i32,
        )
        .expect("manifest");

        assert_eq!(
            manifest
                .files
                .iter()
                .map(|file| file.path.clone())
                .collect::<Vec<_>>(),
            vec![
                temp_dir
                    .path()
                    .join("sub.txt")
                    .canonicalize()
                    .expect("canonicalize sub.txt")
                    .to_str()
                    .expect("sub.txt path utf8")
                    .to_string(),
                temp_dir
                    .path()
                    .join("sub/inner.bin")
                    .canonicalize()
                    .expect("canonicalize inner")
                    .to_str()
                    .expect("inner path utf8")
                    .to_string(),
            ]
        );
        assert_eq!(
            manifest
                .chunks
                .iter()
                .map(|chunk| {
                    (
                        chunk.chunk_index,
                        chunk.file_index,
                        chunk.file_offset,
                        chunk.length,
                    )
                })
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 4), (1, 1, 0, 5)]
        );
    }

    #[test]
    fn sealed_manifest_produces_stable_artifact_metadata() {
        let temp_dir = TempDir::new().expect("create temp dir");
        fs::write(temp_dir.path().join("artifact.bin"), b"artifact").expect("write artifact");

        let left = ArtifactManifest::from_directory(
            temp_dir.path(),
            3,
            MxSourceType::TorchCompileCache as i32,
        )
        .expect("left manifest")
        .seal()
        .expect("left seal");
        let right = ArtifactManifest::from_directory(
            temp_dir.path(),
            3,
            MxSourceType::TorchCompileCache as i32,
        )
        .expect("right manifest")
        .seal()
        .expect("right seal");

        assert_eq!(left.artifact_id, right.artifact_id);
        assert_eq!(left.artifact_id.len(), 64);

        let metadata = left.source_metadata().expect("source metadata");
        assert_eq!(metadata.artifact_id, left.artifact_id);
        assert_eq!(metadata.total_size, 8);
        assert_eq!(metadata.file_count, 1);
        assert_eq!(metadata.chunk_count, 3);
    }

    #[test]
    fn manifest_rejects_invalid_chunk_size() {
        let temp_dir = TempDir::new().expect("create temp dir");
        fs::write(temp_dir.path().join("artifact.bin"), b"artifact").expect("write artifact");

        assert!(
            ArtifactManifest::from_directory(
                temp_dir.path(),
                0,
                MxSourceType::TorchCompileCache as i32,
            )
            .is_err()
        );
        assert!(
            ArtifactManifest::from_directory(
                temp_dir.path(),
                MAX_ARTIFACT_TRANSFER_CHUNK_SIZE + 1,
                MxSourceType::TorchCompileCache as i32,
            )
            .is_err()
        );
    }

    #[test]
    fn empty_files_are_manifested_without_transfer_chunks() {
        let temp_dir = TempDir::new().expect("create temp dir");
        fs::write(temp_dir.path().join("empty"), b"").expect("write empty");

        let manifest = ArtifactManifest::from_directory(
            temp_dir.path(),
            4,
            MxSourceType::TorchCompileCache as i32,
        )
        .expect("manifest");

        assert!(manifest.chunks.is_empty());
        assert_eq!(manifest.total_size().expect("total size"), 0);
        assert_eq!(manifest.chunk_count().expect("chunk count"), 0);
    }

    #[test]
    fn chunks_response_uses_default_page_size_for_zero_max_chunks() {
        let sealed = SealedArtifactManifest {
            artifact_id: "artifact".to_string(),
            manifest: manifest_with_1025_chunks(),
        };

        let response = sealed
            .to_chunks_response("source-123", 0, 0)
            .expect("chunks response");

        assert_eq!(response.chunks.len(), 1024);
        assert_eq!(response.next_page_token, "1024");
    }

    #[test]
    fn chunks_response_caps_requested_page_size() {
        let sealed = SealedArtifactManifest {
            artifact_id: "artifact".to_string(),
            manifest: manifest_with_1025_chunks(),
        };

        let response = sealed
            .to_chunks_response("source-123", 0, 2048)
            .expect("chunks response");

        assert_eq!(response.chunks.len(), 1024);
        assert_eq!(response.next_page_token, "1024");
    }

    #[test]
    fn pinned_artifact_manifest_id_cross_checked_with_python() {
        let manifest = ArtifactManifest {
            manifest_version: ARTIFACT_MANIFEST_VERSION,
            mx_source_type: MxSourceType::TorchCompileCache as i32,
            chunk_size: 8,
            files: vec![
                ArtifactManifestFile {
                    file_index: 0,
                    path: "/cache/sub.txt".to_string(),
                    size: 4,
                    checksum: "text-file-checksum".to_string(),
                },
                ArtifactManifestFile {
                    file_index: 1,
                    path: "/cache/sub/inner.bin".to_string(),
                    size: 5,
                    checksum: "inner-file-checksum".to_string(),
                },
            ],
            chunks: vec![
                ArtifactManifestChunk {
                    chunk_index: 0,
                    file_index: 0,
                    file_offset: 0,
                    length: 4,
                    checksum: "text-chunk-checksum".to_string(),
                },
                ArtifactManifestChunk {
                    chunk_index: 1,
                    file_index: 1,
                    file_offset: 0,
                    length: 5,
                    checksum: "inner-chunk-checksum".to_string(),
                },
            ],
        };

        assert_eq!(
            manifest.seal().expect("seal manifest").artifact_id,
            "a0f08392f2abc45f78bd59f0fe2c601750c2b270dc5cc37c2166d86a65398466"
        );
    }

    #[test]
    #[cfg(unix)]
    fn manifest_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let temp_dir = TempDir::new().expect("create temp dir");
        let target = temp_dir.path().join("target");
        fs::write(&target, b"target").expect("write target");
        symlink(&target, temp_dir.path().join("link")).expect("create symlink");

        assert!(
            ArtifactManifest::from_directory(
                temp_dir.path(),
                4,
                MxSourceType::TorchCompileCache as i32,
            )
            .is_err()
        );
    }

    fn chunk(
        chunk_index: u32,
        file_index: u32,
        file_offset: u64,
        length: u64,
        checksum: &str,
    ) -> ArtifactManifestChunk {
        ArtifactManifestChunk {
            chunk_index,
            file_index,
            file_offset,
            length,
            checksum: checksum.to_string(),
        }
    }

    fn manifest_with_1025_chunks() -> ArtifactManifest {
        ArtifactManifest {
            manifest_version: ARTIFACT_MANIFEST_VERSION,
            mx_source_type: MxSourceType::TorchCompileCache as i32,
            chunk_size: 1,
            files: vec![ArtifactManifestFile {
                file_index: 0,
                path: "/tmp/artifact.bin".to_string(),
                size: 1025,
                checksum: "file".to_string(),
            }],
            chunks: (0..1025)
                .map(|index| ArtifactManifestChunk {
                    chunk_index: index,
                    file_index: 0,
                    file_offset: u64::from(index),
                    length: 1,
                    checksum: format!("chunk-{index}"),
                })
                .collect(),
        }
    }
}
