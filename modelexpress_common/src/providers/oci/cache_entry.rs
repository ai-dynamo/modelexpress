// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::reference::OciReference;
use anyhow::{Context, Result};
use std::{
    fs, io,
    path::{Path, PathBuf},
};
use tracing::warn;
use uuid::Uuid;

pub const CACHE_ROOT_DIR_NAME: &str = "oci";
pub const TMP_DIR_NAME: &str = ".tmp";
pub const FILES_DIR_NAME: &str = "files";
const BLOBS_DIR_NAME: &str = ".blobs";

#[derive(Debug, Clone)]
pub struct CacheEntry {
    path: PathBuf,
}

impl CacheEntry {
    pub fn new(cache_root: &Path, reference: &OciReference) -> Self {
        Self {
            path: Self::path_for(cache_root, reference),
        }
    }

    pub fn path_for(cache_root: &Path, reference: &OciReference) -> PathBuf {
        let mut path = cache_root
            .join(CACHE_ROOT_DIR_NAME)
            .join(reference.registry())
            .join(repository_cache_key(reference.repository()));

        if let Some(digest) = reference.digest() {
            path = path.join("digests").join(digest.replace(':', "-"));
        } else if let Some(tag) = reference.tag() {
            path = path.join("tags").join(tag);
        }

        path
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn files_dir(&self) -> PathBuf {
        Self::files_dir_for(&self.path)
    }

    pub fn existing_files_dir(&self) -> Result<Option<PathBuf>> {
        Self::existing_files_dir_at(&self.path)
    }

    pub fn publish_from(&self, staging: &StagingCacheEntry) -> Result<PathBuf> {
        let result = self.publish(staging);
        staging.cleanup();
        result
    }

    pub fn files_dir_is_non_empty(files_dir: &Path) -> Result<bool> {
        Ok(fs::read_dir(files_dir)
            .with_context(|| format!("Failed to read OCI files directory {files_dir:?}"))?
            .next()
            .is_some())
    }

    fn files_dir_for(entry_path: &Path) -> PathBuf {
        entry_path.join(FILES_DIR_NAME)
    }

    fn publish(&self, staging: &StagingCacheEntry) -> Result<PathBuf> {
        if let Some(existing) = Self::existing_files_dir_at(&self.path)? {
            return Ok(existing);
        }

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create OCI cache parent {parent:?}"))?;
        }

        match fs::rename(staging.path(), &self.path) {
            Ok(()) => Ok(self.files_dir()),
            Err(rename_error) if self.path.exists() => {
                self.recover_existing_after_rename(rename_error)
            }
            Err(rename_error) => Err(anyhow::anyhow!(rename_error).context(format!(
                "Failed to publish OCI cache entry from {} to {}",
                staging.path().display(),
                self.path.display()
            ))),
        }
    }

    fn recover_existing_after_rename(&self, rename_error: io::Error) -> Result<PathBuf> {
        match Self::existing_files_dir_at(&self.path) {
            Ok(Some(existing)) => Ok(existing),
            Ok(None) => Err(anyhow::anyhow!(
                "OCI cache publish failed for {}: {rename_error}",
                self.path.display()
            )),
            Err(error) => Err(error).with_context(|| {
                format!(
                    "OCI cache publish raced with an incomplete or corrupt cache entry at {}",
                    self.path.display()
                )
            }),
        }
    }

    fn validate_path(entry_path: &Path) -> Result<PathBuf> {
        let files_dir = Self::files_dir_for(entry_path);
        if !files_dir.is_dir() {
            anyhow::bail!("missing files directory at {files_dir:?}");
        }

        if !Self::files_dir_is_non_empty(&files_dir)? {
            anyhow::bail!("OCI files directory at {files_dir:?} is empty");
        }

        Ok(files_dir)
    }

    fn existing_files_dir_at(entry_path: &Path) -> Result<Option<PathBuf>> {
        if !entry_path.exists() {
            return Ok(None);
        }

        let files_dir = Self::validate_path(entry_path).with_context(|| {
            format!(
                "OCI cache entry at {} is incomplete or corrupt; remove it before retrying",
                entry_path.display()
            )
        })?;

        Ok(Some(files_dir))
    }
}

pub fn repository_cache_key(repository: &str) -> String {
    let mut key = String::with_capacity(repository.len());
    for character in repository.chars() {
        match character {
            '%' => key.push_str("%25"),
            '/' => key.push_str("%2F"),
            _ => key.push(character),
        }
    }
    key
}

pub fn repository_from_cache_key(key: &str) -> Result<String> {
    let mut repository = String::with_capacity(key.len());
    let mut characters = key.chars();

    while let Some(character) = characters.next() {
        if character != '%' {
            repository.push(character);
            continue;
        }

        let first = characters
            .next()
            .ok_or_else(|| anyhow::anyhow!("Invalid OCI repository cache key '{key}'"))?;
        let second = characters
            .next()
            .ok_or_else(|| anyhow::anyhow!("Invalid OCI repository cache key '{key}'"))?;

        match (first.to_ascii_uppercase(), second.to_ascii_uppercase()) {
            ('2', '5') => repository.push('%'),
            ('2', 'F') => repository.push('/'),
            _ => anyhow::bail!("Invalid OCI repository cache key '{key}'"),
        }
    }

    Ok(repository)
}

#[derive(Debug)]
pub struct StagingCacheEntry {
    path: PathBuf,
}

impl StagingCacheEntry {
    pub fn new(cache_root: &Path) -> Self {
        let path = cache_root
            .join(CACHE_ROOT_DIR_NAME)
            .join(TMP_DIR_NAME)
            .join(Uuid::new_v4().to_string());
        Self { path }
    }

    pub async fn create(&self) -> Result<()> {
        let tmp_root = self.path.parent().ok_or_else(|| {
            anyhow::anyhow!("OCI staging entry '{}' has no parent", self.path.display())
        })?;
        tokio::fs::create_dir_all(tmp_root)
            .await
            .with_context(|| format!("Failed to create OCI temporary cache root {tmp_root:?}"))?;
        tokio::fs::create_dir(&self.path)
            .await
            .with_context(|| format!("Failed to create OCI staging entry {:?}", self.path))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn files_dir(&self) -> PathBuf {
        CacheEntry::files_dir_for(&self.path)
    }

    pub fn blob_root(&self) -> PathBuf {
        self.path.join(BLOBS_DIR_NAME)
    }

    pub fn cleanup(&self) {
        if self.path.exists()
            && let Err(error) = fs::remove_dir_all(&self.path)
        {
            warn!(
                "Failed to remove OCI staging directory {}: {error}",
                self.path.display()
            );
        }
    }
}

impl Drop for StagingCacheEntry {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::providers::oci::reference::OciReference;

    #[test]
    fn test_cache_path_generation() {
        let root = Path::new("/cache");
        let tagged = OciReference::parse("registry.example.com/team/model:v1")
            .expect("tagged reference should parse");
        assert_eq!(
            CacheEntry::path_for(root, &tagged),
            PathBuf::from("/cache/oci/registry.example.com/team%2Fmodel/tags/v1")
        );

        let digest = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let by_digest = OciReference::parse(&format!("registry.example.com/team/model@{digest}"))
            .expect("digest reference should parse");
        assert_eq!(
            CacheEntry::path_for(root, &by_digest),
            PathBuf::from(
                "/cache/oci/registry.example.com/team%2Fmodel/digests/sha256-ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            )
        );
    }

    #[test]
    fn test_cache_path_generation_keeps_repository_from_overlapping_layout() {
        let root = Path::new("/cache");
        let nested = OciReference::parse("registry.example.com/team/model/tags/dev/files/other:v1")
            .expect("nested reference should parse");
        let tagged = OciReference::parse("registry.example.com/team/model:dev")
            .expect("tagged reference should parse");

        assert_eq!(
            CacheEntry::path_for(root, &nested),
            PathBuf::from(
                "/cache/oci/registry.example.com/team%2Fmodel%2Ftags%2Fdev%2Ffiles%2Fother/tags/v1"
            )
        );
        assert!(
            !CacheEntry::path_for(root, &nested)
                .starts_with(CacheEntry::path_for(root, &tagged).join(FILES_DIR_NAME))
        );
    }

    #[test]
    fn test_cache_entry_validation_requires_non_empty_files() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let entry = dir.path().join("entry");
        let files = entry.join(FILES_DIR_NAME);
        fs::create_dir_all(&files).expect("create files dir");

        let err =
            CacheEntry::validate_path(&entry).expect_err("empty cache should fail validation");
        assert!(err.to_string().contains("is empty"));

        fs::write(files.join("config.json"), b"{}").expect("write model file");
        assert_eq!(
            CacheEntry::validate_path(&entry).expect("valid cache"),
            files
        );
    }

    #[test]
    fn test_existing_incomplete_cache_rejected() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let reference = OciReference::parse("registry.example.com/team/model:v1")
            .expect("reference should parse");
        let entry = CacheEntry::path_for(dir.path(), &reference);
        fs::create_dir_all(entry.join(FILES_DIR_NAME)).expect("create incomplete files dir");

        let result = CacheEntry::existing_files_dir_at(&entry);
        let err = result.expect_err("incomplete cache should fail");
        assert!(err.to_string().contains("incomplete or corrupt"));
    }

    #[test]
    fn test_publish_cleans_staging_when_existing_cache_is_corrupt() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let reference = OciReference::parse("registry.example.com/team/model:v1")
            .expect("reference should parse");
        let final_entry = CacheEntry::new(dir.path(), &reference);
        fs::create_dir_all(final_entry.path().join(FILES_DIR_NAME))
            .expect("create incomplete final files dir");

        let staging_entry = StagingCacheEntry::new(dir.path());
        let staging_files = staging_entry.files_dir();
        fs::create_dir_all(&staging_files).expect("create staging files dir");
        fs::write(staging_files.join("config.json"), b"{}").expect("write staging model file");

        let err = final_entry
            .publish_from(&staging_entry)
            .expect_err("corrupt final cache should fail publish");

        assert!(err.to_string().contains("incomplete or corrupt"));
        assert!(!staging_entry.path().exists());
    }

    #[test]
    fn test_existing_cache_entry_uses_non_empty_files_dir() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let reference = OciReference::parse("registry.example.com/team/model:v1")
            .expect("reference should parse");
        let entry = CacheEntry::path_for(dir.path(), &reference);
        let files = entry.join(FILES_DIR_NAME);
        fs::create_dir_all(&files).expect("create files dir");
        fs::write(files.join("config.json"), b"{}").expect("write model file");

        let files_dir = CacheEntry::existing_files_dir_at(&entry)
            .expect("cache lookup should succeed")
            .expect("cache should exist");
        assert_eq!(files_dir, files);
    }
}
