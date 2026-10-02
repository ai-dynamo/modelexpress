// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Utils,
    cache::ProviderCache,
    constants,
    providers::{ModelProviderTrait, ensure_crypto_provider},
};
use anyhow::Result;
use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::info;

mod archive_format;
mod cache_entry;
mod downloader;
mod gbuild;
mod layer_download;
mod path;
mod provider_cache;
mod reference;
mod registry_auth;

use cache_entry::{CacheEntry, StagingCacheEntry};
use downloader::Downloader;
use reference::OciReference;

pub(crate) use provider_cache::OciProviderCache;

/// Files covered by a completed OCI cache entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadMode {
    Full,
    Metadata,
}

impl From<bool> for DownloadMode {
    fn from(ignore_weights: bool) -> Self {
        if ignore_weights {
            Self::Metadata
        } else {
            Self::Full
        }
    }
}

/// Stages one OCI transfer and publishes it only after the caller completes it.
/// Direct pulls and server streams use the same cache layout and completion check.
pub struct OciCacheWrite {
    entry: CacheEntry,
    staging: Arc<StagingCacheEntry>,
}

impl OciCacheWrite {
    pub async fn new(cache_root: &Path, model_name: &str, mode: DownloadMode) -> Result<Self> {
        let reference = OciReference::parse(model_name)?;
        let entry = CacheEntry::for_mode(cache_root, &reference, mode);
        let staging = Arc::new(StagingCacheEntry::new(cache_root));
        staging.create().await?;
        tokio::fs::create_dir_all(staging.files_dir()).await?;
        Ok(Self { entry, staging })
    }

    pub fn files_dir(&self) -> PathBuf {
        self.staging.files_dir()
    }

    pub fn publish(self) -> Result<PathBuf> {
        self.entry.publish_from(&self.staging)
    }
}

/// File-oriented OCI artifact provider implementation.
pub struct OciProvider;

impl OciProvider {
    pub fn cached_model_path(
        cache_root: &Path,
        model_name: &str,
        mode: DownloadMode,
    ) -> Result<PathBuf> {
        let reference = OciReference::parse(model_name)?;
        CacheEntry::for_mode(cache_root, &reference, mode)
            .existing_files_dir()?
            .ok_or_else(|| {
                anyhow::anyhow!("OCI model '{model_name}' not found in cache for {mode:?} download")
            })
    }

    fn cache_root(cache_dir: Option<PathBuf>) -> PathBuf {
        if let Some(dir) = cache_dir {
            return dir;
        }

        if let Ok(cache_path) = env::var(crate::envs::MODEL_EXPRESS_CACHE_DIRECTORY) {
            return PathBuf::from(cache_path);
        }

        let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(constants::DEFAULT_CACHE_PATH)
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for OciProvider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        let cache_root = Self::cache_root(cache_dir);
        let reference = OciReference::parse(model_name)?;
        let mode = DownloadMode::from(ignore_weights);
        let final_entry = CacheEntry::for_mode(&cache_root, &reference, mode);

        if let Some(existing) = final_entry.existing_files_dir()? {
            info!(
                "OCI model '{model_name}' found in cache at {}",
                existing.display()
            );
            return Ok(existing);
        }

        ensure_crypto_provider()?;

        let transfer = OciCacheWrite::new(&cache_root, model_name, mode).await?;

        let downloader = Downloader::new(model_name, &reference);
        downloader
            .download_to_staging(&transfer.staging, ignore_weights)
            .await?;

        let files_dir = transfer.publish()?;
        info!(
            "Downloaded OCI artifact '{model_name}' to {}",
            files_dir.display()
        );
        Ok(files_dir)
    }

    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        OciProviderCache.clear_model(&cache_dir, model_name)
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let reference = OciReference::parse(model_name)?;
        for mode in [DownloadMode::Full, DownloadMode::Metadata] {
            if let Some(path) =
                CacheEntry::for_mode(&cache_dir, &reference, mode).existing_files_dir()?
            {
                return Ok(path);
            }
        }
        anyhow::bail!("OCI model '{model_name}' not found in cache")
    }

    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        Ok(OciReference::parse(model_name)?.canonical_name())
    }

    fn provider_name(&self) -> &'static str {
        "OCI"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    use super::{cache_entry::FILES_DIR_NAME as FILES_DIR, reference::OciReference};

    #[tokio::test]
    async fn cache_writes_publish_only_complete_transfers() {
        let root = TempDir::new().expect("cache");
        let model = "registry.example.com/team/model:v1";
        let interrupted = OciCacheWrite::new(root.path(), model, DownloadMode::Full)
            .await
            .expect("stage");
        fs::write(interrupted.files_dir().join("config.json"), b"partial").expect("partial file");
        drop(interrupted);
        let first = OciCacheWrite::new(root.path(), model, DownloadMode::Full)
            .await
            .expect("first");
        let second = OciCacheWrite::new(root.path(), model, DownloadMode::Full)
            .await
            .expect("second");
        fs::write(first.files_dir().join("config.json"), b"complete").expect("first file");
        fs::write(second.files_dir().join("config.json"), b"complete").expect("second file");
        let (left, right) = tokio::join!(
            tokio::task::spawn_blocking(move || first.publish()),
            tokio::task::spawn_blocking(move || second.publish()),
        );
        let path = left.expect("first task").expect("publish");
        assert_eq!(right.expect("second task").expect("racing publish"), path);
        assert_eq!(
            fs::read(path.join("config.json")).expect("complete file"),
            b"complete"
        );
        assert_eq!(
            OciProvider::cached_model_path(root.path(), model, DownloadMode::Full)
                .expect("cache hit"),
            path
        );
    }

    #[tokio::test]
    async fn blocking_work_retains_staging_until_it_finishes() {
        let root = TempDir::new().expect("cache");
        let transfer = OciCacheWrite::new(
            root.path(),
            "registry.example.com/model:v1",
            DownloadMode::Full,
        )
        .await
        .expect("stage");
        let path = transfer.files_dir();
        let staging = Arc::clone(&transfer.staging);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async move {
            let _transfer = transfer;
            tokio::task::spawn_blocking(move || {
                started_tx.send(()).expect("start");
                release_rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .expect("release");
                fs::write(staging.files_dir().join("payload"), b"complete")
                    .expect("staging remains writable");
                drop(staging);
                finished_tx.send(()).expect("finish");
            })
            .await
            .expect("extraction task");
        });
        started_rx.await.expect("started");
        waiter.abort();
        assert!(waiter.await.expect_err("cancelled").is_cancelled());
        assert!(path.is_dir());
        release_tx.send(()).expect("release blocking work");
        finished_rx.await.expect("finished");
        assert_eq!(
            fs::read_dir(root.path().join("oci/.tmp"))
                .expect("staging root")
                .count(),
            0
        );
    }

    #[test]
    fn test_canonical_model_name_accepts_oci_scheme() {
        assert_eq!(
            OciProvider
                .canonical_model_name("oci://registry.example.com/team/model:v1")
                .expect("canonical ref"),
            "registry.example.com/team/model:v1"
        );

        let digest = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        assert_eq!(
            OciProvider
                .canonical_model_name(&format!(
                    "oci://registry.example.com/team/model:v1@{digest}"
                ))
                .expect("canonical digest ref"),
            format!("registry.example.com/team/model@{digest}")
        );
    }

    #[tokio::test]
    async fn test_get_model_path_rejects_missing_or_incomplete_cache() {
        let dir = TempDir::new().expect("temp dir");
        let missing = OciProvider
            .get_model_path(
                "registry.example.com/team/model:v1",
                dir.path().to_path_buf(),
            )
            .await
            .expect_err("missing cache should fail");
        assert!(missing.to_string().contains("not found in cache"));

        let reference = OciReference::parse("registry.example.com/team/model:v1")
            .expect("reference should parse");
        let entry = CacheEntry::new(dir.path(), &reference).path().to_path_buf();
        fs::create_dir_all(entry.join(FILES_DIR)).expect("create incomplete files dir");

        let incomplete = OciProvider
            .get_model_path(
                "registry.example.com/team/model:v1",
                dir.path().to_path_buf(),
            )
            .await
            .expect_err("incomplete cache should fail");
        assert!(incomplete.to_string().contains("incomplete or corrupt"));
    }
}
