// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Utils, cache::ProviderCache, constants, providers::ModelProviderTrait};
use anyhow::Result;
use std::{env, path::PathBuf};
use tracing::info;

mod archive_format;
mod cache_entry;
mod downloader;
mod layer_download;
mod path;
mod provider_cache;
mod reference;
mod registry_auth;

use cache_entry::{CacheEntry, StagingCacheEntry};
use downloader::Downloader;
use reference::OciReference;

pub(crate) use provider_cache::OciProviderCache;

const MODEL_EXPRESS_CACHE_ENV_VAR: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";

/// File-oriented OCI artifact provider implementation.
pub struct OciProvider;

impl OciProvider {
    fn cache_root(cache_dir: Option<PathBuf>) -> PathBuf {
        if let Some(dir) = cache_dir {
            return dir;
        }

        if let Ok(cache_path) = env::var(MODEL_EXPRESS_CACHE_ENV_VAR) {
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
        let final_entry = CacheEntry::new(&cache_root, &reference);

        // Keep OCI cache reuse NGC-like for now: a non-empty artifact directory is
        // considered usable, even if it was created by an ignore_weights download.
        if let Some(existing) = final_entry.existing_files_dir()? {
            info!(
                "OCI model '{model_name}' found in cache at {}",
                existing.display()
            );
            return Ok(existing);
        }

        let staging_entry = StagingCacheEntry::new(&cache_root);
        staging_entry.create().await?;

        let downloader = Downloader::new(model_name, &reference);
        downloader
            .download_to_staging(&staging_entry, ignore_weights)
            .await?;

        let files_dir = final_entry.publish_from(&staging_entry)?;
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
        let entry = CacheEntry::new(&cache_dir, &reference);
        entry.existing_files_dir()?.ok_or_else(|| {
            anyhow::anyhow!(
                "OCI model '{model_name}' not found in cache (expected {})",
                entry.files_dir().display()
            )
        })
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
        let entry = CacheEntry::path_for(dir.path(), &reference);
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
