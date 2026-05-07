// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    cache_entry::{
        CACHE_ROOT_DIR_NAME, CacheEntry, FILES_DIR_NAME, TMP_DIR_NAME, repository_from_cache_key,
    },
    reference::OciReference,
};
use crate::{
    cache::{ModelInfo, ProviderCache, directory_size},
    models::ModelProvider,
};
use anyhow::{Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tracing::{info, warn};

pub(crate) struct OciProviderCache;

struct CacheWalker<'a> {
    root: &'a Path,
    models: Vec<ModelInfo>,
}

impl<'a> CacheWalker<'a> {
    fn collect(root: &'a Path) -> Result<Vec<ModelInfo>> {
        let mut walker = Self {
            root,
            models: Vec::new(),
        };
        walker.visit(root)?;
        Ok(walker.models)
    }

    fn visit(&mut self, current: &Path) -> Result<()> {
        if current
            .file_name()
            .is_some_and(|name| name == FILES_DIR_NAME)
            && self.push_model(current)?
        {
            return Ok(());
        }

        for entry in fs::read_dir(current).with_context(|| format!("Failed to read {current:?}"))? {
            let entry = entry.with_context(|| format!("Failed to read entry in {current:?}"))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("Failed to inspect {path:?}"))?;

            if file_type.is_dir() && !self.is_staging_path(&path) {
                self.visit(&path)?;
            }
        }

        Ok(())
    }

    fn push_model(&mut self, files_dir: &Path) -> Result<bool> {
        let Some(name) = self.model_name_from_files_dir(files_dir)? else {
            return Ok(false);
        };

        if CacheEntry::files_dir_is_non_empty(files_dir)? {
            self.models.push(ModelInfo {
                provider: ModelProvider::Oci,
                name,
                size: directory_size(files_dir)?,
                path: files_dir.to_path_buf(),
            });
        }
        Ok(true)
    }

    fn is_staging_path(&self, path: &Path) -> bool {
        path.strip_prefix(self.root)
            .ok()
            .and_then(|relative| relative.components().next())
            .is_some_and(|component| component.as_os_str() == TMP_DIR_NAME)
    }

    fn model_name_from_files_dir(&self, files_dir: &Path) -> Result<Option<String>> {
        let relative = files_dir.strip_prefix(self.root).with_context(|| {
            format!("Failed to strip prefix {:?} from {files_dir:?}", self.root)
        })?;
        let parts = Self::path_parts(relative)?;

        if parts.len() != 5 || parts.last().is_none_or(|part| part != FILES_DIR_NAME) {
            return Ok(None);
        }

        let registry = &parts[0];
        let repository = repository_from_cache_key(&parts[1])?;
        let reference = &parts[3];

        match parts[2].as_str() {
            "tags" => Ok(Some(format!("{registry}/{repository}:{reference}"))),
            "digests" => {
                let (algorithm, digest) = reference.rsplit_once('-').ok_or_else(|| {
                    anyhow::anyhow!("Invalid OCI digest cache entry {reference:?}")
                })?;
                Ok(Some(format!(
                    "{registry}/{repository}@{algorithm}:{digest}"
                )))
            }
            _ => Ok(None),
        }
    }

    fn path_parts(path: &Path) -> Result<Vec<String>> {
        path.components()
            .map(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("OCI cache path contains non-UTF-8 data"))
            })
            .collect()
    }
}

impl ProviderCache for OciProviderCache {
    fn clear_model(&self, cache_root: &Path, model_name: &str) -> Result<()> {
        let reference = OciReference::parse(model_name)?;
        let entry = CacheEntry::new(cache_root, &reference);

        if entry.path().exists() {
            fs::remove_dir_all(entry.path())
                .with_context(|| format!("Failed to remove OCI model cache {:?}", entry.path()))?;
            info!("Cleared OCI model: {model_name}");
        } else {
            warn!("OCI model '{model_name}' not found in cache");
        }

        Ok(())
    }

    fn resolve_model_path(
        &self,
        cache_root: &Path,
        model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        let reference = OciReference::parse(model_name)?;
        // This is a deterministic destination path, not an existing-cache check.
        // In no-shared-storage mode the client streams files directly here, so an
        // interrupted transfer can leave a non-empty partial directory. Keep this
        // method as a provider-specific path mapper; direct OCI downloads still use
        // staging plus rename before publishing a final cache entry.
        Ok(CacheEntry::new(cache_root, &reference).files_dir())
    }

    fn list_models(&self, cache_root: &Path) -> Result<Vec<ModelInfo>> {
        let root = cache_root.join(CACHE_ROOT_DIR_NAME);
        if !root.exists() {
            return Ok(Vec::new());
        }

        CacheWalker::collect(&root)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cache::ProviderCache;
    use crate::models::ModelProvider;
    use crate::providers::oci::{cache_entry::FILES_DIR_NAME, reference::OciReference};

    #[test]
    fn test_model_path_returns_layout_without_validating_cache() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = OciProviderCache
            .resolve_model_path(dir.path(), "registry.example.com/team/model:v1", None)
            .expect("model path");

        assert_eq!(
            path,
            dir.path()
                .join("oci/registry.example.com/team%2Fmodel/tags/v1/files")
        );
    }

    #[test]
    fn test_cache_list_and_clear() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let reference = OciReference::parse("registry.example.com/team/model:v1")
            .expect("reference should parse");
        let entry = CacheEntry::path_for(dir.path(), &reference);
        let files = entry.join(FILES_DIR_NAME);
        fs::create_dir_all(&files).expect("create files dir");
        fs::write(files.join("config.json"), b"{}").expect("write model file");

        let cache = OciProviderCache;
        let models = cache.list_models(dir.path()).expect("list models");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, ModelProvider::Oci);
        assert_eq!(models[0].name, "registry.example.com/team/model:v1");
        assert_eq!(models[0].path, files);

        cache
            .clear_model(dir.path(), "registry.example.com/team/model:v1")
            .expect("clear model");
        assert!(!entry.exists());
    }

    #[test]
    fn test_cache_list_allows_files_in_repository_path() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let reference = OciReference::parse("registry.example.com/team/files/model:v1")
            .expect("reference should parse");
        let entry = CacheEntry::path_for(dir.path(), &reference);
        let files = entry.join(FILES_DIR_NAME);
        fs::create_dir_all(&files).expect("create files dir");
        fs::write(files.join("config.json"), b"{}").expect("write model file");

        let models = OciProviderCache
            .list_models(dir.path())
            .expect("list models");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "registry.example.com/team/files/model:v1");
        assert_eq!(models[0].path, files);
    }

    #[test]
    fn test_cache_list_keeps_repository_from_overlapping_layout() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let nested = OciReference::parse("registry.example.com/team/model/tags/dev/files/other:v1")
            .expect("nested reference should parse");
        let nested_entry = CacheEntry::path_for(dir.path(), &nested);
        let nested_files = nested_entry.join(FILES_DIR_NAME);
        fs::create_dir_all(&nested_files).expect("create nested files dir");
        fs::write(nested_files.join("config.json"), b"{}").expect("write model file");

        let alias = OciReference::parse("registry.example.com/team/model:dev")
            .expect("alias reference should parse");
        assert!(
            !CacheEntry::path_for(dir.path(), &alias)
                .join(FILES_DIR_NAME)
                .exists()
        );

        let models = OciProviderCache
            .list_models(dir.path())
            .expect("list models");

        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].name,
            "registry.example.com/team/model/tags/dev/files/other:v1"
        );
        assert_eq!(models[0].path, nested_files);
    }
}
