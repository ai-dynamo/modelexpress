// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    model_dir::ModelDir,
    model_name::{BucketName, CACHE_ROOT_DIR_NAME, ModelName},
};
use crate::cache::{ModelInfo, ProviderCache};
use crate::models::ModelProvider;
use anyhow::{Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tracing::{info, warn};

pub struct GcsProviderCache;

impl GcsProviderCache {
    fn collect_cached_models(
        cache_dir: &Path,
        current_dir: &Path,
        models: &mut Vec<ModelInfo>,
    ) -> Result<()> {
        let current_model_dir = ModelDir::new(cache_dir, current_dir);
        if current_model_dir.has_manifest_file() {
            match current_model_dir.model_name() {
                Ok(model) => {
                    if current_model_dir.cache_satisfies_request(false)? {
                        models.push(ModelInfo {
                            provider: ModelProvider::Gcs,
                            name: model.to_string(),
                            size: current_model_dir.size()?,
                            path: current_dir.to_path_buf(),
                        });
                    }
                    return Ok(());
                }
                Err(err) => {
                    warn!(
                        "Skipping invalid GCS cache entry '{}': {}",
                        current_dir.display(),
                        err
                    );
                }
            }
        }

        for entry in fs::read_dir(current_dir)
            .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_cached_models(cache_dir, &path, models)?;
            }
        }

        Ok(())
    }
}

impl ProviderCache for GcsProviderCache {
    fn clear_model(&self, cache_dir: &Path, model_name: &str) -> Result<()> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(cache_dir);
        let model_dir_state = ModelDir::new(cache_dir, &model_dir);

        if !model_dir_state.is_removable()? {
            info!(
                "Model not found in cache: {} ({:?})",
                model_name,
                ModelProvider::Gcs
            );
            return Ok(());
        }

        model_dir_state.remove()?;
        info!("Cleared model: {} ({:?})", model_name, ModelProvider::Gcs);

        Ok(())
    }

    fn resolve_model_path(
        &self,
        cache_dir: &Path,
        model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        Ok(ModelName::parse(model_name)?.model_dir(cache_dir))
    }

    fn list_models(&self, cache_dir: &Path) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        let root = cache_dir.join(CACHE_ROOT_DIR_NAME);

        if !root.exists() {
            return Ok(models);
        }

        for bucket_entry in fs::read_dir(&root)? {
            let bucket_entry = bucket_entry?;
            let bucket_path = bucket_entry.path();
            if !bucket_path.is_dir() {
                continue;
            }

            let Some(bucket_name) = bucket_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };

            if BucketName::parse(bucket_name).is_err() {
                warn!(
                    "Skipping invalid GCS bucket cache entry '{}'",
                    bucket_path.display()
                );
                continue;
            }

            Self::collect_cached_models(cache_dir, &bucket_path, &mut models)?;
        }

        Ok(models)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::test_support::{
        expected_model_dir, manifest_entry, write_cached_model, write_incomplete_cached_model,
        write_manifest_with_payloads,
    };
    use super::*;
    use crate::cache::ProviderCache;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_list_models_scenarios() {
        let cache = GcsProviderCache;
        for scenario in [
            "partial_manifest",
            "recursive_siblings",
            "model_path_contains_internal_segment",
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            match scenario {
                "partial_manifest" => write_manifest_with_payloads(
                    temp_dir.path(),
                    "gs://bucket/foo/bar/baz",
                    &[("tokenizer.json", b"{}")],
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
                "recursive_siblings" => {
                    write_cached_model(
                        temp_dir.path(),
                        "gs://bucket/foo/bar/baz",
                        "tokenizer.json",
                        b"{}",
                    );
                    write_cached_model(
                        temp_dir.path(),
                        "gs://bucket/foo/bar/buz",
                        "weights/model.bin",
                        b"abcd",
                    );
                }
                "model_path_contains_internal_segment" => {
                    write_cached_model(
                        temp_dir.path(),
                        "gs://bucket/foo/.mx/bar",
                        "tokenizer.json",
                        b"{}",
                    );
                }
                _ => unreachable!("unexpected scenario"),
            }

            let mut models = cache
                .list_models(temp_dir.path())
                .expect("Expected model listing");
            models.sort_by(|left, right| left.name.cmp(&right.name));

            match scenario {
                "partial_manifest" => {
                    assert!(models.is_empty(), "partial manifest should not be listed");
                }
                "recursive_siblings" => {
                    assert_eq!(models.len(), 2);
                    assert_eq!(models[0].name, "gs://bucket/foo/bar/baz");
                    assert_eq!(models[0].size, 2);
                    assert_eq!(models[1].name, "gs://bucket/foo/bar/buz");
                    assert_eq!(models[1].size, 4);
                }
                "model_path_contains_internal_segment" => {
                    assert_eq!(models.len(), 1);
                    assert_eq!(models[0].name, "gs://bucket/foo/.mx/bar");
                    assert_eq!(models[0].size, 2);
                }
                _ => unreachable!("unexpected scenario"),
            }
        }
    }

    #[test]
    fn test_clear_model_scenarios() {
        let cache = GcsProviderCache;
        for scenario in [
            "ancestor_keeps_descendant",
            "incomplete_removed",
            "descendant_keeps_cached_ancestor",
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            match scenario {
                "ancestor_keeps_descendant" => {
                    let ancestor_name = "gs://bucket/foo/bar";
                    let descendant_name = "gs://bucket/foo/bar/baz";
                    write_cached_model(temp_dir.path(), descendant_name, "tokenizer.json", b"{}");
                    cache
                        .clear_model(temp_dir.path(), ancestor_name)
                        .expect("Expected clear to succeed");
                    assert!(expected_model_dir(temp_dir.path(), descendant_name).exists());
                }
                "incomplete_removed" => {
                    let model_name = "gs://bucket/foo/bar";
                    let model_dir = expected_model_dir(temp_dir.path(), model_name);
                    write_incomplete_cached_model(
                        temp_dir.path(),
                        model_name,
                        "tokenizer.json",
                        b"{}",
                    );
                    cache
                        .clear_model(temp_dir.path(), model_name)
                        .expect("Expected clear to succeed");
                    assert!(!model_dir.exists());
                }
                "descendant_keeps_cached_ancestor" => {
                    let ancestor_name = "gs://bucket/foo/bar";
                    let descendant_name = "gs://bucket/foo/bar/baz";
                    let ancestor_dir = expected_model_dir(temp_dir.path(), ancestor_name);
                    let descendant_dir = expected_model_dir(temp_dir.path(), descendant_name);
                    write_cached_model(temp_dir.path(), ancestor_name, "tokenizer.json", b"{}");
                    fs::create_dir_all(&descendant_dir).expect("Failed to create descendant dir");
                    fs::write(descendant_dir.join("partial.bin"), b"partial")
                        .expect("Failed to create descendant payload");
                    cache
                        .clear_model(temp_dir.path(), descendant_name)
                        .expect("Expected clear to succeed");
                    assert!(ancestor_dir.exists());
                    assert!(ancestor_dir.join("tokenizer.json").exists());
                    assert!(descendant_dir.exists());
                }
                _ => unreachable!("unexpected scenario"),
            }
        }
    }
}
