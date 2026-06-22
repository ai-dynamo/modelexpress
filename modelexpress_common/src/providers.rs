// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::path::PathBuf;

/// Trait for model providers
/// This trait provides the framework for supporting multiple model providers.
#[async_trait::async_trait]
pub trait ModelProviderTrait: Send + Sync {
    /// Download a model and return the path where it was downloaded
    async fn download_model(
        &self,
        model_name: &str,
        cache_path: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf>;

    /// Delete a model from the provider's cache
    /// Returns Ok(()) if the model was successfully deleted or didn't exist
    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()>;

    /// Get the full path to the latest model snapshot if it exists
    /// Returns the path if found, or an error if not found
    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf>;

    /// Return the canonical model name for this provider.
    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        Ok(model_name.to_string())
    }

    /// Get the provider name for logging
    fn provider_name(&self) -> &'static str;

    /// Check if a file should be ignored during download
    /// This allows each provider to specify which files to skip
    /// Default implementation ignores dotfiles and common repository metadata files
    fn is_ignored(filename: &str) -> bool
    where
        Self: Sized,
    {
        const DEFAULT_IGNORED: [&str; 1] = ["README.md"];
        let name = std::path::Path::new(filename)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(filename);
        name.starts_with('.') || DEFAULT_IGNORED.contains(&name)
    }

    /// Check if a file is an image file that should be ignored
    /// This allows each provider to customize image file detection
    /// Default implementation recognizes common image file extensions
    fn is_image(path: &std::path::Path) -> bool
    where
        Self: Sized,
    {
        path.extension().is_some_and(|ext| {
            ext.eq_ignore_ascii_case("png")
                || ext.eq_ignore_ascii_case("jpg")
                || ext.eq_ignore_ascii_case("jpeg")
                || ext.eq_ignore_ascii_case("gif")
                || ext.eq_ignore_ascii_case("webp")
                || ext.eq_ignore_ascii_case("svg")
                || ext.eq_ignore_ascii_case("ico")
                || ext.eq_ignore_ascii_case("bmp")
                || ext.eq_ignore_ascii_case("tiff")
                || ext.eq_ignore_ascii_case("tif")
        })
    }

    /// Checks if a file is a model weight file
    fn is_weight_file(filename: &str) -> bool
    where
        Self: Sized,
    {
        filename.ends_with(".bin")
            || filename.ends_with(".safetensors")
            || filename.ends_with(".h5")
            || filename.ends_with(".msgpack")
            || filename.ends_with(".ckpt.index")
            || filename.ends_with(".iop")
            || filename.ends_with(".gas")
    }
}

#[cfg(feature = "gcs")]
pub mod gcs;
pub mod huggingface;
pub mod ngc;

pub use gcs::GcsProvider;
pub use huggingface::HuggingFaceProvider;
pub use ngc::NgcProvider;

#[cfg(not(feature = "gcs"))]
pub mod gcs {
    use super::ModelProviderTrait;
    use crate::cache::{ModelInfo, ProviderCache};
    use crate::models::ModelProvider;
    use anyhow::{Context, Result};
    use std::{
        fs,
        path::{Path, PathBuf},
    };
    use tracing::info;

    const CACHE_ROOT_DIR_NAME: &str = "gcs";

    pub struct GcsProvider;

    pub struct GcsProviderCache;

    fn model_dir(cache_dir: &Path, model_name: &str) -> Result<PathBuf> {
        let Some(full_url) = model_name.strip_prefix("gs://") else {
            anyhow::bail!("GCS model name must be a full gs://<bucket>/<path> URL");
        };
        let (bucket, object_prefix) = full_url
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("GCS model URL must include bucket and object path"))?;
        if bucket.is_empty() || object_prefix.is_empty() {
            anyhow::bail!("GCS model URL must include bucket and object path");
        }

        let mut path = cache_dir.join(CACHE_ROOT_DIR_NAME).join(bucket);
        for component in object_prefix.trim_end_matches('/').split('/') {
            if component.is_empty() || component == "." || component == ".." {
                anyhow::bail!("GCS model path must not contain empty or relative path segments");
            }
            path = path.join(component);
        }

        Ok(path)
    }

    fn canonical_model_name(model_name: &str) -> Result<String> {
        let path = model_dir(Path::new(""), model_name)?;
        let normalized = path
            .strip_prefix(CACHE_ROOT_DIR_NAME)
            .context("Failed to normalize GCS model name")?
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        Ok(format!("gs://{normalized}"))
    }

    #[async_trait::async_trait]
    impl ModelProviderTrait for GcsProvider {
        async fn download_model(
            &self,
            _model_name: &str,
            _cache_dir: Option<PathBuf>,
            _ignore_weights: bool,
        ) -> Result<PathBuf> {
            anyhow::bail!("GCS downloads are disabled; rebuild with the `gcs` feature")
        }

        async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
            let path = model_dir(&cache_dir, model_name)?;
            if path.exists() {
                fs::remove_dir_all(&path).with_context(|| {
                    format!("Failed to remove GCS cache directory '{}'", path.display())
                })?;
                info!(
                    "Deleted cached GCS model '{}' from '{}'",
                    model_name,
                    path.display()
                );
            }
            Ok(())
        }

        async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
            let path = model_dir(&cache_dir, model_name)?;
            if !path.is_dir() {
                anyhow::bail!("GCS model '{model_name}' not found in cache");
            }
            Ok(path)
        }

        fn canonical_model_name(&self, model_name: &str) -> Result<String> {
            canonical_model_name(model_name)
        }

        fn provider_name(&self) -> &'static str {
            "GCS"
        }
    }

    impl ProviderCache for GcsProviderCache {
        fn clear_model(&self, cache_root: &Path, model_name: &str) -> Result<()> {
            let path = model_dir(cache_root, model_name)?;
            if path.exists() {
                fs::remove_dir_all(&path).with_context(|| {
                    format!("Failed to remove GCS cache directory '{}'", path.display())
                })?;
            }
            Ok(())
        }

        fn resolve_model_path(
            &self,
            cache_root: &Path,
            model_name: &str,
            _revision: Option<&str>,
        ) -> Result<PathBuf> {
            model_dir(cache_root, model_name)
        }

        fn list_models(&self, _cache_root: &Path) -> Result<Vec<ModelInfo>> {
            Ok(Vec::new())
        }
    }

    #[allow(dead_code)]
    const _: ModelProvider = ModelProvider::Gcs;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_is_image_function() {
        assert!(HuggingFaceProvider::is_image(Path::new("test.png")));
        assert!(HuggingFaceProvider::is_image(Path::new("test.PNG")));
        assert!(HuggingFaceProvider::is_image(Path::new("test.jpg")));
        assert!(HuggingFaceProvider::is_image(Path::new("test.JPG")));
        assert!(HuggingFaceProvider::is_image(Path::new("test.jpeg")));
        assert!(HuggingFaceProvider::is_image(Path::new("test.JPEG")));

        assert!(!HuggingFaceProvider::is_image(Path::new("test.txt")));
        assert!(!HuggingFaceProvider::is_image(Path::new("test.py")));
        assert!(!HuggingFaceProvider::is_image(Path::new("test")));
        assert!(!HuggingFaceProvider::is_image(Path::new("test.model")));
    }

    #[test]
    fn test_ignored_files() {
        // Dotfiles
        assert!(HuggingFaceProvider::is_ignored(".gitattributes"));
        assert!(HuggingFaceProvider::is_ignored(".gitignore"));
        assert!(HuggingFaceProvider::is_ignored(".gitkeep"));
        assert!(HuggingFaceProvider::is_ignored(".hidden"));

        // Dotfiles in subdirectories
        assert!(HuggingFaceProvider::is_ignored("subdir/.gitkeep"));
        assert!(HuggingFaceProvider::is_ignored("a/b/.hidden"));

        // Explicit files
        assert!(HuggingFaceProvider::is_ignored("README.md"));
        assert!(HuggingFaceProvider::is_ignored("subdir/README.md"));

        // (Not Ignored) Regular files
        assert!(!HuggingFaceProvider::is_ignored("model.bin"));
        assert!(!HuggingFaceProvider::is_ignored("tokenizer.json"));
        assert!(!HuggingFaceProvider::is_ignored("config.json"));
    }

    #[test]
    fn test_is_weight_file() {
        assert!(HuggingFaceProvider::is_weight_file("model.bin"));
        assert!(HuggingFaceProvider::is_weight_file("model.safetensors"));
        assert!(HuggingFaceProvider::is_weight_file("model.h5"));
        assert!(HuggingFaceProvider::is_weight_file("model.msgpack"));
        assert!(HuggingFaceProvider::is_weight_file("model.ckpt.index"));
        assert!(HuggingFaceProvider::is_weight_file("model.iop"));
        assert!(HuggingFaceProvider::is_weight_file("model.gas"));

        assert!(!HuggingFaceProvider::is_weight_file("tokenizer.json"));
        assert!(!HuggingFaceProvider::is_weight_file("config.json"));
        assert!(!HuggingFaceProvider::is_weight_file("README.md"));
    }

    #[test]
    fn test_canonical_model_name_default_preserves_input() {
        let provider = HuggingFaceProvider;
        let canonical = provider.canonical_model_name("test/model");
        assert!(
            canonical
                .as_ref()
                .is_ok_and(|model_name| model_name == "test/model"),
            "Expected canonical model name, got {canonical:?}"
        );
    }

    #[test]
    fn test_canonical_model_name_gcs_trims_trailing_slash() {
        let provider = GcsProvider;
        let canonical = provider.canonical_model_name("gs://test-bucket/org/model/rev-1/");
        assert!(
            canonical
                .as_ref()
                .is_ok_and(|model_name| model_name == "gs://test-bucket/org/model/rev-1"),
            "Expected canonical model name, got {canonical:?}"
        );
    }
}
