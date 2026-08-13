// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::path::PathBuf;

/// Result of a model download.
///
/// `resolved_revision` is the immutable revision the request resolved to (a commit SHA
/// for Hugging Face). It is `None` for providers with no revision concept, so callers
/// can tell "not applicable" apart from "unknown".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadOutcome {
    /// Directory the model snapshot was downloaded into.
    pub path: PathBuf,
    /// Immutable revision the request resolved to, when the provider has one.
    pub resolved_revision: Option<String>,
}

/// Reject a pinned revision for providers that do not support one, so every
/// unsupported-provider error reads the same.
pub fn reject_unsupported_revision(provider_name: &str, revision: Option<&str>) -> Result<()> {
    match revision {
        Some(revision) => anyhow::bail!(
            "Provider '{provider_name}' does not support pinned revisions (requested '{revision}')"
        ),
        None => Ok(()),
    }
}

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

    /// Download a model at a specific revision, reporting the revision it resolved to.
    ///
    /// Providers that expose revisions override this and treat [`Self::download_model`]
    /// as the unpinned special case.
    async fn download_model_revision(
        &self,
        model_name: &str,
        cache_path: Option<PathBuf>,
        ignore_weights: bool,
        revision: Option<&str>,
    ) -> Result<ModelDownloadOutcome> {
        reject_unsupported_revision(self.provider_name(), revision)?;
        let path = self
            .download_model(model_name, cache_path, ignore_weights)
            .await?;
        Ok(ModelDownloadOutcome {
            path,
            resolved_revision: None,
        })
    }

    /// Whether this provider has a revision concept at all.
    ///
    /// Lets callers reject a pinned revision as a bad request up front, rather than
    /// discovering it as a resolution failure that reads like a missing revision.
    fn supports_revisions(&self) -> bool {
        false
    }

    /// Resolve a requested branch, tag, or revision identifier to an immutable one.
    ///
    /// A `None` request means "the provider's default revision". Providers with a
    /// revision concept always return `Some`, including for the default revision, so
    /// callers can pin cache and lease identity to it.
    ///
    /// This must never fall back to the default revision when the requested one does
    /// not exist.
    async fn resolve_revision(
        &self,
        _model_name: &str,
        _cache_dir: Option<PathBuf>,
        revision: Option<&str>,
    ) -> Result<Option<String>> {
        reject_unsupported_revision(self.provider_name(), revision)?;
        Ok(None)
    }

    /// Record locally that the caller's `requested_revision` — or the provider's default
    /// revision when `None` — names `commit`.
    ///
    /// A snapshot installed by streaming from the ModelExpress server never went through
    /// this provider's downloader, so the cache bookkeeping the provider's own tooling
    /// expects is missing. Providers with a revision concept implement this to complete
    /// the install; the default is a no-op.
    async fn record_local_revision(
        &self,
        _model_name: &str,
        _cache_dir: &std::path::Path,
        _requested_revision: Option<&str>,
        _commit: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Delete a model from the provider's cache
    /// Returns Ok(()) if the model was successfully deleted or didn't exist
    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()>;

    /// Delete a single revision of a model from the provider's cache.
    /// A `None` revision deletes everything cached for the model.
    async fn delete_model_revision(
        &self,
        model_name: &str,
        cache_dir: PathBuf,
        revision: Option<&str>,
    ) -> Result<()> {
        reject_unsupported_revision(self.provider_name(), revision)?;
        self.delete_model(model_name, cache_dir).await
    }

    /// Get the full path to the latest model snapshot if it exists
    /// Returns the path if found, or an error if not found
    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf>;

    /// Get the full path to a specific model revision, or to the latest snapshot when no
    /// revision is requested. Returns an error if the revision is not cached.
    async fn get_model_path_revision(
        &self,
        model_name: &str,
        cache_dir: PathBuf,
        revision: Option<&str>,
    ) -> Result<PathBuf> {
        reject_unsupported_revision(self.provider_name(), revision)?;
        self.get_model_path(model_name, cache_dir).await
    }

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
        is_weight_file(filename)
    }
}

/// Provider-agnostic check for whether a file is a model weight file.
///
/// Shared by the provider trait's `is_weight_file` default and by the server's
/// file-streaming path so both use the same definition of "weight file".
pub fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
        || filename.ends_with(".iop")
        || filename.ends_with(".gas")
}

#[cfg(feature = "gcs")]
pub mod gcs;
pub mod huggingface;
pub(crate) mod lock_file;
pub mod ngc;

pub use gcs::GcsProvider;
pub use huggingface::HuggingFaceProvider;
pub use ngc::NgcProvider;

#[cfg(not(feature = "gcs"))]
pub mod gcs {
    //! Stub provider compiled when the `gcs` feature is off. `ModelProvider::Gcs`
    //! is part of the gRPC/serde contract, so the variant always exists and these
    //! types must too; without the feature every GCS operation just reports that
    //! it's disabled. No path/cache logic lives here, that belongs to the real
    //! `gcs.rs` module behind the feature.
    use super::ModelProviderTrait;
    use crate::cache::{ModelInfo, ProviderCache};
    use anyhow::Result;
    use std::path::{Path, PathBuf};

    const FEATURE_DISABLED: &str = "GCS support is disabled; rebuild with the `gcs` feature";

    pub struct GcsProvider;

    pub struct GcsProviderCache;

    #[async_trait::async_trait]
    impl ModelProviderTrait for GcsProvider {
        async fn download_model(
            &self,
            _model_name: &str,
            _cache_dir: Option<PathBuf>,
            _ignore_weights: bool,
        ) -> Result<PathBuf> {
            anyhow::bail!(FEATURE_DISABLED)
        }

        async fn delete_model(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<()> {
            anyhow::bail!(FEATURE_DISABLED)
        }

        async fn get_model_path(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<PathBuf> {
            anyhow::bail!(FEATURE_DISABLED)
        }

        fn provider_name(&self) -> &'static str {
            "GCS"
        }
    }

    impl ProviderCache for GcsProviderCache {
        fn clear_model(&self, _cache_root: &Path, _model_name: &str) -> Result<()> {
            anyhow::bail!(FEATURE_DISABLED)
        }

        fn resolve_model_path(
            &self,
            _cache_root: &Path,
            _model_name: &str,
            _revision: Option<&str>,
        ) -> Result<PathBuf> {
            anyhow::bail!(FEATURE_DISABLED)
        }

        // Returns empty so listing every provider's cache stays infallible in a
        // build without GCS; there are no GCS models to report.
        fn list_models(&self, _cache_root: &Path) -> Result<Vec<ModelInfo>> {
            Ok(Vec::new())
        }
    }
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

    #[cfg(feature = "gcs")]
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

    #[cfg(feature = "gcs")]
    #[test]
    fn test_gcs_rejects_path_traversal_segments() {
        let provider = GcsProvider;
        let escapes = [
            "gs://bucket/org/../../../etc/passwd",
            "gs://bucket/org/./model",
            "gs://bucket/org//model",
            r"gs://bucket/org\..\etc\passwd",
            "gs://bucket/C:/windows/system32",
        ];
        for model_name in escapes {
            let result = provider.canonical_model_name(model_name);
            assert!(
                result.is_err(),
                "Expected '{model_name}' to be rejected, got {result:?}"
            );
        }
    }
}
