// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ) -> Result<PathBuf>;

    /// Delete a model from the provider's cache
    /// Returns Ok(()) if the model was successfully deleted or didn't exist
    async fn delete_model(&self, model_name: &str) -> Result<()>;

    /// Get the full path to the latest model snapshot if it exists
    /// Returns the path if found, or an error if not found
    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf>;

    /// Get the provider name for logging
    fn provider_name(&self) -> &'static str;

    /// Check if a file should be ignored during download
    /// This allows each provider to specify which files to skip
    /// Default implementation ignores common repository metadata files
    fn is_ignored(filename: &str) -> bool
    where
        Self: Sized,
    {
        const DEFAULT_IGNORED: [&str; 3] = [".gitattributes", ".gitignore", "README.md"];
        DEFAULT_IGNORED.contains(&filename)
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
}

pub mod huggingface;

pub use huggingface::HuggingFaceProvider;

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
        assert!(HuggingFaceProvider::is_ignored(".gitattributes"));
        assert!(HuggingFaceProvider::is_ignored(".gitignore"));
        assert!(HuggingFaceProvider::is_ignored("README.md"));

        assert!(!HuggingFaceProvider::is_ignored("model.bin"));
        assert!(!HuggingFaceProvider::is_ignored("tokenizer.json"));
    }
}
