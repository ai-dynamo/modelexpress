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
    async fn delete_model(&self, model_name: &str) -> Result<()>;

    /// Get the full path to the latest model snapshot if it exists
    /// Returns the path if found, or an error if not found
    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf>;

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
    }
}

pub mod huggingface;
pub mod model_streamer;

pub use huggingface::HuggingFaceProvider;
pub use model_streamer::ModelStreamerProvider;

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

        assert!(!HuggingFaceProvider::is_weight_file("tokenizer.json"));
        assert!(!HuggingFaceProvider::is_weight_file("config.json"));
        assert!(!HuggingFaceProvider::is_weight_file("README.md"));
    }
}
