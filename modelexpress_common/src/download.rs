// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::models::ModelProvider;
use crate::providers::{GcsProvider, HuggingFaceProvider, ModelProviderTrait, NgcProvider};
use anyhow::Result;
use std::path::PathBuf;
use tracing::{info, warn};

/// Provider factory to get the appropriate provider implementation
#[must_use]
pub fn get_provider(provider: ModelProvider) -> Box<dyn ModelProviderTrait> {
    match provider {
        ModelProvider::HuggingFace => Box::new(HuggingFaceProvider),
        ModelProvider::Ngc => Box::new(NgcProvider),
        ModelProvider::Gcs => Box::new(GcsProvider),
    }
}

/// Canonicalize a model name using the provider-specific rules.
pub fn canonical_model_name(model_name: &str, provider: ModelProvider) -> Result<String> {
    get_provider(provider).canonical_model_name(model_name)
}

/// Download a model using the specified provider
pub async fn download_model(
    model_name: &str,
    provider: ModelProvider,
    cache_dir: Option<PathBuf>,
    ignore_weights: bool,
) -> Result<PathBuf> {
    let provider_impl = get_provider(provider);
    info!(
        "Downloading model '{}' using provider: {}",
        model_name,
        provider_impl.provider_name()
    );

    if ignore_weights {
        warn!("`ignore_weights` is set to true. All the model weight files will be ignored!");
    }

    provider_impl
        .download_model(model_name, cache_dir, ignore_weights)
        .await
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Mock provider for testing
    struct MockProvider {
        should_succeed: bool,
        return_path: PathBuf,
    }

    #[async_trait::async_trait]
    impl ModelProviderTrait for MockProvider {
        async fn download_model(
            &self,
            _model_name: &str,
            _cache_dir: Option<PathBuf>,
            _ignore_weights: bool,
        ) -> Result<PathBuf> {
            if self.should_succeed {
                Ok(self.return_path.clone())
            } else {
                Err(anyhow::anyhow!("Mock download failed"))
            }
        }

        async fn delete_model(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<()> {
            if self.should_succeed {
                Ok(())
            } else {
                Err(anyhow::anyhow!("Mock delete failed"))
            }
        }

        async fn get_model_path(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<PathBuf> {
            if self.should_succeed {
                Ok(self.return_path.clone())
            } else {
                Err(anyhow::anyhow!("Mock get_model_path failed"))
            }
        }

        fn provider_name(&self) -> &'static str {
            "Mock Provider"
        }

        fn is_ignored(_filename: &str) -> bool {
            false // Mock provider doesn't ignore any files
        }

        fn is_image(_path: &std::path::Path) -> bool {
            false // Mock provider doesn't consider any files as images
        }
    }

    #[test]
    fn test_get_provider() {
        let provider = get_provider(ModelProvider::HuggingFace);
        assert_eq!(provider.provider_name(), "Hugging Face");

        let provider = get_provider(ModelProvider::Ngc);
        assert_eq!(provider.provider_name(), "NGC");

        let provider = get_provider(ModelProvider::Gcs);
        assert_eq!(provider.provider_name(), "GCS");
    }

    #[test]
    fn test_canonical_model_name_routing() {
        assert_eq!(
            canonical_model_name("test/model", ModelProvider::HuggingFace)
                .expect("Expected canonical model name"),
            "test/model"
        );
        assert_eq!(
            canonical_model_name("gs://test-bucket/org/model/rev-1/", ModelProvider::Gcs)
                .expect("Expected canonical model name"),
            "gs://test-bucket/org/model/rev-1"
        );
    }

    #[tokio::test]
    async fn test_mock_provider_success() {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let mock_provider = MockProvider {
            should_succeed: true,
            return_path: temp_dir.path().to_path_buf(),
        };

        let result = mock_provider
            .download_model("test-model", Some(temp_dir.path().to_path_buf()), false)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.expect("Expected successful result"), temp_dir.path());
    }

    #[tokio::test]
    async fn test_mock_provider_failure() {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let mock_provider = MockProvider {
            should_succeed: false,
            return_path: temp_dir.path().to_path_buf(),
        };

        let result = mock_provider
            .download_model("test-model", Some(temp_dir.path().to_path_buf()), false)
            .await;
        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected error result")
                .to_string()
                .contains("Mock download failed")
        );
    }

    #[test]
    fn test_default_trait_implementations() {
        // Create a minimal provider that uses default implementations
        struct DefaultProvider;

        #[async_trait::async_trait]
        impl ModelProviderTrait for DefaultProvider {
            async fn download_model(
                &self,
                _model_name: &str,
                _cache_dir: Option<PathBuf>,
                _ignore_weights: bool,
            ) -> Result<PathBuf> {
                Ok(PathBuf::from("/tmp"))
            }

            async fn delete_model(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<()> {
                Ok(())
            }

            async fn get_model_path(
                &self,
                _model_name: &str,
                _cache_dir: PathBuf,
            ) -> Result<PathBuf> {
                Ok(PathBuf::from("/tmp"))
            }

            fn provider_name(&self) -> &'static str {
                "Default Provider"
            }
            // Note: is_ignored and is_image are not implemented, so they use defaults
        }

        // Test default is_ignored behavior - dotfiles
        assert!(DefaultProvider::is_ignored(".gitattributes"));
        assert!(DefaultProvider::is_ignored(".gitignore"));
        assert!(DefaultProvider::is_ignored(".gitkeep"));
        assert!(DefaultProvider::is_ignored(".hidden"));

        // Test default is_ignored behavior - explicit files
        assert!(DefaultProvider::is_ignored("README.md"));

        // Test default is_ignored behavior - regular files
        assert!(!DefaultProvider::is_ignored("LICENSE"));
        assert!(!DefaultProvider::is_ignored("model.bin"));
        assert!(!DefaultProvider::is_ignored("config.json"));

        // Test default is_image behavior
        use std::path::Path;
        assert!(DefaultProvider::is_image(Path::new("test.png")));
        assert!(DefaultProvider::is_image(Path::new("test.JPG")));
        assert!(DefaultProvider::is_image(Path::new("test.gif")));
        assert!(!DefaultProvider::is_image(Path::new("test.txt")));
        assert!(!DefaultProvider::is_image(Path::new("model.bin")));
    }
}
