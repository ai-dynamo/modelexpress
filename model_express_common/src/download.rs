use crate::models::ModelProvider;
use anyhow::Result;
use hf_hub::api::tokio::ApiBuilder;
use std::env;
use std::path::{Path, PathBuf};
use tracing::info;

const IGNORED: [&str; 3] = [".gitattributes", "LICENSE", "README.md"];

const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";

/// Trait for model providers
/// This trait provides the framework for supporting multiple model providers.
/// It allows for easy extension to support providers like OpenAI, Anthropic, etc.
#[async_trait::async_trait]
pub trait ModelProviderTrait: Send + Sync {
    /// Download a model and return the path where it was downloaded
    async fn download_model(&self, model_name: &str) -> Result<PathBuf>;

    /// Get the provider name for logging
    fn provider_name(&self) -> &'static str;
}

/// Hugging Face model provider implementation
pub struct HuggingFaceProvider;

#[async_trait::async_trait]
impl ModelProviderTrait for HuggingFaceProvider {
    async fn download_model(&self, model_name: &str) -> Result<PathBuf> {
        download_from_hf(model_name).await
    }

    fn provider_name(&self) -> &'static str {
        "Hugging Face"
    }
}

/// Provider factory to get the appropriate provider implementation
pub fn get_provider(provider: ModelProvider) -> Box<dyn ModelProviderTrait> {
    match provider {
        ModelProvider::HuggingFace => Box::new(HuggingFaceProvider),
    }
}

/// Download a model using the specified provider
pub async fn download_model(model_name: &str, provider: ModelProvider) -> Result<PathBuf> {
    let provider_impl = get_provider(provider);
    info!(
        "Downloading model '{}' using provider: {}",
        model_name,
        provider_impl.provider_name()
    );
    provider_impl.download_model(model_name).await
}

/// Attempt to download a model from Hugging Face
/// Returns the directory it is in
pub async fn download_from_hf(name: impl AsRef<Path>) -> Result<PathBuf> {
    info!(
        "Downloading model from Hugging Face: {}",
        name.as_ref().display()
    );
    let name = name.as_ref();
    let token = env::var(HF_TOKEN_ENV_VAR).ok();
    let api = ApiBuilder::new()
        .with_progress(true)
        .with_token(token)
        .build()?;
    let model_name = name.display().to_string();

    let repo = api.model(model_name.clone());

    let info = match repo.info().await {
        Ok(info) => info,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to fetch model '{}' from HuggingFace: {}. Is this a valid HuggingFace ID?",
                model_name,
                e
            ));
        }
    };
    info!("Got model info: {:?}", info);

    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!(
            "Model '{}' exists but contains no downloadable files.",
            model_name
        ));
    }

    let mut p = PathBuf::new();
    let mut files_downloaded = false;

    for sib in info.siblings {
        if IGNORED.contains(&sib.rfilename.as_str()) || is_image(&sib.rfilename) {
            continue;
        }

        match repo.get(&sib.rfilename).await {
            Ok(path) => {
                p = path;
                files_downloaded = true;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to download file '{}' from model '{}': {}",
                    sib.rfilename,
                    model_name,
                    e
                ));
            }
        }
    }

    if !files_downloaded {
        return Err(anyhow::anyhow!(
            "No valid files found for model '{}'.",
            model_name
        ));
    }

    info!("Downloaded model files for {}", model_name);

    match p.parent() {
        Some(p) => Ok(p.to_path_buf()),
        None => Err(anyhow::anyhow!("Invalid HF cache path: {}", p.display())),
    }
}

fn is_image(s: &str) -> bool {
    s.ends_with(".png")
        || s.ends_with("PNG")
        || s.ends_with(".jpg")
        || s.ends_with("JPG")
        || s.ends_with(".jpeg")
        || s.ends_with("JPEG")
}

#[cfg(test)]
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
        async fn download_model(&self, _model_name: &str) -> Result<PathBuf> {
            if self.should_succeed {
                Ok(self.return_path.clone())
            } else {
                Err(anyhow::anyhow!("Mock download failed"))
            }
        }

        fn provider_name(&self) -> &'static str {
            "Mock Provider"
        }
    }

    #[test]
    fn test_is_image_function() {
        assert!(is_image("test.png"));
        assert!(is_image("test.PNG"));
        assert!(is_image("test.jpg"));
        assert!(is_image("test.JPG"));
        assert!(is_image("test.jpeg"));
        assert!(is_image("test.JPEG"));

        assert!(!is_image("test.txt"));
        assert!(!is_image("test.py"));
        assert!(!is_image("test"));
        assert!(!is_image("test.model"));
    }

    #[test]
    fn test_ignored_files() {
        assert!(IGNORED.contains(&".gitattributes"));
        assert!(IGNORED.contains(&"LICENSE"));
        assert!(IGNORED.contains(&"README.md"));
        assert_eq!(IGNORED.len(), 3);
    }

    #[test]
    fn test_get_provider() {
        let provider = get_provider(ModelProvider::HuggingFace);
        assert_eq!(provider.provider_name(), "Hugging Face");
    }

    #[tokio::test]
    async fn test_mock_provider_success() {
        let temp_dir = TempDir::new().unwrap();
        let mock_provider = MockProvider {
            should_succeed: true,
            return_path: temp_dir.path().to_path_buf(),
        };

        let result = mock_provider.download_model("test-model").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), temp_dir.path());
    }

    #[tokio::test]
    async fn test_mock_provider_failure() {
        let temp_dir = TempDir::new().unwrap();
        let mock_provider = MockProvider {
            should_succeed: false,
            return_path: temp_dir.path().to_path_buf(),
        };

        let result = mock_provider.download_model("test-model").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Mock download failed"));
    }

    #[tokio::test]
    async fn test_download_model_routing() {
        // Test that download_model function properly routes to the provider
        // Note: This test doesn't actually download from HF to avoid network dependency
        // In a real scenario, you might want to mock the hf-hub dependency

        let provider = ModelProvider::HuggingFace;
        let provider_impl = get_provider(provider);
        assert_eq!(provider_impl.provider_name(), "Hugging Face");
    }

    #[test]
    fn test_hugging_face_provider_name() {
        let provider = HuggingFaceProvider;
        assert_eq!(provider.provider_name(), "Hugging Face");
    }

    #[test]
    fn test_provider_trait_object() {
        let provider: Box<dyn ModelProviderTrait> = Box::new(HuggingFaceProvider);
        assert_eq!(provider.provider_name(), "Hugging Face");
    }
}
