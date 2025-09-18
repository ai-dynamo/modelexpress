// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Utils, constants, providers::ModelProviderTrait};
use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";
const HF_HUB_CACHE_ENV_VAR: &str = "HF_HUB_CACHE";
const MODEL_EXPRESS_CACHE_ENV_VAR: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";

/// Get the cache directory for Hugging Face models
/// Priority order:
/// 1. Provided cache_dir parameter
/// 2. HF_HUB_CACHE environment variable
/// 3. Default location (~/.cache/huggingface/hub)
fn get_cache_dir(cache_dir: Option<PathBuf>) -> PathBuf {
    // Use provided cache directory if available
    if let Some(dir) = cache_dir {
        return dir;
    }

    // Try MODEL_EXPRESS_CACHE_DIRECTORY environment variable first
    if let Ok(cache_path) = env::var(MODEL_EXPRESS_CACHE_ENV_VAR) {
        return PathBuf::from(cache_path);
    }

    // Try environment variable
    if let Ok(cache_path) = env::var(HF_HUB_CACHE_ENV_VAR) {
        return PathBuf::from(cache_path);
    }

    // Fall back to default location
    let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(constants::DEFAULT_HF_CACHE_PATH)
}

/// Hugging Face model provider implementation
pub struct HuggingFaceProvider;

#[async_trait::async_trait]
impl ModelProviderTrait for HuggingFaceProvider {
    /// Attempt to download a model from Hugging Face
    /// Returns the directory it is in
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
    ) -> Result<PathBuf> {
        info!("Downloading model from Hugging Face: {model_name}");
        let token = env::var(HF_TOKEN_ENV_VAR).ok();

        // Get cache directory and ensure it exists
        let cache_dir = get_cache_dir(cache_dir);
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            anyhow::anyhow!("Failed to create cache directory {:?}: {}", cache_dir, e)
        })?;

        info!("Using cache directory: {:?}", cache_dir);
        // High CPU download
        //
        // This may cause issues on regular desktops as it will saturate
        // CPUs by multiplexing the downloads.
        // However in data-center focused environments with model express
        // this may help saturate the bandwidth (>500MB/s) better.
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(token)
            .high()
            .with_cache_dir(cache_dir)
            .build()?;
        let model_name = model_name.to_string();

        let repo = api.model(model_name.clone());

        let info = repo.info().await.map_err(
            |e| anyhow::anyhow!("Failed to fetch model '{model_name}' from HuggingFace. Is this a valid HuggingFace ID? Error: {e}"),
        )?;
        info!("Got model info: {info:?}");

        if info.siblings.is_empty() {
            anyhow::bail!("Model '{model_name}' exists but contains no downloadable files.");
        }

        let mut p = PathBuf::new();
        let mut files_downloaded = false;

        for sib in info.siblings {
            if HuggingFaceProvider::is_ignored(&sib.rfilename)
                || HuggingFaceProvider::is_image(Path::new(&sib.rfilename))
            {
                continue;
            }

            match repo.get(&sib.rfilename).await {
                Ok(path) => {
                    p = path;
                    files_downloaded = true;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to download file '{sib}' from model '{model_name}': {e}",
                        sib = sib.rfilename,
                        model_name = model_name,
                        e = e
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

        info!("Downloaded model files for {model_name}");

        match p.parent() {
            Some(p) => Ok(p.to_path_buf()),
            None => Err(anyhow::anyhow!("Invalid HF cache path: {}", p.display())),
        }
    }

    /// Attempt to delete a model from Hugging Face cache
    /// Returns Ok(()) if the model was successfully deleted or didn't exist
    async fn delete_model(&self, model_name: &str) -> Result<()> {
        info!("Deleting model from Hugging Face cache: {model_name}");
        let token = env::var(HF_TOKEN_ENV_VAR).ok();
        let api = ApiBuilder::new()
            .with_token(token)
            .build()
            .context("Failed to create Hugging Face API client")?;
        let model_name = model_name.to_string();

        let repo = api.model(model_name.clone());

        let info = match repo.info().await {
            Ok(info) => info,
            Err(_) => {
                // If we can't get model info, assume it doesn't exist or is already deleted
                info!("Model '{model_name}' not found or already deleted");
                return Ok(());
            }
        };

        if info.siblings.is_empty() {
            info!("Model '{model_name}' has no files to delete");
            return Ok(());
        }

        let mut files_deleted: u32 = 0;
        let mut deletion_errors = Vec::new();

        for sib in &info.siblings {
            if HuggingFaceProvider::is_ignored(&sib.rfilename)
                || HuggingFaceProvider::is_image(Path::new(&sib.rfilename))
            {
                continue;
            }

            // Try to get the file path from cache first
            if let Ok(cached_path) = repo.get(&sib.rfilename).await {
                // Delete the cached file
                match std::fs::remove_file(&cached_path) {
                    Ok(_) => {
                        files_deleted = files_deleted.saturating_add(1);
                        info!("Deleted cached file: {}", cached_path.display());
                    }
                    Err(e) => {
                        let error_msg =
                            format!("Failed to delete cached file '{}'", cached_path.display());
                        deletion_errors.push(anyhow::anyhow!(e).context(error_msg));
                    }
                }
            }
        }

        // Try to remove the empty model directory if all files were deleted
        if files_deleted > 0 && deletion_errors.is_empty() {
            // Get any file path to find the model directory
            for sib in &info.siblings {
                if let Ok(cached_path) = repo.get(&sib.rfilename).await
                    && let Some(model_dir) = cached_path.parent()
                    && let Ok(mut entries) = std::fs::read_dir(model_dir)
                    && entries.next().is_none()
                {
                    if let Err(e) = std::fs::remove_dir(model_dir) {
                        info!("Could not remove empty model directory: {e}");
                    } else {
                        info!("Removed empty model directory: {}", model_dir.display());
                    }
                    break;
                }
            }
        }

        if !deletion_errors.is_empty() {
            let mut compound_error =
                anyhow::anyhow!("Failed to delete some files for model '{model_name}'");

            for (i, error) in deletion_errors.into_iter().enumerate() {
                compound_error =
                    compound_error.context(format!("Error {}: {:#}", i.saturating_add(1), error));
            }

            return Err(compound_error);
        }

        if files_deleted == 0 {
            info!("No cached files found to delete for model '{model_name}'");
        } else {
            info!("Successfully deleted {files_deleted} cached files for model '{model_name}'");
        }

        Ok(())
    }

    /// Get the full path to the latest model snapshot if it exists
    /// Returns the path if found, or an error if not found
    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let normalized_name = model_name.replace("/", "--");
        let path = cache_dir
            .join(format!["models--{normalized_name}"])
            .join("snapshots");

        if !path.exists() {
            anyhow::bail!("Model snapshots for '{model_name}' not found in cache");
        }

        let mut files: Vec<fs::DirEntry> = fs::read_dir(path)?.filter_map(Result::ok).collect();
        if files.is_empty() {
            anyhow::bail!("Model snapshots for '{model_name}' is empty");
        }

        // A bit hacky way to figure out the latest snapshot
        files.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.created().or_else(|_| m.modified()))
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        files.reverse();

        // Check against the latest commit hash from HF
        let token = env::var(HF_TOKEN_ENV_VAR).ok();
        let api = ApiBuilder::from_env().with_token(token).build()?;
        let repo = api.model(model_name.to_string());
        let info = repo.info().await.map_err(|e| {
            anyhow::anyhow!("Failed to fetch model '{model_name}' from HuggingFace: {e}")
        })?;

        for file in &files {
            if file.file_name().display().to_string() == info.sha {
                return Ok(file.path());
            }
        }

        warn!(
            "Existing model snapshots do not match the latest commit hash '{0}'. \
            Returning the best-effort, latest model snapshot.",
            info.sha
        );

        Ok(files[0].path())
    }

    fn provider_name(&self) -> &'static str {
        "Hugging Face"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;
    use tokio::time::Duration;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    #[tokio::test]
    async fn test_delete_model_trait() {
        let provider = HuggingFaceProvider;
        // Test that the delete method exists and can be called
        // Note: This won't actually delete anything since we're not providing a real model
        // but it tests the trait implementation
        let result = provider.delete_model("nonexistent/model").await;
        // Should succeed (return Ok(())) even if model doesn't exist
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_model_path_trait() {
        let server = MockServer::start().await;

        // Return the desired sha we want get_model_path to pick
        Mock::given(method("GET"))
            .and(path_regex(r"^/api/models/test/model(?:/.*)?$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test/model",
                "sha": "def5678",
                "siblings": []
            })))
            .mount(&server)
            .await;

        unsafe {
            std::env::set_var("HF_ENDPOINT", server.uri());
        }

        // Construct a temporary cache dir with a model snapshots
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let path = temp_dir
            .path()
            .join("models--test--model")
            .join("snapshots");

        std::fs::create_dir_all(path.join("abc1234")).expect("Failed to create directory");
        tokio::time::sleep(Duration::from_secs(1)).await;
        std::fs::create_dir_all(path.join("def5678")).expect("Failed to create directory");

        let provider = HuggingFaceProvider;
        let result = provider
            .get_model_path("test/model", temp_dir.path().to_path_buf())
            .await;

        assert!(result.is_ok());
        assert_eq!(
            result.expect("Failed to get model path"),
            path.join("def5678")
        );
    }
}
