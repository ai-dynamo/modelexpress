// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Utils, constants, providers::ModelProviderTrait};
use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";
const HF_HUB_CACHE_ENV_VAR: &str = "HF_HUB_CACHE";
const MODEL_EXPRESS_CACHE_ENV_VAR: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";
const HF_HUB_OFFLINE_ENV_VAR: &str = "HF_HUB_OFFLINE";

/// Check if offline mode is enabled via HF_HUB_OFFLINE environment variable.
/// The variable is considered enabled if its value is one of: "1", "ON", "YES", "TRUE" (case-insensitive).
fn is_offline_mode() -> bool {
    env::var(HF_HUB_OFFLINE_ENV_VAR)
        .map(|v| matches!(v.to_uppercase().as_str(), "1" | "ON" | "YES" | "TRUE"))
        .unwrap_or(false)
}

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

impl HuggingFaceProvider {
    /// Determine whether the provided filename refers to a file that lives in a sub-directory.
    /// Hugging Face repositories can contain nested folders, but those are never files
    /// we use to run the model, so Model Express ignores them.
    fn is_subdirectory_file(filename: &str) -> bool {
        Path::new(filename).components().count() > 1
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for HuggingFaceProvider {
    /// Attempt to download a model from Hugging Face.
    /// Returns the directory it is in.
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        let cache_dir = get_cache_dir(cache_dir);
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            anyhow::anyhow!("Failed to create cache directory {:?}: {}", cache_dir, e)
        })?;

        if is_offline_mode() {
            info!("HF_HUB_OFFLINE is set, using cached model for '{model_name}'");
            return self.get_model_path(model_name, cache_dir).await;
        }

        let token = env::var(HF_TOKEN_ENV_VAR).ok();

        info!("Using cache directory: {:?}", cache_dir);
        // High CPU download
        //
        // This may cause issues on regular desktops as it will saturate
        // CPUs by multiplexing the downloads.
        // However in data-center focused environments with model express
        // this may help saturate the bandwidth (>500MB/s) better.
        let api = ApiBuilder::from_env()
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
        debug!("Got model info: {info:?}");

        if info.siblings.is_empty() {
            anyhow::bail!("Model '{model_name}' exists but contains no downloadable files.");
        }

        let mut p = PathBuf::new();
        let mut files_downloaded = false;

        for sib in info.siblings {
            if HuggingFaceProvider::is_subdirectory_file(&sib.rfilename) {
                continue;
            }

            if HuggingFaceProvider::is_ignored(&sib.rfilename)
                || HuggingFaceProvider::is_image(Path::new(&sib.rfilename))
            {
                continue;
            }

            if ignore_weights && HuggingFaceProvider::is_weight_file(&sib.rfilename) {
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
            if HuggingFaceProvider::is_subdirectory_file(&sib.rfilename) {
                continue;
            }

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

    /// Get the full path to the latest model snapshot if it exists.
    /// Returns the path if found, or an error if not found.
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

        // Sort by creation/modification time to get the most recent snapshot
        files.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.created().or_else(|_| m.modified()))
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        files.reverse();

        // In offline mode, skip network validation and return the latest local snapshot
        if is_offline_mode() {
            return Ok(files[0].path());
        }

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
            Returning the best-effort, latest local model snapshot.",
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

    /// Minimal mock of the Hugging Face Hub used by tests.
    ///
    /// This server stubs:
    /// - the model info endpoint (`/api/models/<repo>`), returning a fixed `sha` and file list
    /// - the file resolve endpoints (`/<repo>/resolve/<rev>/<filename>`) for each sibling
    ///
    /// The hf_hub client writes files into `cache_path` when the resolve endpoints return
    /// successful responses with the headers it expects (ETag, commit, range). This allows
    /// us to simulate a real model download without external network access.
    struct MockHFServer {
        /// WireMock instance; keeps the server alive for the lifetime of the test
        _server: MockServer,
        /// Temporary HF cache root that tests pass to `ApiBuilder::with_cache_dir`
        pub cache_path: PathBuf,
    }

    impl MockHFServer {
        /// Start a WireMock server and configure stubs compatible with hf_hub's download flow.
        ///
        /// Notes on headers and status codes expected by hf_hub:
        /// - `etag`: used for dedup and cache validation
        /// - `x-repo-commit`: identifies the snapshot commit (must match `info.sha`)
        /// - Range download: GETs may be partial; we return 206 with `accept-ranges`,
        ///   `content-length` and `content-range` to keep the client happy across versions.
        async fn new() -> Self {
            let temp_dir = TempDir::new().expect("Failed to create temporary directory");
            let server = MockServer::start().await;

            // Return the desired sha we want get_model_path to pick
            // Matches GET /api/models/test/model (and subpaths).
            Mock::given(method("GET"))
                .and(path_regex(r"^/api/models/test/model(?:/.*)?$"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                     "id": "test/model",
                     "sha": "def5678",
                     "siblings": [
                         {"rfilename": "config.json"},
                         {"rfilename": "model.safetensors"},
                         {"rfilename": "tokenizer.json"},
                         {"rfilename": "README.md"},
                         {"rfilename": "subdir/model.safetensors"}
                     ]
                })))
                .mount(&server)
                .await;

            // Mock resolved file contents so hf_hub can populate the cache
            // Matches GET /test/model/resolve/<rev>/(config.json|tokenizer.json|README.md|model.safetensors)
            Mock::given(method("GET"))
                .and(path_regex(r"^/test/model/resolve/(main|[^/]+)/(?:config\.json|tokenizer\.json|README\.md|model\.safetensors)$"))
                .respond_with(
                    ResponseTemplate::new(206)
                        .insert_header("etag", "\"def5678\"")
                        .insert_header("x-repo-commit", "def5678")
                        .insert_header("accept-ranges", "bytes")
                        .insert_header("content-length", "64")
                        .insert_header("content-range", "bytes 0-63/64")
                        .set_body_bytes(vec![0u8; 64]),
                )
                .mount(&server)
                .await;

            unsafe {
                std::env::set_var("HF_ENDPOINT", server.uri());
            }

            Self {
                _server: server,
                cache_path: temp_dir.path().to_path_buf(),
            }
        }
    }

    impl Drop for MockHFServer {
        /// Ensure the temporary cache path is removed even if a test fails.
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.cache_path).unwrap_or_else(|e| {
                warn!("Failed to remove temporary cache path: {e}");
            });
        }
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
        let mock_server = MockHFServer::new().await;

        // Construct a temporary cache dir with a model snapshots
        let path = mock_server
            .cache_path
            .join("models--test--model")
            .join("snapshots");

        std::fs::create_dir_all(path.join("abc1234")).expect("Failed to create directory");
        tokio::time::sleep(Duration::from_secs(1)).await;
        std::fs::create_dir_all(path.join("def5678")).expect("Failed to create directory");

        let provider = HuggingFaceProvider;
        let result = provider
            .get_model_path("test/model", mock_server.cache_path.clone())
            .await;

        assert!(result.is_ok());
        assert_eq!(
            result.expect("Failed to get model path"),
            path.join("def5678")
        );
    }

    #[tokio::test]
    async fn test_download_ignore_weights() {
        let mock_server = MockHFServer::new().await;
        let provider = HuggingFaceProvider;
        let result = provider
            .download_model("test/model", Some(mock_server.cache_path.clone()), false)
            .await
            .expect("Failed to download model");

        let files = fs::read_dir(result)
            .expect("Failed to read directory")
            .filter_map(Result::ok);

        for file in files {
            info!("File: {}", file.path().display());
            assert!(!file.path().ends_with("safetensors"));
        }
    }

    #[tokio::test]
    async fn test_download_ignores_subdirectories() {
        let mock_server = MockHFServer::new().await;
        let provider = HuggingFaceProvider;

        let result = provider
            .download_model("test/model", Some(mock_server.cache_path.clone()), false)
            .await
            .expect("Failed to download model");

        assert!(
            !result.join("subdir").exists(),
            "Expected files located in sub-directories to be ignored"
        );
    }

    #[test]
    fn test_is_offline_mode() {
        unsafe {
            env::set_var(HF_HUB_OFFLINE_ENV_VAR, "1");
            assert!(is_offline_mode());

            env::set_var(HF_HUB_OFFLINE_ENV_VAR, "0");
            assert!(!is_offline_mode());

            env::remove_var(HF_HUB_OFFLINE_ENV_VAR);
        }
        assert!(!is_offline_mode());
    }

    #[tokio::test]
    async fn test_download_model_offline_mode_with_cache() {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let snapshots_path = temp_dir
            .path()
            .join("models--test--model")
            .join("snapshots")
            .join("abc1234");
        std::fs::create_dir_all(&snapshots_path).expect("Failed to create directory");

        unsafe {
            env::set_var(HF_HUB_OFFLINE_ENV_VAR, "1");
        }

        let result = HuggingFaceProvider
            .download_model("test/model", Some(temp_dir.path().into()), false)
            .await;

        unsafe {
            env::remove_var(HF_HUB_OFFLINE_ENV_VAR);
        }

        assert!(result.is_ok());
        assert!(result.expect("Expected path").ends_with("abc1234"));
    }

    #[tokio::test]
    async fn test_download_model_offline_mode_without_cache() {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");

        unsafe {
            env::set_var(HF_HUB_OFFLINE_ENV_VAR, "1");
        }

        let result = HuggingFaceProvider
            .download_model("nonexistent/model", Some(temp_dir.path().into()), false)
            .await;

        unsafe {
            env::remove_var(HF_HUB_OFFLINE_ENV_VAR);
        }

        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected error")
                .to_string()
                .contains("not found in cache")
        );
    }
}
