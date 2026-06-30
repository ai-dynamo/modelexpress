// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Utils,
    cache::{ModelInfo, ProviderCache, directory_size},
    constants,
    models::ModelProvider,
    providers::ModelProviderTrait,
};
use anyhow::{Context, Result};
use futures::StreamExt;
use hf_hub::api::tokio::{ApiBuilder, ApiError};
use hf_hub::{Cache, Repo, RepoType};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";
const HF_HUB_CACHE_ENV_VAR: &str = "HF_HUB_CACHE";
const MODEL_EXPRESS_CACHE_ENV_VAR: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";
const HF_HUB_OFFLINE_ENV_VAR: &str = "HF_HUB_OFFLINE";
const HF_FALLBACK_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HF_FALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(300);

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

pub(crate) struct HuggingFaceProviderCache;

impl HuggingFaceProviderCache {
    fn repo_root(cache_root: &Path, model_name: &str) -> PathBuf {
        cache_root.join(format!("models--{}", model_name.replace('/', "--")))
    }

    fn snapshots_dir(cache_root: &Path, model_name: &str) -> PathBuf {
        Self::repo_root(cache_root, model_name).join("snapshots")
    }

    fn folder_name_to_model_id(folder_name: &str) -> String {
        if let Some(stripped) = folder_name.strip_prefix("models--") {
            stripped.replace("--", "/")
        } else {
            folder_name.to_string()
        }
    }

    fn latest_local_snapshot_path(cache_root: &Path, model_name: &str) -> Result<PathBuf> {
        let path = Self::snapshots_dir(cache_root, model_name);

        if !path.exists() {
            anyhow::bail!("Model snapshots for '{model_name}' not found in cache");
        }

        let mut files: Vec<fs::DirEntry> = fs::read_dir(path)?.filter_map(Result::ok).collect();
        if files.is_empty() {
            anyhow::bail!("Model snapshots for '{model_name}' is empty");
        }

        files.sort_by_key(|entry| {
            entry
                .metadata()
                .and_then(|metadata| metadata.created().or_else(|_| metadata.modified()))
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        files.reverse();

        Ok(files[0].path())
    }
}

impl ProviderCache for HuggingFaceProviderCache {
    fn clear_model(&self, cache_root: &Path, model_name: &str) -> Result<()> {
        let model_path = Self::repo_root(cache_root, model_name);

        if model_path.exists() {
            fs::remove_dir_all(&model_path)
                .with_context(|| format!("Failed to remove model: {model_path:?}"))?;
            info!(
                "Cleared model: {} ({:?})",
                model_name,
                ModelProvider::HuggingFace
            );
        } else {
            warn!(
                "Model not found in cache: {} ({:?})",
                model_name,
                ModelProvider::HuggingFace
            );
        }

        Ok(())
    }

    fn resolve_model_path(
        &self,
        cache_root: &Path,
        model_name: &str,
        revision: Option<&str>,
    ) -> Result<PathBuf> {
        match revision {
            Some(revision) => Ok(Self::snapshots_dir(cache_root, model_name).join(revision)),
            None => Self::latest_local_snapshot_path(cache_root, model_name),
        }
    }

    fn list_models(&self, cache_root: &Path) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();

        if !cache_root.exists() {
            return Ok(models);
        }

        for entry in fs::read_dir(cache_root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let Some(folder_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };

            if !folder_name.starts_with("models--") {
                continue;
            }

            models.push(ModelInfo {
                provider: ModelProvider::HuggingFace,
                name: Self::folder_name_to_model_id(folder_name),
                size: directory_size(&path)?,
                path,
            });
        }

        Ok(models)
    }
}

impl HuggingFaceProvider {
    /// Determine whether the provided filename refers to a file that lives in a sub-directory.
    /// Hugging Face repositories can contain nested folders, but those are never files
    /// we use to run the model, so Model Express ignores them.
    fn is_subdirectory_file(filename: &str) -> bool {
        Path::new(filename).components().count() > 1
    }

    fn is_missing_content_range_error(error: &ApiError) -> bool {
        matches!(
            error,
            ApiError::MissingHeader(header)
                if header.as_str().eq_ignore_ascii_case("content-range")
        )
    }

    async fn download_full_body_file(
        url: &str,
        cache_dir: &Path,
        model_name: &str,
        commit_hash: &str,
        filename: &str,
        token: Option<&str>,
    ) -> Result<PathBuf> {
        // Compatibility shim for mirrors/CDNs that ignore hf-hub's range metadata probe.
        // Long term, this should live in hf-hub so ModelExpress can return to repo.get().
        let client = reqwest::Client::builder()
            .connect_timeout(HF_FALLBACK_CONNECT_TIMEOUT)
            .read_timeout(HF_FALLBACK_READ_TIMEOUT)
            .build()
            .context("Failed to build Hugging Face fallback HTTP client")?;
        let mut request = client.get(url).header(
            "user-agent",
            format!("modelexpress/{}", env!("CARGO_PKG_VERSION")),
        );
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("Failed to request Hugging Face file '{filename}'"))?
            .error_for_status()
            .with_context(|| format!("Failed to download Hugging Face file '{filename}'"))?;

        let response_commit = response
            .headers()
            .get("x-repo-commit")
            .ok_or_else(|| {
                anyhow::anyhow!("Full-body response for '{filename}' is missing x-repo-commit")
            })?
            .to_str()
            .with_context(|| format!("Invalid x-repo-commit header for '{filename}'"))?;
        if response_commit != commit_hash {
            anyhow::bail!(
                "Full-body response for '{filename}' came from commit '{response_commit}', expected '{commit_hash}'"
            );
        }

        let expected_size = response.content_length().ok_or_else(|| {
            anyhow::anyhow!("Full-body response for '{filename}' is missing content-length")
        })?;

        let repo_root = HuggingFaceProviderCache::repo_root(cache_dir, model_name);
        let pointer_path = repo_root.join("snapshots").join(commit_hash).join(filename);
        let snapshot_dir = pointer_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid Hugging Face snapshot path"))?;
        fs::create_dir_all(snapshot_dir)
            .with_context(|| format!("Failed to create snapshot cache for '{model_name}'"))?;

        // Write through a temp file and verify the byte count before publishing the snapshot.
        // A truncated full-body response must not become a trusted cache hit on later runs.
        let temp_file = tempfile::Builder::new()
            .prefix(".modelexpress-")
            .tempfile_in(snapshot_dir)
            .with_context(|| {
                format!(
                    "Failed to create temp cache file in '{}'",
                    snapshot_dir.display()
                )
            })?;
        let tmp_path = temp_file.path().to_path_buf();
        let mut file =
            tokio::fs::File::from_std(temp_file.reopen().with_context(|| {
                format!("Failed to open temp cache file '{}'", tmp_path.display())
            })?);
        let mut downloaded = 0_u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| {
                format!("Failed while streaming Hugging Face file '{filename}'")
            })?;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write cache file '{}'", tmp_path.display()))?;
            downloaded = downloaded
                .checked_add(u64::try_from(chunk.len())?)
                .ok_or_else(|| anyhow::anyhow!("Downloaded byte count overflowed"))?;
        }

        file.flush()
            .await
            .with_context(|| format!("Failed to flush cache file '{}'", tmp_path.display()))?;
        drop(file);

        if downloaded != expected_size {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            anyhow::bail!(
                "Downloaded {downloaded} bytes for '{filename}', expected {expected_size}"
            );
        }

        match temp_file.persist(&pointer_path) {
            Ok(_) => {}
            Err(_) if pointer_path.exists() => {}
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "Failed to commit cache file '{}': {}",
                    pointer_path.display(),
                    error.error
                ));
            }
        }
        Cache::new(cache_dir.to_path_buf())
            .model(model_name.to_string())
            .create_ref(commit_hash)
            .with_context(|| {
                format!("Failed to create Hugging Face cache ref for '{model_name}'")
            })?;

        Ok(pointer_path)
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
            .with_token(token.clone())
            .high()
            .with_cache_dir(cache_dir.clone())
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

        let pinned_repo = api.repo(Repo::with_revision(
            model_name.clone(),
            RepoType::Model,
            info.sha.clone(),
        ));
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

            let path = match repo.get(&sib.rfilename).await {
                Ok(path) => path,
                Err(e) if HuggingFaceProvider::is_missing_content_range_error(&e) => {
                    // hf-hub requires Content-Range for its size probe. Some HF mirrors return a
                    // complete 200 OK body instead, so retry this file without Range headers.
                    warn!(
                        "Hugging Face range metadata missing for '{}' from model '{}'; retrying with full-body download",
                        sib.rfilename, model_name
                    );
                    HuggingFaceProvider::download_full_body_file(
                        &pinned_repo.url(&sib.rfilename),
                        &cache_dir,
                        &model_name,
                        &info.sha,
                        &sib.rfilename,
                        token.as_deref(),
                    )
                    .await
                    .map_err(|fallback_error| {
                        anyhow::anyhow!(
                            "Failed to download file '{sib}' from model '{model_name}': {e}; full-body fallback also failed: {fallback_error:#}",
                            sib = sib.rfilename,
                            model_name = model_name,
                            e = e
                        )
                    })?
                }
                Err(e) => {
                    // HTTP 416 (Range Not Satisfiable) occurs for empty files (0 bytes)
                    // since range requests are invalid on empty content. Skip gracefully.
                    if let ApiError::RequestError(req_err) = &e
                        && req_err.status().is_some_and(|s| s.as_u16() == 416)
                    {
                        warn!(
                            "Skipping empty file '{}' from model '{}': {}",
                            sib.rfilename, model_name, e
                        );
                        continue;
                    }
                    return Err(anyhow::anyhow!(
                        "Failed to download file '{sib}' from model '{model_name}': {e}",
                        sib = sib.rfilename,
                        model_name = model_name,
                        e = e
                    ));
                }
            };

            p = path;
            files_downloaded = true;
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
    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        info!("Deleting model from Hugging Face cache: {model_name}");
        let token = env::var(HF_TOKEN_ENV_VAR).ok();
        let api = ApiBuilder::from_env()
            .with_token(token)
            .with_cache_dir(cache_dir.clone())
            .build()
            .context("Failed to create Hugging Face API client")?;
        let model_name = model_name.to_string();

        let repo = api.model(model_name.clone());
        let cache_repo = Cache::new(cache_dir).model(model_name.clone());

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
        let mut model_dirs: HashSet<PathBuf> = HashSet::new();

        for sib in &info.siblings {
            if HuggingFaceProvider::is_subdirectory_file(&sib.rfilename) {
                continue;
            }

            if HuggingFaceProvider::is_ignored(&sib.rfilename)
                || HuggingFaceProvider::is_image(Path::new(&sib.rfilename))
            {
                continue;
            }

            // Cache-only lookup: avoid network fetch during deletion.
            if let Some(cached_path) = cache_repo.get(&sib.rfilename) {
                // Delete the cached file
                match std::fs::remove_file(&cached_path) {
                    Ok(_) => {
                        files_deleted = files_deleted.saturating_add(1);
                        info!("Deleted cached file: {}", cached_path.display());
                        if let Some(model_dir) = cached_path.parent() {
                            model_dirs.insert(model_dir.to_path_buf());
                        }
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
            for model_dir in model_dirs {
                if let Ok(mut entries) = std::fs::read_dir(&model_dir)
                    && entries.next().is_none()
                {
                    if let Err(e) = std::fs::remove_dir(&model_dir) {
                        info!("Could not remove empty model directory: {e}");
                    } else {
                        info!("Removed empty model directory: {}", model_dir.display());
                    }
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
        let latest_local_snapshot =
            HuggingFaceProviderCache.resolve_model_path(&cache_dir, model_name, None)?;

        // In offline mode, skip network validation and return the latest local snapshot
        if is_offline_mode() {
            return Ok(latest_local_snapshot);
        }

        // Check against the latest commit hash from HF
        let token = env::var(HF_TOKEN_ENV_VAR).ok();
        let api = ApiBuilder::from_env().with_token(token).build()?;
        let repo = api.model(model_name.to_string());
        let info = repo.info().await.map_err(|e| {
            anyhow::anyhow!("Failed to fetch model '{model_name}' from HuggingFace: {e}")
        })?;

        let latest_remote_snapshot =
            HuggingFaceProviderCache.resolve_model_path(&cache_dir, model_name, Some(&info.sha))?;
        if latest_remote_snapshot.exists() {
            return Ok(latest_remote_snapshot);
        }

        warn!(
            "Existing model snapshots do not match the latest commit hash '{0}'. \
            Returning the best-effort, latest local model snapshot.",
            info.sha
        );

        Ok(latest_local_snapshot)
    }

    fn provider_name(&self) -> &'static str {
        "Hugging Face"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, acquire_env_mutex};
    use serde_json::json;
    use std::sync::MutexGuard;
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
    struct MockHFServer<'a> {
        /// WireMock instance; keeps the server alive for the lifetime of the test
        _server: MockServer,
        /// Temporary HF cache root that tests pass to `ApiBuilder::with_cache_dir`
        pub cache_path: PathBuf,
        /// Restores HF_ENDPOINT when the mock server is dropped.
        _hf_endpoint_guard: EnvVarGuard<'a>,
    }

    impl<'a> MockHFServer<'a> {
        /// Start a WireMock server and configure stubs compatible with hf_hub's download flow.
        ///
        /// Notes on headers and status codes expected by hf_hub:
        /// - `etag`: used for dedup and cache validation
        /// - `x-repo-commit`: identifies the snapshot commit (must match `info.sha`)
        /// - Range download: GETs may be partial; we return 206 with `accept-ranges`,
        ///   `content-length` and `content-range` to keep the client happy across versions.
        async fn new(env_lock: &'a MutexGuard<'static, ()>) -> Self {
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

            let hf_endpoint_guard = EnvVarGuard::set(env_lock, "HF_ENDPOINT", &server.uri());

            Self {
                _server: server,
                cache_path: temp_dir.path().to_path_buf(),
                _hf_endpoint_guard: hf_endpoint_guard,
            }
        }
    }

    impl Drop for MockHFServer<'_> {
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
        let cache_dir = TempDir::new().expect("Failed to create temporary cache directory");
        // Test that the delete method exists and can be called
        // Note: This won't actually delete anything since we're not providing a real model
        // but it tests the trait implementation
        let result = provider
            .delete_model("nonexistent/model", cache_dir.path().to_path_buf())
            .await;
        // Should succeed (return Ok(())) even if model doesn't exist
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_accepts_full_body_response_without_content_range() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let server = MockServer::start().await;
        let model_name = "test/model";
        let file_contents = b"license";

        Mock::given(method("GET"))
            .and(path_regex(r"^/api/models/test/model(?:/.*)?$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                 "id": model_name,
                 "sha": "def5678",
                 "siblings": [
                     {"rfilename": "LICENSE"}
                 ]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(r"^/test/model/resolve/main/LICENSE$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("etag", "\"license-etag\"")
                    .insert_header("x-repo-commit", "def5678")
                    .insert_header("accept-ranges", "bytes")
                    .insert_header("content-length", file_contents.len().to_string())
                    .set_body_bytes(file_contents.to_vec()),
            )
            .expect(1)
            .named("hf-hub range probe without content-range")
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(r"^/test/model/resolve/def5678/LICENSE$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("etag", "\"license-etag\"")
                    .insert_header("x-repo-commit", "def5678")
                    .insert_header("accept-ranges", "bytes")
                    .insert_header("content-length", file_contents.len().to_string())
                    .set_body_bytes(file_contents.to_vec()),
            )
            .expect(1)
            .named("commit-pinned full-body fallback")
            .mount(&server)
            .await;

        let _hf_endpoint_guard = EnvVarGuard::set(&env_lock, "HF_ENDPOINT", &server.uri());

        let snapshot = HuggingFaceProvider
            .download_model(model_name, Some(temp_dir.path().to_path_buf()), false)
            .await
            .expect("Download should accept full-body responses");

        assert_eq!(
            fs::read(snapshot.join("LICENSE")).expect("Expected downloaded file"),
            file_contents
        );
        assert_eq!(
            fs::read_to_string(temp_dir.path().join("models--test--model/refs/main"))
                .expect("Expected cache ref"),
            "def5678"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_delete_model_prefers_explicit_cache_dir_over_env() {
        let env_lock = acquire_env_mutex();
        let mock_server = MockHFServer::new(&env_lock).await;
        let provider = HuggingFaceProvider;

        let explicit_cache = TempDir::new().expect("Failed to create explicit cache directory");
        let env_cache = TempDir::new().expect("Failed to create env cache directory");

        let explicit_snapshot = provider
            .download_model(
                "test/model",
                Some(explicit_cache.path().to_path_buf()),
                false,
            )
            .await
            .expect("Failed to seed explicit cache");

        let explicit_config = explicit_snapshot.join("config.json");
        assert!(
            explicit_config.exists(),
            "Expected explicit cache to contain model file before delete"
        );

        let env_snapshot = provider
            .download_model("test/model", Some(env_cache.path().to_path_buf()), false)
            .await
            .expect("Failed to seed env cache");
        let env_config = env_snapshot.join("config.json");
        assert!(
            env_config.exists(),
            "Expected env cache to contain model file before delete"
        );

        let env_cache_path = env_cache.path().to_str().expect("Expected env cache path");
        let _model_express_cache_guard =
            EnvVarGuard::set(&env_lock, MODEL_EXPRESS_CACHE_ENV_VAR, env_cache_path);
        let _hf_hub_cache_guard = EnvVarGuard::set(&env_lock, HF_HUB_CACHE_ENV_VAR, env_cache_path);

        let delete_result = provider
            .delete_model("test/model", explicit_cache.path().to_path_buf())
            .await;

        assert!(
            delete_result.is_ok(),
            "Delete request should succeed: {delete_result:?}"
        );
        assert!(
            !explicit_config.exists(),
            "Expected explicit cache file to be deleted when explicit cache dir is provided"
        );
        assert!(
            env_config.exists(),
            "Expected env cache file to remain untouched when explicit cache dir is provided"
        );

        drop(mock_server);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_delete_model_does_not_download_uncached_files() {
        let env_lock = acquire_env_mutex();
        let temp_cache = TempDir::new().expect("Failed to create temporary cache directory");
        let server = MockServer::start().await;
        let provider = HuggingFaceProvider;
        let model_name = "modelexpress-tests/delete-no-download";

        Mock::given(method("GET"))
            .and(path_regex(
                r"^/api/models/modelexpress-tests/delete-no-download(?:/.*)?$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": model_name,
                "sha": "def5678",
                "siblings": [
                    {"rfilename": "config.json"}
                ]
            })))
            .expect(1)
            .named("delete_model should query repo info exactly once")
            .mount(&server)
            .await;

        // Deleting uncached files must not trigger any remote resolve/download requests.
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/modelexpress-tests/delete-no-download/resolve/(main|[^/]+)/config\.json$",
            ))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("etag", "\"def5678\"")
                    .insert_header("x-repo-commit", "def5678")
                    .insert_header("accept-ranges", "bytes")
                    .insert_header("content-length", "64")
                    .insert_header("content-range", "bytes 0-63/64")
                    .set_body_bytes(vec![0u8; 64]),
            )
            .expect(0)
            .named("delete_model should not call Hugging Face resolve endpoint")
            .mount(&server)
            .await;

        let temp_cache_path = temp_cache.path().to_str().expect("Expected cache path");
        let _model_express_cache_guard =
            EnvVarGuard::set(&env_lock, MODEL_EXPRESS_CACHE_ENV_VAR, temp_cache_path);
        let _hf_endpoint_guard = EnvVarGuard::set(&env_lock, "HF_ENDPOINT", &server.uri());

        let result = provider
            .delete_model(model_name, temp_cache.path().to_path_buf())
            .await;

        assert!(result.is_ok(), "Delete should succeed when cache is empty");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_get_model_path_trait() {
        let env_lock = acquire_env_mutex();
        let mock_server = MockHFServer::new(&env_lock).await;

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
    #[allow(clippy::await_holding_lock)]
    async fn test_download_ignore_weights() {
        let env_lock = acquire_env_mutex();
        let mock_server = MockHFServer::new(&env_lock).await;
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
    #[allow(clippy::await_holding_lock)]
    async fn test_download_ignores_subdirectories() {
        let env_lock = acquire_env_mutex();
        let mock_server = MockHFServer::new(&env_lock).await;
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

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_ignores_dotfiles() {
        let env_lock = acquire_env_mutex();
        // Create a mock server with dotfiles in siblings list but NO endpoint for them.
        // If the code tries to download a dotfile, it will fail since there's no mock.
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path_regex(r"^/api/models/test/model(?:/.*)?$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                 "id": "test/model",
                 "sha": "def5678",
                 "siblings": [
                     {"rfilename": "config.json"},
                     {"rfilename": ".gitkeep"},
                     {"rfilename": ".gitignore"},
                     {"rfilename": ".hidden"}
                 ]
            })))
            .mount(&server)
            .await;

        // Only mock config.json
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/test/model/resolve/(main|[^/]+)/config\.json$",
            ))
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

        let _hf_endpoint_guard = EnvVarGuard::set(&env_lock, "HF_ENDPOINT", &server.uri());

        let provider = HuggingFaceProvider;
        let result = provider
            .download_model("test/model", Some(temp_dir.path().to_path_buf()), false)
            .await;

        assert!(
            result.is_ok(),
            "Download should succeed with dotfiles ignored"
        );
    }

    #[test]
    fn test_is_offline_mode() {
        let env_lock = acquire_env_mutex();
        {
            let _offline_guard = EnvVarGuard::set(&env_lock, HF_HUB_OFFLINE_ENV_VAR, "1");
            assert!(is_offline_mode());
        }

        {
            let _offline_guard = EnvVarGuard::set(&env_lock, HF_HUB_OFFLINE_ENV_VAR, "0");
            assert!(!is_offline_mode());
        }

        {
            let _offline_guard = EnvVarGuard::remove(&env_lock, HF_HUB_OFFLINE_ENV_VAR);
            assert!(!is_offline_mode());
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_model_offline_mode_with_cache() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let snapshots_path = temp_dir
            .path()
            .join("models--test--model")
            .join("snapshots")
            .join("abc1234");
        std::fs::create_dir_all(&snapshots_path).expect("Failed to create directory");

        let _offline_guard = EnvVarGuard::set(&env_lock, HF_HUB_OFFLINE_ENV_VAR, "1");

        let result = HuggingFaceProvider
            .download_model("test/model", Some(temp_dir.path().into()), false)
            .await;

        assert!(result.is_ok());
        assert!(result.expect("Expected path").ends_with("abc1234"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_model_offline_mode_without_cache() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");

        let _offline_guard = EnvVarGuard::set(&env_lock, HF_HUB_OFFLINE_ENV_VAR, "1");

        let result = HuggingFaceProvider
            .download_model("nonexistent/model", Some(temp_dir.path().into()), false)
            .await;

        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected error")
                .to_string()
                .contains("not found in cache")
        );
    }
}
