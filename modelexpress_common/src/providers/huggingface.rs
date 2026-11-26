// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    cache::{ModelInfo, ProviderCache, directory_size},
    constants, envs,
    models::ModelProvider,
    providers::{ModelDownloadOutcome, ModelProviderTrait},
};
use anyhow::{Context, Result};
use futures::StreamExt;
use hf_hub::api::tokio::{ApiBuilder, ApiError};
use hf_hub::{Cache, Repo, RepoType};
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

const HF_FALLBACK_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HF_FALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Revision used when a caller does not pin one. Matches hf-hub's own default branch.
const DEFAULT_HF_REVISION: &str = "main";

/// Check if offline mode is enabled via `HF_HUB_OFFLINE`.
/// See [`crate::envs::hf_offline`] for the accepted truthy values.
fn is_offline_mode() -> bool {
    envs::hf_offline()
}

/// Hugging Face token, preferring `HF_TOKEN` and falling back to the token file that
/// `huggingface-cli login` writes under `HF_HOME`, unless `HF_HUB_DISABLE_IMPLICIT_TOKEN`
/// opts out of the cached one.
///
/// `ApiBuilder::from_env` reads that file itself, so handing it `envs::hf_token()` unchanged
/// discards the token whenever `HF_TOKEN` is unset. Resolving it here keeps the builder and the
/// full-body download fallback, which sets its own bearer header, on the same credential.
fn resolve_hf_token() -> Option<String> {
    if let Some(token) = envs::hf_token() {
        return Some(token);
    }
    if envs::hf_implicit_token_disabled() {
        return None;
    }
    Cache::from_env().token()
}

/// Get the cache directory for Hugging Face models
/// Priority order:
/// 1. Provided cache_dir parameter
/// 2. MODEL_EXPRESS_CACHE_DIRECTORY environment variable
/// 3. HF_HUB_CACHE environment variable
/// 4. Default location (~/.cache/huggingface/hub)
fn get_cache_dir(cache_dir: Option<PathBuf>) -> PathBuf {
    // Use provided cache directory if available
    if let Some(dir) = cache_dir {
        return dir;
    }

    // Try MODEL_EXPRESS_CACHE_DIRECTORY environment variable first
    if let Some(cache_path) = envs::cache_directory() {
        return cache_path;
    }

    // Try HF_HUB_CACHE environment variable
    if let Some(cache_path) = envs::hf_hub_cache() {
        return cache_path;
    }

    // Fall back to default location
    envs::home_dir_or_cwd().join(constants::DEFAULT_HF_CACHE_PATH)
}

/// Hugging Face model provider implementation
pub struct HuggingFaceProvider;

pub(crate) struct HuggingFaceProviderCache;

fn validate_single_segment(segment: &str, label: &str) -> Result<()> {
    if segment.is_empty() || segment.trim() != segment {
        anyhow::bail!("Invalid {label}: must be non-empty without surrounding whitespace");
    }
    if segment.contains('\0') || segment.contains('/') || segment == "." || segment == ".." {
        anyhow::bail!("Invalid {label} '{segment}': unsafe path segment");
    }
    let mut components = Path::new(segment).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => anyhow::bail!("Invalid {label} '{segment}': must be a single normal path segment"),
    }
}

fn validate_model_name(model_name: &str) -> Result<()> {
    if model_name.is_empty() || model_name.trim() != model_name || model_name.contains('\0') {
        anyhow::bail!("Invalid Hugging Face model name '{model_name}'");
    }
    let parts: Vec<&str> = model_name.split('/').collect();
    if parts.is_empty() || parts.len() > 2 {
        anyhow::bail!(
            "Invalid Hugging Face model name '{model_name}': expected '<model>' or '<org>/<model>'"
        );
    }
    for segment in &parts {
        validate_single_segment(segment, "Hugging Face model name segment")?;
        if segment.contains("--") || segment.starts_with('-') || segment.ends_with('-') {
            anyhow::bail!(
                "Invalid Hugging Face model name '{model_name}': segments must not contain '--' or start or end with '-'"
            );
        }
    }
    let folder = format!("models--{}", parts.join("--"));
    validate_single_segment(&folder, "Hugging Face cache directory name")
}

fn validate_snapshot_revision(revision: &str) -> Result<()> {
    validate_single_segment(revision, "Hugging Face snapshot revision")
}

fn validate_ref_name(revision: &str) -> Result<()> {
    if revision.is_empty() || revision.trim() != revision || revision.contains('\0') {
        anyhow::bail!("Invalid Hugging Face revision '{revision}'");
    }
    for segment in revision.split('/') {
        validate_single_segment(segment, "Hugging Face revision segment")?;
    }
    Ok(())
}

fn remove_dir_all_if_exists(cache_root: &Path, path: &Path) -> Result<bool> {
    if !crate::providers::ensure_delete_target_within_cache_root(cache_root, path)? {
        return Ok(false);
    }
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => {
            return Err(err).with_context(|| format!("Failed to inspect {}", path.display()));
        }
    };
    if meta.file_type().is_symlink() || !meta.file_type().is_dir() {
        match fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err).with_context(|| format!("Failed to remove {}", path.display())),
        }
    } else {
        match fs::remove_dir_all(path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err).with_context(|| format!("Failed to remove {}", path.display())),
        }
    }
}

fn snapshot_has_readable_file(snapshot_dir: &Path) -> bool {
    let mut stack = vec![snapshot_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_file() {
                return true;
            }
            if file_type.is_symlink() {
                if fs::metadata(&path).is_ok_and(|meta| meta.is_file()) {
                    return true;
                }
            } else if file_type.is_dir() {
                stack.push(path);
            }
        }
    }
    false
}

fn remove_refs_for_commit(cache_root: &Path, refs_dir: &Path, commit: &str) {
    if !crate::providers::ensure_delete_target_within_cache_root(cache_root, refs_dir)
        .unwrap_or(false)
    {
        return;
    }
    if !fs::symlink_metadata(refs_dir).is_ok_and(|meta| meta.file_type().is_dir()) {
        return;
    }
    let mut stack = vec![refs_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file()
                && fs::read_to_string(&path).is_ok_and(|target| target.trim() == commit)
            {
                match fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => warn!("Failed to remove stale ref {}: {err}", path.display()),
                }
            }
        }
    }
}

impl HuggingFaceProviderCache {
    fn repo_root(cache_root: &Path, model_name: &str) -> Result<PathBuf> {
        validate_model_name(model_name)?;
        Ok(cache_root.join(format!("models--{}", model_name.replace('/', "--"))))
    }

    fn snapshots_dir(cache_root: &Path, model_name: &str) -> Result<PathBuf> {
        Ok(Self::repo_root(cache_root, model_name)?.join("snapshots"))
    }

    fn refs_dir(cache_root: &Path, model_name: &str) -> Result<PathBuf> {
        Ok(Self::repo_root(cache_root, model_name)?.join("refs"))
    }

    /// Read the commit a cached `refs/<revision>` entry points at.
    ///
    /// Returns `None` when the ref does not exist or its target is not a safe
    /// snapshot segment, so callers can fall back or reject safely.
    fn read_ref(cache_root: &Path, model_name: &str, revision: &str) -> Option<String> {
        validate_ref_name(revision).ok()?;
        let ref_path = Self::refs_dir(cache_root, model_name).ok()?.join(revision);
        let commit = fs::read_to_string(ref_path).ok()?;
        let commit = commit.trim();
        validate_snapshot_revision(commit).ok()?;
        Some(commit.to_string())
    }

    fn folder_name_to_model_id(folder_name: &str) -> String {
        if let Some(stripped) = folder_name.strip_prefix("models--") {
            stripped.replace("--", "/")
        } else {
            folder_name.to_string()
        }
    }

    fn latest_local_snapshot_path(cache_root: &Path, model_name: &str) -> Result<PathBuf> {
        let path = Self::snapshots_dir(cache_root, model_name)?;

        if !path.is_dir() {
            anyhow::bail!("Model snapshots for '{model_name}' not found in cache");
        }

        let mut files: Vec<fs::DirEntry> = fs::read_dir(path)?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|ft| ft.is_dir()))
            .collect();
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
        let model_path = Self::repo_root(cache_root, model_name)?;

        if remove_dir_all_if_exists(cache_root, &model_path)? {
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
            Some(revision) => {
                validate_snapshot_revision(revision)?;
                Ok(Self::snapshots_dir(cache_root, model_name)?.join(revision))
            }
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

            let size = directory_size(&path);
            models.push(ModelInfo::measured(
                ModelProvider::HuggingFace,
                Self::folder_name_to_model_id(folder_name),
                path,
                size,
            ));
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

        let repo_root = HuggingFaceProviderCache::repo_root(cache_dir, model_name)?;
        validate_snapshot_revision(commit_hash)?;
        validate_single_segment(filename, "Hugging Face filename")?;
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
        Ok(pointer_path)
    }

    /// Point `refs/<revision>` at `commit_hash` so the snapshot stays resolvable by the
    /// name the caller asked for (a branch, a tag, or the default revision).
    fn create_revision_ref(
        cache_dir: &Path,
        model_name: &str,
        revision: &str,
        commit_hash: &str,
    ) -> Result<()> {
        validate_model_name(model_name)?;
        validate_ref_name(revision)?;
        validate_snapshot_revision(commit_hash)?;
        Cache::new(cache_dir.to_path_buf())
            .repo(Repo::with_revision(
                model_name.to_string(),
                RepoType::Model,
                revision.to_string(),
            ))
            .create_ref(commit_hash)
            .with_context(|| {
                format!("Failed to create Hugging Face cache ref for '{model_name}@{revision}'")
            })
    }

    /// Delete blobs that no surviving snapshot references.
    ///
    /// Snapshot entries are symlinks into the repository's `blobs/` directory, so
    /// deleting a snapshot removes the links but leaves the bytes behind. Without this,
    /// evicting one revision of a multi-revision model reclaims essentially no disk.
    ///
    /// Best effort: a blob we cannot classify is kept. Leaking disk is recoverable,
    /// deleting a blob another snapshot still points at is not.
    fn reclaim_unreferenced_blobs(cache_dir: &Path, model_name: &str) {
        let Ok(repo_root) = HuggingFaceProviderCache::repo_root(cache_dir, model_name) else {
            return;
        };
        let blobs_dir = repo_root.join("blobs");
        if !crate::providers::ensure_delete_target_within_cache_root(cache_dir, &blobs_dir)
            .unwrap_or(false)
            || !fs::symlink_metadata(&blobs_dir).is_ok_and(|meta| meta.file_type().is_dir())
        {
            return;
        }
        let Ok(blobs) = fs::read_dir(&blobs_dir) else {
            return;
        };

        let mut referenced = HashSet::new();
        let snapshots_dir = repo_root.join("snapshots");
        if !fs::symlink_metadata(&snapshots_dir).is_ok_and(|meta| meta.file_type().is_dir()) {
            return;
        }
        let Ok(snapshots) = fs::read_dir(&snapshots_dir) else {
            // No snapshot directory to walk means we cannot prove a blob is unused.
            return;
        };
        let mut dirs_to_scan = Vec::new();
        for snapshot in snapshots {
            let Ok(snapshot) = snapshot else {
                return;
            };
            let Ok(file_type) = snapshot.file_type() else {
                return;
            };
            if file_type.is_dir() {
                dirs_to_scan.push(snapshot.path());
            }
        }
        while let Some(dir) = dirs_to_scan.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                // An unreadable snapshot may reference anything, so keep every blob.
                return;
            };
            for entry in entries {
                let Ok(entry) = entry else {
                    return;
                };
                let Ok(file_type) = entry.file_type() else {
                    return;
                };
                let path = entry.path();
                if file_type.is_dir() {
                    dirs_to_scan.push(path);
                } else if let Ok(target) = fs::canonicalize(&path) {
                    referenced.insert(target);
                }
            }
        }

        for blob in blobs.filter_map(Result::ok) {
            if !blob.file_type().is_ok_and(|ft| ft.is_file()) {
                continue;
            }
            let path = blob.path();
            let is_unreferenced = fs::canonicalize(&path)
                .map(|canonical| !referenced.contains(&canonical))
                .unwrap_or(false);
            if is_unreferenced {
                match fs::remove_file(&path) {
                    Ok(()) => debug!("Reclaimed unreferenced blob {}", path.display()),
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => warn!("Failed to reclaim blob {}: {e}", path.display()),
                }
            }
        }
    }

    /// Fetch repository metadata for a revision.
    ///
    /// A missing branch, tag, or commit surfaces as an error here; we never retry
    /// against the default revision, so a typo can't silently download the wrong model.
    async fn repo_info_at_revision(
        model_name: &str,
        revision: &str,
        cache_dir: Option<&Path>,
    ) -> Result<hf_hub::api::RepoInfo> {
        validate_model_name(model_name)?;
        validate_ref_name(revision)?;
        let mut builder = ApiBuilder::from_env().with_token(resolve_hf_token());
        if let Some(cache_dir) = cache_dir {
            builder = builder.with_cache_dir(cache_dir.to_path_buf());
        }
        let api = builder.build()?;
        api.repo(Repo::with_revision(
            model_name.to_string(),
            RepoType::Model,
            revision.to_string(),
        ))
        .info()
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to fetch model '{model_name}' at revision '{revision}' from HuggingFace. \
                 Is this a valid HuggingFace ID and revision? Error: {e}"
            )
        })
    }

    /// Resolve a requested revision against the local cache only.
    ///
    /// Used in offline mode and by the file-serving paths, which must not reach out to
    /// the Hub for a model that is already downloaded.
    fn resolve_revision_offline(
        cache_dir: &Path,
        model_name: &str,
        revision: Option<&str>,
    ) -> Result<String> {
        let snapshots_dir = HuggingFaceProviderCache::snapshots_dir(cache_dir, model_name)?;
        match revision {
            Some(revision) => {
                validate_ref_name(revision)?;
                if validate_snapshot_revision(revision).is_ok()
                    && snapshots_dir.join(revision).is_dir()
                {
                    return Ok(revision.to_string());
                }
                if let Some(commit) =
                    HuggingFaceProviderCache::read_ref(cache_dir, model_name, revision)
                    && snapshots_dir.join(&commit).is_dir()
                {
                    return Ok(commit);
                }
                anyhow::bail!(
                    "Revision '{revision}' of model '{model_name}' is not present in the cache"
                )
            }
            None => {
                let latest =
                    HuggingFaceProviderCache::latest_local_snapshot_path(cache_dir, model_name)?;
                latest
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(String::from)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Cached snapshot path for '{model_name}' has no revision component"
                        )
                    })
            }
        }
    }

    /// Download every eligible file of `model_name` at `revision`, resolving the request
    /// to an immutable commit first and fetching every file from that commit.
    async fn download_at_revision(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
        revision: Option<&str>,
    ) -> Result<ModelDownloadOutcome> {
        validate_model_name(model_name)?;
        if let Some(revision) = revision {
            validate_ref_name(revision)?;
        }
        let cache_dir = get_cache_dir(cache_dir);
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            anyhow::anyhow!("Failed to create cache directory {:?}: {}", cache_dir, e)
        })?;

        if is_offline_mode() {
            info!("HF_HUB_OFFLINE is set, using cached model for '{model_name}'");
            let resolved = Self::resolve_revision_offline(&cache_dir, model_name, revision)?;
            let path =
                HuggingFaceProviderCache::snapshots_dir(&cache_dir, model_name)?.join(&resolved);
            return Ok(ModelDownloadOutcome {
                path,
                resolved_revision: Some(resolved),
            });
        }

        let token = resolve_hf_token();

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
        let requested_revision = revision.unwrap_or(DEFAULT_HF_REVISION).to_string();

        let info = api
            .repo(Repo::with_revision(
                model_name.clone(),
                RepoType::Model,
                requested_revision.clone(),
            ))
            .info()
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to fetch model '{model_name}' at revision '{requested_revision}' from \
                     HuggingFace. Is this a valid HuggingFace ID and revision? Error: {e}"
                )
            })?;
        debug!("Got model info: {info:?}");
        validate_snapshot_revision(&info.sha)?;

        if info.siblings.is_empty() {
            anyhow::bail!("Model '{model_name}' exists but contains no downloadable files.");
        }

        // Every file is fetched from the resolved commit rather than from the requested
        // branch or tag, so a revision that moves mid-download cannot mix commits into
        // one snapshot.
        let pinned_repo = api.repo(Repo::with_revision(
            model_name.clone(),
            RepoType::Model,
            info.sha.clone(),
        ));
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

            match pinned_repo.get(&sib.rfilename).await {
                Ok(_) => {}
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
                    })?;
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
            }

            files_downloaded = true;
        }

        if !files_downloaded {
            return Err(anyhow::anyhow!(
                "No valid files found for model '{}'.",
                model_name
            ));
        }

        HuggingFaceProvider::create_revision_ref(
            &cache_dir,
            &model_name,
            &requested_revision,
            &info.sha,
        )?;

        info!(
            "Downloaded model files for {model_name} at revision {}",
            info.sha
        );

        Ok(ModelDownloadOutcome {
            path: HuggingFaceProviderCache::snapshots_dir(&cache_dir, &model_name)?.join(&info.sha),
            resolved_revision: Some(info.sha),
        })
    }
}

impl HuggingFaceProvider {
    /// Validate that `revision` is a plain single-component snapshot directory name.
    pub fn validate_snapshot_revision(revision: &str) -> Result<()> {
        validate_snapshot_revision(revision)
    }

    /// Delete a single cached Hugging Face snapshot and stale refs pointing to it,
    /// leaving `blobs/` and the repository root untouched.
    ///
    /// Used when deleting a revision while a sibling revision of the same model is
    /// actively `DOWNLOADING`, so in-flight `.sync.part`, `.lock`, or newly renamed
    /// blobs belonging to the sibling are never disturbed.
    pub fn delete_snapshot_keeping_blobs(
        model_name: &str,
        cache_dir: &Path,
        revision: &str,
    ) -> Result<()> {
        Self::delete_snapshot_internal(model_name, cache_dir, revision, false)
    }

    fn delete_snapshot_internal(
        model_name: &str,
        cache_dir: &Path,
        revision: &str,
        reclaim_blobs: bool,
    ) -> Result<()> {
        validate_snapshot_revision(revision)?;
        let snapshots_dir = HuggingFaceProviderCache::snapshots_dir(cache_dir, model_name)?;
        let snapshot = snapshots_dir.join(revision);

        if remove_dir_all_if_exists(cache_dir, &snapshot)? {
            info!("Deleted snapshot {revision} for model '{model_name}'");
        } else {
            info!("Snapshot {revision} for model '{model_name}' is not cached");
        }

        let refs_dir = HuggingFaceProviderCache::refs_dir(cache_dir, model_name)?;
        remove_refs_for_commit(cache_dir, &refs_dir, revision);

        if !reclaim_blobs {
            return Ok(());
        }

        let no_snapshots_left = fs::read_dir(&snapshots_dir)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false);
        if no_snapshots_left {
            let repo_root = HuggingFaceProviderCache::repo_root(cache_dir, model_name)?;
            if remove_dir_all_if_exists(cache_dir, &repo_root)? {
                info!("Removed empty repository directory for model '{model_name}'");
            }
        } else {
            HuggingFaceProvider::reclaim_unreferenced_blobs(cache_dir, model_name);
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for HuggingFaceProvider {
    /// Attempt to download a model from Hugging Face at its default revision.
    /// Returns the directory it is in.
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        self.download_at_revision(model_name, cache_dir, ignore_weights, None)
            .await
            .map(|outcome| outcome.path)
    }

    async fn download_model_revision(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
        revision: Option<&str>,
    ) -> Result<ModelDownloadOutcome> {
        self.download_at_revision(model_name, cache_dir, ignore_weights, revision)
            .await
    }

    /// Resolve a branch, tag, or commit SHA to the immutable commit SHA it names.
    ///
    /// Offline, this only consults the local cache; a revision that is not cached is an
    /// error rather than a fall back to whatever snapshot happens to be newest.
    async fn resolve_revision(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        revision: Option<&str>,
    ) -> Result<Option<String>> {
        validate_model_name(model_name)?;
        if let Some(revision) = revision {
            validate_ref_name(revision)?;
        }
        let cache_dir = get_cache_dir(cache_dir);

        if is_offline_mode() {
            return Self::resolve_revision_offline(&cache_dir, model_name, revision).map(Some);
        }

        let info = Self::repo_info_at_revision(
            model_name,
            revision.unwrap_or(DEFAULT_HF_REVISION),
            Some(&cache_dir),
        )
        .await?;

        Ok(Some(info.sha))
    }

    /// Delete a model's repository directory from the local Hugging Face cache.
    ///
    /// Never contacts the Hub; returns `Ok(())` when the model is already absent.
    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        info!("Deleting model from Hugging Face cache: {model_name}");
        HuggingFaceProviderCache.clear_model(&cache_dir, model_name)
    }

    /// Delete a single cached revision, leaving the model's other revisions intact.
    ///
    /// Treats `revision` as the literal snapshot directory name without resolving
    /// through `refs/`, so a branch name cannot redirect deletion onto another entry's
    /// snapshot. Still removes any `refs/` entry that pointed to the deleted commit.
    /// Once the last snapshot is gone the whole repository directory is removed.
    async fn delete_model_revision(
        &self,
        model_name: &str,
        cache_dir: PathBuf,
        revision: Option<&str>,
    ) -> Result<()> {
        let Some(revision) = revision else {
            return HuggingFaceProviderCache.clear_model(&cache_dir, model_name);
        };

        Self::delete_snapshot_internal(model_name, &cache_dir, revision, true)
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
        let token = resolve_hf_token();
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

    /// Resolve a cached snapshot directory without contacting the Hub.
    ///
    /// The file-serving paths use this: the model is already downloaded, so a requested
    /// revision either has a snapshot on disk or the request is not serviceable.
    async fn get_model_path_revision(
        &self,
        model_name: &str,
        cache_dir: PathBuf,
        revision: Option<&str>,
    ) -> Result<PathBuf> {
        let Some(revision) = revision else {
            return self.get_model_path(model_name, cache_dir).await;
        };

        let commit = Self::resolve_revision_offline(&cache_dir, model_name, Some(revision))?;
        Ok(HuggingFaceProviderCache::snapshots_dir(&cache_dir, model_name)?.join(commit))
    }

    async fn has_cached_files(
        &self,
        model_name: &str,
        cache_dir: PathBuf,
        revision: Option<&str>,
        _ignore_weights: bool,
    ) -> Result<bool> {
        validate_model_name(model_name)?;
        if let Some(revision) = revision {
            validate_ref_name(revision)?;
        }
        let Ok(commit) = Self::resolve_revision_offline(&cache_dir, model_name, revision) else {
            return Ok(false);
        };
        let snapshot_dir =
            HuggingFaceProviderCache::snapshots_dir(&cache_dir, model_name)?.join(commit);
        Ok(snapshot_has_readable_file(&snapshot_dir))
    }

    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        validate_model_name(model_name)?;
        Ok(model_name.to_string())
    }

    fn supports_revisions(&self) -> bool {
        true
    }

    /// Write `refs/<revision>` for a snapshot that was installed by streaming rather
    /// than downloaded through hf-hub, so `huggingface_hub` can resolve the snapshot by
    /// branch or tag name — including with `HF_HUB_OFFLINE=1`.
    async fn record_local_revision(
        &self,
        model_name: &str,
        cache_dir: &Path,
        requested_revision: Option<&str>,
        commit: &str,
    ) -> Result<()> {
        Self::create_revision_ref(
            cache_dir,
            model_name,
            requested_revision.unwrap_or(DEFAULT_HF_REVISION),
            commit,
        )
    }

    fn provider_name(&self) -> &'static str {
        "Hugging Face"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, acquire_env_mutex};
    use serde_json::json;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::sync::MutexGuard;
    use tempfile::TempDir;
    use tokio::time::Duration;
    use wiremock::matchers::{header, method, path_regex};
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

            let hf_endpoint_guard =
                EnvVarGuard::set(env_lock, crate::envs::HF_ENDPOINT, &server.uri());

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

        // Downloads are pinned to the resolved commit, so both hf-hub's range probe and
        // the full-body fallback hit the commit-pinned URL. A request against the branch
        // would mean a file could come from a commit other than the one we report.
        Mock::given(method("GET"))
            .and(path_regex(r"^/test/model/resolve/main/LICENSE$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .named("branch-resolved download must not happen")
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
            .expect(2)
            .named("commit-pinned range probe, then full-body fallback")
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
        let _model_express_cache_guard = EnvVarGuard::set(
            &env_lock,
            crate::envs::MODEL_EXPRESS_CACHE_DIRECTORY,
            env_cache_path,
        );
        let _hf_hub_cache_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_CACHE, env_cache_path);

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
            .and(path_regex(r"^/.*$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .named("delete_model must be local-only and never contact Hugging Face")
            .mount(&server)
            .await;

        let temp_cache_path = temp_cache.path().to_str().expect("Expected cache path");
        let _model_express_cache_guard = EnvVarGuard::set(
            &env_lock,
            crate::envs::MODEL_EXPRESS_CACHE_DIRECTORY,
            temp_cache_path,
        );
        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let result = provider
            .delete_model(model_name, temp_cache.path().to_path_buf())
            .await;

        assert!(result.is_ok(), "Delete should succeed when cache is empty");

        let repo_root = temp_cache
            .path()
            .join("models--modelexpress-tests--delete-no-download");
        std::fs::create_dir_all(repo_root.join("snapshots/abc1234"))
            .expect("Failed to create snapshot");
        std::fs::create_dir_all(repo_root.join("blobs")).expect("Failed to create blobs");
        std::fs::write(repo_root.join("snapshots/abc1234/config.json"), b"{}")
            .expect("Failed to write config");
        std::fs::write(repo_root.join("blobs/blob1"), b"data").expect("Failed to write blob");

        provider
            .delete_model(model_name, temp_cache.path().to_path_buf())
            .await
            .expect("Local-only delete should remove cached repository");
        assert!(!repo_root.exists());
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

        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

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
            let _offline_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_OFFLINE, "1");
            assert!(is_offline_mode());
        }

        {
            let _offline_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_OFFLINE, "0");
            assert!(!is_offline_mode());
        }

        {
            let _offline_guard = EnvVarGuard::remove(&env_lock, crate::envs::HF_HUB_OFFLINE);
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

        let _offline_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_OFFLINE, "1");

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

        let _offline_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_OFFLINE, "1");

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

    /// Stub the revision-metadata endpoint hf-hub queries for `revision`, reporting
    /// `sha` as the commit that revision currently points at.
    async fn mock_revision_info(server: &MockServer, revision: &str, sha: &str) {
        Mock::given(method("GET"))
            .and(path_regex(format!(
                r"^/api/models/test/model/revision/{revision}$"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test/model",
                "sha": sha,
                "siblings": [{"rfilename": "config.json"}]
            })))
            .mount(server)
            .await;
    }

    /// Stub `config.json` under one specific revision path.
    async fn mock_revision_file(server: &MockServer, revision: &str, sha: &str) {
        Mock::given(method("GET"))
            .and(path_regex(format!(
                r"^/test/model/resolve/{revision}/config\.json$"
            )))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("etag", format!("\"{sha}\""))
                    .insert_header("x-repo-commit", sha)
                    .insert_header("accept-ranges", "bytes")
                    .insert_header("content-length", "64")
                    .insert_header("content-range", "bytes 0-63/64")
                    .set_body_bytes(vec![0u8; 64]),
            )
            .mount(server)
            .await;
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_at_explicit_commit_produces_that_snapshot() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let server = MockServer::start().await;

        mock_revision_info(&server, "abc9999", "abc9999").await;
        mock_revision_file(&server, "abc9999", "abc9999").await;

        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let outcome = HuggingFaceProvider
            .download_model_revision(
                "test/model",
                Some(temp_dir.path().to_path_buf()),
                false,
                Some("abc9999"),
            )
            .await
            .expect("Pinned download should succeed");

        assert_eq!(outcome.resolved_revision.as_deref(), Some("abc9999"));
        assert_eq!(
            outcome.path,
            temp_dir
                .path()
                .join("models--test--model/snapshots/abc9999")
        );
        assert!(outcome.path.join("config.json").exists());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_at_tag_fetches_every_file_from_the_resolved_commit() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let server = MockServer::start().await;

        mock_revision_info(&server, "v1.0", "def5678").await;
        mock_revision_file(&server, "def5678", "def5678").await;

        // A tag can move between the metadata lookup and the file fetch, so downloading
        // through the tag would risk mixing commits into one snapshot.
        Mock::given(method("GET"))
            .and(path_regex(r"^/test/model/resolve/v1\.0/config\.json$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .named("tag-resolved download must not happen")
            .mount(&server)
            .await;

        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let outcome = HuggingFaceProvider
            .download_model_revision(
                "test/model",
                Some(temp_dir.path().to_path_buf()),
                false,
                Some("v1.0"),
            )
            .await
            .expect("Tag download should succeed");

        assert_eq!(outcome.resolved_revision.as_deref(), Some("def5678"));
        assert!(outcome.path.ends_with("snapshots/def5678"));
        assert_eq!(
            fs::read_to_string(temp_dir.path().join("models--test--model/refs/v1.0"))
                .expect("Expected the tag ref to point at the resolved commit"),
            "def5678"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_unpinned_download_reports_the_default_revision_commit() {
        let env_lock = acquire_env_mutex();
        let mock_server = MockHFServer::new(&env_lock).await;

        let outcome = HuggingFaceProvider
            .download_model_revision(
                "test/model",
                Some(mock_server.cache_path.clone()),
                false,
                None,
            )
            .await
            .expect("Unpinned download should succeed");

        assert_eq!(outcome.resolved_revision.as_deref(), Some("def5678"));
        assert!(outcome.path.ends_with("snapshots/def5678"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_missing_revision_fails_without_falling_back_to_the_default() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path_regex(r"^/api/models/test/model/revision/no-such-tag$"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        // The default revision resolves fine. Falling back to it would quietly hand the
        // caller a different model than the one they pinned.
        mock_revision_info(&server, "main", "def5678").await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/test/model/resolve/[^/]+/config\.json$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .named("no file may be downloaded for a missing revision")
            .mount(&server)
            .await;

        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let error = HuggingFaceProvider
            .download_model_revision(
                "test/model",
                Some(temp_dir.path().to_path_buf()),
                false,
                Some("no-such-tag"),
            )
            .await
            .expect_err("A missing revision must fail");

        assert!(
            error.to_string().contains("no-such-tag"),
            "Error should name the requested revision, got: {error}"
        );
        assert!(
            !temp_dir
                .path()
                .join("models--test--model/snapshots/def5678")
                .exists(),
            "The default revision must not have been downloaded"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_resolve_revision_offline_uses_refs_then_snapshot_directories() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let repo_root = temp_dir.path().join("models--test--model");
        std::fs::create_dir_all(repo_root.join("snapshots/abc1234"))
            .expect("Failed to create snapshot");
        std::fs::create_dir_all(repo_root.join("refs")).expect("Failed to create refs");
        std::fs::write(repo_root.join("refs/main"), "abc1234").expect("Failed to write ref");

        let _offline_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_OFFLINE, "1");
        let cache_dir = Some(temp_dir.path().to_path_buf());
        let provider = HuggingFaceProvider;

        for revision in [None, Some("main"), Some("abc1234")] {
            let resolved = provider
                .resolve_revision("test/model", cache_dir.clone(), revision)
                .await
                .unwrap_or_else(|e| panic!("Expected {revision:?} to resolve offline: {e}"));
            assert_eq!(resolved.as_deref(), Some("abc1234"));
        }

        assert!(
            provider
                .resolve_revision("test/model", cache_dir, Some("not-cached"))
                .await
                .is_err(),
            "An uncached revision must not resolve to the newest local snapshot"
        );
    }

    #[tokio::test]
    async fn test_delete_model_revision_keeps_the_other_snapshots() {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let repo_root = temp_dir.path().join("models--test--model");
        for commit in ["abc1234", "def5678"] {
            std::fs::create_dir_all(repo_root.join("snapshots").join(commit))
                .expect("Failed to create snapshot");
            std::fs::write(
                repo_root.join("snapshots").join(commit).join("config.json"),
                b"{}",
            )
            .expect("Failed to write config");
        }
        std::fs::create_dir_all(repo_root.join("refs/release")).expect("Failed to create refs");
        std::fs::write(repo_root.join("refs/main"), "abc1234").expect("Failed to write ref");
        std::fs::write(repo_root.join("refs/release/v1"), "abc1234").expect("Failed to write ref");
        std::fs::write(repo_root.join("refs/stable"), "def5678").expect("Failed to write ref");

        let provider = HuggingFaceProvider;
        provider
            .delete_model_revision("test/model", temp_dir.path().to_path_buf(), Some("abc1234"))
            .await
            .expect("Deleting one revision should succeed");

        assert!(!repo_root.join("snapshots/abc1234").exists());
        assert!(!repo_root.join("refs/main").exists());
        assert!(!repo_root.join("refs/release/v1").exists());
        assert!(repo_root.join("refs/stable").exists());
        assert!(repo_root.join("snapshots/def5678/config.json").exists());

        provider
            .delete_model_revision("test/model", temp_dir.path().to_path_buf(), Some("def5678"))
            .await
            .expect("Deleting the last revision should succeed");

        assert!(
            !repo_root.exists(),
            "The repository directory should be removed once its last snapshot is gone"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_delete_model_revision_reclaims_only_unreferenced_blobs() {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let repo_root = temp_dir.path().join("models--test--model");
        let blobs = repo_root.join("blobs");
        std::fs::create_dir_all(&blobs).expect("Failed to create blobs");

        // `shared` is linked from both snapshots, `only-old` from just the one we delete.
        for blob in ["shared", "only-old"] {
            std::fs::write(blobs.join(blob), vec![0u8; 32]).expect("Failed to write blob");
        }
        for (commit, blob_names) in [
            ("abc1234", vec!["shared", "only-old"]),
            ("def5678", vec!["shared"]),
        ] {
            let snapshot = repo_root.join("snapshots").join(commit);
            std::fs::create_dir_all(&snapshot).expect("Failed to create snapshot");
            for blob in blob_names {
                symlink(blobs.join(blob), snapshot.join(blob)).expect("Failed to link blob");
            }
        }

        HuggingFaceProvider
            .delete_model_revision("test/model", temp_dir.path().to_path_buf(), Some("abc1234"))
            .await
            .expect("Deleting one revision should succeed");

        assert!(
            blobs.join("shared").exists(),
            "A blob the surviving snapshot links to must be kept"
        );
        assert!(
            !blobs.join("only-old").exists(),
            "A blob no snapshot links to any more must be reclaimed"
        );
    }

    #[tokio::test]
    async fn test_delete_model_revision_without_a_revision_is_local_only() {
        // Eviction has to reclaim disk without the Hub being reachable, so the
        // delete-everything path must not enumerate files from the API.
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let repo_root = temp_dir.path().join("models--test--model");
        let snapshot = repo_root.join("snapshots/abc1234");
        std::fs::create_dir_all(&snapshot).expect("Failed to create snapshot");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("Failed to write config");
        std::fs::create_dir_all(repo_root.join("blobs")).expect("Failed to create blobs");
        std::fs::write(repo_root.join("blobs/blob1"), vec![0u8; 32]).expect("Failed to write blob");

        HuggingFaceProvider
            .delete_model_revision("test/model", temp_dir.path().to_path_buf(), None)
            .await
            .expect("Deleting the whole model should succeed");

        assert!(
            !repo_root.exists(),
            "The repository directory should be gone"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_get_model_path_revision_never_reaches_the_hub() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let server = MockServer::start().await;

        // Any request at all fails the test: serving an already-downloaded snapshot must
        // not depend on the Hub being reachable.
        Mock::given(method("GET"))
            .and(path_regex(r"^/.*$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .named("resolving a cached snapshot must not call the Hub")
            .mount(&server)
            .await;

        let repo_root = temp_dir.path().join("models--test--model");
        std::fs::create_dir_all(repo_root.join("snapshots/abc1234"))
            .expect("Failed to create snapshot");

        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let provider = HuggingFaceProvider;
        let path = provider
            .get_model_path_revision("test/model", temp_dir.path().to_path_buf(), Some("abc1234"))
            .await
            .expect("A cached revision should resolve");
        assert_eq!(path, repo_root.join("snapshots/abc1234"));

        assert!(
            provider
                .get_model_path_revision(
                    "test/model",
                    temp_dir.path().to_path_buf(),
                    Some("missing")
                )
                .await
                .is_err(),
            "An uncached revision must not resolve"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_resolve_revision_sends_the_cached_cli_token_when_hf_token_is_unset() {
        let env_lock = acquire_env_mutex();
        let hf_home = TempDir::new().expect("Failed to create temporary directory");
        std::fs::write(hf_home.path().join("token"), "hf_cached_cli_token\n")
            .expect("Failed to write token file");
        let _hf_home_guard =
            EnvVarGuard::set(&env_lock, "HF_HOME", &hf_home.path().display().to_string());
        let _hf_token_guard = EnvVarGuard::remove(&env_lock, crate::envs::HF_TOKEN);
        let _offline_guard = EnvVarGuard::remove(&env_lock, crate::envs::HF_HUB_OFFLINE);
        let _implicit_guard =
            EnvVarGuard::remove(&env_lock, crate::envs::HF_HUB_DISABLE_IMPLICIT_TOKEN);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/api/models/test/model(?:/.*)?$"))
            .and(header("authorization", "Bearer hf_cached_cli_token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test/model",
                "sha": "def5678",
                "siblings": []
            })))
            .mount(&server)
            .await;
        let _endpoint_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let cache_dir = TempDir::new().expect("Failed to create temporary cache directory");
        let resolved = HuggingFaceProvider
            .resolve_revision("test/model", Some(cache_dir.path().to_path_buf()), None)
            .await
            .expect("The token written by `huggingface-cli login` should authenticate the request");

        assert_eq!(resolved.as_deref(), Some("def5678"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_resolve_revision_omits_the_cached_cli_token_when_implicit_tokens_are_disabled() {
        let env_lock = acquire_env_mutex();
        let hf_home = TempDir::new().expect("Failed to create temporary directory");
        std::fs::write(hf_home.path().join("token"), "hf_cached_cli_token")
            .expect("Failed to write token file");
        let _hf_home_guard =
            EnvVarGuard::set(&env_lock, "HF_HOME", &hf_home.path().display().to_string());
        let _hf_token_guard = EnvVarGuard::remove(&env_lock, crate::envs::HF_TOKEN);
        let _offline_guard = EnvVarGuard::remove(&env_lock, crate::envs::HF_HUB_OFFLINE);
        let _implicit_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_HUB_DISABLE_IMPLICIT_TOKEN, "1");

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/api/models/test/model(?:/.*)?$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "test/model",
                "sha": "def5678",
                "siblings": []
            })))
            .mount(&server)
            .await;
        let _endpoint_guard = EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &server.uri());

        let cache_dir = TempDir::new().expect("Failed to create temporary cache directory");
        HuggingFaceProvider
            .resolve_revision("test/model", Some(cache_dir.path().to_path_buf()), None)
            .await
            .expect("The request should still resolve anonymously");

        let requests = server
            .received_requests()
            .await
            .expect("Request recording is enabled");
        assert!(
            requests
                .iter()
                .all(|request| !request.headers.contains_key("authorization")),
            "A cached token must not be sent once HF_HUB_DISABLE_IMPLICIT_TOKEN is set"
        );
    }

    #[tokio::test]
    async fn test_validation_rejects_ambiguous_and_traversal_names_and_revisions() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let victim = temp_dir.path().join("models--org--model/snapshots/abc1234");
        fs::create_dir_all(&victim).expect("Failed to create victim snapshot");
        fs::write(victim.join("config.json"), b"{}").expect("Failed to write victim file");

        let provider = HuggingFaceProvider;
        let bad_names = [
            "",
            " ",
            " org/model",
            "org/model ",
            ".",
            "..",
            "../other",
            "org/..",
            "org/.",
            "/org/model",
            "org/model/",
            "org//model",
            "a/b/c",
            "org--model",
            "org/a--b",
            "org/model\0",
        ];

        for bad in bad_names {
            assert!(
                provider.canonical_model_name(bad).is_err(),
                "canonical_model_name should reject '{bad}'"
            );
            assert!(
                provider
                    .delete_model(bad, temp_dir.path().to_path_buf())
                    .await
                    .is_err(),
                "delete_model should reject '{bad}'"
            );
            assert!(
                HuggingFaceProviderCache
                    .clear_model(temp_dir.path(), bad)
                    .is_err(),
                "clear_model should reject '{bad}'"
            );
            assert!(
                provider
                    .has_cached_files(bad, temp_dir.path().to_path_buf(), None, false)
                    .await
                    .is_err(),
                "has_cached_files should reject '{bad}'"
            );
        }

        let bad_revisions = [
            "",
            " ",
            " abc",
            "abc ",
            ".",
            "..",
            "../abc1234",
            "../../models--org--model",
            "a/b",
            "abc\0",
        ];
        for bad_rev in bad_revisions {
            assert!(
                provider
                    .delete_model_revision(
                        "org/model",
                        temp_dir.path().to_path_buf(),
                        Some(bad_rev)
                    )
                    .await
                    .is_err(),
                "delete_model_revision should reject revision '{bad_rev}'"
            );
        }

        assert!(
            victim.join("config.json").exists(),
            "Invalid delete inputs must leave valid cached models untouched"
        );
    }

    #[tokio::test]
    async fn test_delete_model_revision_does_not_resolve_through_refs() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let repo_root = temp_dir.path().join("models--org--model");
        let snapshot = repo_root.join("snapshots/abc1234");
        fs::create_dir_all(&snapshot).expect("Failed to create snapshot");
        fs::write(snapshot.join("config.json"), b"{}").expect("Failed to write config");
        fs::create_dir_all(repo_root.join("refs")).expect("Failed to create refs");
        fs::write(repo_root.join("refs/main"), "abc1234").expect("Failed to write ref");

        HuggingFaceProvider
            .delete_model_revision("org/model", temp_dir.path().to_path_buf(), Some("main"))
            .await
            .expect("Deleting an uncached literal snapshot name should be idempotent");

        assert!(
            snapshot.join("config.json").exists(),
            "Deleting branch name 'main' must not dereference refs/main and delete snapshots/abc1234"
        );
        assert!(repo_root.join("refs/main").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_delete_does_not_follow_symlinks_outside_cache() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let outside_dir = TempDir::new().expect("Failed to create outside dir");
        let outside_file = outside_dir.path().join("keep.txt");
        fs::write(&outside_file, b"important").expect("Failed to write outside file");

        let linked_repo = cache_dir.path().join("models--org--linked");
        symlink(outside_dir.path(), &linked_repo).expect("Failed to create repo symlink");

        HuggingFaceProvider
            .delete_model("org/linked", cache_dir.path().to_path_buf())
            .await
            .expect("Deleting a symlinked repo root should succeed");

        assert!(linked_repo.symlink_metadata().is_err());
        assert!(
            outside_file.exists(),
            "External target of a symlinked repo root must stay intact"
        );

        let nested_repo = cache_dir.path().join("models--org--nested");
        let snapshot = nested_repo.join("snapshots/abc1234");
        fs::create_dir_all(&snapshot).expect("Failed to create snapshot");
        symlink(outside_dir.path(), snapshot.join("escape")).expect("Failed to create symlink");

        HuggingFaceProvider
            .delete_model_revision(
                "org/nested",
                cache_dir.path().to_path_buf(),
                Some("abc1234"),
            )
            .await
            .expect("Deleting snapshot containing symlink should succeed");

        assert!(!nested_repo.exists());
        assert!(
            outside_file.exists(),
            "External target of a symlink inside a snapshot must stay intact"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_has_cached_files_after_download_in_each_weight_mode_and_offline() {
        let env_lock = acquire_env_mutex();
        let mock_server = MockHFServer::new(&env_lock).await;
        let provider = HuggingFaceProvider;

        for ignore_weights in [false, true] {
            let cache_dir = TempDir::new().expect("Failed to create cache dir");
            assert!(
                !provider
                    .has_cached_files(
                        "test/model",
                        cache_dir.path().to_path_buf(),
                        Some("def5678"),
                        ignore_weights,
                    )
                    .await
                    .expect("has_cached_files should succeed when empty"),
                "Empty cache must report false for ignore_weights={ignore_weights}"
            );

            let outcome = provider
                .download_model_revision(
                    "test/model",
                    Some(cache_dir.path().to_path_buf()),
                    ignore_weights,
                    None,
                )
                .await
                .expect("Download should succeed");

            assert!(
                provider
                    .has_cached_files(
                        "test/model",
                        cache_dir.path().to_path_buf(),
                        outcome.resolved_revision.as_deref(),
                        ignore_weights,
                    )
                    .await
                    .expect("has_cached_files should succeed after download"),
                "Downloaded model must report present for ignore_weights={ignore_weights}"
            );
            assert!(
                provider
                    .has_cached_files(
                        "test/model",
                        cache_dir.path().to_path_buf(),
                        None,
                        ignore_weights,
                    )
                    .await
                    .expect("Unpinned has_cached_files should succeed after download"),
                "Unpinned check must report present for ignore_weights={ignore_weights}"
            );

            // Deleting blobs leaves broken symlinks in snapshots/, which must report absent.
            let blobs_dir = cache_dir.path().join("models--test--model/blobs");
            if blobs_dir.exists() {
                fs::remove_dir_all(&blobs_dir).expect("Failed to remove blobs");
                assert!(
                    !provider
                        .has_cached_files(
                            "test/model",
                            cache_dir.path().to_path_buf(),
                            outcome.resolved_revision.as_deref(),
                            ignore_weights,
                        )
                        .await
                        .expect("has_cached_files should handle broken symlinks"),
                    "Broken blob symlinks must not report cached files present"
                );
            }
        }

        drop(mock_server);

        // Verify has_cached_files never contacts the Hub even when unpinned.
        let failing_hub = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/.*$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .named("has_cached_files must never call the Hub")
            .mount(&failing_hub)
            .await;
        let _hf_endpoint_guard =
            EnvVarGuard::set(&env_lock, crate::envs::HF_ENDPOINT, &failing_hub.uri());
        let _offline_guard = EnvVarGuard::remove(&env_lock, crate::envs::HF_HUB_OFFLINE);

        let local_cache = TempDir::new().expect("Failed to create local cache");
        let snapshot = local_cache
            .path()
            .join("models--test--model/snapshots/abc1234");
        fs::create_dir_all(&snapshot).expect("Failed to create snapshot");
        assert!(
            !provider
                .has_cached_files("test/model", local_cache.path().to_path_buf(), None, false)
                .await
                .expect("Empty snapshot dir should return Ok(false)")
        );
        fs::write(snapshot.join("config.json"), b"{}").expect("Failed to write config");
        assert!(
            provider
                .has_cached_files("test/model", local_cache.path().to_path_buf(), None, false)
                .await
                .expect("Populated snapshot dir should return Ok(true)")
        );
    }

    #[test]
    fn test_hf_validate_model_name_rejects_leading_and_trailing_dashes() {
        for invalid in ["a-/b", "a/-b", "-a/b", "a/b-", "-model", "model-"] {
            assert!(
                validate_model_name(invalid).is_err(),
                "Expected leading/trailing dash model name '{invalid}' to be rejected"
            );
        }
        for valid in ["org-name/model-name", "meta-llama/Llama-3.1-8B", "gpt2"] {
            assert!(
                validate_model_name(valid).is_ok(),
                "Expected valid model name '{valid}' to succeed"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_hf_delete_model_revision_rejects_symlinked_snapshots_ancestor() {
        let cache = TempDir::new().expect("create cache dir");
        let outside = TempDir::new().expect("create outside dir");
        let outside_rev = outside.path().join("rev1");
        fs::create_dir_all(&outside_rev).expect("create outside rev dir");
        fs::write(outside_rev.join("secret.txt"), b"keep").expect("write outside file");

        let repo_root = cache.path().join("models--org--model");
        fs::create_dir_all(&repo_root).expect("create repo root");
        std::os::unix::fs::symlink(outside.path(), repo_root.join("snapshots"))
            .expect("create snapshots symlink");

        let err = HuggingFaceProvider
            .delete_model_revision("org/model", cache.path().to_path_buf(), Some("rev1"))
            .await
            .expect_err("Expected symlinked snapshots ancestor to be rejected");
        assert!(
            err.to_string().contains("symlink"),
            "Unexpected error: {err}"
        );
        assert!(
            outside_rev.join("secret.txt").is_file(),
            "Outside directory must not be deleted through symlinked snapshots ancestor"
        );
    }

    #[test]
    fn test_delete_snapshot_keeping_blobs_preserves_inflight_blobs_and_repo_root() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let repo_root = temp_dir.path().join("models--org--model");
        let snapshot = repo_root.join("snapshots/rev1");
        let blobs = repo_root.join("blobs");
        let refs = repo_root.join("refs/release");
        fs::create_dir_all(&snapshot).expect("create snapshot");
        fs::create_dir_all(&blobs).expect("create blobs");
        fs::create_dir_all(&refs).expect("create refs");

        fs::write(snapshot.join("config.json"), b"{}").expect("write snapshot file");
        fs::write(blobs.join("etag1"), b"blob-data").expect("write blob");
        fs::write(blobs.join("etag2.sync.part"), b"partial").expect("write partial blob");
        fs::write(blobs.join("etag2.lock"), b"").expect("write lock");
        fs::write(repo_root.join("refs/main"), "rev1").expect("write main ref");
        fs::write(refs.join("v1"), "rev1").expect("write nested ref");
        fs::write(repo_root.join("refs/other"), "rev2").expect("write other ref");

        HuggingFaceProvider::delete_snapshot_keeping_blobs("org/model", temp_dir.path(), "rev1")
            .expect("delete_snapshot_keeping_blobs should succeed");

        assert!(!snapshot.exists(), "deleted snapshot must be removed");
        assert!(
            !repo_root.join("refs/main").exists(),
            "stale top-level ref must be removed"
        );
        assert!(
            !refs.join("v1").exists(),
            "stale nested ref must be removed"
        );
        assert!(
            repo_root.join("refs/other").exists(),
            "unrelated ref must remain"
        );
        assert!(
            repo_root.exists(),
            "repo root must remain even when snapshots/ becomes empty"
        );
        assert!(
            blobs.join("etag1").exists()
                && blobs.join("etag2.sync.part").exists()
                && blobs.join("etag2.lock").exists(),
            "all blobs and in-flight partial files must remain untouched"
        );

        assert!(
            HuggingFaceProvider::delete_snapshot_keeping_blobs(
                "org/model",
                temp_dir.path(),
                "../outside"
            )
            .is_err(),
            "traversal revision must be rejected"
        );
        assert!(
            HuggingFaceProvider::delete_snapshot_keeping_blobs(
                "org--model",
                temp_dir.path(),
                "rev1"
            )
            .is_err(),
            "double-dash model segment must be rejected"
        );
    }
}
