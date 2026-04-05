// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cache::{ModelInfo, ProviderCache};
use crate::models::ModelProvider;
use crate::providers::ModelProviderTrait;
use anyhow::{Context, Result};
use crc32c::Crc32cReader;
use fd_lock::{RwLock as FileRwLock, RwLockWriteGuard as FileWriteGuard};
use futures::Future;
use futures::StreamExt;
use futures::stream;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model::{ListObjectsRequest, Object};
use google_cloud_storage::model_ext::ReadRange;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

const CACHE_ROOT_DIR_NAME: &str = "gcs";
const MAX_GCS_LIST_RESULTS: i32 = 1000;
const MAX_PARALLEL_DOWNLOADS: usize = 8;
const PROGRESS_LOG_INTERVAL_BYTES: u64 = 128 * 1024 * 1024;
const INTERNAL_METADATA_DIR_NAME: &str = ".mx";
const MANIFEST_FILE_NAME: &str = "manifest.json";
const LOCKS_DIR_NAME: &str = "locks";
const PARTS_DIR_NAME: &str = "parts";
const MANIFEST_LOCK_FILE_NAME: &str = "manifest.lock";
const MANIFEST_VERSION: u32 = 1;
const FILE_LOCK_POLL_INTERVAL: Duration = Duration::from_secs(1);

fn ensure_crypto_provider() -> Result<()> {
    if rustls::crypto::CryptoProvider::get_default().is_some() {
        return Ok(());
    }

    match rustls::crypto::ring::default_provider().install_default() {
        Ok(()) => Ok(()),
        Err(_) if rustls::crypto::CryptoProvider::get_default().is_some() => Ok(()),
        Err(_) => anyhow::bail!("Failed to install rustls ring CryptoProvider for GCS"),
    }
}

struct LockFile;

impl LockFile {
    fn try_acquire<'a>(
        lock: &'a mut FileRwLock<fs::File>,
        lock_path: &Path,
    ) -> Result<Option<FileWriteGuard<'a, fs::File>>> {
        match lock.try_write() {
            Ok(guard) => Ok(Some(guard)),
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(None)
            }
            Err(err) => Err(anyhow::anyhow!(
                "Failed to acquire lock file '{}': {}",
                lock_path.display(),
                err
            )),
        }
    }

    fn open(path: &Path) -> Result<FileRwLock<fs::File>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create lock directory '{}'", parent.display())
            })?;
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("Failed to create lock file '{}'", path.display()))?;

        Ok(FileRwLock::new(file))
    }

    async fn with_exclusive<T, F, Fut>(lock_path: &Path, op: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut lock_file = Self::open(lock_path)?;
        let _lock = loop {
            if let Some(lock) = Self::try_acquire(&mut lock_file, lock_path)? {
                break lock;
            }
            tokio::time::sleep(FILE_LOCK_POLL_INTERVAL).await;
        };

        op().await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelName {
    bucket: BucketName,
    object_prefix: String,
}

#[derive(Debug, Clone, Copy)]
struct ModelDir<'a> {
    cache_dir: &'a Path,
    model_dir: &'a Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CacheManifest {
    version: u32,
    model: String,
    files: Vec<ManifestEntry>,
}

impl CacheManifest {
    fn new(model: &ModelName, mut files: Vec<ManifestEntry>) -> Self {
        files.sort_by(|left, right| left.path.cmp(&right.path));
        Self {
            version: MANIFEST_VERSION,
            model: model.to_string(),
            files,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestEntry {
    path: String,
    size: u64,
    crc32c: String,
    generation: Option<u64>,
}

impl ModelName {
    fn parse(model_name: &str) -> Result<Self> {
        if model_name.is_empty() {
            anyhow::bail!("Model name must not be empty");
        }

        let Some(full_url) = model_name.strip_prefix("gs://") else {
            anyhow::bail!("GCS model name must be a full gs://<bucket>/<path> URL");
        };
        let (bucket_raw, object_prefix) = full_url
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("GCS model URL must include bucket and object path"))?;
        if object_prefix.is_empty() {
            anyhow::bail!("GCS model URL must include a non-empty object path");
        }

        let bucket = BucketName::parse(bucket_raw)?;
        Self::new(bucket, object_prefix)
    }

    fn new(bucket: BucketName, object_prefix: &str) -> Result<Self> {
        let normalized_object_prefix = object_prefix.trim_end_matches('/');
        if normalized_object_prefix.is_empty() {
            anyhow::bail!("GCS model path must not be empty");
        }

        let mut components = Vec::new();
        for component in normalized_object_prefix.split('/') {
            if component.is_empty() {
                anyhow::bail!("GCS model path must not contain empty path segments");
            }
            if component == "." || component == ".." {
                anyhow::bail!("GCS model path must not contain '.' or '..' segments");
            }
            components.push(component);
        }

        if components.is_empty() {
            anyhow::bail!("GCS model path must not be empty");
        }

        Ok(Self {
            bucket,
            object_prefix: components.join("/"),
        })
    }

    fn model_dir(&self, cache_dir: &Path) -> PathBuf {
        let mut path = self.bucket_dir(cache_dir);
        for component in self.object_prefix.split('/') {
            path = path.join(component);
        }
        path
    }

    fn bucket_dir(&self, cache_dir: &Path) -> PathBuf {
        cache_dir
            .join(CACHE_ROOT_DIR_NAME)
            .join(self.bucket.as_str())
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gs://{}/{}", self.bucket, self.object_prefix)
    }
}

impl<'a> ModelDir<'a> {
    fn new(cache_dir: &'a Path, model_dir: &'a Path) -> Self {
        Self {
            cache_dir,
            model_dir,
        }
    }

    fn model_name(&self) -> Result<ModelName> {
        let gcs_root = self.cache_dir.join(CACHE_ROOT_DIR_NAME);
        let relative = self.model_dir.strip_prefix(&gcs_root).with_context(|| {
            format!(
                "GCS model directory '{}' is outside cache directory '{}'",
                self.model_dir.display(),
                gcs_root.display()
            )
        })?;
        let mut components = relative.components();
        let bucket_component = components.next().ok_or_else(|| {
            anyhow::anyhow!(
                "GCS model directory '{}' is missing bucket and object path",
                self.model_dir.display()
            )
        })?;
        let bucket_name = match bucket_component {
            Component::Normal(component) => component.to_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "GCS bucket component in '{}' is not valid UTF-8",
                    self.model_dir.display()
                )
            })?,
            _ => {
                anyhow::bail!(
                    "GCS model directory '{}' has invalid bucket component",
                    self.model_dir.display()
                )
            }
        };
        let bucket = BucketName::parse(bucket_name)?;

        let mut object_prefix_components = Vec::new();
        for component in components {
            let component = match component {
                Component::Normal(component) => component.to_str().ok_or_else(|| {
                    anyhow::anyhow!(
                        "GCS object path component in '{}' is not valid UTF-8",
                        self.model_dir.display()
                    )
                })?,
                _ => {
                    anyhow::bail!(
                        "GCS model directory '{}' has invalid object path component",
                        self.model_dir.display()
                    )
                }
            };
            object_prefix_components.push(component);
        }

        if object_prefix_components.is_empty() {
            anyhow::bail!(
                "GCS model directory '{}' is missing object path",
                self.model_dir.display()
            );
        }

        ModelName::new(bucket, &object_prefix_components.join("/"))
    }

    fn metadata_dir(&self) -> PathBuf {
        self.model_dir.join(INTERNAL_METADATA_DIR_NAME)
    }

    fn manifest_path(&self) -> PathBuf {
        self.metadata_dir().join(MANIFEST_FILE_NAME)
    }

    fn manifest_temp_path(&self) -> PathBuf {
        self.metadata_dir()
            .join(format!("{MANIFEST_FILE_NAME}.tmp"))
    }

    fn manifest_lock_path(&self) -> PathBuf {
        self.metadata_dir().join(MANIFEST_LOCK_FILE_NAME)
    }

    fn parse_relative_object_path(&self, relative_key: &str) -> Result<PathBuf> {
        if relative_key.split('/').any(str::is_empty) {
            anyhow::bail!(
                "Unsafe object path '{relative_key}' for model '{}': empty path segments are not allowed",
                self.model_name()?
            );
        }

        let path = Path::new(relative_key);
        if !path.is_safe_relative() {
            anyhow::bail!(
                "Unsafe object path '{relative_key}' for model '{}'",
                self.model_name()?
            );
        }
        if Self::is_internal_artifact(path) {
            anyhow::bail!(
                "GCS model '{}' contains reserved internal path '{}'",
                self.model_name()?,
                relative_key
            );
        }

        Ok(path.to_path_buf())
    }

    fn has_manifest_file(&self) -> bool {
        self.manifest_path().is_file()
    }

    fn write_manifest(&self, manifest: &CacheManifest) -> Result<()> {
        let manifest_path = self.manifest_path();
        let temp_path = self.manifest_temp_path();
        fs::create_dir_all(self.metadata_dir()).with_context(|| {
            format!(
                "Failed to create GCS metadata directory '{}'",
                self.metadata_dir().display()
            )
        })?;

        let mut payload = serde_json::to_vec_pretty(manifest).with_context(|| {
            format!(
                "Failed to serialize manifest for '{}'",
                self.model_dir.display()
            )
        })?;
        payload.push(b'\n');

        fs::write(&temp_path, payload)
            .with_context(|| format!("Failed to write manifest '{}'", temp_path.display()))?;
        if let Err(err) = fs::rename(&temp_path, &manifest_path).with_context(|| {
            format!(
                "Failed to move manifest '{}' into '{}'",
                temp_path.display(),
                manifest_path.display()
            )
        }) {
            let _ = fs::remove_file(&temp_path);
            return Err(err);
        }

        Ok(())
    }

    async fn ensure_manifest(
        &self,
        model: &ModelName,
        downloader: &Downloader<'_, impl google_cloud_storage::stub::Storage + 'static>,
    ) -> Result<CacheManifest> {
        if let Some(manifest) = self.load_manifest()? {
            return Ok(manifest);
        }

        LockFile::with_exclusive(&self.manifest_lock_path(), || async move {
            if let Some(manifest) = self.load_manifest()? {
                return Ok(manifest);
            }

            let manifest = CacheManifest::new(model, downloader.collect_manifest_entries().await?);
            self.write_manifest(&manifest)?;
            Ok(manifest)
        })
        .await
    }

    fn load_manifest(&self) -> Result<Option<CacheManifest>> {
        let manifest_path = self.manifest_path();
        if !manifest_path.is_file() {
            return Ok(None);
        }

        let contents = fs::read(&manifest_path)
            .with_context(|| format!("Failed to read manifest '{}'", manifest_path.display()));
        let manifest = contents.and_then(|contents| {
            serde_json::from_slice::<CacheManifest>(&contents)
                .with_context(|| format!("Failed to parse manifest '{}'", manifest_path.display()))
        });
        let Some(manifest) = (match manifest {
            Ok(manifest) => Some(manifest),
            Err(err) => {
                warn!(
                    "Ignoring unreadable GCS manifest '{}': {}",
                    manifest_path.display(),
                    err
                );
                return Ok(None);
            }
        }) else {
            return Ok(None);
        };

        let expected_model = self.model_name()?.to_string();
        let mut seen_paths = HashSet::new();
        let manifest_is_valid = manifest.version == MANIFEST_VERSION
            && manifest.model == expected_model
            && !manifest.files.is_empty()
            && manifest.files.iter().all(|entry| {
                let relative_path = Path::new(&entry.path);
                !entry.path.is_empty()
                    && relative_path.is_safe_relative()
                    && !Self::is_internal_artifact(relative_path)
                    && u32::from_str_radix(&entry.crc32c, 16).is_ok()
                    && seen_paths.insert(entry.path.clone())
            });

        if !manifest_is_valid {
            warn!(
                "Ignoring invalid GCS manifest '{}'",
                manifest_path.display(),
            );
            return Ok(None);
        }

        Ok(Some(manifest))
    }

    fn cache_satisfies_request(&self, ignore_weights: bool) -> Result<bool> {
        let Some(manifest) = self.load_manifest()? else {
            return Ok(false);
        };
        let mut matched_entry = false;
        for entry in &manifest.files {
            if ignore_weights && GcsProvider::is_weight_file(&entry.path) {
                continue;
            }

            matched_entry = true;
            let file_path = self.model_dir.join(&entry.path);
            let metadata = match fs::metadata(&file_path) {
                Ok(metadata) => metadata,
                Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "Failed to inspect manifest file '{}': {}",
                        file_path.display(),
                        err
                    ));
                }
            };
            if !metadata.is_file() {
                return Ok(false);
            }
        }
        Ok(matched_entry)
    }

    fn find_overlap(&self) -> Result<Option<(&'static str, ModelName)>> {
        let model = self.model_name()?;
        let bucket_dir = model.bucket_dir(self.cache_dir);
        let mut current = self.model_dir.parent();
        while let Some(ancestor) = current {
            if !ancestor.starts_with(&bucket_dir) {
                break;
            }

            let ancestor_model_dir = ModelDir::new(self.cache_dir, ancestor);
            if ancestor_model_dir.has_manifest_file() {
                return Ok(Some(("ancestor", ancestor_model_dir.model_name()?)));
            }

            if ancestor == bucket_dir {
                break;
            }
            current = ancestor.parent();
        }

        if !self.model_dir.is_dir() {
            return Ok(None);
        }

        let mut pending = vec![self.model_dir.to_path_buf()];
        while let Some(current_dir) = pending.pop() {
            for entry in fs::read_dir(&current_dir)
                .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let child_model_dir = ModelDir::new(self.cache_dir, &path);
                if child_model_dir.has_manifest_file() {
                    return Ok(Some(("descendant", child_model_dir.model_name()?)));
                }

                pending.push(path);
            }
        }

        Ok(None)
    }

    fn is_removable(&self) -> Result<bool> {
        if self.has_manifest_file() {
            self.model_name()?;
            return Ok(true);
        }

        if !self.model_dir.exists() {
            return Ok(false);
        }

        Ok(self.find_overlap()?.is_none())
    }

    fn remove(&self) -> Result<()> {
        if !self.is_removable()? {
            anyhow::bail!(
                "GCS model directory '{}' is not removable",
                self.model_dir.display()
            );
        }

        if self.model_dir.is_dir() {
            fs::remove_dir_all(self.model_dir).with_context(|| {
                format!(
                    "Failed to remove cached GCS model directory '{}'",
                    self.model_dir.display()
                )
            })?;
        } else {
            fs::remove_file(self.model_dir).with_context(|| {
                format!(
                    "Failed to remove cached GCS model file '{}'",
                    self.model_dir.display()
                )
            })?;
        }

        Ok(())
    }

    fn ensure_available(&self) -> Result<()> {
        let model = self.model_name()?;

        if let Some((kind, overlap)) = self.find_overlap()? {
            anyhow::bail!(
                "GCS model '{}' overlaps cached {} model '{}'",
                model,
                kind,
                overlap
            );
        }

        Ok(())
    }

    fn is_internal_artifact(relative: &Path) -> bool {
        matches!(
            relative.components().next(),
            Some(Component::Normal(component)) if component == OsStr::new(INTERNAL_METADATA_DIR_NAME)
        )
    }

    fn size(&self) -> Result<u64> {
        let mut size = 0u64;
        let mut pending = vec![self.model_dir.to_path_buf()];

        while let Some(current_dir) = pending.pop() {
            for entry in fs::read_dir(&current_dir)
                .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                let relative = path
                    .strip_prefix(self.model_dir)
                    .with_context(|| format!("Failed to make '{}' relative", path.display()))?;

                if Self::is_internal_artifact(relative) {
                    continue;
                }

                if path.is_file() {
                    size = size.saturating_add(fs::metadata(&path)?.len());
                } else if path.is_dir() {
                    pending.push(path);
                }
            }
        }

        Ok(size)
    }
}

pub(crate) struct GcsProviderCache;

impl GcsProviderCache {
    fn collect_cached_models(
        cache_dir: &Path,
        current_dir: &Path,
        models: &mut Vec<ModelInfo>,
    ) -> Result<()> {
        let current_model_dir = ModelDir::new(cache_dir, current_dir);
        if current_model_dir.has_manifest_file() {
            match current_model_dir.model_name() {
                Ok(model) => {
                    if current_model_dir.cache_satisfies_request(false)? {
                        models.push(ModelInfo {
                            provider: ModelProvider::Gcs,
                            name: model.to_string(),
                            size: current_model_dir.size()?,
                            path: current_dir.to_path_buf(),
                        });
                    }
                    return Ok(());
                }
                Err(err) => {
                    warn!(
                        "Skipping invalid GCS cache entry '{}': {}",
                        current_dir.display(),
                        err
                    );
                }
            }
        }

        for entry in fs::read_dir(current_dir)
            .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_cached_models(cache_dir, &path, models)?;
            }
        }

        Ok(())
    }
}

impl ProviderCache for GcsProviderCache {
    fn clear_model(&self, cache_dir: &Path, model_name: &str) -> Result<()> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(cache_dir);
        let model_dir_state = ModelDir::new(cache_dir, &model_dir);

        if !model_dir_state.is_removable()? {
            info!(
                "Model not found in cache: {} ({:?})",
                model_name,
                ModelProvider::Gcs
            );
            return Ok(());
        }

        model_dir_state.remove()?;
        info!("Cleared model: {} ({:?})", model_name, ModelProvider::Gcs);

        Ok(())
    }

    fn resolve_model_path(
        &self,
        cache_dir: &Path,
        model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        Ok(ModelName::parse(model_name)?.model_dir(cache_dir))
    }

    fn list_models(&self, cache_dir: &Path) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        let root = cache_dir.join(CACHE_ROOT_DIR_NAME);

        if !root.exists() {
            return Ok(models);
        }

        for bucket_entry in fs::read_dir(&root)? {
            let bucket_entry = bucket_entry?;
            let bucket_path = bucket_entry.path();
            if !bucket_path.is_dir() {
                continue;
            }

            let Some(bucket_name) = bucket_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };

            if BucketName::parse(bucket_name).is_err() {
                warn!(
                    "Skipping invalid GCS bucket cache entry '{}'",
                    bucket_path.display()
                );
                continue;
            }

            Self::collect_cached_models(cache_dir, &bucket_path, &mut models)?;
        }

        Ok(models)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BucketName {
    normalized: String,
}

impl BucketName {
    fn parse(raw: &str) -> Result<Self> {
        use std::path::{Component, Path};

        let trimmed = raw.trim();
        if trimmed.is_empty() {
            anyhow::bail!("bucket name must not be empty");
        }

        let without_scheme = trimmed.strip_prefix("gs://").unwrap_or(trimmed);
        let without_trailing = without_scheme.trim_end_matches('/');
        if without_trailing.is_empty() {
            anyhow::bail!("trimmed bucket name must not be empty");
        }

        if without_trailing.contains('/') {
            anyhow::bail!("bucket name must contain only the bucket name (no object path)");
        }

        let mut components = Path::new(without_trailing).components();
        match (components.next(), components.next()) {
            (Some(Component::Normal(_)), None) => {}
            _ => anyhow::bail!("bucket name must be a single normal path segment"),
        }

        Ok(Self {
            normalized: without_trailing.to_string(),
        })
    }

    fn as_str(&self) -> &str {
        &self.normalized
    }

    fn resource_name(&self) -> String {
        format!("projects/_/buckets/{}", self.as_str())
    }
}

impl std::fmt::Display for BucketName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
struct DownloadTask {
    object_name: String,
    relative_path: PathBuf,
    destination_path: PathBuf,
    internal_dir: PathBuf,
    expected_size: Option<u64>,
    generation: Option<i64>,
    expected_crc32c: Option<u32>,
}

impl DownloadTask {
    fn sidecar_path(&self, subdir: &str, suffix: &str) -> PathBuf {
        let mut suffixed_name = self
            .relative_path
            .file_name()
            .map(|value| value.to_os_string())
            .unwrap_or_else(|| OsStr::new("download").to_os_string());
        suffixed_name.push(suffix);

        let suffixed_relative_path = self.relative_path.with_file_name(suffixed_name);
        self.internal_dir.join(subdir).join(suffixed_relative_path)
    }

    fn lock_path(&self) -> PathBuf {
        self.sidecar_path(LOCKS_DIR_NAME, ".lock")
    }

    fn temp_path(&self) -> PathBuf {
        self.sidecar_path(PARTS_DIR_NAME, ".part")
    }

    async fn verify_file_crc32c(&self, path: &Path) -> Result<()> {
        let Some(expected_crc32c) = self.expected_crc32c else {
            anyhow::bail!(
                "Cannot verify file '{}' for '{}': remote CRC32C is unavailable",
                path.display(),
                self.object_name
            );
        };

        let verify_path = path.to_path_buf();
        let actual_crc32c =
            tokio::task::spawn_blocking(move || verify_path.calculate_file_crc32c())
                .await
                .with_context(|| {
                    format!("CRC32C verification task panicked for '{}'", path.display())
                })??;

        if actual_crc32c != expected_crc32c {
            anyhow::bail!(
                "CRC32C mismatch for file '{}' for '{}': expected {:08x}, got {:08x}",
                path.display(),
                self.object_name,
                expected_crc32c,
                actual_crc32c
            );
        }

        Ok(())
    }

    async fn has_cached_destination_file(&self) -> Result<bool> {
        let metadata = match tokio::fs::metadata(&self.destination_path).await {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Failed to inspect destination '{}': {}",
                    self.destination_path.display(),
                    err
                ));
            }
        };

        if !metadata.is_file() {
            return Ok(false);
        }

        Ok(true)
    }

    async fn remove_temp_file_if_present(&self, temp_path: &Path) -> Result<()> {
        match tokio::fs::remove_file(temp_path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(anyhow::anyhow!(
                "Failed to remove stale temp file '{}': {}",
                temp_path.display(),
                err
            )),
        }
    }

    async fn promote_verified_temp_file(&self, temp_path: &Path) -> Result<()> {
        if let Err(err) = self.verify_file_crc32c(temp_path).await {
            if let Err(remove_err) = tokio::fs::remove_file(temp_path).await {
                return Err(anyhow::anyhow!(
                    "{}; also failed to discard unverified temp file '{}': {}",
                    err,
                    temp_path.display(),
                    remove_err
                ));
            }
            return Err(err);
        }

        tokio::fs::rename(temp_path, &self.destination_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to move '{}' to '{}'",
                    temp_path.display(),
                    self.destination_path.display()
                )
            })?;

        Ok(())
    }
}

struct DownloadProgress {
    total_files: usize,
    total_bytes: Option<u64>,
    completed_files: AtomicUsize,
    downloaded_bytes: AtomicU64,
    next_log_bytes: AtomicU64,
    started_at: Instant,
}

impl DownloadProgress {
    fn new(tasks: &[DownloadTask]) -> Self {
        let mut total_bytes = Some(0u64);
        for task in tasks {
            total_bytes = match (total_bytes, task.expected_size) {
                (Some(acc), Some(size)) => acc.checked_add(size),
                _ => None,
            };
        }

        Self {
            total_files: tasks.len(),
            total_bytes,
            completed_files: AtomicUsize::new(0),
            downloaded_bytes: AtomicU64::new(0),
            next_log_bytes: AtomicU64::new(PROGRESS_LOG_INTERVAL_BYTES),
            started_at: Instant::now(),
        }
    }

    fn log_start(&self) {
        let total_size = self
            .total_bytes
            .map(Self::format_bytes)
            .unwrap_or_else(|| "unknown".to_string());
        info!(
            "Starting GCS parallel download: {} files ({} total) with {} workers",
            self.total_files, total_size, MAX_PARALLEL_DOWNLOADS
        );
    }

    fn record_downloaded_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }

        let previous = self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
        let current = previous.saturating_add(bytes);
        self.maybe_log_threshold_progress(current);
    }

    fn mark_file_completed(&self, object_name: &str) {
        let previous = self.completed_files.fetch_add(1, Ordering::Relaxed);
        let completed = previous.saturating_add(1);
        self.log_progress(&format!(
            "Completed '{}' ({}/{})",
            object_name, completed, self.total_files
        ));
    }

    fn log_finish(&self) {
        let elapsed = self.started_at.elapsed();
        self.log_progress(&format!(
            "Finished GCS download in {:.1}s",
            elapsed.as_secs_f64()
        ));
    }

    fn maybe_log_threshold_progress(&self, current_bytes: u64) {
        let mut threshold = self.next_log_bytes.load(Ordering::Relaxed);
        while current_bytes >= threshold {
            match self.next_log_bytes.compare_exchange(
                threshold,
                threshold.saturating_add(PROGRESS_LOG_INTERVAL_BYTES),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.log_progress("GCS download progress");
                    break;
                }
                Err(observed) => threshold = observed,
            }
        }
    }

    fn log_progress(&self, prefix: &str) {
        let completed = self.completed_files.load(Ordering::Relaxed);
        let downloaded = self.downloaded_bytes.load(Ordering::Relaxed);
        match self.total_bytes {
            Some(total) if total > 0 => {
                let percent = ((downloaded as f64) * 100.0 / (total as f64)).min(100.0);
                info!(
                    "{}: files {}/{}, bytes {}/{} ({:.1}%)",
                    prefix,
                    completed,
                    self.total_files,
                    Self::format_bytes(downloaded),
                    Self::format_bytes(total),
                    percent
                );
            }
            Some(_) | None => {
                info!(
                    "{}: files {}/{}, bytes {}",
                    prefix,
                    completed,
                    self.total_files,
                    Self::format_bytes(downloaded)
                );
            }
        }
    }

    fn format_bytes(bytes: u64) -> String {
        const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
        let mut value = bytes as f64;
        let mut unit_idx = 0usize;
        let last_unit_idx = UNITS.len().saturating_sub(1);
        while value >= 1024.0 && unit_idx < last_unit_idx {
            value /= 1024.0;
            unit_idx = unit_idx.saturating_add(1);
        }
        format!("{value:.1} {}", UNITS[unit_idx])
    }
}

trait PathExt {
    fn calculate_file_crc32c(&self) -> Result<u32>;
    fn is_safe_relative(&self) -> bool;
}

impl PathExt for Path {
    fn calculate_file_crc32c(&self) -> Result<u32> {
        let file = fs::File::open(self).with_context(|| {
            format!(
                "Failed to open '{}' for CRC32C verification",
                self.display()
            )
        })?;
        let mut reader = Crc32cReader::new(file);
        io::copy(&mut reader, &mut io::sink()).with_context(|| {
            format!(
                "Failed to read '{}' for CRC32C verification",
                self.display()
            )
        })?;
        Ok(reader.crc32c())
    }

    fn is_safe_relative(&self) -> bool {
        if self.is_absolute() {
            return false;
        }

        self.components()
            .all(|component| matches!(component, Component::Normal(_)))
    }
}

struct Downloader<'a, S = google_cloud_storage::stub::DefaultStorage>
where
    S: google_cloud_storage::stub::Storage + 'static,
{
    storage: &'a Storage<S>,
    control: &'a StorageControl,
    bucket: &'a BucketName,
    model: &'a ModelName,
    model_dir: ModelDir<'a>,
    bucket_resource_name: String,
    list_prefix: String,
}

impl<'a, S> Downloader<'a, S>
where
    S: google_cloud_storage::stub::Storage + 'static,
{
    fn new(
        storage: &'a Storage<S>,
        control: &'a StorageControl,
        model: &'a ModelName,
        model_dir: ModelDir<'a>,
    ) -> Self {
        Self {
            storage,
            control,
            bucket: &model.bucket,
            model,
            model_dir,
            bucket_resource_name: model.bucket.resource_name(),
            list_prefix: format!("{}/", model.object_prefix),
        }
    }

    fn build_list_request(&self, page_token: Option<String>) -> ListObjectsRequest {
        let mut request = ListObjectsRequest::new()
            .set_parent(self.bucket_resource_name.clone())
            .set_prefix(self.list_prefix.clone())
            .set_page_size(MAX_GCS_LIST_RESULTS);

        if let Some(page_token) = page_token {
            request = request.set_page_token(page_token);
        }

        request
    }

    fn ensure_parent_directory(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
        }
        Ok(())
    }

    fn candidate_relative_key<'b>(&self, object: &'b Object) -> Option<&'b str> {
        object
            .name
            .strip_prefix(&self.list_prefix)
            .filter(|suffix| !suffix.is_empty() && !suffix.ends_with('/'))
    }

    fn build_download_task(
        &self,
        relative_path: PathBuf,
        expected_size: u64,
        generation: Option<u64>,
        expected_crc32c: u32,
    ) -> Result<DownloadTask> {
        let destination_path = self.model_dir.model_dir.join(&relative_path);

        Ok(DownloadTask {
            object_name: format!("{}/{}", self.model.object_prefix, relative_path.display()),
            relative_path,
            destination_path,
            internal_dir: self.model_dir.model_dir.join(INTERNAL_METADATA_DIR_NAME),
            expected_size: Some(expected_size),
            generation: generation.and_then(|value| i64::try_from(value).ok()),
            expected_crc32c: Some(expected_crc32c),
        })
    }

    fn manifest_entry_if_downloadable(&self, object: &Object) -> Result<Option<ManifestEntry>> {
        let Some(relative_key) = self.candidate_relative_key(object) else {
            return Ok(None);
        };
        let relative_path = self.model_dir.parse_relative_object_path(relative_key)?;
        if !GcsProvider::is_downloadable_path(&relative_path) {
            return Ok(None);
        }

        let expected_size = u64::try_from(object.size).with_context(|| {
            format!(
                "Object '{}' for model '{}' has an invalid size '{}'",
                object.name, self.model, object.size
            )
        })?;
        let expected_crc32c = object
            .checksums
            .as_ref()
            .and_then(|checksums| checksums.crc32c)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Cannot record manifest entry for '{}': remote CRC32C is unavailable",
                    object.name
                )
            })?;

        Ok(Some(ManifestEntry {
            path: relative_path.to_string_lossy().into_owned(),
            size: expected_size,
            crc32c: format!("{expected_crc32c:08x}"),
            generation: (object.generation > 0)
                .then_some(object.generation)
                .and_then(|generation| u64::try_from(generation).ok()),
        }))
    }

    async fn collect_manifest_entries(&self) -> Result<Vec<ManifestEntry>> {
        let mut page_token: Option<String> = None;
        let mut entries = Vec::new();

        loop {
            let response = self
                .control
                .list_objects()
                .with_request(self.build_list_request(page_token.clone()))
                .send()
                .await
                .with_context(|| {
                    format!(
                        "Failed to list objects in gs://{}/{}",
                        self.bucket, self.model.object_prefix
                    )
                })?;

            for object in response.objects {
                if let Some(entry) = self.manifest_entry_if_downloadable(&object)? {
                    entries.push(entry);
                }
            }

            if response.next_page_token.is_empty() {
                break;
            }
            page_token = Some(response.next_page_token);
        }

        Ok(entries)
    }

    fn download_tasks_from_manifest(
        &self,
        manifest: &CacheManifest,
        ignore_weights: bool,
    ) -> Result<Vec<DownloadTask>> {
        let mut tasks = Vec::new();
        for entry in &manifest.files {
            let relative_path = self.model_dir.parse_relative_object_path(&entry.path)?;
            if !GcsProvider::matches_request_path(&relative_path, ignore_weights) {
                continue;
            }

            let expected_crc32c = u32::from_str_radix(&entry.crc32c, 16).with_context(|| {
                format!(
                    "Manifest for '{}' contains invalid CRC32C '{}'",
                    self.model, entry.crc32c
                )
            })?;

            tasks.push(self.build_download_task(
                relative_path,
                entry.size,
                entry.generation,
                expected_crc32c,
            )?);
        }

        Ok(tasks)
    }

    async fn download_tasks_in_parallel(&self, tasks: &[DownloadTask]) -> Result<()> {
        let task_count = tasks.len();
        if task_count == 0 {
            return Ok(());
        }
        let progress = Arc::new(DownloadProgress::new(tasks));
        progress.log_start();

        let download_stream = stream::iter(tasks.iter().cloned().map(|task| {
            let progress = Arc::clone(&progress);
            async move {
                self.download_task_once(&task, progress.as_ref()).await?;
                progress.mark_file_completed(&task.object_name);
                Ok::<(), anyhow::Error>(())
            }
        }))
        .buffer_unordered(MAX_PARALLEL_DOWNLOADS);

        tokio::pin!(download_stream);

        while let Some(result) = download_stream.next().await {
            result?;
        }
        progress.log_finish();

        Ok(())
    }

    async fn download_task_once(
        &self,
        task: &DownloadTask,
        progress: &DownloadProgress,
    ) -> Result<()> {
        Self::ensure_parent_directory(&task.destination_path)?;
        let temp_path = task.temp_path();
        Self::ensure_parent_directory(&temp_path)?;

        LockFile::with_exclusive(&task.lock_path(), || async move {
            let Some(resume_offset) = self.prepare_task_download(task).await? else {
                return Ok(());
            };
            self.stream_task_to_temp(task, resume_offset, progress)
                .await?;
            self.finalize_task_download(task).await
        })
        .await
    }

    async fn prepare_task_download(&self, task: &DownloadTask) -> Result<Option<u64>> {
        let temp_path = task.temp_path();
        if task.has_cached_destination_file().await? {
            task.remove_temp_file_if_present(&temp_path).await?;
            return Ok(None);
        }

        let mut resume_offset = match tokio::fs::metadata(&temp_path).await {
            Ok(metadata) => metadata.len(),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0,
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Failed to inspect '{}': {}",
                    temp_path.display(),
                    err
                ));
            }
        };
        if let Some(expected_size) = task.expected_size {
            if resume_offset > expected_size {
                tokio::fs::remove_file(&temp_path).await.with_context(|| {
                    format!(
                        "Failed to clear oversized temp file '{}'",
                        temp_path.display()
                    )
                })?;
                resume_offset = 0;
            } else if resume_offset == expected_size && resume_offset > 0 {
                match task.promote_verified_temp_file(&temp_path).await {
                    Ok(()) => return Ok(None),
                    Err(err) => {
                        warn!("{err}");
                        resume_offset = 0;
                    }
                }
            }
        }

        if resume_offset > 0 && task.generation.is_none() {
            warn!(
                "Discarding partial temp file '{}' for '{}' because remote generation is unavailable",
                temp_path.display(),
                task.object_name
            );
            tokio::fs::remove_file(&temp_path).await.with_context(|| {
                format!(
                    "Failed to discard temp file '{}' without a known remote generation",
                    temp_path.display()
                )
            })?;
            resume_offset = 0;
        }

        Ok(Some(resume_offset))
    }

    async fn stream_task_to_temp(
        &self,
        task: &DownloadTask,
        resume_offset: u64,
        progress: &DownloadProgress,
    ) -> Result<()> {
        let temp_path = task.temp_path();
        let bucket = self.bucket;
        let read_range = if resume_offset == 0 {
            ReadRange::all()
        } else {
            ReadRange::offset(resume_offset)
        };

        let mut request = self
            .storage
            .read_object(self.bucket_resource_name.clone(), task.object_name.clone())
            .set_read_range(read_range);
        if let Some(generation) = task.generation {
            request = request.set_generation(generation);
        }

        let mut stream = request.send().await.with_context(|| {
            format!(
                "Failed to start download for gs://{bucket}/{}",
                task.object_name
            )
        })?;

        let mut file = if resume_offset == 0 {
            tokio::fs::File::create(&temp_path)
                .await
                .with_context(|| format!("Failed to create '{}'", temp_path.display()))?
        } else {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&temp_path)
                .await
                .with_context(|| {
                    format!(
                        "Failed to open '{}' for resume at byte {}",
                        temp_path.display(),
                        resume_offset
                    )
                })?
        };

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| {
                format!("Failed while streaming gs://{bucket}/{}", task.object_name)
            })?;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write '{}'", temp_path.display()))?;
            progress.record_downloaded_bytes(chunk.len() as u64);
        }

        file.flush()
            .await
            .with_context(|| format!("Failed to flush '{}'", temp_path.display()))?;
        drop(file);

        Ok(())
    }

    async fn finalize_task_download(&self, task: &DownloadTask) -> Result<()> {
        let temp_path = task.temp_path();
        if let Some(expected_size) = task.expected_size {
            let final_size = tokio::fs::metadata(&temp_path)
                .await
                .with_context(|| {
                    format!(
                        "Failed to inspect completed temp file '{}'",
                        temp_path.display()
                    )
                })?
                .len();
            if final_size != expected_size {
                anyhow::bail!(
                    "Incomplete download for '{}': expected {} bytes, got {} bytes",
                    task.object_name,
                    expected_size,
                    final_size
                );
            }
        }

        task.promote_verified_temp_file(&temp_path).await
    }
}

pub struct GcsProvider;

impl GcsProvider {
    fn is_downloadable_path(path: &Path) -> bool {
        !Self::is_ignored(path.to_string_lossy().as_ref()) && !Self::is_image(path)
    }

    fn matches_request_path(path: &Path, ignore_weights: bool) -> bool {
        !ignore_weights || !Self::is_weight_file(path.to_string_lossy().as_ref())
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for GcsProvider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        let cache_dir = cache_dir
            .ok_or_else(|| anyhow::anyhow!("GCS download requires cache_dir to be provided"))?;
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory: {}", cache_dir.display())
        })?;

        let model = ModelName::parse(model_name)?;

        let model_dir = model.model_dir(&cache_dir);
        let current_model_dir = ModelDir::new(&cache_dir, &model_dir);
        if current_model_dir.cache_satisfies_request(ignore_weights)? {
            info!(
                "Using cached GCS model '{}' from '{}'",
                model,
                model_dir.display()
            );
            return Ok(model_dir);
        }

        (ModelDir::new(&cache_dir, &model_dir)).ensure_available()?;

        ensure_crypto_provider()?;
        let storage = Storage::builder()
            .build()
            .await
            .context("Failed to initialize Google Cloud Storage data client")?;
        let control = StorageControl::builder()
            .build()
            .await
            .context("Failed to initialize Google Cloud Storage control client")?;
        let downloader = Downloader::new(&storage, &control, &model, current_model_dir);
        let manifest = current_model_dir
            .ensure_manifest(&model, &downloader)
            .await?;

        if current_model_dir.cache_satisfies_request(ignore_weights)? {
            info!(
                "Using cached GCS model '{}' from '{}'",
                model,
                model_dir.display()
            );
            return Ok(model_dir);
        }

        let download_tasks = downloader.download_tasks_from_manifest(&manifest, ignore_weights)?;

        if download_tasks.is_empty() {
            anyhow::bail!("No downloadable files found in {}", model);
        }

        downloader
            .download_tasks_in_parallel(&download_tasks)
            .await?;

        if !current_model_dir.cache_satisfies_request(ignore_weights)? {
            anyhow::bail!(
                "Downloaded GCS model '{}' is still incomplete for the requested file set",
                model
            );
        }

        info!(
            "Downloaded {} files for model '{}'",
            download_tasks.len(),
            model
        );

        Ok(model_dir)
    }

    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        let model_dir_state = ModelDir::new(&cache_dir, &model_dir);

        if !model_dir_state.is_removable()? {
            info!(
                "GCS model '{}' not found in cache, skipping delete",
                model_name
            );
            return Ok(());
        }

        model_dir_state.remove()?;
        info!(
            "Deleted cached GCS model '{}' from '{}'",
            model_name,
            model_dir.display()
        );
        Ok(())
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        let model_dir_state = ModelDir::new(&cache_dir, &model_dir);
        if !model_dir_state.cache_satisfies_request(false)? {
            anyhow::bail!("GCS model '{model_name}' not found in cache");
        }

        Ok(model_dir)
    }

    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        Ok(ModelName::parse(model_name)?.to_string())
    }

    fn provider_name(&self) -> &'static str {
        "GCS"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use google_cloud_gax::{options::RequestOptions as ControlRequestOptions, response::Response};
    use google_cloud_storage::{
        model::{
            ListObjectsRequest, ListObjectsResponse, Object, ObjectChecksums, ReadObjectRequest,
        },
        model_ext::ObjectHighlights,
        read_object::ReadObjectResponse,
        request_options::RequestOptions as StorageRequestOptions,
    };
    use std::{
        collections::{HashMap, VecDeque},
        sync::{Arc, Mutex},
    };
    use tempfile::TempDir;

    fn expected_model_dir(cache_dir: &Path, model_name: &str) -> PathBuf {
        ModelName::parse(model_name)
            .expect("Expected model parsing")
            .model_dir(cache_dir)
    }

    fn expected_internal_dir(cache_dir: &Path, model_name: &str) -> PathBuf {
        expected_model_dir(cache_dir, model_name).join(INTERNAL_METADATA_DIR_NAME)
    }

    fn manifest_entry_with_generation(
        relative_path: &str,
        contents: &[u8],
        generation: u64,
    ) -> ManifestEntry {
        ManifestEntry {
            path: relative_path.to_string(),
            size: contents.len() as u64,
            crc32c: format!("{:08x}", crc32c::crc32c(contents)),
            generation: Some(generation),
        }
    }

    fn manifest_entry(relative_path: &str, contents: &[u8]) -> ManifestEntry {
        manifest_entry_with_generation(relative_path, contents, 42)
    }

    fn gcs_object(object_name: &str, contents: &[u8], generation: i64) -> Object {
        Object::new()
            .set_name(object_name)
            .set_size(contents.len() as i64)
            .set_generation(generation)
            .set_checksums(ObjectChecksums::new().set_crc32c(crc32c::crc32c(contents)))
    }

    #[derive(Debug, Clone)]
    struct TestStorageStub {
        objects: Arc<HashMap<String, String>>,
        requests: Arc<Mutex<Vec<ReadObjectRequest>>>,
    }

    impl TestStorageStub {
        fn new<I, K, V>(objects: I) -> Self
        where
            I: IntoIterator<Item = (K, V)>,
            K: Into<String>,
            V: Into<String>,
        {
            Self {
                objects: Arc::new(
                    objects
                        .into_iter()
                        .map(|(name, contents)| (name.into(), contents.into()))
                        .collect(),
                ),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn read_requests(&self) -> Vec<ReadObjectRequest> {
            self.requests
                .lock()
                .expect("Failed to lock read requests")
                .clone()
        }
    }

    impl google_cloud_storage::stub::Storage for TestStorageStub {
        fn read_object(
            &self,
            req: ReadObjectRequest,
            _options: StorageRequestOptions,
        ) -> impl Future<Output = google_cloud_storage::Result<ReadObjectResponse>> + Send {
            let objects = Arc::clone(&self.objects);
            let requests = Arc::clone(&self.requests);
            async move {
                requests
                    .lock()
                    .expect("Failed to lock read requests")
                    .push(req.clone());

                let payload = objects
                    .get(&req.object)
                    .expect("Expected test object payload")
                    .clone();
                let start = usize::try_from(req.read_offset.max(0))
                    .expect("Expected non-negative read offset");
                let resumed = payload
                    .get(start..)
                    .expect("Expected resume offset to be within payload")
                    .to_string();

                Ok(ReadObjectResponse::from_source(
                    ObjectHighlights::default(),
                    resumed,
                ))
            }
        }
    }

    #[derive(Debug, Clone)]
    struct TestStorageControlStub {
        responses: Arc<Mutex<VecDeque<Response<ListObjectsResponse>>>>,
        requests: Arc<Mutex<Vec<ListObjectsRequest>>>,
    }

    impl TestStorageControlStub {
        fn new<I>(responses: I) -> Self
        where
            I: IntoIterator<Item = ListObjectsResponse>,
        {
            Self {
                responses: Arc::new(Mutex::new(
                    responses.into_iter().map(Response::from).collect(),
                )),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn list_requests(&self) -> Vec<ListObjectsRequest> {
            self.requests
                .lock()
                .expect("Failed to lock list requests")
                .clone()
        }
    }

    impl google_cloud_storage::stub::StorageControl for TestStorageControlStub {
        fn list_objects(
            &self,
            req: ListObjectsRequest,
            _options: ControlRequestOptions,
        ) -> impl Future<Output = google_cloud_storage::Result<Response<ListObjectsResponse>>> + Send
        {
            let responses = Arc::clone(&self.responses);
            let requests = Arc::clone(&self.requests);
            async move {
                requests
                    .lock()
                    .expect("Failed to lock list requests")
                    .push(req);
                Ok(responses
                    .lock()
                    .expect("Failed to lock list responses")
                    .pop_front()
                    .expect("Expected test list_objects response"))
            }
        }
    }

    fn write_payload(cache_dir: &Path, model_name: &str, relative_path: &str, contents: &[u8]) {
        let model_dir = expected_model_dir(cache_dir, model_name);
        let payload_path = model_dir.join(relative_path);
        if let Some(parent) = payload_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create model payload directory");
        }
        fs::write(&payload_path, contents).expect("Failed to write model payload");
    }

    fn write_manifest(cache_dir: &Path, model_name: &str, entries: Vec<ManifestEntry>) {
        let model = ModelName::parse(model_name).expect("Expected model name parse");
        let model_dir = expected_model_dir(cache_dir, model_name);
        (ModelDir::new(cache_dir, &model_dir))
            .write_manifest(&CacheManifest::new(&model, entries))
            .expect("Failed to write manifest");
    }

    fn write_cached_model(
        cache_dir: &Path,
        model_name: &str,
        relative_path: &str,
        contents: &[u8],
    ) {
        write_payload(cache_dir, model_name, relative_path, contents);
        write_manifest(
            cache_dir,
            model_name,
            vec![manifest_entry(relative_path, contents)],
        );
    }

    fn write_manifest_with_payloads(
        cache_dir: &Path,
        model_name: &str,
        payloads: &[(&str, &[u8])],
        manifest_entries: Vec<ManifestEntry>,
    ) {
        for (relative_path, contents) in payloads {
            write_payload(cache_dir, model_name, relative_path, contents);
        }
        write_manifest(cache_dir, model_name, manifest_entries);
    }

    fn write_incomplete_cached_model(
        cache_dir: &Path,
        model_name: &str,
        relative_path: &str,
        contents: &[u8],
    ) {
        let model_dir = expected_model_dir(cache_dir, model_name);
        let payload_path = model_dir.join(relative_path);
        if let Some(parent) = payload_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create model payload directory");
        }
        fs::write(&payload_path, contents).expect("Failed to write model payload");
    }

    fn test_download_task(
        root: &Path,
        relative_path: &str,
        expected_crc32c: Option<u32>,
    ) -> DownloadTask {
        let relative_path = PathBuf::from(relative_path);
        DownloadTask {
            object_name: format!("org/model/rev/{}", relative_path.display()),
            relative_path: relative_path.clone(),
            destination_path: root.join(&relative_path),
            internal_dir: root.join(INTERNAL_METADATA_DIR_NAME),
            expected_size: None,
            generation: Some(42),
            expected_crc32c,
        }
    }

    #[test]
    fn test_load_manifest_rejects_invalid_entries() {
        let model_name = "gs://test-bucket/org/model/rev-1";
        for (scenario, files) in [
            (
                "duplicate_paths",
                serde_json::json!([
                    {
                        "path": "tokenizer.json",
                        "size": 2,
                        "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                        "generation": 11
                    },
                    {
                        "path": "tokenizer.json",
                        "size": 7,
                        "crc32c": format!("{:08x}", crc32c::crc32c(b"weights")),
                        "generation": 12
                    }
                ]),
            ),
            (
                "reserved_internal_path",
                serde_json::json!([{
                    "path": ".mx/manifest.json",
                    "size": 2,
                    "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                    "generation": 13
                }]),
            ),
            (
                "invalid_crc",
                serde_json::json!([{
                    "path": "tokenizer.json",
                    "size": 2,
                    "crc32c": "not-hex",
                    "generation": 14
                }]),
            ),
            (
                "wrong_model_name",
                serde_json::json!([{
                    "path": "tokenizer.json",
                    "size": 2,
                    "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                    "generation": 15
                }]),
            ),
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            let internal_dir = expected_internal_dir(temp_dir.path(), model_name);
            fs::create_dir_all(&internal_dir).expect("Failed to create metadata dir");

            let manifest = serde_json::json!({
                "version": MANIFEST_VERSION,
                "model": if scenario == "wrong_model_name" {
                    "gs://other-bucket/org/model/rev-1"
                } else {
                    model_name
                },
                "files": files
            });
            fs::write(
                internal_dir.join(MANIFEST_FILE_NAME),
                serde_json::to_vec(&manifest).expect("Expected manifest serialization"),
            )
            .expect("Failed to write manifest");

            let model_dir = ModelDir::new(temp_dir.path(), &model_dir);
            assert!(
                model_dir
                    .load_manifest()
                    .expect("Expected manifest read to succeed")
                    .is_none(),
                "scenario={scenario}"
            );
        }
    }

    #[tokio::test]
    async fn test_downloader_collect_manifest_entries_paginates_and_filters_downloadables() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let object_prefix = model.object_prefix.clone();

        let storage_stub = TestStorageStub::new(Vec::<(String, String)>::new());
        let storage = Storage::from_stub(storage_stub);
        let control_stub = TestStorageControlStub::new([
            ListObjectsResponse::new()
                .set_objects([
                    gcs_object(&format!("{object_prefix}/tokenizer.json"), b"{}", 11),
                    gcs_object(&format!("{object_prefix}/README.md"), b"readme", 12),
                    gcs_object(&format!("{object_prefix}/image.png"), b"png", 13),
                    gcs_object(&format!("{object_prefix}/"), b"", 14),
                    gcs_object("other/prefix/file.bin", b"other", 15),
                ])
                .set_next_page_token("page-2"),
            ListObjectsResponse::new().set_objects([gcs_object(
                &format!("{object_prefix}/weights/model.bin"),
                b"weights",
                16,
            )]),
        ]);
        let control = StorageControl::from_stub(control_stub.clone());
        let downloader = Downloader::new(&storage, &control, &model, model_dir);

        let mut entries = downloader
            .collect_manifest_entries()
            .await
            .expect("Expected manifest entry collection");
        entries.sort_by(|left, right| left.path.cmp(&right.path));

        assert_eq!(
            entries,
            vec![
                manifest_entry_with_generation("tokenizer.json", b"{}", 11),
                manifest_entry_with_generation("weights/model.bin", b"weights", 16),
            ]
        );

        let requests = control_stub.list_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].parent, model.bucket.resource_name());
        assert_eq!(requests[0].prefix, format!("{object_prefix}/"));
        assert_eq!(requests[0].page_size, MAX_GCS_LIST_RESULTS);
        assert!(requests[0].page_token.is_empty());
        assert_eq!(requests[1].page_token, "page-2");
    }

    #[test]
    fn test_download_tasks_from_manifest_filters_weights_and_validates_crc() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let storage = Storage::from_stub(TestStorageStub::new(Vec::<(String, String)>::new()));
        let control = StorageControl::from_stub(TestStorageControlStub::new(Vec::<
            ListObjectsResponse,
        >::new()));
        let downloader = Downloader::new(&storage, &control, &model, model_dir);

        let manifest = CacheManifest::new(
            &model,
            vec![
                manifest_entry_with_generation("tokenizer.json", b"{}", 21),
                manifest_entry_with_generation("weights/model.bin", b"weights", 22),
            ],
        );

        let metadata_only = downloader
            .download_tasks_from_manifest(&manifest, true)
            .expect("Expected task generation");
        assert_eq!(metadata_only.len(), 1);
        assert_eq!(
            metadata_only[0].relative_path,
            PathBuf::from("tokenizer.json")
        );
        assert_eq!(
            metadata_only[0].destination_path,
            model_dir_path.join("tokenizer.json")
        );
        assert_eq!(metadata_only[0].generation, Some(21));

        let full_request = downloader
            .download_tasks_from_manifest(&manifest, false)
            .expect("Expected full task generation");
        assert_eq!(full_request.len(), 2);
        assert_eq!(
            full_request[1].relative_path,
            PathBuf::from("weights/model.bin")
        );
        assert_eq!(
            full_request[1].object_name,
            format!("{}/weights/model.bin", model.object_prefix)
        );

        let invalid_crc_manifest = CacheManifest::new(
            &model,
            vec![ManifestEntry {
                path: "tokenizer.json".to_string(),
                size: 2,
                crc32c: "not-hex".to_string(),
                generation: Some(23),
            }],
        );
        assert!(
            downloader
                .download_tasks_from_manifest(&invalid_crc_manifest, false)
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_download_tasks_in_parallel_resumes_partial_downloads_and_skips_cached_files() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let tokenizer_object = format!("{}/tokenizer.json", model.object_prefix);
        let weights_object = format!("{}/weights/model.bin", model.object_prefix);

        let storage_stub = TestStorageStub::new([(weights_object.clone(), "weights".to_string())]);
        let storage = Storage::from_stub(storage_stub.clone());
        let control = StorageControl::from_stub(TestStorageControlStub::new(Vec::<
            ListObjectsResponse,
        >::new()));
        let downloader = Downloader::new(&storage, &control, &model, model_dir);
        let manifest = CacheManifest::new(
            &model,
            vec![
                manifest_entry_with_generation("tokenizer.json", b"{}", 31),
                manifest_entry_with_generation("weights/model.bin", b"weights", 32),
            ],
        );
        let tasks = downloader
            .download_tasks_from_manifest(&manifest, false)
            .expect("Expected download tasks");

        let cached_task = tasks
            .iter()
            .find(|task| task.object_name == tokenizer_object)
            .expect("Expected cached tokenizer task");
        if let Some(parent) = cached_task.destination_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create cached destination dir");
        }
        fs::write(&cached_task.destination_path, b"{}")
            .expect("Failed to write cached destination");
        if let Some(parent) = cached_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create cached temp dir");
        }
        fs::write(cached_task.temp_path(), b"stale").expect("Failed to write stale temp");

        let resumed_task = tasks
            .iter()
            .find(|task| task.object_name == weights_object)
            .expect("Expected resumable weights task");
        if let Some(parent) = resumed_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create resume temp dir");
        }
        fs::write(resumed_task.temp_path(), b"we").expect("Failed to write partial temp");

        downloader
            .download_tasks_in_parallel(&tasks)
            .await
            .expect("Expected parallel download");

        assert_eq!(
            fs::read(&cached_task.destination_path).expect("Expected cached destination payload"),
            b"{}"
        );
        assert!(
            !cached_task.temp_path().exists(),
            "Expected stale temp file to be removed"
        );
        assert_eq!(
            fs::read(&resumed_task.destination_path).expect("Expected resumed destination payload"),
            b"weights"
        );
        assert!(
            !resumed_task.temp_path().exists(),
            "Expected resumed temp file to be promoted"
        );

        let requests = storage_stub.read_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].bucket, model.bucket.resource_name());
        assert_eq!(requests[0].object, weights_object);
        assert_eq!(requests[0].read_offset, 2);
        assert_eq!(requests[0].generation, 32);
    }

    #[tokio::test]
    async fn test_prepare_and_finalize_task_download_edge_cases() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let storage = Storage::from_stub(TestStorageStub::new(Vec::<(String, String)>::new()));
        let control = StorageControl::from_stub(TestStorageControlStub::new(Vec::<
            ListObjectsResponse,
        >::new()));
        let downloader = Downloader::new(&storage, &control, &model, model_dir);

        let mismatch_task = downloader
            .build_download_task(
                PathBuf::from("weights/checksum.bin"),
                7,
                Some(41),
                crc32c::crc32c(b"different"),
            )
            .expect("Expected mismatch task");
        if let Some(parent) = mismatch_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create mismatch temp dir");
        }
        fs::write(mismatch_task.temp_path(), b"weights").expect("Failed to write mismatch temp");
        assert_eq!(
            downloader
                .prepare_task_download(&mismatch_task)
                .await
                .expect("Expected mismatch temp handling"),
            Some(0)
        );
        assert!(
            !mismatch_task.temp_path().exists(),
            "Expected mismatched temp file to be discarded"
        );

        let oversized_task = downloader
            .build_download_task(
                PathBuf::from("weights/oversized.bin"),
                3,
                Some(42),
                crc32c::crc32c(b"abc"),
            )
            .expect("Expected oversized task");
        if let Some(parent) = oversized_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create oversized temp dir");
        }
        fs::write(oversized_task.temp_path(), b"toolong").expect("Failed to write oversized temp");
        assert_eq!(
            downloader
                .prepare_task_download(&oversized_task)
                .await
                .expect("Expected oversized temp handling"),
            Some(0)
        );
        assert!(
            !oversized_task.temp_path().exists(),
            "Expected oversized temp file to be discarded"
        );

        let generationless_task = DownloadTask {
            generation: None,
            ..downloader
                .build_download_task(
                    PathBuf::from("weights/generationless.bin"),
                    7,
                    Some(43),
                    crc32c::crc32c(b"weights"),
                )
                .expect("Expected generationless task")
        };
        if let Some(parent) = generationless_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create generationless temp dir");
        }
        fs::write(generationless_task.temp_path(), b"we")
            .expect("Failed to write generationless temp");
        assert_eq!(
            downloader
                .prepare_task_download(&generationless_task)
                .await
                .expect("Expected generationless temp handling"),
            Some(0)
        );
        assert!(
            !generationless_task.temp_path().exists(),
            "Expected generationless temp file to be discarded"
        );

        let incomplete_task = downloader
            .build_download_task(
                PathBuf::from("weights/incomplete.bin"),
                7,
                Some(44),
                crc32c::crc32c(b"weights"),
            )
            .expect("Expected incomplete task");
        if let Some(parent) = incomplete_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create incomplete temp dir");
        }
        fs::write(incomplete_task.temp_path(), b"w").expect("Failed to write incomplete temp");
        let err = downloader
            .finalize_task_download(&incomplete_task)
            .await
            .expect_err("Expected finalize to reject incomplete temp");
        assert!(err.to_string().contains("Incomplete download"));
    }

    #[test]
    fn test_progress_and_crc_helpers() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let checksum_path = temp_dir.path().join("weights.bin");
        fs::write(&checksum_path, b"weights").expect("Failed to write checksum payload");
        assert_eq!(
            checksum_path
                .as_path()
                .calculate_file_crc32c()
                .expect("Expected checksum calculation"),
            crc32c::crc32c(b"weights")
        );

        let tasks = vec![
            DownloadTask {
                object_name: "org/model/rev123/tokenizer.json".to_string(),
                relative_path: PathBuf::from("tokenizer.json"),
                destination_path: temp_dir.path().join("tokenizer.json"),
                internal_dir: temp_dir.path().join(INTERNAL_METADATA_DIR_NAME),
                expected_size: Some(2),
                generation: Some(51),
                expected_crc32c: Some(crc32c::crc32c(b"{}")),
            },
            DownloadTask {
                object_name: "org/model/rev123/weights/model.bin".to_string(),
                relative_path: PathBuf::from("weights/model.bin"),
                destination_path: temp_dir.path().join("weights/model.bin"),
                internal_dir: temp_dir.path().join(INTERNAL_METADATA_DIR_NAME),
                expected_size: Some(7),
                generation: Some(52),
                expected_crc32c: Some(crc32c::crc32c(b"weights")),
            },
        ];
        let progress = DownloadProgress::new(&tasks);
        assert_eq!(progress.total_bytes, Some(9));
        progress.record_downloaded_bytes(2);
        progress.mark_file_completed("tokenizer.json");
        assert_eq!(progress.completed_files.load(Ordering::Relaxed), 1);
        assert_eq!(progress.downloaded_bytes.load(Ordering::Relaxed), 2);
        progress.log_start();
        progress.log_finish();

        assert_eq!(DownloadProgress::format_bytes(0), "0.0 B");
        assert_eq!(DownloadProgress::format_bytes(1536), "1.5 KiB");
        assert_eq!(
            BucketName::parse("gs://bucket-name/")
                .expect("Expected bucket parse")
                .resource_name(),
            "projects/_/buckets/bucket-name"
        );
    }

    #[test]
    fn test_mx_path_layout_tables() {
        assert_eq!(GcsProvider.provider_name(), "GCS");

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir);

        for (path, expected) in [
            (
                model_dir.manifest_path(),
                expected_internal_dir(temp_dir.path(), model_name).join(MANIFEST_FILE_NAME),
            ),
            (
                model_dir.manifest_temp_path(),
                expected_internal_dir(temp_dir.path(), model_name).join("manifest.json.tmp"),
            ),
            (
                model_dir.manifest_lock_path(),
                expected_internal_dir(temp_dir.path(), model_name).join(MANIFEST_LOCK_FILE_NAME),
            ),
        ] {
            assert_eq!(path, expected);
        }

        for (relative_path, expected_temp, expected_lock) in [
            (
                "weights/model.bin",
                "/tmp/cache/model/.mx/parts/weights/model.bin.part",
                "/tmp/cache/model/.mx/locks/weights/model.bin.lock",
            ),
            (
                "weights/model.mlir",
                "/tmp/cache/model/.mx/parts/weights/model.mlir.part",
                "/tmp/cache/model/.mx/locks/weights/model.mlir.lock",
            ),
            (
                "weights/model",
                "/tmp/cache/model/.mx/parts/weights/model.part",
                "/tmp/cache/model/.mx/locks/weights/model.lock",
            ),
        ] {
            let task = test_download_task(Path::new("/tmp/cache/model"), relative_path, None);
            assert_eq!(task.temp_path(), PathBuf::from(expected_temp));
            assert_eq!(task.lock_path(), PathBuf::from(expected_lock));
        }

        for (path, expected_reserved) in [
            (".mx/manifest.json", true),
            (".mx/parts/weights/model.bin.part", true),
            ("weights/model.bin", false),
        ] {
            assert_eq!(
                ModelDir::is_internal_artifact(Path::new(path)),
                expected_reserved,
                "path={path}"
            );
        }

        let root = temp_dir.path().join("size-case");
        fs::create_dir_all(root.join("nested")).expect("Failed to create nested dirs");
        fs::create_dir_all(root.join(".mx/parts/nested")).expect("Failed to create parts dir");
        fs::create_dir_all(root.join(".mx/locks/nested")).expect("Failed to create locks dir");
        fs::write(root.join("nested/model.bin"), b"weights").expect("Failed to create model file");
        fs::write(root.join(".mx/parts/nested/model.bin.part"), b"partial")
            .expect("Failed to create temp file");
        fs::write(root.join(".mx/locks/nested/model.bin.lock"), b"lock")
            .expect("Failed to create lock file");
        assert_eq!(
            (ModelDir::new(&root, &root))
                .size()
                .expect("Expected model dir size"),
            7
        );
    }

    #[test]
    fn test_write_manifest_scenarios() {
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model = ModelName::parse(model_name).expect("Expected model parse");
        let cases = [
            (
                "create",
                None,
                CacheManifest::new(&model, vec![manifest_entry("tokenizer.json", b"{}")]),
            ),
            (
                "replace",
                Some(CacheManifest::new(
                    &model,
                    vec![manifest_entry("tokenizer.json", b"{}")],
                )),
                CacheManifest::new(&model, vec![manifest_entry("config.json", br#"{"a":1}"#)]),
            ),
            (
                "expand",
                Some(CacheManifest::new(
                    &model,
                    vec![manifest_entry("tokenizer.json", b"{}")],
                )),
                CacheManifest::new(
                    &model,
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
            ),
        ];

        for (name, initial_manifest, final_manifest) in cases {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            let model_dir_state = ModelDir::new(temp_dir.path(), &model_dir);

            if let Some(initial_manifest) = initial_manifest {
                model_dir_state
                    .write_manifest(&initial_manifest)
                    .expect("Expected initial manifest write");
            }

            model_dir_state
                .write_manifest(&final_manifest)
                .expect("Expected manifest write");

            let parsed: CacheManifest = serde_json::from_slice(
                &fs::read(model_dir_state.manifest_path()).expect("Expected manifest contents"),
            )
            .expect("Expected manifest JSON");
            assert_eq!(parsed, final_manifest, "scenario={name}");
            assert!(
                !model_dir_state.manifest_temp_path().exists(),
                "scenario={name}"
            );
        }
    }

    #[test]
    fn test_model_name_tables_cover_normalization_paths_and_cache_roundtrip() {
        let valid_cases = [
            (
                "gs://testbucket/dev/bake/qwen/rev123",
                "gs://testbucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/testbucket/dev/bake/qwen/rev123"),
            ),
            (
                "gs://testbucket/dev/bake/qwen/rev123/",
                "gs://testbucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/testbucket/dev/bake/qwen/rev123"),
            ),
            (
                "gs://sourcebucket/dev/bake/qwen/rev123",
                "gs://sourcebucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/sourcebucket/dev/bake/qwen/rev123"),
            ),
            (
                "gs://sourcebucket/dev/bake/qwen/rev123/",
                "gs://sourcebucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/sourcebucket/dev/bake/qwen/rev123"),
            ),
        ];

        for (input, canonical, expected_model_dir) in valid_cases {
            let model = ModelName::parse(input).expect("Expected model parsing");
            assert_eq!(model.to_string(), canonical, "input={input}");
            assert_eq!(
                model.model_dir(Path::new("/tmp/cache")),
                expected_model_dir,
                "input={input}"
            );
        }

        for invalid in [
            "",
            "gs://bucket-only",
            "gs://bucket-only/",
            "dev/bake/qwen/rev123",
        ] {
            assert!(ModelName::parse(invalid).is_err(), "input={invalid}");
        }

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        write_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");
        let model = ModelDir::new(temp_dir.path(), &model_dir)
            .model_name()
            .expect("Expected cached model name");
        assert_eq!(model.to_string(), model_name);
    }

    #[test]
    fn test_bucket_name_parse_table() {
        let valid_cases = [("gs://example-bucket/", "example-bucket")];
        for (input, expected) in valid_cases {
            assert_eq!(
                BucketName::parse(input)
                    .expect("Expected bucket normalization")
                    .as_str(),
                expected,
                "input={input}"
            );
        }

        for invalid in ["example-bucket/path", "..", ".", "gs://../"] {
            assert!(BucketName::parse(invalid).is_err(), "input={invalid}");
        }
    }

    #[test]
    fn test_download_model_validation_cases() {
        let provider = GcsProvider;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime");

        for (name, model_name, cache_dir, expected_error) in [
            (
                "missing_cache_dir",
                "gs://test-bucket/test/model/rev-1",
                None,
                "requires cache_dir",
            ),
            (
                "relative_path",
                "org/model/rev-2",
                Some(TempDir::new().expect("Failed to create temp dir")),
                "full gs://<bucket>/<path> URL",
            ),
        ] {
            let result = runtime.block_on(provider.download_model(
                model_name,
                cache_dir.as_ref().map(|dir| dir.path().to_path_buf()),
                false,
            ));
            assert!(
                result
                    .expect_err("Expected validation failure")
                    .to_string()
                    .contains(expected_error),
                "scenario={name}"
            );
        }
    }

    #[test]
    fn test_download_model_cached_cases() {
        let provider = GcsProvider;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime");
        let cases = [
            (
                "full",
                "gs://test-bucket/org/model/rev-1",
                false,
                vec![("weights/model.bin", b"weights".as_slice())],
                vec![manifest_entry("weights/model.bin", b"weights")],
            ),
            (
                "metadata_subset",
                "gs://test-bucket/org/model/rev-meta",
                true,
                vec![("tokenizer.json", b"{}".as_slice())],
                vec![
                    manifest_entry("tokenizer.json", b"{}"),
                    manifest_entry("weights/model.bin", b"weights"),
                ],
            ),
        ];

        for (name, model_name, ignore_weights, payloads, manifest_entries) in cases {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            write_manifest_with_payloads(temp_dir.path(), model_name, &payloads, manifest_entries);

            let result = runtime
                .block_on(provider.download_model(
                    model_name,
                    Some(temp_dir.path().to_path_buf()),
                    ignore_weights,
                ))
                .expect("Expected cached model reuse");

            assert_eq!(result, model_dir, "scenario={name}");
        }
    }

    #[test]
    fn test_relative_object_path_tables_cover_safety_downloadability_and_weight_matching() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = expected_model_dir(temp_dir.path(), "gs://test-bucket/org/model/rev-1");
        let model_dir = ModelDir::new(temp_dir.path(), &model_path);
        let relative_path_cases = [
            ("README.md", false, true, true, false),
            ("nested/README.md", false, true, true, false),
            (".gitattributes", false, true, true, false),
            ("image.png", false, true, true, false),
            ("nested/image.webp", false, true, true, false),
            ("config.json", true, true, true, false),
            ("tokenizer.json", true, true, true, false),
            ("foo.mlir", true, true, true, false),
            ("compile_summary.json", true, true, true, false),
            ("manifest.json", true, true, true, false),
            ("nested/preset.json", true, true, true, false),
            ("model.safetensors", true, false, true, true),
            ("weights/model.bin", true, false, true, true),
            ("weights/model.iop", true, false, true, true),
            ("weights/model.gas", true, false, true, true),
            ("weights/model.h5", true, false, true, true),
            ("weights/model.msgpack", true, false, true, true),
            ("weights/model.ckpt.index", true, false, true, true),
        ];

        for (
            relative_path,
            expected_downloadable,
            expected_metadata_match,
            expected_safe,
            expected_weight,
        ) in relative_path_cases
        {
            let relative = model_dir
                .parse_relative_object_path(relative_path)
                .expect("Expected valid relative object path");
            assert_eq!(
                relative.is_safe_relative(),
                expected_safe,
                "safe check for {relative_path}"
            );
            assert_eq!(
                GcsProvider::is_downloadable_path(&relative),
                expected_downloadable,
                "downloadable check for {relative_path}"
            );
            assert_eq!(
                GcsProvider::matches_request_path(&relative, true),
                expected_metadata_match,
                "metadata match for {relative_path}"
            );
            assert!(
                GcsProvider::matches_request_path(&relative, false),
                "full request match for {relative_path}"
            );
            assert_eq!(
                GcsProvider::is_weight_file(relative.to_string_lossy().as_ref()),
                expected_weight,
                "weight check for {relative_path}"
            );
        }

        for path in ["weights/model.safetensors", "tokenizer.json"] {
            assert!(
                Path::new(path).is_safe_relative(),
                "expected safe relative path: {path}"
            );
        }
        for path in ["../model.safetensors", "/etc/passwd"] {
            assert!(
                !Path::new(path).is_safe_relative(),
                "expected unsafe relative path: {path}"
            );
        }
    }

    #[test]
    fn test_relative_path_validation_tables() {
        fn candidate<'a>(object_name: &'a str, list_prefix: &str) -> Option<&'a str> {
            object_name
                .strip_prefix(list_prefix)
                .filter(|suffix| !suffix.is_empty() && !suffix.ends_with('/'))
        }

        assert!(candidate("other/model/file.bin", "org/model/rev123/").is_none());
        assert!(candidate("org/model/rev123/", "org/model/rev123/").is_none());
        assert!(candidate("org/model/rev123/subdir/", "org/model/rev123/").is_none());
        assert_eq!(
            candidate("org/model/rev123/tokenizer.json", "org/model/rev123/"),
            Some("tokenizer.json")
        );
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = expected_model_dir(temp_dir.path(), "gs://test-bucket/org/model/rev-1");
        let model_dir = ModelDir::new(temp_dir.path(), &model_path);
        for invalid in ["../secret.bin", "a//b.bin", "/a/b.bin", "a/b//c.bin"] {
            assert!(
                model_dir.parse_relative_object_path(invalid).is_err(),
                "input={invalid}"
            );
        }

        for (path, expected_valid) in [
            ("weights/model.bin", true),
            ("", false),
            ("/etc/passwd", false),
            ("../escape", false),
        ] {
            let path = Path::new(path);
            let is_valid_relative = !path.as_os_str().is_empty() && path.is_safe_relative();
            assert_eq!(is_valid_relative, expected_valid, "path={:?}", path);
        }
    }

    #[test]
    fn test_request_is_satisfied_scenarios() {
        for scenario in ["metadata_only_payload", "invalid_manifest"] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_name = match scenario {
                "metadata_only_payload" => "gs://bucket/foo/bar/baz",
                "invalid_manifest" => "gs://bucket/foo/bar/bad",
                _ => unreachable!("unexpected scenario"),
            };
            let model_dir = expected_model_dir(temp_dir.path(), model_name);

            match scenario {
                "metadata_only_payload" => write_manifest_with_payloads(
                    temp_dir.path(),
                    model_name,
                    &[("tokenizer.json", b"{}")],
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
                "invalid_manifest" => {
                    let internal_dir = expected_internal_dir(temp_dir.path(), model_name);
                    fs::create_dir_all(&internal_dir).expect("Failed to create metadata dir");
                    let invalid_manifest = serde_json::json!({
                        "version": MANIFEST_VERSION + 1,
                        "model": model_name,
                        "files": [{
                            "path": "tokenizer.json",
                            "size": 2,
                            "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                            "generation": 42
                        }]
                    });
                    fs::write(
                        internal_dir.join(MANIFEST_FILE_NAME),
                        serde_json::to_vec(&invalid_manifest)
                            .expect("Expected manifest serialization"),
                    )
                    .expect("Failed to write invalid manifest");
                    fs::create_dir_all(&model_dir).expect("Failed to create model dir");
                    fs::write(model_dir.join("tokenizer.json"), b"{}")
                        .expect("Failed to write payload");
                }
                _ => unreachable!("unexpected scenario"),
            }

            let model_dir = ModelDir::new(temp_dir.path(), &model_dir);
            match scenario {
                "metadata_only_payload" => {
                    assert!(
                        model_dir
                            .cache_satisfies_request(true)
                            .expect("Expected metadata request to be satisfied")
                    );
                    assert!(
                        !model_dir
                            .cache_satisfies_request(false)
                            .expect("Expected full request to remain incomplete")
                    );
                }
                "invalid_manifest" => {
                    assert!(
                        !model_dir
                            .cache_satisfies_request(false)
                            .expect("Expected invalid manifest to be ignored")
                    );
                }
                _ => unreachable!("unexpected scenario"),
            }
        }
    }

    #[tokio::test]
    async fn test_download_task_file_verification_cases() {
        struct PromoteCase {
            name: &'static str,
            temp_payload: &'static [u8],
            expected_crc32c: Option<u32>,
            expect_error_substr: Option<&'static str>,
        }

        for case in [
            PromoteCase {
                name: "match",
                temp_payload: b"weights",
                expected_crc32c: Some(crc32c::crc32c(b"weights")),
                expect_error_substr: None,
            },
            PromoteCase {
                name: "mismatch",
                temp_payload: b"weights",
                expected_crc32c: Some(crc32c::crc32c(b"different")),
                expect_error_substr: Some("CRC32C mismatch"),
            },
            PromoteCase {
                name: "missing_crc",
                temp_payload: b"weights",
                expected_crc32c: None,
                expect_error_substr: Some("remote CRC32C is unavailable"),
            },
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let temp_path = temp_dir.path().join("weights.bin.tmp");
            fs::write(&temp_path, case.temp_payload).expect("Failed to write temp file");

            let task = test_download_task(temp_dir.path(), "weights.bin", case.expected_crc32c);
            let destination_path = task.destination_path.clone();

            match case.expect_error_substr {
                None => {
                    task.promote_verified_temp_file(&temp_path)
                        .await
                        .expect("Expected checksum verification");
                    assert!(!temp_path.exists(), "scenario={}", case.name);
                    assert_eq!(
                        fs::read(&destination_path).expect("Expected promoted destination file"),
                        case.temp_payload,
                        "scenario={}",
                        case.name
                    );
                }
                Some(error_substr) => {
                    let err = task
                        .promote_verified_temp_file(&temp_path)
                        .await
                        .expect_err("Expected promote failure");
                    assert!(
                        err.to_string().contains(error_substr),
                        "scenario={} err={err}",
                        case.name
                    );
                    assert!(!temp_path.exists(), "scenario={}", case.name);
                    assert!(!destination_path.exists(), "scenario={}", case.name);
                }
            }
        }

        for (name, setup, expected_match) in [
            ("file", "file", true),
            ("missing", "missing", false),
            ("directory", "directory", false),
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let task = test_download_task(
                temp_dir.path(),
                "weights.bin",
                Some(crc32c::crc32c(b"weights")),
            );

            match setup {
                "file" => {
                    fs::write(&task.destination_path, b"weights")
                        .expect("Failed to create destination file");
                }
                "missing" => {}
                "directory" => {
                    fs::create_dir_all(&task.destination_path)
                        .expect("Failed to create destination directory");
                }
                _ => unreachable!("unexpected setup"),
            }

            assert_eq!(
                task.has_cached_destination_file()
                    .await
                    .expect("Expected destination inspection"),
                expected_match,
                "scenario={name}"
            );
        }
    }

    #[test]
    fn test_list_models_scenarios() {
        let cache = GcsProviderCache;
        for scenario in [
            "partial_manifest",
            "recursive_siblings",
            "model_path_contains_internal_segment",
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            match scenario {
                "partial_manifest" => write_manifest_with_payloads(
                    temp_dir.path(),
                    "gs://bucket/foo/bar/baz",
                    &[("tokenizer.json", b"{}")],
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
                "recursive_siblings" => {
                    write_cached_model(
                        temp_dir.path(),
                        "gs://bucket/foo/bar/baz",
                        "tokenizer.json",
                        b"{}",
                    );
                    write_cached_model(
                        temp_dir.path(),
                        "gs://bucket/foo/bar/buz",
                        "weights/model.bin",
                        b"abcd",
                    );
                }
                "model_path_contains_internal_segment" => {
                    write_cached_model(
                        temp_dir.path(),
                        "gs://bucket/foo/.mx/bar",
                        "tokenizer.json",
                        b"{}",
                    );
                }
                _ => unreachable!("unexpected scenario"),
            }

            let mut models = cache
                .list_models(temp_dir.path())
                .expect("Expected model listing");
            models.sort_by(|left, right| left.name.cmp(&right.name));

            match scenario {
                "partial_manifest" => {
                    assert!(models.is_empty(), "partial manifest should not be listed");
                }
                "recursive_siblings" => {
                    assert_eq!(models.len(), 2);
                    assert_eq!(models[0].name, "gs://bucket/foo/bar/baz");
                    assert_eq!(models[0].size, 2);
                    assert_eq!(models[1].name, "gs://bucket/foo/bar/buz");
                    assert_eq!(models[1].size, 4);
                }
                "model_path_contains_internal_segment" => {
                    assert_eq!(models.len(), 1);
                    assert_eq!(models[0].name, "gs://bucket/foo/.mx/bar");
                    assert_eq!(models[0].size, 2);
                }
                _ => unreachable!("unexpected scenario"),
            }
        }
    }

    #[test]
    fn test_clear_model_scenarios() {
        let cache = GcsProviderCache;
        for scenario in [
            "ancestor_keeps_descendant",
            "incomplete_removed",
            "descendant_keeps_cached_ancestor",
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            match scenario {
                "ancestor_keeps_descendant" => {
                    let ancestor_name = "gs://bucket/foo/bar";
                    let descendant_name = "gs://bucket/foo/bar/baz";
                    write_cached_model(temp_dir.path(), descendant_name, "tokenizer.json", b"{}");
                    cache
                        .clear_model(temp_dir.path(), ancestor_name)
                        .expect("Expected clear to succeed");
                    assert!(expected_model_dir(temp_dir.path(), descendant_name).exists());
                }
                "incomplete_removed" => {
                    let model_name = "gs://bucket/foo/bar";
                    let model_dir = expected_model_dir(temp_dir.path(), model_name);
                    write_incomplete_cached_model(
                        temp_dir.path(),
                        model_name,
                        "tokenizer.json",
                        b"{}",
                    );
                    cache
                        .clear_model(temp_dir.path(), model_name)
                        .expect("Expected clear to succeed");
                    assert!(!model_dir.exists());
                }
                "descendant_keeps_cached_ancestor" => {
                    let ancestor_name = "gs://bucket/foo/bar";
                    let descendant_name = "gs://bucket/foo/bar/baz";
                    let ancestor_dir = expected_model_dir(temp_dir.path(), ancestor_name);
                    let descendant_dir = expected_model_dir(temp_dir.path(), descendant_name);
                    write_cached_model(temp_dir.path(), ancestor_name, "tokenizer.json", b"{}");
                    fs::create_dir_all(&descendant_dir).expect("Failed to create descendant dir");
                    fs::write(descendant_dir.join("partial.bin"), b"partial")
                        .expect("Failed to create descendant payload");
                    cache
                        .clear_model(temp_dir.path(), descendant_name)
                        .expect("Expected clear to succeed");
                    assert!(ancestor_dir.exists());
                    assert!(ancestor_dir.join("tokenizer.json").exists());
                    assert!(descendant_dir.exists());
                }
                _ => unreachable!("unexpected scenario"),
            }
        }
    }

    #[test]
    fn test_ensure_model_dir_available_rejects_ancestor_descendant_overlap() {
        for (cached_model, requested_model) in [
            ("gs://bucket/foo/bar/baz", "gs://bucket/foo/bar"),
            ("gs://bucket/foo/.mx/bar", "gs://bucket/foo"),
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            write_cached_model(temp_dir.path(), cached_model, "tokenizer.json", b"{}");

            let model = ModelName::parse(requested_model).expect("Expected model parse");
            let model_dir = model.model_dir(temp_dir.path());

            let err = (ModelDir::new(temp_dir.path(), &model_dir))
                .ensure_available()
                .expect_err("Expected overlap rejection");

            assert!(
                err.to_string().contains("overlaps cached descendant model"),
                "cached_model={cached_model} requested_model={requested_model} err={err}"
            );
        }
    }

    #[tokio::test]
    async fn test_get_model_path_scenarios() {
        let provider = GcsProvider;
        for scenario in ["missing", "partial_manifest", "complete"] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_name = match scenario {
                "missing" => "gs://test-bucket/org/model/rev-3",
                "partial_manifest" => "gs://test-bucket/org/model/rev-partial",
                "complete" => "gs://test-bucket/org/model/rev-5",
                _ => unreachable!("unexpected scenario"),
            };

            match scenario {
                "partial_manifest" => write_manifest_with_payloads(
                    temp_dir.path(),
                    model_name,
                    &[("tokenizer.json", b"{}")],
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
                "complete" => {
                    write_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");
                }
                "missing" => {}
                _ => unreachable!("unexpected scenario"),
            }

            let result = provider
                .get_model_path(model_name, temp_dir.path().to_path_buf())
                .await;
            match scenario {
                "complete" => assert_eq!(
                    result.expect("Expected model path"),
                    expected_model_dir(temp_dir.path(), model_name)
                ),
                _ => assert!(
                    result
                        .expect_err("Expected cache miss")
                        .to_string()
                        .contains("not found in cache")
                ),
            }
        }
    }

    #[tokio::test]
    async fn test_delete_model_scenarios() {
        let provider = GcsProvider;
        for scenario in ["complete", "incomplete"] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_name = "gs://test-bucket/org/model/rev-1";
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            match scenario {
                "complete" => {
                    write_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");
                }
                "incomplete" => {
                    write_incomplete_cached_model(
                        temp_dir.path(),
                        model_name,
                        "tokenizer.json",
                        b"{}",
                    );
                }
                _ => unreachable!("unexpected scenario"),
            }
            provider
                .delete_model(model_name, temp_dir.path().to_path_buf())
                .await
                .expect("Expected successful delete");
            assert!(!model_dir.exists());
        }
    }
}
