// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    cache::{ModelInfo, ProviderCache},
    models::ModelProvider,
    providers::{ModelProviderTrait, ensure_crypto_provider, reject_unsupported_revision},
};
use anyhow::{Context, Result};
use futures::StreamExt;
use object_store::{
    ClientOptions, ObjectStore, ObjectStoreExt, aws::AmazonS3Builder, path::Path as ObjectPath,
};
use std::{
    fs, io,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::io::AsyncWriteExt;
use tracing::info;

const CACHE_ROOT_DIR_NAME: &str = "s3";
const DEFAULT_REGION: &str = "us-east-1";
const DEFAULT_TIMEOUT_SECS: u64 = 3600;
const MODEL_MARKER_FILE_NAME: &str = ".mx-model";

#[derive(Debug, Clone, PartialEq, Eq)]
struct S3ModelName {
    bucket: String,
    object_prefix: String,
}

fn validate_s3_segment(segment: &str, label: &str) -> Result<()> {
    if segment.is_empty() || segment.trim() != segment {
        anyhow::bail!("S3 {label} must be non-empty without surrounding whitespace");
    }
    if segment.contains('\0') || segment.contains('/') || segment == "." || segment == ".." {
        anyhow::bail!("S3 {label} '{segment}' is not a valid path segment");
    }
    let mut components = Path::new(segment).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => anyhow::bail!("S3 {label} '{segment}' must be a single normal path segment"),
    }
}

fn has_marker_file(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(meta) => Ok(meta.file_type().is_file()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err).with_context(|| format!("Failed to inspect '{}'", path.display())),
    }
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("Failed to remove '{}'", path.display())),
    }
}

/// Recursively delete `dir` while preserving any descendant directory that
/// contains `.mx-model` and the parent chain leading to it.
///
/// Returns `Ok(true)` if `dir` was kept to preserve a cached descendant model,
/// or `Ok(false)` if `dir` was removed (or already absent).
fn prune_s3_dir(dir: &Path, is_root: bool) -> Result<bool> {
    if is_root {
        // Remove the target marker first so a mid-walk failure cannot leave a
        // partially deleted directory looking cached.
        remove_file_if_exists(&dir.join(MODEL_MARKER_FILE_NAME))?;
    } else if has_marker_file(&dir.join(MODEL_MARKER_FILE_NAME))? {
        return Ok(true);
    }

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => {
            return Err(err).with_context(|| format!("Failed to read '{}'", dir.display()));
        }
    };

    let mut preserved_descendant = false;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("Failed to read entry in '{}'", dir.display()));
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("Failed to inspect '{}'", path.display()));
            }
        };
        if file_type.is_dir() {
            if prune_s3_dir(&path, false)? {
                preserved_descendant = true;
            }
        } else {
            remove_file_if_exists(&path)?;
        }
    }

    if preserved_descendant {
        return Ok(true);
    }

    match fs::remove_dir(dir) {
        Ok(()) => Ok(false),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err).with_context(|| format!("Failed to remove '{}'", dir.display())),
    }
}

fn remove_s3_model_dir(cache_dir: &Path, model: &S3ModelName) -> Result<()> {
    let bucket_dir = cache_dir.join(CACHE_ROOT_DIR_NAME).join(&model.bucket);
    let model_dir = model.model_dir(cache_dir);

    if !crate::providers::ensure_delete_target_within_cache_root(cache_dir, &model_dir)? {
        return Ok(());
    }

    let meta = match fs::symlink_metadata(&model_dir) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(err)
                .with_context(|| format!("Failed to inspect '{}'", model_dir.display()));
        }
    };

    if meta.file_type().is_symlink() || !meta.file_type().is_dir() {
        return remove_file_if_exists(&model_dir);
    }

    // If a cached ancestor model owns this prefix, keep all payload files and
    // remove only the target's own marker file (if present).
    let mut cursor = model_dir.parent();
    while let Some(ancestor) = cursor {
        if ancestor == bucket_dir || !ancestor.starts_with(&bucket_dir) {
            break;
        }
        if has_marker_file(&ancestor.join(MODEL_MARKER_FILE_NAME))? {
            remove_file_if_exists(&model_dir.join(MODEL_MARKER_FILE_NAME))?;
            return Ok(());
        }
        cursor = ancestor.parent();
    }

    prune_s3_dir(&model_dir, true)?;
    Ok(())
}

impl S3ModelName {
    fn parse(model_name: &str) -> Result<Self> {
        let model_name = model_name.trim();
        let Some(scheme) = model_name.get(.."s3://".len()) else {
            anyhow::bail!("S3 model name must be a full s3://<bucket>/<path> URL");
        };
        if !scheme.eq_ignore_ascii_case("s3://") {
            anyhow::bail!("S3 model name must be a full s3://<bucket>/<path> URL");
        }
        let full_url = &model_name["s3://".len()..];
        let (bucket, object_prefix) = full_url
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("S3 model URL must include bucket and object path"))?;
        validate_s3_segment(bucket, "bucket")?;
        Self::new(bucket, object_prefix)
    }

    fn new(bucket: &str, object_prefix: &str) -> Result<Self> {
        let object_prefix = object_prefix.trim_end_matches('/');
        if object_prefix.is_empty() {
            anyhow::bail!("S3 model path must not be empty");
        }

        for component in object_prefix.split('/') {
            validate_s3_segment(component, "model path segment")?;
        }

        Ok(Self {
            bucket: bucket.to_string(),
            object_prefix: object_prefix.to_string(),
        })
    }

    fn model_dir(&self, cache_dir: &Path) -> PathBuf {
        let mut path = cache_dir.join(CACHE_ROOT_DIR_NAME).join(&self.bucket);
        for component in self.object_prefix.split('/') {
            path = path.join(component);
        }
        path
    }
}

impl std::fmt::Display for S3ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s3://{}/{}", self.bucket, self.object_prefix)
    }
}

pub struct S3Provider;

impl S3Provider {
    fn is_downloadable_path(path: &Path) -> bool {
        !Self::is_ignored(path.to_string_lossy().as_ref()) && !Self::is_image(path)
    }

    fn build_store_builder(bucket: &str) -> AmazonS3Builder {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());
        let endpoint = std::env::var("AWS_ENDPOINT_URL")
            .or_else(|_| std::env::var("AWS_ENDPOINT"))
            .or_else(|_| std::env::var("S3_ENDPOINT"))
            .ok();
        let timeout_secs = std::env::var("MODEL_EXPRESS_S3_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let mut builder = AmazonS3Builder::from_env()
            .with_region(region)
            .with_bucket_name(bucket)
            .with_client_options(
                ClientOptions::new().with_timeout(Duration::from_secs(timeout_secs)),
            );

        #[cfg(feature = "tls-native")]
        {
            builder = builder
                .with_crypto_provider(Arc::new(crate::providers::s3_crypto::OpensslCryptoProvider));
        }

        if let Some(endpoint) = endpoint {
            let allow_http = endpoint.starts_with("http://")
                || std::env::var("AWS_ALLOW_HTTP")
                    .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
            builder = builder
                .with_endpoint(endpoint)
                .with_virtual_hosted_style_request(false)
                .with_allow_http(allow_http);
        }

        builder
    }

    fn build_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        ensure_crypto_provider()?;
        Ok(Arc::new(Self::build_store_builder(bucket).build()?))
    }

    fn relative_path(model: &S3ModelName, location: &ObjectPath) -> Result<Option<PathBuf>> {
        let key = location.as_ref();
        let Some(relative) = key
            .strip_prefix(&model.object_prefix)
            .map(|value| value.trim_start_matches('/'))
        else {
            return Ok(None);
        };
        if relative.is_empty() || relative.ends_with('/') {
            return Ok(None);
        }
        if relative.split('/').any(str::is_empty) {
            anyhow::bail!("Unsafe S3 object path '{}': empty path segment", key);
        }

        let path = PathBuf::from(relative);
        let is_safe = !path.is_absolute()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_)));
        if !is_safe {
            anyhow::bail!("Unsafe S3 object path '{}'", key);
        }

        Ok(Some(path))
    }

    async fn download_object(
        store: Arc<dyn ObjectStore>,
        location: ObjectPath,
        destination_path: PathBuf,
    ) -> Result<()> {
        if let Some(parent) = destination_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
        }

        let result = store
            .get(&location)
            .await
            .with_context(|| format!("Failed to GET s3 object '{}'", location))?;
        let mut stream = result.into_stream();
        let mut file = tokio::fs::File::create(&destination_path)
            .await
            .with_context(|| format!("Failed to create '{}'", destination_path.display()))?;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .with_context(|| format!("Failed while streaming s3 object '{}'", location))?;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write '{}'", destination_path.display()))?;
        }
        file.flush()
            .await
            .with_context(|| format!("Failed to flush '{}'", destination_path.display()))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for S3Provider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        if !ignore_weights {
            anyhow::bail!(
                "S3 provider only supports metadata downloads with ignore_weights=true; use ModelStreamer for S3 model weights"
            );
        }

        let cache_dir = cache_dir
            .ok_or_else(|| anyhow::anyhow!("S3 download requires cache_dir to be provided"))?;
        tokio::fs::create_dir_all(&cache_dir)
            .await
            .with_context(|| {
                format!("Failed to create cache directory: {}", cache_dir.display())
            })?;

        let model = S3ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        let store = Self::build_store(&model.bucket)?;
        let prefix = ObjectPath::from(model.object_prefix.clone());
        let mut list_stream = store.list(Some(&prefix));
        let mut file_count = 0usize;

        while let Some(meta_result) = list_stream.next().await {
            let meta = meta_result.with_context(|| format!("Failed to list {}", model))?;
            let Some(relative_path) = Self::relative_path(&model, &meta.location)? else {
                continue;
            };
            if !Self::is_downloadable_path(&relative_path)
                || Self::is_weight_file(relative_path.to_string_lossy().as_ref())
            {
                continue;
            }

            let destination_path = model_dir.join(&relative_path);
            Self::download_object(Arc::clone(&store), meta.location, destination_path).await?;
            file_count = file_count.saturating_add(1);
        }

        if file_count == 0 {
            anyhow::bail!("No metadata files found in {}", model);
        }
        tokio::fs::write(model_dir.join(MODEL_MARKER_FILE_NAME), b"")
            .await
            .with_context(|| {
                format!(
                    "Failed to write S3 model marker in '{}'",
                    model_dir.display()
                )
            })?;

        info!(
            "Downloaded {} metadata files for model '{}'",
            file_count, model
        );
        Ok(model_dir)
    }

    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        let model = S3ModelName::parse(model_name)?;
        remove_s3_model_dir(&cache_dir, &model)
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let model = S3ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        if !model_dir.is_dir() || !has_marker_file(&model_dir.join(MODEL_MARKER_FILE_NAME))? {
            anyhow::bail!("S3 model '{model_name}' not found in cache");
        }
        Ok(model_dir)
    }

    async fn has_cached_files(
        &self,
        model_name: &str,
        cache_dir: PathBuf,
        revision: Option<&str>,
        ignore_weights: bool,
    ) -> Result<bool> {
        reject_unsupported_revision(self.provider_name(), revision)?;
        let model = S3ModelName::parse(model_name)?;
        if !ignore_weights {
            return Ok(false);
        }
        let model_dir = model.model_dir(&cache_dir);
        Ok(model_dir.is_dir() && has_marker_file(&model_dir.join(MODEL_MARKER_FILE_NAME))?)
    }

    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        Ok(S3ModelName::parse(model_name)?.to_string())
    }

    fn provider_name(&self) -> &'static str {
        "S3"
    }
}

pub(crate) struct S3ProviderCache;

impl ProviderCache for S3ProviderCache {
    fn clear_model(&self, cache_dir: &Path, model_name: &str) -> Result<()> {
        let model = S3ModelName::parse(model_name)?;
        remove_s3_model_dir(cache_dir, &model)
    }

    fn resolve_model_path(
        &self,
        cache_dir: &Path,
        model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        Ok(S3ModelName::parse(model_name)?.model_dir(cache_dir))
    }

    fn list_models(&self, cache_dir: &Path) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        let root = cache_dir.join(CACHE_ROOT_DIR_NAME);
        if !root.exists() {
            return Ok(models);
        }

        collect_cached_models(cache_dir, &root, &mut models)?;
        Ok(models)
    }
}

fn collect_cached_models(
    cache_dir: &Path,
    current_dir: &Path,
    models: &mut Vec<ModelInfo>,
) -> Result<()> {
    let entries = if current_dir == cache_dir.join(CACHE_ROOT_DIR_NAME) {
        fs::read_dir(current_dir)
            .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
    } else {
        let Some(entries) = crate::cache::read_dir_for_listing(current_dir) else {
            return Ok(());
        };
        entries
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_cached_models(cache_dir, &path, models)?;
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some(MODEL_MARKER_FILE_NAME) {
            continue;
        }

        let Some(model_dir) = path.parent() else {
            continue;
        };
        let relative = model_dir
            .strip_prefix(cache_dir.join(CACHE_ROOT_DIR_NAME))
            .with_context(|| format!("Failed to make '{}' relative", model_dir.display()))?;
        let mut components = relative.components();
        let Some(Component::Normal(bucket)) = components.next() else {
            continue;
        };
        let Some(bucket) = bucket.to_str() else {
            continue;
        };
        let object_prefix = components
            .filter_map(|component| match component {
                Component::Normal(value) => value.to_str(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/");
        if object_prefix.is_empty() {
            continue;
        }
        let name = format!("s3://{bucket}/{object_prefix}");
        if models.iter().any(|model| model.name == name) {
            continue;
        }
        models.push(ModelInfo::measured(
            ModelProvider::S3,
            name,
            model_dir.to_path_buf(),
            crate::cache::directory_size(model_dir),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;
    use tempfile::TempDir;

    #[test]
    fn test_model_name_parse_and_cache_dir() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            S3ModelName::parse("s3://bucket/org/model/rev/").expect("Expected model name parse");
        assert_eq!(model.to_string(), "s3://bucket/org/model/rev");
        assert_eq!(
            model.model_dir(temp_dir.path()),
            temp_dir.path().join("s3/bucket/org/model/rev")
        );
    }

    #[test]
    fn test_model_name_parse_normalizes_scheme() {
        let model =
            S3ModelName::parse(" S3://bucket/org/model/rev/").expect("Expected model name parse");
        assert_eq!(model.to_string(), "s3://bucket/org/model/rev");
    }

    #[test]
    fn test_model_name_parse_rejects_dot_buckets() {
        for invalid in ["s3://./org/model", "s3://../org/model"] {
            assert!(S3ModelName::parse(invalid).is_err(), "input={invalid}");
        }
    }

    #[test]
    fn test_clear_model_with_dot_bucket_leaves_other_caches() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let hf_model = temp_dir.path().join("models--gpt2/snapshots/abc");
        let s3_model = temp_dir.path().join("s3/bucket/org/model");
        fs::create_dir_all(&hf_model).expect("Failed to create HF model dir");
        fs::create_dir_all(&s3_model).expect("Failed to create S3 model dir");

        for name in ["s3://../models--gpt2", "s3://./bucket"] {
            S3ProviderCache
                .clear_model(temp_dir.path(), name)
                .expect_err("Expected dot bucket to be rejected");
        }
        assert!(hf_model.exists());
        assert!(s3_model.exists());
    }

    #[test]
    fn test_s3_builder_supports_default_credential_chain() {
        S3Provider::build_store("bucket").expect("Expected S3 store without static credentials");
    }

    #[test]
    fn test_get_model_path_requires_completion_marker() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model = S3ModelName::parse("s3://bucket/org/model").expect("Expected model name parse");
        let model_dir = model.model_dir(temp_dir.path());
        fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write model file");

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create runtime");
        runtime
            .block_on(
                S3Provider.get_model_path("s3://bucket/org/model", temp_dir.path().to_path_buf()),
            )
            .expect_err("Expected incomplete cache to be rejected");

        fs::write(model_dir.join(MODEL_MARKER_FILE_NAME), b"")
            .expect("Failed to write completion marker");
        assert_eq!(
            runtime
                .block_on(
                    S3Provider
                        .get_model_path("s3://bucket/org/model", temp_dir.path().to_path_buf(),)
                )
                .expect("Expected complete cache"),
            model_dir
        );
    }

    #[test]
    fn test_model_provider_value_enum_accepts_s3() {
        let parsed = ModelProvider::from_str("s3", false).expect("Expected s3 provider parse");
        assert_eq!(parsed, ModelProvider::S3);
    }

    #[test]
    fn test_s3_provider_rejects_full_weight_download() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let runtime = tokio::runtime::Runtime::new().expect("Failed to create runtime");
        let err = runtime
            .block_on(S3Provider.download_model(
                "s3://bucket/org/model",
                Some(temp_dir.path().to_path_buf()),
                false,
            ))
            .expect_err("Expected full S3 weight download rejection");
        assert!(err.to_string().contains("use ModelStreamer"));
    }

    #[tokio::test]
    async fn test_delete_ancestor_preserves_deep_cached_descendant_and_empties_elsewhere() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let root = temp_dir.path().join("s3/bucket/a");
        let deep = root.join("x/y");
        let sibling_empty = root.join("x/empty");
        fs::create_dir_all(&deep).expect("create deep dir");
        fs::create_dir_all(&sibling_empty).expect("create sibling dir");
        fs::write(root.join(MODEL_MARKER_FILE_NAME), b"").expect("write root marker");
        fs::write(root.join("config.json"), b"{}").expect("write root config");
        fs::write(root.join("x/loose.txt"), b"loose").expect("write loose file");
        fs::write(sibling_empty.join("junk.txt"), b"junk").expect("write junk file");
        fs::write(deep.join(MODEL_MARKER_FILE_NAME), b"").expect("write deep marker");
        fs::write(deep.join("config.json"), b"{\"deep\":true}").expect("write deep config");

        S3Provider
            .delete_model("s3://bucket/a", temp_dir.path().to_path_buf())
            .await
            .expect("delete ancestor should succeed");

        assert!(!root.join(MODEL_MARKER_FILE_NAME).exists());
        assert!(!root.join("config.json").exists());
        assert!(!root.join("x/loose.txt").exists());
        assert!(!sibling_empty.exists());
        assert!(deep.join(MODEL_MARKER_FILE_NAME).is_file());
        assert_eq!(
            fs::read(deep.join("config.json")).expect("read preserved descendant"),
            b"{\"deep\":true}"
        );
    }

    #[tokio::test]
    async fn test_delete_target_below_cached_ancestor_keeps_files_with_or_without_target_marker() {
        for (use_clear_model, target_has_marker) in
            [(false, true), (false, false), (true, true), (true, false)]
        {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let ancestor = temp_dir.path().join("s3/bucket/a");
            let target = ancestor.join("b");
            fs::create_dir_all(&target).expect("create target dir");
            fs::write(ancestor.join(MODEL_MARKER_FILE_NAME), b"").expect("write ancestor marker");
            fs::write(ancestor.join("root.json"), b"{}").expect("write ancestor file");
            fs::write(target.join("payload.json"), b"{\"keep\":true}")
                .expect("write target payload");
            if target_has_marker {
                fs::write(target.join(MODEL_MARKER_FILE_NAME), b"").expect("write target marker");
            }

            if use_clear_model {
                S3ProviderCache
                    .clear_model(temp_dir.path(), "s3://bucket/a/b")
                    .expect("clear_model should succeed");
            } else {
                S3Provider
                    .delete_model("s3://bucket/a/b", temp_dir.path().to_path_buf())
                    .await
                    .expect("delete_model should succeed");
            }

            assert!(
                ancestor.join(MODEL_MARKER_FILE_NAME).is_file(),
                "ancestor marker must remain (clear={use_clear_model}, marker={target_has_marker})"
            );
            assert!(
                ancestor.join("root.json").is_file(),
                "ancestor files must remain"
            );
            assert!(
                !target.join(MODEL_MARKER_FILE_NAME).exists(),
                "target marker must be removed when present"
            );
            assert_eq!(
                fs::read(target.join("payload.json")).expect("target payload must remain"),
                b"{\"keep\":true}",
                "payload below cached ancestor must stay (clear={use_clear_model}, marker={target_has_marker})"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_delete_s3_does_not_follow_symlinks() {
        let cache_dir = TempDir::new().expect("create cache dir");
        let outside_dir = TempDir::new().expect("create outside dir");
        let outside_file = outside_dir.path().join("keep.txt");
        fs::write(&outside_file, b"important").expect("write outside file");

        let model_dir = cache_dir.path().join("s3/bucket/org/model");
        fs::create_dir_all(&model_dir).expect("create model dir");
        fs::write(model_dir.join(MODEL_MARKER_FILE_NAME), b"").expect("write marker");
        std::os::unix::fs::symlink(outside_dir.path(), model_dir.join("escape"))
            .expect("create inner symlink");

        S3Provider
            .delete_model("s3://bucket/org/model", cache_dir.path().to_path_buf())
            .await
            .expect("delete model with symlink should succeed");
        assert!(!model_dir.exists());
        assert!(outside_file.exists(), "outside target must remain intact");

        let linked_parent = cache_dir.path().join("s3/bucket/org");
        fs::create_dir_all(&linked_parent).expect("create parent");
        let linked_root = linked_parent.join("linked");
        std::os::unix::fs::symlink(outside_dir.path(), &linked_root).expect("create root symlink");

        S3Provider
            .delete_model("s3://bucket/org/linked", cache_dir.path().to_path_buf())
            .await
            .expect("delete symlinked root should succeed");
        assert!(linked_root.symlink_metadata().is_err());
        assert!(outside_file.exists(), "outside target must remain intact");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_has_cached_files_after_s3_download_and_validation() {
        use crate::test_support::{EnvVarGuard, acquire_env_mutex};
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let env_lock = acquire_env_mutex();
        let server = MockServer::start().await;
        let cache_dir = TempDir::new().expect("create cache dir");

        let list_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>test-bucket</Name>
  <Prefix>org/model</Prefix>
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>org/model/config.json</Key>
    <LastModified>2026-01-01T00:00:00.000Z</LastModified>
    <ETag>"etag-1"</ETag>
    <Size>2</Size>
  </Contents>
  <Contents>
    <Key>org/model/model.safetensors</Key>
    <LastModified>2026-01-01T00:00:00.000Z</LastModified>
    <ETag>"etag-2"</ETag>
    <Size>7</Size>
  </Contents>
</ListBucketResult>"#;

        Mock::given(method("GET"))
            .and(path("/test-bucket"))
            .and(query_param("list-type", "2"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/xml")
                    .set_body_string(list_xml),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/test-bucket/org/model/config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"{}".to_vec()))
            .mount(&server)
            .await;

        let _endpoint = EnvVarGuard::set(&env_lock, "AWS_ENDPOINT_URL", &server.uri());
        let _access = EnvVarGuard::set(&env_lock, "AWS_ACCESS_KEY_ID", "test-access-key");
        let _secret = EnvVarGuard::set(&env_lock, "AWS_SECRET_ACCESS_KEY", "test-secret-key");
        let _region = EnvVarGuard::set(&env_lock, "AWS_REGION", "us-east-1");

        let model_name = "s3://test-bucket/org/model";
        assert!(
            !S3Provider
                .has_cached_files(model_name, cache_dir.path().to_path_buf(), None, true)
                .await
                .expect("empty check")
        );

        S3Provider
            .download_model(model_name, Some(cache_dir.path().to_path_buf()), true)
            .await
            .expect("S3 metadata-only download should succeed");

        assert!(
            S3Provider
                .has_cached_files(model_name, cache_dir.path().to_path_buf(), None, true)
                .await
                .expect("metadata-only check after download")
        );
        assert!(
            !S3Provider
                .has_cached_files(model_name, cache_dir.path().to_path_buf(), None, false)
                .await
                .expect("full-weight check must be false for metadata-only S3 cache")
        );
        assert!(
            S3Provider
                .has_cached_files(
                    model_name,
                    cache_dir.path().to_path_buf(),
                    Some("rev1"),
                    true
                )
                .await
                .is_err()
        );
        for invalid in [
            "s3://bucket/org/model\0",
            "s3://bucket/ /model",
            "s3://bucket/org//model",
            "s3://bucket/../escape",
        ] {
            assert!(
                S3ModelName::parse(invalid).is_err(),
                "expected rejection for '{invalid}'"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_s3_delete_rejects_symlinked_ancestor() {
        let cache = TempDir::new().expect("create cache dir");
        let outside = TempDir::new().expect("create outside dir");
        let outside_model = outside.path().join("m");
        fs::create_dir_all(&outside_model).expect("create outside model");
        fs::write(outside_model.join(MODEL_MARKER_FILE_NAME), b"").expect("write marker");
        fs::write(outside_model.join("keep.txt"), b"keep").expect("write outside file");

        let bucket_dir = cache.path().join(CACHE_ROOT_DIR_NAME).join("bkt");
        fs::create_dir_all(&bucket_dir).expect("create bucket dir");
        std::os::unix::fs::symlink(outside.path(), bucket_dir.join("link"))
            .expect("create ancestor symlink");

        let err = S3Provider
            .delete_model("s3://bkt/link/m", cache.path().to_path_buf())
            .await
            .expect_err("Expected symlinked ancestor to be rejected");
        assert!(
            err.to_string().contains("symlink"),
            "Unexpected error: {err}"
        );
        assert!(
            outside_model.join("keep.txt").is_file(),
            "Outside files must not be deleted through symlinked S3 ancestor"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_s3_delete_removes_marker_before_failing_subdir() {
        use std::os::unix::fs::PermissionsExt;

        let cache = TempDir::new().expect("create cache dir");
        let model_dir = cache.path().join(CACHE_ROOT_DIR_NAME).join("bkt/org/model");
        let locked_subdir = model_dir.join("subdir");
        fs::create_dir_all(&locked_subdir).expect("create locked subdir");
        let marker = model_dir.join(MODEL_MARKER_FILE_NAME);
        fs::write(&marker, b"").expect("write marker");
        fs::write(locked_subdir.join("weight.bin"), b"data").expect("write weight");

        // Make subdir read-only so unlinking weight.bin fails for non-root users.
        fs::set_permissions(&locked_subdir, fs::Permissions::from_mode(0o500))
            .expect("set read-only permissions");

        let result = S3Provider
            .delete_model("s3://bkt/org/model", cache.path().to_path_buf())
            .await;

        fs::set_permissions(&locked_subdir, fs::Permissions::from_mode(0o700))
            .expect("restore permissions");

        // When running as non-root, deletion fails on the read-only subdir,
        // and the root marker must already have been removed first.
        if result.is_err() {
            assert!(
                !marker.exists(),
                "Root .mx-model marker must be removed before walking child files"
            );
        }
    }
}
