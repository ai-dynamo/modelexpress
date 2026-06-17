// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    cache::{ModelInfo, ProviderCache},
    models::ModelProvider,
    providers::ModelProviderTrait,
};
use anyhow::{Context, Result};
use futures::StreamExt;
use object_store::{ClientOptions, ObjectStore, aws::AmazonS3Builder, path::Path as ObjectPath};
use std::{
    fs,
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

impl S3ModelName {
    fn parse(model_name: &str) -> Result<Self> {
        let Some(full_url) = model_name.strip_prefix("s3://") else {
            anyhow::bail!("S3 model name must be a full s3://<bucket>/<path> URL");
        };
        let (bucket, object_prefix) = full_url
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("S3 model URL must include bucket and object path"))?;
        if bucket.is_empty() || bucket.contains('/') {
            anyhow::bail!("S3 model URL must include a valid bucket");
        }
        Self::new(bucket, object_prefix)
    }

    fn new(bucket: &str, object_prefix: &str) -> Result<Self> {
        let object_prefix = object_prefix.trim_end_matches('/');
        if object_prefix.is_empty() {
            anyhow::bail!("S3 model path must not be empty");
        }

        for component in object_prefix.split('/') {
            if component.is_empty() {
                anyhow::bail!("S3 model path must not contain empty path segments");
            }
            if component == "." || component == ".." {
                anyhow::bail!("S3 model path must not contain '.' or '..' segments");
            }
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

    fn build_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let access_key_id =
            std::env::var("AWS_ACCESS_KEY_ID").context("AWS_ACCESS_KEY_ID not set")?;
        let secret_access_key =
            std::env::var("AWS_SECRET_ACCESS_KEY").context("AWS_SECRET_ACCESS_KEY not set")?;
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

        let mut builder = AmazonS3Builder::new()
            .with_access_key_id(access_key_id)
            .with_secret_access_key(secret_access_key)
            .with_region(region)
            .with_bucket_name(bucket)
            .with_client_options(
                ClientOptions::new().with_timeout(Duration::from_secs(timeout_secs)),
            );

        if let Ok(token) = std::env::var("AWS_SESSION_TOKEN") {
            builder = builder.with_token(token);
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

        Ok(Arc::new(builder.build()?))
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
        let model_dir = model.model_dir(&cache_dir);
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)
                .with_context(|| format!("Failed to remove '{}'", model_dir.display()))?;
        }
        Ok(())
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let model = S3ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        if !model_dir.is_dir() {
            anyhow::bail!("S3 model '{model_name}' not found in cache");
        }
        Ok(model_dir)
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
        let model_dir = model.model_dir(cache_dir);
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)
                .with_context(|| format!("Failed to remove '{}'", model_dir.display()))?;
        }
        Ok(())
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
    for entry in fs::read_dir(current_dir)
        .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
    {
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
        models.push(ModelInfo {
            provider: ModelProvider::S3,
            name,
            size: crate::cache::directory_size(model_dir)?,
            path: model_dir.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
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
}
