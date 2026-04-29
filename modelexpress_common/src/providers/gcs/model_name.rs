// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::path::{Path, PathBuf};

pub const CACHE_ROOT_DIR_NAME: &str = "gcs";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelName {
    pub bucket: BucketName,
    pub object_prefix: String,
}

impl ModelName {
    pub fn parse(model_name: &str) -> Result<Self> {
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

    pub fn new(bucket: BucketName, object_prefix: &str) -> Result<Self> {
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

    pub fn model_dir(&self, cache_dir: &Path) -> PathBuf {
        let mut path = self.bucket_dir(cache_dir);
        for component in self.object_prefix.split('/') {
            path = path.join(component);
        }
        path
    }

    pub fn bucket_dir(&self, cache_dir: &Path) -> PathBuf {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketName {
    normalized: String,
}

impl BucketName {
    pub fn parse(raw: &str) -> Result<Self> {
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

    pub fn resource_name(&self) -> String {
        format!("projects/_/buckets/{}", self.as_str())
    }
}

impl std::fmt::Display for BucketName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
