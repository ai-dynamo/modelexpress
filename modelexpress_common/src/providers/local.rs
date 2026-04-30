// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stub provider for the `LOCAL` (peer-served arbitrary directory) variant.
//!
//! `LOCAL` is fundamentally a serving primitive, not a download/cache one:
//! the worker registers a logical id -> directory mapping at startup, and
//! the gRPC server resolves the request directly against that mapping.
//! These trait methods exist only because the project's existing match
//! arms over [`ModelProvider`] need an arm for every variant; in practice
//! the production server intercepts `LOCAL` before routing into provider
//! infrastructure, so these implementations are dead at runtime. They
//! return clear errors if a future caller forgets to intercept.

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};

use crate::cache::{ModelInfo, ProviderCache};
use crate::providers::ModelProviderTrait;

/// No-op provider for the LOCAL variant. See module docs.
pub struct LocalProvider;

#[async_trait::async_trait]
impl ModelProviderTrait for LocalProvider {
    async fn download_model(
        &self,
        _model_name: &str,
        _cache_path: Option<PathBuf>,
        _ignore_weights: bool,
    ) -> Result<PathBuf> {
        Err(anyhow!(
            "LOCAL provider has no download semantics; serve from a registered mount instead"
        ))
    }

    async fn delete_model(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<()> {
        Err(anyhow!("LOCAL provider has no cache; nothing to delete"))
    }

    async fn get_model_path(&self, _model_name: &str, _cache_dir: PathBuf) -> Result<PathBuf> {
        Err(anyhow!(
            "LOCAL provider has no cache layout; resolve the mount registry instead"
        ))
    }

    fn provider_name(&self) -> &'static str {
        "Local"
    }
}

/// No-op cache view for the LOCAL variant. See module docs.
pub(crate) struct LocalProviderCache;

impl ProviderCache for LocalProviderCache {
    fn clear_model(&self, _cache_root: &Path, _model_name: &str) -> Result<()> {
        Err(anyhow!("LOCAL provider has no cache; nothing to clear"))
    }

    fn resolve_model_path(
        &self,
        _cache_root: &Path,
        _model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        Err(anyhow!(
            "LOCAL provider has no cache layout; resolve the mount registry instead"
        ))
    }

    fn list_models(&self, _cache_root: &Path) -> Result<Vec<ModelInfo>> {
        // No cached models for the LOCAL variant; serving paths live in
        // the mount registry, which is owned by the gRPC server.
        Ok(Vec::new())
    }
}
