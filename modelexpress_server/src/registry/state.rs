// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lazy-connect wrapper around `RegistryBackend`, parallel to [`crate::p2p::state`].

use crate::backend_config::BackendConfig;
use crate::registry::backend::{
    ModelRecord, RegistryBackend, RegistryResult, create_registry_backend,
};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Clone)]
pub struct RegistryManager {
    backend: Arc<RwLock<Option<Arc<dyn RegistryBackend>>>>,
    config: Option<BackendConfig>,
}

impl Default for RegistryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RegistryManager {
    pub fn new() -> Self {
        Self {
            backend: Arc::new(RwLock::new(None)),
            config: BackendConfig::from_env().ok(),
        }
    }

    pub fn with_config(config: BackendConfig) -> Self {
        Self {
            backend: Arc::new(RwLock::new(None)),
            config: Some(config),
        }
    }

    /// Inject a pre-built backend directly (tests only).
    #[cfg(test)]
    pub fn with_backend(backend: Arc<dyn RegistryBackend>) -> Self {
        Self {
            backend: Arc::new(RwLock::new(Some(backend))),
            config: None,
        }
    }

    /// Eagerly connect to the configured backend. Returns the backend type name.
    pub async fn connect(&self) -> RegistryResult<String> {
        let config = self.config.clone().ok_or(
            "MX_METADATA_BACKEND is not set or invalid. Set it to 'redis' or 'kubernetes'.",
        )?;
        let backend_name = config.to_string();
        let backend = create_registry_backend(config).await?;
        let mut guard = self.backend.write().await;
        *guard = Some(backend);
        info!("RegistryManager connected (backend: {})", backend_name);
        Ok(backend_name)
    }

    async fn get_backend(&self) -> RegistryResult<Arc<dyn RegistryBackend>> {
        {
            let guard = self.backend.read().await;
            if let Some(backend) = guard.as_ref() {
                return Ok(backend.clone());
            }
        }
        let mut guard = self.backend.write().await;
        if let Some(backend) = guard.as_ref() {
            return Ok(backend.clone());
        }
        let config = self.config.clone().ok_or(
            "MX_METADATA_BACKEND is not set or invalid. Set it to 'redis' or 'kubernetes'.",
        )?;
        let backend = create_registry_backend(config.clone()).await?;
        info!("RegistryManager lazily connected ({:?})", config);
        *guard = Some(backend.clone());
        Ok(backend)
    }

    pub async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        self.get_backend().await?.get_status(model_name).await
    }

    pub async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        self.get_backend().await?.get_model_record(model_name).await
    }

    pub async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        self.get_backend()
            .await?
            .set_status(model_name, provider, status, message)
            .await
    }

    pub async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        self.get_backend().await?.touch_model(model_name).await
    }

    pub async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        self.get_backend().await?.delete_model(model_name).await
    }

    pub async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        self.get_backend()
            .await?
            .get_models_by_last_used(limit)
            .await
    }

    pub async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        self.get_backend().await?.get_status_counts().await
    }

    pub async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<ModelStatus> {
        self.get_backend()
            .await?
            .try_claim_for_download(model_name, provider)
            .await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::registry::backend::MockRegistryBackend;
    use mockall::predicate::eq;

    #[tokio::test]
    async fn connect_fails_when_no_config() {
        let mgr = RegistryManager {
            backend: Arc::new(RwLock::new(None)),
            config: None,
        };
        assert!(mgr.connect().await.is_err());
    }

    #[tokio::test]
    async fn try_claim_delegates_to_backend() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_try_claim_for_download()
            .with(eq("m"), eq(ModelProvider::HuggingFace))
            .once()
            .returning(|_, _| Ok(ModelStatus::DOWNLOADING));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        let status = mgr
            .try_claim_for_download("m", ModelProvider::HuggingFace)
            .await
            .expect("claim");
        assert_eq!(status, ModelStatus::DOWNLOADING);
    }

    #[tokio::test]
    async fn set_status_propagates_errors() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_set_status()
            .once()
            .returning(|_, _, _, _| Err("backend down".into()));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        assert!(
            mgr.set_status("m", ModelProvider::HuggingFace, ModelStatus::ERROR, None)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn get_models_by_last_used_passes_limit() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .with(eq(Some(3_u32)))
            .once()
            .returning(|_| Ok(Vec::new()));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        let v = mgr.get_models_by_last_used(Some(3)).await.expect("call");
        assert!(v.is_empty());
    }

    #[tokio::test]
    async fn get_status_delegates_to_backend() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_status()
            .with(eq("m"))
            .once()
            .returning(|_| Ok(Some(ModelStatus::DOWNLOADED)));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        assert_eq!(
            mgr.get_status("m").await.expect("get_status"),
            Some(ModelStatus::DOWNLOADED)
        );
    }

    #[tokio::test]
    async fn get_model_record_delegates_to_backend() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_model_record()
            .with(eq("m"))
            .once()
            .returning(|_| Ok(None));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        assert!(
            mgr.get_model_record("m")
                .await
                .expect("get_model_record")
                .is_none()
        );
    }

    #[tokio::test]
    async fn touch_model_delegates_to_backend() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_touch_model()
            .with(eq("m"))
            .once()
            .returning(|_| Ok(()));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        mgr.touch_model("m").await.expect("touch");
    }

    #[tokio::test]
    async fn delete_model_delegates_to_backend() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_delete_model()
            .with(eq("m"))
            .once()
            .returning(|_| Ok(()));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        mgr.delete_model("m").await.expect("delete");
    }

    #[tokio::test]
    async fn get_status_counts_delegates_to_backend() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_status_counts()
            .once()
            .returning(|| Ok((2, 3, 1)));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        assert_eq!(mgr.get_status_counts().await.expect("counts"), (2, 3, 1));
    }

    #[tokio::test]
    async fn set_status_forwards_all_args() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_set_status()
            .with(
                eq("m"),
                eq(ModelProvider::HuggingFace),
                eq(ModelStatus::DOWNLOADED),
                eq(Some("done".to_string())),
            )
            .once()
            .returning(|_, _, _, _| Ok(()));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        mgr.set_status(
            "m",
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
            Some("done".to_string()),
        )
        .await
        .expect("set_status");
    }

    #[tokio::test]
    async fn get_backend_caches_first_connection() {
        // With a pre-injected backend, repeated calls to any pass-through method must hit
        // the same backend instance (the double-checked RwLock path returns the cached
        // Arc rather than re-invoking create_registry_backend).
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_status().times(2).returning(|_| Ok(None));
        let mgr = RegistryManager::with_backend(Arc::new(mock));
        assert!(mgr.get_status("a").await.expect("first").is_none());
        assert!(mgr.get_status("b").await.expect("second").is_none());
    }
}
