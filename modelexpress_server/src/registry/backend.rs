// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend abstraction for the distributed model registry.
//!
//! Parallels [`crate::p2p::backend`] in shape: trait, per-store submodules, factory.

use crate::backend_config::BackendConfig;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use std::sync::Arc;

pub mod kubernetes;
pub mod redis;

/// Result type for registry operations. Errors are boxed so backend-specific error types
/// (Redis, kube) can bubble up without the trait needing to know about them.
pub type RegistryResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Full model-lifecycle record returned by registry backends.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRecord {
    pub model_name: String,
    pub provider: ModelProvider,
    pub status: ModelStatus,
    pub created_at: DateTime<Utc>,
    pub last_used_at: DateTime<Utc>,
    pub message: Option<String>,
}

/// Result of a `try_claim_for_download` call. Distributed semantics across replicas:
/// only the caller that sees `Claimed` owns the download. Replicas that see
/// `AlreadyExists` must wait on the owner instead of spawning a duplicate download.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClaimOutcome {
    /// This call atomically created the registry record with status `DOWNLOADING`.
    /// The caller is the download owner.
    Claimed,
    /// The record already existed when we tried to claim. The caller is a waiter, not
    /// the owner; the enclosed status is the snapshot observed during the attempt.
    AlreadyExists(ModelStatus),
}

/// Trait for model-registry backend implementations.
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait RegistryBackend: Send + Sync {
    /// Initialize the connection. Idempotent; safe to call at startup.
    async fn connect(&self) -> RegistryResult<()>;

    /// Return the status of a model, or `None` if the model is unknown.
    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>>;

    /// Return the full record for a model, or `None` if unknown.
    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>>;

    /// Upsert the status, provider, message, and `last_used_at` for a model. Preserves
    /// `created_at` when the record already exists; stamps it to now otherwise.
    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()>;

    /// Bump `last_used_at` to now. No-op if the model is unknown.
    async fn touch_model(&self, model_name: &str) -> RegistryResult<()>;

    /// Delete the model record. No-op if the model is unknown.
    async fn delete_model(&self, model_name: &str) -> RegistryResult<()>;

    /// Return models ordered by `last_used_at` ascending (oldest first), truncated to
    /// `limit` if provided. Drives LRU cache eviction.
    async fn get_models_by_last_used(&self, limit: Option<u32>)
    -> RegistryResult<Vec<ModelRecord>>;

    /// Return (downloading, downloaded, error) counts. Used by the metrics path.
    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)>;

    /// Atomic claim: if the model has no registry record, create one with status
    /// `DOWNLOADING` and return `ClaimOutcome::Claimed`. Otherwise, return
    /// `ClaimOutcome::AlreadyExists(status)` without mutation.
    ///
    /// This is the only way multi-replica servers know which one actually owns the
    /// download. Callers MUST NOT infer ownership from the observed status alone —
    /// two replicas can both observe `DOWNLOADING` but only the one that receives
    /// `Claimed` owns it.
    async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<ClaimOutcome>;

    /// Atomic compare-and-set: if the current status is `ERROR`, flip it to
    /// `DOWNLOADING` with `Retrying download...` as the message and return `true`.
    /// Otherwise return `false` without mutation. Used by the error-retry path so
    /// only one replica spawns the retry even when multiple observe `ERROR`.
    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<bool>;
}

/// Construct a registry backend from config, eagerly connecting before returning.
pub async fn create_registry_backend(
    config: BackendConfig,
) -> RegistryResult<Arc<dyn RegistryBackend>> {
    match config {
        BackendConfig::Redis { url } => {
            let backend = redis::RedisRegistryBackend::new(&url);
            backend.connect().await?;
            Ok(Arc::new(backend) as Arc<dyn RegistryBackend>)
        }
        BackendConfig::Kubernetes { namespace } => {
            let backend = kubernetes::KubernetesRegistryBackend::new(&namespace).await?;
            backend.connect().await?;
            Ok(Arc::new(backend) as Arc<dyn RegistryBackend>)
        }
    }
}
