// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend abstraction for RL refit control-plane metadata.

use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::refit::{
    CreateWeightVersionRequest, DeleteVersionLeaseRequest, DeleteWeightVersionShardRequest,
    RegisterVersionLeaseRequest, UpdateWeightVersionStateRequest, VersionLease, WeightVersion,
    WeightVersionShard, WorkerRegistration,
};

use crate::backend_config::BackendConfig;

pub mod instrumented;
pub mod redis;

pub type RefitResult<T> = Result<T, RefitBackendError>;

#[derive(Debug, thiserror::Error)]
pub enum RefitBackendError {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    FailedPrecondition(String),
    #[error("{0}")]
    AlreadyExists(String),
    #[error("{0}")]
    ResourceExhausted(String),
    #[error("{0}")]
    Internal(String),
    #[error("{0}")]
    Unavailable(String),
}

/// Atomic domain operations required by `RefitService`.
///
/// Each method is one backend transaction boundary. Redis implements these
/// operations with Lua; a Kubernetes implementation can use a version-scoped
/// coordination CR and `resourceVersion` compare-and-swap.
#[async_trait]
pub trait RefitBackend: Send + Sync {
    async fn register_worker(
        &self,
        worker: WorkerRegistration,
        ttl_seconds: u32,
    ) -> RefitResult<WorkerRegistration>;

    async fn create_weight_version(
        &self,
        request: &CreateWeightVersionRequest,
    ) -> RefitResult<WeightVersion>;

    async fn get_weight_version(&self, uid: &str) -> RefitResult<WeightVersion>;

    async fn delete_weight_version(&self, uid: &str) -> RefitResult<WeightVersion>;

    async fn update_weight_version_state(
        &self,
        request: &UpdateWeightVersionStateRequest,
    ) -> RefitResult<WeightVersion>;

    async fn create_weight_version_shard(
        &self,
        shard: WeightVersionShard,
    ) -> RefitResult<(WeightVersionShard, WeightVersion)>;

    async fn list_weight_version_shards(
        &self,
        version_id: &str,
    ) -> RefitResult<Vec<WeightVersionShard>>;

    async fn delete_weight_version_shard(
        &self,
        request: &DeleteWeightVersionShardRequest,
    ) -> RefitResult<bool>;

    async fn register_version_lease(
        &self,
        request: &RegisterVersionLeaseRequest,
    ) -> RefitResult<VersionLease>;

    async fn delete_version_lease(&self, request: &DeleteVersionLeaseRequest) -> RefitResult<bool>;
}

/// Construct the configured Refit backend.
///
/// Refit is not exposed for backends that do not yet implement its atomic
/// lifecycle contract.
pub async fn create_backend(config: &BackendConfig) -> RefitResult<Option<Arc<dyn RefitBackend>>> {
    match config {
        BackendConfig::Redis { url } => Ok(Some(Arc::new(
            redis::RedisRefitBackend::connect(url).await?,
        ))),
        BackendConfig::Kubernetes { .. } => Ok(None),
        #[cfg(feature = "memory-backend")]
        BackendConfig::Memory => Ok(None),
    }
}
