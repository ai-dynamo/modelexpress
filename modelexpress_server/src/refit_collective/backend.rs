// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend abstraction for the NCCL M2N collective control plane.

use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::refit_collective::{
    CollectiveGroup, CollectiveGroupMembership, CollectiveTransfer,
    CreateCollectiveTransferRequest, JoinCollectiveGroupRequest, PublishGroupBootstrapRequest,
    ReportCollectiveTransferRequest,
};

use crate::backend_config::BackendConfig;

pub mod redis;

pub type CollectiveResult<T> = Result<T, CollectiveBackendError>;

#[derive(Debug, thiserror::Error)]
pub enum CollectiveBackendError {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    FailedPrecondition(String),
    #[error("{0}")]
    AlreadyExists(String),
    #[error("{0}")]
    Internal(String),
    #[error("{0}")]
    Unavailable(String),
}

/// Atomic domain operations required by `RefitCollectiveService`.
///
/// Each method is one backend transaction boundary. The atomicity is not
/// incidental: admission, epoch bumps and readiness are evaluated together, so
/// that a group can never be observed as READY with a membership that has
/// already changed underneath it. A rank that acted on such an observation
/// would enter a collective its peers are not in, and block until the deadline.
#[async_trait]
pub trait CollectiveBackend: Send + Sync {
    /// Admit one worker to the group implied by its declared membership,
    /// creating the group on first contact.
    ///
    /// Requires a live matching `WorkerRegistration`. Bumps the group's epoch
    /// when an admitted registration expires, when a worker presents a new
    /// generation for a slot already held, or when the reported plan digest
    /// differs from the group's. Every one of those invalidates the cached
    /// communicator, the cached plan, or both, and the epoch tells a client to
    /// drop them. First admission while a group is still forming does not bump.
    async fn join_group(
        &self,
        request: &JoinCollectiveGroupRequest,
    ) -> CollectiveResult<CollectiveGroupMembership>;

    async fn get_group(&self, group_id: &str) -> CollectiveResult<CollectiveGroup>;

    /// Record one lane's `ncclUniqueId`, stamped with the epoch it was
    /// generated for. A stamp that is not the group's current epoch is
    /// rejected rather than stored: it describes a communicator whose world
    /// size no longer matches the membership.
    async fn publish_bootstrap(
        &self,
        request: &PublishGroupBootstrapRequest,
    ) -> CollectiveResult<CollectiveGroup>;

    /// Idempotent on `idempotency_key`, so an orchestrator retry after a
    /// timeout returns the original operation instead of opening a second one
    /// against the same group.
    async fn create_transfer(
        &self,
        request: &CreateCollectiveTransferRequest,
    ) -> CollectiveResult<CollectiveTransfer>;

    async fn get_transfer(&self, operation_id: &str) -> CollectiveResult<CollectiveTransfer>;

    async fn delete_transfer(&self, operation_id: &str) -> CollectiveResult<CollectiveTransfer>;

    /// Record one participant's terminal result, fenced on the operation, the
    /// group epoch and the reporting worker's admitted generation. A report
    /// from a restarted worker, or against a superseded epoch, is rejected
    /// rather than allowed to complete an operation it is no longer part of.
    async fn report_transfer(
        &self,
        request: &ReportCollectiveTransferRequest,
    ) -> CollectiveResult<CollectiveTransfer>;
}

/// Construct the configured collective backend.
///
/// Mirrors the refit control plane: the path is exposed only for backends that
/// implement its atomic contract, and is absent rather than degraded elsewhere.
pub async fn create_backend(
    config: &BackendConfig,
) -> CollectiveResult<Option<Arc<dyn CollectiveBackend>>> {
    match config {
        BackendConfig::Redis { url } => Ok(Some(Arc::new(
            redis::RedisCollectiveBackend::connect(url).await?,
        ))),
        BackendConfig::Kubernetes { .. } => Ok(None),
        #[cfg(feature = "memory-backend")]
        BackendConfig::Memory => Ok(None),
    }
}
