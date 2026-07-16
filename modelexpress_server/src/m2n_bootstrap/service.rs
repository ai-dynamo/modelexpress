// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC service for immutable NCCL M2N bootstrap records.

use std::sync::Arc;

use modelexpress_common::grpc::m2n_bootstrap::{
    AbortM2nBootstrapRequest, AbortM2nBootstrapResponse, GetM2nBootstrapRequest,
    GetM2nBootstrapResponse, PublishM2nBootstrapRequest, PublishM2nBootstrapResponse,
    m2n_bootstrap_service_server::M2nBootstrapService,
};
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use super::backend::BootstrapError;
use super::state::M2nBootstrapStateManager;

pub struct M2nBootstrapServiceImpl {
    state: Arc<M2nBootstrapStateManager>,
}

impl M2nBootstrapServiceImpl {
    #[must_use]
    pub fn new(state: Arc<M2nBootstrapStateManager>) -> Self {
        Self { state }
    }
}

fn status_from_error(error: BootstrapError) -> Status {
    match error {
        BootstrapError::InvalidArgument(message) => Status::invalid_argument(message),
        BootstrapError::Conflict(message) => Status::already_exists(message),
        BootstrapError::Unsupported(message) => Status::unimplemented(message),
        BootstrapError::InvalidStoredRecord(message) => Status::data_loss(message),
        BootstrapError::Backend(message) => Status::unavailable(message),
    }
}

#[tonic::async_trait]
impl M2nBootstrapService for M2nBootstrapServiceImpl {
    async fn publish_bootstrap(
        &self,
        request: Request<PublishM2nBootstrapRequest>,
    ) -> Result<Response<PublishM2nBootstrapResponse>, Status> {
        let outcome = self
            .state
            .publish(request.into_inner(), chrono::Utc::now().timestamp_millis())
            .await
            .map_err(status_from_error)?;
        let action = if outcome.created {
            "published"
        } else {
            "reused"
        };
        info!(
            "M2N bootstrap {action}: attempt={} cohort={} world_size={}",
            outcome.record.key.attempt_id, outcome.record.key.cohort_id, outcome.record.world_size
        );
        Ok(Response::new(PublishM2nBootstrapResponse {
            success: true,
            message: format!("Bootstrap record {action}"),
            record: Some(outcome.record.to_proto()),
        }))
    }

    async fn get_bootstrap(
        &self,
        request: Request<GetM2nBootstrapRequest>,
    ) -> Result<Response<GetM2nBootstrapResponse>, Status> {
        let record = self
            .state
            .get(
                request.into_inner().key,
                chrono::Utc::now().timestamp_millis(),
            )
            .await
            .map_err(status_from_error)?;
        Ok(Response::new(GetM2nBootstrapResponse {
            found: record.is_some(),
            record,
        }))
    }

    async fn abort_bootstrap(
        &self,
        request: Request<AbortM2nBootstrapRequest>,
    ) -> Result<Response<AbortM2nBootstrapResponse>, Status> {
        let record = self
            .state
            .abort(request.into_inner(), chrono::Utc::now().timestamp_millis())
            .await
            .map_err(status_from_error)?;
        let key = record.key.as_ref();
        warn!(
            "M2N bootstrap aborted: attempt={} cohort={} reason={}",
            key.map_or("", |value| value.attempt_id.as_str()),
            key.map_or("", |value| value.cohort_id.as_str()),
            record.reason
        );
        Ok(Response::new(AbortM2nBootstrapResponse {
            success: true,
            message: "Bootstrap attempt aborted".to_string(),
            record: Some(record),
        }))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::m2n_bootstrap::backend::memory::InMemoryM2nBootstrapBackend;
    use modelexpress_common::grpc::m2n_bootstrap::{M2nBootstrapKey, M2nBootstrapState};

    fn service() -> M2nBootstrapServiceImpl {
        M2nBootstrapServiceImpl::new(Arc::new(M2nBootstrapStateManager::with_backend(Arc::new(
            InMemoryM2nBootstrapBackend::new(),
        ))))
    }

    fn key() -> M2nBootstrapKey {
        M2nBootstrapKey {
            job_id: "job".to_string(),
            attempt_id: "123e4567-e89b-42d3-a456-426614174000".to_string(),
            cohort_id: "cohort".to_string(),
        }
    }

    fn publish_request() -> PublishM2nBootstrapRequest {
        PublishM2nBootstrapRequest {
            key: Some(key()),
            nccl_unique_id: vec![7; 128],
            source_world_size: 1,
            destination_world_size: 1,
            world_size: 2,
            roster_digest: vec![1; 32],
            config_digest: vec![2; 32],
            publisher_participant_id: "source-0".to_string(),
            ttl_ms: 60_000,
        }
    }

    #[tokio::test]
    async fn publish_get_abort_roundtrip() {
        let service = service();
        let published = service
            .publish_bootstrap(Request::new(publish_request()))
            .await
            .expect("publish")
            .into_inner()
            .record
            .expect("record");
        assert_eq!(published.state, M2nBootstrapState::Published as i32);

        let fetched = service
            .get_bootstrap(Request::new(GetM2nBootstrapRequest { key: Some(key()) }))
            .await
            .expect("get")
            .into_inner();
        assert!(fetched.found);

        let aborted = service
            .abort_bootstrap(Request::new(AbortM2nBootstrapRequest {
                key: Some(key()),
                requested_by: "coordinator".to_string(),
                reason: "peer failed".to_string(),
            }))
            .await
            .expect("abort")
            .into_inner()
            .record
            .expect("record");
        assert_eq!(aborted.state, M2nBootstrapState::Aborted as i32);
        assert!(aborted.nccl_unique_id.is_empty());
    }

    #[tokio::test]
    async fn conflicting_publication_returns_already_exists() {
        let service = service();
        service
            .publish_bootstrap(Request::new(publish_request()))
            .await
            .expect("first");
        let mut conflicting = publish_request();
        conflicting.nccl_unique_id[0] = 8;
        let error = service
            .publish_bootstrap(Request::new(conflicting))
            .await
            .expect_err("conflict");
        assert_eq!(error.code(), tonic::Code::AlreadyExists);
    }
}
