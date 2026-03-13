// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P Metadata Service implementation for storing and retrieving NIXL/RDMA metadata.
//!
//! The server stores model metadata (NIXL agent info + tensor descriptors) keyed by model name.
//! Clients query for existing sources and publish their own metadata.

use crate::state::P2pStateManager;
use modelexpress_common::grpc::p2p::{
    GetMetadataRequest, GetMetadataResponse, PublishMetadataRequest, PublishMetadataResponse,
    UpdateStatusRequest, UpdateStatusResponse, WorkerMetadata, p2p_service_server::P2pService,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info};

/// P2P Service implementation
pub struct P2pServiceImpl {
    state: Arc<P2pStateManager>,
}

impl P2pServiceImpl {
    /// Create a new P2P service
    pub fn new(state: Arc<P2pStateManager>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl P2pService for P2pServiceImpl {
    async fn publish_metadata(
        &self,
        request: Request<PublishMetadataRequest>,
    ) -> Result<Response<PublishMetadataResponse>, Status> {
        let req = request.into_inner();

        if req.model_name.is_empty() {
            return Ok(Response::new(PublishMetadataResponse {
                success: false,
                message: "model_name is required".to_string(),
            }));
        }

        let num_workers = req.workers.len();
        let total_tensors: usize = req.workers.iter().map(|w| w.tensors.len()).sum();

        match self
            .state
            .publish_metadata(&req.model_name, req.workers)
            .await
        {
            Ok(()) => Ok(Response::new(PublishMetadataResponse {
                success: true,
                message: format!(
                    "Published metadata for model '{}' ({} workers, {} tensors)",
                    req.model_name, num_workers, total_tensors
                ),
            })),
            Err(e) => {
                error!("Failed to publish metadata: {}", e);
                Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: format!("Failed to publish metadata: {e}"),
                }))
            }
        }
    }

    async fn get_metadata(
        &self,
        request: Request<GetMetadataRequest>,
    ) -> Result<Response<GetMetadataResponse>, Status> {
        let req = request.into_inner();

        if req.model_name.is_empty() {
            return Ok(Response::new(GetMetadataResponse {
                found: false,
                workers: Vec::new(),
            }));
        }

        match self.state.get_metadata(&req.model_name).await {
            Ok(Some(record)) => {
                let total_tensors: usize = record.workers.iter().map(|w| w.tensors.len()).sum();
                info!(
                    "Found metadata for model '{}': {} workers, {} tensors",
                    req.model_name,
                    record.workers.len(),
                    total_tensors
                );

                let workers: Vec<WorkerMetadata> = record
                    .workers
                    .into_iter()
                    .map(WorkerMetadata::from)
                    .collect();

                Ok(Response::new(GetMetadataResponse {
                    found: true,
                    workers,
                }))
            }
            Ok(None) => {
                info!("No metadata found for model '{}'", req.model_name);
                Ok(Response::new(GetMetadataResponse {
                    found: false,
                    workers: Vec::new(),
                }))
            }
            Err(e) => {
                error!("Failed to get metadata: {}", e);
                Ok(Response::new(GetMetadataResponse {
                    found: false,
                    workers: Vec::new(),
                }))
            }
        }
    }

    async fn update_status(
        &self,
        request: Request<UpdateStatusRequest>,
    ) -> Result<Response<UpdateStatusResponse>, Status> {
        let req = request.into_inner();

        if req.model_name.is_empty() {
            return Ok(Response::new(UpdateStatusResponse {
                success: false,
                message: "model_name is required".to_string(),
            }));
        }

        match self
            .state
            .update_worker_status(&req.model_name, req.worker_id, req.status)
            .await
        {
            Ok(()) => Ok(Response::new(UpdateStatusResponse {
                success: true,
                message: format!(
                    "Updated status for model '{}' worker {}",
                    req.model_name, req.worker_id
                ),
            })),
            Err(e) => {
                error!("Failed to update status: {}", e);
                Ok(Response::new(UpdateStatusResponse {
                    success: false,
                    message: format!("Failed to update status: {e}"),
                }))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::metadata_backend::{MockMetadataBackend, ModelMetadataRecord, WorkerRecord};
    use crate::state::P2pStateManager;
    use modelexpress_common::grpc::p2p::SourceStatus;

    fn make_service(mock: MockMetadataBackend) -> P2pServiceImpl {
        P2pServiceImpl::new(Arc::new(P2pStateManager::with_backend(Arc::new(mock))))
    }

    // ── publish_metadata ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_publish_metadata_empty_model_name() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                model_name: "".to_string(),
                workers: vec![],
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_publish_metadata_success() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _| Ok(()));

        let svc = make_service(mock);
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                model_name: "my-model".to_string(),
                workers: vec![WorkerMetadata {
                    worker_rank: 0,
                    nixl_metadata: vec![1, 2, 3],
                    tensors: vec![],
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                }],
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.success);
    }

    #[tokio::test]
    async fn test_publish_metadata_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _| Err("storage unavailable".into()));

        let svc = make_service(mock);
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                model_name: "my-model".to_string(),
                workers: vec![],
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("storage unavailable"));
    }

    // ── get_metadata ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_metadata_empty_model_name() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                model_name: "".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.workers.is_empty());
    }

    #[tokio::test]
    async fn test_get_metadata_found() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata().once().returning(|_| {
            Ok(Some(ModelMetadataRecord {
                model_name: "my-model".to_string(),
                workers: vec![WorkerRecord {
                    worker_rank: 0,
                    nixl_metadata: vec![],
                    tensors: vec![],
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                }],
                published_at: 1234567890,
            }))
        });

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                model_name: "my-model".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.found);
        assert_eq!(resp.workers.len(), 1);
        assert_eq!(resp.workers[0].worker_rank, 0);
        assert_eq!(resp.workers[0].status, SourceStatus::Ready as i32);
    }

    #[tokio::test]
    async fn test_get_metadata_not_found() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata().once().returning(|_| Ok(None));

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                model_name: "missing-model".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.workers.is_empty());
    }

    #[tokio::test]
    async fn test_get_metadata_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|_| Err("timeout".into()));

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                model_name: "my-model".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
    }

    // ── update_status ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_update_status_empty_model_name() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                model_name: "".to_string(),
                worker_id: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_update_status_success() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _| Ok(()));

        let svc = make_service(mock);
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                model_name: "my-model".to_string(),
                worker_id: 3,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.success);
    }

    #[tokio::test]
    async fn test_update_status_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _| Err("write failed".into()));

        let svc = make_service(mock);
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                model_name: "my-model".to_string(),
                worker_id: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("write failed"));
    }
}
