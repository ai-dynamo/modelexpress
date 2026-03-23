// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P Metadata Service implementation for storing and retrieving NIXL/RDMA metadata.
//!
//! Metadata is keyed by mx_source_id, a 16-char hex hash of SourceIdentity.
//! Clients send the full SourceIdentity; the server computes and returns the hash.

use crate::metadata_backend::SourceInstanceInfo;
use crate::source_identity::{compute_mx_source_id, validate_identity};
use crate::state::P2pStateManager;
use modelexpress_common::grpc::p2p::{
    GetMetadataRequest, GetMetadataResponse, ListSourcesRequest, ListSourcesResponse,
    PublishMetadataRequest, PublishMetadataResponse, SourceInstanceRef, SourceStatus,
    UpdateStatusRequest, UpdateStatusResponse, WorkerMetadata, p2p_service_server::P2pService,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};

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

        let identity = match req.identity {
            Some(id) => id,
            None => {
                return Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: "identity is required".to_string(),
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }));
            }
        };

        if let Err(e) = validate_identity(&identity) {
            return Ok(Response::new(PublishMetadataResponse {
                success: false,
                message: e,
                mx_source_id: String::new(),
                worker_id: String::new(),
            }));
        }

        if req.worker_id.is_empty() {
            return Ok(Response::new(PublishMetadataResponse {
                success: false,
                message: "worker_id is required".to_string(),
                mx_source_id: String::new(),
                worker_id: String::new(),
            }));
        }

        let worker = match req.worker {
            Some(w) => w,
            None => {
                return Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: "worker is required".to_string(),
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }));
            }
        };

        let source_id = compute_mx_source_id(&identity);
        let worker_id = req.worker_id.clone();
        let model_name = identity.model_name.clone();
        let worker_rank = worker.worker_rank;
        let tensor_count = worker.tensors.len();

        match self
            .state
            .publish_metadata(&identity, &worker_id, vec![worker])
            .await
        {
            Ok(()) => {
                info!(
                    "PublishMetadata: model='{}' source_id={} worker_id={} worker_rank={} tensors={}",
                    model_name, source_id, worker_id, worker_rank, tensor_count
                );
                Ok(Response::new(PublishMetadataResponse {
                    success: true,
                    message: format!(
                        "Published metadata for '{}' (source_id={}, worker_id={}, worker_rank={}, {} tensors)",
                        model_name, source_id, worker_id, worker_rank, tensor_count
                    ),
                    mx_source_id: source_id,
                    worker_id,
                }))
            }
            Err(e) => {
                error!("Failed to publish metadata: {}", e);
                Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: format!("Failed to publish metadata: {e}"),
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }))
            }
        }
    }

    async fn list_sources(
        &self,
        request: Request<ListSourcesRequest>,
    ) -> Result<Response<ListSourcesResponse>, Status> {
        let req = request.into_inner();

        // Resolve optional source_id filter
        let source_id_filter: Option<String> = req.identity.as_ref().and_then(|id| {
            if id.model_name.is_empty() {
                None
            } else {
                Some(compute_mx_source_id(id))
            }
        });

        // Convert raw proto i32 to typed enum — None means no filter
        let status_filter = req
            .status_filter
            .and_then(|s| SourceStatus::try_from(s).ok());

        let workers: Vec<SourceInstanceInfo> = match self
            .state
            .list_workers(source_id_filter, status_filter)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to list workers: {}", e);
                return Ok(Response::new(ListSourcesResponse {
                    instances: Vec::new(),
                }));
            }
        };

        let refs: Vec<SourceInstanceRef> = workers
            .into_iter()
            .map(|info| SourceInstanceRef {
                mx_source_id: info.source_id,
                worker_id: info.worker_id,
                model_name: info.model_name,
                worker_rank: info.worker_rank,
            })
            .collect();

        debug!("ListSources: returning {} instances", refs.len());

        Ok(Response::new(ListSourcesResponse { instances: refs }))
    }

    async fn get_metadata(
        &self,
        request: Request<GetMetadataRequest>,
    ) -> Result<Response<GetMetadataResponse>, Status> {
        let req = request.into_inner();

        if req.mx_source_id.is_empty() || req.worker_id.is_empty() {
            return Ok(Response::new(GetMetadataResponse {
                found: false,
                worker: None,
                mx_source_id: String::new(),
                worker_id: String::new(),
            }));
        }

        match self
            .state
            .get_metadata(&req.mx_source_id, &req.worker_id)
            .await
        {
            Ok(Some(record)) => {
                // Each worker_id maps to exactly one worker record; take the first.
                let worker = record.workers.into_iter().next().map(WorkerMetadata::from);
                let found = worker.is_some();
                info!(
                    "GetMetadata '{}' (source_id={}, worker_id={}): {} tensors",
                    record.model_name,
                    req.mx_source_id,
                    req.worker_id,
                    worker.as_ref().map_or(0, |w| w.tensors.len()),
                );
                Ok(Response::new(GetMetadataResponse {
                    found,
                    worker,
                    mx_source_id: req.mx_source_id,
                    worker_id: req.worker_id,
                }))
            }
            Ok(None) => {
                info!(
                    "No metadata found for source_id={} worker_id={}",
                    req.mx_source_id, req.worker_id
                );
                Ok(Response::new(GetMetadataResponse {
                    found: false,
                    worker: None,
                    mx_source_id: req.mx_source_id,
                    worker_id: req.worker_id,
                }))
            }
            Err(e) => {
                error!("Failed to get metadata: {}", e);
                Ok(Response::new(GetMetadataResponse {
                    found: false,
                    worker: None,
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }))
            }
        }
    }

    async fn update_status(
        &self,
        request: Request<UpdateStatusRequest>,
    ) -> Result<Response<UpdateStatusResponse>, Status> {
        let req = request.into_inner();

        if req.mx_source_id.is_empty() {
            return Ok(Response::new(UpdateStatusResponse {
                success: false,
                message: "mx_source_id is required".to_string(),
            }));
        }

        if req.worker_id.is_empty() {
            return Ok(Response::new(UpdateStatusResponse {
                success: false,
                message: "worker_id is required".to_string(),
            }));
        }

        let status = match SourceStatus::try_from(req.status) {
            Ok(s) => s,
            Err(_) => {
                return Ok(Response::new(UpdateStatusResponse {
                    success: false,
                    message: format!("invalid status value: {}", req.status),
                }));
            }
        };

        match self
            .state
            .update_worker_status(&req.mx_source_id, &req.worker_id, req.worker_rank, status)
            .await
        {
            Ok(()) => Ok(Response::new(UpdateStatusResponse {
                success: true,
                message: format!(
                    "Updated status for source '{}' worker_id '{}' rank {}",
                    req.mx_source_id, req.worker_id, req.worker_rank
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
    use crate::metadata_backend::{
        BackendMetadataRecord, MockMetadataBackend, ModelMetadataRecord, WorkerRecord,
    };
    use crate::state::P2pStateManager;
    use modelexpress_common::grpc::p2p::{MxSourceType, SourceIdentity, SourceStatus};

    fn make_service(mock: MockMetadataBackend) -> P2pServiceImpl {
        P2pServiceImpl::new(Arc::new(P2pStateManager::with_backend(Arc::new(mock))))
    }

    fn test_identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: MxSourceType::Weights as i32,
            model_name: "my-model".to_string(),
            backend_framework: 1,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
        }
    }

    // ── publish_metadata ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_publish_metadata_missing_identity() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: None,
                worker: None,
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.mx_source_id.is_empty());
    }

    #[tokio::test]
    async fn test_publish_metadata_empty_model_name() {
        let svc = make_service(MockMetadataBackend::new());
        let mut id = test_identity();
        id.model_name = String::new();
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(id),
                worker: None,
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_publish_metadata_missing_worker_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: None,
                worker_id: String::new(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("worker_id"));
    }

    #[tokio::test]
    async fn test_publish_metadata_success() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _, _| Ok(()));

        let svc = make_service(mock);
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: Some(WorkerMetadata {
                    worker_rank: 0,
                    backend_metadata: Some(
                        modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata::NixlMetadata(vec![1, 2, 3]),
                    ),
                    tensors: vec![],
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                }),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.success);
        assert!(!resp.mx_source_id.is_empty());
        assert_eq!(resp.mx_source_id.len(), 16);
        assert_eq!(resp.worker_id, "worker-uuid-1");
    }

    #[tokio::test]
    async fn test_publish_metadata_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _, _| Err("storage unavailable".into()));

        let svc = make_service(mock);
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: Some(WorkerMetadata {
                    worker_rank: 0,
                    backend_metadata: None,
                    tensors: vec![],
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                }),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("storage unavailable"));
    }

    // ── get_metadata ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_metadata_empty_source_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: String::new(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
    }

    #[tokio::test]
    async fn test_get_metadata_found() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|source_id, worker_id| {
                Ok(Some(ModelMetadataRecord {
                    source_id: source_id.to_string(),
                    worker_id: worker_id.to_string(),
                    model_name: "my-model".to_string(),
                    workers: vec![WorkerRecord {
                        worker_rank: 0,
                        backend_metadata: BackendMetadataRecord::None,
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
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.found);
        assert!(resp.worker.is_some());
        assert_eq!(
            resp.worker.expect("worker should be present").status,
            SourceStatus::Ready as i32
        );
        assert_eq!(resp.mx_source_id, "abc123def456abcd");
        assert_eq!(resp.worker_id, "worker-uuid-1");
    }

    #[tokio::test]
    async fn test_get_metadata_not_found() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata().once().returning(|_, _| Ok(None));

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
        assert_eq!(resp.mx_source_id, "abc123def456abcd");
    }

    // ── update_status ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_update_status_invalid_status_value() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                status: 99,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("99"));
    }

    #[tokio::test]
    async fn test_update_status_empty_source_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: String::new(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_update_status_empty_worker_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: String::new(),
                worker_rank: 0,
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
            .returning(|_, _, _, _, _| Ok(()));

        let svc = make_service(mock);
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 3,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.success);
    }

    // ── publish_metadata (missing worker) ────────────────────────────────

    #[tokio::test]
    async fn test_publish_metadata_missing_worker() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: None,
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("worker is required"));
    }

    // ── list_sources ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_sources_returns_instances() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(|_, _| {
            Ok(vec![
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                },
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w2".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 1,
                },
            ])
        });

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(test_identity()),
                status_filter: Some(SourceStatus::Ready as i32),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert_eq!(resp.instances.len(), 2);
        assert_eq!(resp.instances[0].worker_id, "w1");
        assert_eq!(resp.instances[0].worker_rank, 0);
        assert_eq!(resp.instances[1].worker_id, "w2");
        assert_eq!(resp.instances[1].worker_rank, 1);
    }

    #[tokio::test]
    async fn test_list_sources_no_identity() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Ok(vec![]));

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: None,
                status_filter: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.instances.is_empty());
    }

    #[tokio::test]
    async fn test_list_sources_backend_error_returns_empty() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Err("backend down".into()));

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(test_identity()),
                status_filter: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.instances.is_empty());
    }

    #[tokio::test]
    async fn test_list_sources_empty_model_name_no_filter() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .withf(|source_id, _| source_id.is_none())
            .once()
            .returning(|_, _| Ok(vec![]));

        let svc = make_service(mock);
        let mut id = test_identity();
        id.model_name = String::new();
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(id),
                status_filter: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.instances.is_empty());
    }

    // ── get_metadata (additional) ───────────────────────────────────────────

    #[tokio::test]
    async fn test_get_metadata_empty_worker_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: String::new(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
    }

    #[tokio::test]
    async fn test_get_metadata_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|_, _| Err("storage error".into()));

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
        assert!(resp.mx_source_id.is_empty());
    }

    #[tokio::test]
    async fn test_get_metadata_record_with_empty_workers() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|source_id, worker_id| {
                Ok(Some(ModelMetadataRecord {
                    source_id: source_id.to_string(),
                    worker_id: worker_id.to_string(),
                    model_name: "my-model".to_string(),
                    workers: vec![],
                    published_at: 0,
                }))
            });

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
    }

    // ── update_status (additional) ──────────────────────────────────────────

    #[tokio::test]
    async fn test_update_status_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _, _| Err("write failed".into()));

        let svc = make_service(mock);
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("write failed"));
    }
}
