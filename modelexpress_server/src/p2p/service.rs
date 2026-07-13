// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P Metadata Service implementation for storing and retrieving NIXL/RDMA metadata.
//!
//! Metadata is keyed by mx_source_id, a 16-char hex hash of SourceIdentity.
//! Clients send the full SourceIdentity; the server computes and returns the hash.

use crate::p2p::backend::SourceInstanceInfo;
use crate::p2p::load_tracker::SourceLoadTracker;
use crate::p2p::source_identity::{compute_mx_source_id, validate_identity};
use crate::p2p::state::P2pStateManager;
use modelexpress_common::envs;
use modelexpress_common::grpc::p2p::{
    GetMetadataRequest, GetMetadataResponse, ListSourcesRequest, ListSourcesResponse,
    PublishMetadataRequest, PublishMetadataResponse, SourceInstanceRef, SourceStatus,
    UpdateStatusRequest, UpdateStatusResponse, WorkerMetadata, p2p_service_server::P2pService,
};
use std::sync::Arc;
use std::time::Duration;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};

/// P2P Service implementation
pub struct P2pServiceImpl {
    state: Arc<P2pStateManager>,
    /// Transient, in-memory per-source load estimate surfaced in ListSources as
    /// `active_transfers` for the client `load_aware` selector.
    load_tracker: SourceLoadTracker,
}

impl P2pServiceImpl {
    /// Create a new P2P service, sizing the source-load window from
    /// `MX_P2P_SOURCE_LOAD_TTL_SECS` (default 60s).
    pub fn new(state: Arc<P2pStateManager>) -> Self {
        Self::with_load_ttl(state, Duration::from_secs(envs::p2p_source_load_ttl_secs()))
    }

    /// Create a new P2P service with an explicit source-load TTL (used by tests
    /// for deterministic load-tracking behavior).
    pub fn with_load_ttl(state: Arc<P2pStateManager>, load_ttl: Duration) -> Self {
        Self {
            state,
            load_tracker: SourceLoadTracker::new(load_ttl),
        }
    }
}

fn ready_source_is_fresh(
    info: &SourceInstanceInfo,
    now_ms: i64,
    heartbeat_timeout_ms: u64,
) -> bool {
    if info.updated_at <= 0 {
        return false;
    }

    let age_ms = now_ms.saturating_sub(info.updated_at).max(0) as u64;
    age_ms <= heartbeat_timeout_ms
}

fn worker_tensor_count(worker: &WorkerMetadata) -> usize {
    use modelexpress_common::grpc::p2p::worker_metadata::SourcePayload;

    match &worker.source_payload {
        Some(SourcePayload::TensorSource(tensor_source)) => tensor_source.tensors.len(),
        Some(SourcePayload::ArtifactSource(_)) => 0,
        _ => legacy_worker_tensor_count(worker),
    }
}

#[allow(deprecated)]
fn legacy_worker_tensor_count(worker: &WorkerMetadata) -> usize {
    worker.tensors.len()
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
        let tensor_count = worker_tensor_count(&worker);

        match self
            .state
            .publish_metadata(&identity, &worker_id, worker)
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

        let workers = if status_filter == Some(SourceStatus::Ready) {
            let now_ms = chrono::Utc::now().timestamp_millis();
            let heartbeat_timeout_ms =
                modelexpress_common::envs::heartbeat_timeout_secs().saturating_mul(1000);
            workers
                .into_iter()
                .filter(|info| ready_source_is_fresh(info, now_ms, heartbeat_timeout_ms))
                .collect()
        } else {
            workers
        };

        let refs: Vec<SourceInstanceRef> = workers
            .into_iter()
            .map(|info| SourceInstanceRef {
                // Read the load estimate before moving the id fields into the ref.
                active_transfers: self
                    .load_tracker
                    .active_count(&info.source_id, &info.worker_id),
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
                if found {
                    // A target that reached GetMetadata has selected this source
                    // and is about to pull from it; count it toward the source's
                    // load estimate for the load_aware selector.
                    self.load_tracker
                        .record_selection(&req.mx_source_id, &req.worker_id);
                }
                info!(
                    "GetMetadata '{}' (source_id={}, worker_id={}): {} tensors",
                    record.model_name,
                    req.mx_source_id,
                    req.worker_id,
                    worker.as_ref().map_or(0, worker_tensor_count),
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
    use crate::p2p::backend::{
        BackendMetadataRecord, MockMetadataBackend, ModelMetadataRecord, WorkerRecord,
    };
    use crate::p2p::state::P2pStateManager;
    use modelexpress_common::grpc::p2p::worker_metadata::SourcePayload;
    use modelexpress_common::grpc::p2p::{
        ArtifactSourceMetadata, MxSourceType, SourceIdentity, SourceStatus, TensorSourceMetadata,
    };

    fn make_service(mock: MockMetadataBackend) -> P2pServiceImpl {
        P2pServiceImpl::new(Arc::new(P2pStateManager::with_backend(Arc::new(mock))))
    }

    fn make_service_with_load_ttl(mock: MockMetadataBackend, ttl: Duration) -> P2pServiceImpl {
        P2pServiceImpl::with_load_ttl(Arc::new(P2pStateManager::with_backend(Arc::new(mock))), ttl)
    }

    fn empty_tensor_source() -> Option<SourcePayload> {
        Some(SourcePayload::TensorSource(TensorSourceMetadata {
            tensors: vec![],
        }))
    }

    fn test_identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.5.0".to_string(),
            mx_source_type: MxSourceType::Weights as i32,
            model_name: "my-model".to_string(),
            backend_framework: 1,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
            revision: String::new(),
            backend_framework_version: String::new(),
            torch_version: String::new(),
            cuda_version: String::new(),
            triton_version: String::new(),
            gpu_arch: String::new(),
            compile_config_digest: String::new(),
        }
    }

    fn test_artifact_identity() -> SourceIdentity {
        SourceIdentity {
            mx_source_type: MxSourceType::TorchCompileCache as i32,
            ..test_identity()
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
                    source_payload: empty_tensor_source(),
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                    ..Default::default()
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
                    source_payload: empty_tensor_source(),
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                    ..Default::default()
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
                        metadata_endpoint: String::new(),
                        agent_name: String::new(),
                        worker_grpc_endpoint: String::new(),
                        artifact_source: None,
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
        let now = chrono::Utc::now().timestamp_millis();
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(move |_, _| {
            Ok(vec![
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: now,
                },
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w2".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 1,
                    status: SourceStatus::Ready as i32,
                    updated_at: now,
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
    async fn test_list_sources_filters_artifact_sources_by_worker_status() {
        let now = chrono::Utc::now().timestamp_millis();
        let identity = test_artifact_identity();
        let expected_source_id = compute_mx_source_id(&identity);
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .withf(move |source_id, status_filter| {
                source_id.as_deref() == Some(expected_source_id.as_str())
                    && *status_filter == Some(SourceStatus::Ready)
            })
            .once()
            .returning(move |source_id, _| {
                Ok(vec![SourceInstanceInfo {
                    source_id: source_id.expect("source id"),
                    worker_id: "artifact-worker".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: now,
                }])
            });

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(identity),
                status_filter: Some(SourceStatus::Ready as i32),
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert_eq!(resp.instances.len(), 1);
        assert_eq!(resp.instances[0].worker_id, "artifact-worker");
    }

    #[tokio::test]
    async fn test_list_sources_ready_filter_excludes_expired_heartbeats() {
        let now = chrono::Utc::now().timestamp_millis();
        let expired_updated_at =
            now - ((modelexpress_common::envs::heartbeat_timeout_secs() + 1) * 1000) as i64;

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(move |_, _| {
            Ok(vec![
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "fresh-worker".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: now,
                },
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "expired-worker".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 1,
                    status: SourceStatus::Ready as i32,
                    updated_at: expired_updated_at,
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

        assert_eq!(resp.instances.len(), 1);
        assert_eq!(resp.instances[0].worker_id, "fresh-worker");
    }

    #[tokio::test]
    async fn test_get_metadata_preserves_artifact_source_status() {
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
                        metadata_endpoint: "10.0.0.1:5555".to_string(),
                        agent_name: "artifact-agent".to_string(),
                        worker_grpc_endpoint: "10.0.0.1:6555".to_string(),
                        artifact_source: Some(
                            ArtifactSourceMetadata {
                                artifact_id: "sha256:artifact".to_string(),
                                total_size: 1024,
                                file_count: 1,
                                chunk_count: 2,
                                node_rank: 0,
                            }
                            .into(),
                        ),
                    }],
                    published_at: 1234567890,
                }))
            });

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "artifact-source-id".to_string(),
                worker_id: "artifact-worker".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();

        let worker = resp.worker.expect("worker should be present");
        assert_eq!(worker.status, SourceStatus::Ready as i32);
        assert!(matches!(
            worker.source_payload,
            Some(SourcePayload::ArtifactSource(ref artifact))
                if artifact.artifact_id == "sha256:artifact"
        ));
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

    #[tokio::test]
    async fn test_list_sources_reports_active_transfers_after_selection() {
        let mut mock = MockMetadataBackend::new();
        // Every GetMetadata resolves to a ready worker for the requested source.
        mock.expect_get_metadata()
            .times(2)
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
                        metadata_endpoint: String::new(),
                        agent_name: String::new(),
                        worker_grpc_endpoint: String::new(),
                        artifact_source: None,
                    }],
                    published_at: 1234567890,
                }))
            });
        // Two ready workers of the same rank; only w1 gets selected below.
        mock.expect_list_workers().times(1).returning(|_, _| {
            Ok(vec![
                SourceInstanceInfo {
                    source_id: "srcaaaabbbbccccd".to_string(),
                    worker_id: "w1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                },
                SourceInstanceInfo {
                    source_id: "srcaaaabbbbccccd".to_string(),
                    worker_id: "w2".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                },
            ])
        });

        // Long TTL so both selections stay in-window for the test.
        let svc = make_service_with_load_ttl(mock, Duration::from_secs(300));

        // Select w1 twice via GetMetadata (the load signal).
        for _ in 0..2 {
            svc.get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "srcaaaabbbbccccd".to_string(),
                worker_id: "w1".to_string(),
            }))
            .await
            .expect("rpc");
        }

        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(test_identity()),
                status_filter: Some(SourceStatus::Ready as i32),
            }))
            .await
            .expect("rpc")
            .into_inner();

        let w1 = resp
            .instances
            .iter()
            .find(|r| r.worker_id == "w1")
            .expect("w1 present");
        let w2 = resp
            .instances
            .iter()
            .find(|r| r.worker_id == "w2")
            .expect("w2 present");
        assert_eq!(w1.active_transfers, 2, "selected source reflects its load");
        assert_eq!(w2.active_transfers, 0, "unselected source stays idle");
    }

    fn ready_record(source_id: &str, worker_id: &str) -> ModelMetadataRecord {
        ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name: "my-model".to_string(),
            workers: vec![WorkerRecord {
                worker_rank: 0,
                backend_metadata: BackendMetadataRecord::None,
                tensors: vec![],
                status: SourceStatus::Ready as i32,
                updated_at: 1234567890000,
                metadata_endpoint: String::new(),
                agent_name: String::new(),
                worker_grpc_endpoint: String::new(),
                artifact_source: None,
            }],
            published_at: 1234567890,
        }
    }

    fn single_instance_listing(worker_id: &'static str) -> Vec<SourceInstanceInfo> {
        vec![SourceInstanceInfo {
            source_id: "srcaaaabbbbccccd".to_string(),
            worker_id: worker_id.to_string(),
            model_name: "my-model".to_string(),
            worker_rank: 0,
            status: SourceStatus::Ready as i32,
            updated_at: 1234567890000,
        }]
    }

    async fn list_first_active_transfers(svc: &P2pServiceImpl) -> u32 {
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(test_identity()),
                status_filter: Some(SourceStatus::Ready as i32),
            }))
            .await
            .expect("rpc")
            .into_inner();
        resp.instances[0].active_transfers
    }

    // ── load signal: end-to-end robustness ─────────────────────────────────

    #[tokio::test]
    async fn test_load_signal_decays_after_ttl_end_to_end() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|s, w| Ok(Some(ready_record(s, w))));
        mock.expect_list_workers()
            .times(2)
            .returning(|_, _| Ok(single_instance_listing("w1")));

        let svc = make_service_with_load_ttl(mock, Duration::from_millis(50));
        svc.get_metadata(Request::new(GetMetadataRequest {
            mx_source_id: "srcaaaabbbbccccd".to_string(),
            worker_id: "w1".to_string(),
        }))
        .await
        .expect("rpc");

        assert_eq!(list_first_active_transfers(&svc).await, 1);
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(
            list_first_active_transfers(&svc).await,
            0,
            "selection must age out of the TTL window"
        );
    }

    #[tokio::test]
    async fn test_load_signal_resets_on_service_restart() {
        // Restart contract: a fresh service instance (new in-memory tracker)
        // reports zero load, which makes the client load_aware policy collapse
        // to rendezvous_hash until the window repopulates.
        let mut mock1 = MockMetadataBackend::new();
        mock1
            .expect_get_metadata()
            .times(3)
            .returning(|s, w| Ok(Some(ready_record(s, w))));
        mock1
            .expect_list_workers()
            .once()
            .returning(|_, _| Ok(single_instance_listing("w1")));
        let svc1 = make_service_with_load_ttl(mock1, Duration::from_secs(300));
        for _ in 0..3 {
            svc1.get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "srcaaaabbbbccccd".to_string(),
                worker_id: "w1".to_string(),
            }))
            .await
            .expect("rpc");
        }
        assert_eq!(list_first_active_transfers(&svc1).await, 3);

        let mut mock2 = MockMetadataBackend::new();
        mock2
            .expect_list_workers()
            .once()
            .returning(|_, _| Ok(single_instance_listing("w1")));
        let svc2 = make_service_with_load_ttl(mock2, Duration::from_secs(300));
        assert_eq!(
            list_first_active_transfers(&svc2).await,
            0,
            "restarted server must report zero load, not stale counts"
        );
    }

    #[tokio::test]
    async fn test_concurrent_selections_all_counted() {
        // 64 targets selecting the same source concurrently: every selection
        // lands in the window, no deadlock or lost update.
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .times(64)
            .returning(|s, w| Ok(Some(ready_record(s, w))));
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Ok(single_instance_listing("w1")));

        let svc = Arc::new(make_service_with_load_ttl(mock, Duration::from_secs(300)));

        let mut handles = Vec::new();
        for _ in 0..64 {
            let svc = Arc::clone(&svc);
            handles.push(tokio::spawn(async move {
                svc.get_metadata(Request::new(GetMetadataRequest {
                    mx_source_id: "srcaaaabbbbccccd".to_string(),
                    worker_id: "w1".to_string(),
                }))
                .await
                .expect("rpc")
            }));
        }
        for h in handles {
            h.await.expect("task");
        }
        assert_eq!(list_first_active_transfers(&svc).await, 64);
    }

    #[tokio::test]
    async fn test_failed_metadata_lookup_not_counted_as_load() {
        // A metadata miss means no transfer follows; it must not inflate the
        // source's load estimate.
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata().once().returning(|_, _| Ok(None));
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Ok(single_instance_listing("w1")));

        let svc = make_service_with_load_ttl(mock, Duration::from_secs(300));
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "srcaaaabbbbccccd".to_string(),
                worker_id: "w1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert_eq!(list_first_active_transfers(&svc).await, 0);
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
