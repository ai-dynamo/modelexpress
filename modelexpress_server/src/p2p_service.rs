// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P Metadata Service implementation for storing and retrieving NIXL/RDMA metadata.
//!
//! The server stores model metadata (NIXL agent info + tensor descriptors) keyed by model name.
//! Clients query for existing sources and publish their own metadata.

use crate::state::P2pStateManager;
use modelexpress_common::grpc::p2p::{
    GetMetadataRequest, GetMetadataResponse, GetReadyRequest, GetReadyResponse,
    PublishMetadataRequest, PublishMetadataResponse, PublishReadyRequest, PublishReadyResponse,
    WorkerMetadata, p2p_service_server::P2pService,
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

    async fn publish_ready(
        &self,
        request: Request<PublishReadyRequest>,
    ) -> Result<Response<PublishReadyResponse>, Status> {
        let req = request.into_inner();

        if req.model_name.is_empty() {
            return Ok(Response::new(PublishReadyResponse {
                success: false,
                message: "model_name is required".to_string(),
            }));
        }

        if req.session_id.is_empty() {
            return Ok(Response::new(PublishReadyResponse {
                success: false,
                message: "session_id is required".to_string(),
            }));
        }

        match self
            .state
            .publish_ready(
                &req.model_name,
                req.worker_id,
                &req.session_id,
                &req.metadata_hash,
                req.nixl_ready,
                req.stability_verified,
            )
            .await
        {
            Ok(()) => Ok(Response::new(PublishReadyResponse {
                success: true,
                message: format!(
                    "Published ready flag for model '{}' worker {}",
                    req.model_name, req.worker_id
                ),
            })),
            Err(e) => {
                error!("Failed to publish ready flag: {}", e);
                Ok(Response::new(PublishReadyResponse {
                    success: false,
                    message: format!("Failed to publish ready flag: {e}"),
                }))
            }
        }
    }

    async fn get_ready(
        &self,
        request: Request<GetReadyRequest>,
    ) -> Result<Response<GetReadyResponse>, Status> {
        let req = request.into_inner();

        if req.model_name.is_empty() {
            return Ok(Response::new(GetReadyResponse {
                found: false,
                ready: false,
                session_id: String::new(),
                metadata_hash: String::new(),
                timestamp: 0.0,
            }));
        }

        match self.state.get_ready(&req.model_name, req.worker_id).await {
            Ok(Some(record)) => {
                let ready = record.nixl_ready && record.stability_verified;
                info!(
                    "Ready status for model '{}' worker {}: ready={}, session={}...",
                    req.model_name,
                    req.worker_id,
                    ready,
                    record.session_id.chars().take(8).collect::<String>()
                );

                Ok(Response::new(GetReadyResponse {
                    found: true,
                    ready,
                    session_id: record.session_id,
                    metadata_hash: record.metadata_hash,
                    timestamp: record.timestamp,
                }))
            }
            Ok(None) => {
                info!(
                    "No ready flag found for model '{}' worker {}",
                    req.model_name, req.worker_id
                );
                Ok(Response::new(GetReadyResponse {
                    found: false,
                    ready: false,
                    session_id: String::new(),
                    metadata_hash: String::new(),
                    timestamp: 0.0,
                }))
            }
            Err(e) => {
                error!("Failed to get ready flag: {}", e);
                Ok(Response::new(GetReadyResponse {
                    found: false,
                    ready: false,
                    session_id: String::new(),
                    metadata_hash: String::new(),
                    timestamp: 0.0,
                }))
            }
        }
    }
}
