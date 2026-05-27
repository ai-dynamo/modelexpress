// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer lease RPC request handling.

use crate::p2p::state::{BeginTransferLeaseParams, P2pStateManager};
use modelexpress_common::grpc::p2p::{
    BeginTransferLeaseRequest, CompleteTransferLeaseRequest, GetTransferLeaseRequest,
    GetTransferLeaseResponse, ListTransferLeasesRequest, ListTransferLeasesResponse,
    RenewTransferLeaseRequest, TransferLeaseResponse, TransferLeaseStatus,
};
use tonic::{Request, Response, Status};
use tracing::error;

pub(super) async fn begin_transfer_lease(
    state: &P2pStateManager,
    request: Request<BeginTransferLeaseRequest>,
) -> Result<Response<TransferLeaseResponse>, Status> {
    let req = request.into_inner();
    if req.mx_source_id.is_empty()
        || req.source_worker_id.is_empty()
        || req.target_worker_id.is_empty()
    {
        return Ok(Response::new(TransferLeaseResponse {
            success: false,
            message: "mx_source_id, source_worker_id, and target_worker_id are required"
                .to_string(),
            lease: None,
        }));
    }

    let requested_lease_id = if req.lease_id.is_empty() {
        None
    } else {
        Some(req.lease_id)
    };
    let lease_params = BeginTransferLeaseParams {
        lease_id: requested_lease_id,
        mx_source_id: req.mx_source_id,
        source_worker_id: req.source_worker_id,
        target_worker_id: req.target_worker_id,
        target_worker_rank: req.target_worker_rank,
        model_version: req.model_version,
        ttl_millis: req.ttl_millis,
        metadata: req.metadata,
    };
    match state.begin_transfer_lease(lease_params).await {
        Ok(lease) => Ok(Response::new(TransferLeaseResponse {
            success: true,
            message: "transfer lease started".to_string(),
            lease: Some(lease.into()),
        })),
        Err(e) => {
            error!("Failed to begin transfer lease: {}", e);
            Ok(Response::new(TransferLeaseResponse {
                success: false,
                message: format!("Failed to begin transfer lease: {e}"),
                lease: None,
            }))
        }
    }
}

pub(super) async fn renew_transfer_lease(
    state: &P2pStateManager,
    request: Request<RenewTransferLeaseRequest>,
) -> Result<Response<TransferLeaseResponse>, Status> {
    let req = request.into_inner();
    if req.lease_id.is_empty() {
        return Ok(Response::new(TransferLeaseResponse {
            success: false,
            message: "lease_id is required".to_string(),
            lease: None,
        }));
    }

    match state
        .renew_transfer_lease(&req.lease_id, req.ttl_millis)
        .await
    {
        Ok(lease) => Ok(Response::new(TransferLeaseResponse {
            success: true,
            message: "transfer lease renewed".to_string(),
            lease: Some(lease.into()),
        })),
        Err(e) => Ok(Response::new(TransferLeaseResponse {
            success: false,
            message: format!("Failed to renew transfer lease: {e}"),
            lease: None,
        })),
    }
}

pub(super) async fn complete_transfer_lease(
    state: &P2pStateManager,
    request: Request<CompleteTransferLeaseRequest>,
) -> Result<Response<TransferLeaseResponse>, Status> {
    let req = request.into_inner();
    if req.lease_id.is_empty() {
        return Ok(Response::new(TransferLeaseResponse {
            success: false,
            message: "lease_id is required".to_string(),
            lease: None,
        }));
    }
    let status = match TransferLeaseStatus::try_from(req.status) {
        Ok(
            status @ (TransferLeaseStatus::Completed
            | TransferLeaseStatus::Failed
            | TransferLeaseStatus::Expired),
        ) => status,
        _ => {
            return Ok(Response::new(TransferLeaseResponse {
                success: false,
                message: "status must be COMPLETED, FAILED, or EXPIRED".to_string(),
                lease: None,
            }));
        }
    };

    match state
        .complete_transfer_lease(&req.lease_id, status, &req.error_message)
        .await
    {
        Ok(lease) => Ok(Response::new(TransferLeaseResponse {
            success: true,
            message: "transfer lease completed".to_string(),
            lease: Some(lease.into()),
        })),
        Err(e) => Ok(Response::new(TransferLeaseResponse {
            success: false,
            message: format!("Failed to complete transfer lease: {e}"),
            lease: None,
        })),
    }
}

pub(super) async fn get_transfer_lease(
    state: &P2pStateManager,
    request: Request<GetTransferLeaseRequest>,
) -> Result<Response<GetTransferLeaseResponse>, Status> {
    let req = request.into_inner();
    if req.lease_id.is_empty() {
        return Ok(Response::new(GetTransferLeaseResponse {
            found: false,
            lease: None,
        }));
    }

    match state.get_transfer_lease(&req.lease_id).await {
        Ok(Some(lease)) => Ok(Response::new(GetTransferLeaseResponse {
            found: true,
            lease: Some(lease.into()),
        })),
        Ok(None) => Ok(Response::new(GetTransferLeaseResponse {
            found: false,
            lease: None,
        })),
        Err(e) => {
            error!("Failed to get transfer lease: {}", e);
            Ok(Response::new(GetTransferLeaseResponse {
                found: false,
                lease: None,
            }))
        }
    }
}

pub(super) async fn list_transfer_leases(
    state: &P2pStateManager,
    request: Request<ListTransferLeasesRequest>,
) -> Result<Response<ListTransferLeasesResponse>, Status> {
    let req = request.into_inner();
    let status_filter = req
        .status_filter
        .and_then(|status| TransferLeaseStatus::try_from(status).ok());
    let mx_source_id = if req.mx_source_id.is_empty() {
        None
    } else {
        Some(req.mx_source_id)
    };
    let target_worker_id = if req.target_worker_id.is_empty() {
        None
    } else {
        Some(req.target_worker_id)
    };
    let source_worker_id = if req.source_worker_id.is_empty() {
        None
    } else {
        Some(req.source_worker_id)
    };
    let model_version_filter = req.model_version_filter;

    match state
        .list_transfer_leases(
            mx_source_id,
            target_worker_id,
            status_filter,
            model_version_filter,
            source_worker_id,
        )
        .await
    {
        Ok(leases) => Ok(Response::new(ListTransferLeasesResponse {
            leases: leases.into_iter().map(Into::into).collect(),
        })),
        Err(e) => {
            error!("Failed to list transfer leases: {}", e);
            Ok(Response::new(ListTransferLeasesResponse {
                leases: Vec::new(),
            }))
        }
    }
}
