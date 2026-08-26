// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral gRPC service for RL refit control-plane metadata.

#![allow(clippy::result_large_err)] // tonic service helpers return tonic::Status.

use std::collections::HashSet;
use std::sync::Arc;

use modelexpress_common::grpc::refit::{
    CreateWeightVersionRequest, CreateWeightVersionShardRequest, CreateWeightVersionShardResponse,
    DeleteVersionLeaseRequest, DeleteVersionLeaseResponse, DeleteWeightVersionRequest,
    DeleteWeightVersionShardRequest, DeleteWeightVersionShardResponse, GetWeightVersionRequest,
    ListWeightVersionShardsRequest, ListWeightVersionShardsResponse, RegisterVersionLeaseRequest,
    RegisterWorkerRequest, VersionLease, WeightPayloadFormat, WeightVersion, WeightVersionState,
    WorkerRegistration, WorkerRole, refit_service_server::RefitService,
};
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use super::backend::{RefitBackend, RefitBackendError};

fn required(value: &str, field: &str) -> Result<(), Status> {
    if value.trim().is_empty() {
        Err(Status::invalid_argument(format!("{field} is required")))
    } else {
        Ok(())
    }
}

fn validate_ttl(ttl_seconds: u32) -> Result<(), Status> {
    if ttl_seconds == 0 {
        Err(Status::invalid_argument(
            "ttl_seconds must be greater than zero",
        ))
    } else {
        Ok(())
    }
}

/// Converts a backend error into a `Status`, logging it on the way out.
///
/// Every backend failure in this service funnels through here, so severity is
/// assigned once: operator-actionable faults are logged loudly, caller faults at
/// debug so a misbehaving client cannot flood the server log.
fn backend_status(error: RefitBackendError) -> Status {
    match error {
        RefitBackendError::InvalidArgument(message) => {
            debug!("Refit backend rejected request: {message}");
            Status::invalid_argument(message)
        }
        RefitBackendError::NotFound(message) => {
            debug!("Refit backend reported not found: {message}");
            Status::not_found(message)
        }
        RefitBackendError::FailedPrecondition(message) => {
            debug!("Refit backend precondition failed: {message}");
            Status::failed_precondition(message)
        }
        RefitBackendError::AlreadyExists(message) => {
            debug!("Refit backend reported conflict: {message}");
            Status::already_exists(message)
        }
        RefitBackendError::ResourceExhausted(message) => {
            warn!("Refit backend exhausted: {message}");
            Status::resource_exhausted(message)
        }
        RefitBackendError::Internal(message) => {
            error!("Refit backend internal error: {message}");
            Status::internal(message)
        }
        RefitBackendError::Unavailable(message) => {
            error!("Refit metadata backend unavailable: {message}");
            Status::unavailable(format!("Refit metadata backend error: {message}"))
        }
    }
}

#[derive(Clone)]
pub struct RefitServiceImpl {
    backend: Arc<dyn RefitBackend>,
}

impl RefitServiceImpl {
    pub fn new(backend: Arc<dyn RefitBackend>) -> Self {
        Self { backend }
    }
}

#[tonic::async_trait]
impl RefitService for RefitServiceImpl {
    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<WorkerRegistration>, Status> {
        let request = request.into_inner();
        let worker = request
            .worker
            .ok_or_else(|| Status::invalid_argument("worker is required"))?;
        required(&worker.worker_id, "worker.worker_id")?;
        required(&worker.model_name, "worker.model_name")?;
        if WorkerRole::try_from(worker.role).unwrap_or(WorkerRole::Unspecified)
            == WorkerRole::Unspecified
        {
            return Err(Status::invalid_argument("worker.role must be specified"));
        }
        validate_ttl(request.ttl_seconds)?;

        info!(
            "Registering refit worker '{}' for model '{}' (ttl {}s)",
            worker.worker_id, worker.model_name, request.ttl_seconds
        );
        self.backend
            .register_worker(worker, request.ttl_seconds)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn create_weight_version(
        &self,
        request: Request<CreateWeightVersionRequest>,
    ) -> Result<Response<WeightVersion>, Status> {
        let request = request.into_inner();
        required(&request.model_name, "model_name")?;
        required(&request.idempotency_key, "idempotency_key")?;
        let payload_format = WeightPayloadFormat::try_from(request.payload_format)
            .unwrap_or(WeightPayloadFormat::Unspecified);
        match payload_format {
            WeightPayloadFormat::FullTensor if request.base_version_id.is_some() => {
                return Err(Status::invalid_argument(
                    "base_version_id must be omitted for FULL_TENSOR",
                ));
            }
            WeightPayloadFormat::XorDelta if request.base_version_id.is_none() => {
                return Err(Status::invalid_argument(
                    "base_version_id is required for XOR_DELTA",
                ));
            }
            WeightPayloadFormat::Unspecified => {
                return Err(Status::invalid_argument("payload_format must be specified"));
            }
            _ => {}
        }
        if request.expected_source_slots.is_empty() {
            return Err(Status::invalid_argument(
                "expected_source_slots must not be empty",
            ));
        }
        let mut unique = HashSet::new();
        for source_slot_id in &request.expected_source_slots {
            required(source_slot_id, "expected_source_slots entry")?;
            if !unique.insert(source_slot_id) {
                return Err(Status::invalid_argument(
                    "expected_source_slots must not contain duplicates",
                ));
            }
        }
        if let Some(base_version_id) = request.base_version_id.as_deref()
            && payload_format == WeightPayloadFormat::XorDelta
        {
            let base = self
                .backend
                .get_weight_version(base_version_id)
                .await
                .map_err(backend_status)?;
            if base.model_name != request.model_name
                || base.state != i32::from(WeightVersionState::Ready)
            {
                return Err(Status::failed_precondition(
                    "XOR_DELTA base version must be READY and have the same model_name",
                ));
            }
        }
        info!(
            "Creating weight version for model '{}' ({:?}, {} expected source slots)",
            request.model_name,
            payload_format,
            request.expected_source_slots.len()
        );
        self.backend
            .create_weight_version(&request)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn get_weight_version(
        &self,
        request: Request<GetWeightVersionRequest>,
    ) -> Result<Response<WeightVersion>, Status> {
        let uid = request.into_inner().uid;
        required(&uid, "uid")?;
        debug!("Fetching weight version '{uid}'");
        self.backend
            .get_weight_version(&uid)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn delete_weight_version(
        &self,
        request: Request<DeleteWeightVersionRequest>,
    ) -> Result<Response<WeightVersion>, Status> {
        let uid = request.into_inner().uid;
        required(&uid, "uid")?;
        info!("Deleting weight version '{uid}'");
        self.backend
            .delete_weight_version(&uid)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn create_weight_version_shard(
        &self,
        request: Request<CreateWeightVersionShardRequest>,
    ) -> Result<Response<CreateWeightVersionShardResponse>, Status> {
        let shard = request
            .into_inner()
            .shard
            .ok_or_else(|| Status::invalid_argument("shard is required"))?;
        required(&shard.version_id, "shard.version_id")?;
        required(&shard.source_slot_id, "shard.source_slot_id")?;
        required(&shard.worker_id, "shard.worker_id")?;
        required(&shard.manifest_digest, "shard.manifest_digest")?;
        required(&shard.manifest_endpoint, "shard.manifest_endpoint")?;
        required(&shard.transport, "shard.transport")?;

        info!(
            "Registering shard for version '{}' from worker '{}' (source slot '{}', transport '{}')",
            shard.version_id, shard.worker_id, shard.source_slot_id, shard.transport
        );
        let (shard, version) = self
            .backend
            .create_weight_version_shard(shard)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(CreateWeightVersionShardResponse {
            shard: Some(shard),
            version: Some(version),
        }))
    }

    async fn list_weight_version_shards(
        &self,
        request: Request<ListWeightVersionShardsRequest>,
    ) -> Result<Response<ListWeightVersionShardsResponse>, Status> {
        let version_id = request.into_inner().version_id;
        required(&version_id, "version_id")?;
        debug!("Listing shards for weight version '{version_id}'");
        self.backend
            .list_weight_version_shards(&version_id)
            .await
            .map(|shards| Response::new(ListWeightVersionShardsResponse { shards }))
            .map_err(backend_status)
    }

    async fn delete_weight_version_shard(
        &self,
        request: Request<DeleteWeightVersionShardRequest>,
    ) -> Result<Response<DeleteWeightVersionShardResponse>, Status> {
        let request = request.into_inner();
        required(&request.version_id, "version_id")?;
        required(&request.source_slot_id, "source_slot_id")?;
        required(&request.worker_id, "worker_id")?;
        info!(
            "Deleting shard for version '{}' from worker '{}' (source slot '{}')",
            request.version_id, request.worker_id, request.source_slot_id
        );
        self.backend
            .delete_weight_version_shard(&request)
            .await
            .map(|deleted| Response::new(DeleteWeightVersionShardResponse { deleted }))
            .map_err(backend_status)
    }

    async fn register_version_lease(
        &self,
        request: Request<RegisterVersionLeaseRequest>,
    ) -> Result<Response<VersionLease>, Status> {
        let request = request.into_inner();
        required(&request.version_id, "version_id")?;
        required(&request.worker_id, "worker_id")?;
        validate_ttl(request.ttl_seconds)?;
        info!(
            "Registering lease on version '{}' for worker '{}' (ttl {}s)",
            request.version_id, request.worker_id, request.ttl_seconds
        );
        self.backend
            .register_version_lease(&request)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn delete_version_lease(
        &self,
        request: Request<DeleteVersionLeaseRequest>,
    ) -> Result<Response<DeleteVersionLeaseResponse>, Status> {
        let request = request.into_inner();
        required(&request.version_id, "version_id")?;
        required(&request.lease_id, "lease_id")?;
        required(&request.worker_id, "worker_id")?;
        info!(
            "Releasing lease '{}' on version '{}' for worker '{}'",
            request.lease_id, request.version_id, request.worker_id
        );
        self.backend
            .delete_version_lease(&request)
            .await
            .map(|deleted| Response::new(DeleteVersionLeaseResponse { deleted }))
            .map_err(backend_status)
    }
}
