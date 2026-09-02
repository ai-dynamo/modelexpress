// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral gRPC service for RL refit control-plane metadata.

#![allow(clippy::result_large_err)] // tonic service helpers return tonic::Status.

use std::collections::HashSet;
use std::sync::Arc;

use modelexpress_common::grpc::refit::{
    CreateWeightVersionRequest, CreateWeightVersionResponse, CreateWeightVersionShardRequest,
    CreateWeightVersionShardResponse, DeleteVersionLeaseRequest, DeleteVersionLeaseResponse,
    DeleteWeightVersionRequest, DeleteWeightVersionResponse, DeleteWeightVersionShardRequest,
    DeleteWeightVersionShardResponse, GetWeightVersionRequest, GetWeightVersionResponse,
    ListWeightVersionShardsRequest, ListWeightVersionShardsResponse, ObjectStorageType,
    RegisterVersionLeaseRequest, RegisterVersionLeaseResponse, RegisterWorkerRequest,
    RegisterWorkerResponse, UpdateWeightVersionStateRequest, UpdateWeightVersionStateResponse,
    WeightPayloadFormat, WeightVersionState, WorkerRole, refit_service_server::RefitService,
};
use tonic::{Request, Response, Status};

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

fn validate_s3_uri(uri: &str) -> Result<(), Status> {
    if uri.contains('?') || uri.contains('#') {
        return Err(Status::invalid_argument(
            "object_storage.uri must not contain a query or fragment",
        ));
    }
    let Some(location) = uri.strip_prefix("s3://") else {
        return Err(Status::invalid_argument(
            "object_storage.uri must use the s3:// scheme",
        ));
    };
    let Some((bucket, key)) = location.split_once('/') else {
        return Err(Status::invalid_argument(
            "object_storage.uri must include a bucket and key",
        ));
    };
    if bucket.trim().is_empty() || key.trim().is_empty() || key.starts_with('/') {
        return Err(Status::invalid_argument(
            "object_storage.uri must include a bucket and key",
        ));
    }
    Ok(())
}

fn validate_payload_format(
    request: &CreateWeightVersionRequest,
) -> Result<WeightPayloadFormat, Status> {
    let payload_format = WeightPayloadFormat::try_from(request.payload_format)
        .unwrap_or(WeightPayloadFormat::Unspecified);
    match payload_format {
        WeightPayloadFormat::FullTensor if request.base_version_id.is_some() => Err(
            Status::invalid_argument("base_version_id must be omitted for FULL_TENSOR"),
        ),
        WeightPayloadFormat::XorDelta if request.base_version_id.is_none() => Err(
            Status::invalid_argument("base_version_id is required for XOR_DELTA"),
        ),
        WeightPayloadFormat::FullHfCheckpoint if request.base_version_id.is_some() => Err(
            Status::invalid_argument("base_version_id must be omitted for FULL_HF_CHECKPOINT"),
        ),
        WeightPayloadFormat::FullHfCheckpoint
            if request.object_storage.as_ref().is_none_or(|source| {
                ObjectStorageType::try_from(source.storage_type)
                    .unwrap_or(ObjectStorageType::Unspecified)
                    != ObjectStorageType::S3
            }) =>
        {
            Err(Status::invalid_argument(
                "FULL_HF_CHECKPOINT requires S3 object_storage",
            ))
        }
        WeightPayloadFormat::Unspecified => {
            Err(Status::invalid_argument("payload_format must be specified"))
        }
        _ => Ok(payload_format),
    }
}

fn validate_publication(request: &CreateWeightVersionRequest) -> Result<(), Status> {
    if let Some(uid) = request.uid.as_deref() {
        required(uid, "uid")?;
    }
    let state =
        WeightVersionState::try_from(request.state).unwrap_or(WeightVersionState::Unspecified);
    if let Some(object_storage) = request.object_storage.as_ref() {
        if ObjectStorageType::try_from(object_storage.storage_type)
            .unwrap_or(ObjectStorageType::Unspecified)
            != ObjectStorageType::S3
        {
            return Err(Status::invalid_argument(
                "only S3 object storage is currently supported",
            ));
        }
        required(&object_storage.uri, "object_storage.uri")?;
        validate_s3_uri(&object_storage.uri)?;
        if !request.expected_source_slots.is_empty() {
            return Err(Status::invalid_argument(
                "expected_source_slots must be empty for S3 publication",
            ));
        }
        if !matches!(
            state,
            WeightVersionState::Staging | WeightVersionState::Ready
        ) {
            return Err(Status::invalid_argument(
                "S3 state must be STAGING or READY",
            ));
        }
        return Ok(());
    }

    if state != WeightVersionState::Staging {
        return Err(Status::invalid_argument(
            "worker-sharded state must be STAGING",
        ));
    }
    if request.expected_source_slots.is_empty() {
        return Err(Status::invalid_argument(
            "expected_source_slots must not be empty for worker-sharded publication",
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
    Ok(())
}

fn backend_status(error: RefitBackendError) -> Status {
    match error {
        RefitBackendError::InvalidArgument(message) => Status::invalid_argument(message),
        RefitBackendError::NotFound(message) => Status::not_found(message),
        RefitBackendError::FailedPrecondition(message) => Status::failed_precondition(message),
        RefitBackendError::AlreadyExists(message) => Status::already_exists(message),
        RefitBackendError::ResourceExhausted(message) => Status::resource_exhausted(message),
        RefitBackendError::Internal(message) => Status::internal(message),
        RefitBackendError::Unavailable(message) => {
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
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
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

        let worker = self
            .backend
            .register_worker(worker, request.ttl_seconds)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(RegisterWorkerResponse {
            worker: Some(worker),
        }))
    }

    async fn create_weight_version(
        &self,
        request: Request<CreateWeightVersionRequest>,
    ) -> Result<Response<CreateWeightVersionResponse>, Status> {
        let request = request.into_inner();
        required(&request.model_name, "model_name")?;
        required(&request.idempotency_key, "idempotency_key")?;
        let payload_format = validate_payload_format(&request)?;
        validate_publication(&request)?;
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
        let version = self
            .backend
            .create_weight_version(&request)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(CreateWeightVersionResponse {
            version: Some(version),
        }))
    }

    async fn get_weight_version(
        &self,
        request: Request<GetWeightVersionRequest>,
    ) -> Result<Response<GetWeightVersionResponse>, Status> {
        let uid = request.into_inner().uid;
        required(&uid, "uid")?;
        let version = self
            .backend
            .get_weight_version(&uid)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(GetWeightVersionResponse {
            version: Some(version),
        }))
    }

    async fn update_weight_version_state(
        &self,
        request: Request<UpdateWeightVersionStateRequest>,
    ) -> Result<Response<UpdateWeightVersionStateResponse>, Status> {
        let request = request.into_inner();
        required(&request.uid, "uid")?;
        if WeightVersionState::try_from(request.state).unwrap_or(WeightVersionState::Unspecified)
            == WeightVersionState::Unspecified
        {
            return Err(Status::invalid_argument("state must be specified"));
        }
        let version = self
            .backend
            .update_weight_version_state(&request)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(UpdateWeightVersionStateResponse {
            version: Some(version),
        }))
    }

    async fn delete_weight_version(
        &self,
        request: Request<DeleteWeightVersionRequest>,
    ) -> Result<Response<DeleteWeightVersionResponse>, Status> {
        let uid = request.into_inner().uid;
        required(&uid, "uid")?;
        let version = self
            .backend
            .delete_weight_version(&uid)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(DeleteWeightVersionResponse {
            version: Some(version),
        }))
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
        self.backend
            .delete_weight_version_shard(&request)
            .await
            .map(|deleted| Response::new(DeleteWeightVersionShardResponse { deleted }))
            .map_err(backend_status)
    }

    async fn register_version_lease(
        &self,
        request: Request<RegisterVersionLeaseRequest>,
    ) -> Result<Response<RegisterVersionLeaseResponse>, Status> {
        let request = request.into_inner();
        required(&request.version_id, "version_id")?;
        required(&request.worker_id, "worker_id")?;
        validate_ttl(request.ttl_seconds)?;
        let lease = self
            .backend
            .register_version_lease(&request)
            .await
            .map_err(backend_status)?;
        Ok(Response::new(RegisterVersionLeaseResponse {
            lease: Some(lease),
        }))
    }

    async fn delete_version_lease(
        &self,
        request: Request<DeleteVersionLeaseRequest>,
    ) -> Result<Response<DeleteVersionLeaseResponse>, Status> {
        let request = request.into_inner();
        required(&request.version_id, "version_id")?;
        required(&request.lease_id, "lease_id")?;
        required(&request.worker_id, "worker_id")?;
        self.backend
            .delete_version_lease(&request)
            .await
            .map(|deleted| Response::new(DeleteVersionLeaseResponse { deleted }))
            .map_err(backend_status)
    }
}

#[cfg(test)]
mod tests {
    use modelexpress_common::grpc::refit::ObjectStorageSource;
    use tonic::Code;

    use super::*;

    fn error_code<T>(result: Result<T, Status>) -> Code {
        match result {
            Ok(_) => panic!("publication validation unexpectedly succeeded"),
            Err(status) => status.code(),
        }
    }

    fn s3_source(uri: &str) -> ObjectStorageSource {
        ObjectStorageSource {
            uri: uri.to_string(),
            storage_type: ObjectStorageType::S3.into(),
        }
    }

    #[test]
    fn full_hf_checkpoint_requires_s3_and_omits_base() {
        let full_hf = CreateWeightVersionRequest {
            payload_format: WeightPayloadFormat::FullHfCheckpoint.into(),
            object_storage: Some(s3_source(
                "s3://weights/run/policy/v25/model.safetensors.index.json",
            )),
            state: WeightVersionState::Staging.into(),
            ..Default::default()
        };
        assert!(matches!(
            validate_payload_format(&full_hf),
            Ok(WeightPayloadFormat::FullHfCheckpoint)
        ));
        assert!(validate_publication(&full_hf).is_ok());

        let mut with_base = full_hf.clone();
        with_base.base_version_id = Some("previous-version".to_string());
        assert_eq!(
            error_code(validate_payload_format(&with_base)),
            Code::InvalidArgument
        );

        let mut without_storage = full_hf.clone();
        without_storage.object_storage = None;
        assert_eq!(
            error_code(validate_payload_format(&without_storage)),
            Code::InvalidArgument
        );

        let mut with_gcs = full_hf;
        with_gcs.object_storage = Some(ObjectStorageSource {
            uri: "gs://weights/run/policy/v25/model.safetensors.index.json".to_string(),
            storage_type: ObjectStorageType::Gcs.into(),
        });
        assert_eq!(
            error_code(validate_payload_format(&with_gcs)),
            Code::InvalidArgument
        );
    }

    #[test]
    fn full_tensor_remains_valid_for_s3_publication() {
        let request = CreateWeightVersionRequest {
            payload_format: WeightPayloadFormat::FullTensor.into(),
            object_storage: Some(s3_source(
                "s3://weights/run/policy/v42/model.safetensors.index.json",
            )),
            state: WeightVersionState::Staging.into(),
            ..Default::default()
        };

        assert!(matches!(
            validate_payload_format(&request),
            Ok(WeightPayloadFormat::FullTensor)
        ));
        assert!(validate_publication(&request).is_ok());
    }

    #[test]
    fn valid_object_storage_and_worker_sharded_publications_are_accepted() {
        for state in [WeightVersionState::Staging, WeightVersionState::Ready] {
            assert!(
                validate_publication(&CreateWeightVersionRequest {
                    uid: Some("caller-version".to_string()),
                    object_storage: Some(s3_source(
                        "s3://weights/run/policy/v42/model.safetensors.index.json",
                    )),
                    state: state.into(),
                    ..Default::default()
                })
                .is_ok()
            );
        }
        assert!(
            validate_publication(&CreateWeightVersionRequest {
                expected_source_slots: vec!["rank:0".to_string()],
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn invalid_object_storage_and_worker_sharded_publications_are_rejected() {
        for request in [
            CreateWeightVersionRequest {
                uid: Some(" \t".to_string()),
                expected_source_slots: vec!["rank:0".to_string()],
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(ObjectStorageSource::default()),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(s3_source("https://weights/root")),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(s3_source("s3://weights")),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(s3_source("s3:///root")),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                expected_source_slots: vec!["rank:0".to_string()],
                object_storage: Some(s3_source("s3://weights/root")),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(s3_source("s3://weights/root")),
                state: WeightVersionState::Releasing.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(s3_source("s3://weights//root")),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(s3_source("s3://weights/root?query")),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(ObjectStorageSource {
                    uri: "az://weights/root".to_string(),
                    storage_type: ObjectStorageType::Azure.into(),
                }),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                object_storage: Some(ObjectStorageSource {
                    uri: "gs://weights/root".to_string(),
                    storage_type: ObjectStorageType::Gcs.into(),
                }),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                expected_source_slots: vec!["rank:0".to_string()],
                state: WeightVersionState::Ready.into(),
                ..Default::default()
            },
            CreateWeightVersionRequest {
                expected_source_slots: vec!["rank:0".to_string(), "rank:0".to_string()],
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            },
        ] {
            assert_eq!(
                error_code(validate_publication(&request)),
                Code::InvalidArgument
            );
        }
    }
}
