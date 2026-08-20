// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral gRPC service for the NCCL M2N collective control plane.

#![allow(clippy::result_large_err)] // tonic service helpers return tonic::Status.

use std::sync::Arc;

use modelexpress_common::grpc::refit_collective::{
    CollectiveGroup, CollectiveGroupMembership, CollectiveGroupSpec, CollectiveRole,
    CollectiveTransfer, CreateCollectiveTransferRequest, DeleteCollectiveTransferRequest,
    GetCollectiveGroupRequest, GetCollectiveTransferRequest, JoinCollectiveGroupRequest,
    PublishGroupBootstrapRequest, ReportCollectiveTransferRequest,
    refit_collective_service_server::RefitCollectiveService,
};
use tonic::{Request, Response, Status};

use super::backend::{CollectiveBackend, CollectiveBackendError};

/// Size of an `ncclUniqueId`. Checked here rather than at
/// `Communicator.init`, because a truncated identifier surfaces there as every
/// rank of the lane blocking rather than as an error.
const NCCL_UNIQUE_ID_BYTES: usize = 128;

fn required(value: &str, field: &str) -> Result<(), Status> {
    if value.trim().is_empty() {
        Err(Status::invalid_argument(format!("{field} is required")))
    } else {
        Ok(())
    }
}

fn delimiter_free(value: &str, field: &str, delimiters: &[char]) -> Result<(), Status> {
    if let Some(delimiter) = delimiters
        .iter()
        .find(|delimiter| value.contains(**delimiter))
    {
        return Err(Status::invalid_argument(format!(
            "{field} must not contain the reserved delimiter {:?}",
            delimiter
        )));
    }
    Ok(())
}

fn backend_status(error: CollectiveBackendError) -> Status {
    match error {
        CollectiveBackendError::InvalidArgument(message) => Status::invalid_argument(message),
        CollectiveBackendError::NotFound(message) => Status::not_found(message),
        CollectiveBackendError::FailedPrecondition(message) => Status::failed_precondition(message),
        CollectiveBackendError::AlreadyExists(message) => Status::already_exists(message),
        CollectiveBackendError::Internal(message) => Status::internal(message),
        CollectiveBackendError::Unavailable(message) => {
            Status::unavailable(format!("Collective metadata backend error: {message}"))
        }
    }
}

/// Validate the membership declaration every participant of one operation must
/// send identically.
fn validate_spec(spec: Option<&CollectiveGroupSpec>) -> Result<&CollectiveGroupSpec, Status> {
    let spec = spec.ok_or_else(|| Status::invalid_argument("spec is required"))?;
    required(&spec.model_name, "spec.model_name")?;
    delimiter_free(&spec.model_name, "spec.model_name", &['\0'])?;
    if spec.expected_trainer_slots.is_empty() {
        return Err(Status::invalid_argument(
            "spec.expected_trainer_slots must not be empty",
        ));
    }
    if spec.expected_generator_slots.is_empty() {
        return Err(Status::invalid_argument(
            "spec.expected_generator_slots must not be empty",
        ));
    }
    if spec.source_partition_count == 0 {
        return Err(Status::invalid_argument(
            "spec.source_partition_count must be greater than zero",
        ));
    }
    // Duplicate slots would make the expected count disagree with the number
    // of distinct participants, so the group could never reach READY and the
    // failure would present as a timeout rather than as the typo it is.
    if has_duplicates(&spec.expected_trainer_slots) {
        return Err(Status::invalid_argument(
            "spec.expected_trainer_slots must not contain duplicates",
        ));
    }
    if has_duplicates(&spec.expected_generator_slots) {
        return Err(Status::invalid_argument(
            "spec.expected_generator_slots must not contain duplicates",
        ));
    }
    for (field, slots) in [
        ("spec.expected_trainer_slots", &spec.expected_trainer_slots),
        (
            "spec.expected_generator_slots",
            &spec.expected_generator_slots,
        ),
    ] {
        for slot in slots {
            required(slot, field)?;
            delimiter_free(slot, field, &['\0', '\n', '\r', '|'])?;
        }
    }
    if spec
        .expected_trainer_slots
        .iter()
        .any(|slot| spec.expected_generator_slots.contains(slot))
    {
        return Err(Status::invalid_argument(
            "trainer and generator slot namespaces must not overlap",
        ));
    }
    Ok(spec)
}

fn has_duplicates(slots: &[String]) -> bool {
    let mut sorted: Vec<&String> = slots.iter().collect();
    sorted.sort();
    sorted.windows(2).any(|pair| pair[0] == pair[1])
}

#[derive(Clone)]
pub struct RefitCollectiveServiceImpl {
    backend: Arc<dyn CollectiveBackend>,
}

impl RefitCollectiveServiceImpl {
    pub fn new(backend: Arc<dyn CollectiveBackend>) -> Self {
        Self { backend }
    }
}

#[tonic::async_trait]
impl RefitCollectiveService for RefitCollectiveServiceImpl {
    async fn create_collective_transfer(
        &self,
        request: Request<CreateCollectiveTransferRequest>,
    ) -> Result<Response<CollectiveTransfer>, Status> {
        let request = request.into_inner();
        validate_spec(request.spec.as_ref())?;
        required(&request.version_id, "version_id")?;
        required(&request.idempotency_key, "idempotency_key")?;

        self.backend
            .create_transfer(&request)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn get_collective_transfer(
        &self,
        request: Request<GetCollectiveTransferRequest>,
    ) -> Result<Response<CollectiveTransfer>, Status> {
        let request = request.into_inner();
        required(&request.operation_id, "operation_id")?;
        self.backend
            .get_transfer(&request.operation_id)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn delete_collective_transfer(
        &self,
        request: Request<DeleteCollectiveTransferRequest>,
    ) -> Result<Response<CollectiveTransfer>, Status> {
        let request = request.into_inner();
        required(&request.operation_id, "operation_id")?;
        self.backend
            .delete_transfer(&request.operation_id)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn join_collective_group(
        &self,
        request: Request<JoinCollectiveGroupRequest>,
    ) -> Result<Response<CollectiveGroupMembership>, Status> {
        let request = request.into_inner();
        validate_spec(request.spec.as_ref())?;
        required(&request.slot_id, "slot_id")?;
        required(&request.worker_id, "worker_id")?;
        required(&request.plan_digest, "plan_digest")?;
        delimiter_free(&request.slot_id, "slot_id", &['\0', '\n', '\r', '|'])?;
        delimiter_free(&request.worker_id, "worker_id", &['\0', '|'])?;

        let role = CollectiveRole::try_from(request.role).unwrap_or(CollectiveRole::Unspecified);
        if role == CollectiveRole::Unspecified {
            return Err(Status::invalid_argument("role must be specified"));
        }
        if let Some(source) = request.plan_source.as_ref() {
            required(&source.worker_id, "plan_source.worker_id")?;
            required(&source.endpoint, "plan_source.endpoint")?;
            required(&source.digest, "plan_source.digest")?;
            // The digest MX advertises is what generators verify the fetched
            // plan against, so a source that advertises a different one would
            // make every generator fail closed at fetch time.
            if source.digest != request.plan_digest {
                return Err(Status::invalid_argument(
                    "plan_source.digest must match plan_digest",
                ));
            }
            if source.worker_id != request.worker_id {
                return Err(Status::invalid_argument(
                    "plan_source.worker_id must match worker_id",
                ));
            }
            if role != CollectiveRole::Trainer || request.index_in_role != 0 {
                return Err(Status::invalid_argument(
                    "only trainer index_in_role 0 may advertise plan_source",
                ));
            }
        }

        self.backend
            .join_group(&request)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn get_collective_group(
        &self,
        request: Request<GetCollectiveGroupRequest>,
    ) -> Result<Response<CollectiveGroup>, Status> {
        let request = request.into_inner();
        required(&request.group_id, "group_id")?;
        self.backend
            .get_group(&request.group_id)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn publish_group_bootstrap(
        &self,
        request: Request<PublishGroupBootstrapRequest>,
    ) -> Result<Response<CollectiveGroup>, Status> {
        let request = request.into_inner();
        required(&request.group_id, "group_id")?;
        required(&request.worker_id, "worker_id")?;
        if request.epoch == 0 {
            return Err(Status::invalid_argument(
                "epoch must be the group's current epoch",
            ));
        }
        if request.nccl_unique_id.len() != NCCL_UNIQUE_ID_BYTES {
            return Err(Status::invalid_argument(format!(
                "nccl_unique_id must be {NCCL_UNIQUE_ID_BYTES} bytes, got {}",
                request.nccl_unique_id.len()
            )));
        }

        self.backend
            .publish_bootstrap(&request)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }

    async fn report_collective_transfer(
        &self,
        request: Request<ReportCollectiveTransferRequest>,
    ) -> Result<Response<CollectiveTransfer>, Status> {
        let request = request.into_inner();
        required(&request.operation_id, "operation_id")?;
        required(&request.group_id, "group_id")?;
        required(&request.worker_id, "worker_id")?;
        if request.epoch == 0 {
            return Err(Status::invalid_argument(
                "epoch must be the epoch the operation was admitted against",
            ));
        }
        if !request.succeeded {
            required(&request.message, "message")?;
        }

        self.backend
            .report_transfer(&request)
            .await
            .map(Response::new)
            .map_err(backend_status)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn spec() -> CollectiveGroupSpec {
        CollectiveGroupSpec {
            model_name: "m".to_string(),
            expected_trainer_slots: vec!["t0".to_string(), "t1".to_string()],
            expected_generator_slots: vec!["g0".to_string()],
            source_partition_count: 1,
        }
    }

    #[test]
    fn a_valid_spec_passes() {
        assert!(validate_spec(Some(&spec())).is_ok());
    }

    #[test]
    fn an_absent_spec_is_rejected() {
        assert_eq!(
            validate_spec(None).expect_err("absent spec").code(),
            tonic::Code::InvalidArgument
        );
    }

    #[test]
    fn empty_membership_is_rejected() {
        let mut s = spec();
        s.expected_trainer_slots.clear();
        assert!(validate_spec(Some(&s)).is_err());

        let mut s = spec();
        s.expected_generator_slots.clear();
        assert!(validate_spec(Some(&s)).is_err());

        let mut s = spec();
        s.source_partition_count = 0;
        assert!(validate_spec(Some(&s)).is_err());

        let mut s = spec();
        s.model_name = "  ".to_string();
        assert!(validate_spec(Some(&s)).is_err());
    }

    #[test]
    fn duplicate_slots_are_rejected_rather_than_timing_out() {
        // A duplicated slot makes the expected count exceed the number of
        // distinct participants, so the group would sit in FORMING until the
        // client deadline instead of reporting the typo.
        let mut s = spec();
        s.expected_trainer_slots = vec!["t0".to_string(), "t0".to_string()];
        assert!(validate_spec(Some(&s)).is_err());

        let mut s = spec();
        s.expected_generator_slots = vec!["g0".to_string(), "g0".to_string()];
        assert!(validate_spec(Some(&s)).is_err());
    }

    #[test]
    fn duplicate_detection_does_not_reject_distinct_slots() {
        assert!(!has_duplicates(&[
            "a".to_string(),
            "b".to_string(),
            "c".to_string()
        ]));
        assert!(has_duplicates(&[
            "a".to_string(),
            "c".to_string(),
            "a".to_string()
        ]));
        assert!(!has_duplicates(&[]));
    }

    #[test]
    fn slots_must_not_repeat_across_roles() {
        // Redis admission is keyed by slot_id, so sharing a name across roles
        // would overwrite one participant and leave the group permanently short.
        let mut s = spec();
        s.expected_trainer_slots = vec!["r0".to_string()];
        s.expected_generator_slots = vec!["r0".to_string()];
        assert!(validate_spec(Some(&s)).is_err());
    }

    #[test]
    fn slots_reject_the_redis_record_delimiters() {
        let mut s = spec();
        s.expected_trainer_slots = vec!["trainer\n0".to_string()];
        assert!(validate_spec(Some(&s)).is_err());

        let mut s = spec();
        s.expected_generator_slots = vec!["generator|0".to_string()];
        assert!(validate_spec(Some(&s)).is_err());
    }
}
