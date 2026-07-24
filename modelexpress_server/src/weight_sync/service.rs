// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WeightSyncService gRPC implementation.
//!
//! Responsibilities:
//!   - Store/retrieve TrainerTable and InferenceTable blobs in Redis.
//!   - Accept resolved element-run regions from inference workers.
//!   - Route regions against the stored TrainerTable (in Rust, zero torch).
//!   - Cache the resulting RdmaDescriptor plans by plan_key.
//!   - Serve cached plans to all workers sharing the same model topology.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use modelexpress_common::grpc::weight_sync::{
    BuildPlanRequest, BuildPlanResponse, GetM2nPlanRequest, GetM2nPlanResponse,
    GetNcclBootstrapRequest, GetNcclBootstrapResponse, GetPlanRequest, GetPlanResponse,
    GetTrainerTableRequest, GetTrainerTableResponse, InvalidateM2nPlanRequest,
    InvalidateM2nPlanResponse, InvalidatePlanRequest, InvalidatePlanResponse, M2nDescriptorProto,
    NcclBootstrapRecord, PublishInferenceTableRequest, PublishInferenceTableResponse,
    PublishNcclBootstrapRequest, PublishNcclBootstrapResponse, PublishTrainerTableRequest,
    PublishTrainerTableResponse, RdmaDescriptorProto, RegisterM2nWorkerRequest,
    RegisterM2nWorkerResponse,
    weight_sync_service_server::WeightSyncService as WeightSyncServiceTrait,
};

use super::router::{TrainerTableJson, route_all_workers, route_regions};

// ---------------------------------------------------------------------------
// In-memory state
// ---------------------------------------------------------------------------

/// Cached plan: the list of RDMA descriptors built from one worker's regions.
#[derive(Clone)]
struct CachedPlan {
    descriptors: Vec<RdmaDescriptorProto>,
}

/// One worker's registration data for an M2N collective plan.
struct M2nWorkerRegistration {
    regions: Vec<modelexpress_common::grpc::weight_sync::ResolvedRegionProto>,
    _nixl_metadata: Vec<u8>,
}

/// Pending M2N plan accumulating worker registrations until the barrier fires.
struct M2nPlanEntry {
    total_workers: i32,
    registrations: HashMap<i32, M2nWorkerRegistration>,
    /// Set once all workers have registered and the plan is built.
    plan_id: Option<String>,
}

/// Shared server state (plan cache + raw table blobs).
struct WeightSyncState {
    /// plan_id -> built plan
    plans: HashMap<String, CachedPlan>,
    /// plan_key -> plan_id  (so workers with the same key reuse the plan)
    plan_key_to_id: HashMap<String, String>,
    /// model_key -> raw JSON-encoded TrainerTable bytes
    trainer_tables: HashMap<String, Vec<u8>>,
    /// (model_key, worker_rank) -> raw JSON-encoded InferenceTable bytes
    inference_tables: HashMap<(String, i32), Vec<u8>>,
    /// model_key -> M2N plan accumulator
    m2n_pending: HashMap<String, M2nPlanEntry>,
    /// (plan_id, worker_rank) -> this worker's M2N descriptors
    m2n_worker_plans: HashMap<(String, i32), Vec<M2nDescriptorProto>>,
    /// attempt_id -> coordinator-defined NCCL communicator bootstrap record
    nccl_bootstraps: HashMap<String, NcclBootstrapRecord>,
}

impl WeightSyncState {
    fn new() -> Self {
        Self {
            plans: HashMap::new(),
            plan_key_to_id: HashMap::new(),
            trainer_tables: HashMap::new(),
            inference_tables: HashMap::new(),
            m2n_pending: HashMap::new(),
            m2n_worker_plans: HashMap::new(),
            nccl_bootstraps: HashMap::new(),
        }
    }
}

const NCCL_UNIQUE_ID_BYTES: usize = 128;
const SHA256_BYTES: usize = 32;

fn validate_attempt_id(attempt_id: &str) -> Result<(), Status> {
    let parsed = Uuid::parse_str(attempt_id).map_err(|_| {
        Status::invalid_argument(
            "attempt_id must be a canonical random UUIDv4 and must never be reused",
        )
    })?;
    if parsed.get_version() != Some(uuid::Version::Random)
        || parsed.hyphenated().to_string() != attempt_id
    {
        return Err(Status::invalid_argument(
            "attempt_id must be a canonical random UUIDv4 and must never be reused",
        ));
    }
    Ok(())
}

fn validate_nccl_bootstrap_record(record: &NcclBootstrapRecord) -> Result<(), Status> {
    validate_attempt_id(&record.attempt_id)?;
    if record.cohort_id.is_empty() {
        return Err(Status::invalid_argument("cohort_id is required"));
    }
    if record.nccl_unique_id.len() != NCCL_UNIQUE_ID_BYTES {
        return Err(Status::invalid_argument(format!(
            "nccl_unique_id must be exactly {NCCL_UNIQUE_ID_BYTES} bytes"
        )));
    }
    if record.source_world_size == 0 || record.destination_world_size == 0 {
        return Err(Status::invalid_argument(
            "source_world_size and destination_world_size must be non-zero",
        ));
    }
    let expected_world_size = record
        .source_world_size
        .checked_add(record.destination_world_size)
        .ok_or_else(|| Status::invalid_argument("NCCL world size overflow"))?;
    if record.world_size != expected_world_size {
        return Err(Status::invalid_argument(
            "world_size must equal source_world_size + destination_world_size",
        ));
    }
    if record.roster_digest.len() != SHA256_BYTES {
        return Err(Status::invalid_argument(
            "roster_digest must be a 32-byte SHA-256 digest",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

pub struct WeightSyncServiceImpl {
    state: Arc<RwLock<WeightSyncState>>,
}

impl Default for WeightSyncServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightSyncServiceImpl {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(WeightSyncState::new())),
        }
    }
}

#[tonic::async_trait]
impl WeightSyncServiceTrait for WeightSyncServiceImpl {
    async fn publish_nccl_bootstrap(
        &self,
        request: Request<PublishNcclBootstrapRequest>,
    ) -> Result<Response<PublishNcclBootstrapResponse>, Status> {
        let record = request
            .into_inner()
            .record
            .ok_or_else(|| Status::invalid_argument("record is required"))?;
        validate_nccl_bootstrap_record(&record)?;

        self.state
            .write()
            .await
            .nccl_bootstraps
            .insert(record.attempt_id.clone(), record);
        Ok(Response::new(PublishNcclBootstrapResponse { ok: true }))
    }

    async fn get_nccl_bootstrap(
        &self,
        request: Request<GetNcclBootstrapRequest>,
    ) -> Result<Response<GetNcclBootstrapResponse>, Status> {
        let attempt_id = request.into_inner().attempt_id;
        validate_attempt_id(&attempt_id)?;

        let record = self
            .state
            .read()
            .await
            .nccl_bootstraps
            .get(&attempt_id)
            .cloned();
        Ok(Response::new(GetNcclBootstrapResponse {
            found: record.is_some(),
            record,
        }))
    }

    async fn publish_trainer_table(
        &self,
        request: Request<PublishTrainerTableRequest>,
    ) -> Result<Response<PublishTrainerTableResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;
        state
            .trainer_tables
            .insert(req.model_key.clone(), req.table_payload);
        // Invalidate any cached plans that used this model's table
        state
            .plan_key_to_id
            .retain(|k, _| !k.starts_with(&req.model_key));
        Ok(Response::new(PublishTrainerTableResponse { ok: true }))
    }

    async fn get_trainer_table(
        &self,
        request: Request<GetTrainerTableRequest>,
    ) -> Result<Response<GetTrainerTableResponse>, Status> {
        let req = request.into_inner();
        let state = self.state.read().await;
        match state.trainer_tables.get(&req.model_key) {
            Some(payload) => Ok(Response::new(GetTrainerTableResponse {
                found: true,
                table_payload: payload.clone(),
            })),
            None => Ok(Response::new(GetTrainerTableResponse {
                found: false,
                table_payload: vec![],
            })),
        }
    }

    async fn publish_inference_table(
        &self,
        request: Request<PublishInferenceTableRequest>,
    ) -> Result<Response<PublishInferenceTableResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;
        state
            .inference_tables
            .insert((req.model_key, req.worker_rank), req.table_payload);
        Ok(Response::new(PublishInferenceTableResponse { ok: true }))
    }

    async fn build_plan(
        &self,
        request: Request<BuildPlanRequest>,
    ) -> Result<Response<BuildPlanResponse>, Status> {
        let req = request.into_inner();

        // Reuse an existing plan for this plan_key if one was already built.
        {
            let state = self.state.read().await;
            if let Some(plan_id) = state.plan_key_to_id.get(&req.plan_key) {
                return Ok(Response::new(BuildPlanResponse {
                    plan_id: plan_id.clone(),
                }));
            }
        }

        // Fetch the TrainerTable and deserialize it.
        let table_bytes = {
            let state = self.state.read().await;
            state
                .trainer_tables
                .get(&req.model_key)
                .cloned()
                .ok_or_else(|| {
                    Status::not_found(format!(
                        "TrainerTable not found for model_key {:?}",
                        req.model_key
                    ))
                })?
        };

        let table: TrainerTableJson = serde_json::from_slice(&table_bytes)
            .map_err(|e| Status::internal(format!("Failed to decode TrainerTable: {e}")))?;

        // Route the resolved regions in Rust (no torch).
        let descriptors = route_regions(&req.regions, &table);

        let plan_id = Uuid::new_v4().to_string();
        let mut state = self.state.write().await;
        state
            .plans
            .insert(plan_id.clone(), CachedPlan { descriptors });
        state.plan_key_to_id.insert(req.plan_key, plan_id.clone());

        Ok(Response::new(BuildPlanResponse { plan_id }))
    }

    async fn get_plan(
        &self,
        request: Request<GetPlanRequest>,
    ) -> Result<Response<GetPlanResponse>, Status> {
        let req = request.into_inner();
        let state = self.state.read().await;
        match state.plans.get(&req.plan_id) {
            Some(plan) => Ok(Response::new(GetPlanResponse {
                ready: true,
                descriptors: plan.descriptors.clone(),
            })),
            None => Ok(Response::new(GetPlanResponse {
                ready: false,
                descriptors: vec![],
            })),
        }
    }

    async fn invalidate_plan(
        &self,
        request: Request<InvalidatePlanRequest>,
    ) -> Result<Response<InvalidatePlanResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;
        if let Some(plan_id) = state.plan_key_to_id.remove(&req.plan_key) {
            state.plans.remove(&plan_id);
        }
        Ok(Response::new(InvalidatePlanResponse { ok: true }))
    }

    async fn register_m2n_worker(
        &self,
        request: Request<RegisterM2nWorkerRequest>,
    ) -> Result<Response<RegisterM2nWorkerResponse>, Status> {
        let req = request.into_inner();

        let mut state = self.state.write().await;

        // Reuse an existing plan_id if this worker already registered.
        if let Some(entry) = state.m2n_pending.get(&req.model_key)
            && let Some(plan_id) = &entry.plan_id
        {
            return Ok(Response::new(RegisterM2nWorkerResponse {
                m2n_plan_id: plan_id.clone(),
            }));
        }

        // Accumulate this worker's registration, then check the barrier.
        // Collect all data needed from `entry` inside a scoped block so the
        // mutable borrow of state.m2n_pending is dropped before we read
        // state.trainer_tables (rustc cannot prove disjointness through
        // HashMap::entry() across a function-call boundary).
        let worker_data: Option<
            Vec<(
                i32,
                Vec<modelexpress_common::grpc::weight_sync::ResolvedRegionProto>,
            )>,
        > = {
            let entry = state
                .m2n_pending
                .entry(req.model_key.clone())
                .or_insert_with(|| M2nPlanEntry {
                    total_workers: req.total_workers,
                    registrations: HashMap::new(),
                    plan_id: None,
                });

            entry.registrations.insert(
                req.worker_rank,
                M2nWorkerRegistration {
                    regions: req.regions,
                    _nixl_metadata: req.nixl_metadata,
                },
            );

            if entry.registrations.len() < entry.total_workers as usize {
                None
            } else {
                Some(
                    entry
                        .registrations
                        .iter()
                        .map(|(rank, reg)| (*rank, reg.regions.clone()))
                        .collect(),
                )
            }
        };

        let worker_data = match worker_data {
            None => {
                return Ok(Response::new(RegisterM2nWorkerResponse {
                    m2n_plan_id: String::new(),
                }));
            }
            Some(d) => d,
        };

        // All workers registered: fetch the TrainerTable and route.
        let table_bytes = state
            .trainer_tables
            .get(&req.model_key)
            .cloned()
            .ok_or_else(|| {
                Status::not_found(format!(
                    "TrainerTable not found for model_key {:?}",
                    req.model_key
                ))
            })?;

        let table: TrainerTableJson = serde_json::from_slice(&table_bytes)
            .map_err(|e| Status::internal(format!("Failed to decode TrainerTable: {e}")))?;

        let worker_refs: Vec<(
            i32,
            &[modelexpress_common::grpc::weight_sync::ResolvedRegionProto],
        )> = worker_data
            .iter()
            .map(|(r, v)| (*r, v.as_slice()))
            .collect();

        let per_worker = route_all_workers(&worker_refs, &table);

        let plan_id = Uuid::new_v4().to_string();

        for (rank, descs) in per_worker {
            state
                .m2n_worker_plans
                .insert((plan_id.clone(), rank), descs);
        }

        // Record the plan_id so subsequent RegisterM2nWorker calls return it.
        if let Some(entry) = state.m2n_pending.get_mut(&req.model_key) {
            entry.plan_id = Some(plan_id.clone());
        }

        Ok(Response::new(RegisterM2nWorkerResponse {
            m2n_plan_id: plan_id,
        }))
    }

    async fn get_m2n_plan(
        &self,
        request: Request<GetM2nPlanRequest>,
    ) -> Result<Response<GetM2nPlanResponse>, Status> {
        let req = request.into_inner();

        if req.m2n_plan_id.is_empty() {
            // Barrier not yet satisfied; caller should poll again.
            return Ok(Response::new(GetM2nPlanResponse {
                ready: false,
                descriptors: vec![],
            }));
        }

        let state = self.state.read().await;
        match state
            .m2n_worker_plans
            .get(&(req.m2n_plan_id.clone(), req.worker_rank))
        {
            Some(descs) => Ok(Response::new(GetM2nPlanResponse {
                ready: true,
                descriptors: descs.clone(),
            })),
            None => Ok(Response::new(GetM2nPlanResponse {
                ready: false,
                descriptors: vec![],
            })),
        }
    }

    async fn invalidate_m2n_plan(
        &self,
        request: Request<InvalidateM2nPlanRequest>,
    ) -> Result<Response<InvalidateM2nPlanResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;

        if let Some(entry) = state.m2n_pending.remove(&req.model_key)
            && let Some(plan_id) = entry.plan_id
        {
            // Remove all per-worker slices for this plan.
            state.m2n_worker_plans.retain(|(pid, _), _| pid != &plan_id);
        }

        Ok(Response::new(InvalidateM2nPlanResponse { ok: true }))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    const ATTEMPT_A: &str = "123e4567-e89b-42d3-a456-426614174000";
    const ATTEMPT_B: &str = "123e4567-e89b-42d3-a456-426614174001";

    fn record(attempt_id: &str, cohort_id: &str, marker: u8) -> NcclBootstrapRecord {
        NcclBootstrapRecord {
            cohort_id: cohort_id.to_string(),
            attempt_id: attempt_id.to_string(),
            nccl_unique_id: vec![marker; NCCL_UNIQUE_ID_BYTES],
            source_world_size: 2,
            destination_world_size: 2,
            world_size: 4,
            roster_digest: vec![marker; SHA256_BYTES],
        }
    }

    async fn publish(service: &WeightSyncServiceImpl, record: NcclBootstrapRecord) {
        let response = service
            .publish_nccl_bootstrap(Request::new(PublishNcclBootstrapRequest {
                record: Some(record),
            }))
            .await
            .expect("publish")
            .into_inner();
        assert!(response.ok);
    }

    async fn get(
        service: &WeightSyncServiceImpl,
        attempt_id: &str,
    ) -> GetNcclBootstrapResponse {
        service
            .get_nccl_bootstrap(Request::new(GetNcclBootstrapRequest {
                attempt_id: attempt_id.to_string(),
            }))
            .await
            .expect("get")
            .into_inner()
    }

    #[tokio::test]
    async fn nccl_bootstrap_publish_get_and_identical_retry() {
        let service = WeightSyncServiceImpl::new();
        let expected = record(ATTEMPT_A, "pp-0", 7);

        publish(&service, expected.clone()).await;
        publish(&service, expected.clone()).await;

        let response = get(&service, ATTEMPT_A).await;
        assert!(response.found);
        assert_eq!(response.record, Some(expected));
    }

    #[tokio::test]
    async fn nccl_bootstrap_unknown_attempt_is_not_found() {
        let response = get(&WeightSyncServiceImpl::new(), ATTEMPT_A).await;
        assert!(!response.found);
        assert!(response.record.is_none());
    }

    #[tokio::test]
    async fn nccl_bootstrap_attempts_and_cohorts_are_isolated() {
        let service = WeightSyncServiceImpl::new();
        let first = record(ATTEMPT_A, "pp-0", 1);
        let second = record(ATTEMPT_B, "pp-1", 2);
        publish(&service, first.clone()).await;
        publish(&service, second.clone()).await;

        assert_eq!(get(&service, ATTEMPT_A).await.record, Some(first));
        assert_eq!(get(&service, ATTEMPT_B).await.record, Some(second));
    }

    #[tokio::test]
    async fn nccl_bootstrap_same_cohort_new_attempt_is_independent() {
        let service = WeightSyncServiceImpl::new();
        let first = record(ATTEMPT_A, "pp-0", 1);
        let retry = record(ATTEMPT_B, "pp-0", 2);
        publish(&service, first.clone()).await;
        publish(&service, retry.clone()).await;

        assert_eq!(get(&service, ATTEMPT_A).await.record, Some(first));
        assert_eq!(get(&service, ATTEMPT_B).await.record, Some(retry));
    }

    #[tokio::test]
    async fn nccl_bootstrap_rejects_invalid_records() {
        let mut cases = Vec::new();

        for invalid_attempt in [
            "not-a-uuid",
            "123E4567-E89B-42D3-A456-426614174000",
            "123e4567-e89b-12d3-a456-426614174000",
        ] {
            cases.push(record(invalid_attempt, "pp-0", 1));
        }

        let mut invalid = record(ATTEMPT_A, "", 1);
        cases.push(invalid.clone());

        invalid = record(ATTEMPT_A, "pp-0", 1);
        invalid.nccl_unique_id.pop();
        cases.push(invalid.clone());

        invalid = record(ATTEMPT_A, "pp-0", 1);
        invalid.source_world_size = 0;
        cases.push(invalid.clone());

        invalid = record(ATTEMPT_A, "pp-0", 1);
        invalid.world_size = 5;
        cases.push(invalid.clone());

        invalid = record(ATTEMPT_A, "pp-0", 1);
        invalid.source_world_size = u32::MAX;
        invalid.destination_world_size = 1;
        invalid.world_size = u32::MAX;
        cases.push(invalid.clone());

        invalid = record(ATTEMPT_A, "pp-0", 1);
        invalid.roster_digest.pop();
        cases.push(invalid);

        let service = WeightSyncServiceImpl::new();
        for invalid in cases {
            let error = service
                .publish_nccl_bootstrap(Request::new(PublishNcclBootstrapRequest {
                    record: Some(invalid),
                }))
                .await
                .expect_err("invalid record");
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
        }
    }
}
