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
    weight_sync_service_server::WeightSyncService as WeightSyncServiceTrait,
    BuildPlanRequest, BuildPlanResponse,
    GetPlanRequest, GetPlanResponse,
    GetTrainerTableRequest, GetTrainerTableResponse,
    InvalidatePlanRequest, InvalidatePlanResponse,
    PublishInferenceTableRequest, PublishInferenceTableResponse,
    PublishTrainerTableRequest, PublishTrainerTableResponse,
    RdmaDescriptorProto,
};

use super::router::{route_regions, TrainerTableJson};

// ---------------------------------------------------------------------------
// In-memory state
// ---------------------------------------------------------------------------

/// Cached plan: the list of RDMA descriptors built from one worker's regions.
#[derive(Clone)]
struct CachedPlan {
    descriptors: Vec<RdmaDescriptorProto>,
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
}

impl WeightSyncState {
    fn new() -> Self {
        Self {
            plans: HashMap::new(),
            plan_key_to_id: HashMap::new(),
            trainer_tables: HashMap::new(),
            inference_tables: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

pub struct WeightSyncServiceImpl {
    state: Arc<RwLock<WeightSyncState>>,
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
    async fn publish_trainer_table(
        &self,
        request: Request<PublishTrainerTableRequest>,
    ) -> Result<Response<PublishTrainerTableResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;
        state.trainer_tables.insert(req.model_key.clone(), req.table_payload);
        // Invalidate any cached plans that used this model's table
        state.plan_key_to_id.retain(|k, _| !k.starts_with(&req.model_key));
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
        state.inference_tables.insert(
            (req.model_key, req.worker_rank),
            req.table_payload,
        );
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
        state.plans.insert(plan_id.clone(), CachedPlan { descriptors });
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
}
