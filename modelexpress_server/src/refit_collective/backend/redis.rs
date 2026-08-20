// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis implementation of the collective control-plane backend.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use modelexpress_common::grpc::refit_collective::{
    CollectiveGroup, CollectiveGroupMembership, CollectiveGroupSpec, CollectiveGroupState,
    CollectiveLane, CollectiveParticipant, CollectiveRole, CollectiveTransfer,
    CollectiveTransferState, CreateCollectiveTransferRequest, JoinCollectiveGroupRequest,
    LaneAssignment, LaneKind, PlanSource, PublishGroupBootstrapRequest,
    ReportCollectiveTransferRequest,
};
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Script};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{CollectiveBackend, CollectiveBackendError, CollectiveResult};
use crate::refit_collective::lanes::LaneLayout;

const JOIN_GROUP_LUA: &str = include_str!("redis/scripts/join_collective_group.lua");
const PUBLISH_BOOTSTRAP_LUA: &str = include_str!("redis/scripts/publish_group_bootstrap.lua");
const CREATE_TRANSFER_LUA: &str = include_str!("redis/scripts/create_collective_transfer.lua");
const REPORT_TRANSFER_LUA: &str = include_str!("redis/scripts/report_collective_transfer.lua");
const REFRESH_GROUP_LUA: &str = include_str!("redis/scripts/refresh_collective_group.lua");
const DELETE_TRANSFER_LUA: &str = include_str!("redis/scripts/delete_collective_transfer.lua");

fn group_key(group_id: &str) -> String {
    format!("mx:refitc:group:{group_id}")
}

fn participants_key(group_id: &str) -> String {
    format!("mx:refitc:group:{group_id}:participants")
}

fn digests_key(group_id: &str) -> String {
    format!("mx:refitc:group:{group_id}:digests")
}

fn lane_key(group_id: &str, lane_id: u32) -> String {
    format!("mx:refitc:group:{group_id}:lane:{lane_id}")
}

fn operation_key(operation_id: &str) -> String {
    format!("mx:refitc:op:{operation_id}")
}

fn reported_key(operation_id: &str) -> String {
    format!("mx:refitc:op:{operation_id}:reported")
}

fn operation_idempotency_key(model_name: &str, request_key: &str) -> String {
    format!(
        "mx:refitc:op-request:{}:{model_name}{request_key}",
        model_name.len()
    )
}

fn worker_key(worker_id: &str) -> String {
    format!("mx:refit:worker:{worker_id}")
}

/// Derive a stable group identity from the declared membership.
///
/// Every participant of one operation sends an identical declaration, so they
/// all resolve the same group without a separate create call. A participant
/// that declares a different membership resolves a *different* group, which
/// then never reaches its expected count -- a bounded timeout naming the
/// missing slots, rather than one group with inconsistent geometry.
fn group_id_for(spec: &CollectiveGroupSpec) -> String {
    let mut trainers = spec.expected_trainer_slots.clone();
    let mut generators = spec.expected_generator_slots.clone();
    trainers.sort();
    generators.sort();

    let mut hasher = Sha256::new();
    hasher.update(spec.model_name.as_bytes());
    hasher.update([0]);
    hasher.update(spec.source_partition_count.to_le_bytes());
    hasher.update([0]);
    for slot in &trainers {
        hasher.update(slot.as_bytes());
        hasher.update([0]);
    }
    hasher.update([1]);
    for slot in &generators {
        hasher.update(slot.as_bytes());
        hasher.update([0]);
    }

    let digest = hasher.finalize();
    let mut id = String::with_capacity(32);
    for byte in &digest[..16] {
        let _ = write!(id, "{byte:02x}");
    }
    id
}

fn now_unix_ms() -> CollectiveResult<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| CollectiveBackendError::Internal(format!("system clock error: {error}")))?
        .as_millis();
    u64::try_from(millis).map_err(|_| {
        CollectiveBackendError::Internal("system time does not fit in uint64".to_string())
    })
}

fn redis_error(error: redis::RedisError) -> CollectiveBackendError {
    if error.is_io_error()
        || error.is_cluster_error()
        || matches!(
            error.kind(),
            redis::ErrorKind::BusyLoadingError
                | redis::ErrorKind::MasterDown
                | redis::ErrorKind::ClusterConnectionNotFound
        )
    {
        CollectiveBackendError::Unavailable(error.to_string())
    } else {
        CollectiveBackendError::Internal(error.to_string())
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn hex_decode(text: &str) -> CollectiveResult<Vec<u8>> {
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return Err(CollectiveBackendError::Internal(
            "bootstrap identifier is not valid hex".to_string(),
        ));
    }
    bytes
        .chunks_exact(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair).map_err(|error| {
                CollectiveBackendError::Internal(format!("invalid bootstrap identifier: {error}"))
            })?;
            u8::from_str_radix(digits, 16).map_err(|error| {
                CollectiveBackendError::Internal(format!("invalid bootstrap identifier: {error}"))
            })
        })
        .collect()
}

fn field<'a>(fields: &'a HashMap<String, String>, name: &str) -> CollectiveResult<&'a str> {
    fields.get(name).map(String::as_str).ok_or_else(|| {
        CollectiveBackendError::Internal(format!("collective record is missing {name}"))
    })
}

fn parse_field<T>(fields: &HashMap<String, String>, name: &str) -> CollectiveResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    field(fields, name)?.parse().map_err(|error| {
        CollectiveBackendError::Internal(format!("invalid {name} in collective record: {error}"))
    })
}

fn group_state_from_str(text: &str) -> CollectiveGroupState {
    match text {
        "READY" => CollectiveGroupState::Ready,
        "RELEASING" => CollectiveGroupState::Releasing,
        _ => CollectiveGroupState::Forming,
    }
}

fn transfer_state_from_str(text: &str) -> CollectiveTransferState {
    match text {
        "RUNNING" => CollectiveTransferState::Running,
        "COMPLETE" => CollectiveTransferState::Complete,
        "FAILED" => CollectiveTransferState::Failed,
        "ABORTED" => CollectiveTransferState::Aborted,
        _ => CollectiveTransferState::Pending,
    }
}

fn split_slots(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split('\n').map(str::to_string).collect()
}

/// Parse one `worker_id|role|index_in_role|source_partition|joined_epoch` record.
fn participant_from_record(
    slot_id: &str,
    record: &str,
) -> CollectiveResult<(CollectiveParticipant, Option<u32>)> {
    let mut parts = record.split('|');
    let worker_id = parts.next().unwrap_or_default().to_string();
    let role = match parts.next().unwrap_or_default() {
        "TRAINER" => CollectiveRole::Trainer,
        "GENERATOR" => CollectiveRole::Generator,
        other => {
            return Err(CollectiveBackendError::Internal(format!(
                "unknown participant role {other}"
            )));
        }
    };
    let index_in_role: u32 = parts.next().unwrap_or_default().parse().map_err(|error| {
        CollectiveBackendError::Internal(format!("invalid participant index: {error}"))
    })?;
    let partition = match parts.next().unwrap_or_default() {
        "" => None,
        value => Some(value.parse::<u32>().map_err(|error| {
            CollectiveBackendError::Internal(format!("invalid participant partition: {error}"))
        })?),
    };
    let _: u64 = parts.next().unwrap_or_default().parse().map_err(|error| {
        CollectiveBackendError::Internal(format!("invalid participant epoch: {error}"))
    })?;
    if parts.next().is_some() {
        return Err(CollectiveBackendError::Internal(
            "participant record has extra fields".to_string(),
        ));
    }

    Ok((
        CollectiveParticipant {
            slot_id: slot_id.to_string(),
            worker_id,
            role: role.into(),
            index_in_role,
            rank_in_lane: 0,
        },
        partition,
    ))
}

pub struct RedisCollectiveBackend {
    connection: ConnectionManager,
}

impl RedisCollectiveBackend {
    pub async fn connect(url: &str) -> CollectiveResult<Self> {
        let client = redis::Client::open(url).map_err(redis_error)?;
        let connection = ConnectionManager::new(client).await.map_err(redis_error)?;
        Ok(Self { connection })
    }

    fn layout_for(spec: &CollectiveGroupSpec) -> CollectiveResult<LaneLayout> {
        let trainers = u32::try_from(spec.expected_trainer_slots.len()).map_err(|_| {
            CollectiveBackendError::InvalidArgument("too many trainer slots".to_string())
        })?;
        let generators = u32::try_from(spec.expected_generator_slots.len()).map_err(|_| {
            CollectiveBackendError::InvalidArgument("too many generator slots".to_string())
        })?;
        LaneLayout::new(spec.source_partition_count, trainers, generators)
            .map_err(|error| CollectiveBackendError::InvalidArgument(error.to_string()))
    }

    async fn read_group_once(&self, group_id: &str) -> CollectiveResult<CollectiveGroup> {
        let mut connection = self.connection.clone();
        let fields: HashMap<String, String> = connection
            .hgetall(group_key(group_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective group {group_id} was not found"
            )));
        }

        let epoch: u64 = parse_field(&fields, "epoch")?;
        let partitions: u32 = parse_field(&fields, "source_partition_count")?;
        let trainer_slots = split_slots(field(&fields, "expected_trainer_slots")?);
        let generator_slots = split_slots(field(&fields, "expected_generator_slots")?);
        let layout = LaneLayout::new(
            partitions,
            u32::try_from(trainer_slots.len()).unwrap_or(0),
            u32::try_from(generator_slots.len()).unwrap_or(0),
        )
        .map_err(|error| CollectiveBackendError::Internal(error.to_string()))?;

        let records: HashMap<String, String> = connection
            .hgetall(participants_key(group_id))
            .await
            .map_err(redis_error)?;

        let mut lanes: Vec<CollectiveLane> = Vec::with_capacity(layout.lane_count() as usize);
        for lane_id in 0..layout.lane_count() {
            let kind = if lane_id == layout.broadcast_lane_id() {
                LaneKind::Broadcast
            } else {
                LaneKind::Reshard
            };
            let world_size = if kind == LaneKind::Broadcast {
                layout.broadcast_world_size()
            } else {
                layout.reshard_world_size()
            };
            let lane_fields: HashMap<String, String> = connection
                .hgetall(lane_key(group_id, lane_id))
                .await
                .map_err(redis_error)?;
            let nccl_unique_id = match lane_fields.get("nccl_unique_id") {
                Some(text) => hex_decode(text)?,
                None => Vec::new(),
            };
            let bootstrap_epoch: u64 = lane_fields
                .get("bootstrap_epoch")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);

            let mut participants: Vec<CollectiveParticipant> = Vec::new();
            for (slot_id, record) in &records {
                let (mut participant, partition) = participant_from_record(slot_id, record)?;
                let role = CollectiveRole::try_from(participant.role)
                    .unwrap_or(CollectiveRole::Unspecified);
                let assignments = layout
                    .assign(role, participant.index_in_role, partition)
                    .map_err(|error| {
                        CollectiveBackendError::Internal(format!(
                            "stored participant {slot_id} has an invalid lane assignment: {error}"
                        ))
                    })?;
                if let Some(assignment) = assignments.iter().find(|a| a.lane_id == lane_id) {
                    participant.rank_in_lane = assignment.rank_in_lane;
                    participants.push(participant);
                }
            }
            participants.sort_by_key(|p| p.rank_in_lane);

            lanes.push(CollectiveLane {
                lane_id,
                kind: kind.into(),
                world_size,
                nccl_unique_id,
                bootstrap_epoch,
                participants,
            });
        }

        let plan_source_worker = field(&fields, "plan_source_worker_id")?.to_string();
        let plan_source = if plan_source_worker.is_empty() {
            None
        } else {
            Some(PlanSource {
                worker_id: plan_source_worker,
                endpoint: field(&fields, "plan_source_endpoint")?.to_string(),
                digest: field(&fields, "plan_source_digest")?.to_string(),
            })
        };

        Ok(CollectiveGroup {
            group_id: group_id.to_string(),
            model_name: field(&fields, "model_name")?.to_string(),
            epoch,
            state: group_state_from_str(field(&fields, "state")?).into(),
            lanes,
            plan_source,
            plan_digest: field(&fields, "plan_digest")?.to_string(),
            expected_trainer_slots: trainer_slots,
            expected_generator_slots: generator_slots,
            created_at_unix_ms: parse_field(&fields, "created_at_unix_ms")?,
        })
    }

    async fn refresh_group_state(&self, group_id: &str) -> CollectiveResult<()> {
        let mut connection = self.connection.clone();
        let partitions: Option<u32> = connection
            .hget(group_key(group_id), "source_partition_count")
            .await
            .map_err(redis_error)?;
        let Some(partitions) = partitions else {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective group {group_id} was not found"
            )));
        };
        let lane_count = partitions.checked_add(1).ok_or_else(|| {
            CollectiveBackendError::Internal("collective lane count overflowed".to_string())
        })?;

        let refresh_script = Script::new(REFRESH_GROUP_LUA);
        let mut script = refresh_script.prepare_invoke();
        script.key(group_key(group_id));
        script.key(participants_key(group_id));
        script.key(digests_key(group_id));
        for lane_id in 0..lane_count {
            script.key(lane_key(group_id, lane_id));
        }
        let outcome: String = script
            .invoke_async(&mut connection)
            .await
            .map_err(redis_error)?;
        if outcome == "NOTFOUND" {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective group {group_id} was not found"
            )));
        }
        if !outcome.starts_with("OK:") {
            return Err(CollectiveBackendError::Internal(format!(
                "unexpected group refresh outcome {outcome}"
            )));
        }
        Ok(())
    }

    async fn read_group(&self, group_id: &str) -> CollectiveResult<CollectiveGroup> {
        // A group spans several Redis hashes. Retry if a concurrent join or
        // bootstrap changes the root record while those hashes are being read,
        // so callers never receive READY paired with another epoch's lanes.
        for _ in 0..5 {
            self.refresh_group_state(group_id).await?;
            let group = self.read_group_once(group_id).await?;
            let mut connection = self.connection.clone();
            let fields: HashMap<String, String> = connection
                .hgetall(group_key(group_id))
                .await
                .map_err(redis_error)?;
            if !fields.is_empty()
                && parse_field::<u64>(&fields, "epoch")? == group.epoch
                && i32::from(group_state_from_str(field(&fields, "state")?)) == group.state
            {
                return Ok(group);
            }
        }
        Err(CollectiveBackendError::Unavailable(format!(
            "collective group {group_id} changed while it was being read"
        )))
    }

    async fn read_transfer(&self, operation_id: &str) -> CollectiveResult<CollectiveTransfer> {
        let mut connection = self.connection.clone();
        let fields: HashMap<String, String> = connection
            .hgetall(operation_key(operation_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective transfer {operation_id} was not found"
            )));
        }
        let reported: Vec<String> = connection
            .smembers(reported_key(operation_id))
            .await
            .map_err(redis_error)?;

        Ok(CollectiveTransfer {
            operation_id: operation_id.to_string(),
            group_id: field(&fields, "group_id")?.to_string(),
            epoch: parse_field(&fields, "epoch")?,
            version_id: field(&fields, "version_id")?.to_string(),
            model_name: field(&fields, "model_name")?.to_string(),
            idempotency_key: field(&fields, "idempotency_key")?.to_string(),
            state: transfer_state_from_str(field(&fields, "state")?).into(),
            created_at_unix_ms: parse_field(&fields, "created_at_unix_ms")?,
            reported_worker_ids: reported,
            failure_message: field(&fields, "failure_message")?.to_string(),
        })
    }
}

#[async_trait]
impl CollectiveBackend for RedisCollectiveBackend {
    async fn join_group(
        &self,
        request: &JoinCollectiveGroupRequest,
    ) -> CollectiveResult<CollectiveGroupMembership> {
        let spec = request.spec.as_ref().ok_or_else(|| {
            CollectiveBackendError::InvalidArgument("spec is required".to_string())
        })?;
        let layout = Self::layout_for(spec)?;
        let role = CollectiveRole::try_from(request.role).unwrap_or(CollectiveRole::Unspecified);
        let assignments = layout
            .assign(role, request.index_in_role, request.source_partition)
            .map_err(|error| CollectiveBackendError::InvalidArgument(error.to_string()))?;

        let group_id = group_id_for(spec);
        let mut keys = vec![
            group_key(&group_id),
            participants_key(&group_id),
            digests_key(&group_id),
            worker_key(&request.worker_id),
        ];
        for lane_id in 0..layout.lane_count() {
            keys.push(lane_key(&group_id, lane_id));
        }

        let role_text = match role {
            CollectiveRole::Trainer => "TRAINER",
            CollectiveRole::Generator => "GENERATOR",
            CollectiveRole::Unspecified => {
                return Err(CollectiveBackendError::InvalidArgument(
                    "role must be specified".to_string(),
                ));
            }
        };
        let expected_total = layout.trainer_count.saturating_add(layout.generator_count);
        let plan_source = request.plan_source.clone().unwrap_or_default();

        let join_script = Script::new(JOIN_GROUP_LUA);
        let mut script = join_script.prepare_invoke();
        for key in &keys {
            script.key(key);
        }
        let outcome: String = script
            .arg(&group_id)
            .arg(&spec.model_name)
            .arg(spec.source_partition_count)
            .arg(expected_total)
            .arg(&request.slot_id)
            .arg(&request.worker_id)
            .arg(role_text)
            .arg(request.index_in_role)
            .arg(
                request
                    .source_partition
                    .map(|p| p.to_string())
                    .unwrap_or_default(),
            )
            .arg(&request.plan_digest)
            .arg(&plan_source.worker_id)
            .arg(&plan_source.endpoint)
            .arg(&plan_source.digest)
            .arg(spec.expected_trainer_slots.join("\n"))
            .arg(spec.expected_generator_slots.join("\n"))
            .arg(now_unix_ms()?)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;

        match outcome.as_str() {
            "UNEXPECTED_SLOT" => {
                return Err(CollectiveBackendError::InvalidArgument(format!(
                    "slot {} is not expected for role {role_text}",
                    request.slot_id
                )));
            }
            "UNREGISTERED" => {
                return Err(CollectiveBackendError::FailedPrecondition(format!(
                    "worker {} has no live matching registration",
                    request.worker_id
                )));
            }
            "INVALID_PLAN_SOURCE" => {
                return Err(CollectiveBackendError::InvalidArgument(
                    "plan_source must be the registered trainer coordinator endpoint".to_string(),
                ));
            }
            "CONFLICTING_ASSIGNMENT" => {
                return Err(CollectiveBackendError::AlreadyExists(format!(
                    "slot {} is already bound to a different role, ordinal, or partition",
                    request.slot_id
                )));
            }
            "DUPLICATE_RANK" => {
                return Err(CollectiveBackendError::AlreadyExists(format!(
                    "another slot already owns {role_text} index {}",
                    request.index_in_role
                )));
            }
            "DUPLICATE_WORKER" => {
                return Err(CollectiveBackendError::AlreadyExists(format!(
                    "worker {} is already admitted under another slot",
                    request.worker_id
                )));
            }
            "CORRUPT_PARTICIPANT" => {
                return Err(CollectiveBackendError::Internal(
                    "collective group contains a malformed participant record".to_string(),
                ));
            }
            _ => {}
        }

        let mut parts = outcome.split(':');
        if parts.next() != Some("OK") {
            return Err(CollectiveBackendError::Internal(format!(
                "unexpected join outcome {outcome}"
            )));
        }
        let epoch: u64 = parts
            .next()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| {
                CollectiveBackendError::Internal("join returned no epoch".to_string())
            })?;
        let state = group_state_from_str(parts.next().unwrap_or("FORMING"));

        Ok(CollectiveGroupMembership {
            group_id,
            epoch,
            assignments: assignments
                .into_iter()
                .map(|a| LaneAssignment {
                    lane_id: a.lane_id,
                    kind: a.kind.into(),
                    rank_in_lane: a.rank_in_lane,
                    world_size: a.world_size,
                })
                .collect(),
            state: state.into(),
            is_bootstrap_leader: layout.is_bootstrap_leader(role, request.index_in_role),
        })
    }

    async fn get_group(&self, group_id: &str) -> CollectiveResult<CollectiveGroup> {
        self.read_group(group_id).await
    }

    async fn publish_bootstrap(
        &self,
        request: &PublishGroupBootstrapRequest,
    ) -> CollectiveResult<CollectiveGroup> {
        let group = self.read_group(&request.group_id).await?;
        let lane_count = u32::try_from(group.lanes.len()).unwrap_or(0);
        if request.lane_id >= lane_count {
            return Err(CollectiveBackendError::InvalidArgument(format!(
                "lane {} is out of range for {lane_count} lanes",
                request.lane_id
            )));
        }
        let lane = group
            .lanes
            .iter()
            .find(|lane| lane.lane_id == request.lane_id)
            .ok_or_else(|| {
                CollectiveBackendError::Internal(format!(
                    "collective group {} is missing lane {}",
                    request.group_id, request.lane_id
                ))
            })?;
        let leader_slot = lane
            .participants
            .iter()
            .find(|participant| participant.rank_in_lane == 0)
            .map(|participant| participant.slot_id.as_str())
            .ok_or_else(|| {
                CollectiveBackendError::FailedPrecondition(format!(
                    "lane {} has no admitted rank-0 participant",
                    request.lane_id
                ))
            })?;

        let publish_script = Script::new(PUBLISH_BOOTSTRAP_LUA);
        let mut script = publish_script.prepare_invoke();
        script.key(group_key(&request.group_id));
        script.key(participants_key(&request.group_id));
        script.key(digests_key(&request.group_id));
        script.key(lane_key(&request.group_id, request.lane_id));
        for lane_id in 0..lane_count {
            script.key(lane_key(&request.group_id, lane_id));
        }

        let outcome: String = script
            .arg(request.epoch)
            .arg(&request.worker_id)
            .arg(hex_encode(&request.nccl_unique_id))
            .arg(leader_slot)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;

        if outcome == "NOTFOUND" {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective group {} was not found",
                request.group_id
            )));
        }
        if let Some(current) = outcome.strip_prefix("STALE:") {
            return Err(CollectiveBackendError::FailedPrecondition(format!(
                "bootstrap for epoch {} was rejected; the group is at epoch {current}",
                request.epoch
            )));
        }
        if outcome == "NOTLEADER" {
            return Err(CollectiveBackendError::FailedPrecondition(format!(
                "worker {} is not the live rank-0 participant for lane {}",
                request.worker_id, request.lane_id
            )));
        }
        if outcome == "CONFLICT" {
            return Err(CollectiveBackendError::AlreadyExists(format!(
                "lane {} already has a different bootstrap for epoch {}",
                request.lane_id, request.epoch
            )));
        }
        if !outcome.starts_with("OK:") {
            return Err(CollectiveBackendError::Internal(format!(
                "unexpected bootstrap outcome {outcome}"
            )));
        }

        self.read_group(&request.group_id).await
    }

    async fn create_transfer(
        &self,
        request: &CreateCollectiveTransferRequest,
    ) -> CollectiveResult<CollectiveTransfer> {
        let spec = request.spec.as_ref().ok_or_else(|| {
            CollectiveBackendError::InvalidArgument("spec is required".to_string())
        })?;
        Self::layout_for(spec)?;
        let group_id = group_id_for(spec);
        let operation_id = Uuid::new_v4().simple().to_string();

        match self.refresh_group_state(&group_id).await {
            Ok(()) | Err(CollectiveBackendError::NotFound(_)) => {}
            Err(error) => return Err(error),
        }

        let outcome: String = Script::new(CREATE_TRANSFER_LUA)
            .key(operation_key(&operation_id))
            .key(operation_idempotency_key(
                &spec.model_name,
                &request.idempotency_key,
            ))
            .key(group_key(&group_id))
            .arg(&operation_id)
            .arg(&group_id)
            .arg(&request.version_id)
            .arg(&spec.model_name)
            .arg(&request.idempotency_key)
            .arg("PENDING")
            .arg(now_unix_ms()?)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;

        match outcome.as_str() {
            "CREATED" => self.read_transfer(&operation_id).await,
            "COLLISION" => Err(CollectiveBackendError::Internal(
                "generated operation id already exists".to_string(),
            )),
            "NOGROUP" => Err(CollectiveBackendError::FailedPrecondition(
                "no collective group has formed for this membership yet".to_string(),
            )),
            other => match other.strip_prefix("EXISTING:") {
                Some(existing) => {
                    let transfer = self.read_transfer(existing).await?;
                    if transfer.group_id == group_id
                        && transfer.version_id == request.version_id
                        && transfer.model_name == spec.model_name
                        && transfer.idempotency_key == request.idempotency_key
                    {
                        Ok(transfer)
                    } else {
                        Err(CollectiveBackendError::AlreadyExists(
                            "idempotency_key was already used for a different collective transfer"
                                .to_string(),
                        ))
                    }
                }
                None => Err(CollectiveBackendError::Internal(format!(
                    "unexpected create outcome {other}"
                ))),
            },
        }
    }

    async fn get_transfer(&self, operation_id: &str) -> CollectiveResult<CollectiveTransfer> {
        self.read_transfer(operation_id).await
    }

    async fn delete_transfer(&self, operation_id: &str) -> CollectiveResult<CollectiveTransfer> {
        let transfer = self.read_transfer(operation_id).await?;
        if !matches!(
            CollectiveTransferState::try_from(transfer.state),
            Ok(CollectiveTransferState::Complete)
                | Ok(CollectiveTransferState::Failed)
                | Ok(CollectiveTransferState::Aborted)
        ) {
            return Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective transfer {operation_id} is not terminal"
            )));
        }

        let outcome: String = Script::new(DELETE_TRANSFER_LUA)
            .key(operation_key(operation_id))
            .key(reported_key(operation_id))
            .key(operation_idempotency_key(
                &transfer.model_name,
                &transfer.idempotency_key,
            ))
            .arg(operation_id)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;
        match outcome.as_str() {
            "DELETED" => Ok(transfer),
            "NOTFOUND" => Err(CollectiveBackendError::NotFound(format!(
                "collective transfer {operation_id} was not found"
            ))),
            "NOTTERMINAL" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective transfer {operation_id} is not terminal"
            ))),
            other => Err(CollectiveBackendError::Internal(format!(
                "unexpected delete outcome {other}"
            ))),
        }
    }

    async fn report_transfer(
        &self,
        request: &ReportCollectiveTransferRequest,
    ) -> CollectiveResult<CollectiveTransfer> {
        let group = self.read_group(&request.group_id).await?;
        let report_script = Script::new(REPORT_TRANSFER_LUA);
        let mut script = report_script.prepare_invoke();
        script.key(operation_key(&request.operation_id));
        script.key(reported_key(&request.operation_id));
        script.key(group_key(&request.group_id));
        script.key(participants_key(&request.group_id));
        for lane in &group.lanes {
            script.key(lane_key(&request.group_id, lane.lane_id));
        }
        let outcome: String = script
            .arg(&request.operation_id)
            .arg(&request.group_id)
            .arg(request.epoch)
            .arg(&request.worker_id)
            .arg(i32::from(request.succeeded))
            .arg(&request.message)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;

        match outcome.as_str() {
            "NOTFOUND" => Err(CollectiveBackendError::NotFound(format!(
                "collective transfer {} was not found",
                request.operation_id
            ))),
            "WRONGGROUP" => Err(CollectiveBackendError::InvalidArgument(
                "the report names a different group than the operation".to_string(),
            )),
            "NOTADMITTED" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "worker {} is not an admitted generation of this group",
                request.worker_id
            ))),
            "NOTREADY" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective group {} is not READY for reports",
                request.group_id
            ))),
            other => {
                if let Some(operation_epoch) = other.strip_prefix("OPSTALE:") {
                    return Err(CollectiveBackendError::FailedPrecondition(format!(
                        "report for epoch {} does not match the operation epoch {operation_epoch}",
                        request.epoch
                    )));
                }
                if let Some(current) = other.strip_prefix("STALE:") {
                    return Err(CollectiveBackendError::FailedPrecondition(format!(
                        "report for epoch {} was rejected; the group is at epoch {current}",
                        request.epoch
                    )));
                }
                if other.starts_with("OK:") {
                    self.read_transfer(&request.operation_id).await
                } else {
                    Err(CollectiveBackendError::Internal(format!(
                        "unexpected report outcome {other}"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn spec(
        model: &str,
        trainers: &[&str],
        generators: &[&str],
        partitions: u32,
    ) -> CollectiveGroupSpec {
        CollectiveGroupSpec {
            model_name: model.to_string(),
            expected_trainer_slots: trainers.iter().map(|s| (*s).to_string()).collect(),
            expected_generator_slots: generators.iter().map(|s| (*s).to_string()).collect(),
            source_partition_count: partitions,
        }
    }

    #[test]
    fn group_id_is_stable_under_slot_ordering() {
        // Workers enumerate their peers in whatever order the framework hands
        // them over; declaring the same set must still land on one group.
        let a = spec("m", &["t0", "t1"], &["g0", "g1"], 1);
        let b = spec("m", &["t1", "t0"], &["g1", "g0"], 1);
        assert_eq!(group_id_for(&a), group_id_for(&b));
    }

    #[test]
    fn group_id_separates_distinct_memberships() {
        let base = spec("m", &["t0", "t1"], &["g0"], 1);
        assert_ne!(
            group_id_for(&base),
            group_id_for(&spec("other", &["t0", "t1"], &["g0"], 1))
        );
        // A different admitted generator subset is a different communicator.
        assert_ne!(
            group_id_for(&base),
            group_id_for(&spec("m", &["t0", "t1"], &["g0", "g1"], 1))
        );
        assert_ne!(
            group_id_for(&base),
            group_id_for(&spec("m", &["t0", "t1"], &["g0"], 2))
        );
    }

    #[test]
    fn group_id_does_not_collide_across_the_role_boundary() {
        // Without a separator between the two slot lists, moving a name from
        // one role to the other would hash identically.
        assert_ne!(
            group_id_for(&spec("m", &["a", "b"], &["c"], 1)),
            group_id_for(&spec("m", &["a"], &["b", "c"], 1))
        );
    }

    #[test]
    fn hex_round_trips_a_bootstrap_identifier() {
        let id: Vec<u8> = (0..128u32).map(|i| u8::try_from(i).unwrap_or(0)).collect();
        assert_eq!(hex_decode(&hex_encode(&id)).expect("round trip"), id);
    }

    #[test]
    fn malformed_bootstrap_identifiers_are_rejected() {
        assert!(hex_decode("abc").is_err());
        assert!(hex_decode("zz").is_err());
    }

    #[test]
    fn participant_records_round_trip() {
        let (trainer, partition) =
            participant_from_record("t0", "w1|TRAINER|3|1|7").expect("trainer record");
        assert_eq!(trainer.worker_id, "w1");
        assert_eq!(trainer.index_in_role, 3);
        assert_eq!(partition, Some(1));

        let (generator, partition) =
            participant_from_record("g0", "w2|GENERATOR|0||7").expect("generator record");
        assert_eq!(generator.worker_id, "w2");
        assert_eq!(partition, None);

        assert!(participant_from_record("x", "w|BOGUS|0||7").is_err());
        assert!(participant_from_record("x", "w|TRAINER|0|0").is_err());
    }
}
