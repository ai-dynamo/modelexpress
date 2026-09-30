// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis implementation of the collective control-plane backend.
//!
//! One group's root hash, participants, digests and lane hashes are mutated
//! together by a single Lua script, and none of the key helpers below carries a
//! hash tag. That makes the whole keyspace single-slot: this backend targets
//! standalone or replicated Redis, not Redis Cluster, where those keys would
//! land on different slots and every script would fail CROSSSLOT.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use modelexpress_common::grpc::refit_collective::{
    CollectiveBootstrapFence, CollectiveGroup, CollectiveGroupMembership, CollectiveGroupSpec,
    CollectiveGroupState, CollectiveLane, CollectiveParticipant, CollectiveRole,
    CollectiveTransfer, CollectiveTransferState, CreateCollectiveTransferRequest,
    JoinCollectiveGroupRequest, LaneAssignment, LaneKind, PlanSource, PublishGroupBootstrapRequest,
    ReachCollectiveBootstrapFenceRequest, ReportCollectiveTransferRequest,
};
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Script};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{CollectiveBackend, CollectiveBackendError, CollectiveResult};
use crate::refit_collective::lanes::{Lane, LaneLayout};

const JOIN_GROUP_LUA: &str = include_str!("redis/scripts/join_collective_group.lua");
const PUBLISH_BOOTSTRAP_LUA: &str = include_str!("redis/scripts/publish_group_bootstrap.lua");
const CREATE_TRANSFER_LUA: &str = include_str!("redis/scripts/create_collective_transfer.lua");
const REPORT_TRANSFER_LUA: &str = include_str!("redis/scripts/report_collective_transfer.lua");
const REFRESH_GROUP_LUA: &str = include_str!("redis/scripts/refresh_collective_group.lua");
const DELETE_TRANSFER_LUA: &str = include_str!("redis/scripts/delete_collective_transfer.lua");
const REACH_BOOTSTRAP_FENCE_LUA: &str =
    include_str!("redis/scripts/reach_collective_bootstrap_fence.lua");

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

fn fence_key(group_id: &str, lane_id: u32) -> String {
    format!("mx:refitc:group:{group_id}:fence:{lane_id}")
}

const OPERATION_KEY_PREFIX: &str = "mx:refitc:op:";

fn operation_key(operation_id: &str) -> String {
    format!("{OPERATION_KEY_PREFIX}{operation_id}")
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

    // Lane order is normalized for the same reason the slot lists are: two
    // participants declaring the same membership in a different vector order
    // would otherwise resolve two groups, each stuck below its expected count.
    // Slot order WITHIN a lane stays significant - it is the rank assignment.
    let mut lanes: Vec<&_> = spec.lanes.iter().collect();
    lanes.sort_by_key(|lane| (lane.lane_id, lane.kind));

    let mut hasher = Sha256::new();
    hasher.update(spec.model_name.as_bytes());
    hasher.update([0]);
    for lane in lanes {
        hasher.update(lane.lane_id.to_le_bytes());
        hasher.update(lane.kind.to_le_bytes());
        for slot in lane.trainer_slots.iter().chain(lane.generator_slots.iter()) {
            hasher.update(slot.as_bytes());
            hasher.update([0]);
        }
        hasher.update([2]);
    }
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
        .as_chunks::<2>()
        .0
        .iter()
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

/// The declared lane set, flattened for the group hash: one line per lane,
/// `lane_id|kind|trainer_slots|generator_slots`, slot lists comma separated.
/// Slot ids in this path are role-ordinal strings, so none of the three
/// separators can occur inside one.
fn encode_lanes(lanes: &[Lane]) -> String {
    lanes
        .iter()
        .map(|lane| {
            format!(
                "{}|{}|{}|{}",
                lane.lane_id,
                i32::from(lane.kind),
                lane.trainer_slots.join(","),
                lane.generator_slots.join(",")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_lanes(text: &str) -> CollectiveResult<Vec<Lane>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split('\n')
        .map(|line| {
            let mut parts = line.split('|');
            let lane_id: u32 = parts.next().unwrap_or_default().parse().map_err(|error| {
                CollectiveBackendError::Internal(format!("invalid stored lane_id: {error}"))
            })?;
            let kind_code: i32 = parts.next().unwrap_or_default().parse().map_err(|error| {
                CollectiveBackendError::Internal(format!("invalid stored lane kind: {error}"))
            })?;
            let kind = LaneKind::try_from(kind_code).map_err(|_| {
                CollectiveBackendError::Internal(format!("unknown stored lane kind {kind_code}"))
            })?;
            let trainer_slots = split_csv(parts.next().unwrap_or_default());
            let generator_slots = split_csv(parts.next().unwrap_or_default());
            if parts.next().is_some() {
                return Err(CollectiveBackendError::Internal(
                    "stored lane record has extra fields".to_string(),
                ));
            }
            Ok(Lane {
                lane_id,
                kind,
                trainer_slots,
                generator_slots,
            })
        })
        .collect()
}

fn lanes_from_spec(spec: &CollectiveGroupSpec) -> Vec<Lane> {
    spec.lanes
        .iter()
        .map(|lane| Lane {
            lane_id: lane.lane_id,
            kind: LaneKind::try_from(lane.kind).unwrap_or(LaneKind::Unspecified),
            trainer_slots: lane.trainer_slots.clone(),
            generator_slots: lane.generator_slots.clone(),
        })
        .collect()
}

fn split_csv(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split(',').map(str::to_string).collect()
}

fn split_slots(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split('\n').map(str::to_string).collect()
}

/// Parse one `worker_id|role|index_in_role|joined_epoch` record.
fn participant_from_record(slot_id: &str, record: &str) -> CollectiveResult<CollectiveParticipant> {
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
    let _: u64 = parts.next().unwrap_or_default().parse().map_err(|error| {
        CollectiveBackendError::Internal(format!("invalid participant epoch: {error}"))
    })?;
    if parts.next().is_some() {
        return Err(CollectiveBackendError::Internal(
            "participant record has extra fields".to_string(),
        ));
    }

    Ok(CollectiveParticipant {
        slot_id: slot_id.to_string(),
        worker_id,
        role: role.into(),
        index_in_role,
        rank_in_lane: 0,
    })
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
        LaneLayout::new(
            lanes_from_spec(spec),
            &spec.expected_trainer_slots,
            &spec.expected_generator_slots,
        )
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
        let trainer_slots = split_slots(field(&fields, "expected_trainer_slots")?);
        let generator_slots = split_slots(field(&fields, "expected_generator_slots")?);
        let layout = LaneLayout::new(
            decode_lanes(field(&fields, "lanes")?)?,
            &trainer_slots,
            &generator_slots,
        )
        .map_err(|error| CollectiveBackendError::Internal(error.to_string()))?;

        let records: HashMap<String, String> = connection
            .hgetall(participants_key(group_id))
            .await
            .map_err(redis_error)?;

        // Place every participant in one pass. Rebuilding the assignment per
        // lane instead would parse and re-assign the whole membership
        // `lane_count` times, and `publish_bootstrap` reads a group twice.
        let mut lane_participants: HashMap<u32, Vec<CollectiveParticipant>> = layout
            .lanes()
            .iter()
            .map(|lane| (lane.lane_id, Vec::new()))
            .collect();
        for (slot_id, record) in &records {
            let participant = participant_from_record(slot_id, record)?;
            let assignments = layout.assign(slot_id).map_err(|error| {
                CollectiveBackendError::Internal(format!(
                    "stored participant {slot_id} has an invalid lane assignment: {error}"
                ))
            })?;
            for assignment in assignments {
                if let Some(lane) = lane_participants.get_mut(&assignment.lane_id) {
                    let mut placed = participant.clone();
                    placed.rank_in_lane = assignment.rank_in_lane;
                    lane.push(placed);
                }
            }
        }
        for participants in lane_participants.values_mut() {
            participants.sort_by_key(|p| p.rank_in_lane);
        }

        let mut lane_pipe = redis::pipe();
        for lane in layout.lanes() {
            lane_pipe.hgetall(lane_key(group_id, lane.lane_id));
        }
        let lane_hashes: Vec<HashMap<String, String>> = lane_pipe
            .query_async(&mut connection)
            .await
            .map_err(redis_error)?;

        let mut lanes: Vec<CollectiveLane> = Vec::with_capacity(layout.lanes().len());
        for (position, declared) in layout.lanes().iter().enumerate() {
            let lane_id = declared.lane_id;
            let kind = declared.kind;
            let world_size = declared.world_size();
            let participants = lane_participants.remove(&lane_id).unwrap_or_default();
            let empty = HashMap::new();
            let lane_fields = lane_hashes.get(position).unwrap_or(&empty);
            let nccl_unique_id = match lane_fields.get("nccl_unique_id") {
                Some(text) => hex_decode(text)?,
                None => Vec::new(),
            };
            let bootstrap_epoch: u64 = lane_fields
                .get("bootstrap_epoch")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);

            lanes.push(CollectiveLane {
                lane_id,
                kind: kind.into(),
                world_size,
                nccl_unique_id,
                bootstrap_epoch,
                participants,
            });
        }

        // The digests hash already holds every slot's reported digest; the
        // group only ever stored the latest. Surfacing the disagreement is the
        // difference between a diagnosable cohort split and an unexplained
        // FORMING that never resolves.
        let group_digest = field(&fields, "plan_digest")?.to_string();
        let reported: HashMap<String, String> = connection
            .hgetall(digests_key(group_id))
            .await
            .map_err(redis_error)?;
        let mut disagreeing_slots: Vec<String> = reported
            .into_iter()
            .filter(|(_, digest)| *digest != group_digest)
            .map(|(slot_id, _)| slot_id)
            .collect();
        disagreeing_slots.sort();

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
            plan_digest: group_digest,
            expected_trainer_slots: trainer_slots,
            expected_generator_slots: generator_slots,
            created_at_unix_ms: parse_field(&fields, "created_at_unix_ms")?,
            disagreeing_slots,
            requires_bootstrap_fence: fields
                .get("requires_bootstrap_fence")
                .is_some_and(|value| value == "1"),
        })
    }

    async fn refresh_group_state(&self, group_id: &str) -> CollectiveResult<()> {
        let mut connection = self.connection.clone();
        let stored: Option<String> = connection
            .hget(group_key(group_id), "lanes")
            .await
            .map_err(redis_error)?;
        let Some(stored) = stored else {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective group {group_id} was not found"
            )));
        };
        let lanes = decode_lanes(&stored)?;

        let refresh_script = Script::new(REFRESH_GROUP_LUA);
        let mut script = refresh_script.prepare_invoke();
        script.key(group_key(group_id));
        script.key(participants_key(group_id));
        script.key(digests_key(group_id));
        for lane in &lanes {
            script.key(lane_key(group_id, lane.lane_id));
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
            .assign(&request.slot_id)
            .map_err(|error| CollectiveBackendError::InvalidArgument(error.to_string()))?;

        let group_id = group_id_for(spec);
        let mut keys = vec![
            group_key(&group_id),
            participants_key(&group_id),
            digests_key(&group_id),
            worker_key(&request.worker_id),
        ];
        for lane in layout.lanes() {
            keys.push(lane_key(&group_id, lane.lane_id));
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
            .arg(encode_lanes(layout.lanes()))
            .arg(expected_total)
            .arg(&request.slot_id)
            .arg(&request.worker_id)
            .arg(role_text)
            .arg(request.index_in_role)
            .arg(&request.plan_digest)
            .arg(&plan_source.worker_id)
            .arg(&plan_source.endpoint)
            .arg(&plan_source.digest)
            .arg(spec.expected_trainer_slots.join("\n"))
            .arg(spec.expected_generator_slots.join("\n"))
            .arg(now_unix_ms()?)
            .arg(if spec.requires_bootstrap_fence {
                "1"
            } else {
                "0"
            })
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
                    "only a live trainer at index 0 may advertise a plan source, and only \
                     under its own worker_id"
                        .to_string(),
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
            "CONFLICTING_FENCE_REQUIREMENT" => {
                return Err(CollectiveBackendError::FailedPrecondition(format!(
                    "requires_bootstrap_fence={} conflicts with collective group {group_id}, \
                     which was formed with the opposite requirement; every participant must \
                     declare the same value",
                    spec.requires_bootstrap_fence
                )));
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
            is_bootstrap_leader: layout.is_bootstrap_leader(&request.slot_id),
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
        // Lane ids are whatever the caller declared, so membership is the
        // test, not a range check against the count.
        if !group
            .lanes
            .iter()
            .any(|lane| lane.lane_id == request.lane_id)
        {
            return Err(CollectiveBackendError::InvalidArgument(format!(
                "lane {} is not declared by this group",
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
        for lane in &group.lanes {
            script.key(lane_key(&request.group_id, lane.lane_id));
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

    async fn reach_bootstrap_fence(
        &self,
        request: &ReachCollectiveBootstrapFenceRequest,
    ) -> CollectiveResult<CollectiveBootstrapFence> {
        let outcome: String = Script::new(REACH_BOOTSTRAP_FENCE_LUA)
            .key(group_key(&request.group_id))
            .key(participants_key(&request.group_id))
            .key(fence_key(&request.group_id, request.lane_id))
            .arg(request.epoch)
            .arg(request.lane_id)
            .arg(request.phase)
            .arg(&request.slot_id)
            .arg(&request.worker_id)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;

        match outcome.as_str() {
            "NOTFOUND" => Err(CollectiveBackendError::NotFound(format!(
                "collective group {} was not found",
                request.group_id
            ))),
            "NOTREADY" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective group {} is not READY for bootstrap fence {}",
                request.group_id, request.lane_id
            ))),
            "NOTADMITTED" | "NOTLIVE" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "worker {} is not the live admitted generation for slot {}",
                request.worker_id, request.slot_id
            ))),
            "NOLANE" => Err(CollectiveBackendError::InvalidArgument(format!(
                "lane {} is not declared by group {}",
                request.lane_id, request.group_id
            ))),
            "NOPHASE" => Err(CollectiveBackendError::InvalidArgument(
                "bootstrap fence phase must be specified".to_string(),
            )),
            other => {
                if let Some(lane_id) = other.strip_prefix("PREINCOMPLETE:") {
                    return Err(CollectiveBackendError::FailedPrecondition(format!(
                        "bootstrap completion for group {} was rejected because lane {lane_id} has not released its PRE_BARRIER fence",
                        request.group_id
                    )));
                }
                if let Some(current) = other.strip_prefix("STALE:") {
                    return Err(CollectiveBackendError::FailedPrecondition(format!(
                        "bootstrap fence for epoch {} was rejected; the group is at epoch {current}",
                        request.epoch
                    )));
                }
                let Some(payload) = other.strip_prefix("OK:") else {
                    return Err(CollectiveBackendError::Internal(format!(
                        "unexpected bootstrap fence outcome {other}"
                    )));
                };
                let (released, missing) = payload.split_once(':').ok_or_else(|| {
                    CollectiveBackendError::Internal(
                        "bootstrap fence outcome is missing release state".to_string(),
                    )
                })?;
                Ok(CollectiveBootstrapFence {
                    group_id: request.group_id.clone(),
                    epoch: request.epoch,
                    lane_id: request.lane_id,
                    released: released == "1",
                    missing_slots: if missing.is_empty() {
                        Vec::new()
                    } else {
                        missing.lines().map(str::to_string).collect()
                    },
                    phase: request.phase,
                })
            }
        }
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
            .arg(OPERATION_KEY_PREFIX)
            .arg(if spec.requires_bootstrap_fence {
                "1"
            } else {
                "0"
            })
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
            "FENCEMISMATCH" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "requires_bootstrap_fence={} conflicts with collective group {group_id}, \
                 which was formed with the opposite requirement; every participant must \
                 declare the same value",
                spec.requires_bootstrap_fence
            ))),
            "NOTBOOTSTRAPPED" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective group {group_id} has not completed all-rank bootstrap"
            ))),
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
    use modelexpress_common::grpc::refit_collective::{BootstrapFencePhase, LaneSpec};

    /// One reshard lane per `lane_count`, splitting the trainers evenly across
    /// them, plus a broadcast lane spanning everyone. The split is the TEST's
    /// choice: nothing in MX computes it.
    fn spec(
        model: &str,
        trainers: &[&str],
        generators: &[&str],
        lane_count: u32,
    ) -> CollectiveGroupSpec {
        let trainer_slots: Vec<String> = trainers.iter().map(|s| (*s).to_string()).collect();
        let generator_slots: Vec<String> = generators.iter().map(|s| (*s).to_string()).collect();
        let per_lane = trainer_slots.len().div_ceil(lane_count.max(1) as usize);
        let mut lanes: Vec<LaneSpec> = trainer_slots
            .chunks(per_lane.max(1))
            .enumerate()
            .map(|(index, chunk)| LaneSpec {
                lane_id: u32::try_from(index).unwrap_or(0),
                kind: LaneKind::Reshard.into(),
                trainer_slots: chunk.to_vec(),
                generator_slots: generator_slots.clone(),
            })
            .collect();
        lanes.push(LaneSpec {
            lane_id: u32::try_from(lanes.len()).unwrap_or(0),
            kind: LaneKind::Broadcast.into(),
            trainer_slots: trainer_slots.clone(),
            generator_slots: generator_slots.clone(),
        });
        CollectiveGroupSpec {
            model_name: model.to_string(),
            expected_trainer_slots: trainer_slots,
            expected_generator_slots: generator_slots,
            lanes,
            requires_bootstrap_fence: false,
        }
    }

    #[test]
    fn group_id_is_stable_under_membership_ordering() {
        // Workers enumerate their peers in whatever order the framework hands
        // them over; declaring the same MEMBERSHIP must still land on one
        // group, so the expected-slot lists are order insensitive.
        let mut a = spec("m", &["t0", "t1"], &["g0", "g1"], 1);
        let mut b = spec("m", &["t0", "t1"], &["g0", "g1"], 1);
        a.expected_trainer_slots.reverse();
        b.expected_generator_slots.reverse();
        assert_eq!(group_id_for(&a), group_id_for(&b));
    }

    #[test]
    fn group_id_separates_lanes_that_order_their_ranks_differently() {
        // Lane order is NOT membership: it is the rank assignment. Two callers
        // that put different slots at rank 0 need different communicators, so
        // they must not resolve to the same group and silently disagree.
        let a = spec("m", &["t0", "t1"], &["g0"], 1);
        let mut b = spec("m", &["t0", "t1"], &["g0"], 1);
        for lane in &mut b.lanes {
            lane.trainer_slots.reverse();
        }
        assert_ne!(group_id_for(&a), group_id_for(&b));
    }

    #[test]
    fn group_id_is_stable_under_lane_declaration_order() {
        // The lane VECTOR's order is not membership either. Two participants
        // that list the same lanes in a different order must resolve one
        // group; resolving two would leave each below its expected count and
        // report that as a missing-slot timeout.
        let a = spec("m", &["t0", "t1"], &["g0"], 2);
        let mut b = spec("m", &["t0", "t1"], &["g0"], 2);
        b.lanes.reverse();
        assert_eq!(group_id_for(&a), group_id_for(&b));
    }

    #[test]
    fn group_id_separates_different_lane_counts() {
        assert_ne!(
            group_id_for(&spec("m", &["t0", "t1"], &["g0"], 1)),
            group_id_for(&spec("m", &["t0", "t1"], &["g0"], 2))
        );
    }

    #[test]
    fn group_id_ignores_the_fence_requirement() {
        // The requirement must not split group identity: a cohort that
        // disagrees about it is rejected at join with a clear error, not split
        // into two groups that each wait forever for their missing half.
        let mut fenced = spec("m", &["t0", "t1"], &["g0"], 1);
        fenced.requires_bootstrap_fence = true;
        assert_eq!(
            group_id_for(&spec("m", &["t0", "t1"], &["g0"], 1)),
            group_id_for(&fenced)
        );
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
        let trainer = participant_from_record("t0", "w1|TRAINER|3|7").expect("trainer record");
        assert_eq!(trainer.worker_id, "w1");
        assert_eq!(trainer.index_in_role, 3);

        let generator = participant_from_record("g0", "w2|GENERATOR|0|7").expect("generator");
        assert_eq!(generator.worker_id, "w2");

        assert!(participant_from_record("x", "w|BOGUS|0|7").is_err());
        // A record carrying the retired partition component has one field too
        // many and must be refused rather than silently re-parsed.
        assert!(participant_from_record("x", "w|TRAINER|3|1|7").is_err());
    }

    #[test]
    fn the_declared_lane_set_round_trips_through_the_group_hash() {
        let lanes = vec![
            Lane {
                lane_id: 0,
                kind: LaneKind::Reshard,
                trainer_slots: vec!["t0".to_string(), "t1".to_string()],
                generator_slots: vec!["g0".to_string()],
            },
            Lane {
                lane_id: 9,
                kind: LaneKind::Broadcast,
                trainer_slots: vec!["t0".to_string(), "t1".to_string()],
                generator_slots: vec!["g0".to_string()],
            },
        ];
        let decoded = decode_lanes(&encode_lanes(&lanes)).expect("round trip");
        assert_eq!(decoded, lanes);
        assert_eq!(decode_lanes("").expect("empty"), Vec::new());
        assert!(decode_lanes("0|1|t0|g0|extra").is_err());
        assert!(decode_lanes("notanumber|1|t0|g0").is_err());
    }

    #[tokio::test]
    #[ignore = "requires MX_TEST_REDIS_URL pointing at an isolated Redis"]
    async fn bootstrap_fence_is_idempotent_epoch_fenced_and_reset_on_membership_change() {
        let url = std::env::var("MX_TEST_REDIS_URL")
            .expect("MX_TEST_REDIS_URL must point at an isolated Redis");
        let backend = RedisCollectiveBackend::connect(&url)
            .await
            .expect("connect");
        let mut redis = backend.connection.clone();
        redis::cmd("FLUSHDB")
            .query_async::<()>(&mut redis)
            .await
            .expect("flush");
        for (worker_id, role) in [
            ("w-t0", 1),
            ("w-t1", 1),
            ("w-t1-replacement", 1),
            ("w-g0", 2),
        ] {
            redis::cmd("HSET")
                .arg(worker_key(worker_id))
                .arg("worker_id")
                .arg(worker_id)
                .arg("role")
                .arg(role)
                .arg("model_name")
                .arg("m")
                .query_async::<()>(&mut redis)
                .await
                .expect("register worker");
            redis::cmd("EXPIRE")
                .arg(worker_key(worker_id))
                .arg(60)
                .query_async::<()>(&mut redis)
                .await
                .expect("expire registration");
        }

        // The group opts in so the create gate applies: the gate is opt-in per
        // group, and an undeclared group keeps the historical ungated creates.
        let mut group_spec = spec("m", &["t0", "t1"], &["g0"], 1);
        group_spec.requires_bootstrap_fence = true;
        // One rank index per slot: the join script rejects two slots of the
        // same role sharing an index_in_role as DUPLICATE_RANK.
        let join = |slot_id: &str,
                    worker_id: &str,
                    role: CollectiveRole,
                    index_in_role: u32|
         -> JoinCollectiveGroupRequest {
            JoinCollectiveGroupRequest {
                spec: Some(group_spec.clone()),
                slot_id: slot_id.to_string(),
                worker_id: worker_id.to_string(),
                role: role.into(),
                index_in_role,
                plan_digest: "digest".to_string(),
                plan_source: (slot_id == "t0").then(|| PlanSource {
                    worker_id: worker_id.to_string(),
                    endpoint: "trainer:9000".to_string(),
                    digest: "digest".to_string(),
                }),
            }
        };
        let trainer = join("t0", "w-t0", CollectiveRole::Trainer, 0);
        let trainer_one = join("t1", "w-t1", CollectiveRole::Trainer, 1);
        let generator = join("g0", "w-g0", CollectiveRole::Generator, 0);
        let first = backend.join_group(&trainer).await.expect("trainer join");
        backend
            .join_group(&trainer_one)
            .await
            .expect("second trainer join");
        backend
            .join_group(&generator)
            .await
            .expect("generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: first.group_id.clone(),
                    epoch: first.epoch,
                    lane_id,
                    worker_id: "w-t0".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id).unwrap_or(0); 128],
                })
                .await
                .expect("publish");
        }

        let trainer_arrival = ReachCollectiveBootstrapFenceRequest {
            group_id: first.group_id.clone(),
            epoch: first.epoch,
            lane_id: 1,
            slot_id: "t0".to_string(),
            worker_id: "w-t0".to_string(),
            phase: BootstrapFencePhase::PreBarrier.into(),
        };
        let waiting = backend
            .reach_bootstrap_fence(&trainer_arrival)
            .await
            .expect("first arrival");
        assert!(!waiting.released);
        assert_eq!(waiting.missing_slots, ["g0", "t1"]);
        assert_eq!(
            backend
                .reach_bootstrap_fence(&trainer_arrival)
                .await
                .expect("idempotent arrival"),
            waiting
        );

        let still_waiting = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                group_id: first.group_id.clone(),
                epoch: first.epoch,
                lane_id: 1,
                slot_id: "g0".to_string(),
                worker_id: "w-g0".to_string(),
                phase: BootstrapFencePhase::PreBarrier.into(),
            })
            .await
            .expect("generator arrival");
        assert!(!still_waiting.released);
        assert_eq!(still_waiting.missing_slots, ["t1"]);
        let released = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                group_id: first.group_id.clone(),
                epoch: first.epoch,
                lane_id: 1,
                slot_id: "t1".to_string(),
                worker_id: "w-t1".to_string(),
                phase: BootstrapFencePhase::PreBarrier.into(),
            })
            .await
            .expect("last arrival");
        assert!(released.released);
        assert!(released.missing_slots.is_empty());

        let wrong_generation = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                worker_id: "w-t1".to_string(),
                ..trainer_arrival.clone()
            })
            .await
            .expect_err("wrong slot generation must be rejected");
        assert!(matches!(
            wrong_generation,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        let unknown_lane = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                lane_id: 99,
                ..trainer_arrival.clone()
            })
            .await
            .expect_err("undeclared lane must be rejected");
        assert!(matches!(
            unknown_lane,
            CollectiveBackendError::InvalidArgument(_)
        ));
        let stale = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                epoch: first.epoch + 1,
                ..trainer_arrival.clone()
            })
            .await
            .expect_err("future epoch must be rejected");
        assert!(matches!(
            stale,
            CollectiveBackendError::FailedPrecondition(_)
        ));

        // A membership change moves the epoch and wipes the fence, so the
        // reset is driven by a replacement join.
        let moved = backend
            .join_group(&join("t1", "w-t1-replacement", CollectiveRole::Trainer, 1))
            .await
            .expect("replacement trainer join moves the epoch");
        assert_eq!(moved.epoch, first.epoch + 1);
        assert_eq!(moved.state, i32::from(CollectiveGroupState::Forming));
        let fence_exists: bool = redis
            .exists(fence_key(&first.group_id, 1))
            .await
            .expect("fence existence");
        assert!(!fence_exists);
        let old_arrival = backend
            .reach_bootstrap_fence(&trainer_arrival)
            .await
            .expect_err("old arrival must not enter the new epoch");
        assert!(matches!(
            old_arrival,
            CollectiveBackendError::FailedPrecondition(_)
        ));

        let retry_trainer = backend.join_group(&trainer).await.expect("retry trainer");
        let retry_generator = backend
            .join_group(&generator)
            .await
            .expect("retry generator");
        assert_eq!(retry_trainer.epoch, moved.epoch);
        assert_eq!(retry_generator.epoch, moved.epoch);
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: first.group_id.clone(),
                    epoch: moved.epoch,
                    lane_id,
                    worker_id: "w-t0".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 2).unwrap_or(0); 128],
                })
                .await
                .expect("retry publish");
        }
        let completion_arrival = || ReachCollectiveBootstrapFenceRequest {
            group_id: first.group_id.clone(),
            epoch: moved.epoch,
            lane_id: 1,
            slot_id: "t0".to_string(),
            worker_id: "w-t0".to_string(),
            phase: BootstrapFencePhase::Complete.into(),
        };
        let complete_before_pre = backend
            .reach_bootstrap_fence(&completion_arrival())
            .await
            .expect_err("COMPLETE before any PRE_BARRIER release must be rejected");
        assert!(matches!(
            complete_before_pre,
            CollectiveBackendError::FailedPrecondition(_)
        ));

        let retry_waiting = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                epoch: moved.epoch,
                ..trainer_arrival
            })
            .await
            .expect("retry arrival");
        assert!(!retry_waiting.released);
        assert_eq!(retry_waiting.missing_slots, ["g0", "t1"]);
        let complete_during_partial_pre = backend
            .reach_bootstrap_fence(&completion_arrival())
            .await
            .expect_err("COMPLETE during a partial PRE_BARRIER must be rejected");
        assert!(matches!(
            complete_during_partial_pre,
            CollectiveBackendError::FailedPrecondition(_)
        ));

        let before_completion = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(group_spec.clone()),
                version_id: "v0".to_string(),
                idempotency_key: "before-completion".to_string(),
            })
            .await
            .expect_err("READY without all-rank completion must reject create");
        assert!(matches!(
            before_completion,
            CollectiveBackendError::FailedPrecondition(_)
        ));

        for (slot_id, worker_id) in [("g0", "w-g0"), ("t1", "w-t1-replacement")] {
            let released = backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: moved.epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::PreBarrier.into(),
                })
                .await
                .expect("finish retry pre-barrier fence");
            if slot_id == "t1" {
                assert!(released.released);
            }
        }
        let complete_with_missing_lane = backend
            .reach_bootstrap_fence(&completion_arrival())
            .await
            .expect_err("COMPLETE with an unreleased declared lane must be rejected");
        assert!(matches!(
            complete_with_missing_lane,
            CollectiveBackendError::FailedPrecondition(_)
        ));

        let lane_zero_partial = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                group_id: first.group_id.clone(),
                epoch: moved.epoch,
                lane_id: 0,
                slot_id: "t0".to_string(),
                worker_id: "w-t0".to_string(),
                phase: BootstrapFencePhase::PreBarrier.into(),
            })
            .await
            .expect("partial lane zero PRE_BARRIER");
        assert!(!lane_zero_partial.released);
        let complete_with_partial_lane = backend
            .reach_bootstrap_fence(&completion_arrival())
            .await
            .expect_err("COMPLETE with a partial declared lane must be rejected");
        assert!(matches!(
            complete_with_partial_lane,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        for (slot_id, worker_id) in [("g0", "w-g0"), ("t1", "w-t1-replacement")] {
            let released = backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: moved.epoch,
                    lane_id: 0,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::PreBarrier.into(),
                })
                .await
                .expect("finish lane zero PRE_BARRIER");
            if slot_id == "t1" {
                assert!(released.released);
            }
        }
        let mut completion = None;
        for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0"), ("t1", "w-t1-replacement")] {
            completion = Some(
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: first.group_id.clone(),
                        epoch: moved.epoch,
                        lane_id: 1,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::Complete.into(),
                    })
                    .await
                    .expect("bootstrap completion arrival"),
            );
        }
        assert!(completion.as_ref().is_some_and(|fence| fence.released));
        let replayed_completion = backend
            .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                group_id: first.group_id.clone(),
                epoch: moved.epoch,
                lane_id: 1,
                slot_id: "t1".to_string(),
                worker_id: "w-t1-replacement".to_string(),
                phase: BootstrapFencePhase::Complete.into(),
            })
            .await
            .expect("lost completion response is idempotent");
        assert!(replayed_completion.released);

        let bootstrap_complete_epoch: u64 = redis
            .hget(group_key(&first.group_id), "bootstrap_complete_epoch")
            .await
            .expect("read bootstrap completion epoch");
        assert_eq!(bootstrap_complete_epoch, moved.epoch);

        backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(group_spec.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "after-completion".to_string(),
            })
            .await
            .expect("a fenced group creates once the fence completes");
    }

    #[tokio::test]
    #[ignore = "requires MX_TEST_REDIS_URL pointing at an isolated Redis"]
    async fn the_bootstrap_fence_gate_is_opt_in_per_group() {
        let url = std::env::var("MX_TEST_REDIS_URL")
            .expect("MX_TEST_REDIS_URL must point at an isolated Redis");
        let backend = RedisCollectiveBackend::connect(&url)
            .await
            .expect("connect");
        let mut redis = backend.connection.clone();
        redis::cmd("FLUSHDB")
            .query_async::<()>(&mut redis)
            .await
            .expect("flush");
        for (worker_id, role, model) in [
            ("w-t0", 1, "m"),
            ("w-g0", 2, "m"),
            ("w-t1", 1, "m2"),
            ("w-g1", 2, "m2"),
            ("w-g0-new", 2, "m"),
        ] {
            redis::cmd("HSET")
                .arg(worker_key(worker_id))
                .arg("worker_id")
                .arg(worker_id)
                .arg("role")
                .arg(role)
                .arg("model_name")
                .arg(model)
                .query_async::<()>(&mut redis)
                .await
                .expect("register worker");
            redis::cmd("EXPIRE")
                .arg(worker_key(worker_id))
                .arg(60)
                .query_async::<()>(&mut redis)
                .await
                .expect("expire registration");
        }

        let join = |group_spec: &CollectiveGroupSpec,
                    slot_id: &str,
                    worker_id: &str,
                    role: CollectiveRole| JoinCollectiveGroupRequest {
            spec: Some(group_spec.clone()),
            slot_id: slot_id.to_string(),
            worker_id: worker_id.to_string(),
            role: role.into(),
            index_in_role: 0,
            plan_digest: "digest".to_string(),
            plan_source: None,
        };

        // A group that never declared the fence creates without a single fence
        // arrival: the pre-fence client behavior an upgraded server must keep.
        let unfenced = spec("m", &["t0"], &["g0"], 1);
        backend
            .join_group(&join(&unfenced, "t0", "w-t0", CollectiveRole::Trainer))
            .await
            .expect("unfenced trainer join");
        let unfenced_membership = backend
            .join_group(&join(&unfenced, "g0", "w-g0", CollectiveRole::Generator))
            .await
            .expect("unfenced generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: unfenced_membership.group_id.clone(),
                    epoch: unfenced_membership.epoch,
                    lane_id,
                    worker_id: "w-t0".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id).unwrap_or(0); 128],
                })
                .await
                .expect("unfenced publish");
        }
        let created = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(unfenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "unfenced".to_string(),
            })
            .await
            .expect("an unfenced group creates without reaching the fence");
        assert_eq!(created.epoch, unfenced_membership.epoch);
        let unfenced_group = backend
            .read_group(&unfenced_membership.group_id)
            .await
            .expect("read unfenced group");
        assert!(!unfenced_group.requires_bootstrap_fence);

        // A replay that flips only the fence flag hits the idempotency
        // reservation, but still answers to the group's declared requirement.
        let mut flipped = unfenced.clone();
        flipped.requires_bootstrap_fence = true;
        let flipped_replay = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(flipped),
                version_id: "v1".to_string(),
                idempotency_key: "unfenced".to_string(),
            })
            .await
            .expect_err("a replay that flips the fence flag is refused");
        assert!(matches!(
            flipped_replay,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        assert!(
            flipped_replay
                .to_string()
                .contains("requires_bootstrap_fence")
        );

        let replayed = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(unfenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "unfenced".to_string(),
            })
            .await
            .expect("a faithful replay returns the existing operation");
        assert_eq!(replayed.operation_id, created.operation_id);
        assert_eq!(replayed.epoch, created.epoch);

        // A replacement join (same slot, fresh worker identity) bumps the
        // epoch and wipes lane/fence state. A replay from the superseded
        // epoch no longer returns the old operation: it reclaims the
        // reservation into a fresh create that answers to the new epoch's
        // gate, and tombstones the superseded row.
        let replacement = backend
            .join_group(&join(
                &unfenced,
                "g0",
                "w-g0-new",
                CollectiveRole::Generator,
            ))
            .await
            .expect("a replacement join with a fresh worker identity");
        assert_eq!(replacement.epoch, created.epoch + 1);

        let reclaimed = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(unfenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "unfenced".to_string(),
            })
            .await
            .expect("a replay from a superseded epoch reclaims the reservation");
        assert_ne!(reclaimed.operation_id, created.operation_id);
        assert_eq!(reclaimed.epoch, replacement.epoch);
        let superseded = backend
            .read_transfer(&created.operation_id)
            .await
            .expect("the superseded operation row remains readable");
        assert!(matches!(
            CollectiveTransferState::try_from(superseded.state),
            Ok(CollectiveTransferState::Aborted)
        ));
        assert!(superseded.failure_message.contains("superseded"));

        let replayed_again = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(unfenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "unfenced".to_string(),
            })
            .await
            .expect("the reclaimed reservation replays the new operation");
        assert_eq!(replayed_again.operation_id, reclaimed.operation_id);
        assert_eq!(replayed_again.epoch, replacement.epoch);

        let mut fenced = spec("m2", &["t1"], &["g1"], 1);
        fenced.requires_bootstrap_fence = true;
        backend
            .join_group(&join(&fenced, "t1", "w-t1", CollectiveRole::Trainer))
            .await
            .expect("fenced trainer join");
        let fenced_membership = backend
            .join_group(&join(&fenced, "g1", "w-g1", CollectiveRole::Generator))
            .await
            .expect("fenced generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: fenced_membership.group_id.clone(),
                    epoch: fenced_membership.epoch,
                    lane_id,
                    worker_id: "w-t1".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 2).unwrap_or(0); 128],
                })
                .await
                .expect("fenced publish");
        }

        // A create that under-declares is refused before the gate is consulted.
        let mut under_declared = fenced.clone();
        under_declared.requires_bootstrap_fence = false;
        let mismatched_create = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(under_declared.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "mismatched".to_string(),
            })
            .await
            .expect_err("a create that under-declares the fence is refused");
        assert!(matches!(
            mismatched_create,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        assert!(
            mismatched_create
                .to_string()
                .contains("requires_bootstrap_fence")
        );

        let gated = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(fenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "gated".to_string(),
            })
            .await
            .expect_err("a fenced group rejects creates before the fence completes");
        assert!(matches!(
            gated,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        assert!(
            gated
                .to_string()
                .contains("has not completed all-rank bootstrap")
        );

        let rejected_join = backend
            .join_group(&join(
                &under_declared,
                "g1",
                "w-g1",
                CollectiveRole::Generator,
            ))
            .await
            .expect_err("a join that under-declares the fence is refused");
        assert!(matches!(
            rejected_join,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        assert!(
            rejected_join
                .to_string()
                .contains("requires_bootstrap_fence")
        );

        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t1", "w-t1"), ("g1", "w-g1")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: fenced_membership.group_id.clone(),
                        epoch: fenced_membership.epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("fenced PRE_BARRIER");
            }
        }
        for (slot_id, worker_id) in [("t1", "w-t1"), ("g1", "w-g1")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: fenced_membership.group_id.clone(),
                    epoch: fenced_membership.epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("fenced COMPLETE");
        }
        let created = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(fenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "fenced".to_string(),
            })
            .await
            .expect("a fenced group creates once the fence completes");
        assert_eq!(created.epoch, fenced_membership.epoch);
        let fenced_group = backend
            .read_group(&fenced_membership.group_id)
            .await
            .expect("read fenced group");
        assert!(fenced_group.requires_bootstrap_fence);
    }

    #[tokio::test]
    #[ignore = "requires MX_TEST_REDIS_URL pointing at an isolated Redis"]
    async fn a_superseded_epoch_replay_is_guarded_by_identity_terminal_state_and_the_fence() {
        let url = std::env::var("MX_TEST_REDIS_URL")
            .expect("MX_TEST_REDIS_URL must point at an isolated Redis");
        let backend = RedisCollectiveBackend::connect(&url)
            .await
            .expect("connect");
        let mut redis = backend.connection.clone();
        redis::cmd("FLUSHDB")
            .query_async::<()>(&mut redis)
            .await
            .expect("flush");
        for (worker_id, role, model) in [
            ("w-t2", 1, "m3"),
            ("w-g2", 2, "m3"),
            ("w-g2-new", 2, "m3"),
            ("w-t3", 1, "m4"),
            ("w-g3", 2, "m4"),
            ("w-g3-new", 2, "m4"),
            ("w-t4", 1, "m5"),
            ("w-g4", 2, "m5"),
            ("w-g4-new", 2, "m5"),
        ] {
            redis::cmd("HSET")
                .arg(worker_key(worker_id))
                .arg("worker_id")
                .arg(worker_id)
                .arg("role")
                .arg(role)
                .arg("model_name")
                .arg(model)
                .query_async::<()>(&mut redis)
                .await
                .expect("register worker");
            redis::cmd("EXPIRE")
                .arg(worker_key(worker_id))
                .arg(60)
                .query_async::<()>(&mut redis)
                .await
                .expect("expire registration");
        }

        let join = |group_spec: &CollectiveGroupSpec,
                    slot_id: &str,
                    worker_id: &str,
                    role: CollectiveRole| JoinCollectiveGroupRequest {
            spec: Some(group_spec.clone()),
            slot_id: slot_id.to_string(),
            worker_id: worker_id.to_string(),
            role: role.into(),
            index_in_role: 0,
            plan_digest: "digest".to_string(),
            plan_source: None,
        };

        // Phase A: a replay naming a different version is answered EXISTING
        // (AlreadyExists) even when the group epoch has moved; the identity
        // guard runs before any stale-epoch reclaim, so the old row is left
        // intact.
        let guard = spec("m3", &["t2"], &["g2"], 1);
        backend
            .join_group(&join(&guard, "t2", "w-t2", CollectiveRole::Trainer))
            .await
            .expect("guard trainer join");
        let guard_membership = backend
            .join_group(&join(&guard, "g2", "w-g2", CollectiveRole::Generator))
            .await
            .expect("guard generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: guard_membership.group_id.clone(),
                    epoch: guard_membership.epoch,
                    lane_id,
                    worker_id: "w-t2".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id).unwrap_or(0); 128],
                })
                .await
                .expect("guard publish");
        }
        let guard_created = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(guard.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "guard".to_string(),
            })
            .await
            .expect("guard create");
        let guard_replacement = backend
            .join_group(&join(&guard, "g2", "w-g2-new", CollectiveRole::Generator))
            .await
            .expect("guard replacement join");
        assert_eq!(guard_replacement.epoch, guard_created.epoch + 1);

        let different_version = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(guard.clone()),
                version_id: "v2".to_string(),
                idempotency_key: "guard".to_string(),
            })
            .await
            .expect_err("a different-version replay is refused even after the epoch moved");
        assert!(matches!(
            different_version,
            CollectiveBackendError::AlreadyExists(_)
        ));
        let guard_row = backend
            .read_transfer(&guard_created.operation_id)
            .await
            .expect("guard row remains readable");
        assert!(matches!(
            CollectiveTransferState::try_from(guard_row.state),
            Ok(CollectiveTransferState::Pending)
        ));
        assert!(guard_row.failure_message.is_empty());

        // Phase B: a terminal row is never tombstoned by a stale-epoch
        // reclaim — the reservation is recycled but the FAILED row keeps its
        // state and its original failure message.
        let terminal = spec("m4", &["t3"], &["g3"], 1);
        backend
            .join_group(&join(&terminal, "t3", "w-t3", CollectiveRole::Trainer))
            .await
            .expect("terminal trainer join");
        let terminal_membership = backend
            .join_group(&join(&terminal, "g3", "w-g3", CollectiveRole::Generator))
            .await
            .expect("terminal generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: terminal_membership.group_id.clone(),
                    epoch: terminal_membership.epoch,
                    lane_id,
                    worker_id: "w-t3".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 2).unwrap_or(0); 128],
                })
                .await
                .expect("terminal publish");
        }
        let terminal_created = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(terminal.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "terminal".to_string(),
            })
            .await
            .expect("terminal create");
        redis::cmd("HSET")
            .arg(operation_key(&terminal_created.operation_id))
            .arg("state")
            .arg("FAILED")
            .arg("failure_message")
            .arg("weights checksum mismatch on lane 1")
            .query_async::<()>(&mut redis)
            .await
            .expect("mark the operation failed");
        let terminal_replacement = backend
            .join_group(&join(&terminal, "g3", "w-g3-new", CollectiveRole::Generator))
            .await
            .expect("terminal replacement join");
        assert_eq!(terminal_replacement.epoch, terminal_created.epoch + 1);

        let terminal_reclaimed = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(terminal.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "terminal".to_string(),
            })
            .await
            .expect("a stale-epoch replay of a terminal row still reclaims the reservation");
        assert_ne!(
            terminal_reclaimed.operation_id,
            terminal_created.operation_id
        );
        assert_eq!(terminal_reclaimed.epoch, terminal_replacement.epoch);
        let failed_row = backend
            .read_transfer(&terminal_created.operation_id)
            .await
            .expect("terminal row remains readable");
        assert!(matches!(
            CollectiveTransferState::try_from(failed_row.state),
            Ok(CollectiveTransferState::Failed)
        ));
        assert_eq!(
            failed_row.failure_message,
            "weights checksum mismatch on lane 1"
        );

        // Phase C: a fenced group's stale-epoch replay tombstones the
        // superseded row but the fresh create is held to the NEW epoch's
        // fence — membership change reset it — until the fence completes
        // again.
        let mut fenced = spec("m5", &["t4"], &["g4"], 1);
        fenced.requires_bootstrap_fence = true;
        backend
            .join_group(&join(&fenced, "t4", "w-t4", CollectiveRole::Trainer))
            .await
            .expect("fenced trainer join");
        let fenced_membership = backend
            .join_group(&join(&fenced, "g4", "w-g4", CollectiveRole::Generator))
            .await
            .expect("fenced generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: fenced_membership.group_id.clone(),
                    epoch: fenced_membership.epoch,
                    lane_id,
                    worker_id: "w-t4".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 4).unwrap_or(0); 128],
                })
                .await
                .expect("fenced publish");
        }
        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t4", "w-t4"), ("g4", "w-g4")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: fenced_membership.group_id.clone(),
                        epoch: fenced_membership.epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("fenced PRE_BARRIER");
            }
        }
        for (slot_id, worker_id) in [("t4", "w-t4"), ("g4", "w-g4")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: fenced_membership.group_id.clone(),
                    epoch: fenced_membership.epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("fenced COMPLETE");
        }
        let fenced_created = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(fenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "fenced-stale".to_string(),
            })
            .await
            .expect("fenced create once the fence completes");
        let fenced_replacement = backend
            .join_group(&join(&fenced, "g4", "w-g4-new", CollectiveRole::Generator))
            .await
            .expect("fenced replacement join");
        assert_eq!(fenced_replacement.epoch, fenced_created.epoch + 1);

        let stale_replay = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(fenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "fenced-stale".to_string(),
            })
            .await
            .expect_err("a stale-epoch replay is held to the new epoch's fence");
        assert!(matches!(
            stale_replay,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        assert!(
            stale_replay
                .to_string()
                .contains("has not completed all-rank bootstrap")
        );
        let tombstoned = backend
            .read_transfer(&fenced_created.operation_id)
            .await
            .expect("superseded row remains readable");
        assert!(matches!(
            CollectiveTransferState::try_from(tombstoned.state),
            Ok(CollectiveTransferState::Aborted)
        ));
        assert!(tombstoned.failure_message.contains("superseded"));

        // The trainer acknowledges the new epoch, both lanes re-publish (READY
        // requires current-epoch bootstrap state), the fence re-completes, and
        // the reclaimed key opens a fresh operation in it.
        backend
            .join_group(&join(&fenced, "t4", "w-t4", CollectiveRole::Trainer))
            .await
            .expect("trainer acknowledges the new epoch");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: fenced_membership.group_id.clone(),
                    epoch: fenced_replacement.epoch,
                    lane_id,
                    worker_id: "w-t4".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 6).unwrap_or(0); 128],
                })
                .await
                .expect("re-publish at the new epoch");
        }
        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t4", "w-t4"), ("g4", "w-g4-new")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: fenced_membership.group_id.clone(),
                        epoch: fenced_replacement.epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("re-fence PRE_BARRIER");
            }
        }
        for (slot_id, worker_id) in [("t4", "w-t4"), ("g4", "w-g4-new")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: fenced_membership.group_id.clone(),
                    epoch: fenced_replacement.epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("re-fence COMPLETE");
        }
        let refenced = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(fenced.clone()),
                version_id: "v1".to_string(),
                idempotency_key: "fenced-stale".to_string(),
            })
            .await
            .expect("the reclaimed key creates once the new epoch's fence completes");
        assert_ne!(refenced.operation_id, fenced_created.operation_id);
        assert_eq!(refenced.epoch, fenced_replacement.epoch);
    }
}
