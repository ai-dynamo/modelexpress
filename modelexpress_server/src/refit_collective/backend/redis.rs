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
    AbortCollectiveBootstrapRequest, CollectiveBootstrapFence, CollectiveGroup,
    CollectiveGroupMembership, CollectiveGroupSpec, CollectiveGroupState, CollectiveLane,
    CollectiveParticipant, CollectiveRole, CollectiveTransfer, CollectiveTransferState,
    CreateCollectiveTransferRequest, JoinCollectiveGroupRequest, LaneAssignment, LaneKind,
    PlanSource, PublishGroupBootstrapRequest, ReachCollectiveBootstrapFenceRequest,
    ReportCollectiveTransferRequest,
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
const EXPIRE_TRANSFER_LUA: &str = include_str!("redis/scripts/expire_collective_transfer.lua");
const REPORT_TRANSFER_LUA: &str = include_str!("redis/scripts/report_collective_transfer.lua");
const REFRESH_GROUP_LUA: &str = include_str!("redis/scripts/refresh_collective_group.lua");
const DELETE_TRANSFER_LUA: &str = include_str!("redis/scripts/delete_collective_transfer.lua");
const REACH_BOOTSTRAP_FENCE_LUA: &str =
    include_str!("redis/scripts/reach_collective_bootstrap_fence.lua");
const ABORT_BOOTSTRAP_LUA: &str = include_str!("redis/scripts/abort_collective_bootstrap.lua");
const TRANSFER_DEADLINE_MESSAGE: &str =
    "collective transfer exceeded MX_NCCL_REFIT_TRANSFER_TIMEOUT_S";

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

fn canonicalize_membership_spec(spec: &CollectiveGroupSpec) -> CollectiveGroupSpec {
    let mut canonical = spec.clone();
    canonical.expected_trainer_slots.sort();
    canonical.expected_generator_slots.sort();
    canonical
}

fn validate_slot_encoding(spec: &CollectiveGroupSpec) -> CollectiveResult<()> {
    for slot in spec
        .expected_trainer_slots
        .iter()
        .chain(&spec.expected_generator_slots)
        .chain(
            spec.lanes
                .iter()
                .flat_map(|lane| lane.trainer_slots.iter().chain(&lane.generator_slots)),
        )
    {
        if slot.contains(['\0', '\n', '\r', '|', ',']) {
            return Err(CollectiveBackendError::InvalidArgument(
                "collective slot ids must not contain Redis record delimiters".to_string(),
            ));
        }
    }
    Ok(())
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
    transfer_timeout: std::time::Duration,
}

impl RedisCollectiveBackend {
    pub async fn connect(url: &str) -> CollectiveResult<Self> {
        let transfer_timeout = modelexpress_common::envs::nccl_refit_transfer_timeout()
            .map_err(CollectiveBackendError::InvalidArgument)?;
        Self::connect_with_transfer_timeout(url, transfer_timeout).await
    }

    pub async fn connect_with_transfer_timeout(
        url: &str,
        transfer_timeout: std::time::Duration,
    ) -> CollectiveResult<Self> {
        let client = redis::Client::open(url).map_err(redis_error)?;
        let connection = ConnectionManager::new(client).await.map_err(redis_error)?;
        Ok(Self {
            connection,
            transfer_timeout,
        })
    }

    fn layout_for(spec: &CollectiveGroupSpec) -> CollectiveResult<LaneLayout> {
        validate_slot_encoding(spec)?;
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

    async fn expire_transfer(&self, operation_id: &str) -> CollectiveResult<()> {
        let mut connection = self.connection.clone();
        let fields: HashMap<String, String> = connection
            .hgetall(operation_key(operation_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Ok(());
        }
        let group_id = field(&fields, "group_id")?;
        let model_name = field(&fields, "model_name")?;
        let idempotency_key = field(&fields, "idempotency_key")?;
        let outcome: String = Script::new(EXPIRE_TRANSFER_LUA)
            .key(operation_key(operation_id))
            .key(reported_key(operation_id))
            .key(operation_idempotency_key(model_name, idempotency_key))
            .key(group_key(group_id))
            .arg(operation_id)
            .arg(group_id)
            .arg(TRANSFER_DEADLINE_MESSAGE)
            .invoke_async(&mut connection)
            .await
            .map_err(redis_error)?;
        match outcome.as_str() {
            "ACTIVE" | "ABORTED" | "NOTFOUND" => Ok(()),
            other => Err(CollectiveBackendError::Internal(format!(
                "unexpected transfer expiry outcome {other}"
            ))),
        }
    }

    async fn read_transfer(&self, operation_id: &str) -> CollectiveResult<CollectiveTransfer> {
        self.expire_transfer(operation_id).await?;
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
        let request_spec = request.spec.as_ref().ok_or_else(|| {
            CollectiveBackendError::InvalidArgument("spec is required".to_string())
        })?;
        let spec = canonicalize_membership_spec(request_spec);
        let layout = Self::layout_for(&spec)?;
        let role = CollectiveRole::try_from(request.role).unwrap_or(CollectiveRole::Unspecified);
        let role_slots = match role {
            CollectiveRole::Trainer => &spec.expected_trainer_slots,
            CollectiveRole::Generator => &spec.expected_generator_slots,
            CollectiveRole::Unspecified => {
                return Err(CollectiveBackendError::InvalidArgument(
                    "role must be specified".to_string(),
                ));
            }
        };
        let index_in_role =
            u32::try_from(role_slots.binary_search(&request.slot_id).map_err(|_| {
                CollectiveBackendError::InvalidArgument(format!(
                    "slot {} is not expected for its role",
                    request.slot_id
                ))
            })?)
            .map_err(|_| {
                CollectiveBackendError::InvalidArgument("role contains too many slots".to_string())
            })?;
        let assignments = layout
            .assign(&request.slot_id)
            .map_err(|error| CollectiveBackendError::InvalidArgument(error.to_string()))?;

        let group_id = group_id_for(&spec);
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
            .arg(index_in_role)
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
                    index_in_role
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

    async fn abort_bootstrap(
        &self,
        request: &AbortCollectiveBootstrapRequest,
    ) -> CollectiveResult<CollectiveGroup> {
        let outcome: String = Script::new(ABORT_BOOTSTRAP_LUA)
            .key(group_key(&request.group_id))
            .key(participants_key(&request.group_id))
            .key(worker_key(&request.worker_id))
            .arg(request.epoch)
            .arg(&request.slot_id)
            .arg(&request.worker_id)
            .arg(&request.message)
            .invoke_async(&mut self.connection.clone())
            .await
            .map_err(redis_error)?;
        match outcome.as_str() {
            "NOTFOUND" => Err(CollectiveBackendError::NotFound(format!(
                "collective group {} was not found",
                request.group_id
            ))),
            "NOTREADY" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective group {} is not READY for bootstrap abort",
                request.group_id
            ))),
            "NOTADMITTED" | "NOTLIVE" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "worker {} is not the live admitted generation for slot {}",
                request.worker_id, request.slot_id
            ))),
            other => {
                if let Some(operation_id) = other.strip_prefix("ACTIVE:") {
                    return Err(CollectiveBackendError::FailedPrecondition(format!(
                        "collective group {} is locked by active operation {operation_id}",
                        request.group_id
                    )));
                }
                if let Some(current) = other.strip_prefix("STALE:") {
                    return Err(CollectiveBackendError::FailedPrecondition(format!(
                        "bootstrap abort for epoch {} was rejected; the group is at epoch {current}",
                        request.epoch
                    )));
                }
                if !other.starts_with("OK:") {
                    return Err(CollectiveBackendError::Internal(format!(
                        "unexpected bootstrap abort outcome {other}"
                    )));
                }
                self.read_group(&request.group_id).await
            }
        }
    }

    async fn create_transfer(
        &self,
        request: &CreateCollectiveTransferRequest,
    ) -> CollectiveResult<CollectiveTransfer> {
        let request_spec = request.spec.as_ref().ok_or_else(|| {
            CollectiveBackendError::InvalidArgument("spec is required".to_string())
        })?;
        let spec = canonicalize_membership_spec(request_spec);
        Self::layout_for(&spec)?;
        let group_id = group_id_for(&spec);
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
            .arg(
                u64::try_from(self.transfer_timeout.as_millis()).map_err(|_| {
                    CollectiveBackendError::InvalidArgument(
                        "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S is too large to represent as milliseconds"
                            .to_string(),
                    )
                })?,
            )
            .arg(OPERATION_KEY_PREFIX)
            .arg(TRANSFER_DEADLINE_MESSAGE)
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
            "NOTREADY" => Err(CollectiveBackendError::FailedPrecondition(format!(
                "collective group {group_id} is not READY for a new transfer"
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
                None => match other.strip_prefix("ACTIVE:") {
                    Some(active) => Err(CollectiveBackendError::FailedPrecondition(format!(
                        "collective group {group_id} already has active operation {active}"
                    ))),
                    None => Err(CollectiveBackendError::Internal(format!(
                        "unexpected create outcome {other}"
                    ))),
                },
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
        let mut connection = self.connection.clone();
        let fields: HashMap<String, String> = connection
            .hgetall(operation_key(&request.operation_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Err(CollectiveBackendError::NotFound(format!(
                "collective transfer {} was not found",
                request.operation_id
            )));
        }
        let model_name = field(&fields, "model_name")?;
        let idempotency_key = field(&fields, "idempotency_key")?;
        let group = self.read_group(&request.group_id).await?;
        let report_script = Script::new(REPORT_TRANSFER_LUA);
        let mut script = report_script.prepare_invoke();
        script.key(operation_key(&request.operation_id));
        script.key(reported_key(&request.operation_id));
        script.key(group_key(&request.group_id));
        script.key(participants_key(&request.group_id));
        script.key(operation_idempotency_key(model_name, idempotency_key));
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
            .arg(TRANSFER_DEADLINE_MESSAGE)
            .invoke_async(&mut connection)
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
    use modelexpress_common::grpc::refit_collective::{
        AbortCollectiveBootstrapRequest, BootstrapFencePhase, JoinCollectiveGroupRequest, LaneSpec,
        PlanSource, PublishGroupBootstrapRequest, ReachCollectiveBootstrapFenceRequest,
    };

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
    async fn transfer_idempotency_reclaims_a_stale_epoch_reservation() {
        let url = std::env::var("MX_TEST_REDIS_URL")
            .expect("MX_TEST_REDIS_URL must point at an isolated Redis");
        let backend = RedisCollectiveBackend::connect_with_transfer_timeout(
            &url,
            std::time::Duration::from_secs(30),
        )
        .await
        .expect("connect");
        let mut redis = backend.connection.clone();
        redis::cmd("FLUSHDB")
            .query_async::<()>(&mut redis)
            .await
            .expect("flush");
        for (worker_id, role) in [("w-t0", 1), ("w-t0-replacement", 1), ("w-g0", 2)] {
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

        let group_spec = spec("m", &["t0"], &["g0"], 1);
        let trainer = |worker_id: &str| JoinCollectiveGroupRequest {
            spec: Some(group_spec.clone()),
            slot_id: "t0".to_string(),
            worker_id: worker_id.to_string(),
            role: CollectiveRole::Trainer.into(),
            index_in_role: 0,
            plan_digest: "digest".to_string(),
            plan_source: Some(PlanSource {
                worker_id: worker_id.to_string(),
                endpoint: "trainer:9000".to_string(),
                digest: "digest".to_string(),
            }),
        };
        let generator = JoinCollectiveGroupRequest {
            spec: Some(group_spec.clone()),
            slot_id: "g0".to_string(),
            worker_id: "w-g0".to_string(),
            role: CollectiveRole::Generator.into(),
            index_in_role: 0,
            plan_digest: "digest".to_string(),
            plan_source: None,
        };

        let first = backend
            .join_group(&trainer("w-t0"))
            .await
            .expect("trainer join");
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
                    nccl_unique_id: vec![u8::try_from(lane_id + 1).unwrap_or(0); 128],
                })
                .await
                .expect("publish first-epoch bootstrap");
        }
        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: first.group_id.clone(),
                        epoch: first.epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("release first-epoch pre-barrier");
            }
        }
        for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: first.epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("complete first-epoch bootstrap");
        }

        let request = CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "v1".to_string(),
            idempotency_key: "miles-weight-version-1".to_string(),
        };
        let first_operation = backend
            .create_transfer(&request)
            .await
            .expect("create first-epoch operation");
        let same_epoch_retry = backend
            .create_transfer(&request)
            .await
            .expect("same-epoch retry");
        assert_eq!(same_epoch_retry.operation_id, first_operation.operation_id);

        let second = backend
            .join_group(&trainer("w-t0-replacement"))
            .await
            .expect("replace trainer for the next epoch");
        assert_eq!(second.epoch, first.epoch + 1);
        let second_generator = backend
            .join_group(&generator)
            .await
            .expect("refresh generator for the next epoch");
        assert_eq!(second_generator.epoch, second.epoch);
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: second.group_id.clone(),
                    epoch: second.epoch,
                    lane_id,
                    worker_id: "w-t0-replacement".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 3).unwrap_or(0); 128],
                })
                .await
                .expect("publish second-epoch bootstrap");
        }
        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t0", "w-t0-replacement"), ("g0", "w-g0")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: second.group_id.clone(),
                        epoch: second.epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("release second-epoch pre-barrier");
            }
        }
        for (slot_id, worker_id) in [("t0", "w-t0-replacement"), ("g0", "w-g0")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: second.group_id.clone(),
                    epoch: second.epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("complete second-epoch bootstrap");
        }

        let reservation_key = operation_idempotency_key("m", &request.idempotency_key);
        let stale_reservation: Option<String> = redis
            .get(&reservation_key)
            .await
            .expect("read stale reservation");
        assert_eq!(
            stale_reservation.as_deref(),
            Some(first_operation.operation_id.as_str())
        );

        let collision = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                spec: Some(group_spec.clone()),
                version_id: "v2".to_string(),
                idempotency_key: request.idempotency_key.clone(),
            })
            .await
            .expect_err("the same key must not be reclaimed for another version");
        assert!(matches!(
            collision,
            CollectiveBackendError::AlreadyExists(_)
        ));
        let unchanged_state: String = redis
            .hget(operation_key(&first_operation.operation_id), "state")
            .await
            .expect("read operation after collision");
        assert_eq!(unchanged_state, "PENDING");
        let unchanged_reservation: Option<String> = redis
            .get(&reservation_key)
            .await
            .expect("read reservation after collision");
        assert_eq!(
            unchanged_reservation.as_deref(),
            Some(first_operation.operation_id.as_str())
        );

        let replacement = backend
            .create_transfer(&request)
            .await
            .expect("reclaim stale reservation into the current epoch");
        assert_ne!(replacement.operation_id, first_operation.operation_id);
        assert_eq!(replacement.epoch, second.epoch);
        let stale_state: String = redis
            .hget(operation_key(&first_operation.operation_id), "state")
            .await
            .expect("read stale operation");
        assert_eq!(stale_state, "ABORTED");
        let current_reservation: Option<String> = redis
            .get(&reservation_key)
            .await
            .expect("read replacement reservation");
        assert_eq!(
            current_reservation.as_deref(),
            Some(replacement.operation_id.as_str())
        );

        let group = backend
            .read_group(&second.group_id)
            .await
            .expect("read current group");
        assert_eq!(group.epoch, second.epoch);
        assert_eq!(group.state, i32::from(CollectiveGroupState::Ready));
        let bootstrap_complete_epoch: u64 = redis
            .hget(group_key(&second.group_id), "bootstrap_complete_epoch")
            .await
            .expect("read bootstrap completion epoch");
        assert_eq!(bootstrap_complete_epoch, second.epoch);
    }

    #[tokio::test]
    #[ignore = "requires MX_TEST_REDIS_URL pointing at an isolated Redis"]
    async fn bootstrap_fence_is_idempotent_epoch_fenced_and_reset_on_abort() {
        let url = std::env::var("MX_TEST_REDIS_URL")
            .expect("MX_TEST_REDIS_URL must point at an isolated Redis");
        let backend = RedisCollectiveBackend::connect_with_transfer_timeout(
            &url,
            std::time::Duration::from_secs(30),
        )
        .await
        .expect("connect");
        let mut redis = backend.connection.clone();
        redis::cmd("FLUSHDB")
            .query_async::<()>(&mut redis)
            .await
            .expect("flush");
        for (worker_id, role) in [("w-t0", 1), ("w-t1", 1), ("w-g0", 2)] {
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

        let group_spec = spec("m", &["t0", "t1"], &["g0"], 1);
        let join =
            |slot_id: &str, worker_id: &str, role: CollectiveRole| JoinCollectiveGroupRequest {
                spec: Some(group_spec.clone()),
                slot_id: slot_id.to_string(),
                worker_id: worker_id.to_string(),
                role: role.into(),
                index_in_role: 0,
                plan_digest: "digest".to_string(),
                plan_source: (slot_id == "t0").then(|| PlanSource {
                    worker_id: worker_id.to_string(),
                    endpoint: "trainer:9000".to_string(),
                    digest: "digest".to_string(),
                }),
            };
        let trainer = join("t0", "w-t0", CollectiveRole::Trainer);
        let trainer_one = join("t1", "w-t1", CollectiveRole::Trainer);
        let generator = join("g0", "w-g0", CollectiveRole::Generator);
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

        let aborted = backend
            .abort_bootstrap(&AbortCollectiveBootstrapRequest {
                group_id: first.group_id.clone(),
                epoch: first.epoch,
                slot_id: "t0".to_string(),
                worker_id: "w-t0".to_string(),
                message: "bootstrap timeout".to_string(),
            })
            .await
            .expect("abort");
        assert_eq!(aborted.epoch, first.epoch + 1);
        assert_eq!(aborted.state, i32::from(CollectiveGroupState::Forming));
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
        backend
            .join_group(&trainer_one)
            .await
            .expect("retry second trainer");
        let retry_generator = backend
            .join_group(&generator)
            .await
            .expect("retry generator");
        assert_eq!(retry_trainer.epoch, aborted.epoch);
        assert_eq!(retry_generator.epoch, aborted.epoch);
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: first.group_id.clone(),
                    epoch: aborted.epoch,
                    lane_id,
                    worker_id: "w-t0".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 2).unwrap_or(0); 128],
                })
                .await
                .expect("retry publish");
        }
        let completion_arrival = || ReachCollectiveBootstrapFenceRequest {
            group_id: first.group_id.clone(),
            epoch: aborted.epoch,
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
                epoch: aborted.epoch,
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

        for (slot_id, worker_id) in [("g0", "w-g0"), ("t1", "w-t1")] {
            let released = backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: aborted.epoch,
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
                epoch: aborted.epoch,
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
        for (slot_id, worker_id) in [("g0", "w-g0"), ("t1", "w-t1")] {
            let released = backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: aborted.epoch,
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
        for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0"), ("t1", "w-t1")] {
            completion = Some(
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: first.group_id.clone(),
                        epoch: aborted.epoch,
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
                epoch: aborted.epoch,
                lane_id: 1,
                slot_id: "t1".to_string(),
                worker_id: "w-t1".to_string(),
                phase: BootstrapFencePhase::Complete.into(),
            })
            .await
            .expect("lost completion response is idempotent");
        assert!(replayed_completion.released);

        let create = CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "v1".to_string(),
            idempotency_key: "existing".to_string(),
        };
        let operation = backend
            .create_transfer(&create)
            .await
            .expect("create while READY");

        let late_abort = backend
            .abort_bootstrap(&AbortCollectiveBootstrapRequest {
                group_id: first.group_id.clone(),
                epoch: aborted.epoch,
                slot_id: "t0".to_string(),
                worker_id: "w-t0".to_string(),
                message: "lost completion response".to_string(),
            })
            .await
            .expect_err("an active operation must exclude a late bootstrap abort");
        assert!(matches!(
            late_abort,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        assert_eq!(
            backend
                .read_group(&first.group_id)
                .await
                .expect("group after late abort")
                .epoch,
            aborted.epoch
        );

        for worker_id in ["w-t0", "w-t1", "w-g0"] {
            backend
                .report_transfer(&ReportCollectiveTransferRequest {
                    operation_id: operation.operation_id.clone(),
                    group_id: first.group_id.clone(),
                    epoch: aborted.epoch,
                    worker_id: worker_id.to_string(),
                    succeeded: true,
                    message: String::new(),
                })
                .await
                .expect("complete operation");
        }
        let post_operation_abort = backend
            .abort_bootstrap(&AbortCollectiveBootstrapRequest {
                group_id: first.group_id.clone(),
                epoch: aborted.epoch,
                slot_id: "t0".to_string(),
                worker_id: "w-t0".to_string(),
                message: "abort wins before next create".to_string(),
            })
            .await
            .expect("abort after operation completes");
        assert_eq!(post_operation_abort.epoch, aborted.epoch + 1);

        let stale_retry = backend
            .create_transfer(&create)
            .await
            .expect_err("a prior-epoch reservation must not be returned");
        assert!(matches!(
            stale_retry,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        let stale_reservation: Option<String> = redis
            .get(operation_idempotency_key("m", &create.idempotency_key))
            .await
            .expect("stale reservation is reclaimed");
        assert!(stale_reservation.is_none());

        let rejected = backend
            .create_transfer(&CreateCollectiveTransferRequest {
                idempotency_key: "new".to_string(),
                ..create
            })
            .await
            .expect_err("abort-before-create must reject a new operation");
        assert!(matches!(
            rejected,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        let all_operation_keys: Vec<String> = redis
            .keys(format!("{OPERATION_KEY_PREFIX}*"))
            .await
            .expect("operation keys");
        let operation_keys: Vec<String> = all_operation_keys
            .into_iter()
            .filter(|key: &String| !key.ends_with(":reported"))
            .collect();
        assert_eq!(operation_keys, [operation_key(&operation.operation_id)]);

        let expiry_epoch = post_operation_abort.epoch;
        backend
            .join_group(&trainer)
            .await
            .expect("expiry epoch trainer join");
        backend
            .join_group(&trainer_one)
            .await
            .expect("expiry epoch second trainer join");
        backend
            .join_group(&generator)
            .await
            .expect("expiry epoch generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: first.group_id.clone(),
                    epoch: expiry_epoch,
                    lane_id,
                    worker_id: "w-t0".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 4).unwrap_or(0); 128],
                })
                .await
                .expect("expiry epoch publish");
        }
        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0"), ("t1", "w-t1")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: first.group_id.clone(),
                        epoch: expiry_epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("expiry epoch PRE_BARRIER");
            }
        }
        for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0"), ("t1", "w-t1")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: expiry_epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("expiry epoch COMPLETE");
        }

        let expiring_request = CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "v-expired".to_string(),
            idempotency_key: "expired-a".to_string(),
        };
        let expiring = backend
            .create_transfer(&expiring_request)
            .await
            .expect("create operation that will be abandoned");
        redis::cmd("HSET")
            .arg(operation_key(&expiring.operation_id))
            .arg("deadline_unix_ms")
            .arg(0)
            .query_async::<()>(&mut redis)
            .await
            .expect("expire operation deadline");

        let replacement_request = CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "v-replacement".to_string(),
            idempotency_key: "replacement-b".to_string(),
        };
        let first_replacement = backend
            .create_transfer(&replacement_request)
            .await
            .expect_err("distinct-key create must fence the expired active operation");
        assert!(matches!(
            first_replacement,
            CollectiveBackendError::FailedPrecondition(_)
        ));
        let expired_state: String = redis
            .hget(operation_key(&expiring.operation_id), "state")
            .await
            .expect("expired operation state");
        assert_eq!(expired_state, "ABORTED");
        let after_expiry = backend
            .read_group(&first.group_id)
            .await
            .expect("group after active operation expiry");
        assert_eq!(after_expiry.epoch, expiry_epoch + 1);
        assert_eq!(after_expiry.state, i32::from(CollectiveGroupState::Forming));

        let replacement_epoch = after_expiry.epoch;
        backend
            .join_group(&trainer)
            .await
            .expect("replacement epoch trainer join");
        backend
            .join_group(&trainer_one)
            .await
            .expect("replacement epoch second trainer join");
        backend
            .join_group(&generator)
            .await
            .expect("replacement epoch generator join");
        for lane_id in [0, 1] {
            backend
                .publish_bootstrap(&PublishGroupBootstrapRequest {
                    group_id: first.group_id.clone(),
                    epoch: replacement_epoch,
                    lane_id,
                    worker_id: "w-t0".to_string(),
                    nccl_unique_id: vec![u8::try_from(lane_id + 6).unwrap_or(0); 128],
                })
                .await
                .expect("replacement epoch publish");
        }
        for lane_id in [0, 1] {
            for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0"), ("t1", "w-t1")] {
                backend
                    .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                        group_id: first.group_id.clone(),
                        epoch: replacement_epoch,
                        lane_id,
                        slot_id: slot_id.to_string(),
                        worker_id: worker_id.to_string(),
                        phase: BootstrapFencePhase::PreBarrier.into(),
                    })
                    .await
                    .expect("replacement epoch PRE_BARRIER");
            }
        }
        for (slot_id, worker_id) in [("t0", "w-t0"), ("g0", "w-g0"), ("t1", "w-t1")] {
            backend
                .reach_bootstrap_fence(&ReachCollectiveBootstrapFenceRequest {
                    group_id: first.group_id.clone(),
                    epoch: replacement_epoch,
                    lane_id: 1,
                    slot_id: slot_id.to_string(),
                    worker_id: worker_id.to_string(),
                    phase: BootstrapFencePhase::Complete.into(),
                })
                .await
                .expect("replacement epoch COMPLETE");
        }
        let replacement = backend
            .create_transfer(&replacement_request)
            .await
            .expect("replacement create succeeds after re-bootstrap");
        assert_ne!(replacement.operation_id, expiring.operation_id);
    }
}
