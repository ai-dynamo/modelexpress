// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis implementation of the Refit control-plane backend.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use modelexpress_common::grpc::refit::{
    CreateWeightVersionRequest, DeleteVersionLeaseRequest, DeleteWeightVersionShardRequest,
    ObjectStorageSource, ObjectStorageType, RegisterVersionLeaseRequest,
    UpdateWeightVersionStateRequest, VersionLease, WeightVersion, WeightVersionShard,
    WeightVersionState, WorkerRegistration,
};
use prost::Message;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Script};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{RefitBackend, RefitBackendError, RefitResult};

const CREATE_VERSION_LUA: &str = include_str!("redis/scripts/create_weight_version.lua");
const REGISTER_WORKER_LUA: &str = include_str!("redis/scripts/register_worker.lua");
const CREATE_SHARD_LUA: &str = include_str!("redis/scripts/create_weight_version_shard.lua");
const UPDATE_VERSION_STATE_LUA: &str =
    include_str!("redis/scripts/update_weight_version_state.lua");
const DELETE_SHARD_LUA: &str = include_str!("redis/scripts/delete_weight_version_shard.lua");
const REGISTER_LEASE_LUA: &str = include_str!("redis/scripts/register_version_lease.lua");
const DELETE_LEASE_LUA: &str = include_str!("redis/scripts/delete_version_lease.lua");

fn version_key(version_id: &str) -> String {
    format!("mx:refit:version:metadata:{version_id}")
}

fn shards_key(version_id: &str) -> String {
    format!("mx:refit:version:shards:{version_id}")
}

fn publication_key(worker_id: &str, source_slot_id: &str) -> String {
    format!("{}:{worker_id}{source_slot_id}", worker_id.len())
}

fn coverage_key(version_id: &str) -> String {
    format!("mx:refit:version:coverage:{version_id}")
}

fn expected_source_slots_key(version_id: &str) -> String {
    format!("mx:refit:version:expected-source-slots:{version_id}")
}

fn worker_key(worker_id: &str) -> String {
    format!("mx:refit:worker:{worker_id}")
}

fn leases_key(version_id: &str) -> String {
    format!("mx:refit:version:leases:{version_id}")
}

fn lease_key(version_id: &str, lease_id: &str) -> String {
    format!("mx:refit:version:lease:{version_id}:{lease_id}")
}

fn idempotency_key(model_name: &str, request_key: &str) -> String {
    format!("mx:refit:version-request:{model_name}:{request_key}")
}

fn lease_id(version_id: &str, worker_id: &str) -> String {
    let digest = Sha256::digest(format!("{version_id}\0{worker_id}").as_bytes());
    let mut id = String::with_capacity(8);
    for byte in &digest[..4] {
        let _ = write!(id, "{byte:02x}");
    }
    id
}

fn now_unix_ms() -> RefitResult<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| RefitBackendError::Internal(format!("system clock error: {error}")))?
        .as_millis();
    u64::try_from(millis)
        .map_err(|_| RefitBackendError::Internal("system time does not fit in uint64".to_string()))
}

fn redis_error(error: redis::RedisError) -> RefitBackendError {
    if error.is_io_error()
        || error.is_cluster_error()
        || matches!(
            error.kind(),
            redis::ErrorKind::BusyLoadingError
                | redis::ErrorKind::MasterDown
                | redis::ErrorKind::ClusterConnectionNotFound
        )
    {
        RefitBackendError::Unavailable(error.to_string())
    } else {
        RefitBackendError::Internal(error.to_string())
    }
}

fn hash_field<'a>(fields: &'a HashMap<String, String>, name: &str) -> RefitResult<&'a str> {
    fields.get(name).map(String::as_str).ok_or_else(|| {
        RefitBackendError::Internal(format!("Refit metadata record is missing {name}"))
    })
}

fn parse_hash_field<T>(fields: &HashMap<String, String>, name: &str) -> RefitResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    hash_field(fields, name)?.parse().map_err(|error| {
        RefitBackendError::Internal(format!("invalid {name} in Refit metadata: {error}"))
    })
}

fn version_from_hash(fields: HashMap<String, String>) -> RefitResult<WeightVersion> {
    Ok(WeightVersion {
        uid: hash_field(&fields, "uid")?.to_string(),
        model_name: hash_field(&fields, "model_name")?.to_string(),
        idempotency_key: hash_field(&fields, "idempotency_key")?.to_string(),
        payload_format: parse_hash_field(&fields, "payload_format")?,
        base_version_id: match hash_field(&fields, "base_version_id")? {
            "" => None,
            value => Some(value.to_string()),
        },
        expected_source_slots: serde_json::from_str(hash_field(&fields, "expected_source_slots")?)
            .map_err(|error| {
                RefitBackendError::Internal(format!("invalid expected_source_slots: {error}"))
            })?,
        layout_signature: hash_field(&fields, "layout_signature")?.to_string(),
        state: parse_hash_field(&fields, "state")?,
        created_at_unix_ms: parse_hash_field(&fields, "created_at_unix_ms")?,
        object_storage: fields
            .get("s3_uri")
            .filter(|uri| !uri.is_empty())
            .map(|uri| ObjectStorageSource {
                uri: uri.clone(),
                storage_type: ObjectStorageType::S3.into(),
            }),
    })
}

fn lease_from_hash(fields: HashMap<String, String>) -> RefitResult<VersionLease> {
    Ok(VersionLease {
        lease_id: hash_field(&fields, "lease_id")?.to_string(),
        version_id: hash_field(&fields, "version_id")?.to_string(),
        worker_id: hash_field(&fields, "worker_id")?.to_string(),
        expires_at_unix_ms: parse_hash_field(&fields, "expires_at_unix_ms")?,
    })
}

#[derive(Clone)]
pub struct RedisRefitBackend {
    redis: ConnectionManager,
}

impl RedisRefitBackend {
    pub async fn connect(redis_url: &str) -> RefitResult<Self> {
        let client = redis::Client::open(redis_url).map_err(redis_error)?;
        let redis = ConnectionManager::new(client).await.map_err(redis_error)?;
        Ok(Self { redis })
    }

    async fn get_version_fields(&self, uid: &str) -> RefitResult<HashMap<String, String>> {
        let mut redis = self.redis.clone();
        let fields: HashMap<String, String> =
            redis.hgetall(version_key(uid)).await.map_err(redis_error)?;
        if fields.is_empty() {
            return Err(RefitBackendError::NotFound(format!(
                "weight version UID {uid:?} was not found"
            )));
        }
        Ok(fields)
    }

    async fn transition_weight_version_state(
        &self,
        uid: &str,
        state: i32,
    ) -> RefitResult<WeightVersion> {
        let mut redis = self.redis.clone();
        let result: String = Script::new(UPDATE_VERSION_STATE_LUA)
            .key(version_key(uid))
            .arg(state)
            .arg(i32::from(WeightVersionState::Staging))
            .arg(i32::from(WeightVersionState::Ready))
            .arg(i32::from(WeightVersionState::Releasing))
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            "OK" => self.get_weight_version(uid).await,
            "VERSION_NOT_FOUND" => Err(RefitBackendError::NotFound(
                "weight version was not found".to_string(),
            )),
            "INVALID_TRANSITION" => Err(RefitBackendError::FailedPrecondition(
                "weight version state transition is not allowed".to_string(),
            )),
            _ => Err(RefitBackendError::Internal(format!(
                "unexpected Redis response: {result}"
            ))),
        }
    }

    async fn get_lease(&self, version_id: &str, lease_id: &str) -> RefitResult<VersionLease> {
        let mut redis = self.redis.clone();
        let fields: HashMap<String, String> = redis
            .hgetall(lease_key(version_id, lease_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Err(RefitBackendError::NotFound(format!(
                "version lease {lease_id:?} was not found"
            )));
        }
        lease_from_hash(fields)
    }

    async fn create_version_once(
        &self,
        request: &CreateWeightVersionRequest,
        uid: &str,
    ) -> RefitResult<String> {
        let expected_source_slots =
            serde_json::to_string(&request.expected_source_slots).map_err(|error| {
                RefitBackendError::Internal(format!("encode expected_source_slots: {error}"))
            })?;
        let script = Script::new(CREATE_VERSION_LUA);
        let mut invocation = script.prepare_invoke();
        invocation
            .key(version_key(uid))
            .key(idempotency_key(
                &request.model_name,
                &request.idempotency_key,
            ))
            .key(expected_source_slots_key(uid))
            .arg(uid)
            .arg(&request.model_name)
            .arg(&request.idempotency_key)
            .arg(request.payload_format)
            .arg(request.base_version_id.as_deref().unwrap_or_default())
            .arg(expected_source_slots)
            .arg(request.expected_source_slots.len())
            .arg(
                request
                    .object_storage
                    .as_ref()
                    .map_or("", |source| source.uri.as_str()),
            )
            .arg(request.state)
            .arg(request.state)
            .arg(now_unix_ms()?);
        for source_slot_id in &request.expected_source_slots {
            invocation.arg(source_slot_id);
        }
        let mut redis = self.redis.clone();
        invocation
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)
    }
}

#[async_trait]
impl RefitBackend for RedisRefitBackend {
    async fn register_worker(
        &self,
        mut worker: WorkerRegistration,
        ttl_seconds: u32,
    ) -> RefitResult<WorkerRegistration> {
        let mut redis = self.redis.clone();
        let result: String = Script::new(REGISTER_WORKER_LUA)
            .key(worker_key(&worker.worker_id))
            .arg(&worker.worker_id)
            .arg(worker.role)
            .arg(&worker.model_name)
            .arg(u64::from(ttl_seconds).saturating_mul(1000))
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        if result == "CONFLICT" {
            return Err(RefitBackendError::AlreadyExists(
                "worker_id is already registered with different metadata".to_string(),
            ));
        }
        worker.expires_at_unix_ms = result
            .strip_prefix("OK:")
            .ok_or_else(|| {
                RefitBackendError::Internal(format!("unexpected Redis response: {result}"))
            })?
            .parse()
            .map_err(|error| {
                RefitBackendError::Internal(format!("invalid expiry from Redis: {error}"))
            })?;
        Ok(worker)
    }

    async fn create_weight_version(
        &self,
        request: &CreateWeightVersionRequest,
    ) -> RefitResult<WeightVersion> {
        let requested_uid = request.uid.as_deref();
        let attempts = if requested_uid.is_some() { 1 } else { 5 };
        for _ in 0..attempts {
            let uid = match requested_uid {
                Some(uid) => uid.to_string(),
                None => Uuid::new_v4()
                    .simple()
                    .to_string()
                    .chars()
                    .take(8)
                    .collect(),
            };
            let result = self.create_version_once(request, &uid).await?;
            if result == "CREATED" {
                return self.get_weight_version(&uid).await;
            }
            if let Some(existing_uid) = result.strip_prefix("EXISTING:") {
                if requested_uid.is_some_and(|uid| uid != existing_uid) {
                    return Err(RefitBackendError::AlreadyExists(
                        "idempotency_key was already used for a different WeightVersion"
                            .to_string(),
                    ));
                }
                let fields = self.get_version_fields(existing_uid).await?;
                let initial_state = fields.get("initial_state").map_or_else(
                    || Ok(i32::from(WeightVersionState::Staging)),
                    |value| {
                        value.parse().map_err(|error| {
                            RefitBackendError::Internal(format!(
                                "invalid initial_state in Refit metadata: {error}"
                            ))
                        })
                    },
                )?;
                let existing = version_from_hash(fields)?;
                if existing.model_name == request.model_name
                    && existing.payload_format == request.payload_format
                    && existing.base_version_id == request.base_version_id
                    && existing.expected_source_slots == request.expected_source_slots
                    && existing.object_storage == request.object_storage
                    && initial_state == request.state
                {
                    return Ok(existing);
                }
                return Err(RefitBackendError::AlreadyExists(
                    "idempotency_key was already used for a different WeightVersion".to_string(),
                ));
            }
            if result == "COLLISION" && requested_uid.is_some() {
                return Err(RefitBackendError::AlreadyExists(format!(
                    "weight version UID {uid:?} already exists"
                )));
            }
            if result != "COLLISION" {
                return Err(RefitBackendError::Internal(format!(
                    "unexpected Redis response: {result}"
                )));
            }
        }
        Err(RefitBackendError::ResourceExhausted(
            "could not allocate a unique weight version ID".to_string(),
        ))
    }

    async fn get_weight_version(&self, uid: &str) -> RefitResult<WeightVersion> {
        version_from_hash(self.get_version_fields(uid).await?)
    }

    async fn delete_weight_version(&self, uid: &str) -> RefitResult<WeightVersion> {
        self.transition_weight_version_state(uid, WeightVersionState::Releasing.into())
            .await
    }

    async fn update_weight_version_state(
        &self,
        request: &UpdateWeightVersionStateRequest,
    ) -> RefitResult<WeightVersion> {
        self.transition_weight_version_state(&request.uid, request.state)
            .await
    }

    async fn create_weight_version_shard(
        &self,
        shard: WeightVersionShard,
    ) -> RefitResult<(WeightVersionShard, WeightVersion)> {
        let mut version = self.get_weight_version(&shard.version_id).await?;
        let publication_key = publication_key(&shard.worker_id, &shard.source_slot_id);
        let encoded = shard.encode_to_vec();
        let mut redis = self.redis.clone();
        let result: String = Script::new(CREATE_SHARD_LUA)
            .key(version_key(&shard.version_id))
            .key(worker_key(&shard.worker_id))
            .key(shards_key(&shard.version_id))
            .key(coverage_key(&shard.version_id))
            .key(expected_source_slots_key(&shard.version_id))
            .arg(publication_key)
            .arg(encoded)
            .arg(&version.model_name)
            .arg(&shard.source_slot_id)
            .arg(i32::from(WeightVersionState::Staging))
            .arg(i32::from(WeightVersionState::Ready))
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        let state = match result.as_str() {
            "VERSION_NOT_FOUND" => {
                return Err(RefitBackendError::NotFound(
                    "weight version was not found".to_string(),
                ));
            }
            "WORKER_NOT_FOUND" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "worker registration is missing or expired".to_string(),
                ));
            }
            "MODEL_MISMATCH" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "worker and weight version model_name differ".to_string(),
                ));
            }
            "SOURCE_SLOT_NOT_REQUIRED" => {
                return Err(RefitBackendError::InvalidArgument(
                    "source_slot_id is not required by the weight version".to_string(),
                ));
            }
            "VERSION_NOT_WRITABLE" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "weight version does not accept shard publication".to_string(),
                ));
            }
            "SHARD_CONFLICT" => {
                return Err(RefitBackendError::AlreadyExists(
                    "worker and source_slot_id already published different metadata".to_string(),
                ));
            }
            value => {
                let Some(state) = value.strip_prefix("OK:") else {
                    return Err(RefitBackendError::Internal(format!(
                        "unexpected Redis response: {result}"
                    )));
                };
                state.parse().map_err(|error| {
                    RefitBackendError::Internal(format!("invalid publication state: {error}"))
                })?
            }
        };
        WeightVersionState::try_from(state).map_err(|_| {
            RefitBackendError::Internal(format!("invalid publication state: {state}"))
        })?;
        version.state = state;
        Ok((shard, version))
    }

    async fn list_weight_version_shards(
        &self,
        version_id: &str,
    ) -> RefitResult<Vec<WeightVersionShard>> {
        self.get_weight_version(version_id).await?;
        let mut redis = self.redis.clone();
        let encoded: Vec<Vec<u8>> = redis
            .hvals(shards_key(version_id))
            .await
            .map_err(redis_error)?;
        let mut shards = encoded
            .into_iter()
            .map(|bytes| {
                WeightVersionShard::decode(bytes.as_slice()).map_err(|error| {
                    RefitBackendError::Internal(format!(
                        "invalid WeightVersionShard in Redis: {error}"
                    ))
                })
            })
            .collect::<RefitResult<Vec<_>>>()?;
        shards.sort_by(|left, right| {
            (&left.source_slot_id, &left.worker_id).cmp(&(&right.source_slot_id, &right.worker_id))
        });
        Ok(shards)
    }

    async fn delete_weight_version_shard(
        &self,
        request: &DeleteWeightVersionShardRequest,
    ) -> RefitResult<bool> {
        let publication_key = publication_key(&request.worker_id, &request.source_slot_id);
        let mut redis = self.redis.clone();
        let encoded: Option<Vec<u8>> = redis
            .hget(shards_key(&request.version_id), &publication_key)
            .await
            .map_err(redis_error)?;
        let encoded = encoded.ok_or_else(|| {
            RefitBackendError::NotFound("weight version shard not found".to_string())
        })?;
        let shard = WeightVersionShard::decode(encoded.as_slice()).map_err(|error| {
            RefitBackendError::Internal(format!("invalid WeightVersionShard in Redis: {error}"))
        })?;
        if shard.worker_id != request.worker_id {
            return Err(RefitBackendError::FailedPrecondition(
                "only the publishing worker can delete its shard".to_string(),
            ));
        }

        let result: String = Script::new(DELETE_SHARD_LUA)
            .key(version_key(&request.version_id))
            .key(worker_key(&request.worker_id))
            .key(shards_key(&request.version_id))
            .key(leases_key(&request.version_id))
            .arg(publication_key)
            .arg(encoded)
            .arg(i32::from(WeightVersionState::Releasing))
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            "DELETED" => Ok(true),
            "VERSION_NOT_FOUND" => Err(RefitBackendError::NotFound(
                "weight version was not found".to_string(),
            )),
            "VERSION_NOT_RELEASING" => Err(RefitBackendError::FailedPrecondition(
                "weight version must be RELEASING before its shards can be deleted".to_string(),
            )),
            "WORKER_NOT_FOUND" => Err(RefitBackendError::FailedPrecondition(
                "worker registration is missing or expired".to_string(),
            )),
            "SHARD_NOT_FOUND" => Err(RefitBackendError::NotFound(
                "weight version shard not found".to_string(),
            )),
            "SHARD_CONFLICT" => Err(RefitBackendError::FailedPrecondition(
                "weight version shard changed while it was being deleted".to_string(),
            )),
            "VERSION_LEASED" => Err(RefitBackendError::FailedPrecondition(
                "weight version has an active lease".to_string(),
            )),
            _ => Err(RefitBackendError::Internal(format!(
                "unexpected Redis response: {result}"
            ))),
        }
    }

    async fn register_version_lease(
        &self,
        request: &RegisterVersionLeaseRequest,
    ) -> RefitResult<VersionLease> {
        let lease_id = lease_id(&request.version_id, &request.worker_id);
        let mut redis = self.redis.clone();
        let result: String = Script::new(REGISTER_LEASE_LUA)
            .key(version_key(&request.version_id))
            .key(worker_key(&request.worker_id))
            .key(lease_key(&request.version_id, &lease_id))
            .key(leases_key(&request.version_id))
            .arg(&lease_id)
            .arg(&request.version_id)
            .arg(&request.worker_id)
            .arg(u64::from(request.ttl_seconds).saturating_mul(1000))
            .arg(i32::from(WeightVersionState::Ready))
            .arg(i32::from(WeightVersionState::Releasing))
            .arg(i32::from(
                modelexpress_common::grpc::refit::WorkerRole::Generator,
            ))
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            value if value.starts_with("OK:") => {
                self.get_lease(&request.version_id, &lease_id).await
            }
            "VERSION_NOT_FOUND" => Err(RefitBackendError::NotFound(
                "weight version was not found".to_string(),
            )),
            "VERSION_NOT_LEASEABLE" => Err(RefitBackendError::FailedPrecondition(
                "weight version does not accept this lease registration".to_string(),
            )),
            "WORKER_NOT_FOUND" => Err(RefitBackendError::FailedPrecondition(
                "worker registration is missing or expired".to_string(),
            )),
            "WORKER_NOT_GENERATOR" => Err(RefitBackendError::FailedPrecondition(
                "only a generator worker can hold a version lease".to_string(),
            )),
            "MODEL_MISMATCH" => Err(RefitBackendError::FailedPrecondition(
                "worker and weight version model_name differ".to_string(),
            )),
            "LEASE_CONFLICT" => Err(RefitBackendError::AlreadyExists(
                "version lease ID is already used by a different worker".to_string(),
            )),
            _ => Err(RefitBackendError::Internal(format!(
                "unexpected Redis response: {result}"
            ))),
        }
    }

    async fn delete_version_lease(&self, request: &DeleteVersionLeaseRequest) -> RefitResult<bool> {
        let mut redis = self.redis.clone();
        let result: String = Script::new(DELETE_LEASE_LUA)
            .key(lease_key(&request.version_id, &request.lease_id))
            .key(leases_key(&request.version_id))
            .arg(&request.lease_id)
            .arg(&request.version_id)
            .arg(&request.worker_id)
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            "DELETED" => Ok(true),
            "NOT_FOUND" => Ok(false),
            "LEASE_CONFLICT" => Err(RefitBackendError::FailedPrecondition(
                "version lease is owned by a different worker".to_string(),
            )),
            _ => Err(RefitBackendError::Internal(format!(
                "unexpected Redis response: {result}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_ids_cannot_collide_with_derived_keys() {
        assert_ne!(version_key("foo:shards"), shards_key("foo"));
        assert_ne!(version_key("foo:coverage"), coverage_key("foo"));
        assert_ne!(
            version_key("foo:expected-source-slots"),
            expected_source_slots_key("foo")
        );
        assert_ne!(version_key("foo:leases"), leases_key("foo"));
        assert_ne!(version_key("foo:lease:bar"), lease_key("foo", "bar"));
    }

    #[test]
    fn redis_errors_distinguish_transient_and_internal_failures() {
        let transient: redis::RedisError = (redis::ErrorKind::TryAgain, "retry").into();
        assert!(matches!(
            redis_error(transient),
            RefitBackendError::Unavailable(_)
        ));

        let internal: redis::RedisError =
            (redis::ErrorKind::ResponseError, "invalid script").into();
        assert!(matches!(
            redis_error(internal),
            RefitBackendError::Internal(_)
        ));
    }
}
