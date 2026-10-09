// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis implementation of the Refit control-plane backend.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use modelexpress_common::grpc::refit::{
    CreateTrainerMeshRequest, CreateWeightVersionRequest, DeleteVersionLeaseRequest,
    DeleteWeightVersionShardRequest, ObjectStorageSource, ObjectStorageType,
    RegisterVersionLeaseRequest, TrainerMesh, TrainerTensorsMetadata,
    UpdateWeightVersionStateRequest, VersionLease, WeightVersion, WeightVersionShard,
    WeightVersionState, WorkerRegistration, WorkerRole,
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
const CREATE_MESH_LUA: &str = include_str!("redis/scripts/create_trainer_mesh.lua");
const UPDATE_MESH_LUA: &str = include_str!("redis/scripts/update_trainer_mesh.lua");
const DELETE_MESH_LUA: &str = include_str!("redis/scripts/delete_trainer_mesh.lua");

fn mesh_key(mesh_id: &str) -> String {
    format!("mx:refit:trainer-mesh:metadata:{mesh_id}")
}

fn mesh_versions_key(mesh_id: &str) -> String {
    format!("mx:refit:trainer-mesh:versions:{mesh_id}")
}

fn mesh_idempotency_key(model_name: &str, request_key: &str) -> String {
    format!("mx:refit:trainer-mesh-request:{model_name}:{request_key}")
}

fn version_key(version_id: &str) -> String {
    format!("mx:refit:version:metadata:{version_id}")
}

fn shards_key(version_id: &str) -> String {
    format!("mx:refit:version:shards:{version_id}")
}

fn publication_key(worker_id: &str, logical_shard_id: &str) -> String {
    format!("{}:{worker_id}{logical_shard_id}", worker_id.len())
}

fn publication_endpoints_key(version_id: &str) -> String {
    format!("mx:refit:version:publication-endpoints:{version_id}")
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
    let trainer_mesh_id = fields
        .get("trainer_mesh_id")
        .filter(|mesh_id| !mesh_id.is_empty())
        .cloned();
    let trainer_mesh_generation = if trainer_mesh_id.is_some() {
        fields
            .get("trainer_mesh_generation")
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|generation| *generation > 0)
            .ok_or_else(|| {
                RefitBackendError::FailedPrecondition(
                    "mesh-backed weight version is missing a positive trainer_mesh_generation; recreate the version".to_string(),
                )
            })?
    } else {
        if fields
            .get("trainer_mesh_generation")
            .is_some_and(|value| value != "0")
        {
            return Err(RefitBackendError::FailedPrecondition(
                "weight version without a trainer mesh must have generation zero".to_string(),
            ));
        }
        0
    };
    Ok(WeightVersion {
        uid: hash_field(&fields, "uid")?.to_string(),
        model_name: hash_field(&fields, "model_name")?.to_string(),
        idempotency_key: hash_field(&fields, "idempotency_key")?.to_string(),
        payload_format: parse_hash_field(&fields, "payload_format")?,
        base_version_id: match hash_field(&fields, "base_version_id")? {
            "" => None,
            value => Some(value.to_string()),
        },
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
        trainer_mesh_id,
        trainer_mesh_generation,
        version_number: fields
            .get("version_number")
            .filter(|number| !number.is_empty())
            .map(|number| {
                number.parse().map_err(|error| {
                    RefitBackendError::Internal(format!("invalid version_number: {error}"))
                })
            })
            .transpose()?,
    })
}

fn mesh_from_hash(fields: HashMap<String, String>) -> RefitResult<TrainerMesh> {
    let workers = decode_mesh_workers(hash_field(&fields, "workers")?)?;
    Ok(TrainerMesh {
        mesh_id: hash_field(&fields, "mesh_id")?.to_string(),
        model_name: hash_field(&fields, "model_name")?.to_string(),
        generation: parse_hash_field(&fields, "generation")?,
        workers,
    })
}

fn mesh_workers_json(workers: &HashMap<String, TrainerTensorsMetadata>) -> RefitResult<String> {
    let workers: HashMap<&str, serde_json::Value> = workers
        .iter()
        .map(|(worker_id, metadata)| {
            (
                worker_id.as_str(),
                serde_json::json!({
                    "logical_shard_id": metadata.logical_shard_id,
                    "metadata_endpoint": metadata.metadata_endpoint,
                }),
            )
        })
        .collect();
    serde_json::to_string(&workers)
        .map_err(|error| RefitBackendError::Internal(format!("encode mesh workers: {error}")))
}

fn decode_mesh_workers(encoded: &str) -> RefitResult<HashMap<String, TrainerTensorsMetadata>> {
    let workers: HashMap<String, HashMap<String, String>> = serde_json::from_str(encoded)
        .map_err(|error| RefitBackendError::Internal(format!("invalid mesh workers: {error}")))?;
    workers
        .into_iter()
        .map(|(worker_id, metadata)| {
            Ok((
                worker_id,
                TrainerTensorsMetadata {
                    logical_shard_id: hash_field(&metadata, "logical_shard_id")?.to_string(),
                    metadata_endpoint: hash_field(&metadata, "metadata_endpoint")?.to_string(),
                },
            ))
        })
        .collect()
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
        let script = Script::new(CREATE_VERSION_LUA);
        let mut invocation = script.prepare_invoke();
        invocation
            .key(version_key(uid))
            .key(idempotency_key(
                &request.model_name,
                &request.idempotency_key,
            ))
            .key(mesh_key(
                request.trainer_mesh_id.as_deref().unwrap_or_default(),
            ))
            .key(mesh_versions_key(
                request.trainer_mesh_id.as_deref().unwrap_or_default(),
            ))
            .arg(uid)
            .arg(&request.model_name)
            .arg(&request.idempotency_key)
            .arg(request.payload_format)
            .arg(request.base_version_id.as_deref().unwrap_or_default())
            .arg(
                request
                    .object_storage
                    .as_ref()
                    .map_or("", |source| source.uri.as_str()),
            )
            .arg(request.state)
            .arg(request.state)
            .arg(now_unix_ms()?)
            .arg(request.trainer_mesh_id.as_deref().unwrap_or_default());
        invocation.arg(
            request
                .version_number
                .map(|number| number.to_string())
                .unwrap_or_default(),
        );
        let mut redis = self.redis.clone();
        invocation
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)
    }
}

#[async_trait]
impl RefitBackend for RedisRefitBackend {
    async fn validate_trainer_mesh_workers(
        &self,
        model_name: &str,
        workers: &HashMap<String, TrainerTensorsMetadata>,
    ) -> RefitResult<()> {
        let members: Vec<_> = workers.iter().collect();
        let mut pipeline = redis::pipe();
        for (worker_id, _) in &members {
            pipeline.hgetall(worker_key(worker_id));
        }
        let registrations: Vec<HashMap<String, String>> = pipeline
            .query_async(&mut self.redis.clone())
            .await
            .map_err(redis_error)?;
        let trainer_role = i32::from(WorkerRole::Trainer).to_string();
        for ((worker_id, metadata), registration) in members.into_iter().zip(registrations) {
            if registration.get("worker_id") != Some(worker_id)
                || registration.get("model_name").map(String::as_str) != Some(model_name)
                || registration.get("role") != Some(&trainer_role)
                || registration.get("refit_endpoint") != Some(&metadata.metadata_endpoint)
            {
                return Err(RefitBackendError::FailedPrecondition(format!(
                    "mesh worker {worker_id:?} requires an active trainer registration for the model and binding endpoint"
                )));
            }
        }
        Ok(())
    }

    async fn find_trainer_mesh_for_request(
        &self,
        request: &CreateTrainerMeshRequest,
    ) -> RefitResult<Option<TrainerMesh>> {
        let mut redis = self.redis.clone();
        let existing_id: Option<String> = redis
            .get(mesh_idempotency_key(
                &request.model_name,
                &request.idempotency_key,
            ))
            .await
            .map_err(redis_error)?;
        let Some(existing_id) = existing_id else {
            return Ok(None);
        };
        let fields: HashMap<String, String> = redis
            .hgetall(mesh_key(&existing_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Err(RefitBackendError::AlreadyExists(
                "idempotency_key belongs to a deleted TrainerMesh".to_string(),
            ));
        }
        let initial_workers = decode_mesh_workers(hash_field(&fields, "initial_workers")?)?;
        let existing = mesh_from_hash(fields)?;
        if existing.model_name != request.model_name || initial_workers != request.workers {
            return Err(RefitBackendError::AlreadyExists(
                "idempotency_key was already used for a different TrainerMesh".to_string(),
            ));
        }
        Ok(Some(existing))
    }
    async fn create_trainer_mesh(
        &self,
        request: &CreateTrainerMeshRequest,
    ) -> RefitResult<TrainerMesh> {
        let logical_shards: std::collections::BTreeSet<_> = request
            .workers
            .values()
            .map(|metadata| &metadata.logical_shard_id)
            .collect();
        let logical_shards = serde_json::to_string(&logical_shards).map_err(|error| {
            RefitBackendError::Internal(format!("encode mesh logical_shards: {error}"))
        })?;
        let workers = mesh_workers_json(&request.workers)?;
        for _ in 0..5 {
            let mesh_id: String = Uuid::new_v4()
                .simple()
                .to_string()
                .chars()
                .take(8)
                .collect();
            let mut redis = self.redis.clone();
            let script = Script::new(CREATE_MESH_LUA);
            let mut invocation = script.prepare_invoke();
            invocation
                .key(mesh_key(&mesh_id))
                .key(mesh_idempotency_key(
                    &request.model_name,
                    &request.idempotency_key,
                ))
                .arg(&mesh_id)
                .arg(&request.model_name)
                .arg(&logical_shards)
                .arg(&workers)
                .arg(i32::from(WorkerRole::Trainer));
            for worker_id in request.workers.keys() {
                invocation.key(worker_key(worker_id));
            }
            let result: String = invocation
                .invoke_async(&mut redis)
                .await
                .map_err(redis_error)?;
            if result == "CREATED" {
                return self.get_trainer_mesh(&mesh_id).await;
            }
            if result.starts_with("EXISTING:") {
                return self
                    .find_trainer_mesh_for_request(request)
                    .await?
                    .ok_or_else(|| {
                        RefitBackendError::Internal("mesh idempotency key disappeared".to_string())
                    });
            }
            if result == "WORKER_NOT_FOUND" || result == "WORKER_MISMATCH" {
                return Err(RefitBackendError::FailedPrecondition(
                    "mesh workers require active trainer registrations for the model".to_string(),
                ));
            }
            if result != "COLLISION" {
                return Err(RefitBackendError::Internal(format!(
                    "unexpected Redis response: {result}"
                )));
            }
        }
        Err(RefitBackendError::ResourceExhausted(
            "could not allocate a unique trainer mesh ID".to_string(),
        ))
    }

    async fn get_trainer_mesh(&self, mesh_id: &str) -> RefitResult<TrainerMesh> {
        let mut redis = self.redis.clone();
        let fields: HashMap<String, String> = redis
            .hgetall(mesh_key(mesh_id))
            .await
            .map_err(redis_error)?;
        if fields.is_empty() {
            return Err(RefitBackendError::NotFound(format!(
                "trainer mesh {mesh_id:?} was not found"
            )));
        }
        mesh_from_hash(fields)
    }

    async fn update_trainer_mesh(
        &self,
        mesh_id: &str,
        expected_generation: u64,
        workers: HashMap<String, TrainerTensorsMetadata>,
    ) -> RefitResult<TrainerMesh> {
        let current = self.get_trainer_mesh(mesh_id).await?;
        if current.generation != expected_generation {
            return Err(RefitBackendError::FailedPrecondition(
                "trainer mesh generation changed".to_string(),
            ));
        }
        let script = Script::new(UPDATE_MESH_LUA);
        let mut invocation = script.prepare_invoke();
        invocation
            .key(mesh_key(mesh_id))
            .key(mesh_versions_key(mesh_id))
            .arg(expected_generation)
            .arg(mesh_workers_json(&workers)?)
            .arg(&current.model_name)
            .arg(i32::from(WorkerRole::Trainer));
        for worker_id in workers.keys() {
            invocation.key(worker_key(worker_id));
        }
        let mut redis = self.redis.clone();
        let result: String = invocation
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            "UPDATED" | "UNCHANGED" => self.get_trainer_mesh(mesh_id).await,
            "COVERAGE_MISMATCH" => Err(RefitBackendError::InvalidArgument(
                "mesh update must preserve logical shard coverage".to_string(),
            )),
            "NOT_FOUND" => Err(RefitBackendError::NotFound(
                "trainer mesh was not found".to_string(),
            )),
            "GENERATION_MISMATCH" => Err(RefitBackendError::FailedPrecondition(
                "trainer mesh generation changed".to_string(),
            )),
            "WORKER_NOT_FOUND" | "WORKER_MISMATCH" => Err(RefitBackendError::FailedPrecondition(
                "new mesh worker must have an active trainer registration for the model"
                    .to_string(),
            )),
            "VERSION_LEASED" => Err(RefitBackendError::FailedPrecondition(
                "cannot rebind a worker endpoint while a linked version has an active lease"
                    .to_string(),
            )),
            _ => Err(RefitBackendError::Internal(format!(
                "unexpected Redis response: {result}"
            ))),
        }
    }

    async fn delete_trainer_mesh(&self, mesh_id: &str) -> RefitResult<()> {
        let mut redis = self.redis.clone();
        let result: String = Script::new(DELETE_MESH_LUA)
            .key(mesh_key(mesh_id))
            .key(mesh_versions_key(mesh_id))
            .arg(i32::from(WeightVersionState::Releasing))
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            "DELETED" => Ok(()),
            "NOT_FOUND" => Err(RefitBackendError::NotFound(
                "trainer mesh was not found".to_string(),
            )),
            "IN_USE" => Err(RefitBackendError::FailedPrecondition(
                "trainer mesh still has active weight versions or resources".to_string(),
            )),
            _ => Err(RefitBackendError::Internal(format!(
                "unexpected Redis response: {result}"
            ))),
        }
    }

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
            .arg(&worker.refit_endpoint)
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
                    && existing.object_storage == request.object_storage
                    && existing.trainer_mesh_id == request.trainer_mesh_id
                    && existing.version_number == request.version_number
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
            if result == "MESH_NOT_FOUND" {
                return Err(RefitBackendError::NotFound(
                    "trainer mesh was not found".to_string(),
                ));
            }
            if result == "MESH_MODEL_MISMATCH" {
                return Err(RefitBackendError::FailedPrecondition(
                    "trainer mesh and weight version model_name differ".to_string(),
                ));
            }
            if result == "MESH_GENERATION_MISSING" {
                return Err(RefitBackendError::FailedPrecondition(
                    "trainer mesh is missing a positive generation".to_string(),
                ));
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

    async fn list_weight_versions(
        &self,
        model_name: &str,
        trainer_mesh_id: Option<&str>,
    ) -> RefitResult<Vec<WeightVersion>> {
        let mut redis = self.redis.clone();
        let keys = if let Some(mesh_id) = trainer_mesh_id {
            let ids: Vec<String> = redis
                .smembers(mesh_versions_key(mesh_id))
                .await
                .map_err(redis_error)?;
            ids.iter().map(|id| version_key(id)).collect::<Vec<_>>()
        } else {
            let mut cursor = 0_u64;
            let mut keys = Vec::new();
            loop {
                let (next, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg("mx:refit:version:metadata:*")
                    .arg("COUNT")
                    .arg(100)
                    .query_async(&mut redis)
                    .await
                    .map_err(redis_error)?;
                keys.extend(batch);
                cursor = next;
                if cursor == 0 {
                    break;
                }
            }
            keys
        };
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut pipeline = redis::pipe();
        for key in keys {
            pipeline.hgetall(key);
        }
        let records: Vec<HashMap<String, String>> = pipeline
            .query_async(&mut redis)
            .await
            .map_err(redis_error)?;
        let mut versions = HashMap::new();
        for fields in records {
            if fields.is_empty() {
                continue;
            }
            let version = version_from_hash(fields)?;
            if version.model_name == model_name
                && trainer_mesh_id
                    .is_none_or(|mesh_id| version.trainer_mesh_id.as_deref() == Some(mesh_id))
            {
                versions.insert(version.uid.clone(), version);
            }
        }
        let mut versions: Vec<_> = versions.into_values().collect();
        versions
            .sort_by(|a, b| (&b.created_at_unix_ms, &b.uid).cmp(&(&a.created_at_unix_ms, &a.uid)));
        Ok(versions)
    }

    async fn delete_weight_version(&self, uid: &str) -> RefitResult<WeightVersion> {
        self.transition_weight_version_state(uid, WeightVersionState::Releasing.into())
            .await
    }

    async fn update_weight_version_state(
        &self,
        request: &UpdateWeightVersionStateRequest,
    ) -> RefitResult<WeightVersion> {
        if self
            .get_weight_version(&request.uid)
            .await?
            .trainer_mesh_id
            .is_some()
        {
            return Err(RefitBackendError::FailedPrecondition(
                "mesh-backed weight version state is managed by MX".to_string(),
            ));
        }
        self.transition_weight_version_state(&request.uid, request.state)
            .await
    }

    async fn create_weight_version_shard(
        &self,
        shard: WeightVersionShard,
    ) -> RefitResult<(WeightVersionShard, WeightVersion)> {
        let mut version = self.get_weight_version(&shard.version_id).await?;
        let publication_key = publication_key(&shard.worker_id, &shard.logical_shard_id);
        let encoded = shard.encode_to_vec();
        let mut redis = self.redis.clone();
        let result: String = Script::new(CREATE_SHARD_LUA)
            .key(version_key(&shard.version_id))
            .key(worker_key(&shard.worker_id))
            .key(shards_key(&shard.version_id))
            .key(mesh_key(
                version.trainer_mesh_id.as_deref().unwrap_or_default(),
            ))
            .key(publication_endpoints_key(&shard.version_id))
            .arg(publication_key)
            .arg(encoded)
            .arg(&version.model_name)
            .arg(&shard.logical_shard_id)
            .arg(i32::from(WeightVersionState::Staging))
            .arg(i32::from(WeightVersionState::Ready))
            .arg(&shard.worker_id)
            .arg(i32::from(WorkerRole::Trainer))
            .arg(&shard.manifest_endpoint)
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
            "MESH_NOT_FOUND" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "trainer mesh is missing".to_string(),
                ));
            }
            "MESH_GENERATION_MISSING" | "MESH_GENERATION_MISMATCH" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "weight version trainer_mesh_generation does not match the current mesh; create a new version".to_string(),
                ));
            }
            "WORKER_NOT_TRAINER" | "WORKER_NOT_IN_MESH" | "WORKER_ENDPOINT_MISMATCH" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "worker is not an active trainer member of this logical shard".to_string(),
                ));
            }
            "VERSION_NOT_WRITABLE" => {
                return Err(RefitBackendError::FailedPrecondition(
                    "weight version does not accept shard publication".to_string(),
                ));
            }
            "SHARD_CONFLICT" => {
                return Err(RefitBackendError::AlreadyExists(
                    "worker and logical_shard_id already published different metadata".to_string(),
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
        let version = self.get_weight_version(version_id).await?;
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
        if let Some(mesh_id) = version.trainer_mesh_id.as_deref() {
            let mesh = self.get_trainer_mesh(mesh_id).await?;
            shards.retain(|shard| {
                mesh.workers.get(&shard.worker_id).is_some_and(|metadata| {
                    metadata.logical_shard_id == shard.logical_shard_id
                        && metadata.metadata_endpoint == shard.manifest_endpoint
                })
            });
            if !shards.is_empty() {
                let mut registrations = redis::pipe();
                for shard in &shards {
                    registrations.hget(worker_key(&shard.worker_id), "refit_endpoint");
                }
                let endpoints: Vec<Option<String>> = registrations
                    .query_async(&mut redis)
                    .await
                    .map_err(redis_error)?;
                shards = shards
                    .into_iter()
                    .zip(endpoints)
                    .filter_map(|(shard, endpoint)| {
                        let metadata = mesh.workers.get(&shard.worker_id)?;
                        (endpoint.as_deref() == Some(metadata.metadata_endpoint.as_str()))
                            .then_some(shard)
                    })
                    .collect();
            }
        }
        shards.sort_by(|left, right| {
            (&left.logical_shard_id, &left.worker_id)
                .cmp(&(&right.logical_shard_id, &right.worker_id))
        });
        Ok(shards)
    }

    async fn delete_weight_version_shard(
        &self,
        request: &DeleteWeightVersionShardRequest,
    ) -> RefitResult<bool> {
        let publication_key = publication_key(&request.worker_id, &request.logical_shard_id);
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
            .key(publication_endpoints_key(&request.version_id))
            .arg(publication_key)
            .arg(encoded)
            .invoke_async(&mut redis)
            .await
            .map_err(redis_error)?;
        match result.as_str() {
            "DELETED" => Ok(true),
            "VERSION_NOT_FOUND" => Err(RefitBackendError::NotFound(
                "weight version was not found".to_string(),
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
        self.get_weight_version(&request.version_id).await?;
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

    fn stored_version(mesh_id: &str, generation: Option<&str>) -> HashMap<String, String> {
        let mut fields = HashMap::from([
            ("uid".to_string(), "version".to_string()),
            ("model_name".to_string(), "model".to_string()),
            ("idempotency_key".to_string(), "request".to_string()),
            ("payload_format".to_string(), "1".to_string()),
            ("base_version_id".to_string(), String::new()),
            ("layout_signature".to_string(), String::new()),
            ("state".to_string(), "1".to_string()),
            ("created_at_unix_ms".to_string(), "123".to_string()),
            ("trainer_mesh_id".to_string(), mesh_id.to_string()),
        ]);
        if let Some(generation) = generation {
            fields.insert(
                "trainer_mesh_generation".to_string(),
                generation.to_string(),
            );
        }
        fields
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn stored_mesh_versions_require_positive_generation() {
        for generation in [None, Some("0"), Some("invalid")] {
            assert!(matches!(
                version_from_hash(stored_version("mesh", generation)),
                Err(RefitBackendError::FailedPrecondition(_))
            ));
        }
        let version = version_from_hash(stored_version("mesh", Some("18446744073709551615")))
            .expect("positive generation decodes");
        assert_eq!(version.trainer_mesh_generation, u64::MAX);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn stored_nonmesh_versions_have_generation_zero() {
        let version = version_from_hash(stored_version("", Some("0")))
            .expect("generation is not applicable without a mesh");
        assert_eq!(version.trainer_mesh_generation, 0);
        assert!(matches!(
            version_from_hash(stored_version("", Some("1"))),
            Err(RefitBackendError::FailedPrecondition(_))
        ));
    }

    #[test]
    fn version_ids_cannot_collide_with_derived_keys() {
        assert_ne!(mesh_key("versions:foo"), mesh_versions_key("foo"));
        assert_ne!(version_key("foo:shards"), shards_key("foo"));
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
