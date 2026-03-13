// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis backend for P2P model metadata storage.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, TensorRecord, WorkerRecord};
use async_trait::async_trait;
use modelexpress_common::grpc::p2p::WorkerMetadata;
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Redis key prefixes
mod keys {
    pub const MODEL_PREFIX: &str = "mx:model:";
    pub const MODELS_SET: &str = "mx:models";
}

/// Serializable version of TensorRecord for Redis storage
/// NOTE: addr and size are serialized as strings to avoid Lua cjson precision issues
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorRecordJson {
    pub name: String,
    #[serde(
        serialize_with = "serialize_u64_as_string",
        deserialize_with = "deserialize_u64_from_any"
    )]
    pub addr: u64,
    #[serde(
        serialize_with = "serialize_u64_as_string",
        deserialize_with = "deserialize_u64_from_any"
    )]
    pub size: u64,
    pub device_id: u32,
    pub dtype: String,
}

fn serialize_u64_as_string<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_string())
}

fn deserialize_u64_from_any<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct U64Visitor;

    impl<'de> Visitor<'de> for U64Visitor {
        type Value = u64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a u64 as string or number")
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            u64::try_from(value).map_err(|_| E::custom("negative value"))
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Handle floats from cjson (the problematic case)
            Ok(value as u64)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            value.parse::<u64>().map_err(de::Error::custom)
        }
    }

    deserializer.deserialize_any(U64Visitor)
}

impl From<TensorRecord> for TensorRecordJson {
    fn from(record: TensorRecord) -> Self {
        Self {
            name: record.name,
            addr: record.addr,
            size: record.size,
            device_id: record.device_id,
            dtype: record.dtype,
        }
    }
}

impl From<TensorRecordJson> for TensorRecord {
    fn from(json: TensorRecordJson) -> Self {
        Self {
            name: json.name,
            addr: json.addr,
            size: json.size,
            device_id: json.device_id,
            dtype: json.dtype,
        }
    }
}

/// Serializable version of WorkerRecord for Redis storage
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRecordJson {
    pub worker_rank: u32,
    pub nixl_metadata: Vec<u8>,
    pub tensors: Vec<TensorRecordJson>,
    #[serde(default)]
    pub status: i32,
    #[serde(default)]
    pub updated_at: i64,
}

impl From<WorkerRecord> for WorkerRecordJson {
    fn from(record: WorkerRecord) -> Self {
        Self {
            worker_rank: record.worker_rank,
            nixl_metadata: record.nixl_metadata,
            tensors: record
                .tensors
                .into_iter()
                .map(TensorRecordJson::from)
                .collect(),
            status: record.status,
            updated_at: record.updated_at,
        }
    }
}

impl From<WorkerRecordJson> for WorkerRecord {
    fn from(json: WorkerRecordJson) -> Self {
        Self {
            worker_rank: json.worker_rank,
            nixl_metadata: json.nixl_metadata,
            tensors: json.tensors.into_iter().map(TensorRecord::from).collect(),
            status: json.status,
            updated_at: json.updated_at,
        }
    }
}

/// Model metadata stored in Redis
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadataJson {
    pub model_name: String,
    pub workers: Vec<WorkerRecordJson>,
    pub published_at: i64,
}

impl From<ModelMetadataJson> for ModelMetadataRecord {
    fn from(json: ModelMetadataJson) -> Self {
        Self {
            model_name: json.model_name,
            workers: json.workers.into_iter().map(WorkerRecord::from).collect(),
            published_at: json.published_at,
        }
    }
}

/// Redis backend for metadata storage
pub struct RedisBackend {
    redis: Arc<RwLock<Option<ConnectionManager>>>,
    redis_url: String,
}

impl RedisBackend {
    /// Create a new Redis backend
    pub fn new(redis_url: &str) -> Self {
        Self {
            redis: Arc::new(RwLock::new(None)),
            redis_url: redis_url.to_string(),
        }
    }

    /// Get a Redis connection, reconnecting if necessary
    async fn get_conn(&self) -> MetadataResult<ConnectionManager> {
        // Fast path: read lock
        {
            let guard = self.redis.read().await;
            if let Some(conn) = guard.as_ref() {
                return Ok(conn.clone());
            }
        }

        // Slow path: write lock with double-check
        let mut guard = self.redis.write().await;
        if let Some(conn) = guard.as_ref() {
            return Ok(conn.clone());
        }

        let client = redis::Client::open(self.redis_url.as_str())?;
        let conn = ConnectionManager::new(client).await?;
        *guard = Some(conn.clone());
        Ok(conn)
    }
}

#[async_trait]
impl MetadataBackend for RedisBackend {
    async fn connect(&self) -> MetadataResult<()> {
        let client = redis::Client::open(self.redis_url.as_str())?;
        let conn = ConnectionManager::new(client).await?;

        let mut guard = self.redis.write().await;
        *guard = Some(conn);

        // Redact credentials from URL before logging (e.g., redis://user:pass@host → redis://user:***@host)
        let safe_url = if self.redis_url.contains('@') {
            // Has credentials — redact password between : and @
            if let Some(at_pos) = self.redis_url.rfind('@') {
                let prefix = &self.redis_url[..at_pos];
                let suffix = &self.redis_url[at_pos..];
                if let Some(colon_pos) = prefix.rfind(':') {
                    format!("{}:***{}", &prefix[..colon_pos], suffix)
                } else {
                    self.redis_url.clone()
                }
            } else {
                self.redis_url.clone()
            }
        } else {
            self.redis_url.clone()
        };
        info!("Connected to Redis at {}", safe_url);
        Ok(())
    }

    async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}", keys::MODEL_PREFIX, model_name);

        // Convert new workers to records and serialize
        let new_workers: Vec<WorkerRecordJson> = workers
            .into_iter()
            .map(|w| WorkerRecordJson::from(WorkerRecord::from(w)))
            .collect();
        let new_workers_json = serde_json::to_string(&new_workers)?;
        let timestamp = chrono::Utc::now().timestamp();

        // Lua script for atomic read-modify-write merge
        let script = redis::Script::new(
            r#"
            local key = KEYS[1]
            local models_set = ARGV[1]
            local model_name = ARGV[2]
            local new_workers_json = ARGV[3]
            local timestamp = tonumber(ARGV[4])

            local new_workers = cjson.decode(new_workers_json)

            local existing_json = redis.call('GET', key)
            local existing_workers = {}

            if existing_json then
                local existing = cjson.decode(existing_json)
                if existing.workers then
                    existing_workers = existing.workers
                end
            end

            for _, new_worker in ipairs(new_workers) do
                local found = false
                for i, existing_worker in ipairs(existing_workers) do
                    if existing_worker.worker_rank == new_worker.worker_rank then
                        existing_workers[i] = new_worker
                        found = true
                        break
                    end
                end
                if not found then
                    table.insert(existing_workers, new_worker)
                end
            end

            table.sort(existing_workers, function(a, b)
                return a.worker_rank < b.worker_rank
            end)

            local record = {
                model_name = model_name,
                workers = existing_workers,
                published_at = timestamp
            }

            redis.call('SET', key, cjson.encode(record))
            redis.call('SADD', models_set, model_name)
            return #existing_workers
            "#,
        );

        let worker_count: i32 = script
            .key(&key)
            .arg(keys::MODELS_SET)
            .arg(model_name)
            .arg(&new_workers_json)
            .arg(timestamp)
            .invoke_async(&mut conn)
            .await?;

        let total_tensors: usize = new_workers.iter().map(|w| w.tensors.len()).sum();
        info!(
            "Published metadata for model '{}': {} workers total ({} new tensors)",
            model_name, worker_count, total_tensors
        );
        Ok(())
    }

    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}", keys::MODEL_PREFIX, model_name);
        let json: Option<String> = conn.get(&key).await?;

        match json {
            Some(data) => {
                let record: ModelMetadataJson = serde_json::from_str(&data)?;
                debug!("Retrieved metadata for model '{}'", model_name);
                Ok(Some(ModelMetadataRecord::from(record)))
            }
            None => {
                debug!("No metadata found for model '{}'", model_name);
                Ok(None)
            }
        }
    }

    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}", keys::MODEL_PREFIX, model_name);

        conn.del::<_, ()>(&key).await?;
        conn.srem::<_, _, ()>(keys::MODELS_SET, model_name).await?;

        info!("Removed metadata for model '{}'", model_name);
        Ok(())
    }

    async fn list_models(&self) -> MetadataResult<Vec<String>> {
        let mut conn = self.get_conn().await?;
        let models: Vec<String> = conn.smembers(keys::MODELS_SET).await?;
        Ok(models)
    }

    async fn update_status(
        &self,
        model_name: &str,
        worker_id: u32,
        status: i32,
        updated_at: i64,
    ) -> MetadataResult<()> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}", keys::MODEL_PREFIX, model_name);

        // Lua script: patch status + updated_at for a specific worker in the model record
        let script = redis::Script::new(
            r#"
            local key = KEYS[1]
            local worker_rank = tonumber(ARGV[1])
            local new_status = tonumber(ARGV[2])
            local new_updated_at = tonumber(ARGV[3])

            local json = redis.call('GET', key)
            if not json then return 0 end

            local record = cjson.decode(json)
            for _, worker in ipairs(record.workers) do
                if worker.worker_rank == worker_rank then
                    worker.status = new_status
                    worker.updated_at = new_updated_at
                    redis.call('SET', key, cjson.encode(record))
                    return 1
                end
            end
            return 0
            "#,
        );

        let patched: i32 = script
            .key(&key)
            .arg(worker_id)
            .arg(status)
            .arg(updated_at)
            .invoke_async(&mut conn)
            .await?;

        check_patched(patched, worker_id, model_name)?;
        debug!(
            "Updated status for model '{}' worker {}",
            model_name, worker_id
        );
        Ok(())
    }
}

/// Return `Ok(())` when `patched == 1`, or a descriptive error when `patched == 0`
/// (model key absent or worker not found).
fn check_patched(patched: i32, worker_id: u32, model_name: &str) -> MetadataResult<()> {
    if patched == 0 {
        return Err(format!(
            "update_status: worker {} not found in model '{}'",
            worker_id, model_name
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    // ── TensorRecordJson serialization ──────────────────────────────────────

    #[test]
    fn test_tensor_record_json_roundtrip() {
        let record = TensorRecord {
            name: "model.layers.0.weight".to_string(),
            addr: 0x7f00_0000_0000,
            size: 1_073_741_824,
            device_id: 3,
            dtype: "bfloat16".to_string(),
        };
        let json_record = TensorRecordJson::from(record.clone());
        let json = serde_json::to_string(&json_record).expect("serialize");

        // addr and size must be serialized as strings
        assert!(json.contains(r#""addr":"#));
        let parsed: TensorRecordJson = serde_json::from_str(&json).expect("deserialize");
        let back = TensorRecord::from(parsed);

        assert_eq!(back.name, record.name);
        assert_eq!(back.addr, record.addr);
        assert_eq!(back.size, record.size);
        assert_eq!(back.device_id, record.device_id);
        assert_eq!(back.dtype, record.dtype);
    }

    #[test]
    fn test_deserialize_u64_from_string() {
        let json = r#"{"name":"w","addr":"139948187451390","size":"134217728","device_id":0,"dtype":"f16"}"#;
        let t: TensorRecordJson = serde_json::from_str(json).expect("parse string");
        assert_eq!(t.addr, 139948187451390);
        assert_eq!(t.size, 134217728);
    }

    #[test]
    fn test_deserialize_u64_from_number() {
        let json = r#"{"name":"w","addr":1234567890,"size":4096,"device_id":0,"dtype":"f16"}"#;
        let t: TensorRecordJson = serde_json::from_str(json).expect("parse number");
        assert_eq!(t.addr, 1234567890);
    }

    #[test]
    fn test_deserialize_u64_from_float() {
        // cjson can emit floats for large integers
        let json = r#"{"name":"w","addr":1048576.0,"size":4096.0,"device_id":0,"dtype":"f16"}"#;
        let t: TensorRecordJson = serde_json::from_str(json).expect("parse float");
        assert_eq!(t.addr, 1048576);
    }

    // ── WorkerRecordJson serialization ──────────────────────────────────────

    #[test]
    fn test_worker_record_json_roundtrip_with_status() {
        let record = WorkerRecord {
            worker_rank: 2,
            nixl_metadata: vec![0xde, 0xad, 0xbe, 0xef],
            tensors: vec![TensorRecord {
                name: "t".to_string(),
                addr: 0x1000,
                size: 512,
                device_id: 2,
                dtype: "float16".to_string(),
            }],
            status: 2, // SOURCE_STATUS_READY
            updated_at: 1_700_000_000_000,
        };

        let json_record = WorkerRecordJson::from(record.clone());
        let json = serde_json::to_string(&json_record).expect("serialize");
        let parsed: WorkerRecordJson = serde_json::from_str(&json).expect("deserialize");
        let back = WorkerRecord::from(parsed);

        assert_eq!(back.worker_rank, record.worker_rank);
        assert_eq!(back.nixl_metadata, record.nixl_metadata);
        assert_eq!(back.status, record.status);
        assert_eq!(back.updated_at, record.updated_at);
        assert_eq!(back.tensors.len(), 1);
    }

    #[test]
    fn test_worker_record_json_backward_compat_missing_status() {
        // Records written before status/updated_at fields existed must default to 0
        let json = r#"{"worker_rank":0,"nixl_metadata":[],"tensors":[]}"#;
        let parsed: WorkerRecordJson = serde_json::from_str(json).expect("parse legacy");
        assert_eq!(parsed.status, 0);
        assert_eq!(parsed.updated_at, 0);
    }

    // ── check_patched ────────────────────────────────────────────────────────

    #[test]
    fn test_check_patched_ok() {
        assert!(check_patched(1, 0, "my-model").is_ok());
    }

    #[test]
    fn test_check_patched_zero_returns_err() {
        let err = check_patched(0, 3, "my-model").expect_err("expected Err for patched==0");
        let msg = err.to_string();
        assert!(
            msg.contains("worker 3"),
            "error should name the worker: {msg}"
        );
        assert!(
            msg.contains("my-model"),
            "error should name the model: {msg}"
        );
    }

    #[test]
    fn test_check_patched_zero_unknown_worker() {
        // Absent model key also returns 0 from the Lua script.
        let err = check_patched(0, 0, "nonexistent-model").expect_err("absent model should be Err");
        let msg = err.to_string();
        assert!(
            msg.contains("nonexistent-model"),
            "error should name the model: {msg}"
        );
    }

    // ── ModelMetadataJson serialization ─────────────────────────────────────

    #[test]
    fn test_model_metadata_json_roundtrip() {
        let json = r#"{
            "model_name": "meta-llama/Llama-3.1-70B",
            "workers": [
                {"worker_rank": 0, "nixl_metadata": [], "tensors": [], "status": 2, "updated_at": 1000}
            ],
            "published_at": 1700000000
        }"#;
        let parsed: ModelMetadataJson = serde_json::from_str(json).expect("parse");
        let record = ModelMetadataRecord::from(parsed);

        assert_eq!(record.model_name, "meta-llama/Llama-3.1-70B");
        assert_eq!(record.workers.len(), 1);
        assert_eq!(record.workers[0].status, 2);
        assert_eq!(record.published_at, 1700000000);
    }
}
