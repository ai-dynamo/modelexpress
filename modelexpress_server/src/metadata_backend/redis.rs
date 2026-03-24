// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis backend for P2P model metadata storage.
//!
//! Storage layout:
//!   `mx:source:{source_id}`               — Redis Hash; field `__attributes__` stores
//!                                            JSON-serialized SourceAttributesJson (once
//!                                            per source); each other field is an worker_id
//!                                            with an empty-string value (presence marker).
//!   `mx:source:{source_id}:{worker_id}` — Redis Hash; field = worker_rank (string),
//!                                            value = JSON-serialized WorkerRecordJson.
//!
//! Global listing uses SCAN with pattern `mx:source:????????????????` (16-char source IDs)
//! to enumerate source index keys without a separate secondary index.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, TensorRecord, WorkerRecord};
use async_trait::async_trait;
use modelexpress_common::grpc::p2p::WorkerMetadata;
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Redis key prefixes and reserved field names
mod keys {
    pub const SOURCE_PREFIX: &str = "mx:source:";
    /// SCAN pattern matching source index keys: `mx:source:{16-char-id}`
    pub const SOURCE_SCAN_PATTERN: &str = "mx:source:????????????????";
    /// Reserved hash field in the source index key that stores SourceAttributesJson.
    pub const ATTRIBUTES_FIELD: &str = "__attributes__";
}

/// All fields of a SourceIdentity stored once per source in the index hash.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SourceAttributesJson {
    pub model_name: String,
    #[serde(default)]
    pub mx_version: String,
    #[serde(default)]
    pub mx_source_type: i32,
    #[serde(default)]
    pub backend_framework: i32,
    #[serde(default)]
    pub tensor_parallel_size: u32,
    #[serde(default)]
    pub pipeline_parallel_size: u32,
    #[serde(default)]
    pub expert_parallel_size: u32,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub quantization: String,
}

impl From<&SourceIdentity> for SourceAttributesJson {
    fn from(id: &SourceIdentity) -> Self {
        Self {
            model_name: id.model_name.clone(),
            mx_version: id.mx_version.clone(),
            mx_source_type: id.mx_source_type,
            backend_framework: id.backend_framework,
            tensor_parallel_size: id.tensor_parallel_size,
            pipeline_parallel_size: id.pipeline_parallel_size,
            expert_parallel_size: id.expert_parallel_size,
            dtype: id.dtype.clone(),
            quantization: id.quantization.clone(),
        }
    }
}

/// Scan Redis for all keys matching `pattern`, iterating through all SCAN cursors.
async fn scan_keys(conn: &mut ConnectionManager, pattern: &str) -> MetadataResult<Vec<String>> {
    let mut all_keys = Vec::new();
    let mut cursor: u64 = 0;
    loop {
        let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(pattern)
            .arg("COUNT")
            .arg(100)
            .query_async(conn)
            .await?;
        all_keys.extend(batch);
        cursor = next_cursor;
        if cursor == 0 {
            break;
        }
    }
    Ok(all_keys)
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

/// Serializable version of WorkerRecord stored as a hash field value
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRecordJson {
    pub worker_rank: u32,
    /// Explicit backend type discriminator ("nixl", "transfer_engine", "none").
    #[serde(default)]
    pub backend_type: Option<String>,
    #[serde(default)]
    pub nixl_metadata: Vec<u8>,
    #[serde(default)]
    pub transfer_engine_session_id: Option<String>,
    pub tensors: Vec<TensorRecordJson>,
    #[serde(default)]
    pub status: i32,
    #[serde(default)]
    pub updated_at: i64,
}

impl WorkerRecordJson {
    fn from_worker_record(record: WorkerRecord) -> Self {
        let backend_type = record.backend_metadata.backend_type_str().to_string();
        let (nixl_metadata, transfer_engine_session_id) = match record.backend_metadata {
            super::BackendMetadataRecord::Nixl(data) => (data, None),
            super::BackendMetadataRecord::TransferEngine(sid) => (Vec::new(), Some(sid)),
            super::BackendMetadataRecord::None => (Vec::new(), None),
        };
        Self {
            worker_rank: record.worker_rank,
            backend_type: Some(backend_type),
            nixl_metadata,
            transfer_engine_session_id,
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
            backend_metadata: super::BackendMetadataRecord::from_flat(
                json.nixl_metadata,
                json.transfer_engine_session_id,
                json.backend_type.as_deref(),
            ),
            tensors: json.tensors.into_iter().map(TensorRecord::from).collect(),
            status: json.status,
            updated_at: json.updated_at,
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

        // Redact credentials from URL before logging
        let safe_url = if self.redis_url.contains('@') {
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
        identity: &SourceIdentity,
        worker_id: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let source_id = crate::source_identity::compute_mx_source_id(identity);
        let mut conn = self.get_conn().await?;
        let worker_key = format!("{}{}:{}", keys::SOURCE_PREFIX, source_id, worker_id);
        let source_key = format!("{}{}", keys::SOURCE_PREFIX, source_id);

        // Each worker is stored as an independent hash field — no merge needed.
        let worker_records: Vec<WorkerRecord> =
            workers.into_iter().map(WorkerRecord::from).collect();

        let attr_json = serde_json::to_string(&SourceAttributesJson::from(identity))?;

        let mut pipe = redis::pipe();
        for worker in &worker_records {
            let json = WorkerRecordJson::from_worker_record(worker.clone());
            let field = worker.worker_rank.to_string();
            let value = serde_json::to_string(&json)?;
            pipe.hset(worker_key.clone(), field, value);
        }
        // Store identity attributes once per source.
        // Register each instance with its global rank as the value for fast rank lookup.
        let rank = worker_records
            .first()
            .map(|w| w.worker_rank.to_string())
            .unwrap_or_default();
        pipe.hset(&source_key, keys::ATTRIBUTES_FIELD, &attr_json);
        pipe.hset(&source_key, worker_id, &rank);

        pipe.exec_async(&mut conn).await?;

        let total_tensors: usize = worker_records.iter().map(|w| w.tensors.len()).sum();
        info!(
            "Published metadata for '{}' (source_id={source_id}, worker_id={}): {} workers ({} tensors)",
            identity.model_name,
            worker_id,
            worker_records.len(),
            total_tensors
        );
        Ok(())
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}:{}", keys::SOURCE_PREFIX, source_id, worker_id);

        let fields: std::collections::HashMap<String, String> = conn.hgetall(&key).await?;
        if fields.is_empty() {
            debug!(
                "No metadata found for source_id={} worker_id={}",
                source_id, worker_id
            );
            return Ok(None);
        }

        // Fetch model_name from the source index key's __attributes__ field.
        let source_key = format!("{}{}", keys::SOURCE_PREFIX, source_id);
        let attr_json: Option<String> = conn.hget(&source_key, keys::ATTRIBUTES_FIELD).await?;
        let model_name = attr_json
            .and_then(|v| serde_json::from_str::<SourceAttributesJson>(&v).ok())
            .map(|a| a.model_name)
            .unwrap_or_default();

        let mut workers: Vec<WorkerRecord> = Vec::with_capacity(fields.len());
        for value in fields.values() {
            let json: WorkerRecordJson = serde_json::from_str(value)?;
            workers.push(WorkerRecord::from(json));
        }
        workers.sort_by_key(|w| w.worker_rank);

        debug!(
            "Retrieved metadata for source_id={} worker_id={}: {} workers",
            source_id,
            worker_id,
            workers.len()
        );

        Ok(Some(ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name,
            workers,
            published_at: 0,
        }))
    }

    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<super::SourceInstanceInfo>> {
        let mut conn = self.get_conn().await?;

        // Collect source_ids to query
        let source_ids: Vec<String> = if let Some(sid) = source_id {
            vec![sid]
        } else {
            scan_keys(&mut conn, keys::SOURCE_SCAN_PATTERN)
                .await?
                .into_iter()
                .map(|k| k[keys::SOURCE_PREFIX.len()..].to_string())
                .collect()
        };

        let mut result = Vec::new();

        for sid in &source_ids {
            let source_key = format!("{}{}", keys::SOURCE_PREFIX, sid);
            let instance_map: std::collections::HashMap<String, String> =
                conn.hgetall(&source_key).await?;

            let model_name = instance_map
                .get(keys::ATTRIBUTES_FIELD)
                .and_then(|v| serde_json::from_str::<SourceAttributesJson>(v).ok())
                .map(|a| a.model_name)
                .unwrap_or_default();

            for (iid, rank_str) in instance_map
                .iter()
                .filter(|(k, _)| k.as_str() != keys::ATTRIBUTES_FIELD)
            {
                let worker_rank: u32 = rank_str.parse().unwrap_or(0);
                let worker_key = format!("{}{}:{}", keys::SOURCE_PREFIX, sid, iid);
                let fields: std::collections::HashMap<String, String> =
                    conn.hgetall(&worker_key).await?;
                if fields.is_empty() {
                    continue;
                }

                if let Some(required_status) = status_filter {
                    let matches = fields.values().any(|v| {
                        serde_json::from_str::<WorkerRecordJson>(v)
                            .map(|j| j.status == required_status as i32)
                            .unwrap_or(false)
                    });
                    if !matches {
                        continue;
                    }
                }

                let (status, updated_at) = fields
                    .values()
                    .find_map(|v| {
                        serde_json::from_str::<WorkerRecordJson>(v)
                            .ok()
                            .map(|j| (j.status, j.updated_at))
                    })
                    .unwrap_or((0, 0));

                result.push(super::SourceInstanceInfo {
                    source_id: sid.clone(),
                    worker_id: iid.to_string(),
                    model_name: model_name.clone(),
                    worker_rank,
                    status,
                    updated_at,
                });
            }
        }

        Ok(result)
    }

    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        let mut conn = self.get_conn().await?;
        let source_key = format!("{}{}", keys::SOURCE_PREFIX, source_id);

        let instance_map: std::collections::HashMap<String, String> =
            conn.hgetall(&source_key).await?;

        let mut pipe = redis::pipe();
        for iid in instance_map
            .keys()
            .filter(|k| k.as_str() != keys::ATTRIBUTES_FIELD)
        {
            let worker_key = format!("{}{}:{}", keys::SOURCE_PREFIX, source_id, iid);
            pipe.del(worker_key);
        }
        pipe.del(&source_key);

        pipe.exec_async(&mut conn).await?;
        info!("Removed metadata for source_id={}", source_id);
        Ok(())
    }

    async fn remove_worker(&self, source_id: &str, worker_id: &str) -> MetadataResult<()> {
        let mut conn = self.get_conn().await?;
        let source_key = format!("{}{}", keys::SOURCE_PREFIX, source_id);
        let worker_key = format!("{}{}:{}", keys::SOURCE_PREFIX, source_id, worker_id);

        let mut pipe = redis::pipe();
        pipe.del(&worker_key);
        pipe.hdel(&source_key, worker_id);
        pipe.exec_async(&mut conn).await?;

        // Clean up source index if no workers remain (only __attributes__ left)
        let remaining: usize = conn.hlen(&source_key).await?;
        if remaining <= 1 {
            conn.del::<_, ()>(&source_key).await?;
            info!(
                "Removed last worker '{}' and source index for source_id={}",
                worker_id, source_id
            );
        } else {
            info!(
                "Removed worker '{}' from source_id={}",
                worker_id, source_id
            );
        }

        Ok(())
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        let mut conn = self.get_conn().await?;
        let source_keys = scan_keys(&mut conn, keys::SOURCE_SCAN_PATTERN).await?;

        let mut sources = Vec::new();
        for key in source_keys {
            let source_id = key[keys::SOURCE_PREFIX.len()..].to_string();
            let attr_json: Option<String> = conn.hget(&key, keys::ATTRIBUTES_FIELD).await?;
            if let Some(json) = attr_json {
                let model_name = serde_json::from_str::<SourceAttributesJson>(&json)
                    .map(|a| a.model_name)
                    .unwrap_or_default();
                sources.push((source_id, model_name));
            }
        }
        Ok(sources)
    }

    async fn update_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}:{}", keys::SOURCE_PREFIX, source_id, worker_id);
        let field = worker_rank.to_string();

        let value: Option<String> = conn.hget(&key, &field).await?;
        let json_str = value.ok_or_else(|| {
            format!(
                "update_status: rank {} not found in source '{}' worker '{}'",
                worker_rank, source_id, worker_id
            )
        })?;

        let mut record: WorkerRecordJson = serde_json::from_str(&json_str)?;
        record.status = status as i32;
        record.updated_at = updated_at;

        let updated = serde_json::to_string(&record)?;
        conn.hset::<_, _, _, ()>(&key, &field, &updated).await?;

        debug!(
            "Updated status for source '{}' worker '{}' rank {} -> {}",
            source_id, worker_id, worker_rank, status as i32
        );
        Ok(())
    }
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
            backend_metadata: super::super::BackendMetadataRecord::Nixl(vec![
                0xde, 0xad, 0xbe, 0xef,
            ]),
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

        let json_record = WorkerRecordJson::from_worker_record(record.clone());
        let json = serde_json::to_string(&json_record).expect("serialize");
        let parsed: WorkerRecordJson = serde_json::from_str(&json).expect("deserialize");
        let back = WorkerRecord::from(parsed);

        assert_eq!(back.worker_rank, record.worker_rank);
        assert_eq!(back.backend_metadata, record.backend_metadata);
        assert_eq!(back.status, record.status);
        assert_eq!(back.updated_at, record.updated_at);
        assert_eq!(back.tensors.len(), 1);
    }

    #[test]
    fn test_worker_record_json_backward_compat_missing_status() {
        // Records written before status/updated_at fields existed must default to 0.
        // model_name field (removed) is silently ignored by serde.
        let json = r#"{"worker_rank":0,"model_name":"m","nixl_metadata":[],"tensors":[]}"#;
        let parsed: WorkerRecordJson = serde_json::from_str(json).expect("parse legacy");
        assert_eq!(parsed.status, 0);
        assert_eq!(parsed.updated_at, 0);
    }

    // ── SourceAttributesJson ────────────────────────────────────────────────

    fn test_identity() -> modelexpress_common::grpc::p2p::SourceIdentity {
        modelexpress_common::grpc::p2p::SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: 0,
            model_name: "deepseek-ai/DeepSeek-V3".to_string(),
            backend_framework: 1,
            tensor_parallel_size: 8,
            pipeline_parallel_size: 2,
            expert_parallel_size: 4,
            dtype: "bfloat16".to_string(),
            quantization: "fp8".to_string(),
            extra_parameters: Default::default(),
        }
    }

    #[test]
    fn test_source_attributes_from_identity() {
        let id = test_identity();
        let attr = SourceAttributesJson::from(&id);

        assert_eq!(attr.model_name, "deepseek-ai/DeepSeek-V3");
        assert_eq!(attr.mx_version, "0.3.0");
        assert_eq!(attr.tensor_parallel_size, 8);
        assert_eq!(attr.pipeline_parallel_size, 2);
        assert_eq!(attr.expert_parallel_size, 4);
        assert_eq!(attr.dtype, "bfloat16");
        assert_eq!(attr.quantization, "fp8");
        assert_eq!(attr.backend_framework, 1);
    }

    #[test]
    fn test_source_attributes_json_roundtrip() {
        let id = test_identity();
        let attr = SourceAttributesJson::from(&id);
        let json = serde_json::to_string(&attr).expect("serialize");
        let back: SourceAttributesJson = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.model_name, attr.model_name);
        assert_eq!(back.tensor_parallel_size, attr.tensor_parallel_size);
        assert_eq!(back.pipeline_parallel_size, attr.pipeline_parallel_size);
        assert_eq!(back.expert_parallel_size, attr.expert_parallel_size);
        assert_eq!(back.dtype, attr.dtype);
        assert_eq!(back.quantization, attr.quantization);
    }

    #[test]
    fn test_source_attributes_defaults_for_missing_fields() {
        // Old records that only stored model_name should deserialize with zero defaults.
        let json = r#"{"model_name":"my-model"}"#;
        let attr: SourceAttributesJson = serde_json::from_str(json).expect("deserialize");

        assert_eq!(attr.model_name, "my-model");
        assert_eq!(attr.tensor_parallel_size, 0);
        assert_eq!(attr.pipeline_parallel_size, 0);
        assert_eq!(attr.expert_parallel_size, 0);
        assert_eq!(attr.dtype, "");
        assert_eq!(attr.quantization, "");
    }
}
