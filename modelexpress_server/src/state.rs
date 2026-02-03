// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis-based state management for P2P model metadata.
//!
//! Stores model metadata (NIXL agent info + tensor descriptors) keyed by model name.
//! This allows clients to discover existing sources for the same model.

use modelexpress_common::grpc::p2p::{TensorDescriptor, WorkerMetadata};
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
    pub const READY_PREFIX: &str = "mx:nixl_ready:";
}

/// Serializable version of TensorDescriptor for Redis storage
/// NOTE: addr and size are serialized as strings to avoid Lua cjson precision issues
/// with large u64 values (GPU addresses like 139948187451390 get converted to floats)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRecord {
    pub name: String,
    #[serde(
        serialize_with = "serialize_u64_as_string",
        deserialize_with = "deserialize_u64_from_string_or_number"
    )]
    pub addr: u64,
    #[serde(
        serialize_with = "serialize_u64_as_string",
        deserialize_with = "deserialize_u64_from_string_or_number"
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

fn deserialize_u64_from_string_or_number<'de, D>(deserializer: D) -> Result<u64, D::Error>
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

impl From<TensorDescriptor> for TensorRecord {
    fn from(desc: TensorDescriptor) -> Self {
        Self {
            name: desc.name,
            addr: desc.addr,
            size: desc.size,
            device_id: desc.device_id,
            dtype: desc.dtype,
        }
    }
}

impl From<TensorRecord> for TensorDescriptor {
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

/// Serializable version of WorkerMetadata for Redis storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRecord {
    pub worker_rank: u32,
    pub nixl_metadata: Vec<u8>,
    pub tensors: Vec<TensorRecord>,
}

impl From<WorkerMetadata> for WorkerRecord {
    fn from(meta: WorkerMetadata) -> Self {
        Self {
            worker_rank: meta.worker_rank,
            nixl_metadata: meta.nixl_metadata,
            tensors: meta.tensors.into_iter().map(TensorRecord::from).collect(),
        }
    }
}

impl From<WorkerRecord> for WorkerMetadata {
    fn from(record: WorkerRecord) -> Self {
        Self {
            worker_rank: record.worker_rank,
            nixl_metadata: record.nixl_metadata,
            tensors: record
                .tensors
                .into_iter()
                .map(TensorDescriptor::from)
                .collect(),
        }
    }
}

/// Model metadata stored in Redis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataRecord {
    pub model_name: String,
    pub workers: Vec<WorkerRecord>,
    pub published_at: i64,
}

/// Ready flag stored in Redis for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyRecord {
    pub session_id: String,
    pub metadata_hash: String,
    pub nixl_ready: bool,
    pub stability_verified: bool,
    pub timestamp: f64,
}

/// State manager that handles Redis operations for P2P metadata
#[derive(Clone)]
pub struct P2pStateManager {
    redis: Arc<RwLock<Option<ConnectionManager>>>,
    redis_url: String,
}

impl P2pStateManager {
    /// Create a new state manager
    pub fn new(redis_url: &str) -> Self {
        Self {
            redis: Arc::new(RwLock::new(None)),
            redis_url: redis_url.to_string(),
        }
    }

    /// Initialize the Redis connection
    pub async fn connect(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let client = redis::Client::open(self.redis_url.as_str())?;
        let conn = ConnectionManager::new(client).await?;

        let mut guard = self.redis.write().await;
        *guard = Some(conn);

        info!("Connected to Redis at {}", self.redis_url);
        Ok(())
    }

    /// Get a Redis connection, reconnecting if necessary
    async fn get_conn(
        &self,
    ) -> Result<ConnectionManager, Box<dyn std::error::Error + Send + Sync>> {
        let guard = self.redis.read().await;
        if let Some(conn) = guard.as_ref() {
            return Ok(conn.clone());
        }
        drop(guard);

        // Need to reconnect
        self.connect().await?;

        let guard = self.redis.read().await;
        guard
            .as_ref()
            .cloned()
            .ok_or_else(|| "Redis connection not available".into())
    }

    // ========================================================================
    // Model Metadata Management
    // ========================================================================

    /// Publish metadata for a model
    /// NOTE: This MERGES workers with existing data, allowing incremental publishing
    /// from multiple workers in a distributed system.
    /// Uses a Lua script for ATOMIC read-modify-write to handle concurrent updates.
    pub async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}", keys::MODEL_PREFIX, model_name);

        // Convert new workers to records and serialize
        let new_workers: Vec<WorkerRecord> = workers.into_iter().map(WorkerRecord::from).collect();
        let new_workers_json = serde_json::to_string(&new_workers)?;
        let timestamp = chrono::Utc::now().timestamp();

        // Lua script for atomic read-modify-write merge
        // This runs atomically in Redis, preventing race conditions
        let script = redis::Script::new(
            r#"
            local key = KEYS[1]
            local models_set = ARGV[1]
            local model_name = ARGV[2]
            local new_workers_json = ARGV[3]
            local timestamp = tonumber(ARGV[4])

            -- Parse new workers
            local new_workers = cjson.decode(new_workers_json)

            -- Get existing data
            local existing_json = redis.call('GET', key)
            local existing_workers = {}

            if existing_json then
                local existing = cjson.decode(existing_json)
                if existing.workers then
                    existing_workers = existing.workers
                end
            end

            -- Merge: update existing workers or add new ones
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

            -- Sort by worker_rank
            table.sort(existing_workers, function(a, b)
                return a.worker_rank < b.worker_rank
            end)

            -- Create final record
            local record = {
                model_name = model_name,
                workers = existing_workers,
                published_at = timestamp
            }

            -- Store and return worker count
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

    /// Get metadata for a model
    pub async fn get_metadata(
        &self,
        model_name: &str,
    ) -> Result<Option<ModelMetadataRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}", keys::MODEL_PREFIX, model_name);
        let json: Option<String> = conn.get(&key).await?;

        match json {
            Some(data) => {
                let record: ModelMetadataRecord = serde_json::from_str(&data)?;
                debug!("Retrieved metadata for model '{}'", model_name);
                Ok(Some(record))
            }
            None => {
                debug!("No metadata found for model '{}'", model_name);
                Ok(None)
            }
        }
    }

    // ========================================================================
    // Ready Flag Management (coordination between source and target)
    // ========================================================================

    /// Publish ready flag for a worker
    pub async fn publish_ready(
        &self,
        model_name: &str,
        worker_id: u32,
        session_id: &str,
        metadata_hash: &str,
        nixl_ready: bool,
        stability_verified: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}:worker:{}", keys::READY_PREFIX, model_name, worker_id);

        let record = ReadyRecord {
            session_id: session_id.to_string(),
            metadata_hash: metadata_hash.to_string(),
            nixl_ready,
            stability_verified,
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        };

        let json = serde_json::to_string(&record)?;

        // Set with 2 hour TTL (matches warmup timeout)
        conn.set_ex::<_, _, ()>(&key, &json, 7200).await?;

        info!(
            "Published ready flag for model '{}' worker {}: nixl_ready={}, stability_verified={}, session={}",
            model_name,
            worker_id,
            nixl_ready,
            stability_verified,
            &session_id[..8.min(session_id.len())]
        );
        Ok(())
    }

    /// Get ready status for a worker
    pub async fn get_ready(
        &self,
        model_name: &str,
        worker_id: u32,
    ) -> Result<Option<ReadyRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}{}:worker:{}", keys::READY_PREFIX, model_name, worker_id);

        let json: Option<String> = conn.get(&key).await?;

        match json {
            Some(data) => {
                let record: ReadyRecord = serde_json::from_str(&data)?;
                debug!(
                    "Retrieved ready flag for model '{}' worker {}: nixl_ready={}, stability_verified={}",
                    model_name, worker_id, record.nixl_ready, record.stability_verified
                );
                Ok(Some(record))
            }
            None => {
                debug!(
                    "No ready flag found for model '{}' worker {}",
                    model_name, worker_id
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_record_conversion() {
        let desc = TensorDescriptor {
            name: "model.layers.0.weight".to_string(),
            addr: 0x7f0000000000,
            size: 1024 * 1024 * 1024,
            device_id: 0,
            dtype: "bfloat16".to_string(),
        };

        let record = TensorRecord::from(desc.clone());
        assert_eq!(record.name, "model.layers.0.weight");
        assert_eq!(record.size, 1024 * 1024 * 1024);

        let back: TensorDescriptor = record.into();
        assert_eq!(back.name, desc.name);
        assert_eq!(back.addr, desc.addr);
    }

    #[test]
    fn test_worker_record_conversion() {
        let meta = WorkerMetadata {
            worker_rank: 3,
            nixl_metadata: vec![1, 2, 3, 4, 5],
            tensors: vec![TensorDescriptor {
                name: "test.weight".to_string(),
                addr: 0x1000,
                size: 4096,
                device_id: 3,
                dtype: "float16".to_string(),
            }],
        };

        let record = WorkerRecord::from(meta.clone());
        assert_eq!(record.worker_rank, 3);
        assert_eq!(record.nixl_metadata, vec![1, 2, 3, 4, 5]);
        assert_eq!(record.tensors.len(), 1);

        let back: WorkerMetadata = record.into();
        assert_eq!(back.worker_rank, meta.worker_rank);
        assert_eq!(back.nixl_metadata, meta.nixl_metadata);
    }

    #[test]
    fn test_model_record_serialization() {
        let record = ModelMetadataRecord {
            model_name: "meta-llama/Llama-3.1-70B".to_string(),
            workers: vec![
                WorkerRecord {
                    worker_rank: 0,
                    nixl_metadata: vec![10, 20, 30],
                    tensors: vec![TensorRecord {
                        name: "layer.0.weight".to_string(),
                        addr: 0x7f00_0000_0000,
                        size: 1_000_000,
                        device_id: 0,
                        dtype: "bfloat16".to_string(),
                    }],
                },
                WorkerRecord {
                    worker_rank: 1,
                    nixl_metadata: vec![40, 50, 60],
                    tensors: vec![TensorRecord {
                        name: "layer.0.weight".to_string(),
                        addr: 0x7f00_0000_0000,
                        size: 1_000_000,
                        device_id: 1,
                        dtype: "bfloat16".to_string(),
                    }],
                },
            ],
            published_at: 1234567890,
        };

        // Test serialization round-trip
        let json = serde_json::to_string(&record).expect("serialization failed");
        let deserialized: ModelMetadataRecord =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(deserialized.model_name, record.model_name);
        assert_eq!(deserialized.workers.len(), 2);
    }
}
