// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory metadata backend for standalone and development deployments.

use crate::metadata_backend::{
    MetadataBackend, MetadataResult, ModelMetadataRecord, SourceInstanceInfo, WorkerRecord,
};
use crate::source_identity::compute_mx_source_id;
use async_trait::async_trait;
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Debug, Clone)]
struct SourceEntry {
    model_name: String,
    published_at: i64,
    workers: HashMap<String, WorkerRecord>,
}

#[derive(Default)]
pub struct MemoryBackend {
    entries: RwLock<HashMap<String, SourceEntry>>,
}

impl MemoryBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MetadataBackend for MemoryBackend {
    async fn connect(&self) -> MetadataResult<()> {
        info!("Memory metadata backend initialized");
        Ok(())
    }

    async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        worker: WorkerMetadata,
    ) -> MetadataResult<()> {
        let source_id = compute_mx_source_id(identity);
        let mut entries = self.entries.write().await;
        let published_at = chrono::Utc::now().timestamp_millis();
        let entry = entries.entry(source_id).or_insert_with(|| SourceEntry {
            model_name: identity.model_name.clone(),
            published_at,
            workers: HashMap::new(),
        });
        entry.model_name = identity.model_name.clone();
        entry.published_at = published_at;
        entry
            .workers
            .insert(worker_id.to_string(), WorkerRecord::from(worker));
        Ok(())
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        let entries = self.entries.read().await;
        let Some(entry) = entries.get(source_id) else {
            return Ok(None);
        };
        let Some(worker) = entry.workers.get(worker_id) else {
            return Ok(None);
        };
        Ok(Some(ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name: entry.model_name.clone(),
            workers: vec![worker.clone()],
            published_at: entry.published_at,
        }))
    }

    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>> {
        let entries = self.entries.read().await;
        let mut results = Vec::new();

        for (entry_source_id, entry) in entries.iter() {
            if source_id
                .as_ref()
                .is_some_and(|wanted| wanted != entry_source_id)
            {
                continue;
            }

            for (worker_id, worker) in &entry.workers {
                if status_filter.is_some_and(|status| worker.status != status as i32) {
                    continue;
                }

                results.push(SourceInstanceInfo {
                    source_id: entry_source_id.clone(),
                    worker_id: worker_id.clone(),
                    model_name: entry.model_name.clone(),
                    worker_rank: worker.worker_rank,
                    status: worker.status,
                    updated_at: worker.updated_at,
                });
            }
        }

        Ok(results)
    }

    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        self.entries.write().await.remove(source_id);
        Ok(())
    }

    async fn remove_worker(&self, source_id: &str, worker_id: &str) -> MetadataResult<()> {
        let mut entries = self.entries.write().await;
        let should_remove_source = if let Some(entry) = entries.get_mut(source_id) {
            entry.workers.remove(worker_id);
            entry.workers.is_empty()
        } else {
            false
        };

        if should_remove_source {
            entries.remove(source_id);
        }

        Ok(())
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        let entries = self.entries.read().await;
        Ok(entries
            .iter()
            .map(|(source_id, entry)| (source_id.clone(), entry.model_name.clone()))
            .collect())
    }

    async fn update_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()> {
        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(source_id)
            && let Some(worker) = entry.workers.get_mut(worker_id)
            && worker.worker_rank == worker_rank
        {
            worker.status = status as i32;
            worker.updated_at = updated_at;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::p2p::{
        BackendFramework, MxSourceType, TensorDescriptor, worker_metadata::BackendMetadata,
    };

    fn identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: MxSourceType::Weights as i32,
            model_name: "test/model".to_string(),
            backend_framework: BackendFramework::Vllm as i32,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
        }
    }

    fn worker(rank: u32) -> WorkerMetadata {
        WorkerMetadata {
            worker_rank: rank,
            backend_metadata: Some(BackendMetadata::NixlMetadata(vec![1, 2, 3])),
            tensors: vec![TensorDescriptor {
                name: "weight".to_string(),
                addr: 123,
                size: 456,
                device_id: rank,
                dtype: "bfloat16".to_string(),
            }],
            status: SourceStatus::Ready as i32,
            updated_at: 1000,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_publish_and_get_metadata() {
        let backend = MemoryBackend::new();
        let identity = identity();
        let source_id = compute_mx_source_id(&identity);

        backend
            .publish_metadata(&identity, "worker-a", worker(0))
            .await
            .expect("publish should succeed");

        let record = backend
            .get_metadata(&source_id, "worker-a")
            .await
            .expect("lookup should succeed")
            .expect("record should exist");

        assert_eq!(record.model_name, "test/model");
        assert_eq!(record.workers.len(), 1);
        assert_eq!(record.workers[0].worker_rank, 0);
    }

    #[tokio::test]
    async fn test_list_and_remove_workers() {
        let backend = MemoryBackend::new();
        let identity = identity();
        let source_id = compute_mx_source_id(&identity);

        backend
            .publish_metadata(&identity, "worker-a", worker(0))
            .await
            .expect("publish rank 0");
        backend
            .publish_metadata(&identity, "worker-b", worker(1))
            .await
            .expect("publish rank 1");

        let workers = backend
            .list_workers(Some(source_id.clone()), Some(SourceStatus::Ready))
            .await
            .expect("list should succeed");
        assert_eq!(workers.len(), 2);

        backend
            .remove_worker(&source_id, "worker-a")
            .await
            .expect("remove worker");
        let workers = backend
            .list_workers(Some(source_id.clone()), None)
            .await
            .expect("list after removal");
        assert_eq!(workers.len(), 1);

        backend
            .remove_metadata(&source_id)
            .await
            .expect("remove source");
        let sources = backend.list_sources().await.expect("list sources");
        assert!(sources.is_empty());
    }
}
