// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory P2P metadata backend for tests and local dev. Not persistent, single
//! process. Mirrors the Redis backend's key layout (`(source_id, worker_id)` ->
//! `rank -> record`, plus a per-worker reported rank), but is not behaviorally
//! identical: `remove_worker` drops a source once its last worker is gone, where Redis
//! leaves the orphaned source index behind. So `list_sources` here won't report empty
//! sources.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};

use crate::p2p::backend::{
    MetadataBackend, MetadataResult, ModelMetadataRecord, SourceInstanceInfo, WorkerRecord,
};

#[derive(Default)]
struct WorkerEntry {
    ranks: BTreeMap<u32, WorkerRecord>,
    // the rank list_workers reports for this worker
    index_rank: u32,
}

#[derive(Default)]
struct SourceEntry {
    model_name: String,
    workers: HashMap<String, WorkerEntry>,
}

#[derive(Default)]
pub struct InMemoryMetadataBackend {
    sources: Mutex<HashMap<String, SourceEntry>>,
}

impl InMemoryMetadataBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, SourceEntry>> {
        // poisoned just means someone panicked mid-write; the map is still usable
        self.sources.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl MetadataBackend for InMemoryMetadataBackend {
    async fn connect(&self) -> MetadataResult<()> {
        Ok(())
    }

    async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        worker: WorkerMetadata,
    ) -> MetadataResult<()> {
        let source_id = crate::p2p::source_identity::compute_mx_source_id(identity);
        let record = WorkerRecord::from(worker);
        let rank = record.worker_rank;

        let mut sources = self.lock();
        let source = sources.entry(source_id).or_default();
        source.model_name = identity.model_name.clone();
        let entry = source.workers.entry(worker_id.to_string()).or_default();
        entry.ranks.insert(rank, record);
        entry.index_rank = rank;
        Ok(())
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        let sources = self.lock();
        let Some(source) = sources.get(source_id) else {
            return Ok(None);
        };
        let Some(entry) = source.workers.get(worker_id) else {
            return Ok(None);
        };
        if entry.ranks.is_empty() {
            return Ok(None);
        }
        Ok(Some(ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name: source.model_name.clone(),
            workers: entry.ranks.values().cloned().collect(),
            published_at: 0,
        }))
    }

    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>> {
        let sources = self.lock();
        let mut result = Vec::new();
        for (sid, source) in sources.iter() {
            if let Some(filter) = &source_id
                && sid != filter
            {
                continue;
            }
            for (worker_id, entry) in &source.workers {
                if let Some(status) = status_filter
                    && !entry.ranks.values().any(|r| r.status == status as i32)
                {
                    continue;
                }
                let reported = entry
                    .ranks
                    .get(&entry.index_rank)
                    .or_else(|| entry.ranks.values().next());
                let (status, updated_at, accelerator) = reported.map_or_else(
                    || (0, 0, String::new()),
                    |r| (r.status, r.updated_at, r.accelerator.clone()),
                );
                result.push(SourceInstanceInfo {
                    source_id: sid.clone(),
                    worker_id: worker_id.clone(),
                    model_name: source.model_name.clone(),
                    worker_rank: entry.index_rank,
                    status,
                    updated_at,
                    accelerator,
                });
            }
        }
        Ok(result)
    }

    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        self.lock().remove(source_id);
        Ok(())
    }

    async fn remove_worker(&self, source_id: &str, worker_id: &str) -> MetadataResult<()> {
        // drop the source once its last worker is gone, so empties don't pile up
        let mut sources = self.lock();
        if let Some(source) = sources.get_mut(source_id) {
            source.workers.remove(worker_id);
            if source.workers.is_empty() {
                sources.remove(source_id);
            }
        }
        Ok(())
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        Ok(self
            .lock()
            .iter()
            .map(|(sid, source)| (sid.clone(), source.model_name.clone()))
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
        let mut sources = self.lock();
        let record = sources
            .get_mut(source_id)
            .and_then(|source| source.workers.get_mut(worker_id))
            .and_then(|entry| entry.ranks.get_mut(&worker_rank));
        match record {
            Some(record) => {
                record.status = status as i32;
                record.updated_at = updated_at;
                Ok(())
            }
            None => Err(format!(
                "update_status: rank {worker_rank} not found in source '{source_id}' worker '{worker_id}'"
            )
            .into()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::p2p::source_identity::compute_mx_source_id;
    use modelexpress_common::grpc::p2p::MxSourceType;

    fn identity(model: &str) -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: MxSourceType::Weights as i32,
            model_name: model.to_string(),
            backend_framework: 1,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
            revision: String::new(),
            ..Default::default()
        }
    }

    fn worker(rank: u32, status: SourceStatus) -> WorkerMetadata {
        WorkerMetadata {
            worker_rank: rank,
            backend_metadata: None,
            status: status as i32,
            updated_at: 0,
            ..Default::default()
        }
    }

    // store/fetch works, and the source goes away with its last worker
    #[tokio::test]
    async fn publish_get_remove_roundtrip() {
        let backend = InMemoryMetadataBackend::new();
        let id = identity("m");
        let source_id = compute_mx_source_id(&id);

        backend
            .publish_metadata(&id, "w1", worker(0, SourceStatus::Ready))
            .await
            .expect("publish");
        let record = backend
            .get_metadata(&source_id, "w1")
            .await
            .expect("get")
            .expect("present");
        assert_eq!(record.model_name, "m");
        assert_eq!(record.workers.len(), 1);

        backend
            .remove_worker(&source_id, "w1")
            .await
            .expect("remove");
        assert!(
            backend
                .get_metadata(&source_id, "w1")
                .await
                .expect("get")
                .is_none()
        );
        assert!(backend.list_sources().await.expect("list").is_empty());
    }

    // multiple ranks under one worker_id come back sorted by rank
    #[tokio::test]
    async fn get_metadata_returns_ranks_sorted() {
        let backend = InMemoryMetadataBackend::new();
        let id = identity("m");
        let source_id = compute_mx_source_id(&id);

        for rank in [2, 0, 1] {
            backend
                .publish_metadata(&id, "w1", worker(rank, SourceStatus::Ready))
                .await
                .expect("publish");
        }

        let record = backend
            .get_metadata(&source_id, "w1")
            .await
            .expect("get")
            .expect("present");
        let ranks: Vec<u32> = record.workers.iter().map(|w| w.worker_rank).collect();
        assert_eq!(ranks, [0, 1, 2], "ranks sorted ascending");
    }

    // a worker is listed when ANY of its ranks matches the status filter
    #[tokio::test]
    async fn list_workers_status_filter_matches_any_rank() {
        let backend = InMemoryMetadataBackend::new();
        let id = identity("m");

        backend
            .publish_metadata(&id, "w1", worker(0, SourceStatus::Initializing))
            .await
            .expect("publish r0");
        backend
            .publish_metadata(&id, "w1", worker(1, SourceStatus::Ready))
            .await
            .expect("publish r1");

        assert_eq!(
            backend
                .list_workers(None, Some(SourceStatus::Ready))
                .await
                .expect("ready")
                .len(),
            1,
            "Ready matches rank 1"
        );
        assert_eq!(
            backend
                .list_workers(None, Some(SourceStatus::Initializing))
                .await
                .expect("init")
                .len(),
            1,
            "Initializing matches rank 0"
        );
        assert!(
            backend
                .list_workers(None, Some(SourceStatus::Stale))
                .await
                .expect("stale")
                .is_empty(),
            "no rank is Stale"
        );
    }

    // list_workers reports the last-published (index) rank and its status
    #[tokio::test]
    async fn list_workers_reports_index_rank() {
        let backend = InMemoryMetadataBackend::new();
        let id = identity("m");
        let source_id = compute_mx_source_id(&id);

        backend
            .publish_metadata(&id, "w1", worker(0, SourceStatus::Initializing))
            .await
            .expect("publish r0");
        backend
            .publish_metadata(&id, "w1", worker(3, SourceStatus::Ready))
            .await
            .expect("publish r3");

        let listed = backend
            .list_workers(Some(source_id), None)
            .await
            .expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].worker_rank, 3, "reports the last-published rank");
        assert_eq!(listed[0].status, SourceStatus::Ready as i32);
    }

    // update_status patches an existing rank and errors on a missing rank or worker
    #[tokio::test]
    async fn update_status_patches_then_errors_on_missing() {
        let backend = InMemoryMetadataBackend::new();
        let id = identity("m");
        let source_id = compute_mx_source_id(&id);

        backend
            .publish_metadata(&id, "w1", worker(0, SourceStatus::Initializing))
            .await
            .expect("publish");
        backend
            .update_status(&source_id, "w1", 0, SourceStatus::Ready, 123)
            .await
            .expect("patch existing rank");

        let record = backend
            .get_metadata(&source_id, "w1")
            .await
            .expect("get")
            .expect("present");
        assert_eq!(record.workers[0].status, SourceStatus::Ready as i32);
        assert_eq!(record.workers[0].updated_at, 123);

        assert!(
            backend
                .update_status(&source_id, "w1", 99, SourceStatus::Ready, 1)
                .await
                .is_err(),
            "unknown rank errors"
        );
        assert!(
            backend
                .update_status(&source_id, "ghost", 0, SourceStatus::Ready, 1)
                .await
                .is_err(),
            "unknown worker errors"
        );
    }

    // a source survives while it has any worker, and is dropped with its last one
    #[tokio::test]
    async fn remove_worker_drops_source_only_when_empty() {
        let backend = InMemoryMetadataBackend::new();
        let id = identity("m");
        let source_id = compute_mx_source_id(&id);

        backend
            .publish_metadata(&id, "w1", worker(0, SourceStatus::Ready))
            .await
            .expect("publish w1");
        backend
            .publish_metadata(&id, "w2", worker(0, SourceStatus::Ready))
            .await
            .expect("publish w2");

        backend
            .remove_worker(&source_id, "w1")
            .await
            .expect("remove w1");
        assert_eq!(
            backend.list_sources().await.expect("list").len(),
            1,
            "source stays while w2 remains"
        );
        assert!(
            backend
                .get_metadata(&source_id, "w2")
                .await
                .expect("get")
                .is_some()
        );

        backend
            .remove_worker(&source_id, "w2")
            .await
            .expect("remove w2");
        assert!(
            backend.list_sources().await.expect("list").is_empty(),
            "source dropped with its last worker"
        );
    }
}
