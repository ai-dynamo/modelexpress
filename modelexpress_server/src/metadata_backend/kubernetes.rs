// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD backend for P2P model metadata storage.
//!
//! Uses ModelMetadata CRD for P2P transfer coordination state.
//! NIXL agent blobs are exchanged peer-to-peer via NIXL's native listen thread, never stored in K8s.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, WorkerRecord};
use crate::k8s_types::{ModelMetadata, ModelMetadataSpec, WorkerStatus};
use async_trait::async_trait;
use kube::{
    Client,
    api::{Api, ListParams, Patch, PatchParams, PostParams},
};
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};
use serde_json::json;
use std::collections::BTreeMap;
use tracing::{debug, info};

/// Kubernetes backend for metadata storage
pub struct KubernetesBackend {
    client: Client,
    namespace: String,
}

impl KubernetesBackend {
    /// Create a new Kubernetes backend
    pub async fn new(namespace: &str) -> MetadataResult<Self> {
        let client = Client::try_default().await?;
        Ok(Self {
            client,
            namespace: namespace.to_string(),
        })
    }

    /// Get the API handle for ModelMetadata CRD
    fn model_metadata_api(&self) -> Api<ModelMetadata> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }
}

#[async_trait]
impl MetadataBackend for KubernetesBackend {
    async fn connect(&self) -> MetadataResult<()> {
        // Test connection by listing CRDs (will fail if no permissions)
        let api = self.model_metadata_api();
        let _ = api.list(&ListParams::default().limit(1)).await?;
        info!(
            "Connected to Kubernetes, using namespace '{}'",
            self.namespace
        );
        Ok(())
    }

    async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let source_id = crate::source_identity::compute_mx_source_id(identity);
        let source_id = source_id.as_str();
        let model_name = &identity.model_name;
        let api = self.model_metadata_api();
        let cr_name = format!("mx-source-{}-{}", source_id, worker_id);
        let now = chrono::Utc::now().to_rfc3339();

        // Convert workers to internal format
        let worker_records: Vec<WorkerRecord> =
            workers.into_iter().map(WorkerRecord::from).collect();

        // First, ensure the CR exists
        let existing = api.get_opt(&cr_name).await?;

        if existing.is_none() {
            let new_cr = ModelMetadata {
                metadata: kube::api::ObjectMeta {
                    name: Some(cr_name.clone()),
                    namespace: Some(self.namespace.clone()),
                    labels: Some({
                        let mut labels = BTreeMap::new();
                        labels.insert(
                            "modelexpress.nvidia.com/mx-source-id".to_string(),
                            source_id.to_string(),
                        );
                        labels.insert(
                            "modelexpress.nvidia.com/mx-worker-id".to_string(),
                            worker_id.to_string(),
                        );
                        labels
                    }),
                    ..Default::default()
                },
                spec: ModelMetadataSpec {
                    model_name: model_name.to_string(),
                },
                status: None,
            };

            match api.create(&PostParams::default(), &new_cr).await {
                Ok(_) => {
                    info!("Created ModelMetadata CR '{}'", cr_name);
                }
                Err(kube::Error::Api(err)) if err.code == 409 => {
                    debug!(
                        "ModelMetadata CR '{}' already exists, proceeding to update",
                        cr_name
                    );
                }
                Err(e) => return Err(e.into()),
            }
        }

        // Build worker status updates
        let mut worker_statuses = Vec::new();
        for worker in &worker_records {
            let backend_type = worker.backend_metadata.backend_type_str().to_string();
            let metadata_endpoint = super::none_if_empty(worker.metadata_endpoint.clone());
            let agent_name = super::none_if_empty(worker.agent_name.clone());
            let worker_grpc_endpoint = super::none_if_empty(worker.worker_grpc_endpoint.clone());
            let transfer_engine_session_id = match &worker.backend_metadata {
                super::BackendMetadataRecord::TransferEngine(sid) => Some(sid.clone()),
                _ => None,
            };

            worker_statuses.push(WorkerStatus {
                worker_rank: worker.worker_rank as i32,
                backend_type: Some(backend_type),
                metadata_endpoint,
                agent_name,
                transfer_engine_session_id,
                worker_grpc_endpoint,
                status: WorkerStatus::status_name_from_proto(worker.status),
                updated_at: Some(now.clone()),
            });
        }

        // Merge with existing workers in status (retry on conflict)
        let max_retries: u32 = 5;
        let mut status_updated = false;
        for attempt in 0..max_retries {
            let current = api.get(&cr_name).await?;
            let mut all_workers: Vec<WorkerStatus> =
                current.status.map(|s| s.workers).unwrap_or_default();

            for new_worker in &worker_statuses {
                if let Some(existing) = all_workers
                    .iter_mut()
                    .find(|w| w.worker_rank == new_worker.worker_rank)
                {
                    *existing = new_worker.clone();
                } else {
                    all_workers.push(new_worker.clone());
                }
            }

            all_workers.sort_by_key(|w| w.worker_rank);

            let resource_version = current.metadata.resource_version.unwrap_or_default();
            let status_patch = json!({
                "metadata": { "resourceVersion": resource_version },
                "status": {
                    "workers": all_workers,
                    "publishedAt": now
                }
            });

            match api
                .patch_status(
                    &cr_name,
                    &PatchParams::default(),
                    &Patch::Merge(&status_patch),
                )
                .await
            {
                Ok(_) => {
                    status_updated = true;
                    break;
                }
                Err(kube::Error::Api(err)) if err.code == 409 => {
                    debug!(
                        "Conflict updating status for source '{}' instance '{}', retrying ({}/{})",
                        source_id,
                        worker_id,
                        attempt.saturating_add(1),
                        max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(
                        100_u64.saturating_mul(u64::from(attempt).saturating_add(1)),
                    ))
                    .await;
                }
                Err(e) => return Err(e.into()),
            }
        }

        if !status_updated {
            return Err(format!(
                "Failed to update status for source '{}' instance '{}' after {} retries",
                source_id, worker_id, max_retries
            )
            .into());
        }

        info!(
            "Published metadata for '{}' (source_id={}, worker_id={}): {} workers",
            model_name,
            source_id,
            worker_id,
            worker_records.len(),
        );

        Ok(())
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        let api = self.model_metadata_api();
        let cr_name = format!("mx-source-{}-{}", source_id, worker_id);

        let cr = match api.get_opt(&cr_name).await? {
            Some(cr) => cr,
            None => {
                debug!(
                    "No ModelMetadata CR found for source_id={} worker_id={}",
                    source_id, worker_id
                );
                return Ok(None);
            }
        };

        let status = match cr.status {
            Some(s) => s,
            None => {
                debug!("ModelMetadata CR '{}' has no status", cr_name);
                return Ok(None);
            }
        };

        let mut workers = Vec::new();
        for worker_status in status.workers {
            let backend_metadata = super::BackendMetadataRecord::from_flat(
                worker_status.transfer_engine_session_id.clone(),
                worker_status.backend_type.as_deref(),
            );

            let status = WorkerStatus::status_proto_from_name(&worker_status.status);
            let updated_at = worker_status
                .updated_at
                .as_deref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.timestamp_millis())
                .unwrap_or(0);

            workers.push(WorkerRecord {
                worker_rank: worker_status.worker_rank as u32,
                backend_metadata,
                metadata_endpoint: worker_status.metadata_endpoint.unwrap_or_default(),
                agent_name: worker_status.agent_name.unwrap_or_default(),
                worker_grpc_endpoint: worker_status.worker_grpc_endpoint.unwrap_or_default(),
                status,
                updated_at,
            });
        }

        let published_at = status
            .published_at
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.timestamp())
            .unwrap_or(0);

        debug!(
            "Retrieved metadata for source_id={} worker_id={}: {} workers",
            source_id,
            worker_id,
            workers.len()
        );

        Ok(Some(ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name: cr.spec.model_name.clone(),
            workers,
            published_at,
        }))
    }

    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<super::SourceInstanceInfo>> {
        let api = self.model_metadata_api();

        let label_selector = match source_id {
            Some(sid) => format!("modelexpress.nvidia.com/mx-source-id={}", sid),
            None => String::new(),
        };

        let list_params = if label_selector.is_empty() {
            ListParams::default()
        } else {
            ListParams::default().labels(&label_selector)
        };

        let crs = api.list(&list_params).await?;
        let mut result = Vec::new();
        for cr in crs.items {
            let sid = cr
                .metadata
                .labels
                .as_ref()
                .and_then(|l| l.get("modelexpress.nvidia.com/mx-source-id"))
                .cloned()
                .unwrap_or_default();
            let iid = cr
                .metadata
                .labels
                .as_ref()
                .and_then(|l| l.get("modelexpress.nvidia.com/mx-worker-id"))
                .cloned()
                .unwrap_or_default();

            let worker_rank = cr
                .status
                .as_ref()
                .and_then(|s| s.workers.first())
                .map(|w| w.worker_rank as u32)
                .unwrap_or(0);

            if let Some(required_status) = status_filter {
                let required_name = WorkerStatus::status_name_from_proto(required_status as i32);
                let matches = cr
                    .status
                    .as_ref()
                    .map(|s| s.workers.iter().any(|w| w.status == required_name))
                    .unwrap_or(false);
                if !matches {
                    continue;
                }
            }

            result.push(super::SourceInstanceInfo {
                source_id: sid,
                worker_id: iid,
                model_name: cr.spec.model_name,
                worker_rank,
            });
        }

        Ok(result)
    }

    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        let api = self.model_metadata_api();

        // Delete all CRs for this source_id via label selector
        let crs = api
            .list(&ListParams::default().labels(&format!(
                "modelexpress.nvidia.com/mx-source-id={}",
                source_id
            )))
            .await?;

        for cr in crs.items {
            if let Some(name) = cr.metadata.name {
                match api.delete(&name, &kube::api::DeleteParams::default()).await {
                    Ok(_) => info!("Deleted ModelMetadata CR '{}'", name),
                    Err(kube::Error::Api(err)) if err.code == 404 => {
                        debug!("ModelMetadata CR '{}' not found", name);
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }

        Ok(())
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        let api = self.model_metadata_api();
        let crs = api.list(&ListParams::default()).await?;

        // De-duplicate by source_id (multiple instances share the same source_id)
        let mut seen = BTreeMap::new();
        for cr in crs.items {
            let source_id = cr
                .metadata
                .labels
                .as_ref()
                .and_then(|l| l.get("modelexpress.nvidia.com/mx-source-id"))
                .cloned();
            if let Some(sid) = source_id {
                seen.entry(sid).or_insert_with(|| cr.spec.model_name);
            }
        }

        Ok(seen.into_iter().collect())
    }

    async fn update_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()> {
        let api = self.model_metadata_api();
        let cr_name = format!("mx-source-{}-{}", source_id, worker_id);
        let status_name = WorkerStatus::status_name_from_proto(status as i32);
        let updated_at_rfc3339 = chrono::DateTime::from_timestamp_millis(updated_at)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

        let max_retries: u32 = 5;
        for attempt in 0..max_retries {
            let current = api.get(&cr_name).await?;
            let mut workers: Vec<WorkerStatus> =
                current.status.map(|s| s.workers).unwrap_or_default();

            if workers.is_empty() {
                return Err(format!(
                    "update_status: no workers in source '{}' worker '{}'",
                    source_id, worker_id
                )
                .into());
            }

            for w in &mut workers {
                w.status = status_name.clone();
                w.updated_at = Some(updated_at_rfc3339.clone());
            }

            let resource_version = current.metadata.resource_version.unwrap_or_default();
            let status_patch = serde_json::json!({
                "metadata": { "resourceVersion": resource_version },
                "status": { "workers": workers }
            });

            match api
                .patch_status(
                    &cr_name,
                    &PatchParams::default(),
                    &Patch::Merge(&status_patch),
                )
                .await
            {
                Ok(_) => {
                    debug!(
                        "Updated status for source '{}' worker '{}' rank {} -> {}",
                        source_id, worker_id, worker_rank, status_name
                    );
                    return Ok(());
                }
                Err(kube::Error::Api(err)) if err.code == 409 => {
                    debug!(
                        "Conflict updating status for source '{}' worker '{}', retrying ({}/{})",
                        source_id,
                        worker_id,
                        attempt.saturating_add(1),
                        max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(
                        100_u64.saturating_mul(u64::from(attempt).saturating_add(1)),
                    ))
                    .await;
                }
                Err(e) => return Err(e.into()),
            }
        }

        Err(format!(
            "Failed to update status for source '{}' worker '{}' rank {} after {} retries",
            source_id, worker_id, worker_rank, max_retries
        )
        .into())
    }
}
