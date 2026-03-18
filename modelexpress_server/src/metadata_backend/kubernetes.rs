// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD backend for P2P model metadata storage.
//!
//! Uses ModelMetadata CRD and ConfigMaps for tensor descriptors.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, TensorRecord, WorkerRecord};
use crate::k8s_types::{ModelMetadata, ModelMetadataSpec, TensorDescriptorJson, WorkerStatus};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use k8s_openapi::api::core::v1::ConfigMap;
use kube::{
    Client,
    api::{Api, ListParams, Patch, PatchParams, PostParams},
};
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};
use serde_json::json;
use std::collections::BTreeMap;
use tracing::{debug, info, warn};

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

    /// Get the API handle for ConfigMaps
    fn configmap_api(&self) -> Api<ConfigMap> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    /// Create or update a ConfigMap with tensor descriptors for a worker.
    /// If `owner_uid` and `owner_name` are provided, sets ownerReferences
    /// so K8s garbage-collects ConfigMaps when the parent CR is deleted.
    async fn upsert_tensor_configmap(
        &self,
        source_id: &str,
        instance_id: &str,
        worker_rank: u32,
        tensors: &[TensorRecord],
        owner_name: Option<&str>,
        owner_uid: Option<&str>,
    ) -> MetadataResult<String> {
        let cr_name = format!("mx-source-{}-{}", source_id, instance_id);
        let cm_name = format!("{}-tensors-worker-{}", cr_name, worker_rank);

        // Convert tensors to JSON
        let tensor_json: Vec<TensorDescriptorJson> = tensors
            .iter()
            .map(|t| TensorDescriptorJson {
                name: t.name.clone(),
                addr: t.addr.to_string(),
                size: t.size.to_string(),
                device_id: t.device_id,
                dtype: t.dtype.clone(),
            })
            .collect();

        let tensors_data = serde_json::to_string_pretty(&tensor_json)?;

        let mut data = BTreeMap::new();
        data.insert("tensors.json".to_string(), tensors_data);

        let mut labels = BTreeMap::new();
        labels.insert(
            "modelexpress.nvidia.com/mx-source-id".to_string(),
            source_id.to_string(),
        );
        labels.insert(
            "modelexpress.nvidia.com/worker".to_string(),
            worker_rank.to_string(),
        );

        let owner_references = match (owner_name, owner_uid) {
            (Some(name), Some(uid)) => Some(vec![
                k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference {
                    api_version: "modelexpress.nvidia.com/v1alpha1".to_string(),
                    kind: "ModelMetadata".to_string(),
                    name: name.to_string(),
                    uid: uid.to_string(),
                    controller: Some(true),
                    block_owner_deletion: Some(true),
                },
            ]),
            _ => None,
        };

        let cm = ConfigMap {
            metadata: kube::api::ObjectMeta {
                name: Some(cm_name.clone()),
                namespace: Some(self.namespace.clone()),
                labels: Some(labels),
                owner_references,
                ..Default::default()
            },
            data: Some(data),
            ..Default::default()
        };

        let api = self.configmap_api();

        // Try to create, if exists then patch
        match api.create(&PostParams::default(), &cm).await {
            Ok(_) => {
                debug!("Created ConfigMap {} for worker {}", cm_name, worker_rank);
            }
            Err(kube::Error::Api(err)) if err.code == 409 => {
                // Already exists — use merge patch to avoid SSA field manager conflicts
                api.patch(&cm_name, &PatchParams::default(), &Patch::Merge(&cm))
                    .await?;
                debug!("Updated ConfigMap {} for worker {}", cm_name, worker_rank);
            }
            Err(e) => return Err(e.into()),
        }

        Ok(cm_name)
    }

    /// Read tensor descriptors from a ConfigMap
    async fn read_tensor_configmap(&self, cm_name: &str) -> MetadataResult<Vec<TensorRecord>> {
        let api = self.configmap_api();
        let cm = api.get(cm_name).await?;

        let tensors_json = cm
            .data
            .and_then(|d| d.get("tensors.json").cloned())
            .ok_or("ConfigMap missing tensors.json")?;

        let tensor_descs: Vec<TensorDescriptorJson> = serde_json::from_str(&tensors_json)?;

        let tensors = tensor_descs
            .into_iter()
            .map(|t| {
                let addr = t.addr.parse::<u64>().map_err(|e| {
                    format!("Invalid tensor addr '{}' for '{}': {}", t.addr, t.name, e)
                })?;
                let size = t.size.parse::<u64>().map_err(|e| {
                    format!("Invalid tensor size '{}' for '{}': {}", t.size, t.name, e)
                })?;
                Ok(TensorRecord {
                    name: t.name,
                    addr,
                    size,
                    device_id: t.device_id,
                    dtype: t.dtype,
                })
            })
            .collect::<MetadataResult<Vec<_>>>()?;

        Ok(tensors)
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
        instance_id: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let source_id = crate::source_identity::compute_mx_source_id(identity);
        let source_id = source_id.as_str();
        let model_name = &identity.model_name;
        let api = self.model_metadata_api();
        let cr_name = format!("mx-source-{}-{}", source_id, instance_id);
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
                            "modelexpress.nvidia.com/mx-instance-id".to_string(),
                            instance_id.to_string(),
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

        // Get CR UID for ownerReferences on ConfigMaps
        let cr = api.get(&cr_name).await?;
        let owner_uid = cr.metadata.uid.as_deref();
        let owner_name = cr.metadata.name.as_deref();

        // Build worker status updates and create ConfigMaps
        let mut worker_statuses = Vec::new();
        for worker in &worker_records {
            let cm_name = self
                .upsert_tensor_configmap(
                    source_id,
                    instance_id,
                    worker.worker_rank,
                    &worker.tensors,
                    owner_name,
                    owner_uid,
                )
                .await?;

            let backend_type = worker.backend_metadata.backend_type_str().to_string();
            let (nixl_metadata, transfer_engine_session_id) = match &worker.backend_metadata {
                super::BackendMetadataRecord::Nixl(data) => (BASE64.encode(data), None),
                super::BackendMetadataRecord::TransferEngine(sid) => {
                    (String::new(), Some(sid.clone()))
                }
                super::BackendMetadataRecord::None => (String::new(), None),
            };

            worker_statuses.push(WorkerStatus {
                worker_rank: worker.worker_rank as i32,
                backend_type: Some(backend_type),
                nixl_metadata,
                transfer_engine_session_id,
                tensor_count: worker.tensors.len() as i32,
                tensor_config_map: Some(cm_name),
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
                        instance_id,
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
                source_id, instance_id, max_retries
            )
            .into());
        }

        let total_tensors: usize = worker_records.iter().map(|w| w.tensors.len()).sum();
        info!(
            "Published metadata for '{}' (source_id={}, instance_id={}): {} workers ({} tensors)",
            model_name,
            source_id,
            instance_id,
            worker_records.len(),
            total_tensors
        );

        Ok(())
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        instance_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        let api = self.model_metadata_api();
        let cr_name = format!("mx-source-{}-{}", source_id, instance_id);

        let cr = match api.get_opt(&cr_name).await? {
            Some(cr) => cr,
            None => {
                debug!(
                    "No ModelMetadata CR found for source_id={} instance_id={}",
                    source_id, instance_id
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
            let nixl_bytes = if !worker_status.nixl_metadata.is_empty() {
                BASE64.decode(&worker_status.nixl_metadata).map_err(|e| {
                    format!(
                        "Failed to decode NIXL metadata for worker {}: {}",
                        worker_status.worker_rank, e
                    )
                })?
            } else {
                Vec::new()
            };
            let backend_metadata = super::BackendMetadataRecord::from_flat(
                nixl_bytes,
                worker_status.transfer_engine_session_id.clone(),
                worker_status.backend_type.as_deref(),
            );

            let tensors = if let Some(cm_name) = &worker_status.tensor_config_map {
                match self.read_tensor_configmap(cm_name).await {
                    Ok(t) => t,
                    Err(e) => {
                        warn!("Failed to read tensor ConfigMap '{}': {}", cm_name, e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            let worker_status_proto = WorkerStatus::status_proto_from_name(&worker_status.status)
                .ok_or_else(|| {
                format!(
                    "Unknown status string '{}' for worker {}",
                    worker_status.status, worker_status.worker_rank
                )
            })?;
            let updated_at = worker_status
                .updated_at
                .as_deref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.timestamp_millis())
                .unwrap_or(0);

            workers.push(WorkerRecord {
                worker_rank: worker_status.worker_rank as u32,
                backend_metadata,
                tensors,
                status: worker_status_proto,
                updated_at,
            });
        }

        let published_at = status
            .published_at
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.timestamp())
            .unwrap_or(0);

        debug!(
            "Retrieved metadata for source_id={} instance_id={}: {} workers",
            source_id,
            instance_id,
            workers.len()
        );

        Ok(Some(ModelMetadataRecord {
            source_id: source_id.to_string(),
            instance_id: instance_id.to_string(),
            model_name: cr.spec.model_name.clone(),
            workers,
            published_at,
        }))
    }

    async fn list_instances(
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
                .and_then(|l| l.get("modelexpress.nvidia.com/mx-instance-id"))
                .cloned()
                .unwrap_or_default();

            let worker_rank = cr
                .status
                .as_ref()
                .and_then(|s| s.workers.first())
                .map(|w| w.worker_rank as u32)
                .unwrap_or(0);

            if let Some(required_status) = status_filter {
                let required_name =
                    crate::k8s_types::WorkerStatus::status_name_from_proto(required_status as i32);
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
                instance_id: iid,
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

        // ConfigMaps are garbage-collected via ownerReferences; also sweep by label
        let cm_api = self.configmap_api();
        let cms = cm_api
            .list(&ListParams::default().labels(&format!(
                "modelexpress.nvidia.com/mx-source-id={}",
                source_id
            )))
            .await?;

        for cm in cms {
            if let Some(name) = cm.metadata.name {
                match cm_api
                    .delete(&name, &kube::api::DeleteParams::default())
                    .await
                {
                    Ok(_) => debug!("Deleted ConfigMap '{}'", name),
                    Err(e) => warn!("Failed to delete ConfigMap '{}': {}", name, e),
                }
            }
        }

        Ok(())
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        let api = self.model_metadata_api();
        let crs = api.list(&ListParams::default()).await?;

        // De-duplicate by source_id (multiple instances share the same source_id)
        let mut seen = std::collections::BTreeMap::new();
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
        instance_id: &str,
        worker_id: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()> {
        let api = self.model_metadata_api();
        let cr_name = format!("mx-source-{}-{}", source_id, instance_id);
        let status_name = WorkerStatus::status_name_from_proto(status as i32);
        let updated_at_rfc3339 = chrono::DateTime::from_timestamp_millis(updated_at)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

        let max_retries: u32 = 5;
        for attempt in 0..max_retries {
            let current = api.get(&cr_name).await?;
            let mut all_workers: Vec<WorkerStatus> =
                current.status.map(|s| s.workers).unwrap_or_default();

            if let Some(worker) = all_workers
                .iter_mut()
                .find(|w| w.worker_rank == worker_id as i32)
            {
                worker.status = status_name.clone();
                worker.updated_at = Some(updated_at_rfc3339.clone());
            } else {
                debug!(
                    "update_status: worker {} not found in source '{}' instance '{}', skipping",
                    worker_id, source_id, instance_id
                );
                return Ok(());
            }

            let resource_version = current.metadata.resource_version.unwrap_or_default();
            let status_patch = serde_json::json!({
                "metadata": { "resourceVersion": resource_version },
                "status": { "workers": all_workers }
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
                        "Updated status for source '{}' instance '{}' worker {} -> {}",
                        source_id, instance_id, worker_id, status_name
                    );
                    return Ok(());
                }
                Err(kube::Error::Api(err)) if err.code == 409 => {
                    debug!(
                        "Conflict updating status for source '{}' instance '{}', retrying ({}/{})",
                        source_id,
                        instance_id,
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
            "Failed to update status for source '{}' instance '{}' worker {} after {} retries",
            source_id, instance_id, worker_id, max_retries
        )
        .into())
    }
}
