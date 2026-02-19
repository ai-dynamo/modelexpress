// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD backend for P2P model metadata storage.
//!
//! Uses ModelMetadata CRD and ConfigMaps for tensor descriptors.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, TensorRecord, WorkerRecord};
use crate::k8s_types::{
    ModelMetadata, ModelMetadataPhase, ModelMetadataSpec, TensorDescriptorJson, WorkerStatus,
    sanitize_model_name,
};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use k8s_openapi::api::core::v1::ConfigMap;
use kube::{
    Client,
    api::{Api, ListParams, Patch, PatchParams, PostParams},
};
use modelexpress_common::grpc::p2p::WorkerMetadata;
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
        model_name: &str,
        worker_rank: u32,
        tensors: &[TensorRecord],
        owner_name: Option<&str>,
        owner_uid: Option<&str>,
    ) -> MetadataResult<String> {
        let cr_name = sanitize_model_name(model_name);
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
            "modelexpress.nvidia.com/model".to_string(),
            cr_name.clone(), // Use sanitized name for label value
        );
        labels.insert(
            "modelexpress.nvidia.com/worker".to_string(),
            worker_rank.to_string(),
        );

        let owner_references = match (owner_name, owner_uid) {
            (Some(name), Some(uid)) => Some(vec![k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference {
                api_version: "modelexpress.nvidia.com/v1alpha1".to_string(),
                kind: "ModelMetadata".to_string(),
                name: name.to_string(),
                uid: uid.to_string(),
                controller: Some(true),
                block_owner_deletion: Some(true),
            }]),
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
                // Already exists, patch it
                api.patch(
                    &cm_name,
                    &PatchParams::apply("modelexpress"),
                    &Patch::Apply(&cm),
                )
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
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let api = self.model_metadata_api();
        let cr_name = sanitize_model_name(model_name);
        let now = chrono::Utc::now().to_rfc3339();

        // Convert workers to internal format
        let worker_records: Vec<WorkerRecord> =
            workers.into_iter().map(WorkerRecord::from).collect();

        // First, ensure the CR exists
        let existing = api.get_opt(&cr_name).await?;

        if existing.is_none() {
            // Create new CR
            let new_cr = ModelMetadata {
                metadata: kube::api::ObjectMeta {
                    name: Some(cr_name.clone()),
                    namespace: Some(self.namespace.clone()),
                    labels: Some({
                        let mut labels = BTreeMap::new();
                        labels.insert(
                            "modelexpress.nvidia.com/model".to_string(),
                            cr_name.clone(), // Use sanitized name for label value
                        );
                        labels
                    }),
                    ..Default::default()
                },
                spec: ModelMetadataSpec {
                    model_name: model_name.to_string(),
                    expected_workers: std::env::var("MX_EXPECTED_WORKERS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(worker_records.len() as i32),
                },
                status: None,
            };

            match api.create(&PostParams::default(), &new_cr).await {
                Ok(_) => {
                    info!("Created ModelMetadata CR '{}'", cr_name);
                }
                Err(kube::Error::Api(err)) if err.code == 409 => {
                    debug!("ModelMetadata CR '{}' already exists, proceeding to update", cr_name);
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
                    model_name, worker.worker_rank, &worker.tensors,
                    owner_name, owner_uid,
                )
                .await?;

            worker_statuses.push(WorkerStatus {
                worker_rank: worker.worker_rank as i32,
                nixl_metadata: BASE64.encode(&worker.nixl_metadata),
                tensor_count: worker.tensors.len() as i32,
                tensor_config_map: Some(cm_name),
                ready: true,
                stability_verified: false, // Set by source after warmup
                session_id: None,
                published_at: Some(now.clone()),
            });
        }

        // Merge with existing workers in status (retry on conflict)
        let max_retries = 5;
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

            let phase = if all_workers.is_empty() {
                ModelMetadataPhase::Pending
            } else if all_workers.iter().all(|w| w.ready && w.stability_verified) {
                ModelMetadataPhase::Ready
            } else if all_workers.iter().any(|w| w.ready) {
                ModelMetadataPhase::Initializing
            } else {
                ModelMetadataPhase::Pending
            };

            let resource_version = current.metadata.resource_version.unwrap_or_default();
            let status_patch = json!({
                "metadata": { "resourceVersion": resource_version },
                "status": {
                    "phase": phase.to_string(),
                    "workers": all_workers,
                    "publishedAt": now
                }
            });

            match api
                .patch_status(
                    &cr_name,
                    &PatchParams::apply("modelexpress"),
                    &Patch::Merge(&status_patch),
                )
                .await
            {
                Ok(_) => break,
                Err(kube::Error::Api(err)) if err.code == 409 && attempt < max_retries - 1 => {
                    debug!("Conflict updating status for '{}', retrying ({}/{})", cr_name, attempt + 1, max_retries);
                    tokio::time::sleep(std::time::Duration::from_millis(100 * (attempt as u64 + 1))).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }

        let total_tensors: usize = worker_records.iter().map(|w| w.tensors.len()).sum();
        info!(
            "Published metadata for model '{}': {} workers ({} tensors)",
            model_name,
            worker_records.len(),
            total_tensors
        );

        Ok(())
    }

    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>> {
        let api = self.model_metadata_api();
        let cr_name = sanitize_model_name(model_name);

        let cr = match api.get_opt(&cr_name).await? {
            Some(cr) => cr,
            None => {
                debug!("No ModelMetadata CR found for model '{}'", model_name);
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

        // Reconstruct workers from status + ConfigMaps
        let mut workers = Vec::new();
        for worker_status in status.workers {
            // Decode NIXL metadata
            let nixl_metadata = BASE64
                .decode(&worker_status.nixl_metadata)
                .unwrap_or_default();

            // Read tensors from ConfigMap
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

            workers.push(WorkerRecord {
                worker_rank: worker_status.worker_rank as u32,
                nixl_metadata,
                tensors,
            });
        }

        // Parse published_at timestamp
        let published_at = status
            .published_at
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.timestamp())
            .unwrap_or(0);

        debug!(
            "Retrieved metadata for model '{}': {} workers",
            model_name,
            workers.len()
        );

        Ok(Some(ModelMetadataRecord {
            model_name: cr.spec.model_name,
            workers,
            published_at,
        }))
    }

    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        let cr_name = sanitize_model_name(model_name);

        // Delete the CR (ConfigMaps are garbage-collected via ownerReferences)
        let api = self.model_metadata_api();
        match api
            .delete(&cr_name, &kube::api::DeleteParams::default())
            .await
        {
            Ok(_) => {
                info!("Deleted ModelMetadata CR '{}'", cr_name);
            }
            Err(kube::Error::Api(err)) if err.code == 404 => {
                debug!("ModelMetadata CR '{}' not found", cr_name);
            }
            Err(e) => return Err(e.into()),
        }

        // Also delete associated ConfigMaps
        let cm_api = self.configmap_api();
        let cms = cm_api
            .list(&ListParams::default().labels(&format!(
                "modelexpress.nvidia.com/model={}",
                cr_name // Use sanitized name for label selector
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

    async fn list_models(&self) -> MetadataResult<Vec<String>> {
        let api = self.model_metadata_api();
        let crs = api.list(&ListParams::default()).await?;

        let models: Vec<String> = crs.items.into_iter().map(|cr| cr.spec.model_name).collect();

        Ok(models)
    }
}
