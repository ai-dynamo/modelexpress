// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD backend for the model registry.
//!
//! One `ModelCacheEntry` CR per cached model, with lifecycle state stored in the status
//! subresource. etcd's name-uniqueness enforces the claim atomicity that Redis gets from
//! `HSETNX`: the first replica to `create` a given CR wins; others see a 409 Conflict and
//! read back the existing status.
//!
//! CR names must be DNS-1123 compliant; model names like `meta-llama/Llama-3.1-70B` run
//! through [`sanitize_registry_name`] (lowercase + `/` -> `--`, trailing dashes
//! trimmed, sha256 suffix when over the 253-char budget) and gain a `mx-cache-` prefix
//! so they don't share a name space with the P2P `ModelMetadata` CRs. The original
//! model name lives in `spec.modelName` for human readability.

use super::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};
use crate::registry::k8s_types::{ModelCacheEntry, ModelCacheEntrySpec, phase};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use kube::{
    Client,
    api::{Api, ListParams, Patch, PatchParams, PostParams},
};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

const CR_NAME_PREFIX: &str = "mx-cache-";

/// DNS-1123 `metadata.name` hard limit.
const K8S_NAME_MAX: usize = 253;
/// Budget for the model-name-derived portion of the CR name.
const NAME_BUDGET: usize = K8S_NAME_MAX - CR_NAME_PREFIX.len();
/// Hex chars of SHA256 suffix appended when the sanitized name exceeds the budget.
const HASH_SUFFIX_LEN: usize = 12;

/// Sanitize a HuggingFace/NGC model name into a DNS-1123 `metadata.name` component.
///
/// Transform rules:
/// - `/` → `--`
/// - ASCII uppercase → lowercase
/// - `-` and `.` pass through
/// - other characters → `-`
/// - leading/trailing `-` or `.` trimmed (DNS-1123 requires alphanumeric boundaries)
///
/// The transform is lossy (case-folding, non-alphanumeric collapse), so every output
/// carries a 12-hex-char sha256 suffix derived from the **original** model name. That
/// way `google-T5/model` and `google-t5/model` never collide on the same CR name even
/// though the visible prefix is identical.
fn sanitize_registry_name(model_name: &str) -> String {
    let mut out = String::with_capacity(model_name.len());
    for c in model_name.chars() {
        match c {
            '/' => out.push_str("--"),
            c if c.is_ascii_alphanumeric() => out.push(c.to_ascii_lowercase()),
            '-' | '.' => out.push(c),
            _ => out.push('-'),
        }
    }
    let trimmed = out.trim_matches(|c: char| c == '-' || c == '.');
    let hash = hex_sha256(model_name);
    let hash_suffix = &hash[..HASH_SUFFIX_LEN];
    if trimmed.is_empty() {
        // Degenerate input ("", "///", "---"): emit just the hash.
        return hash_suffix.to_string();
    }
    // Reserve space for `-{hash}`; truncate the readable prefix if we're over budget.
    let max_prefix = NAME_BUDGET.saturating_sub(HASH_SUFFIX_LEN + 1);
    let prefix_len = trimmed.len().min(max_prefix);
    let prefix = &trimmed[..prefix_len];
    format!("{prefix}-{hash_suffix}")
}

fn hex_sha256(s: &str) -> String {
    let digest = Sha256::digest(s.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

pub struct KubernetesRegistryBackend {
    client: Client,
    namespace: String,
}

impl KubernetesRegistryBackend {
    /// Build a new backend. Actual API handshake happens in `connect`.
    pub async fn new(namespace: &str) -> RegistryResult<Self> {
        let client = Client::try_default().await?;
        Ok(Self {
            client,
            namespace: namespace.to_string(),
        })
    }

    fn api(&self) -> Api<ModelCacheEntry> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    fn cr_name_for(model_name: &str) -> String {
        format!("{CR_NAME_PREFIX}{}", sanitize_registry_name(model_name))
    }

    fn provider_str(p: ModelProvider) -> &'static str {
        match p {
            ModelProvider::HuggingFace => "HuggingFace",
            ModelProvider::Ngc => "Ngc",
        }
    }

    fn provider_from_str(s: &str) -> RegistryResult<ModelProvider> {
        match s {
            "HuggingFace" => Ok(ModelProvider::HuggingFace),
            "Ngc" => Ok(ModelProvider::Ngc),
            other => Err(format!("unknown provider in CR spec: {other:?}").into()),
        }
    }

    fn status_from_phase(phase: &str) -> ModelStatus {
        match phase {
            phase::DOWNLOADING => ModelStatus::DOWNLOADING,
            phase::DOWNLOADED => ModelStatus::DOWNLOADED,
            phase::ERROR => ModelStatus::ERROR,
            // Freshly-created CR whose status patch hasn't landed yet: treat as
            // DOWNLOADING so callers wait rather than see a missing record.
            "" => ModelStatus::DOWNLOADING,
            other => {
                warn!("Unknown ModelCacheEntry phase {other:?}, treating as ERROR");
                ModelStatus::ERROR
            }
        }
    }

    fn phase_from_status(status: ModelStatus) -> &'static str {
        match status {
            ModelStatus::DOWNLOADING => phase::DOWNLOADING,
            ModelStatus::DOWNLOADED => phase::DOWNLOADED,
            ModelStatus::ERROR => phase::ERROR,
        }
    }

    fn parse_rfc3339(s: &str, field: &str) -> RegistryResult<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| format!("invalid RFC3339 in field '{field}' ({s:?}): {e}").into())
    }

    fn record_from_cr(cr: &ModelCacheEntry) -> RegistryResult<ModelRecord> {
        let status = cr.status.clone().unwrap_or_default();
        let provider = Self::provider_from_str(&cr.spec.provider)?;
        let model_status = Self::status_from_phase(&status.phase);

        // Missing timestamps on a freshly-created CR: stamp now so downstream code
        // doesn't hit an error when the status patch is still in flight. This mirrors
        // the "create + patch-status" ordering used by try_claim_for_download.
        let now = Utc::now();
        let created_at = match status.created_at.as_deref() {
            Some(s) => Self::parse_rfc3339(s, "createdAt")?,
            None => now,
        };
        let last_used_at = match status.last_used_at.as_deref() {
            Some(s) => Self::parse_rfc3339(s, "lastUsedAt")?,
            None => now,
        };

        Ok(ModelRecord {
            model_name: cr.spec.model_name.clone(),
            provider,
            status: model_status,
            created_at,
            last_used_at,
            message: status.message,
        })
    }

    /// PATCH /status with a partial ModelCacheEntryStatus. Fields present on the patch
    /// overwrite; fields absent are preserved by the Kubernetes strategic-merge semantics
    /// (merge-patch is fine here because the status object is flat).
    async fn patch_status(
        &self,
        cr_name: &str,
        new_phase: Option<&str>,
        last_used_at: Option<&str>,
        created_at: Option<&str>,
        message: Option<Option<&str>>,
    ) -> RegistryResult<()> {
        let mut status_patch = serde_json::Map::new();
        if let Some(p) = new_phase {
            status_patch.insert("phase".into(), json!(p));
        }
        if let Some(ts) = last_used_at {
            status_patch.insert("lastUsedAt".into(), json!(ts));
        }
        if let Some(ts) = created_at {
            status_patch.insert("createdAt".into(), json!(ts));
        }
        if let Some(msg) = message {
            status_patch.insert("message".into(), json!(msg));
        }
        if status_patch.is_empty() {
            return Ok(());
        }
        let patch = json!({ "status": status_patch });
        self.api()
            .patch_status(cr_name, &PatchParams::default(), &Patch::Merge(&patch))
            .await?;
        Ok(())
    }

    /// Read back the current status for a CR name, tolerating 404 (not-found).
    async fn get_cr(&self, cr_name: &str) -> RegistryResult<Option<ModelCacheEntry>> {
        match self.api().get(cr_name).await {
            Ok(cr) => Ok(Some(cr)),
            Err(kube::Error::Api(e)) if e.code == 404 => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[async_trait]
impl RegistryBackend for KubernetesRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        // Exercise the ModelCacheEntry API to surface missing CRDs or RBAC errors early.
        let _ = self.api().list(&ListParams::default().limit(1)).await?;
        info!(
            "Registry: connected to Kubernetes, namespace '{}'",
            self.namespace
        );
        Ok(())
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        let cr_name = Self::cr_name_for(model_name);
        match self.get_cr(&cr_name).await? {
            Some(cr) => {
                let phase = cr.status.unwrap_or_default().phase;
                Ok(Some(Self::status_from_phase(&phase)))
            }
            None => Ok(None),
        }
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        let cr_name = Self::cr_name_for(model_name);
        match self.get_cr(&cr_name).await? {
            Some(cr) => Ok(Some(Self::record_from_cr(&cr)?)),
            None => Ok(None),
        }
    }

    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        let cr_name = Self::cr_name_for(model_name);
        let now = Utc::now().to_rfc3339();

        // Track whether the CR is brand new so we only stamp createdAt on first write —
        // otherwise a subsequent DOWNLOADING -> DOWNLOADED transition would clobber the
        // original timestamp.
        let existing = self.get_cr(&cr_name).await?;
        let is_new = existing.is_none();
        let needs_created_at = existing
            .as_ref()
            .and_then(|cr| cr.status.as_ref())
            .and_then(|s| s.created_at.as_deref())
            .is_none();

        if is_new {
            let cr = ModelCacheEntry::new(
                &cr_name,
                ModelCacheEntrySpec {
                    model_name: model_name.to_string(),
                    provider: Self::provider_str(provider).to_string(),
                },
            );
            match self.api().create(&PostParams::default(), &cr).await {
                Ok(_) => debug!("Created ModelCacheEntry {cr_name} via set_status"),
                Err(kube::Error::Api(e)) if e.code == 409 => {
                    debug!("ModelCacheEntry {cr_name} already exists (raced)");
                }
                Err(e) => return Err(e.into()),
            }
        }

        self.patch_status(
            &cr_name,
            Some(Self::phase_from_status(status)),
            Some(&now),
            if needs_created_at { Some(&now) } else { None },
            Some(message.as_deref()),
        )
        .await?;
        Ok(())
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        let cr_name = Self::cr_name_for(model_name);
        if self.get_cr(&cr_name).await?.is_none() {
            return Ok(()); // no-op on missing record
        }
        let now = Utc::now().to_rfc3339();
        self.patch_status(&cr_name, None, Some(&now), None, None)
            .await?;
        Ok(())
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        let cr_name = Self::cr_name_for(model_name);
        match self
            .api()
            .delete(&cr_name, &kube::api::DeleteParams::default())
            .await
        {
            Ok(_) => Ok(()),
            Err(kube::Error::Api(e)) if e.code == 404 => Ok(()), // no-op
            Err(e) => Err(e.into()),
        }
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        let crs = self.api().list(&ListParams::default()).await?;
        let mut records: Vec<ModelRecord> = Vec::with_capacity(crs.items.len());
        for cr in &crs.items {
            match Self::record_from_cr(cr) {
                Ok(r) => records.push(r),
                Err(e) => {
                    let name = cr.metadata.name.as_deref().unwrap_or("<no-name>");
                    warn!("Skipping malformed ModelCacheEntry {name}: {e}");
                }
            }
        }
        records.sort_by_key(|r| r.last_used_at);
        if let Some(n) = limit {
            records.truncate(n as usize);
        }
        Ok(records)
    }

    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        let crs = self.api().list(&ListParams::default()).await?;
        let mut downloading = 0u32;
        let mut downloaded = 0u32;
        let mut error = 0u32;
        for cr in &crs.items {
            let phase = cr
                .status
                .as_ref()
                .map(|s| s.phase.as_str())
                .unwrap_or_default();
            match Self::status_from_phase(phase) {
                ModelStatus::DOWNLOADING => downloading = downloading.saturating_add(1),
                ModelStatus::DOWNLOADED => downloaded = downloaded.saturating_add(1),
                ModelStatus::ERROR => error = error.saturating_add(1),
            }
        }
        Ok((downloading, downloaded, error))
    }

    async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<ClaimOutcome> {
        let cr_name = Self::cr_name_for(model_name);
        let cr = ModelCacheEntry::new(
            &cr_name,
            ModelCacheEntrySpec {
                model_name: model_name.to_string(),
                provider: Self::provider_str(provider).to_string(),
            },
        );

        match self.api().create(&PostParams::default(), &cr).await {
            Ok(_) => {
                // We won the claim: stamp phase + timestamps on the status subresource.
                // If patch_status fails, rollback the CR we just created so a retry
                // can try again and other replicas don't observe a CR with an empty
                // phase (which status_from_phase treats as DOWNLOADING — a lie).
                let now = Utc::now().to_rfc3339();
                if let Err(patch_err) = self
                    .patch_status(
                        &cr_name,
                        Some(phase::DOWNLOADING),
                        Some(&now),
                        Some(&now),
                        Some(Some("Starting download...")),
                    )
                    .await
                {
                    if let Err(delete_err) = self
                        .api()
                        .delete(&cr_name, &kube::api::DeleteParams::default())
                        .await
                    {
                        warn!(
                            "patch_status failed for {cr_name}; rollback delete also \
                             failed: {delete_err}. CR may be left with empty phase."
                        );
                    } else {
                        debug!("Rolled back {cr_name} after patch_status failure: {patch_err}");
                    }
                    return Err(patch_err);
                }
                Ok(ClaimOutcome::Claimed)
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                // Already exists: someone else claimed it (or an earlier run did). Read
                // back the current phase.
                let existing = self
                    .get_cr(&cr_name)
                    .await?
                    .ok_or("ModelCacheEntry disappeared between 409 and GET")?;
                let phase_str = existing.status.unwrap_or_default().phase;
                Ok(ClaimOutcome::AlreadyExists(Self::status_from_phase(
                    &phase_str,
                )))
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        _provider: ModelProvider,
    ) -> RegistryResult<bool> {
        let cr_name = Self::cr_name_for(model_name);
        let Some(existing) = self.get_cr(&cr_name).await? else {
            return Ok(false);
        };
        let current_phase = existing
            .status
            .as_ref()
            .map(|s| s.phase.as_str())
            .unwrap_or_default()
            .to_string();
        if current_phase != phase::ERROR {
            return Ok(false);
        }
        // Use a JSON Patch `test` op as a server-side precondition — kube rejects the
        // patch with 422 if the status phase was flipped out from under us between
        // GET and PATCH, and we report the CAS miss.
        let now = Utc::now().to_rfc3339();
        let patch = json!([
            { "op": "test", "path": "/status/phase", "value": phase::ERROR },
            { "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADING },
            { "op": "replace", "path": "/status/message", "value": "Retrying download..." },
            { "op": "replace", "path": "/status/lastUsedAt", "value": now },
        ]);
        match self
            .api()
            .patch_status(
                &cr_name,
                &PatchParams::default(),
                &Patch::<()>::Json(serde_json::from_value(patch).map_err(|e| e.to_string())?),
            )
            .await
        {
            Ok(_) => Ok(true),
            Err(kube::Error::Api(e)) if e.code == 422 || e.code == 409 => {
                debug!("Error-retry CAS for {cr_name} lost to a concurrent write");
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_preserves_readable_prefix() {
        // The readable prefix is still present; the hash suffix disambiguates collisions.
        assert!(sanitize_registry_name("org/model").starts_with("org--model-"));
        assert!(
            sanitize_registry_name("meta-llama/Llama-3.1-70B")
                .starts_with("meta-llama--llama-3.1-70b-")
        );
    }

    #[test]
    fn sanitize_distinguishes_slash_from_single_dash() {
        assert_ne!(
            sanitize_registry_name("org/model"),
            sanitize_registry_name("org-model")
        );
    }

    #[test]
    fn sanitize_distinguishes_case() {
        // Case-folding used to collide silently; the always-on hash suffix (of the
        // original case-preserving name) prevents that.
        assert_ne!(
            sanitize_registry_name("Foo/Bar"),
            sanitize_registry_name("foo/bar")
        );
    }

    #[test]
    fn sanitize_handles_degenerate_input() {
        let hashed = sanitize_registry_name("");
        assert_eq!(hashed.len(), HASH_SUFFIX_LEN);
        let hashed = sanitize_registry_name("///");
        assert_eq!(hashed.len(), HASH_SUFFIX_LEN);
        let hashed = sanitize_registry_name("---");
        assert_eq!(hashed.len(), HASH_SUFFIX_LEN);
    }

    #[test]
    fn sanitize_fits_dns_1123_budget() {
        let long: String = "a".repeat(300);
        let out = sanitize_registry_name(&long);
        assert!(out.len() <= NAME_BUDGET);
        // Still distinguishes two different 300-char names via the hash suffix.
        let other: String = format!("{}b", "a".repeat(299));
        assert_ne!(
            sanitize_registry_name(&long),
            sanitize_registry_name(&other)
        );
    }

    #[test]
    fn cr_name_stays_within_k8s_limit() {
        let long = "a".repeat(300);
        let name = KubernetesRegistryBackend::cr_name_for(&long);
        assert!(name.len() <= K8S_NAME_MAX);
        assert!(name.starts_with(CR_NAME_PREFIX));
    }

    #[test]
    fn sanitize_trims_leading_trailing_dashes() {
        assert!(sanitize_registry_name("-model-").starts_with("model-"));
        assert!(sanitize_registry_name(".model.").starts_with("model-"));
    }
}
