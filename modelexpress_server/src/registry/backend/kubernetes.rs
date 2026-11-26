// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD backend for the model registry.
//!
//! One `ModelCacheEntry` CR per (provider, cached model), state in the status subresource.
//! etcd name-uniqueness establishes one record, then a resource-version-fenced status patch
//! establishes the download owner. This is the analogue of Redis's atomic claim script.
//!
//! CR name: `mx-cache-{sanitize(provider/model_name)}` (DNS-1123, sha256 suffix). The
//! provider is in the name so the same model under different providers maps to distinct CRs;
//! `spec.modelName`/`spec.provider` keep the originals. Pre-0.5.0 name-only CRs are migrated
//! lazily on claim (see `try_claim_for_download`); name-addressed reads cover both forms.

use super::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};
use crate::registry::k8s_types::{ModelCacheEntry, ModelCacheEntrySpec, phase};

pub mod finalizer;

/// Finalizer attached to `ModelCacheEntry` CRs so deleting a CR reclaims its
/// cached files before the CR disappears from Kubernetes.
pub const CACHE_FILES_FINALIZER: &str = "modelexpress.nvidia.com/cache-files";
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use k8s_openapi::api::authorization::v1::{
    ResourceAttributes, SelfSubjectAccessReview, SelfSubjectAccessReviewSpec,
};
use kube::{
    Client,
    api::{Api, ListParams, Patch, PatchParams, PostParams},
};
use modelexpress_common::digest::{DigestError, sha256_hex};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Mutex, PoisonError},
    time::Instant,
};
use tracing::{debug, info, warn};

const CR_NAME_PREFIX: &str = "mx-cache-";

/// Every provider, for enumerating candidate CR names in name-addressed lookups.
const ALL_PROVIDERS: [ModelProvider; 4] = [
    ModelProvider::HuggingFace,
    ModelProvider::Ngc,
    ModelProvider::Gcs,
    ModelProvider::S3,
];

/// DNS-1123 `metadata.name` hard limit.
const K8S_NAME_MAX: usize = 253;
/// Budget for the model-name-derived portion of the CR name.
const NAME_BUDGET: usize = K8S_NAME_MAX - CR_NAME_PREFIX.len();
/// Hex chars of SHA256 suffix appended when the sanitized name exceeds the budget.
const HASH_SUFFIX_LEN: usize = 12;

#[derive(Default)]
struct StatusPatch<'a> {
    phase: Option<&'a str>,
    last_used_at: Option<&'a str>,
    created_at: Option<&'a str>,
    message: Option<Option<&'a str>>,
    claim_id: Option<Option<&'a str>>,
    lease_expires_at: Option<Option<&'a str>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LeaseFingerprint {
    claim_id: Option<String>,
    lease_expires_at: Option<String>,
}

#[derive(Debug)]
struct LeaseObservation {
    fingerprint: LeaseFingerprint,
    observed_at: Instant,
}

fn lease_observation_expired(
    observations: &mut HashMap<String, LeaseObservation>,
    cr_name: &str,
    fingerprint: LeaseFingerprint,
    now: Instant,
    lease_duration: std::time::Duration,
) -> bool {
    // Kubernetes lease expiry is measured from the first local observation of an
    // unchanged owner/heartbeat fingerprint, not from the owner's wall-clock deadline.
    // A waiter must therefore observe the same fingerprint again after `lease_duration`;
    // restarting this backend clears the observation and restarts that interval. With the
    // service's 10-second heartbeat and 30-second lease, continuously polling waiters
    // reclaim about 30 seconds after the final observed heartbeat. Redis instead uses
    // Redis server time and can reclaim on the first request after the stored deadline.
    match observations.get_mut(cr_name) {
        Some(observation) if observation.fingerprint == fingerprint => {
            now.duration_since(observation.observed_at) >= lease_duration
        }
        Some(observation) => {
            observation.fingerprint = fingerprint;
            observation.observed_at = now;
            lease_duration.is_zero()
        }
        None => {
            observations.insert(
                cr_name.to_string(),
                LeaseObservation {
                    fingerprint,
                    observed_at: now,
                },
            );
            lease_duration.is_zero()
        }
    }
}

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
fn sanitize_registry_name(model_name: &str) -> Result<String, DigestError> {
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
    let hash = sha256_hex(model_name.as_bytes())?;
    let hash_suffix = &hash[..HASH_SUFFIX_LEN];
    if trimmed.is_empty() {
        // Degenerate input ("", "///", "---"): emit just the hash.
        return Ok(hash_suffix.to_string());
    }
    // Reserve space for `-{hash}`; truncate the readable prefix if we're over budget.
    let max_prefix = NAME_BUDGET.saturating_sub(HASH_SUFFIX_LEN + 1);
    let prefix_len = trimmed.len().min(max_prefix);
    let prefix = &trimmed[..prefix_len];
    Ok(format!("{prefix}-{hash_suffix}"))
}

pub struct KubernetesRegistryBackend {
    client: Client,
    namespace: String,
    lease_observations: Mutex<HashMap<String, LeaseObservation>>,
}

impl KubernetesRegistryBackend {
    /// Build a new backend. Actual API handshake happens in `connect`.
    pub async fn new(namespace: &str) -> RegistryResult<Self> {
        let client = Client::try_default().await?;
        Ok(Self {
            client,
            namespace: namespace.to_string(),
            lease_observations: Mutex::new(HashMap::new()),
        })
    }

    fn api(&self) -> Api<ModelCacheEntry> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    /// Provider-scoped CR name: `sanitize("{provider}/{model_name}")`, so the sha256 suffix
    /// binds (provider, name) and providers never collide.
    fn cr_name_for(provider: ModelProvider, model_name: &str) -> Result<String, DigestError> {
        Ok(format!(
            "{CR_NAME_PREFIX}{}",
            sanitize_registry_name(&format!("{}/{model_name}", Self::provider_str(provider)))?
        ))
    }

    /// Legacy name-only CR name, pre-provider-scoped CRs.
    /// TODO(0.5.0 migration): remove once no deployment has pre-0.5.0 CRs; see
    /// `try_claim_for_download`.
    fn legacy_cr_name_for(model_name: &str) -> Result<String, DigestError> {
        Ok(format!(
            "{CR_NAME_PREFIX}{}",
            sanitize_registry_name(model_name)?
        ))
    }

    /// CR names a record for `model_name` may live under when the provider isn't known up
    /// front: the provider-scoped name for every provider, plus the legacy name-only name.
    fn candidate_cr_names(model_name: &str) -> Result<Vec<String>, DigestError> {
        let mut names = ALL_PROVIDERS
            .iter()
            .map(|p| Self::cr_name_for(*p, model_name))
            .collect::<Result<Vec<_>, _>>()?;
        names.push(Self::legacy_cr_name_for(model_name)?);
        Ok(names)
    }

    /// Build a new `ModelCacheEntry` carrying the cache-files finalizer at creation.
    fn new_cr_with_finalizer(cr_name: &str, model_name: &str, provider: &str) -> ModelCacheEntry {
        let mut cr = ModelCacheEntry::new(
            cr_name,
            ModelCacheEntrySpec {
                model_name: model_name.to_string(),
                provider: provider.to_string(),
            },
        );
        cr.metadata.finalizers = Some(vec![CACHE_FILES_FINALIZER.to_string()]);
        cr
    }

    pub(crate) fn is_terminating(cr: &ModelCacheEntry) -> bool {
        cr.metadata.deletion_timestamp.is_some()
    }

    /// Status reported to download callers inspecting an existing CR.
    ///
    /// A terminating CR is always observed as `DOWNLOADING` so callers wait for
    /// finalizer cleanup and CR disappearance rather than treating a deleting CR as
    /// `DOWNLOADED`, retrying it, or taking over its lease.
    fn observed_claim_status(cr: &ModelCacheEntry) -> ModelStatus {
        if Self::is_terminating(cr) {
            return ModelStatus::DOWNLOADING;
        }
        let phase = cr
            .status
            .as_ref()
            .map(|s| s.phase.as_str())
            .unwrap_or_default();
        Self::status_from_phase(phase)
    }

    /// First live (non-terminating) existing CR among the candidates (provider-scoped
    /// keys, then legacy). Used by name-addressed reads; in practice a name maps to a
    /// single provider.
    async fn find_existing_cr(&self, model_name: &str) -> RegistryResult<Option<ModelCacheEntry>> {
        for cr_name in Self::candidate_cr_names(model_name)? {
            if let Some(cr) = self.get_cr(&cr_name).await?
                && !Self::is_terminating(&cr)
            {
                return Ok(Some(cr));
            }
        }
        Ok(None)
    }

    /// Lazy migration: if a matching-provider legacy CR exists, recreate it under the
    /// provider-scoped name (copying status), delete the legacy CR, return `AlreadyExists`.
    /// `None` if no legacy CR or it's a different provider (caller then claims fresh).
    /// TODO(0.5.0 migration): remove once all deployments have drained legacy CRs.
    async fn adopt_legacy_cr(
        &self,
        model_name: &str,
        provider: ModelProvider,
        scoped_name: &str,
    ) -> RegistryResult<Option<ClaimOutcome>> {
        let legacy_name = Self::legacy_cr_name_for(model_name)?;
        let Some(legacy_cr) = self.get_cr(&legacy_name).await? else {
            return Ok(None);
        };
        if Self::provider_from_str(&legacy_cr.spec.provider)? != provider {
            return Ok(None);
        }
        if Self::is_terminating(&legacy_cr) {
            return Ok(Some(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)));
        }

        let legacy_status = legacy_cr.status.clone().unwrap_or_default();
        let new_cr = Self::new_cr_with_finalizer(
            scoped_name,
            &legacy_cr.spec.model_name,
            &legacy_cr.spec.provider,
        );
        // create is atomic; a 409 means a concurrent claim/migration already made the
        // provider-scoped CR, so we adopt whatever it holds.
        match self.api().create(&PostParams::default(), &new_cr).await {
            Ok(_) => {
                self.patch_status(
                    scoped_name,
                    StatusPatch {
                        phase: Some(Self::phase_from_status(Self::status_from_phase(
                            &legacy_status.phase,
                        ))),
                        last_used_at: legacy_status.last_used_at.as_deref(),
                        created_at: legacy_status.created_at.as_deref(),
                        message: Some(legacy_status.message.as_deref()),
                        claim_id: Some(legacy_status.claim_id.as_deref()),
                        lease_expires_at: Some(legacy_status.lease_expires_at.as_deref()),
                    },
                )
                .await?;
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                debug!("{scoped_name} already exists during legacy migration; adopting it");
            }
            Err(e) => return Err(e.into()),
        }

        // Best-effort delete of the now-migrated legacy CR (idempotent; tolerate 404).
        if let Err(e) = self
            .api()
            .delete(&legacy_name, &kube::api::DeleteParams::default())
            .await
            && !matches!(&e, kube::Error::Api(a) if a.code == 404)
        {
            warn!("Failed to delete legacy CR {legacy_name} after migration: {e}");
        }

        let status = self
            .get_cr(scoped_name)
            .await?
            .map_or(ModelStatus::DOWNLOADING, |cr| {
                Self::observed_claim_status(&cr)
            });
        Ok(Some(ClaimOutcome::AlreadyExists(status)))
    }

    fn provider_str(p: ModelProvider) -> &'static str {
        match p {
            ModelProvider::HuggingFace => "HuggingFace",
            ModelProvider::Ngc => "Ngc",
            ModelProvider::Gcs => "Gcs",
            ModelProvider::S3 => "S3",
        }
    }

    pub(crate) fn provider_from_str(s: &str) -> RegistryResult<ModelProvider> {
        match s {
            "HuggingFace" => Ok(ModelProvider::HuggingFace),
            "Ngc" => Ok(ModelProvider::Ngc),
            "Gcs" => Ok(ModelProvider::Gcs),
            "S3" => Ok(ModelProvider::S3),
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

    fn lease_deadline(duration: std::time::Duration) -> DateTime<Utc> {
        let now = Utc::now();
        chrono::TimeDelta::from_std(duration)
            .ok()
            .and_then(|duration| now.checked_add_signed(duration))
            .unwrap_or(now)
    }

    fn lease_has_expired(
        &self,
        cr_name: &str,
        status: &crate::registry::k8s_types::ModelCacheEntryStatus,
        lease_duration: std::time::Duration,
    ) -> bool {
        let fingerprint = LeaseFingerprint {
            claim_id: status.claim_id.clone(),
            lease_expires_at: status.lease_expires_at.clone(),
        };
        let now = Instant::now();
        let mut observations = self
            .lease_observations
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        lease_observation_expired(&mut observations, cr_name, fingerprint, now, lease_duration)
    }

    fn clear_lease_observation(&self, cr_name: &str) {
        self.lease_observations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(cr_name);
    }

    pub(crate) fn record_from_cr(cr: &ModelCacheEntry) -> RegistryResult<ModelRecord> {
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
    async fn patch_status(&self, cr_name: &str, update: StatusPatch<'_>) -> RegistryResult<()> {
        let mut status_patch = serde_json::Map::new();
        if let Some(p) = update.phase {
            status_patch.insert("phase".into(), json!(p));
        }
        if let Some(ts) = update.last_used_at {
            status_patch.insert("lastUsedAt".into(), json!(ts));
        }
        if let Some(ts) = update.created_at {
            status_patch.insert("createdAt".into(), json!(ts));
        }
        if let Some(msg) = update.message {
            status_patch.insert("message".into(), json!(msg));
        }
        if let Some(id) = update.claim_id {
            status_patch.insert("claimId".into(), json!(id));
        }
        if let Some(expires_at) = update.lease_expires_at {
            status_patch.insert("leaseExpiresAt".into(), json!(expires_at));
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

    fn build_reclaim_downloaded_patch(
        existing: &ModelCacheEntry,
        claim_id: &str,
        now: &str,
        lease_expires_at: &str,
    ) -> RegistryResult<Option<serde_json::Value>> {
        if existing.metadata.deletion_timestamp.is_some() {
            return Ok(None);
        }
        let current_phase = existing
            .status
            .as_ref()
            .map(|s| s.phase.as_str())
            .unwrap_or_default();
        if current_phase != phase::DOWNLOADED {
            return Ok(None);
        }
        let resource_version = existing
            .metadata
            .resource_version
            .as_deref()
            .ok_or("ModelCacheEntry has no resourceVersion")?;
        Ok(Some(json!([
            { "op": "test", "path": "/metadata/resourceVersion", "value": resource_version },
            { "op": "test", "path": "/status/phase", "value": phase::DOWNLOADED },
            { "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADING },
            { "op": "add", "path": "/status/message", "value": "Re-downloading missing cached files..." },
            { "op": "add", "path": "/status/lastUsedAt", "value": now },
            { "op": "add", "path": "/status/claimId", "value": claim_id },
            { "op": "add", "path": "/status/leaseExpiresAt", "value": lease_expires_at },
        ])))
    }

    async fn try_claim_existing(
        &self,
        cr: ModelCacheEntry,
        claim_id: &str,
        lease_duration: std::time::Duration,
        claim_uninitialized: bool,
    ) -> RegistryResult<ClaimOutcome> {
        if Self::is_terminating(&cr) {
            if let Some(cr_name) = cr.metadata.name.as_deref() {
                self.clear_lease_observation(cr_name);
            }
            return Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING));
        }
        let status_missing = cr.status.is_none();
        let status = cr.status.unwrap_or_default();
        let current = Self::status_from_phase(&status.phase);
        if current != ModelStatus::DOWNLOADING {
            if let Some(cr_name) = cr.metadata.name.as_deref() {
                self.clear_lease_observation(cr_name);
            }
            return Ok(ClaimOutcome::AlreadyExists(current));
        }

        let cr_name = cr
            .metadata
            .name
            .as_deref()
            .ok_or("ModelCacheEntry has no name")?;
        let resource_version = cr
            .metadata
            .resource_version
            .as_deref()
            .ok_or("ModelCacheEntry has no resourceVersion")?;
        // Already computed here, so the fresh/takeover distinction costs nothing:
        // an uninitialized claim is a first download, anything else reaching this
        // point got here because the previous owner's lease expired.
        let is_fresh_claim = status_missing && claim_uninitialized;
        if !is_fresh_claim && !self.lease_has_expired(cr_name, &status, lease_duration) {
            return Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING));
        }
        let now = Utc::now().to_rfc3339();
        let lease_expires_at = Self::lease_deadline(lease_duration).to_rfc3339();
        let mut patch = vec![
            json!({ "op": "test", "path": "/metadata/resourceVersion", "value": resource_version }),
        ];
        if status_missing {
            patch.push(json!({
                "op": "add",
                "path": "/status",
                "value": {
                    "phase": phase::DOWNLOADING,
                    "createdAt": now,
                    "lastUsedAt": now,
                    "message": "Starting download...",
                    "claimId": claim_id,
                    "leaseExpiresAt": lease_expires_at,
                }
            }));
        } else {
            patch.extend([
                json!({ "op": "test", "path": "/status/phase", "value": phase::DOWNLOADING }),
                json!({ "op": "add", "path": "/status/claimId", "value": claim_id }),
                json!({ "op": "add", "path": "/status/leaseExpiresAt", "value": lease_expires_at }),
                json!({ "op": "add", "path": "/status/message", "value": "Taking over expired download lease..." }),
                json!({ "op": "add", "path": "/status/lastUsedAt", "value": now }),
            ]);
        }
        match self
            .api()
            .patch_status(
                cr_name,
                &PatchParams::default(),
                &Patch::<()>::Json(
                    serde_json::from_value(serde_json::Value::Array(patch))
                        .map_err(|e| e.to_string())?,
                ),
            )
            .await
        {
            Ok(_) => {
                self.clear_lease_observation(cr_name);
                if is_fresh_claim {
                    Ok(ClaimOutcome::Claimed)
                } else {
                    Ok(ClaimOutcome::TookOver)
                }
            }
            Err(kube::Error::Api(e)) if e.code == 409 || e.code == 422 => {
                let current = self
                    .get_cr(cr_name)
                    .await?
                    .map_or(ModelStatus::DOWNLOADING, |entry| {
                        Self::observed_claim_status(&entry)
                    });
                Ok(ClaimOutcome::AlreadyExists(current))
            }
            Err(e) => Err(e.into()),
        }
    }
    /// Maximum retries for `refresh_download_claim` / `finish_download_claim`
    /// when a concurrent metadata-only update bumps `resourceVersion`.
    const MAX_CLAIM_PATCH_RETRIES: usize = 5;

    /// Patch `/status` on an owned `DOWNLOADING` CR while guarding UID,
    /// `resourceVersion`, phase, and `claimId`. Retries with a bound on 409/422
    /// conflicts caused by unrelated metadata updates (such as finalizer or
    /// annotation edits) so long as the CR still has the same UID, is not
    /// terminating, and still belongs to `claim_id` in phase `Downloading`.
    async fn patch_owned_downloading_status<F>(
        &self,
        cr_name: &str,
        claim_id: &str,
        op_label: &str,
        build_mutations: F,
    ) -> RegistryResult<bool>
    where
        F: Fn(&str) -> Vec<serde_json::Value>,
    {
        let mut expected_uid: Option<String> = None;

        for attempt in 0..Self::MAX_CLAIM_PATCH_RETRIES {
            let Some(existing) = self.get_cr(cr_name).await? else {
                return Ok(false);
            };
            if Self::is_terminating(&existing) {
                debug!("{op_label} for {cr_name} rejected: CR is terminating");
                return Ok(false);
            }

            let current_uid = existing.metadata.uid.as_deref();
            if let Some(uid) = expected_uid.as_deref() {
                if current_uid != Some(uid) {
                    debug!("{op_label} for {cr_name} rejected: CR UID changed");
                    return Ok(false);
                }
            } else {
                expected_uid = current_uid.map(str::to_string);
            }

            let Some(status) = existing.status.as_ref() else {
                return Ok(false);
            };
            if status.phase != phase::DOWNLOADING || status.claim_id.as_deref() != Some(claim_id) {
                debug!("{op_label} for {cr_name} rejected: lease ownership or phase changed");
                return Ok(false);
            }

            let resource_version = existing
                .metadata
                .resource_version
                .as_deref()
                .ok_or("ModelCacheEntry has no resourceVersion")?;
            let now = Utc::now().to_rfc3339();
            let mut ops = Vec::with_capacity(8);
            if let Some(uid) = expected_uid.as_deref() {
                ops.push(json!({ "op": "test", "path": "/metadata/uid", "value": uid }));
            }
            ops.push(json!({ "op": "test", "path": "/metadata/resourceVersion", "value": resource_version }));
            ops.push(json!({ "op": "test", "path": "/status/phase", "value": phase::DOWNLOADING }));
            ops.push(json!({ "op": "test", "path": "/status/claimId", "value": claim_id }));
            ops.extend(build_mutations(&now));

            match self
                .api()
                .patch_status(
                    cr_name,
                    &PatchParams::default(),
                    &Patch::<()>::Json(
                        serde_json::from_value(serde_json::Value::Array(ops))
                            .map_err(|e| e.to_string())?,
                    ),
                )
                .await
            {
                Ok(_) => return Ok(true),
                Err(kube::Error::Api(e)) if e.code == 404 => return Ok(false),
                Err(kube::Error::Api(e)) if e.code == 409 || e.code == 422 => {
                    debug!(
                        "{op_label} for {cr_name} hit conflict/precondition (attempt {}); re-checking",
                        attempt.saturating_add(1)
                    );
                }
                Err(e) => return Err(e.into()),
            }
        }

        Err(format!(
            "{op_label} for {cr_name} exhausted {} retries on concurrent resourceVersion conflicts",
            Self::MAX_CLAIM_PATCH_RETRIES
        )
        .into())
    }

    /// Verify via `SelfSubjectAccessReview` that the server ServiceAccount has `verb`
    /// on `modelcacheentries` (or a subresource such as `"status"`) in `self.namespace`.
    async fn verify_rbac_permission(
        &self,
        verb: &str,
        subresource: Option<&str>,
    ) -> RegistryResult<()> {
        let review_api: Api<SelfSubjectAccessReview> = Api::all(self.client.clone());
        let review = SelfSubjectAccessReview {
            spec: SelfSubjectAccessReviewSpec {
                resource_attributes: Some(ResourceAttributes {
                    group: Some("modelexpress.nvidia.com".to_string()),
                    namespace: Some(self.namespace.clone()),
                    resource: Some("modelcacheentries".to_string()),
                    subresource: subresource.map(str::to_string),
                    verb: Some(verb.to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let response = review_api.create(&PostParams::default(), &review).await?;
        let allowed = response
            .status
            .as_ref()
            .is_some_and(|status| status.allowed);
        if !allowed {
            let target = match subresource {
                Some(sub) => format!("modelcacheentries/{sub}"),
                None => "modelcacheentries".to_string(),
            };
            let reason = response
                .status
                .and_then(|status| status.reason)
                .filter(|r| !r.is_empty())
                .map(|r| format!(" ({r})"))
                .unwrap_or_default();
            return Err(format!(
                "Kubernetes RBAC check failed: ServiceAccount lacks '{verb}' permission on \
                 '{target}.modelexpress.nvidia.com' in namespace '{}'{reason}",
                self.namespace
            )
            .into());
        }
        Ok(())
    }
}

#[async_trait]
impl RegistryBackend for KubernetesRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        // Exercise the ModelCacheEntry API to surface missing CRDs or RBAC errors early,
        // then verify the verbs required by the finalizer controller (watch, metadata
        // patch, and status patch). Intentionally does not check `delete` so read-only
        // or non-deleting roles remain valid.
        let _ = self.api().list(&ListParams::default().limit(1)).await?;
        self.verify_rbac_permission("watch", None).await?;
        self.verify_rbac_permission("patch", None).await?;
        self.verify_rbac_permission("patch", Some("status")).await?;
        info!(
            "Registry: connected to Kubernetes, namespace '{}'",
            self.namespace
        );
        Ok(())
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        match self.find_existing_cr(model_name).await? {
            Some(cr) => {
                let phase = cr.status.unwrap_or_default().phase;
                Ok(Some(Self::status_from_phase(&phase)))
            }
            None => Ok(None),
        }
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        match self.find_existing_cr(model_name).await? {
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
        let cr_name = Self::cr_name_for(provider, model_name)?;
        let now = Utc::now().to_rfc3339();

        // Track whether the CR is brand new so we only stamp createdAt on first write —
        // otherwise a subsequent DOWNLOADING -> DOWNLOADED transition would clobber the
        // original timestamp.
        let existing = self.get_cr(&cr_name).await?;
        if existing.as_ref().is_some_and(Self::is_terminating) {
            return Err(format!("ModelCacheEntry {cr_name} is terminating").into());
        }
        let is_new = existing.is_none();
        let needs_created_at = existing
            .as_ref()
            .and_then(|cr| cr.status.as_ref())
            .and_then(|s| s.created_at.as_deref())
            .is_none();

        if is_new {
            let cr =
                Self::new_cr_with_finalizer(&cr_name, model_name, Self::provider_str(provider));
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
            StatusPatch {
                phase: Some(Self::phase_from_status(status)),
                last_used_at: Some(&now),
                created_at: if needs_created_at { Some(&now) } else { None },
                message: Some(message.as_deref()),
                claim_id: if status == ModelStatus::DOWNLOADING {
                    None
                } else {
                    Some(None)
                },
                lease_expires_at: if status == ModelStatus::DOWNLOADING {
                    None
                } else {
                    Some(None)
                },
            },
        )
        .await?;
        Ok(())
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        // Provider-agnostic: patch lastUsedAt on whichever candidate CR holds the record.
        let Some(cr) = self.find_existing_cr(model_name).await? else {
            return Ok(()); // no-op on missing record
        };
        let Some(cr_name) = cr.metadata.name.as_deref() else {
            return Ok(());
        };
        let now = Utc::now().to_rfc3339();
        self.patch_status(
            cr_name,
            StatusPatch {
                last_used_at: Some(&now),
                ..StatusPatch::default()
            },
        )
        .await?;
        Ok(())
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        // Delete every variant (all providers + legacy); 404 is a no-op.
        for cr_name in Self::candidate_cr_names(model_name)? {
            match self
                .api()
                .delete(&cr_name, &kube::api::DeleteParams::default())
                .await
            {
                Ok(_) => {}
                Err(kube::Error::Api(e)) if e.code == 404 => {}
                Err(e) => return Err(e.into()),
            }
            self.clear_lease_observation(&cr_name);
        }
        Ok(())
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        let crs = self.api().list(&ListParams::default()).await?;
        let mut records: Vec<ModelRecord> = Vec::with_capacity(crs.items.len());
        for cr in &crs.items {
            if Self::is_terminating(cr) {
                continue;
            }
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
            if Self::is_terminating(cr) {
                continue;
            }
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
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<ClaimOutcome> {
        let cr_name = Self::cr_name_for(provider, model_name)?;

        // Fast path (mirrors Redis's scoped-first check): a non-mutating GET returns the
        // existing status without committing a claim, skipping the legacy lookup in steady state.
        if let Some(cr) = self.get_cr(&cr_name).await? {
            return self
                .try_claim_existing(cr, claim_id, lease_duration, false)
                .await;
        }

        // Scoped CR absent: adopt a matching-provider legacy CR if present; a different-provider
        // legacy CR is left untouched (the fix: a GCS record can't answer a HuggingFace claim).
        if let Some(outcome) = self.adopt_legacy_cr(model_name, provider, &cr_name).await? {
            return Ok(outcome);
        }

        let cr = Self::new_cr_with_finalizer(&cr_name, model_name, Self::provider_str(provider));

        match self.api().create(&PostParams::default(), &cr).await {
            Ok(created) => {
                self.try_claim_existing(created, claim_id, lease_duration, true)
                    .await
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                // Lost a race: another replica created the scoped CR between our fast-path
                // GET and this create. Read back the current phase.
                let existing = self
                    .get_cr(&cr_name)
                    .await?
                    .ok_or("ModelCacheEntry disappeared between 409 and GET")?;
                self.try_claim_existing(existing, claim_id, lease_duration, false)
                    .await
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        // Retry only runs after a claim observed AlreadyExists (which already migrated any
        // legacy CR), so the CAS targets the provider-scoped CR directly.
        let cr_name = Self::cr_name_for(provider, model_name)?;
        let Some(existing) = self.get_cr(&cr_name).await? else {
            return Ok(false);
        };
        if Self::is_terminating(&existing) {
            return Ok(false);
        }
        let current_phase = existing
            .status
            .as_ref()
            .map(|s| s.phase.as_str())
            .unwrap_or_default()
            .to_string();
        if current_phase != phase::ERROR {
            return Ok(false);
        }
        let resource_version = existing
            .metadata
            .resource_version
            .as_deref()
            .ok_or("ModelCacheEntry has no resourceVersion")?;
        // Use JSON Patch `test` ops on resourceVersion and phase so a concurrent
        // delete (setting deletionTimestamp) or retry flips the CAS to Ok(false).
        let now = Utc::now().to_rfc3339();
        let lease_expires_at = Self::lease_deadline(lease_duration).to_rfc3339();
        let patch = json!([
            { "op": "test", "path": "/metadata/resourceVersion", "value": resource_version },
            { "op": "test", "path": "/status/phase", "value": phase::ERROR },
            { "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADING },
            { "op": "replace", "path": "/status/message", "value": "Retrying download..." },
            { "op": "replace", "path": "/status/lastUsedAt", "value": now },
            { "op": "add", "path": "/status/claimId", "value": claim_id },
            { "op": "add", "path": "/status/leaseExpiresAt", "value": lease_expires_at },
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
            Ok(_) => {
                self.clear_lease_observation(&cr_name);
                Ok(true)
            }
            Err(kube::Error::Api(e)) if e.code == 404 || e.code == 422 || e.code == 409 => {
                debug!("Error-retry CAS for {cr_name} lost to a concurrent write");
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn try_reclaim_downloaded(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let cr_name = Self::cr_name_for(provider, model_name)?;
        let Some(existing) = self.get_cr(&cr_name).await? else {
            return Ok(false);
        };
        let now = Utc::now().to_rfc3339();
        let lease_expires_at = Self::lease_deadline(lease_duration).to_rfc3339();
        let Some(patch) =
            Self::build_reclaim_downloaded_patch(&existing, claim_id, &now, &lease_expires_at)?
        else {
            return Ok(false);
        };
        match self
            .api()
            .patch_status(
                &cr_name,
                &PatchParams::default(),
                &Patch::<()>::Json(serde_json::from_value(patch).map_err(|e| e.to_string())?),
            )
            .await
        {
            Ok(_) => {
                self.clear_lease_observation(&cr_name);
                Ok(true)
            }
            Err(kube::Error::Api(e)) if e.code == 404 || e.code == 409 || e.code == 422 => {
                debug!("Stale-downloaded reclaim CAS for {cr_name} lost to a concurrent write");
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn refresh_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let cr_name = Self::cr_name_for(provider, model_name)?;
        self.patch_owned_downloading_status(&cr_name, claim_id, "Download lease refresh", |now| {
            let lease_expires_at = Self::lease_deadline(lease_duration).to_rfc3339();
            vec![
                json!({ "op": "add", "path": "/status/leaseExpiresAt", "value": lease_expires_at }),
                json!({ "op": "add", "path": "/status/lastUsedAt", "value": now }),
            ]
        })
        .await
    }

    async fn finish_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<bool> {
        let cr_name = Self::cr_name_for(provider, model_name)?;
        let phase_str = Self::phase_from_status(status);
        let updated = self
            .patch_owned_downloading_status(&cr_name, claim_id, "Download completion", |now| {
                vec![
                    json!({ "op": "replace", "path": "/status/phase", "value": phase_str }),
                    json!({ "op": "add", "path": "/status/message", "value": message }),
                    json!({ "op": "add", "path": "/status/lastUsedAt", "value": now }),
                    json!({ "op": "add", "path": "/status/claimId", "value": null }),
                    json!({ "op": "add", "path": "/status/leaseExpiresAt", "value": null }),
                ]
            })
            .await?;
        if updated {
            self.clear_lease_observation(&cr_name);
        }
        Ok(updated)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use kube::CustomResourceExt;
    use std::collections::BTreeSet;

    #[test]
    fn lease_expires_only_after_unchanged_heartbeat_duration() {
        let mut observations = HashMap::new();
        let started = Instant::now();
        let duration = std::time::Duration::from_secs(30);
        let first = LeaseFingerprint {
            claim_id: Some("owner".to_string()),
            lease_expires_at: Some("heartbeat-1".to_string()),
        };
        assert!(!lease_observation_expired(
            &mut observations,
            "model",
            first.clone(),
            started,
            duration,
        ));
        assert!(!lease_observation_expired(
            &mut observations,
            "model",
            first.clone(),
            started + std::time::Duration::from_secs(29),
            duration,
        ));

        let renewed = LeaseFingerprint {
            claim_id: Some("owner".to_string()),
            lease_expires_at: Some("heartbeat-2".to_string()),
        };
        assert!(!lease_observation_expired(
            &mut observations,
            "model",
            renewed.clone(),
            started + duration,
            duration,
        ));
        assert!(lease_observation_expired(
            &mut observations,
            "model",
            renewed,
            started + duration + duration,
            duration,
        ));
    }

    #[test]
    fn sanitize_preserves_readable_prefix() {
        // The readable prefix is still present; the hash suffix disambiguates collisions.
        assert!(
            sanitize_registry_name("org/model")
                .expect("name")
                .starts_with("org--model-")
        );
        assert!(
            sanitize_registry_name("meta-llama/Llama-3.1-70B")
                .expect("name")
                .starts_with("meta-llama--llama-3.1-70b-")
        );
    }

    #[test]
    fn sanitize_distinguishes_slash_from_single_dash() {
        assert_ne!(
            sanitize_registry_name("org/model").expect("name"),
            sanitize_registry_name("org-model").expect("name")
        );
    }

    #[test]
    fn sanitize_distinguishes_case() {
        // Case-folding used to collide silently; the always-on hash suffix (of the
        // original case-preserving name) prevents that.
        assert_ne!(
            sanitize_registry_name("Foo/Bar").expect("name"),
            sanitize_registry_name("foo/bar").expect("name")
        );
    }

    #[test]
    fn sanitize_handles_degenerate_input() {
        let hashed = sanitize_registry_name("").expect("name");
        assert_eq!(hashed.len(), HASH_SUFFIX_LEN);
        let hashed = sanitize_registry_name("///").expect("name");
        assert_eq!(hashed.len(), HASH_SUFFIX_LEN);
        let hashed = sanitize_registry_name("---").expect("name");
        assert_eq!(hashed.len(), HASH_SUFFIX_LEN);
    }

    #[test]
    fn sanitize_fits_dns_1123_budget() {
        let long: String = "a".repeat(300);
        let out = sanitize_registry_name(&long).expect("name");
        assert!(out.len() <= NAME_BUDGET);
        // Still distinguishes two different 300-char names via the hash suffix.
        let other: String = format!("{}b", "a".repeat(299));
        assert_ne!(
            sanitize_registry_name(&long).expect("name"),
            sanitize_registry_name(&other).expect("name")
        );
    }

    #[test]
    fn cr_name_stays_within_k8s_limit() {
        let long = "a".repeat(300);
        let name = KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, &long)
            .expect("name");
        assert!(name.len() <= K8S_NAME_MAX);
        assert!(name.starts_with(CR_NAME_PREFIX));
    }

    #[test]
    fn cr_name_distinguishes_provider_and_legacy() {
        let n = "google-t5/t5-small";
        let hf =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, n).expect("name");
        let ngc = KubernetesRegistryBackend::cr_name_for(ModelProvider::Ngc, n).expect("name");
        let gcs = KubernetesRegistryBackend::cr_name_for(ModelProvider::Gcs, n).expect("name");
        let s3 = KubernetesRegistryBackend::cr_name_for(ModelProvider::S3, n).expect("name");
        let legacy = KubernetesRegistryBackend::legacy_cr_name_for(n).expect("name");
        // Same name, different provider -> distinct CR names, all distinct from legacy.
        assert_ne!(hf, ngc);
        assert_ne!(hf, gcs);
        assert_ne!(ngc, gcs);
        assert_ne!(hf, s3);
        assert_ne!(ngc, s3);
        assert_ne!(gcs, s3);
        assert_ne!(hf, legacy);
        assert_ne!(ngc, legacy);
        assert_ne!(gcs, legacy);
        assert_ne!(s3, legacy);
        for name in [&hf, &ngc, &gcs, &s3, &legacy] {
            assert!(name.starts_with(CR_NAME_PREFIX));
            assert!(name.len() <= K8S_NAME_MAX);
        }
    }

    #[test]
    fn candidate_cr_names_cover_all_providers_and_legacy() {
        let n = "org/model";
        let candidates = KubernetesRegistryBackend::candidate_cr_names(n).expect("names");
        assert_eq!(candidates.len(), ALL_PROVIDERS.len() + 1);
        for p in ALL_PROVIDERS {
            assert!(
                candidates.contains(&KubernetesRegistryBackend::cr_name_for(p, n).expect("name"))
            );
        }
        assert!(
            candidates.contains(&KubernetesRegistryBackend::legacy_cr_name_for(n).expect("name"))
        );
    }

    #[test]
    fn sanitize_trims_leading_trailing_dashes() {
        assert!(
            sanitize_registry_name("-model-")
                .expect("name")
                .starts_with("model-")
        );
        assert!(
            sanitize_registry_name(".model.")
                .expect("name")
                .starts_with("model-")
        );
    }

    /// Variant names of a unit-variant enum, read from its derived `Deserialize` impl.
    ///
    /// The derive hands the deserializer the complete variant list, so a new variant shows up
    /// here without anyone having to extend a hand-written list.
    fn serde_variant_names<'de, T: serde::Deserialize<'de>>() -> &'static [&'static str] {
        struct VariantCapture(Option<&'static [&'static str]>);

        impl<'de> serde::Deserializer<'de> for &mut VariantCapture {
            type Error = serde::de::value::Error;

            fn deserialize_any<V: serde::de::Visitor<'de>>(
                self,
                _visitor: V,
            ) -> Result<V::Value, Self::Error> {
                Err(serde::de::Error::custom("not an enum"))
            }

            fn deserialize_enum<V: serde::de::Visitor<'de>>(
                self,
                _name: &'static str,
                variants: &'static [&'static str],
                _visitor: V,
            ) -> Result<V::Value, Self::Error> {
                self.0 = Some(variants);
                Err(serde::de::Error::custom("variants captured"))
            }

            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
                bytes byte_buf option unit unit_struct newtype_struct seq tuple
                tuple_struct map struct identifier ignored_any
            }
        }

        let mut capture = VariantCapture(None);
        let _ = T::deserialize(&mut capture);
        capture
            .0
            .expect("type should deserialize as an enum and report its variants")
    }

    fn crd_string_enum(pointer: &str) -> BTreeSet<String> {
        let crd = serde_json::to_value(ModelCacheEntry::crd()).expect("CRD should serialize");
        let schema = crd
            .pointer("/spec/versions/0/schema/openAPIV3Schema/properties")
            .expect("CRD should carry an openAPIV3Schema");
        schema
            .pointer(pointer)
            .and_then(|values| values.as_array())
            .unwrap_or_else(|| panic!("CRD schema should list an enum at {pointer}"))
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .expect("CRD enum values should be strings")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn crd_provider_enum_matches_every_model_provider() {
        let variants = serde_variant_names::<ModelProvider>();
        let providers: Vec<ModelProvider> = variants
            .iter()
            .map(|name| {
                serde_json::from_value(json!(name)).expect("variant name should deserialize")
            })
            .collect();

        for provider in &providers {
            let stored = KubernetesRegistryBackend::provider_str(*provider);
            assert_eq!(
                KubernetesRegistryBackend::provider_from_str(stored)
                    .expect("stored provider string should parse back"),
                *provider,
                "provider_from_str does not round-trip {stored:?}"
            );
            assert!(
                ALL_PROVIDERS.contains(provider),
                "ALL_PROVIDERS is missing {provider:?}"
            );
        }
        assert_eq!(ALL_PROVIDERS.len(), providers.len());

        let written: BTreeSet<String> = providers
            .iter()
            .map(|p| KubernetesRegistryBackend::provider_str(*p).to_string())
            .collect();
        assert_eq!(
            written,
            crd_string_enum("/spec/properties/provider/enum"),
            "the CRD's spec.provider enum must accept exactly the provider strings the backend writes"
        );
    }

    #[test]
    fn crd_phase_enum_matches_every_model_status() {
        let mut written = BTreeSet::from([String::new()]);
        for name in serde_variant_names::<ModelStatus>() {
            let status: ModelStatus =
                serde_json::from_value(json!(name)).expect("variant name should deserialize");
            let phase = KubernetesRegistryBackend::phase_from_status(status);
            assert_eq!(
                KubernetesRegistryBackend::status_from_phase(phase),
                status,
                "status_from_phase does not round-trip {phase:?}"
            );
            written.insert(phase.to_string());
        }
        assert_eq!(
            written,
            crd_string_enum("/status/properties/phase/enum"),
            "the CRD's status.phase enum must accept exactly the phases the backend writes, plus the empty initial phase"
        );
    }

    type RecordedRequests = std::sync::Arc<Mutex<Vec<(http::Method, String, serde_json::Value)>>>;

    fn backend_with_fake_responses(
        responses: Vec<(http::StatusCode, serde_json::Value)>,
    ) -> (KubernetesRegistryBackend, RecordedRequests) {
        use std::collections::VecDeque;
        use std::sync::Arc;

        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let service = tower::service_fn({
            let queue = Arc::clone(&queue);
            let recorded = Arc::clone(&recorded);
            move |request: http::Request<kube::client::Body>| {
                let queue = Arc::clone(&queue);
                let recorded = Arc::clone(&recorded);
                async move {
                    let method = request.method().clone();
                    let uri = request.uri().path().to_string();
                    let body_bytes = request
                        .into_body()
                        .collect_bytes()
                        .await
                        .expect("read request body");
                    let body_json = if body_bytes.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::from_slice(&body_bytes).unwrap_or(serde_json::Value::Null)
                    };
                    recorded
                        .lock()
                        .expect("record lock")
                        .push((method, uri, body_json));
                    let (status, response_json) = queue
                        .lock()
                        .expect("queue lock")
                        .pop_front()
                        .expect("expected fake k8s response");
                    let bytes = serde_json::to_vec(&response_json).expect("serialize response");
                    let response = http::Response::builder()
                        .status(status)
                        .header(http::header::CONTENT_TYPE, "application/json")
                        .body(kube::client::Body::from(bytes))
                        .expect("build response");
                    Ok::<_, tower::BoxError>(response)
                }
            }
        });
        let backend = KubernetesRegistryBackend {
            client: Client::new(service, "default"),
            namespace: "default".to_string(),
            lease_observations: Mutex::new(HashMap::new()),
        };
        (backend, recorded)
    }

    fn downloaded_cr_json(
        name: &str,
        model_name: &str,
        deletion_timestamp: Option<&str>,
    ) -> serde_json::Value {
        let mut meta = json!({
            "name": name,
            "namespace": "default",
            "resourceVersion": "42",
        });
        if let Some(ts) = deletion_timestamp {
            meta["deletionTimestamp"] = json!(ts);
        }
        json!({
            "apiVersion": "modelexpress.nvidia.com/v1alpha1",
            "kind": "ModelCacheEntry",
            "metadata": meta,
            "spec": {
                "modelName": model_name,
                "provider": "HuggingFace",
            },
            "status": {
                "phase": phase::DOWNLOADED,
                "createdAt": "2026-01-01T00:00:00Z",
                "lastUsedAt": "2026-01-01T00:00:00Z",
            }
        })
    }

    #[tokio::test]
    async fn try_reclaim_downloaded_patches_status_with_rv_and_phase_guards() {
        let model_name = "org/model";
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, model_name)
                .expect("cr name");
        let cr = downloaded_cr_json(&cr_name, model_name, None);
        let (backend, recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, cr.clone()),
            (http::StatusCode::OK, cr),
        ]);

        let won = backend
            .try_reclaim_downloaded(
                model_name,
                ModelProvider::HuggingFace,
                "claim-reclaim",
                std::time::Duration::from_secs(30),
            )
            .await
            .expect("reclaim should succeed");
        assert!(won);

        let calls = recorded.lock().expect("recorded").clone();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, http::Method::GET);
        assert_eq!(calls[1].0, http::Method::PATCH);
        assert!(calls[1].1.ends_with("/status"));
        let ops = calls[1].2.as_array().expect("JSON Patch array");
        assert!(
            ops.contains(
                &json!({ "op": "test", "path": "/metadata/resourceVersion", "value": "42" })
            ),
            "patch must guard resourceVersion: {ops:?}"
        );
        assert!(
            ops.contains(
                &json!({ "op": "test", "path": "/status/phase", "value": phase::DOWNLOADED })
            ),
            "patch must guard status.phase: {ops:?}"
        );
        assert!(
            ops.contains(
                &json!({ "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADING })
            ),
            "patch must flip phase to Downloading: {ops:?}"
        );
        assert!(
            ops.contains(
                &json!({ "op": "add", "path": "/status/claimId", "value": "claim-reclaim" })
            ),
            "patch must stamp claimId: {ops:?}"
        );
    }

    #[tokio::test]
    async fn try_reclaim_downloaded_skips_terminating_cr_without_patching() {
        let model_name = "org/terminating";
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, model_name)
                .expect("cr name");
        let cr = downloaded_cr_json(&cr_name, model_name, Some("2026-01-01T00:01:00Z"));
        let (backend, recorded) = backend_with_fake_responses(vec![(http::StatusCode::OK, cr)]);

        let won = backend
            .try_reclaim_downloaded(
                model_name,
                ModelProvider::HuggingFace,
                "claim-reclaim",
                std::time::Duration::from_secs(30),
            )
            .await
            .expect("reclaim should return Ok(false)");
        assert!(!won);
        assert_eq!(
            recorded.lock().expect("recorded").len(),
            1,
            "must not PATCH terminating CR"
        );
    }

    #[tokio::test]
    async fn try_reclaim_downloaded_returns_false_on_conflict() {
        let model_name = "org/raced";
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, model_name)
                .expect("cr name");
        let cr = downloaded_cr_json(&cr_name, model_name, None);
        let conflict = json!({
            "kind": "Status",
            "apiVersion": "v1",
            "status": "Failure",
            "message": "test operation failed",
            "reason": "Invalid",
            "code": 422,
        });
        let (backend, _recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, cr),
            (http::StatusCode::UNPROCESSABLE_ENTITY, conflict),
        ]);

        let won = backend
            .try_reclaim_downloaded(
                model_name,
                ModelProvider::HuggingFace,
                "claim-reclaim",
                std::time::Duration::from_secs(30),
            )
            .await
            .expect("422 conflict maps to Ok(false)");
        assert!(!won);
    }

    fn empty_cr_list_json() -> serde_json::Value {
        json!({
            "apiVersion": "modelexpress.nvidia.com/v1alpha1",
            "kind": "ModelCacheEntryList",
            "metadata": { "resourceVersion": "1" },
            "items": [],
        })
    }

    fn ssar_response_json(allowed: bool, reason: Option<&str>) -> serde_json::Value {
        json!({
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectAccessReview",
            "metadata": {},
            "spec": {},
            "status": {
                "allowed": allowed,
                "reason": reason,
            },
        })
    }

    #[tokio::test]
    async fn connect_checks_list_watch_patch_and_status_patch_without_checking_delete() {
        let (backend, recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, empty_cr_list_json()),
            (http::StatusCode::CREATED, ssar_response_json(true, None)),
            (http::StatusCode::CREATED, ssar_response_json(true, None)),
            (http::StatusCode::CREATED, ssar_response_json(true, None)),
        ]);

        backend
            .connect()
            .await
            .expect("connect should succeed when list, watch, patch, and status patch are allowed");

        let calls = recorded.lock().expect("recorded").clone();
        assert_eq!(calls.len(), 4, "expected list + 3 SSAR checks: {calls:?}");
        assert_eq!(calls[0].0, http::Method::GET);
        assert!(calls[0].1.ends_with("/modelcacheentries"));

        let expected_checks = [("watch", None), ("patch", None), ("patch", Some("status"))];
        for (idx, (verb, subresource)) in expected_checks.iter().enumerate() {
            let (method, path, body) = &calls[idx + 1];
            assert_eq!(*method, http::Method::POST);
            assert!(
                path.ends_with("/selfsubjectaccessreviews"),
                "unexpected path: {path}"
            );
            let attrs = body
                .pointer("/spec/resourceAttributes")
                .expect("resourceAttributes in SSAR");
            assert_eq!(attrs["group"], "modelexpress.nvidia.com");
            assert_eq!(attrs["resource"], "modelcacheentries");
            assert_eq!(attrs["namespace"], "default");
            assert_eq!(attrs["verb"], *verb);
            assert_eq!(
                attrs.get("subresource").and_then(|v| v.as_str()),
                *subresource
            );
        }
    }

    #[tokio::test]
    async fn connect_fails_when_watch_permission_is_denied() {
        let (backend, _recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, empty_cr_list_json()),
            (
                http::StatusCode::CREATED,
                ssar_response_json(false, Some("watch denied")),
            ),
        ]);

        let err = backend
            .connect()
            .await
            .expect_err("connect must fail when watch is denied");
        let msg = err.to_string();
        assert!(
            msg.contains("watch") && msg.contains("modelcacheentries"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn connect_fails_when_metadata_patch_permission_is_denied() {
        let (backend, _recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, empty_cr_list_json()),
            (http::StatusCode::CREATED, ssar_response_json(true, None)),
            (
                http::StatusCode::CREATED,
                ssar_response_json(false, Some("patch denied")),
            ),
        ]);

        let err = backend
            .connect()
            .await
            .expect_err("connect must fail when metadata patch is denied");
        let msg = err.to_string();
        assert!(
            msg.contains("patch") && msg.contains("modelcacheentries"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn connect_fails_when_status_patch_permission_is_denied() {
        let (backend, _recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, empty_cr_list_json()),
            (http::StatusCode::CREATED, ssar_response_json(true, None)),
            (http::StatusCode::CREATED, ssar_response_json(true, None)),
            (
                http::StatusCode::CREATED,
                ssar_response_json(false, Some("status patch denied")),
            ),
        ]);

        let err = backend
            .connect()
            .await
            .expect_err("connect must fail when status patch is denied");
        let msg = err.to_string();
        assert!(
            msg.contains("patch") && msg.contains("modelcacheentries/status"),
            "unexpected error: {msg}"
        );
    }

    fn cr_json(
        name: &str,
        model_name: &str,
        provider: &str,
        phase: &str,
        claim_id: Option<&str>,
        deleting: bool,
    ) -> serde_json::Value {
        let mut metadata = json!({
            "name": name,
            "namespace": "default",
            "uid": format!("uid-{name}"),
            "resourceVersion": "42",
            "finalizers": [CACHE_FILES_FINALIZER],
        });
        if deleting {
            metadata["deletionTimestamp"] = json!("2026-01-01T00:00:00Z");
        }
        let mut status = json!({
            "phase": phase,
            "createdAt": "2026-01-01T00:00:00Z",
            "lastUsedAt": "2026-01-01T00:00:00Z",
        });
        if let Some(id) = claim_id {
            status["claimId"] = json!(id);
            status["leaseExpiresAt"] = json!("2026-01-01T00:05:00Z");
        }
        json!({
            "apiVersion": "modelexpress.nvidia.com/v1alpha1",
            "kind": "ModelCacheEntry",
            "metadata": metadata,
            "spec": {
                "modelName": model_name,
                "provider": provider,
            },
            "status": status,
        })
    }

    fn not_found_json() -> serde_json::Value {
        json!({
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "reason": "NotFound",
            "code": 404,
            "message": "not found",
        })
    }

    #[tokio::test]
    async fn try_claim_for_download_attaches_finalizer_at_creation() {
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, "org/model")
                .expect("cr name");
        let created = json!({
            "apiVersion": "modelexpress.nvidia.com/v1alpha1",
            "kind": "ModelCacheEntry",
            "metadata": {
                "name": cr_name,
                "namespace": "default",
                "uid": "uid-created",
                "resourceVersion": "1",
                "finalizers": [CACHE_FILES_FINALIZER],
            },
            "spec": {
                "modelName": "org/model",
                "provider": "HuggingFace",
            },
        });
        let patched = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::DOWNLOADING,
            Some("claim-1"),
            false,
        );
        let (backend, recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::NOT_FOUND, not_found_json()), // scoped GET
            (http::StatusCode::NOT_FOUND, not_found_json()), // legacy GET
            (http::StatusCode::CREATED, created),            // POST create
            (http::StatusCode::OK, patched),                 // PATCH status
        ]);

        let outcome = backend
            .try_claim_for_download(
                "org/model",
                ModelProvider::HuggingFace,
                "claim-1",
                std::time::Duration::from_secs(30),
            )
            .await
            .expect("claim should succeed");
        assert_eq!(outcome, ClaimOutcome::Claimed);

        let calls = recorded.lock().expect("recorded").clone();
        let create_call = calls
            .iter()
            .find(|(method, _, _)| *method == http::Method::POST)
            .expect("POST create call");
        assert_eq!(
            create_call.2.pointer("/metadata/finalizers"),
            Some(&json!([CACHE_FILES_FINALIZER]))
        );
    }

    #[tokio::test]
    async fn terminating_crs_are_excluded_from_reads_lists_counts_and_mutations() {
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, "org/model")
                .expect("cr name");
        let terminating_downloaded = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::DOWNLOADED,
            None,
            true,
        );
        let terminating_downloading = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::DOWNLOADING,
            Some("claim-1"),
            true,
        );
        let terminating_error = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::ERROR,
            None,
            true,
        );
        let live_cr = cr_json(
            "hf-live",
            "org/live",
            "HuggingFace",
            phase::DOWNLOADED,
            None,
            false,
        );
        let list_with_both = json!({
            "apiVersion": "modelexpress.nvidia.com/v1alpha1",
            "kind": "ModelCacheEntryList",
            "metadata": { "resourceVersion": "50" },
            "items": [terminating_downloaded.clone(), live_cr],
        });

        // 1. get_models_by_last_used and get_status_counts skip terminating CRs.
        let (backend, _) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, list_with_both.clone()),
            (http::StatusCode::OK, list_with_both),
        ]);
        let records = backend
            .get_models_by_last_used(None)
            .await
            .expect("list records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model_name, "org/live");
        assert_eq!(
            backend.get_status_counts().await.expect("status counts"),
            (0, 1, 0)
        );

        // 2. get_status, get_model_record, and touch_model ignore terminating CRs.
        let candidate_count = KubernetesRegistryBackend::candidate_cr_names("org/model")
            .expect("candidates")
            .len();
        let mut responses = Vec::new();
        for _ in 0..3 {
            responses.push((http::StatusCode::OK, terminating_downloaded.clone()));
            for _ in 1..candidate_count {
                responses.push((http::StatusCode::NOT_FOUND, not_found_json()));
            }
        }
        let (backend, recorded) = backend_with_fake_responses(responses);
        assert_eq!(
            backend.get_status("org/model").await.expect("get_status"),
            None
        );
        assert!(
            backend
                .get_model_record("org/model")
                .await
                .expect("get_model_record")
                .is_none()
        );
        backend
            .touch_model("org/model")
            .await
            .expect("touch_model should be no-op");
        assert!(
            recorded
                .lock()
                .expect("recorded")
                .iter()
                .all(|(method, _, _)| *method == http::Method::GET),
            "touch_model must not patch a terminating CR"
        );

        // 3. try_claim_for_download treats a terminating CR as AlreadyExists(DOWNLOADING)
        //    even if its phase was Downloaded.
        let (backend, recorded) = backend_with_fake_responses(vec![(
            http::StatusCode::OK,
            terminating_downloaded.clone(),
        )]);
        let claim = backend
            .try_claim_for_download(
                "org/model",
                ModelProvider::HuggingFace,
                "claim-2",
                std::time::Duration::from_secs(30),
            )
            .await
            .expect("claim on terminating CR");
        assert_eq!(claim, ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING));
        assert_eq!(recorded.lock().expect("recorded").len(), 1);

        // 4. set_status rejects terminating CRs; try_reset_error_for_retry,
        //    refresh_download_claim, and finish_download_claim return Ok(false).
        let (backend, recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, terminating_downloaded),
            (http::StatusCode::OK, terminating_error),
            (http::StatusCode::OK, terminating_downloading.clone()),
            (http::StatusCode::OK, terminating_downloading),
        ]);
        assert!(
            backend
                .set_status(
                    "org/model",
                    ModelProvider::HuggingFace,
                    ModelStatus::DOWNLOADED,
                    None
                )
                .await
                .is_err()
        );
        assert!(
            !backend
                .try_reset_error_for_retry(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-2",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("reset error on terminating CR")
        );
        assert!(
            !backend
                .refresh_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("refresh on terminating CR")
        );
        assert!(
            !backend
                .finish_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    ModelStatus::DOWNLOADED,
                    None,
                )
                .await
                .expect("finish on terminating CR")
        );
        assert!(
            recorded
                .lock()
                .expect("recorded")
                .iter()
                .all(|(method, _, _)| *method == http::Method::GET),
            "no PATCH or POST may be issued against terminating CRs"
        );
    }

    fn conflict_status_json(code: u16, reason: &str, message: &str) -> serde_json::Value {
        json!({
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "reason": reason,
            "code": code,
            "message": message,
        })
    }

    #[tokio::test]
    async fn refresh_and_finish_retry_after_metadata_only_conflict() {
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, "org/model")
                .expect("cr name");
        let rv42 = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::DOWNLOADING,
            Some("claim-1"),
            false,
        );
        let mut rv43 = rv42.clone();
        rv43["metadata"]["resourceVersion"] = json!("43");
        rv43["metadata"]["annotations"] = json!({"unrelated": "updated"});

        // Refresh: GET rv=42 -> PATCH 422 (unrelated metadata bump) -> GET rv=43 -> PATCH 200.
        // Finish:  GET rv=42 -> PATCH 409 (unrelated metadata bump) -> GET rv=43 -> PATCH 200.
        let (backend, recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, rv42.clone()),
            (
                http::StatusCode::UNPROCESSABLE_ENTITY,
                conflict_status_json(422, "Invalid", "resourceVersion mismatch"),
            ),
            (http::StatusCode::OK, rv43.clone()),
            (http::StatusCode::OK, rv43.clone()),
            (http::StatusCode::OK, rv42),
            (
                http::StatusCode::CONFLICT,
                conflict_status_json(409, "Conflict", "resourceVersion mismatch"),
            ),
            (http::StatusCode::OK, rv43.clone()),
            (http::StatusCode::OK, rv43),
        ]);

        assert!(
            backend
                .refresh_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("refresh should succeed after metadata-only conflict")
        );
        assert!(
            backend
                .finish_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    ModelStatus::DOWNLOADED,
                    None,
                )
                .await
                .expect("finish should succeed after metadata-only conflict")
        );

        let calls = recorded.lock().expect("recorded").clone();
        let patch_rvs: Vec<_> = calls
            .iter()
            .filter(|(method, _, _)| *method == http::Method::PATCH)
            .map(|(_, _, body)| {
                body.as_array()
                    .expect("patch ops")
                    .iter()
                    .find(|op| op["path"] == "/metadata/resourceVersion")
                    .expect("resourceVersion test op")["value"]
                    .clone()
            })
            .collect();
        assert_eq!(
            patch_rvs,
            vec![json!("42"), json!("43"), json!("42"), json!("43")],
            "both refresh and finish must retry with the refreshed resourceVersion"
        );
    }

    #[tokio::test]
    async fn refresh_and_finish_fail_when_conflict_is_from_delete_or_uid_change() {
        let cr_name =
            KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, "org/model")
                .expect("cr name");
        let live_rv42 = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::DOWNLOADING,
            Some("claim-1"),
            false,
        );
        let mut terminating_rv43 = cr_json(
            &cr_name,
            "org/model",
            "HuggingFace",
            phase::DOWNLOADING,
            Some("claim-1"),
            true,
        );
        terminating_rv43["metadata"]["resourceVersion"] = json!("43");

        let mut recreated_uid = live_rv42.clone();
        recreated_uid["metadata"]["uid"] = json!("uid-recreated");
        recreated_uid["metadata"]["resourceVersion"] = json!("43");

        // 1. Delete conflict on refresh: GET live -> PATCH 422 -> GET terminating -> Ok(false).
        // 2. Delete conflict on finish:  GET live -> PATCH 422 -> GET terminating -> Ok(false).
        // 3. UID change on refresh:      GET uid-1 -> PATCH 422 -> GET uid-recreated -> Ok(false).
        let (backend, recorded) = backend_with_fake_responses(vec![
            (http::StatusCode::OK, live_rv42.clone()),
            (
                http::StatusCode::UNPROCESSABLE_ENTITY,
                conflict_status_json(422, "Invalid", "resourceVersion mismatch"),
            ),
            (http::StatusCode::OK, terminating_rv43.clone()),
            (http::StatusCode::OK, live_rv42.clone()),
            (
                http::StatusCode::UNPROCESSABLE_ENTITY,
                conflict_status_json(422, "Invalid", "resourceVersion mismatch"),
            ),
            (http::StatusCode::OK, terminating_rv43),
            (http::StatusCode::OK, live_rv42),
            (
                http::StatusCode::UNPROCESSABLE_ENTITY,
                conflict_status_json(422, "Invalid", "resourceVersion mismatch"),
            ),
            (http::StatusCode::OK, recreated_uid),
        ]);

        assert!(
            !backend
                .refresh_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("refresh after delete conflict")
        );
        assert!(
            !backend
                .finish_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    ModelStatus::DOWNLOADED,
                    None,
                )
                .await
                .expect("finish after delete conflict")
        );
        assert!(
            !backend
                .refresh_download_claim(
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("refresh after UID change conflict")
        );

        let patch_count = recorded
            .lock()
            .expect("recorded")
            .iter()
            .filter(|(method, _, _)| *method == http::Method::PATCH)
            .count();
        assert_eq!(
            patch_count, 3,
            "no retry PATCH may be sent once re-read shows deletion or UID change"
        );
    }
}
