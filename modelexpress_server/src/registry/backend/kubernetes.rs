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
//!
//! Every CR carries [`CACHE_FILES_FINALIZER`], so a delete from any client only marks it
//! for deletion until [`finalizer`] has deleted the cached files it covers.

#[cfg(test)]
mod fake_api;
pub mod finalizer;

use super::{ClaimOutcome, ModelRecord, RECLAIM_MESSAGE, RegistryBackend, RegistryResult};
use crate::registry::k8s_types::{
    CACHE_FILES_FINALIZER, ModelCacheEntry, ModelCacheEntrySpec, phase,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
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
/// How many times a download completion reads the CR again after a write that changed
/// its resourceVersion but not its claim.
const FINISH_CLAIM_ATTEMPTS: usize = 3;

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

    /// A new CR that already carries the finalizer. Adding it after create would bump
    /// the resourceVersion that the creator's claim patch is fenced on.
    fn new_cr(cr_name: &str, model_name: &str, provider: &str) -> ModelCacheEntry {
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

    fn is_terminating(cr: &ModelCacheEntry) -> bool {
        cr.metadata.deletion_timestamp.is_some()
    }

    /// Only a watching server removes the finalizer, so without `watch` every delete
    /// would hang.
    async fn ensure_can_watch(&self) -> RegistryResult<()> {
        use k8s_openapi::api::authorization::v1::{
            ResourceAttributes, SelfSubjectAccessReview, SelfSubjectAccessReviewSpec,
        };
        use kube::Resource;

        let review = SelfSubjectAccessReview {
            spec: SelfSubjectAccessReviewSpec {
                resource_attributes: Some(ResourceAttributes {
                    group: Some(ModelCacheEntry::group(&()).into_owned()),
                    resource: Some(ModelCacheEntry::plural(&()).into_owned()),
                    verb: Some("watch".to_string()),
                    namespace: Some(self.namespace.clone()),
                    ..ResourceAttributes::default()
                }),
                ..SelfSubjectAccessReviewSpec::default()
            },
            ..SelfSubjectAccessReview::default()
        };
        let reviewed = Api::<SelfSubjectAccessReview>::all(self.client.clone())
            .create(&PostParams::default(), &review)
            .await?;
        if reviewed.status.is_some_and(|status| status.allowed) {
            Ok(())
        } else {
            Err(format!(
                "the server cannot watch modelcacheentries in namespace '{}': grant the \
                 `watch` verb, which deleting cached files on ModelCacheEntry delete requires",
                self.namespace
            )
            .into())
        }
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

    /// First existing CR among the candidates (provider-scoped keys, then legacy). Used by
    /// name-addressed reads; in practice a name maps to a single provider. A marked CR
    /// is a record that is already deleted, so reads do not see it.
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

    /// Every CR that is not marked for deletion. Callers count and evict records, and
    /// a marked CR only waits for its files to go.
    async fn list_live(&self) -> RegistryResult<Vec<ModelCacheEntry>> {
        let crs = self.api().list(&ListParams::default()).await?;
        Ok(crs
            .items
            .into_iter()
            .filter(|cr| !Self::is_terminating(cr))
            .collect())
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
            // Its files are being deleted; copying it would keep a record of them.
            return Ok(Some(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)));
        }

        let legacy_status = legacy_cr.status.clone().unwrap_or_default();
        let new_cr = Self::new_cr(
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

        let phase = self
            .get_cr(scoped_name)
            .await?
            .and_then(|cr| cr.status)
            .unwrap_or_default()
            .phase;
        Ok(Some(ClaimOutcome::AlreadyExists(Self::status_from_phase(
            &phase,
        ))))
    }

    fn provider_str(p: ModelProvider) -> &'static str {
        match p {
            ModelProvider::HuggingFace => "HuggingFace",
            ModelProvider::Ngc => "Ngc",
            ModelProvider::Gcs => "Gcs",
            ModelProvider::S3 => "S3",
        }
    }

    fn provider_from_str(s: &str) -> RegistryResult<ModelProvider> {
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

    async fn try_claim_existing(
        &self,
        cr: ModelCacheEntry,
        claim_id: &str,
        lease_duration: std::time::Duration,
        claim_uninitialized: bool,
    ) -> RegistryResult<ClaimOutcome> {
        if Self::is_terminating(&cr) {
            // Neither report files that are about to be deleted nor start a download into
            // them. Waiters poll until the CR is gone, then claim a fresh one.
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
                    .and_then(|entry| entry.status)
                    .map_or(ModelStatus::DOWNLOADING, |status| {
                        Self::status_from_phase(&status.phase)
                    });
                Ok(ClaimOutcome::AlreadyExists(current))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Flip `from_phase` to DOWNLOADING and make `claim_id` the lease owner. Only the
    /// caller that wins the CAS starts the download.
    async fn claim_from_phase(
        &self,
        model_name: &str,
        provider: ModelProvider,
        from_phase: &str,
        message: &str,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        // Callers run after a claim observed AlreadyExists (which already migrated any
        // legacy CR), so the CAS targets the provider-scoped CR directly.
        let cr_name = Self::cr_name_for(provider, model_name)?;
        let Some(existing) = self.get_cr(&cr_name).await? else {
            return Ok(false);
        };
        if Self::is_terminating(&existing) {
            return Ok(false);
        }
        if existing.status.as_ref().map(|s| s.phase.as_str()) != Some(from_phase) {
            return Ok(false);
        }
        let resource_version = existing
            .metadata
            .resource_version
            .as_deref()
            .ok_or("ModelCacheEntry has no resourceVersion")?;
        // The resourceVersion test rejects the patch with 409 or 422 when anything
        // changed the CR after the GET: a concurrent claim, or a delete that marked it.
        let now = Utc::now().to_rfc3339();
        let lease_expires_at = Self::lease_deadline(lease_duration).to_rfc3339();
        let patch = json!([
            { "op": "test", "path": "/metadata/resourceVersion", "value": resource_version },
            { "op": "test", "path": "/status/phase", "value": from_phase },
            { "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADING },
            { "op": "add", "path": "/status/message", "value": message },
            { "op": "add", "path": "/status/lastUsedAt", "value": now },
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
            Err(kube::Error::Api(e)) if e.code == 422 || e.code == 409 => {
                debug!("Claim of {cr_name} from {from_phase} lost to a concurrent write");
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }
}

#[async_trait]
impl RegistryBackend for KubernetesRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        // Exercise the ModelCacheEntry API to surface missing CRDs or RBAC errors early.
        let _ = self.api().list(&ListParams::default().limit(1)).await?;
        self.ensure_can_watch().await?;
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
        let is_new = existing.is_none();
        let needs_created_at = existing
            .as_ref()
            .and_then(|cr| cr.status.as_ref())
            .and_then(|s| s.created_at.as_deref())
            .is_none();

        if is_new {
            let cr = Self::new_cr(&cr_name, model_name, Self::provider_str(provider));
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
        let crs = self.list_live().await?;
        let mut records: Vec<ModelRecord> = Vec::with_capacity(crs.len());
        for cr in &crs {
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
        let crs = self.list_live().await?;
        let mut downloading = 0u32;
        let mut downloaded = 0u32;
        let mut error = 0u32;
        for cr in &crs {
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

        let cr = Self::new_cr(&cr_name, model_name, Self::provider_str(provider));

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
        self.claim_from_phase(
            model_name,
            provider,
            phase::ERROR,
            "Retrying download...",
            claim_id,
            lease_duration,
        )
        .await
    }

    async fn try_reclaim_downloaded(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        self.claim_from_phase(
            model_name,
            provider,
            phase::DOWNLOADED,
            RECLAIM_MESSAGE,
            claim_id,
            lease_duration,
        )
        .await
    }

    async fn refresh_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let cr_name = Self::cr_name_for(provider, model_name)?;
        let now = Utc::now().to_rfc3339();
        let lease_expires_at = Self::lease_deadline(lease_duration).to_rfc3339();
        let patch = json!([
            { "op": "test", "path": "/status/phase", "value": phase::DOWNLOADING },
            { "op": "test", "path": "/status/claimId", "value": claim_id },
            { "op": "add", "path": "/status/leaseExpiresAt", "value": lease_expires_at },
            { "op": "add", "path": "/status/lastUsedAt", "value": now },
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
            Err(kube::Error::Api(e)) if e.code == 404 || e.code == 409 || e.code == 422 => {
                warn!(
                    "Download lease refresh for {cr_name} lost ownership: {}: {}",
                    e.reason, e.message
                );
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
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
        for _ in 0..FINISH_CLAIM_ATTEMPTS {
            let Some(existing) = self.get_cr(&cr_name).await? else {
                return Ok(false);
            };
            let owns_claim = existing.status.as_ref().is_some_and(|current| {
                current.phase == phase::DOWNLOADING && current.claim_id.as_deref() == Some(claim_id)
            });
            // A delete that marked the CR removes the files of this download next, so
            // the download must not be published as complete.
            if !owns_claim || Self::is_terminating(&existing) {
                debug!("Download completion for {cr_name} lost ownership");
                return Ok(false);
            }
            let resource_version = existing
                .metadata
                .resource_version
                .as_deref()
                .ok_or("ModelCacheEntry has no resourceVersion")?;
            let now = Utc::now().to_rfc3339();
            // The resourceVersion test covers a delete that marks the CR after the GET.
            let patch = json!([
                { "op": "test", "path": "/metadata/resourceVersion", "value": resource_version },
                { "op": "test", "path": "/status/phase", "value": phase::DOWNLOADING },
                { "op": "test", "path": "/status/claimId", "value": claim_id },
                { "op": "replace", "path": "/status/phase", "value": Self::phase_from_status(status) },
                { "op": "add", "path": "/status/message", "value": message },
                { "op": "add", "path": "/status/lastUsedAt", "value": now },
                { "op": "add", "path": "/status/claimId", "value": null },
                { "op": "add", "path": "/status/leaseExpiresAt", "value": null },
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
                    return Ok(true);
                }
                Err(kube::Error::Api(e)) if e.code == 404 => return Ok(false),
                // Something wrote the CR after the GET. A write that is not a delete
                // or a takeover leaves the claim with this caller, so read again.
                Err(kube::Error::Api(e)) if e.code == 409 || e.code == 422 => {}
                Err(e) => return Err(e.into()),
            }
        }
        Err(format!(
            "ModelCacheEntry {cr_name} changed during each of {FINISH_CLAIM_ATTEMPTS} attempts \
             to finish its download"
        )
        .into())
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

    // --- marked CRs -------------------------------------------------------------
    //
    // A delete only marks a CR until the finalizer has deleted its files. These tests
    // run against a fake API server, so they need no cluster.

    use super::fake_api::{self, Calls};
    use http::{Method, StatusCode};

    const LEASE: std::time::Duration = std::time::Duration::from_secs(30);
    const MODEL: &str = "org/model";

    fn backend_on(client: Client) -> KubernetesRegistryBackend {
        KubernetesRegistryBackend {
            client,
            namespace: fake_api::NAMESPACE.to_string(),
            lease_observations: Mutex::new(HashMap::new()),
        }
    }

    fn scoped_name() -> String {
        KubernetesRegistryBackend::cr_name_for(ModelProvider::HuggingFace, MODEL).expect("CR name")
    }

    fn hf_entry(name: &str, phase: &str) -> ModelCacheEntry {
        fake_api::entry(
            name,
            MODEL,
            "HuggingFace",
            Some(fake_api::status(phase, None, None)),
        )
    }

    /// A backend whose every GET by name answers with `cr`.
    fn backend_serving(cr: ModelCacheEntry) -> (KubernetesRegistryBackend, Calls) {
        let (client, calls) = fake_api::client(move |_| fake_api::ok(&cr));
        (backend_on(client), calls)
    }

    fn writes(calls: &Calls) -> Vec<fake_api::Call> {
        calls
            .all()
            .into_iter()
            .filter(|call| call.method != Method::GET)
            .collect()
    }

    /// A claim on a marked CR must neither report its files, which are about to go,
    /// nor take over the download. The caller waits until the CR is gone.
    #[tokio::test]
    async fn a_marked_cr_answers_every_claim_with_downloading() {
        for phase in [phase::DOWNLOADED, phase::ERROR, phase::DOWNLOADING] {
            let marked = fake_api::marked(hf_entry(&scoped_name(), phase));
            let (backend, calls) = backend_serving(marked);
            let outcome = backend
                .try_claim_for_download(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
                .await
                .expect("claim");
            assert_eq!(
                outcome,
                ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING),
                "phase {phase}"
            );
            assert!(writes(&calls).is_empty(), "{:?}", calls.all());
        }
    }

    /// A zero lease makes every unowned download claimable at once. The mark must
    /// still win.
    #[tokio::test]
    async fn a_marked_cr_is_not_taken_over_when_its_lease_ends() {
        let marked = fake_api::marked(hf_entry(&scoped_name(), phase::DOWNLOADING));
        let (backend, calls) = backend_serving(marked.clone());
        let outcome = backend
            .try_claim_existing(marked, "claim", std::time::Duration::ZERO, false)
            .await
            .expect("claim");
        assert_eq!(
            outcome,
            ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)
        );
        assert!(calls.all().is_empty(), "{:?}", calls.all());
    }

    /// Migrating a marked legacy CR would copy a record whose files are being deleted.
    #[tokio::test]
    async fn a_marked_legacy_cr_is_not_migrated() {
        let legacy_name = KubernetesRegistryBackend::legacy_cr_name_for(MODEL).expect("CR name");
        let legacy = fake_api::marked(hf_entry(&legacy_name, phase::DOWNLOADED));
        let (client, calls) = fake_api::client(move |call| {
            if call.is(Method::GET, &legacy_name) {
                fake_api::ok(&legacy)
            } else {
                fake_api::failure(StatusCode::NOT_FOUND)
            }
        });

        let outcome = backend_on(client)
            .try_claim_for_download(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
            .await
            .expect("claim");
        assert_eq!(
            outcome,
            ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)
        );
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());
    }

    #[tokio::test]
    async fn a_marked_cr_refuses_the_retry_and_the_reclaim() {
        let marked_error = fake_api::marked(hf_entry(&scoped_name(), phase::ERROR));
        let (backend, calls) = backend_serving(marked_error);
        assert!(
            !backend
                .try_reset_error_for_retry(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
                .await
                .expect("retry")
        );
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());

        let marked_downloaded = fake_api::marked(hf_entry(&scoped_name(), phase::DOWNLOADED));
        let (backend, calls) = backend_serving(marked_downloaded);
        assert!(
            !backend
                .try_reclaim_downloaded(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
                .await
                .expect("reclaim")
        );
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());
    }

    /// A delete can mark the CR between the read and the patch. The resourceVersion
    /// test makes the API server refuse the patch in that case.
    #[tokio::test]
    async fn the_reclaim_patch_is_fenced_on_the_version_it_read() {
        let (backend, calls) = backend_serving(hf_entry(&scoped_name(), phase::DOWNLOADED));
        assert!(
            backend
                .try_reclaim_downloaded(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
                .await
                .expect("reclaim")
        );
        let patches = calls.matching(Method::PATCH, "/status");
        assert_eq!(patches.len(), 1, "{:?}", calls.all());
        let ops = patches[0].body.as_array().expect("JSON patch");
        assert_eq!(
            ops[..3],
            [
                json!({ "op": "test", "path": "/metadata/resourceVersion", "value": "7" }),
                json!({ "op": "test", "path": "/status/phase", "value": phase::DOWNLOADED }),
                json!({ "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADING }),
            ]
        );
        assert!(ops.contains(&json!({ "op": "add", "path": "/status/claimId", "value": "claim" })));

        // The record is in another phase: nothing is written.
        let (backend, calls) = backend_serving(hf_entry(&scoped_name(), phase::ERROR));
        assert!(
            !backend
                .try_reclaim_downloaded(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
                .await
                .expect("reclaim")
        );
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());

        // The API server refuses the patch: the claim is lost, not failed.
        for code in [StatusCode::CONFLICT, StatusCode::UNPROCESSABLE_ENTITY] {
            let cr = hf_entry(&scoped_name(), phase::ERROR);
            let (client, _calls) = fake_api::client(move |call| {
                if call.method == Method::GET {
                    fake_api::ok(&cr)
                } else {
                    fake_api::failure(code)
                }
            });
            assert!(
                !backend_on(client)
                    .try_reset_error_for_retry(MODEL, ModelProvider::HuggingFace, "claim", LEASE)
                    .await
                    .expect("retry")
            );
        }
    }

    fn downloading_entry(claim_id: &str) -> ModelCacheEntry {
        fake_api::entry(
            &scoped_name(),
            MODEL,
            "HuggingFace",
            Some(fake_api::status(phase::DOWNLOADING, Some(claim_id), None)),
        )
    }

    async fn finish(backend: &KubernetesRegistryBackend, claim_id: &str) -> RegistryResult<bool> {
        backend
            .finish_download_claim(
                MODEL,
                ModelProvider::HuggingFace,
                claim_id,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
    }

    /// A download can end after a delete marked its CR and before the finalizer
    /// revokes the claim. The files are about to go, so the completion must not
    /// publish DOWNLOADED.
    #[tokio::test]
    async fn a_marked_cr_refuses_the_completion() {
        let (backend, calls) = backend_serving(fake_api::marked(downloading_entry("claim")));
        assert!(!finish(&backend, "claim").await.expect("finish"));
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());

        // Another owner, or no CR: also no write.
        let (backend, calls) = backend_serving(downloading_entry("other"));
        assert!(!finish(&backend, "claim").await.expect("finish"));
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());

        let (client, calls) = fake_api::client(|_| fake_api::failure(StatusCode::NOT_FOUND));
        assert!(!finish(&backend_on(client), "claim").await.expect("finish"));
        assert!(writes(&calls).is_empty(), "{:?}", calls.all());
    }

    /// A delete can mark the CR between the read and the patch. The resourceVersion
    /// test makes the API server refuse the patch, and the next read sees the mark.
    #[tokio::test]
    async fn the_completion_patch_is_fenced_on_the_version_it_read() {
        let (backend, calls) = backend_serving(downloading_entry("claim"));
        assert!(finish(&backend, "claim").await.expect("finish"));
        let patches = calls.matching(Method::PATCH, "/status");
        assert_eq!(patches.len(), 1, "{:?}", calls.all());
        let ops = patches[0].body.as_array().expect("JSON patch");
        assert_eq!(
            ops[0],
            json!({ "op": "test", "path": "/metadata/resourceVersion", "value": "7" })
        );
        assert!(ops.contains(
            &json!({ "op": "replace", "path": "/status/phase", "value": phase::DOWNLOADED })
        ));

        for code in [StatusCode::CONFLICT, StatusCode::UNPROCESSABLE_ENTITY] {
            let refused = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (client, calls) = fake_api::client({
                let refused = refused.clone();
                move |call| {
                    if call.method != Method::GET {
                        refused.store(true, std::sync::atomic::Ordering::SeqCst);
                        fake_api::failure(code)
                    } else if refused.load(std::sync::atomic::Ordering::SeqCst) {
                        fake_api::ok(&fake_api::marked(downloading_entry("claim")))
                    } else {
                        fake_api::ok(&downloading_entry("claim"))
                    }
                }
            });
            assert!(!finish(&backend_on(client), "claim").await.expect("finish"));
            assert_eq!(writes(&calls).len(), 1, "{:?}", calls.all());
        }
    }

    /// A write that changes the version but not the claim, such as a `lastUsedAt`
    /// update, must not cost the owner its completed download.
    #[tokio::test]
    async fn a_completion_that_loses_the_version_tries_again() {
        let patches = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (client, calls) = fake_api::client({
            let patches = patches.clone();
            move |call| {
                if call.method == Method::GET {
                    fake_api::ok(&downloading_entry("claim"))
                } else if patches.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    fake_api::failure(StatusCode::CONFLICT)
                } else {
                    fake_api::ok(&downloading_entry("claim"))
                }
            }
        });
        assert!(finish(&backend_on(client), "claim").await.expect("finish"));
        assert_eq!(writes(&calls).len(), 2, "{:?}", calls.all());

        // A CR that changes on each attempt is an error, not a lost claim.
        let (client, calls) = fake_api::client(|call| {
            if call.method == Method::GET {
                fake_api::ok(&downloading_entry("claim"))
            } else {
                fake_api::failure(StatusCode::CONFLICT)
            }
        });
        assert!(finish(&backend_on(client), "claim").await.is_err());
        assert_eq!(
            writes(&calls).len(),
            FINISH_CLAIM_ATTEMPTS,
            "{:?}",
            calls.all()
        );
    }

    /// A marked CR is a record that is already deleted. Counting or listing it would
    /// evict one live model too many and book a deleted download twice.
    #[tokio::test]
    async fn registry_reads_do_not_see_marked_crs() {
        let live = hf_entry("live", phase::DOWNLOADED);
        let marked = fake_api::marked(fake_api::entry(
            "marked",
            "org/deleted",
            "HuggingFace",
            Some(fake_api::status(phase::DOWNLOADING, Some("owner"), None)),
        ));
        let (client, _calls) = fake_api::client({
            let items = vec![live, marked.clone()];
            move |call| {
                if call.path.ends_with("/modelcacheentries") {
                    fake_api::list(&items)
                } else {
                    fake_api::ok(&marked)
                }
            }
        });
        let backend = backend_on(client);

        let records = backend.get_models_by_last_used(None).await.expect("list");
        let names: Vec<&str> = records.iter().map(|r| r.model_name.as_str()).collect();
        assert_eq!(names, [MODEL]);
        assert_eq!(
            backend.get_status_counts().await.expect("counts"),
            (0, 1, 0)
        );
        assert_eq!(
            backend.get_status("org/deleted").await.expect("status"),
            None
        );
        assert_eq!(
            backend
                .get_model_record("org/deleted")
                .await
                .expect("record"),
            None
        );
    }

    fn access_review(allowed: bool) -> serde_json::Value {
        json!({
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectAccessReview",
            "spec": {},
            "status": { "allowed": allowed },
        })
    }

    /// Without `watch` no server removes the finalizer, and every delete would hang.
    #[tokio::test]
    async fn connect_fails_when_the_server_cannot_watch() {
        for allowed in [true, false] {
            let (client, calls) = fake_api::client(move |call| {
                if call.method == Method::POST {
                    (StatusCode::CREATED, access_review(allowed))
                } else {
                    fake_api::list(&[])
                }
            });
            let result = backend_on(client).connect().await;
            assert_eq!(result.is_ok(), allowed, "{result:?}");
            if let Err(e) = result {
                assert!(e.to_string().contains("`watch`"), "{e}");
            }

            let reviews = calls.matching(Method::POST, "/selfsubjectaccessreviews");
            assert_eq!(reviews.len(), 1, "{:?}", calls.all());
            assert_eq!(
                reviews[0].body["spec"]["resourceAttributes"],
                json!({
                    "group": "modelexpress.nvidia.com",
                    "resource": "modelcacheentries",
                    "verb": "watch",
                    "namespace": fake_api::NAMESPACE,
                })
            );
        }
    }
}
