// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes finalizer controller for `ModelCacheEntry` custom resources.
//!
//! Attaches [`CACHE_FILES_FINALIZER`] to `ModelCacheEntry` objects (catching
//! pre-upgrade or manually created CRs) and reclaims cached model files when a
//! CR receives a `deletionTimestamp`.
//!
//! All metadata and status mutations use RFC 6902 JSON Patch preconditions on
//! `/metadata/uid`, `/metadata/resourceVersion`, and (for finalizer removal)
//! the exact finalizer array index. This prevents a stale patch from touching
//! a recreated CR with the same name and preserves any foreign finalizers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use kube::api::{ListParams, Patch, PatchParams};
use kube::runtime::controller::Action;
use kube::runtime::{Controller, watcher};
use kube::{Api, Client};
use serde_json::json;
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::registry::backend::ModelRecord;
use crate::registry::backend::kubernetes::{CACHE_FILES_FINALIZER, KubernetesRegistryBackend};
use crate::registry::cache_cleanup::{CleanupPlan, execute_cleanup_plan, plan_cleanup_for_entry};
use crate::registry::entry_key::EntryKey;
use crate::registry::k8s_types::{ModelCacheEntry, phase};
use crate::services::DOWNLOAD_LEASE_DURATION;
use modelexpress_common::download::canonical_model_name;
use modelexpress_common::providers::lock_file::LockFile;

/// Maximum JSON Patch retries when `resourceVersion` changes concurrently.
const MAX_PATCH_RETRIES: usize = 5;

/// Short requeue delay when a status patch loses an optimistic concurrency race.
const REQUEUE_CONFLICT_DELAY: Duration = Duration::from_secs(1);

/// Short requeue delay when the per-model advisory file lock is currently busy.
const REQUEUE_LOCK_BUSY_DELAY: Duration = Duration::from_secs(1);

/// Backoff delay when reconciliation returns an error (for example, disk cleanup failure).
const REQUEUE_ERROR_DELAY: Duration = Duration::from_secs(5);

/// Errors returned by a single finalizer reconciliation pass.
#[derive(Debug, thiserror::Error)]
pub enum FinalizerError {
    #[error("Kubernetes API error: {0}")]
    Kube(#[from] kube::Error),

    #[error("Failed to serialize JSON patch: {0}")]
    JsonPatch(String),

    #[error("ModelCacheEntry '{name}' is missing required metadata field '{field}'")]
    MissingMetadata { name: String, field: &'static str },

    #[error("Optimistic concurrency retries exhausted for ModelCacheEntry '{name}': {reason}")]
    PatchConflict { name: String, reason: String },

    #[error("Failed to remove cached files for ModelCacheEntry '{name}': {source}")]
    Cleanup {
        name: String,
        #[source]
        source: anyhow::Error,
    },
}

/// Shared state for [`reconcile_entry`].
pub struct FinalizerControllerContext {
    client: Client,
    namespace: String,
    cache_directory: PathBuf,
    writer_wait_duration: Duration,
    /// Monotonic start instant of the writer wait window for terminating
    /// `DOWNLOADING` (or uninitialized) CRs, keyed strictly by CR `metadata.uid`.
    /// Keying by UID guarantees that a recreated CR with the same `metadata.name`
    /// never inherits an already-elapsed wait from a previous CR instance.
    writer_waits: Mutex<HashMap<String, Instant>>,
}

impl FinalizerControllerContext {
    pub fn new(client: Client, namespace: String, cache_directory: PathBuf) -> Self {
        Self::with_writer_wait(client, namespace, cache_directory, DOWNLOAD_LEASE_DURATION)
    }

    pub(crate) fn with_writer_wait(
        client: Client,
        namespace: String,
        cache_directory: PathBuf,
        writer_wait_duration: Duration,
    ) -> Self {
        Self {
            client,
            namespace,
            cache_directory,
            writer_wait_duration,
            writer_waits: Mutex::new(HashMap::new()),
        }
    }

    fn api(&self) -> Api<ModelCacheEntry> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    /// Record the start of the writer wait for `uid` (if not already started)
    /// and return `Some(remaining)` while the wait window is still open.
    fn check_or_start_writer_wait(&self, uid: &str) -> Option<Duration> {
        if self.writer_wait_duration.is_zero() {
            return None;
        }
        let now = Instant::now();
        let mut waits = self
            .writer_waits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let started = *waits.entry(uid.to_string()).or_insert(now);
        let elapsed = now.saturating_duration_since(started);
        if elapsed < self.writer_wait_duration {
            Some(self.writer_wait_duration.saturating_sub(elapsed))
        } else {
            None
        }
    }

    fn clear_writer_wait(&self, uid: &str) {
        let mut waits = self
            .writer_waits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        waits.remove(uid);
    }
}

fn has_cache_files_finalizer(cr: &ModelCacheEntry) -> bool {
    cache_files_finalizer_index(cr).is_some()
}

fn cache_files_finalizer_index(cr: &ModelCacheEntry) -> Option<usize> {
    cr.metadata
        .finalizers
        .as_ref()
        .and_then(|finalizers| finalizers.iter().position(|f| f == CACHE_FILES_FINALIZER))
}

/// Returns `true` when the CR might have an active writer whose self-fence
/// window must elapse before cached files are deleted.
fn needs_writer_wait(cr: &ModelCacheEntry) -> bool {
    match cr.status.as_ref() {
        None => true,
        Some(status) => status.phase.is_empty() || status.phase == phase::DOWNLOADING,
    }
}

fn has_active_claim(cr: &ModelCacheEntry) -> bool {
    cr.status
        .as_ref()
        .is_some_and(|status| status.claim_id.is_some() || status.lease_expires_at.is_some())
}

fn is_not_found(err: &kube::Error) -> bool {
    matches!(err, kube::Error::Api(api) if api.code == 404)
}

fn is_conflict_or_precondition(err: &kube::Error) -> bool {
    matches!(err, kube::Error::Api(api) if api.code == 409 || api.code == 422)
}

fn parse_json_patch(value: serde_json::Value) -> Result<Patch<()>, FinalizerError> {
    let patch =
        serde_json::from_value(value).map_err(|e| FinalizerError::JsonPatch(e.to_string()))?;
    Ok(Patch::Json(patch))
}

/// Append [`CACHE_FILES_FINALIZER`] to a live CR using UID + resourceVersion
/// preconditions while preserving all foreign finalizers.
async fn attach_finalizer(
    api: &Api<ModelCacheEntry>,
    name: &str,
    uid: &str,
    initial: &ModelCacheEntry,
) -> Result<(), FinalizerError> {
    let mut current = initial.clone();

    for attempt in 0..MAX_PATCH_RETRIES {
        if current.metadata.uid.as_deref() != Some(uid) {
            debug!("Aborting finalizer attach for '{name}': UID changed");
            return Ok(());
        }
        if KubernetesRegistryBackend::is_terminating(&current)
            || has_cache_files_finalizer(&current)
        {
            return Ok(());
        }

        let rv = current
            .metadata
            .resource_version
            .as_deref()
            .ok_or_else(|| FinalizerError::MissingMetadata {
                name: name.to_string(),
                field: "resourceVersion",
            })?;

        let add_op = if current.metadata.finalizers.is_some() {
            json!({
                "op": "add",
                "path": "/metadata/finalizers/-",
                "value": CACHE_FILES_FINALIZER,
            })
        } else {
            json!({
                "op": "add",
                "path": "/metadata/finalizers",
                "value": [CACHE_FILES_FINALIZER],
            })
        };

        let patch = parse_json_patch(json!([
            { "op": "test", "path": "/metadata/uid", "value": uid },
            { "op": "test", "path": "/metadata/resourceVersion", "value": rv },
            add_op,
        ]))?;

        match api.patch(name, &PatchParams::default(), &patch).await {
            Ok(_) => {
                debug!("Attached finalizer '{CACHE_FILES_FINALIZER}' to ModelCacheEntry '{name}'");
                return Ok(());
            }
            Err(err) if is_not_found(&err) => return Ok(()),
            Err(err) if is_conflict_or_precondition(&err) => {
                debug!(
                    "Finalizer attach for '{name}' hit conflict/precondition (attempt {}); re-fetching",
                    attempt.saturating_add(1)
                );
                match api.get(name).await {
                    Ok(refreshed) => current = refreshed,
                    Err(get_err) if is_not_found(&get_err) => return Ok(()),
                    Err(get_err) => return Err(get_err.into()),
                }
            }
            Err(err) => return Err(err.into()),
        }
    }

    Err(FinalizerError::PatchConflict {
        name: name.to_string(),
        reason: "attaching finalizer".to_string(),
    })
}

/// Outcome of attempting to revoke an active download claim on a terminating CR.
enum RevokeOutcome {
    Revoked,
    NotFound,
    Conflict,
}

/// Clear `status.claimId` and `status.leaseExpiresAt` on a terminating CR using
/// UID + resourceVersion preconditions so an active writer loses lease
/// ownership immediately.
async fn revoke_download_claim(
    api: &Api<ModelCacheEntry>,
    name: &str,
    uid: &str,
    cr: &ModelCacheEntry,
) -> Result<RevokeOutcome, FinalizerError> {
    let rv =
        cr.metadata
            .resource_version
            .as_deref()
            .ok_or_else(|| FinalizerError::MissingMetadata {
                name: name.to_string(),
                field: "resourceVersion",
            })?;

    let patch = parse_json_patch(json!([
        { "op": "test", "path": "/metadata/uid", "value": uid },
        { "op": "test", "path": "/metadata/resourceVersion", "value": rv },
        { "op": "add", "path": "/status/claimId", "value": null },
        { "op": "add", "path": "/status/leaseExpiresAt", "value": null },
    ]))?;

    match api
        .patch_status(name, &PatchParams::default(), &patch)
        .await
    {
        Ok(_) => {
            debug!("Revoked download claim on terminating ModelCacheEntry '{name}' (uid={uid})");
            Ok(RevokeOutcome::Revoked)
        }
        Err(err) if is_not_found(&err) => Ok(RevokeOutcome::NotFound),
        Err(err) if is_conflict_or_precondition(&err) => {
            debug!(
                "Claim revocation on terminating ModelCacheEntry '{name}' hit conflict/precondition; requeueing"
            );
            Ok(RevokeOutcome::Conflict)
        }
        Err(err) => Err(err.into()),
    }
}

/// Remove only [`CACHE_FILES_FINALIZER`] from `name` while asserting UID,
/// resourceVersion, and the exact index of our finalizer. Retries on 409/422
/// by re-fetching and recomputing the index so concurrent foreign finalizer
/// edits are preserved, and aborts immediately if the UID changes.
async fn remove_finalizer(
    api: &Api<ModelCacheEntry>,
    name: &str,
    uid: &str,
    initial: ModelCacheEntry,
) -> Result<(), FinalizerError> {
    let mut current = initial;

    for attempt in 0..MAX_PATCH_RETRIES {
        if current.metadata.uid.as_deref() != Some(uid) {
            debug!(
                "Aborting finalizer removal for '{name}': UID changed from '{uid}' to '{:?}'",
                current.metadata.uid
            );
            return Ok(());
        }

        let Some(index) = cache_files_finalizer_index(&current) else {
            return Ok(());
        };

        let rv = current
            .metadata
            .resource_version
            .as_deref()
            .ok_or_else(|| FinalizerError::MissingMetadata {
                name: name.to_string(),
                field: "resourceVersion",
            })?;

        let finalizer_path = format!("/metadata/finalizers/{index}");
        let patch = parse_json_patch(json!([
            { "op": "test", "path": "/metadata/uid", "value": uid },
            { "op": "test", "path": "/metadata/resourceVersion", "value": rv },
            { "op": "test", "path": finalizer_path, "value": CACHE_FILES_FINALIZER },
            { "op": "remove", "path": finalizer_path },
        ]))?;

        match api.patch(name, &PatchParams::default(), &patch).await {
            Ok(_) => {
                debug!(
                    "Removed finalizer '{CACHE_FILES_FINALIZER}' from ModelCacheEntry '{name}' (uid={uid})"
                );
                return Ok(());
            }
            Err(err) if is_not_found(&err) => return Ok(()),
            Err(err) if is_conflict_or_precondition(&err) => {
                debug!(
                    "Finalizer removal for '{name}' hit conflict/precondition (attempt {}); re-fetching",
                    attempt.saturating_add(1)
                );
                match api.get(name).await {
                    Ok(refreshed) => current = refreshed,
                    Err(get_err) if is_not_found(&get_err) => return Ok(()),
                    Err(get_err) => return Err(get_err.into()),
                }
            }
            Err(err) => return Err(err.into()),
        }
    }

    Err(FinalizerError::PatchConflict {
        name: name.to_string(),
        reason: "removing finalizer".to_string(),
    })
}

/// Reconcile a single `ModelCacheEntry` object.
pub async fn reconcile_entry(
    entry: Arc<ModelCacheEntry>,
    ctx: Arc<FinalizerControllerContext>,
) -> Result<Action, FinalizerError> {
    let Some(name) = entry.metadata.name.as_deref() else {
        warn!("Skipping ModelCacheEntry without metadata.name");
        return Ok(Action::await_change());
    };
    let Some(uid) = entry.metadata.uid.as_deref() else {
        warn!("Skipping ModelCacheEntry '{name}' without metadata.uid");
        return Ok(Action::await_change());
    };

    let api = ctx.api();

    if !KubernetesRegistryBackend::is_terminating(&entry) {
        if !has_cache_files_finalizer(&entry) {
            attach_finalizer(&api, name, uid, &entry).await?;
        }
        return Ok(Action::await_change());
    }

    if !has_cache_files_finalizer(&entry) {
        ctx.clear_writer_wait(uid);
        return Ok(Action::await_change());
    }

    // Re-read the CR directly from the API server so we verify the UID,
    // terminating state, finalizer presence, and latest phase before revoking
    // claims, waiting for active writers, or touching disk.
    let latest = match api.get(name).await {
        Ok(cr) => cr,
        Err(err) if is_not_found(&err) => {
            ctx.clear_writer_wait(uid);
            return Ok(Action::await_change());
        }
        Err(err) => return Err(err.into()),
    };

    if latest.metadata.uid.as_deref() != Some(uid)
        || !KubernetesRegistryBackend::is_terminating(&latest)
        || !has_cache_files_finalizer(&latest)
    {
        ctx.clear_writer_wait(uid);
        return Ok(Action::await_change());
    }

    // Revoke any active download lease and enforce the UID-scoped writer wait
    // window before touching cached files.
    if needs_writer_wait(&latest) {
        if has_active_claim(&latest) {
            match revoke_download_claim(&api, name, uid, &latest).await? {
                RevokeOutcome::Revoked => {}
                RevokeOutcome::NotFound => {
                    ctx.clear_writer_wait(uid);
                    return Ok(Action::await_change());
                }
                RevokeOutcome::Conflict => {
                    return Ok(Action::requeue(REQUEUE_CONFLICT_DELAY));
                }
            }
        }
        if let Some(remaining) = ctx.check_or_start_writer_wait(uid) {
            debug!(
                "Terminating ModelCacheEntry '{name}' (uid={uid}) waiting {:?} for active writer to fence",
                remaining
            );
            return Ok(Action::requeue(remaining));
        }
    }

    // Perform survivor-aware cleanup and finalizer removal under the per-model
    // exclusive advisory file lock. Unknown providers and malformed targets skip
    // disk access and never acquire a lock so invalid CRs do not wedge termination.
    match KubernetesRegistryBackend::provider_from_str(&latest.spec.provider) {
        Ok(provider) => {
            if let CleanupPlan::InvalidTarget { reason } =
                plan_cleanup_for_entry(provider, &latest.spec.model_name, &[])
            {
                warn!(
                    "Skipping disk cleanup for terminating ModelCacheEntry '{name}' (uid={uid}) \
                     with invalid target '{}': {reason}",
                    latest.spec.model_name
                );
                remove_finalizer(&api, name, uid, latest).await?;
                ctx.clear_writer_wait(uid);
                return Ok(Action::await_change());
            }

            let parsed = EntryKey::parse(&latest.spec.model_name);
            let canonical_name =
                canonical_model_name(&parsed.model_name, provider).map_err(|e| {
                    FinalizerError::Cleanup {
                        name: name.to_string(),
                        source: e,
                    }
                })?;
            let lock_path =
                LockFile::model_lock_path(&ctx.cache_directory, provider, &canonical_name)
                    .map_err(|e| FinalizerError::Cleanup {
                        name: name.to_string(),
                        source: anyhow::Error::new(e),
                    })?;

            let exclusive_lock = match LockFile::try_acquire_exclusive(&lock_path).await {
                Ok(Some(lock)) => lock,
                Ok(None) => {
                    debug!(
                        "Terminating ModelCacheEntry '{name}' (uid={uid}) model lock is busy; \
                         requeueing in {:?}",
                        REQUEUE_LOCK_BUSY_DELAY
                    );
                    return Ok(Action::requeue(REQUEUE_LOCK_BUSY_DELAY));
                }
                Err(err) => {
                    error!(
                        "Failed to acquire exclusive model lock for terminating ModelCacheEntry \
                         '{name}' (model='{}', uid={uid}); keeping finalizer for retry: {err}",
                        latest.spec.model_name
                    );
                    return Err(FinalizerError::Cleanup {
                        name: name.to_string(),
                        source: anyhow::Error::new(err),
                    });
                }
            };

            // Re-fetch and re-verify UID, terminating state, finalizer, and locked
            // model identity under the exclusive lock before listing survivors or
            // touching cached files.
            let locked_latest = match api.get(name).await {
                Ok(cr) => cr,
                Err(err) if is_not_found(&err) => {
                    ctx.clear_writer_wait(uid);
                    return Ok(Action::await_change());
                }
                Err(err) => return Err(err.into()),
            };
            if locked_latest.metadata.uid.as_deref() != Some(uid)
                || !KubernetesRegistryBackend::is_terminating(&locked_latest)
                || !has_cache_files_finalizer(&locked_latest)
            {
                ctx.clear_writer_wait(uid);
                return Ok(Action::await_change());
            }

            let spec_matches_lock =
                KubernetesRegistryBackend::provider_from_str(&locked_latest.spec.provider)
                    .ok()
                    .filter(|locked_provider| *locked_provider == provider)
                    .and_then(|locked_provider| {
                        let locked_parsed = EntryKey::parse(&locked_latest.spec.model_name);
                        canonical_model_name(&locked_parsed.model_name, locked_provider).ok()
                    })
                    .is_some_and(|locked_canonical| locked_canonical == canonical_name);
            if !spec_matches_lock {
                warn!(
                    "Terminating ModelCacheEntry '{name}' (uid={uid}) changed provider or model \
                     from ({provider:?}, '{canonical_name}') to ('{}', '{}') before lock \
                     acquisition; releasing lock and requeueing",
                    locked_latest.spec.provider, locked_latest.spec.model_name
                );
                drop(exclusive_lock);
                return Ok(Action::requeue(REQUEUE_CONFLICT_DELAY));
            }

            let list = api.list(&ListParams::default()).await?;
            let mut live_records: Vec<ModelRecord> = Vec::with_capacity(list.items.len());
            for cr in &list.items {
                if KubernetesRegistryBackend::is_terminating(cr) {
                    continue;
                }
                match KubernetesRegistryBackend::record_from_cr(cr) {
                    Ok(record) => live_records.push(record),
                    Err(err) => {
                        let cr_name = cr.metadata.name.as_deref().unwrap_or("<no-name>");
                        warn!(
                            "Skipping malformed ModelCacheEntry '{cr_name}' during finalizer survivor check: {err}"
                        );
                    }
                }
            }

            let plan =
                plan_cleanup_for_entry(provider, &locked_latest.spec.model_name, &live_records);
            let (_deleted, _exclusive_lock) = match execute_cleanup_plan(
                &plan,
                &ctx.cache_directory,
                exclusive_lock,
            )
            .await
            {
                Ok(result) => result,
                Err(err) => {
                    error!(
                        "Failed to remove cached files for terminating ModelCacheEntry '{name}' \
                             (model='{}', uid={uid}); keeping finalizer for retry: {err}",
                        locked_latest.spec.model_name
                    );
                    return Err(FinalizerError::Cleanup {
                        name: name.to_string(),
                        source: err,
                    });
                }
            };

            remove_finalizer(&api, name, uid, locked_latest).await?;
        }
        Err(err) => {
            warn!(
                "Skipping disk cleanup for terminating ModelCacheEntry '{name}' (uid={uid}) \
                 with unknown provider '{}': {err}",
                latest.spec.provider
            );
            remove_finalizer(&api, name, uid, latest).await?;
        }
    }

    ctx.clear_writer_wait(uid);
    Ok(Action::await_change())
}

/// Error backoff policy for [`Controller::run`].
fn error_policy(
    entry: Arc<ModelCacheEntry>,
    err: &FinalizerError,
    _ctx: Arc<FinalizerControllerContext>,
) -> Action {
    let name = entry.metadata.name.as_deref().unwrap_or("<no-name>");
    warn!("Requeueing ModelCacheEntry '{name}' after finalizer error: {err}");
    Action::requeue(REQUEUE_ERROR_DELAY)
}

/// Reconcile existing CRs once at startup so operators can observe when the
/// initial pass over pre-upgrade entries completes.
async fn run_initial_pass(
    api: &Api<ModelCacheEntry>,
    namespace: &str,
    ctx: &Arc<FinalizerControllerContext>,
) {
    match api.list(&ListParams::default()).await {
        Ok(list) => {
            let total = list.items.len();
            let mut reconciled: usize = 0;
            for cr in list.items {
                let cr_name = cr
                    .metadata
                    .name
                    .as_deref()
                    .unwrap_or("<no-name>")
                    .to_string();
                match reconcile_entry(Arc::new(cr), Arc::clone(ctx)).await {
                    Ok(_) => reconciled = reconciled.saturating_add(1),
                    Err(err) => {
                        warn!(
                            "Initial finalizer pass hit error on ModelCacheEntry '{cr_name}' \
                             (will retry via controller watch): {err}"
                        );
                    }
                }
            }
            info!(
                "Kubernetes ModelCacheEntry finalizer controller completed initial pass \
                 in namespace '{namespace}': reconciled {reconciled}/{total} existing entries"
            );
        }
        Err(err) => {
            warn!(
                "Kubernetes ModelCacheEntry finalizer controller failed initial list in \
                 namespace '{namespace}' (continuing with controller watch): {err}"
            );
        }
    }
}

/// Spawn the background `ModelCacheEntry` finalizer controller task.
pub fn spawn_finalizer_controller(
    client: Client,
    namespace: String,
    cache_directory: PathBuf,
    mut shutdown_rx: oneshot::Receiver<()>,
) -> tokio::task::JoinHandle<()> {
    let api: Api<ModelCacheEntry> = Api::namespaced(client.clone(), &namespace);
    let ctx = Arc::new(FinalizerControllerContext::new(
        client,
        namespace.clone(),
        cache_directory,
    ));

    info!("Starting Kubernetes ModelCacheEntry finalizer controller in namespace '{namespace}'");

    tokio::spawn(async move {
        tokio::select! {
            _ = run_initial_pass(&api, &namespace, &ctx) => {}
            _ = &mut shutdown_rx => {
                info!("Kubernetes ModelCacheEntry finalizer controller shutting down");
                info!("Kubernetes ModelCacheEntry finalizer controller stopped");
                return;
            }
        }

        Controller::new(api, watcher::Config::default())
            .graceful_shutdown_on(async move {
                let _ = shutdown_rx.await;
                info!("Kubernetes ModelCacheEntry finalizer controller shutting down");
            })
            .run(reconcile_entry, error_policy, ctx)
            .for_each(|result| async move {
                match result {
                    Ok((object_ref, _action)) => {
                        debug!("Reconciled ModelCacheEntry '{}'", object_ref.name);
                    }
                    Err(err) => {
                        warn!("ModelCacheEntry finalizer reconciliation error: {err}");
                    }
                }
            })
            .await;

        info!("Kubernetes ModelCacheEntry finalizer controller stopped");
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::registry::k8s_types::{ModelCacheEntrySpec, ModelCacheEntryStatus};
    use http::{Method, Request, Response, StatusCode};
    use kube::client::Body;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::service_fn;

    const S3_MODEL_MARKER: &str = ".mx-model";

    /// Helper for building mock Kubernetes API responses.
    fn json_response(status: StatusCode, body: &serde_json::Value) -> Response<Body> {
        let bytes = serde_json::to_vec(body).expect("serialize mock response");
        Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(bytes))
            .expect("build response")
    }

    fn k8s_status_error(code: u16, reason: &str, message: &str) -> Response<Body> {
        json_response(
            StatusCode::from_u16(code).expect("valid HTTP status"),
            &json!({
                "kind": "Status",
                "apiVersion": "v1",
                "status": "Failure",
                "message": message,
                "reason": reason,
                "code": code,
            }),
        )
    }

    fn mock_client<F>(handler: F) -> Client
    where
        F: Fn(Method, String, serde_json::Value) -> Response<Body> + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        let svc = service_fn(move |req: Request<Body>| {
            let handler = Arc::clone(&handler);
            async move {
                let method = req.method().clone();
                let path = req.uri().path().to_string();
                let bytes = req
                    .into_body()
                    .collect_bytes()
                    .await
                    .expect("collect mock request body");
                let body: serde_json::Value = if bytes.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::from_slice(&bytes).expect("parse mock request JSON")
                };
                Ok::<_, std::convert::Infallible>(handler(method, path, body))
            }
        });
        Client::new(svc, "test-ns")
    }

    fn cr_json(cr: &ModelCacheEntry) -> serde_json::Value {
        serde_json::to_value(cr).expect("serialize ModelCacheEntry")
    }

    /// Construct a `ModelCacheEntry` directly with standard metadata for `s3://bucket/model`.
    fn base_cr(name: &str, uid: &str, rv: &str, deleting: bool) -> ModelCacheEntry {
        let mut cr = ModelCacheEntry::new(
            name,
            ModelCacheEntrySpec {
                model_name: "s3://bucket/model".to_string(),
                provider: "S3".to_string(),
            },
        );
        cr.metadata.namespace = Some("test-ns".to_string());
        cr.metadata.uid = Some(uid.to_string());
        cr.metadata.resource_version = Some(rv.to_string());
        cr.metadata.finalizers = Some(vec![CACHE_FILES_FINALIZER.to_string()]);
        if deleting {
            cr.metadata.deletion_timestamp = Some(
                k8s_openapi::apimachinery::pkg::apis::meta::v1::Time(chrono::Utc::now()),
            );
        }
        cr.status = Some(status_downloaded());
        cr
    }

    fn status_downloaded() -> ModelCacheEntryStatus {
        ModelCacheEntryStatus {
            phase: phase::DOWNLOADED.to_string(),
            last_used_at: Some("2026-01-01T00:00:00Z".to_string()),
            created_at: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        }
    }

    fn status_downloading(claim_id: Option<&str>) -> ModelCacheEntryStatus {
        ModelCacheEntryStatus {
            phase: phase::DOWNLOADING.to_string(),
            last_used_at: Some("2026-01-01T00:00:00Z".to_string()),
            created_at: Some("2026-01-01T00:00:00Z".to_string()),
            claim_id: claim_id.map(str::to_string),
            lease_expires_at: claim_id.map(|_| "2026-01-01T00:05:00Z".to_string()),
            ..Default::default()
        }
    }

    fn cr_list_json(items: &[ModelCacheEntry]) -> serde_json::Value {
        let json_items: Vec<_> = items.iter().map(cr_json).collect();
        json!({
            "apiVersion": "modelexpress.nvidia.com/v1alpha1",
            "kind": "ModelCacheEntryList",
            "metadata": { "resourceVersion": "99" },
            "items": json_items,
        })
    }

    #[tokio::test]
    async fn attaches_finalizer_to_pre_upgrade_cr_preserving_foreign_finalizers() {
        let mut initial = base_cr("s3-my-model", "uid-pre-upgrade", "10", false);
        initial.metadata.finalizers = Some(vec!["foreign.io/keep".to_string()]);
        let state = Arc::new(Mutex::new(initial));
        let captured_patches = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));

        let state_h = Arc::clone(&state);
        let patches_h = Arc::clone(&captured_patches);
        let client = mock_client(move |method, _path, body| {
            let mut cr = state_h.lock().expect("lock cr");
            match method {
                Method::PATCH => {
                    patches_h.lock().expect("lock patches").push(body.clone());
                    let ops = body.as_array().expect("patch array");
                    assert_eq!(ops[0]["op"], "test");
                    assert_eq!(ops[0]["path"], "/metadata/uid");
                    assert_eq!(ops[0]["value"], cr.metadata.uid.as_deref().expect("uid"));
                    assert_eq!(ops[1]["op"], "test");
                    assert_eq!(ops[1]["path"], "/metadata/resourceVersion");
                    assert_eq!(
                        ops[1]["value"],
                        cr.metadata.resource_version.as_deref().expect("rv")
                    );
                    assert_eq!(ops[2]["op"], "add");
                    assert_eq!(ops[2]["path"], "/metadata/finalizers/-");
                    assert_eq!(ops[2]["value"], CACHE_FILES_FINALIZER);
                    cr.metadata
                        .finalizers
                        .get_or_insert_with(Vec::new)
                        .push(CACHE_FILES_FINALIZER.to_string());
                    cr.metadata.resource_version = Some("11".to_string());
                    json_response(StatusCode::OK, &cr_json(&cr))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let entry = Arc::new(state.lock().expect("lock").clone());

        let action = reconcile_entry(entry, ctx).await.expect("reconcile");
        assert_eq!(action, Action::await_change());
        assert_eq!(
            state.lock().expect("lock").metadata.finalizers,
            Some(vec![
                "foreign.io/keep".to_string(),
                CACHE_FILES_FINALIZER.to_string(),
            ])
        );
        assert_eq!(captured_patches.lock().expect("lock").len(), 1);
    }

    #[tokio::test]
    async fn uid_change_between_read_and_finalizer_remove_aborts_without_touching_recreated_cr() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"data").expect("weights");

        let old_cr = base_cr("s3-my-model", "uid-old", "20", true);
        let mut recreated_cr = base_cr("s3-my-model", "uid-new", "30", false);
        recreated_cr.status = Some(status_downloading(Some("new-claim")));

        let get_count = Arc::new(AtomicUsize::new(0));
        let patch_count = Arc::new(AtomicUsize::new(0));
        let old_h = old_cr.clone();
        let new_h = recreated_cr.clone();
        let get_c = Arc::clone(&get_count);
        let patch_c = Arc::clone(&patch_count);

        let client = mock_client(move |method, path, body| match method {
            Method::GET if path.ends_with("/modelcacheentries") => {
                json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&old_h)))
            }
            Method::GET => {
                // First two GETs (initial check + under-lock re-fetch) return the
                // terminating old CR; third GET (after 422) returns the recreated CR
                // with a new UID.
                let call = get_c.fetch_add(1, Ordering::SeqCst);
                if call < 2 {
                    json_response(StatusCode::OK, &cr_json(&old_h))
                } else {
                    json_response(StatusCode::OK, &cr_json(&new_h))
                }
            }
            Method::PATCH => {
                let call = patch_c.fetch_add(1, Ordering::SeqCst);
                let ops = body.as_array().expect("patch ops");
                // The patch MUST assert the old UID so apiserver rejects it
                // once the CR is recreated under uid-new.
                assert_eq!(ops[0]["op"], "test");
                assert_eq!(ops[0]["path"], "/metadata/uid");
                assert_eq!(ops[0]["value"], "uid-old");
                assert_eq!(call, 0, "must not send a second patch after UID changed");
                k8s_status_error(422, "Invalid", "test operation failed: /metadata/uid")
            }
            other => panic!("unexpected HTTP method {other}"),
        });

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let action = reconcile_entry(Arc::new(old_cr), ctx)
            .await
            .expect("reconcile aborts cleanly on UID change");
        assert_eq!(action, Action::await_change());
        assert_eq!(patch_count.load(Ordering::SeqCst), 1);
        assert_eq!(get_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn concurrent_foreign_finalizer_insertion_retries_and_preserves_foreign_finalizer() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"data").expect("weights");

        // Initially our finalizer is at index 0. Between GET and PATCH, another
        // controller inserts "foreign.io/keep" at index 0, shifting ours to index 1.
        let initial_cr = base_cr("s3-my-model", "uid-foreign", "40", true);

        let server_state = Arc::new(Mutex::new(initial_cr.clone()));
        let get_count = Arc::new(AtomicUsize::new(0));
        let patch_count = Arc::new(AtomicUsize::new(0));

        let state_h = Arc::clone(&server_state);
        let get_c = Arc::clone(&get_count);
        let patch_c = Arc::clone(&patch_count);
        let client = mock_client(move |method, path, body| {
            let mut cr = state_h.lock().expect("lock state");
            match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cr)))
                }
                Method::GET => {
                    let call = get_c.fetch_add(1, Ordering::SeqCst);
                    let resp = json_response(StatusCode::OK, &cr_json(&cr));
                    if call == 1 {
                        // Simulate concurrent foreign finalizer insertion right
                        // after the under-lock read.
                        cr.metadata.finalizers = Some(vec![
                            "foreign.io/keep".to_string(),
                            CACHE_FILES_FINALIZER.to_string(),
                        ]);
                        cr.metadata.resource_version = Some("41".to_string());
                    }
                    resp
                }
                Method::PATCH => {
                    let call = patch_c.fetch_add(1, Ordering::SeqCst);
                    let ops = body.as_array().expect("patch ops");
                    let tested_rv = ops[1]["value"].as_str().expect("rv string");
                    if Some(tested_rv) != cr.metadata.resource_version.as_deref() {
                        assert_eq!(call, 0);
                        return k8s_status_error(
                            422,
                            "Invalid",
                            "test operation failed: /metadata/resourceVersion",
                        );
                    }
                    assert_eq!(call, 1);
                    assert_eq!(ops[0]["value"], cr.metadata.uid.as_deref().expect("uid"));
                    assert_eq!(ops[2]["path"], "/metadata/finalizers/1");
                    assert_eq!(ops[2]["value"], CACHE_FILES_FINALIZER);
                    assert_eq!(ops[3]["op"], "remove");
                    assert_eq!(ops[3]["path"], "/metadata/finalizers/1");

                    let finalizers = cr.metadata.finalizers.as_mut().expect("finalizers");
                    let removed = finalizers.remove(1);
                    assert_eq!(removed, CACHE_FILES_FINALIZER);
                    cr.metadata.resource_version = Some("42".to_string());
                    json_response(StatusCode::OK, &cr_json(&cr))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let action = reconcile_entry(Arc::new(initial_cr), ctx)
            .await
            .expect("reconcile succeeds after retry");
        assert_eq!(action, Action::await_change());
        assert_eq!(patch_count.load(Ordering::SeqCst), 2);
        assert_eq!(
            server_state.lock().expect("lock").metadata.finalizers,
            Some(vec!["foreign.io/keep".to_string()])
        );
        assert!(!model_dir.exists(), "cached files should be deleted");
    }

    #[tokio::test]
    async fn terminating_downloading_cr_revokes_claim_and_keys_writer_wait_by_uid() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"data").expect("weights");

        let mut cr1 = base_cr("s3-shared-name", "uid-first", "50", true);
        cr1.status = Some(status_downloading(Some("claim-1")));
        let state = Arc::new(Mutex::new(cr1));
        let status_revokes = Arc::new(AtomicUsize::new(0));

        let state_h = Arc::clone(&state);
        let revokes_h = Arc::clone(&status_revokes);
        let client = mock_client(move |method, path, body| {
            let mut cr = state_h.lock().expect("lock");
            match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cr)))
                }
                Method::GET => json_response(StatusCode::OK, &cr_json(&cr)),
                Method::PATCH if path.ends_with("/status") => {
                    let ops = body.as_array().expect("status patch");
                    assert_eq!(ops[0]["path"], "/metadata/uid");
                    assert_eq!(ops[0]["value"], cr.metadata.uid.as_deref().expect("uid"));
                    assert_eq!(ops[1]["path"], "/metadata/resourceVersion");
                    assert_eq!(
                        ops[1]["value"],
                        cr.metadata.resource_version.as_deref().expect("rv")
                    );
                    assert_eq!(ops[2]["path"], "/status/claimId");
                    assert!(ops[2]["value"].is_null());
                    assert_eq!(ops[3]["path"], "/status/leaseExpiresAt");
                    assert!(ops[3]["value"].is_null());
                    if let Some(status) = cr.status.as_mut() {
                        status.claim_id = None;
                        status.lease_expires_at = None;
                    }
                    let rv = cr.metadata.resource_version.as_deref().unwrap_or("");
                    cr.metadata.resource_version = Some(format!("{rv}-revoked"));
                    revokes_h.fetch_add(1, Ordering::SeqCst);
                    json_response(StatusCode::OK, &cr_json(&cr))
                }
                Method::PATCH => {
                    cr.metadata.finalizers = Some(Vec::new());
                    let rv = cr.metadata.resource_version.as_deref().unwrap_or("");
                    cr.metadata.resource_version = Some(format!("{rv}-finalized"));
                    json_response(StatusCode::OK, &cr_json(&cr))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });

        let wait_duration = Duration::from_millis(80);
        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            wait_duration,
        ));

        // First reconcile: revokes claim and requeues without deleting files yet.
        let snapshot1 = Arc::new(state.lock().expect("lock").clone());
        let action1 = reconcile_entry(snapshot1, Arc::clone(&ctx))
            .await
            .expect("first reconcile");
        assert_ne!(
            action1,
            Action::await_change(),
            "must wait for writer window"
        );
        assert_eq!(status_revokes.load(Ordering::SeqCst), 1);
        assert!(
            model_dir.exists(),
            "files must NOT be deleted before writer wait elapses"
        );

        // Wait for uid-first's window to expire, then reconcile again.
        tokio::time::sleep(wait_duration + Duration::from_millis(20)).await;
        let snapshot2 = Arc::new(state.lock().expect("lock").clone());
        let action2 = reconcile_entry(snapshot2, Arc::clone(&ctx))
            .await
            .expect("second reconcile");
        assert_eq!(action2, Action::await_change());
        assert!(!model_dir.exists(), "files should be deleted after wait");
        assert_eq!(
            state.lock().expect("lock").metadata.finalizers,
            Some(Vec::new())
        );

        // Recreate CR with the EXACT SAME metadata.name but a NEW UID and new files.
        fs::create_dir_all(&model_dir).expect("recreate dir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"new-data").expect("weights");
        {
            let mut cr = state.lock().expect("lock");
            cr.metadata.uid = Some("uid-second".to_string());
            cr.metadata.resource_version = Some("60".to_string());
            cr.metadata.finalizers = Some(vec![CACHE_FILES_FINALIZER.to_string()]);
            cr.status = Some(status_downloading(Some("claim-2")));
        }

        // First reconcile for uid-second MUST start a fresh wait window instead
        // of reusing uid-first's elapsed window.
        let snapshot3 = Arc::new(state.lock().expect("lock").clone());
        let action3 = reconcile_entry(snapshot3, Arc::clone(&ctx))
            .await
            .expect("recreated CR first reconcile");
        assert_ne!(
            action3,
            Action::await_change(),
            "recreated CR with new UID must get its own full writer wait"
        );
        assert_eq!(status_revokes.load(Ordering::SeqCst), 2);
        assert!(
            model_dir.exists(),
            "recreated CR files must stay intact during uid-second wait window"
        );
    }

    #[tokio::test]
    async fn disk_cleanup_failure_keeps_finalizer_attached_for_retry() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside dir");
        let outside_model = outside.path().join("model");
        fs::create_dir_all(&outside_model).expect("mkdir outside");
        fs::write(outside_model.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(outside_model.join("secret.bin"), b"keep").expect("secret");

        // Place an ancestor symlink inside cache_dir/s3 pointing outside cache_dir.
        fs::create_dir_all(temp_dir.path().join("s3")).expect("mkdir s3");
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), temp_dir.path().join("s3/bucket"))
            .expect("create ancestor symlink");

        let cr = base_cr("s3-escaped", "uid-escape", "70", true);
        let patch_calls = Arc::new(AtomicUsize::new(0));
        let cr_h = cr.clone();
        let patch_c = Arc::clone(&patch_calls);

        let client = mock_client(move |method, path, _body| match method {
            Method::GET if path.ends_with("/modelcacheentries") => {
                json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cr_h)))
            }
            Method::GET => json_response(StatusCode::OK, &cr_json(&cr_h)),
            Method::PATCH => {
                patch_c.fetch_add(1, Ordering::SeqCst);
                panic!("finalizer MUST NOT be removed when disk cleanup fails");
            }
            other => panic!("unexpected HTTP method {other}"),
        });

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let err = reconcile_entry(Arc::new(cr), ctx)
            .await
            .expect_err("cleanup failure must return Err so controller retries");
        assert!(matches!(err, FinalizerError::Cleanup { .. }));
        assert_eq!(patch_calls.load(Ordering::SeqCst), 0);
        assert!(
            outside_model.join("secret.bin").exists(),
            "outside target must never be deleted"
        );
    }

    #[tokio::test]
    async fn invalid_target_cr_skips_disk_and_strips_finalizer() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let sentinel = temp_dir.path().join("keep.txt");
        fs::write(&sentinel, b"untouched").expect("write sentinel");

        let mut cr = base_cr("hf-traversal", "uid-invalid", "80", true);
        cr.spec.model_name = "org/model@../escape".to_string();
        cr.spec.provider = "HuggingFace".to_string();
        let state = Arc::new(Mutex::new(cr.clone()));
        let state_h = Arc::clone(&state);

        let client = mock_client(move |method, path, _body| {
            let mut cr = state_h.lock().expect("lock");
            match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cr)))
                }
                Method::GET => json_response(StatusCode::OK, &cr_json(&cr)),
                Method::PATCH => {
                    cr.metadata.finalizers = Some(Vec::new());
                    json_response(StatusCode::OK, &cr_json(&cr))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let action = reconcile_entry(Arc::new(cr), ctx)
            .await
            .expect("invalid target strips finalizer cleanly");
        assert_eq!(action, Action::await_change());
        assert_eq!(
            state.lock().expect("lock").metadata.finalizers,
            Some(Vec::new())
        );
        assert!(sentinel.exists(), "unrelated files must remain untouched");
    }

    #[tokio::test]
    async fn surviving_live_cr_protects_files_while_terminating_cr_strips_finalizer() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"keep-me").expect("weights");

        let terminating_cr = base_cr("s3-legacy", "uid-term", "90", true);
        let survivor_cr = base_cr("s3-scoped", "uid-live", "91", false);

        let state = Arc::new(Mutex::new(terminating_cr.clone()));
        let state_h = Arc::clone(&state);
        let survivor_h = survivor_cr.clone();

        let client = mock_client(move |method, path, _body| {
            let mut cr = state_h.lock().expect("lock");
            match method {
                Method::GET if path.ends_with("/modelcacheentries") => json_response(
                    StatusCode::OK,
                    &cr_list_json(&[cr.clone(), survivor_h.clone()]),
                ),
                Method::GET => json_response(StatusCode::OK, &cr_json(&cr)),
                Method::PATCH => {
                    cr.metadata.finalizers = Some(Vec::new());
                    json_response(StatusCode::OK, &cr_json(&cr))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let action = reconcile_entry(Arc::new(terminating_cr), ctx)
            .await
            .expect("reconcile with survivor succeeds");
        assert_eq!(action, Action::await_change());
        assert_eq!(
            state.lock().expect("lock").metadata.finalizers,
            Some(Vec::new())
        );
        assert!(
            model_dir.join("weights.bin").exists(),
            "surviving live CR must protect cached files"
        );
    }

    #[tokio::test]
    async fn busy_model_lock_requeues_finalizer_and_lock_io_error_fails_closed() {
        // 1. Held shared lock causes immediate 1 s requeue without deleting files or
        //    removing the finalizer; once released, reconcile cleans up cleanly.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"data").expect("weights");

        let cr = base_cr("s3-my-model", "uid-lock-busy", "80", true);
        let state = Arc::new(Mutex::new(cr.clone()));
        let patch_count = Arc::new(AtomicUsize::new(0));
        let state_h = Arc::clone(&state);
        let patch_c = Arc::clone(&patch_count);

        let client = mock_client(move |method, path, _body| {
            let mut cur = state_h.lock().expect("lock state");
            match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cur)))
                }
                Method::GET => json_response(StatusCode::OK, &cr_json(&cur)),
                Method::PATCH => {
                    patch_c.fetch_add(1, Ordering::SeqCst);
                    cur.metadata.finalizers = Some(Vec::new());
                    json_response(StatusCode::OK, &cr_json(&cur))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });

        let lock_path = LockFile::model_lock_path(
            temp_dir.path(),
            modelexpress_common::models::ModelProvider::S3,
            "s3://bucket/model",
        )
        .expect("lock path");
        let shared = LockFile::acquire_shared_timeout(&lock_path, Duration::from_secs(1))
            .await
            .expect("hold shared lock");

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let busy_action = reconcile_entry(Arc::new(cr.clone()), Arc::clone(&ctx))
            .await
            .expect("busy lock should requeue without error");
        assert_eq!(busy_action, Action::requeue(REQUEUE_LOCK_BUSY_DELAY));
        assert_eq!(patch_count.load(Ordering::SeqCst), 0);
        assert!(
            model_dir.join("weights.bin").exists(),
            "files must stay intact while model lock is busy"
        );

        drop(shared);

        let done_action = reconcile_entry(Arc::new(cr), ctx)
            .await
            .expect("reconcile succeeds once lock is free");
        assert_eq!(done_action, Action::await_change());
        assert_eq!(patch_count.load(Ordering::SeqCst), 1);
        assert!(
            !model_dir.exists(),
            "files should be removed once lock is free"
        );

        // 2. Lock I/O error fails closed, keeping finalizer attached and files intact.
        let io_dir = tempfile::tempdir().expect("io tempdir");
        let io_model_dir = io_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&io_model_dir).expect("mkdir");
        fs::write(io_model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(io_model_dir.join("weights.bin"), b"data").expect("weights");
        fs::write(io_dir.path().join(".mx"), b"blocker").expect("block lock dir");

        let io_cr = base_cr("s3-my-model", "uid-lock-io", "81", true);
        let io_cr_h = io_cr.clone();
        let io_client = mock_client(move |method, _path, _body| match method {
            Method::GET => json_response(StatusCode::OK, &cr_json(&io_cr_h)),
            Method::PATCH => panic!("must not remove finalizer on lock I/O error"),
            other => panic!("unexpected HTTP method {other}"),
        });
        let io_ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            io_client,
            "test-ns".to_string(),
            io_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let err = reconcile_entry(Arc::new(io_cr), io_ctx)
            .await
            .expect_err("lock I/O error must fail closed");
        assert!(matches!(err, FinalizerError::Cleanup { .. }));
        assert!(io_model_dir.join("weights.bin").exists());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_reconcile_keeps_exclusive_lock_until_blocking_cleanup_ends() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"data").expect("weights");

        let cr = base_cr("s3-cancel", "uid-cancel", "82", true);
        let cr_h = cr.clone();
        let client = mock_client(move |method, path, _body| match method {
            Method::GET if path.ends_with("/modelcacheentries") => {
                json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cr_h)))
            }
            Method::GET => json_response(StatusCode::OK, &cr_json(&cr_h)),
            Method::PATCH => panic!("cancelled reconcile must not reach finalizer patch"),
            other => panic!("unexpected HTTP method {other}"),
        });

        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        crate::registry::cache_cleanup::register_before_cleanup_hook(temp_dir.path(), move || {
            let _ = entered_tx.send(());
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        });

        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));
        let cancelled = {
            let reconcile_fut = reconcile_entry(Arc::new(cr), ctx);
            tokio::pin!(reconcile_fut);
            tokio::select! {
                _ = &mut reconcile_fut => false,
                res = entered_rx => {
                    res.expect("blocking cleanup should enter before first file op");
                    true
                }
            }
        };
        assert!(
            cancelled,
            "reconcile_entry future must be dropped while blocking cleanup is held"
        );
        assert!(
            model_dir.join("weights.bin").exists(),
            "files must still exist while blocking cleanup is held before its first op"
        );

        let lock_path = LockFile::model_lock_path(
            temp_dir.path(),
            modelexpress_common::models::ModelProvider::S3,
            "s3://bucket/model",
        )
        .expect("lock path");
        let busy = LockFile::acquire_shared_timeout(&lock_path, Duration::from_millis(120)).await;
        assert!(
            matches!(
                busy,
                Err(modelexpress_common::providers::lock_file::LockFileError::Busy { .. })
            ),
            "shared lock must stay busy while cancelled reconcile's blocking cleanup runs: {busy:?}"
        );

        release_tx.send(()).expect("release blocking cleanup");
        let _shared = LockFile::acquire_shared_timeout(&lock_path, Duration::from_secs(2))
            .await
            .expect("shared lock must succeed once blocking cleanup finishes");
        assert!(
            !model_dir.exists(),
            "blocking cleanup should remove model files before dropping exclusive lock"
        );
    }

    #[tokio::test]
    async fn spec_change_between_two_gets_requeues_without_touching_new_model_while_same_model_revision_change_proceeds()
     {
        // 1. Model changes from model-a to model-b between GET 1 and locked GET 2:
        //    must release A's lock, touch neither A nor B, keep finalizer, and requeue.
        {
            let temp_dir = tempfile::tempdir().expect("tempdir");
            let dir_a = temp_dir.path().join("s3/bucket/model-a");
            let dir_b = temp_dir.path().join("s3/bucket/model-b");
            for dir in [&dir_a, &dir_b] {
                fs::create_dir_all(dir).expect("mkdir");
                fs::write(dir.join(S3_MODEL_MARKER), b"").expect("marker");
                fs::write(dir.join("weights.bin"), b"keep").expect("weights");
            }

            let mut cr_a = base_cr("s3-mutated", "uid-mut", "90", true);
            cr_a.spec.model_name = "s3://bucket/model-a".to_string();
            let mut cr_b = cr_a.clone();
            cr_b.spec.model_name = "s3://bucket/model-b".to_string();
            cr_b.metadata.resource_version = Some("91".to_string());

            let get_count = Arc::new(AtomicUsize::new(0));
            let list_count = Arc::new(AtomicUsize::new(0));
            let patch_count = Arc::new(AtomicUsize::new(0));
            let cr_a_h = cr_a.clone();
            let cr_b_h = cr_b.clone();
            let get_c = Arc::clone(&get_count);
            let list_c = Arc::clone(&list_count);
            let patch_c = Arc::clone(&patch_count);

            let client = mock_client(move |method, path, _body| match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    list_c.fetch_add(1, Ordering::SeqCst);
                    panic!("must not list survivors after model identity changed under lock");
                }
                Method::GET => {
                    if get_c.fetch_add(1, Ordering::SeqCst) == 0 {
                        json_response(StatusCode::OK, &cr_json(&cr_a_h))
                    } else {
                        json_response(StatusCode::OK, &cr_json(&cr_b_h))
                    }
                }
                Method::PATCH => {
                    patch_c.fetch_add(1, Ordering::SeqCst);
                    panic!("must not remove finalizer when spec changed under wrong lock");
                }
                other => panic!("unexpected HTTP method {other}"),
            });

            let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
                client,
                "test-ns".to_string(),
                temp_dir.path().to_path_buf(),
                Duration::ZERO,
            ));
            let action = reconcile_entry(Arc::new(cr_a), ctx)
                .await
                .expect("spec change should requeue cleanly");
            assert_eq!(action, Action::requeue(REQUEUE_CONFLICT_DELAY));
            assert_eq!(get_count.load(Ordering::SeqCst), 2);
            assert_eq!(list_count.load(Ordering::SeqCst), 0);
            assert_eq!(patch_count.load(Ordering::SeqCst), 0);
            assert!(
                dir_a.join("weights.bin").exists(),
                "A files must stay intact"
            );
            assert!(
                dir_b.join("weights.bin").exists(),
                "B files must stay intact"
            );

            let lock_a = LockFile::model_lock_path(
                temp_dir.path(),
                modelexpress_common::models::ModelProvider::S3,
                "s3://bucket/model-a",
            )
            .expect("lock path A");
            assert!(
                LockFile::try_acquire_exclusive(&lock_a)
                    .await
                    .expect("try_acquire_exclusive A")
                    .is_some(),
                "lock A must be released on requeue"
            );
        }

        // 2. Provider changes between GET 1 (S3) and GET 2 (GCS): same requeue guard.
        {
            let temp_dir = tempfile::tempdir().expect("tempdir");
            let cr_s3 = base_cr("provider-mutated", "uid-prov", "92", true);
            let mut cr_gcs = cr_s3.clone();
            cr_gcs.spec.provider = "Gcs".to_string();
            cr_gcs.spec.model_name = "gs://bucket/model".to_string();

            let get_count = Arc::new(AtomicUsize::new(0));
            let cr_s3_h = cr_s3.clone();
            let cr_gcs_h = cr_gcs.clone();
            let get_c = Arc::clone(&get_count);
            let client = mock_client(move |method, path, _body| match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    panic!("must not list survivors after provider changed under lock");
                }
                Method::GET => {
                    if get_c.fetch_add(1, Ordering::SeqCst) == 0 {
                        json_response(StatusCode::OK, &cr_json(&cr_s3_h))
                    } else {
                        json_response(StatusCode::OK, &cr_json(&cr_gcs_h))
                    }
                }
                Method::PATCH => panic!("must not remove finalizer when provider changed"),
                other => panic!("unexpected HTTP method {other}"),
            });

            let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
                client,
                "test-ns".to_string(),
                temp_dir.path().to_path_buf(),
                Duration::ZERO,
            ));
            let action = reconcile_entry(Arc::new(cr_s3), ctx)
                .await
                .expect("provider change should requeue cleanly");
            assert_eq!(action, Action::requeue(REQUEUE_CONFLICT_DELAY));
        }

        // 3. Same-model revision change (HuggingFace rev1 -> rev2) keeps the same lock key
        //    and proceeds without requeueing.
        {
            let temp_dir = tempfile::tempdir().expect("tempdir");
            let snap_rev1 = temp_dir.path().join("models--org--model/snapshots/rev1");
            let snap_rev2 = temp_dir.path().join("models--org--model/snapshots/rev2");
            for snap in [&snap_rev1, &snap_rev2] {
                fs::create_dir_all(snap).expect("mkdir");
                fs::write(snap.join("config.json"), b"{}").expect("write config");
            }

            let rev1_key = EntryKey::new("org/model", Some("rev1".to_string()), false).to_string();
            let rev2_key = EntryKey::new("org/model", Some("rev2".to_string()), false).to_string();
            let mut cr_rev1 = base_cr("hf-same-model", "uid-hf", "94", true);
            cr_rev1.spec.provider = "HuggingFace".to_string();
            cr_rev1.spec.model_name = rev1_key.clone();
            let mut cr_rev2 = cr_rev1.clone();
            cr_rev2.spec.model_name = rev2_key;

            let mut survivor_rev1 = base_cr("hf-survivor", "uid-surv", "95", false);
            survivor_rev1.spec.provider = "HuggingFace".to_string();
            survivor_rev1.spec.model_name = rev1_key;

            let get_count = Arc::new(AtomicUsize::new(0));
            let patch_count = Arc::new(AtomicUsize::new(0));
            let cr_rev1_h = cr_rev1.clone();
            let cr_rev2_h = cr_rev2.clone();
            let surv_h = survivor_rev1.clone();
            let get_c = Arc::clone(&get_count);
            let patch_c = Arc::clone(&patch_count);
            let client = mock_client(move |method, path, _body| match method {
                Method::GET if path.ends_with("/modelcacheentries") => json_response(
                    StatusCode::OK,
                    &cr_list_json(&[cr_rev2_h.clone(), surv_h.clone()]),
                ),
                Method::GET => {
                    if get_c.fetch_add(1, Ordering::SeqCst) == 0 {
                        json_response(StatusCode::OK, &cr_json(&cr_rev1_h))
                    } else {
                        json_response(StatusCode::OK, &cr_json(&cr_rev2_h))
                    }
                }
                Method::PATCH => {
                    patch_c.fetch_add(1, Ordering::SeqCst);
                    let mut patched = cr_rev2_h.clone();
                    patched.metadata.finalizers = Some(Vec::new());
                    json_response(StatusCode::OK, &cr_json(&patched))
                }
                other => panic!("unexpected HTTP method {other}"),
            });

            let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
                client,
                "test-ns".to_string(),
                temp_dir.path().to_path_buf(),
                Duration::ZERO,
            ));
            let action = reconcile_entry(Arc::new(cr_rev1), ctx)
                .await
                .expect("same-model revision change should proceed without requeue");
            assert_eq!(action, Action::await_change());
            assert_eq!(patch_count.load(Ordering::SeqCst), 1);
            assert!(
                snap_rev1.join("config.json").exists(),
                "surviving rev1 stays"
            );
            assert!(
                !snap_rev2.exists(),
                "locked rev2 snapshot should be removed"
            );
        }
    }

    /// Ruling ^138 P2 / ^140 / ^141 item 8: many concurrent waiters polling a terminating
    /// CR (60 waiters x 20 ms claim latency, 20 ms poll interval) must hold ZERO shared
    /// file locks, so the Kubernetes finalizer acquires the exclusive lock on its very
    /// first try, deletes the files, and strips the finalizer without requeueing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn many_waiters_on_terminating_cr_hold_no_shared_lock_and_finalizer_acquires_exclusive() {
        use crate::metrics::registry::{DownloadMetrics, RegistryMetrics};
        use crate::registry::backend::{ClaimOutcome, RegistryBackend, RegistryResult};
        use crate::registry::state::RegistryManager;
        use crate::services::{DownloadTarget, LeaseTimings, ModelDownloadTracker};
        use modelexpress_common::models::{ModelProvider, ModelStatus};

        struct SlowTerminatingBackend {
            claim_polls: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl RegistryBackend for SlowTerminatingBackend {
            async fn connect(&self) -> RegistryResult<()> {
                Ok(())
            }
            async fn get_status(&self, _model_name: &str) -> RegistryResult<Option<ModelStatus>> {
                Ok(Some(ModelStatus::DOWNLOADING))
            }
            async fn get_model_record(
                &self,
                _model_name: &str,
            ) -> RegistryResult<Option<ModelRecord>> {
                Ok(None)
            }
            async fn set_status(
                &self,
                _model_name: &str,
                _provider: ModelProvider,
                _status: ModelStatus,
                _message: Option<String>,
            ) -> RegistryResult<()> {
                Ok(())
            }
            async fn touch_model(&self, _model_name: &str) -> RegistryResult<()> {
                Ok(())
            }
            async fn delete_model(&self, _model_name: &str) -> RegistryResult<()> {
                Ok(())
            }
            async fn get_models_by_last_used(
                &self,
                _limit: Option<u32>,
            ) -> RegistryResult<Vec<ModelRecord>> {
                Ok(Vec::new())
            }
            async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
                Ok((1, 0, 0))
            }
            async fn try_claim_for_download(
                &self,
                _model_name: &str,
                _provider: ModelProvider,
                _claim_id: &str,
                _lease_duration: Duration,
            ) -> RegistryResult<ClaimOutcome> {
                // Emulate a 20 ms API server round-trip returning DOWNLOADING while
                // the CR is terminating (Odo ^138 starvation setup).
                tokio::time::sleep(Duration::from_millis(20)).await;
                self.claim_polls.fetch_add(1, Ordering::SeqCst);
                Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING))
            }
            async fn try_reset_error_for_retry(
                &self,
                _model_name: &str,
                _provider: ModelProvider,
                _claim_id: &str,
                _lease_duration: Duration,
            ) -> RegistryResult<bool> {
                Ok(false)
            }
            async fn try_reclaim_downloaded(
                &self,
                _model_name: &str,
                _provider: ModelProvider,
                _claim_id: &str,
                _lease_duration: Duration,
            ) -> RegistryResult<bool> {
                Ok(false)
            }
            async fn refresh_download_claim(
                &self,
                _model_name: &str,
                _provider: ModelProvider,
                _claim_id: &str,
                _lease_duration: Duration,
            ) -> RegistryResult<bool> {
                Ok(false)
            }
            async fn finish_download_claim(
                &self,
                _model_name: &str,
                _provider: ModelProvider,
                _claim_id: &str,
                _status: ModelStatus,
                _message: Option<String>,
            ) -> RegistryResult<bool> {
                Ok(false)
            }
        }

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = temp_dir.path().join("s3/bucket/model");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(model_dir.join(S3_MODEL_MARKER), b"").expect("marker");
        fs::write(model_dir.join("weights.bin"), b"data").expect("weights");

        let claim_polls = Arc::new(AtomicUsize::new(0));
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(
            SlowTerminatingBackend {
                claim_polls: Arc::clone(&claim_polls),
            },
        )));
        let mut metrics_reg = crate::metrics::new_registry();
        let tracker = ModelDownloadTracker::new(
            registry,
            temp_dir.path().to_path_buf(),
            DownloadMetrics::register(&mut metrics_reg),
            RegistryMetrics::register(&mut metrics_reg),
        )
        .with_lease_timings(LeaseTimings {
            poll_interval: Duration::from_millis(20),
            ..LeaseTimings::default()
        });

        let target = DownloadTarget {
            model_name: "s3://bucket/model".to_string(),
            provider: ModelProvider::S3,
            revision: None,
            ignore_weights: true,
        };
        let mut waiter_rxs = Vec::with_capacity(60);
        let mut waiter_handles = Vec::with_capacity(60);
        for _ in 0..60 {
            let (tx, rx) = tokio::sync::mpsc::channel(4);
            let t = tracker.clone();
            let tgt = target.clone();
            waiter_rxs.push(rx);
            waiter_handles.push(tokio::spawn(async move {
                t.ensure_model_downloaded(&tgt, &tx).await
            }));
        }

        // Wait until the 60 waiters are actively polling concurrently.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while claim_polls.load(Ordering::SeqCst) < 120 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "waiters should complete at least 120 polls"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Finalizer reconciles the terminating CR while all 60 waiters continue polling:
        // must acquire the exclusive lock on attempt 1 without requeueing.
        let cr = base_cr("s3-many-waiters", "uid-many-waiters", "99", true);
        let state = Arc::new(Mutex::new(cr.clone()));
        let state_h = Arc::clone(&state);
        let patch_count = Arc::new(AtomicUsize::new(0));
        let patch_c = Arc::clone(&patch_count);
        let client = mock_client(move |method, path, _body| {
            let mut cur = state_h.lock().expect("lock");
            match method {
                Method::GET if path.ends_with("/modelcacheentries") => {
                    json_response(StatusCode::OK, &cr_list_json(std::slice::from_ref(&cur)))
                }
                Method::GET => json_response(StatusCode::OK, &cr_json(&cur)),
                Method::PATCH => {
                    patch_c.fetch_add(1, Ordering::SeqCst);
                    cur.metadata.finalizers = Some(Vec::new());
                    json_response(StatusCode::OK, &cr_json(&cur))
                }
                other => panic!("unexpected HTTP method {other}"),
            }
        });
        let ctx = Arc::new(FinalizerControllerContext::with_writer_wait(
            client,
            "test-ns".to_string(),
            temp_dir.path().to_path_buf(),
            Duration::ZERO,
        ));

        let action = reconcile_entry(Arc::new(cr), ctx)
            .await
            .expect("finalizer must succeed while 60 waiters poll");
        assert_eq!(
            action,
            Action::await_change(),
            "finalizer must acquire exclusive lock immediately, not requeue due to waiter lock starvation"
        );
        assert_eq!(patch_count.load(Ordering::SeqCst), 1);
        assert!(!model_dir.exists(), "cached files must be removed");

        // Disconnect all waiters so their tasks exit cleanly.
        drop(waiter_rxs);
        for handle in waiter_handles {
            let _ = tokio::time::timeout(Duration::from_secs(2), handle)
                .await
                .expect("waiter exit timeout");
        }
    }
}
