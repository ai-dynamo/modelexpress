// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deletes the cached files of a `ModelCacheEntry` when the CR is deleted.
//!
//! Every CR carries [`CACHE_FILES_FINALIZER`], so a delete from any client
//! (`kubectl delete mxcache`, the server itself) only marks the CR. This controller
//! sees the mark, deletes the files the entry covers, and then removes the finalizer so
//! Kubernetes finishes the delete. A delete made while no server runs is finished when
//! one starts. A CR without the finalizer gets it on its first reconcile.
//!
//! Every replica runs this controller. Deleting files and removing the finalizer are
//! both idempotent, so replicas that race on one CR converge.
//!
//! A CR marked while its model downloads has a writer that must stop first. The
//! controller takes the claim away, which fences the owner exactly like a lease
//! takeover does: its next heartbeat fails, so it stops, and it cannot publish the
//! entry as downloaded. The files go once the lease has stayed unchanged for a full
//! lease duration.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use futures::StreamExt;
use kube::api::{Api, ListParams, Patch, PatchParams};
use kube::runtime::controller::{self, Action, Controller};
use kube::runtime::finalizer::{self as kube_finalizer, Event};
use kube::runtime::watcher;
use kube::{Client, ResourceExt};
use modelexpress_common::models::ModelStatus;
use serde_json::json;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use super::{
    KubernetesRegistryBackend, LeaseFingerprint, LeaseObservation, lease_observation_expired,
};
use crate::registry::entry_files::{self, FileScope, Survivor};
use crate::registry::entry_key::EntryKey;
use crate::registry::k8s_types::{
    CACHE_FILES_FINALIZER, ModelCacheEntry, ModelCacheEntryStatus, phase,
};

/// How often a CR that waits for its download to stop is checked again.
const WRITER_POLL_INTERVAL: Duration = Duration::from_secs(5);
const RETRY_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error)]
pub enum CleanupError {
    /// Not a failure. `finalizer()` removes the finalizer on any `Ok`, so a cleanup that
    /// must wait reports this error and requeues.
    #[error("waiting for the download of {0} to stop before deleting its files")]
    DownloadInFlight(String),
    #[error(transparent)]
    Kube(#[from] kube::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("failed to delete cached files of {name}: {reason}")]
    Files { name: String, reason: String },
}

/// What a deletion-marked CR needs next.
#[derive(Debug, PartialEq, Eq)]
enum CleanupStep<'a> {
    /// A download owner still holds the claim. Take it away.
    RevokeClaim(&'a str),
    /// The last owner may still be writing.
    WaitForWriter,
    DeleteFiles,
}

fn next_step<'a>(
    cr_name: &str,
    status: &'a ModelCacheEntryStatus,
    observations: &mut HashMap<String, LeaseObservation>,
    now: Instant,
    lease_duration: Duration,
) -> CleanupStep<'a> {
    if KubernetesRegistryBackend::status_from_phase(&status.phase) != ModelStatus::DOWNLOADING {
        return CleanupStep::DeleteFiles;
    }
    if let Some(claim_id) = status.claim_id.as_deref() {
        return CleanupStep::RevokeClaim(claim_id);
    }
    let fingerprint = LeaseFingerprint {
        claim_id: None,
        lease_expires_at: status.lease_expires_at.clone(),
    };
    if lease_observation_expired(observations, cr_name, fingerprint, now, lease_duration) {
        CleanupStep::DeleteFiles
    } else {
        CleanupStep::WaitForWriter
    }
}

fn entry_key(cr: &ModelCacheEntry) -> EntryKey {
    match KubernetesRegistryBackend::provider_from_str(&cr.spec.provider) {
        Ok(provider) => entry_files::record_key(&cr.spec.model_name, provider),
        Err(_) => EntryKey::parse(&cr.spec.model_name),
    }
}

fn survivors(items: &[ModelCacheEntry]) -> Vec<Survivor> {
    items
        .iter()
        .filter(|cr| !KubernetesRegistryBackend::is_terminating(cr))
        .map(|cr| {
            let phase = cr
                .status
                .as_ref()
                .map_or("", |status| status.phase.as_str());
            Survivor {
                key: entry_key(cr),
                downloading: KubernetesRegistryBackend::status_from_phase(phase)
                    == ModelStatus::DOWNLOADING,
            }
        })
        .collect()
}

struct Context {
    api: Api<ModelCacheEntry>,
    cache_dir: PathBuf,
    lease_duration: Duration,
    observations: Mutex<HashMap<String, LeaseObservation>>,
}

impl Context {
    async fn cleanup(&self, cr: &ModelCacheEntry) -> Result<Action, CleanupError> {
        let cr_name = cr.name_any();
        let status = cr.status.clone().unwrap_or_default();
        let step = {
            let mut observations = self
                .observations
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            next_step(
                &cr_name,
                &status,
                &mut observations,
                Instant::now(),
                self.lease_duration,
            )
        };
        match step {
            CleanupStep::RevokeClaim(claim_id) => {
                self.revoke_claim(&cr_name, claim_id).await?;
                return Err(CleanupError::DownloadInFlight(cr_name));
            }
            CleanupStep::WaitForWriter => return Err(CleanupError::DownloadInFlight(cr_name)),
            CleanupStep::DeleteFiles => {}
        }

        let provider = KubernetesRegistryBackend::provider_from_str(&cr.spec.provider)
            .map_err(|e| CleanupError::Invalid(format!("{cr_name}: {e}")))?;
        let key = entry_files::record_key(&cr.spec.model_name, provider);
        let survivors = survivors(&self.api.list(&ListParams::default()).await?.items);
        let scope = entry_files::deletable_scope(&key, &survivors);
        entry_files::delete_files(provider, &key.model_name, &self.cache_dir, scope)
            .await
            .map_err(|e| CleanupError::Files {
                name: cr_name.clone(),
                reason: format!("{e:#}"),
            })?;
        // Every replica gets here for the same CR, and most find the files already
        // gone. The lines state the result, not that this replica removed something.
        if scope == FileScope::Keep {
            info!(
                "Kept the cached files of '{}' for deleted ModelCacheEntry {cr_name}: another \
                 entry of the model still uses them",
                key.model_name
            );
        } else {
            info!(
                "No cached files of '{}' ({scope:?}) remain under {} for deleted \
                 ModelCacheEntry {cr_name}",
                key.model_name,
                self.cache_dir.display()
            );
        }
        self.observations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&cr_name);
        Ok(Action::await_change())
    }

    /// Clear the claim so the owner's next heartbeat and its completion both fail.
    async fn revoke_claim(&self, cr_name: &str, claim_id: &str) -> Result<(), CleanupError> {
        let patch = json!([
            { "op": "test", "path": "/status/phase", "value": phase::DOWNLOADING },
            { "op": "test", "path": "/status/claimId", "value": claim_id },
            { "op": "add", "path": "/status/claimId", "value": null },
            { "op": "add", "path": "/status/message", "value": "Deleting cache entry..." },
        ]);
        let patch = serde_json::from_value(patch)
            .map_err(|e| CleanupError::Invalid(format!("revoke patch for {cr_name}: {e}")))?;
        match self
            .api
            .patch_status(cr_name, &PatchParams::default(), &Patch::<()>::Json(patch))
            .await
        {
            Ok(_) => {
                info!("Stopping the download of deleted ModelCacheEntry {cr_name}");
                Ok(())
            }
            // The claim changed or the CR is gone. The next reconcile reads it again.
            Err(kube::Error::Api(e)) if matches!(e.code, 404 | 409 | 422) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

async fn reconcile(
    cr: Arc<ModelCacheEntry>,
    ctx: Arc<Context>,
) -> Result<Action, kube_finalizer::Error<CleanupError>> {
    kube_finalizer::finalizer(&ctx.api, CACHE_FILES_FINALIZER, cr, |event| async {
        match event {
            Event::Apply(_) => Ok(Action::await_change()),
            Event::Cleanup(cr) => ctx.cleanup(&cr).await,
        }
    })
    .await
}

fn error_policy(
    cr: Arc<ModelCacheEntry>,
    error: &kube_finalizer::Error<CleanupError>,
    _ctx: Arc<Context>,
) -> Action {
    match error {
        kube_finalizer::Error::CleanupFailed(CleanupError::DownloadInFlight(_)) => {
            debug!("{error}");
            Action::requeue(WRITER_POLL_INTERVAL)
        }
        // Another replica changed the finalizers first. The resulting watch event, or
        // this requeue, reconciles the CR again.
        kube_finalizer::Error::AddFinalizer(kube::Error::Api(e))
        | kube_finalizer::Error::RemoveFinalizer(kube::Error::Api(e))
            if matches!(e.code, 404 | 409 | 422) =>
        {
            debug!("ModelCacheEntry {}: {error}", cr.name_any());
            Action::requeue(Duration::from_secs(1))
        }
        _ => {
            warn!("ModelCacheEntry {}: {error}", cr.name_any());
            Action::requeue(RETRY_INTERVAL)
        }
    }
}

/// `lease_duration` is the download lease a revoked owner may still write for.
pub async fn run(
    client: Client,
    namespace: String,
    cache_dir: PathBuf,
    lease_duration: Duration,
    shutdown: oneshot::Receiver<()>,
) {
    let api: Api<ModelCacheEntry> = Api::namespaced(client, &namespace);
    let ctx = Arc::new(Context {
        api: api.clone(),
        cache_dir,
        lease_duration,
        observations: Mutex::new(HashMap::new()),
    });
    info!("ModelCacheEntry finalizer started (namespace '{namespace}')");
    Controller::new(api, watcher::Config::default())
        .graceful_shutdown_on(async move {
            let _ = shutdown.await;
        })
        .run(reconcile, error_policy, ctx)
        .for_each(|result| async move {
            match result {
                Ok(_) | Err(controller::Error::ReconcilerFailed(..)) => {}
                Err(controller::Error::ObjectNotFound(object)) => {
                    debug!("ModelCacheEntry {object} was deleted before its requeue");
                }
                // The message of these errors names no cause. The cause is in the chain.
                Err(e) => warn!("ModelCacheEntry finalizer: {:#}", anyhow::Error::from(e)),
            }
        })
        .await;
    info!("ModelCacheEntry finalizer stopped");
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::fake_api::{self, status};
    use super::*;
    use http::{Method, StatusCode};
    use std::path::Path;
    use tempfile::TempDir;

    const LEASE: Duration = Duration::from_secs(30);
    const CR_NAME: &str = "mx-cache-entry";

    fn context(client: Client, cache_dir: &Path) -> Arc<Context> {
        Arc::new(Context {
            api: Api::namespaced(client, fake_api::NAMESPACE),
            cache_dir: cache_dir.to_path_buf(),
            lease_duration: LEASE,
            observations: Mutex::new(HashMap::new()),
        })
    }

    /// A marked Hugging Face CR named [`CR_NAME`].
    fn marked_entry(model_name: &str, status: Option<ModelCacheEntryStatus>) -> ModelCacheEntry {
        fake_api::marked(fake_api::entry(CR_NAME, model_name, "HuggingFace", status))
    }

    fn pinned(revision: &str) -> String {
        EntryKey::new("org/model", Some(revision.to_string()), false).to_string()
    }

    /// A cache with one model that has the snapshots `abc` and `def`.
    fn cache_with_two_snapshots() -> (TempDir, std::path::PathBuf) {
        let cache = TempDir::new().expect("cache dir");
        let repo = cache.path().join("models--org--model");
        for revision in ["abc", "def"] {
            std::fs::create_dir_all(repo.join("snapshots").join(revision)).expect("snapshot");
        }
        (cache, repo)
    }

    #[test]
    fn finished_entries_delete_their_files_at_once() {
        let mut observations = HashMap::new();
        for phase in [phase::DOWNLOADED, phase::ERROR] {
            let status = status(phase, None, None);
            assert_eq!(
                next_step("cr", &status, &mut observations, Instant::now(), LEASE),
                CleanupStep::DeleteFiles
            );
        }
    }

    /// A CR with no status reads as DOWNLOADING: its creator can have a claim patch in
    /// flight, so it waits like every other download with no known owner.
    #[test]
    fn an_entry_that_never_got_a_status_waits_one_lease_duration() {
        let mut observations = HashMap::new();
        let status = ModelCacheEntryStatus::default();
        let start = Instant::now();
        assert_eq!(
            next_step("cr", &status, &mut observations, start, LEASE),
            CleanupStep::WaitForWriter
        );
        assert_eq!(
            next_step("cr", &status, &mut observations, start + LEASE, LEASE),
            CleanupStep::DeleteFiles
        );
    }

    #[test]
    fn a_claimed_download_is_revoked_first() {
        let mut observations = HashMap::new();
        let status = status(phase::DOWNLOADING, Some("owner"), Some("t1"));
        assert_eq!(
            next_step("cr", &status, &mut observations, Instant::now(), LEASE),
            CleanupStep::RevokeClaim("owner")
        );
    }

    #[test]
    fn a_revoked_download_waits_one_lease_duration() {
        let mut observations = HashMap::new();
        let status = status(phase::DOWNLOADING, None, Some("t1"));
        let start = Instant::now();
        assert_eq!(
            next_step("cr", &status, &mut observations, start, LEASE),
            CleanupStep::WaitForWriter
        );
        assert_eq!(
            next_step(
                "cr",
                &status,
                &mut observations,
                start + LEASE - Duration::from_secs(1),
                LEASE
            ),
            CleanupStep::WaitForWriter
        );
        assert_eq!(
            next_step("cr", &status, &mut observations, start + LEASE, LEASE),
            CleanupStep::DeleteFiles
        );
    }

    #[test]
    fn survivors_skip_marked_crs_and_carry_the_download_state() {
        let downloaded = fake_api::entry(
            "a",
            "org/model",
            "HuggingFace",
            Some(status(phase::DOWNLOADED, None, None)),
        );
        let downloading = fake_api::entry(
            "b",
            &pinned("def"),
            "HuggingFace",
            Some(status(phase::DOWNLOADING, Some("owner"), Some("t1"))),
        );
        let never_claimed = fake_api::entry("c", "org/new", "HuggingFace", None);
        let marked = fake_api::marked(fake_api::entry("d", "org/other", "HuggingFace", None));

        assert_eq!(
            survivors(&[downloaded, downloading, never_claimed, marked]),
            vec![
                Survivor {
                    key: EntryKey::parse("org/model"),
                    downloading: false,
                },
                Survivor {
                    key: EntryKey::parse(&pinned("def")),
                    downloading: true,
                },
                Survivor {
                    key: EntryKey::parse("org/new"),
                    downloading: true,
                },
            ]
        );
    }

    /// `finalizer()` removes the finalizer on any `Ok`. While a download can still
    /// write, the error is the only thing that keeps the CR and its files.
    #[tokio::test]
    async fn a_claimed_download_is_revoked_and_the_cr_keeps_its_finalizer() {
        let (cache, repo) = cache_with_two_snapshots();
        let cr = marked_entry(
            "org/model",
            Some(status(phase::DOWNLOADING, Some("owner"), Some("t1"))),
        );
        let (client, calls) = fake_api::client({
            let cr = cr.clone();
            move |_| fake_api::ok(&cr)
        });

        let result = reconcile(Arc::new(cr), context(client, cache.path())).await;
        assert!(
            matches!(
                result,
                Err(kube_finalizer::Error::CleanupFailed(
                    CleanupError::DownloadInFlight(_)
                ))
            ),
            "{result:?}"
        );

        let calls = calls.all();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert!(calls[0].is(Method::PATCH, "/mx-cache-entry/status"));
        assert_eq!(
            calls[0].body,
            json!([
                { "op": "test", "path": "/status/phase", "value": phase::DOWNLOADING },
                { "op": "test", "path": "/status/claimId", "value": "owner" },
                { "op": "add", "path": "/status/claimId", "value": null },
                { "op": "add", "path": "/status/message", "value": "Deleting cache entry..." },
            ])
        );
        assert!(repo.exists(), "the owner can still write");
    }

    #[tokio::test]
    async fn a_revoked_download_keeps_cr_and_files_until_the_lease_ends() {
        let (cache, repo) = cache_with_two_snapshots();
        let cr = marked_entry(
            "org/model",
            Some(status(phase::DOWNLOADING, None, Some("t1"))),
        );
        let (client, calls) = fake_api::client(|_| fake_api::failure(StatusCode::NOT_FOUND));

        let result = reconcile(Arc::new(cr), context(client, cache.path())).await;
        assert!(
            matches!(
                result,
                Err(kube_finalizer::Error::CleanupFailed(
                    CleanupError::DownloadInFlight(_)
                ))
            ),
            "{result:?}"
        );
        assert!(calls.all().is_empty(), "{:?}", calls.all());
        assert!(repo.exists());
    }

    /// The claim can change, or the CR can go, between the read and the patch. The next
    /// reconcile reads the CR again, so these answers are not failures.
    #[tokio::test]
    async fn a_lost_revoke_patch_is_not_an_error() {
        let cache = TempDir::new().expect("cache dir");
        for code in [
            StatusCode::NOT_FOUND,
            StatusCode::CONFLICT,
            StatusCode::UNPROCESSABLE_ENTITY,
        ] {
            let (client, _calls) = fake_api::client(move |_| fake_api::failure(code));
            let ctx = context(client, cache.path());
            ctx.revoke_claim(CR_NAME, "owner").await.expect("tolerated");
        }

        let (client, _calls) =
            fake_api::client(|_| fake_api::failure(StatusCode::INTERNAL_SERVER_ERROR));
        let ctx = context(client, cache.path());
        assert!(matches!(
            ctx.revoke_claim(CR_NAME, "owner").await,
            Err(CleanupError::Kube(_))
        ));
    }

    #[tokio::test]
    async fn a_finished_entry_loses_its_files_and_then_its_finalizer() {
        let (cache, repo) = cache_with_two_snapshots();
        let cr = marked_entry("org/model", Some(status(phase::DOWNLOADED, None, None)));
        let (client, calls) = fake_api::client({
            let cr = cr.clone();
            move |call| {
                if call.method == Method::GET {
                    fake_api::list(std::slice::from_ref(&cr))
                } else {
                    fake_api::ok(&cr)
                }
            }
        });

        reconcile(Arc::new(cr), context(client, cache.path()))
            .await
            .expect("cleanup");

        assert!(
            !repo.exists(),
            "the last entry of a model takes all its files"
        );
        let calls = calls.all();
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert!(calls[0].is(Method::GET, "/modelcacheentries"));
        assert!(calls[1].is(Method::PATCH, "/mx-cache-entry"));
        assert_eq!(
            calls[1].body,
            json!([
                { "op": "test", "path": "/metadata/finalizers/0", "value": CACHE_FILES_FINALIZER },
                { "op": "remove", "path": "/metadata/finalizers/0" },
            ])
        );
    }

    #[tokio::test]
    async fn only_the_snapshot_of_the_deleted_revision_goes() {
        let (cache, repo) = cache_with_two_snapshots();
        let cr = marked_entry(&pinned("abc"), Some(status(phase::DOWNLOADED, None, None)));
        let sibling = fake_api::entry(
            "sibling",
            &pinned("def"),
            "HuggingFace",
            Some(status(phase::DOWNLOADED, None, None)),
        );
        let (client, _calls) = fake_api::client({
            let items = vec![cr.clone(), sibling];
            move |_| fake_api::list(&items)
        });

        context(client, cache.path())
            .cleanup(&cr)
            .await
            .expect("cleanup");
        assert!(!repo.join("snapshots/abc").exists());
        assert!(repo.join("snapshots/def").exists());
    }

    /// A download in flight for another revision has blobs that no snapshot links yet.
    /// A revision delete would reclaim them and make that download fail.
    #[tokio::test]
    async fn a_sibling_download_in_flight_keeps_all_files() {
        let (cache, repo) = cache_with_two_snapshots();
        let partial_blob = repo.join("blobs/etag.sync.part");
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs dir");
        std::fs::write(&partial_blob, b"partial").expect("partial blob");

        let cr = marked_entry(&pinned("abc"), Some(status(phase::DOWNLOADED, None, None)));
        let sibling = fake_api::entry(
            "sibling",
            &pinned("ghi"),
            "HuggingFace",
            Some(status(phase::DOWNLOADING, Some("owner"), Some("t1"))),
        );
        let (client, _calls) = fake_api::client({
            let items = vec![cr.clone(), sibling];
            move |_| fake_api::list(&items)
        });

        context(client, cache.path())
            .cleanup(&cr)
            .await
            .expect("cleanup");
        assert!(repo.join("snapshots/abc").exists());
        assert!(partial_blob.exists());
    }

    /// `spec.modelName` is free text for anyone who can create a CR. A revision that
    /// is not a commit identifier must not move the delete out of the snapshots.
    #[tokio::test]
    async fn a_crafted_revision_deletes_nothing_and_releases_the_cr() {
        let (cache, repo) = cache_with_two_snapshots();
        let neighbor = cache.path().join("models--org--neighbor");
        std::fs::create_dir_all(&neighbor).expect("neighbor model");

        let cr = marked_entry("mx1:../..:-:org/model", None);
        let sibling = fake_api::entry(
            "sibling",
            &pinned("def"),
            "HuggingFace",
            Some(status(phase::DOWNLOADED, None, None)),
        );
        let (client, _calls) = fake_api::client({
            let items = vec![cr.clone(), sibling];
            move |_| fake_api::list(&items)
        });
        let ctx = context(client, cache.path());
        // A CR with no status waits one lease duration. Its second look is after that.
        ctx.observations.lock().expect("observations lock").insert(
            CR_NAME.to_string(),
            LeaseObservation {
                fingerprint: LeaseFingerprint {
                    claim_id: None,
                    lease_expires_at: None,
                },
                observed_at: Instant::now()
                    .checked_sub(LEASE)
                    .expect("a monotonic clock older than the lease"),
            },
        );

        ctx.cleanup(&cr).await.expect("the CR is released");
        assert!(cache.path().exists());
        assert!(neighbor.exists());
        assert!(repo.join("snapshots/abc").exists());
        assert!(repo.join("snapshots/def").exists());
    }

    #[tokio::test]
    async fn a_failed_file_delete_keeps_the_finalizer() {
        let cache = TempDir::new().expect("cache dir");
        // A regular file where the repository directory belongs makes the delete fail.
        std::fs::write(cache.path().join("models--org--model"), b"").expect("blocking file");
        let cr = marked_entry("org/model", Some(status(phase::ERROR, None, None)));
        let (client, calls) = fake_api::client({
            let cr = cr.clone();
            move |_| fake_api::list(std::slice::from_ref(&cr))
        });

        let result = reconcile(Arc::new(cr), context(client, cache.path())).await;
        assert!(
            matches!(
                result,
                Err(kube_finalizer::Error::CleanupFailed(
                    CleanupError::Files { .. }
                ))
            ),
            "{result:?}"
        );
        assert!(calls.matching(Method::PATCH, CR_NAME).is_empty());
    }

    #[tokio::test]
    async fn a_cr_without_the_finalizer_gets_it() {
        let cache = TempDir::new().expect("cache dir");
        let mut cr = fake_api::entry(CR_NAME, "org/model", "HuggingFace", None);
        cr.metadata.finalizers = None;
        let (client, calls) = fake_api::client({
            let cr = cr.clone();
            move |_| fake_api::ok(&cr)
        });

        reconcile(Arc::new(cr), context(client, cache.path()))
            .await
            .expect("apply");
        let calls = calls.all();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert!(calls[0].is(Method::PATCH, "/mx-cache-entry"));
        assert_eq!(
            calls[0].body,
            json!([
                { "op": "test", "path": "/metadata/finalizers", "value": null },
                { "op": "add", "path": "/metadata/finalizers", "value": [CACHE_FILES_FINALIZER] },
            ])
        );
    }

    #[tokio::test]
    async fn error_policy_separates_waits_races_and_failures() {
        let cache = TempDir::new().expect("cache dir");
        let (client, _calls) = fake_api::client(|_| fake_api::failure(StatusCode::NOT_FOUND));
        let ctx = context(client, cache.path());
        let cr = Arc::new(marked_entry("org/model", None));
        let api_error = |code: u16| {
            kube::Error::Api(kube::core::ErrorResponse {
                status: "Failure".to_string(),
                message: String::new(),
                reason: String::new(),
                code,
            })
        };

        let waiting =
            kube_finalizer::Error::CleanupFailed(CleanupError::DownloadInFlight("cr".to_string()));
        assert_eq!(
            error_policy(cr.clone(), &waiting, ctx.clone()),
            Action::requeue(WRITER_POLL_INTERVAL)
        );

        for code in [404, 409, 422] {
            for raced in [
                kube_finalizer::Error::AddFinalizer(api_error(code)),
                kube_finalizer::Error::RemoveFinalizer(api_error(code)),
            ] {
                assert_eq!(
                    error_policy(cr.clone(), &raced, ctx.clone()),
                    Action::requeue(Duration::from_secs(1))
                );
            }
        }

        for failed in [
            kube_finalizer::Error::RemoveFinalizer(api_error(403)),
            kube_finalizer::Error::CleanupFailed(CleanupError::Files {
                name: "cr".to_string(),
                reason: "denied".to_string(),
            }),
        ] {
            assert_eq!(
                error_policy(cr.clone(), &failed, ctx.clone()),
                Action::requeue(RETRY_INTERVAL)
            );
        }
    }

    #[test]
    fn new_crs_carry_the_finalizer() {
        let cr = KubernetesRegistryBackend::new_cr("mx-cache-x", "org/model", "HuggingFace");
        assert_eq!(cr.finalizers(), [CACHE_FILES_FINALIZER.to_string()]);
        assert!(!KubernetesRegistryBackend::is_terminating(&cr));
    }
}
