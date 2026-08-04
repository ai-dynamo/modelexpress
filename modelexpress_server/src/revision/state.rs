// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
#[cfg(any(test, feature = "integration-tests"))]
use std::sync::Arc;

use modelexpress_common::grpc::revision::{
    ReceiverRevisionState, ReceiverStateRecord, RecoveryCandidate, RecoveryCandidateKind,
    RevisionLifecycleState, RevisionManifest, RevisionRecord,
};
use modelexpress_common::revision::validate_revision_manifest;
use thiserror::Error;

use super::backend::{CommitOutcome, DynRevisionCatalogBackend, PublishReadyOutcome};

#[derive(Debug, Clone, PartialEq)]
pub struct PublicationResult {
    pub record: RevisionRecord,
    pub created: bool,
}

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("invalid revision manifest: {0}")]
    InvalidManifest(String),
    #[error("revision '{model_id}/{version}' already exists with a different manifest")]
    ManifestConflict { model_id: String, version: String },
    #[error("revision '{model_id}/{version}' was not found")]
    RevisionNotFound { model_id: String, version: String },
    #[error("revision '{model_id}/{version}' has invalid lifecycle state {state}")]
    InvalidLifecycle {
        model_id: String,
        version: String,
        state: i32,
    },
    #[error("receiver state must not be unspecified")]
    InvalidReceiverState,
    #[error("revision catalog backend error: {0}")]
    Backend(String),
}

#[derive(Clone)]
pub struct RevisionCatalogState {
    backend: DynRevisionCatalogBackend,
}

impl RevisionCatalogState {
    #[must_use]
    pub fn with_backend(backend: DynRevisionCatalogBackend) -> Self {
        Self { backend }
    }

    #[cfg(any(test, feature = "integration-tests"))]
    #[must_use]
    pub fn for_tests() -> Self {
        Self::with_backend(Arc::new(
            super::backend::testing::TestRevisionCatalogBackend::new(),
        ))
    }

    pub async fn connect(&self) -> Result<(), CatalogError> {
        self.backend
            .connect()
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))
    }

    pub async fn publish(
        &self,
        manifest: RevisionManifest,
        now_unix_ms: u64,
    ) -> Result<PublicationResult, CatalogError> {
        validate_revision_manifest(&manifest)
            .map_err(|error| CatalogError::InvalidManifest(error.to_string()))?;
        let model_id = manifest.model_id.clone();
        let version = manifest.version.clone();
        let record = RevisionRecord {
            manifest: Some(manifest),
            state: RevisionLifecycleState::Ready as i32,
            created_at_unix_ms: now_unix_ms,
            state_changed_at_unix_ms: now_unix_ms,
        };
        match self
            .backend
            .publish_ready(record)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))?
        {
            PublishReadyOutcome::Created(record) => Ok(PublicationResult {
                record,
                created: true,
            }),
            PublishReadyOutcome::Existing(record) => Ok(PublicationResult {
                record,
                created: false,
            }),
            PublishReadyOutcome::Conflict => {
                Err(CatalogError::ManifestConflict { model_id, version })
            }
        }
    }

    pub async fn get(
        &self,
        model_id: &str,
        version: &str,
    ) -> Result<Option<RevisionRecord>, CatalogError> {
        self.backend
            .get_revision(model_id, version)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))
    }

    pub async fn list_visible(&self, model_id: &str) -> Result<Vec<RevisionRecord>, CatalogError> {
        let records = self.list_model_revisions(model_id).await?;
        Ok(records
            .into_iter()
            .filter(|record| record.state == RevisionLifecycleState::Ready as i32)
            .collect())
    }

    async fn list_model_revisions(
        &self,
        model_id: &str,
    ) -> Result<Vec<RevisionRecord>, CatalogError> {
        let records = self
            .backend
            .list_revisions(model_id)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))?;
        Ok(records
            .into_iter()
            .filter(|record| {
                matches!(
                    RevisionLifecycleState::try_from(record.state).ok(),
                    Some(RevisionLifecycleState::Ready | RevisionLifecycleState::Committed)
                )
            })
            .collect())
    }

    pub async fn commit(
        &self,
        model_id: &str,
        version: &str,
        changed_at_unix_ms: u64,
    ) -> Result<RevisionRecord, CatalogError> {
        match self
            .backend
            .commit_revision(model_id, version, changed_at_unix_ms)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))?
        {
            CommitOutcome::Committed(record) | CommitOutcome::AlreadyCommitted(record) => {
                Ok(record)
            }
            CommitOutcome::NotFound => Err(CatalogError::RevisionNotFound {
                model_id: model_id.to_string(),
                version: version.to_string(),
            }),
            CommitOutcome::InvalidState(record) => Err(CatalogError::InvalidLifecycle {
                model_id: model_id.to_string(),
                version: version.to_string(),
                state: record.state,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn update_receiver_state(
        &self,
        model_id: &str,
        version: &str,
        receiver_id: &str,
        state: ReceiverRevisionState,
        installed_version: Option<String>,
        detail: String,
        observed_at_unix_ms: u64,
    ) -> Result<ReceiverStateRecord, CatalogError> {
        if state == ReceiverRevisionState::Unspecified {
            return Err(CatalogError::InvalidReceiverState);
        }
        if self.get(model_id, version).await?.is_none() {
            return Err(CatalogError::RevisionNotFound {
                model_id: model_id.to_string(),
                version: version.to_string(),
            });
        }
        self.backend
            .upsert_receiver_state(ReceiverStateRecord {
                model_id: model_id.to_string(),
                version: version.to_string(),
                receiver_id: receiver_id.to_string(),
                state: state as i32,
                installed_version,
                detail,
                observed_at_unix_ms,
            })
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))
    }

    pub async fn receiver_states(
        &self,
        model_id: &str,
        version: &str,
    ) -> Result<Vec<ReceiverStateRecord>, CatalogError> {
        self.backend
            .list_receiver_states(model_id, version)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))
    }

    pub async fn recovery_candidates(
        &self,
        model_id: &str,
        installed_version: Option<&str>,
        target_version: &str,
        max_delta_replay_length: Option<u32>,
    ) -> Result<Vec<RecoveryCandidate>, CatalogError> {
        let records = self.list_model_revisions(model_id).await?;
        let by_version: HashMap<_, _> = records
            .into_iter()
            .filter_map(|record| {
                record
                    .manifest
                    .as_ref()
                    .map(|manifest| (manifest.version.clone(), record.clone()))
            })
            .collect();
        let Some(target) = by_version.get(target_version).cloned() else {
            return Err(CatalogError::RevisionNotFound {
                model_id: model_id.to_string(),
                version: target_version.to_string(),
            });
        };
        let target_manifest =
            target
                .manifest
                .as_ref()
                .ok_or_else(|| CatalogError::InvalidLifecycle {
                    model_id: model_id.to_string(),
                    version: target_version.to_string(),
                    state: target.state,
                })?;

        if target_manifest.base_version.is_none() {
            return Ok(vec![RecoveryCandidate {
                kind: RecoveryCandidateKind::FullTarget as i32,
                revisions: vec![target],
            }]);
        }

        let Some(installed_version) = installed_version else {
            return Ok(Vec::new());
        };
        let max_length = max_delta_replay_length.unwrap_or(u32::MAX) as usize;
        if max_length == 0 {
            return Ok(Vec::new());
        }

        let mut reverse_chain = Vec::new();
        let mut current = target;
        let mut seen = HashSet::new();
        loop {
            let Some(manifest) = current.manifest.as_ref() else {
                return Ok(Vec::new());
            };
            if !seen.insert(manifest.version.clone()) {
                return Ok(Vec::new());
            }
            let Some(base_version) = manifest.base_version.as_deref() else {
                return Ok(Vec::new());
            };
            reverse_chain.push(current.clone());
            if reverse_chain.len() > max_length {
                return Ok(Vec::new());
            }
            if base_version == installed_version {
                if let Some(installed) = by_version.get(installed_version) {
                    let Some(installed_manifest) = installed.manifest.as_ref() else {
                        return Ok(Vec::new());
                    };
                    if manifest.base_digest.as_deref()
                        != Some(installed_manifest.target_digest.as_str())
                        || manifest.format_digest != installed_manifest.format_digest
                    {
                        return Ok(Vec::new());
                    }
                }
                break;
            }
            let Some(base) = by_version.get(base_version).cloned() else {
                return Ok(Vec::new());
            };
            let Some(base_manifest) = base.manifest.as_ref() else {
                return Ok(Vec::new());
            };
            if manifest.base_digest.as_deref() != Some(base_manifest.target_digest.as_str())
                || manifest.format_digest != base_manifest.format_digest
            {
                return Ok(Vec::new());
            }
            current = base;
        }
        reverse_chain.reverse();
        let kind = if reverse_chain.len() == 1 {
            RecoveryCandidateKind::DirectDelta
        } else {
            RecoveryCandidateKind::DeltaReplay
        };
        Ok(vec![RecoveryCandidate {
            kind: kind as i32,
            revisions: reverse_chain,
        }])
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::revision::{
        ChangeState, DeltaLocation, DeltaTransferMethod, RankDelta, ReceiverRevisionState,
        RecoveryCandidateKind, RevisionLifecycleState, RevisionManifest, RevisionRank, S3Location,
        delta_location,
    };

    fn manifest(version: &str, target_digest: &str) -> RevisionManifest {
        RevisionManifest {
            model_id: "model".to_string(),
            version: version.to_string(),
            base_version: Some("v0".to_string()),
            transfer_method: DeltaTransferMethod::Canonical as i32,
            delta_method: Some("xor".to_string()),
            compression_algorithm: Some("zstd".to_string()),
            format_digest: "format".to_string(),
            base_digest: Some("base".to_string()),
            target_digest: target_digest.to_string(),
            ranks: vec![RevisionRank {
                trainer_rank: 0,
                producer_id: "publisher-0".to_string(),
                source_layout_digest: "layout".to_string(),
                delta: Some(RankDelta {
                    change_state: ChangeState::Dirty as i32,
                    checksum: Some("deadbeef".to_string()),
                    location: Some(DeltaLocation {
                        transport: Some(delta_location::Transport::S3(S3Location {
                            bucket: "bucket".to_string(),
                            key: format!("models/model/{version}/root.json"),
                            object_version: Some("object-v1".to_string()),
                        })),
                    }),
                    delta_descriptor: None,
                }),
                shards: vec![],
            }],
        }
    }

    #[tokio::test]
    async fn immutable_publication_is_idempotent_and_rejects_conflicts() {
        let state = RevisionCatalogState::for_tests();

        let first = state
            .publish(manifest("v1", "target-1"), 100)
            .await
            .expect("first publication");
        assert!(first.created);
        assert_eq!(first.record.state, RevisionLifecycleState::Ready as i32);
        assert_eq!(first.record.created_at_unix_ms, 100);

        let retry = state
            .publish(manifest("v1", "target-1"), 999)
            .await
            .expect("identical retry");
        assert!(!retry.created);
        assert_eq!(retry.record, first.record);

        let error = state
            .publish(manifest("v1", "different-target"), 1000)
            .await
            .expect_err("conflicting retry");
        assert!(matches!(error, CatalogError::ManifestConflict { .. }));
    }

    #[tokio::test]
    async fn commit_is_an_idempotent_ready_to_committed_transition() {
        let state = RevisionCatalogState::for_tests();
        state
            .publish(manifest("v1", "target-1"), 100)
            .await
            .expect("publication");

        let committed = state.commit("model", "v1", 200).await.expect("commit");
        assert_eq!(committed.state, RevisionLifecycleState::Committed as i32);
        assert_eq!(committed.state_changed_at_unix_ms, 200);

        let retry = state
            .commit("model", "v1", 999)
            .await
            .expect("commit retry");
        assert_eq!(retry, committed);
        assert!(matches!(
            state.commit("model", "missing", 300).await,
            Err(CatalogError::RevisionNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn receiver_reports_are_idempotent_and_do_not_change_revision_lifecycle() {
        let state = RevisionCatalogState::for_tests();
        state
            .publish(manifest("v1", "target-1"), 100)
            .await
            .expect("publication");

        let report = state
            .update_receiver_state(
                "model",
                "v1",
                "rollout-0",
                ReceiverRevisionState::Verified,
                Some("v1".to_string()),
                "device verified".to_string(),
                150,
            )
            .await
            .expect("report");
        let retry = state
            .update_receiver_state(
                "model",
                "v1",
                "rollout-0",
                ReceiverRevisionState::Verified,
                Some("v1".to_string()),
                "device verified".to_string(),
                999,
            )
            .await
            .expect("report retry");
        assert_eq!(retry, report);
        assert_eq!(report.observed_at_unix_ms, 150);
        assert_eq!(
            state
                .receiver_states("model", "v1")
                .await
                .expect("receiver audit"),
            vec![report]
        );

        let revision = state
            .get("model", "v1")
            .await
            .expect("get")
            .expect("record");
        assert_eq!(revision.state, RevisionLifecycleState::Ready as i32);
    }

    #[tokio::test]
    async fn list_visible_is_deterministic_and_excludes_other_models() {
        let state = RevisionCatalogState::for_tests();
        state
            .publish(manifest("v2", "target-2"), 200)
            .await
            .expect("v2");
        state
            .publish(manifest("v1", "target-1"), 100)
            .await
            .expect("v1");
        let mut other = manifest("v1", "other-target");
        other.model_id = "other".to_string();
        state.publish(other, 50).await.expect("other");

        let records = state.list_visible("model").await.expect("list");
        let versions: Vec<_> = records
            .iter()
            .filter_map(|record| record.manifest.as_ref())
            .map(|manifest| manifest.version.as_str())
            .collect();
        assert_eq!(versions, vec!["v1", "v2"]);
    }

    #[tokio::test]
    async fn recovery_rejects_digest_mismatch_at_installed_base() {
        let state = RevisionCatalogState::for_tests();
        let mut v1 = manifest("v1", "target-1");
        v1.base_version = Some("v0".to_string());
        let mut v2 = manifest("v2", "target-2");
        v2.base_version = Some("v1".to_string());
        v2.base_digest = Some("wrong-target".to_string());
        state.publish(v1, 100).await.expect("v1");
        state.publish(v2, 200).await.expect("v2");

        assert!(
            state
                .recovery_candidates("model", Some("v1"), "v2", Some(1))
                .await
                .expect("candidates")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn recovery_rejects_a_version_chain_with_digest_mismatch() {
        let state = RevisionCatalogState::for_tests();
        let mut v1 = manifest("v1", "target-1");
        v1.base_version = Some("v0".to_string());
        let mut v2 = manifest("v2", "target-2");
        v2.base_version = Some("v1".to_string());
        v2.base_digest = Some("wrong-target".to_string());
        state.publish(v1, 100).await.expect("v1");
        state.publish(v2, 200).await.expect("v2");

        assert!(
            state
                .recovery_candidates("model", Some("v0"), "v2", Some(2))
                .await
                .expect("candidates")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn recovery_candidates_follow_exact_lineage_and_replay_limit() {
        let state = RevisionCatalogState::for_tests();
        let mut v1 = manifest("v1", "target-1");
        v1.base_version = Some("v0".to_string());
        let mut v2 = manifest("v2", "target-2");
        v2.base_version = Some("v1".to_string());
        v2.base_digest = Some("target-1".to_string());
        state.publish(v1, 100).await.expect("v1");
        state.publish(v2, 200).await.expect("v2");

        let direct = state
            .recovery_candidates("model", Some("v1"), "v2", Some(1))
            .await
            .expect("direct");
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0].kind, RecoveryCandidateKind::DirectDelta as i32);
        assert_eq!(direct[0].revisions.len(), 1);

        let replay = state
            .recovery_candidates("model", Some("v0"), "v2", Some(2))
            .await
            .expect("replay");
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].kind, RecoveryCandidateKind::DeltaReplay as i32);
        assert_eq!(replay[0].revisions.len(), 2);

        assert!(
            state
                .recovery_candidates("model", Some("v0"), "v2", Some(1))
                .await
                .expect("bounded replay")
                .is_empty()
        );
    }
}
