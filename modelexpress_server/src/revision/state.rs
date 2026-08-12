// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(test, feature = "integration-tests"))]
use std::sync::Arc;

use modelexpress_common::grpc::revision::{RevisionManifest, RevisionRecord, RevisionState};
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
    #[error("revision '{model_id}/{target_version}' already exists with a different manifest")]
    ManifestConflict {
        model_id: String,
        target_version: String,
    },
    #[error("revision '{model_id}/{target_version}' was not found")]
    RevisionNotFound {
        model_id: String,
        target_version: String,
    },
    #[error("revision '{model_id}/{target_version}' has invalid lifecycle state {state}")]
    InvalidLifecycle {
        model_id: String,
        target_version: String,
        state: i32,
    },
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

    pub async fn publish(
        &self,
        manifest: RevisionManifest,
    ) -> Result<PublicationResult, CatalogError> {
        validate_revision_manifest(&manifest)
            .map_err(|error| CatalogError::InvalidManifest(error.to_string()))?;
        let model_id = manifest.model_id.clone();
        let target_version = manifest.target_version.clone();
        let record = RevisionRecord {
            manifest: Some(manifest),
            state: RevisionState::Ready as i32,
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
            PublishReadyOutcome::Conflict => Err(CatalogError::ManifestConflict {
                model_id,
                target_version,
            }),
        }
    }

    pub async fn get(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> Result<Option<RevisionRecord>, CatalogError> {
        self.backend
            .get_revision(model_id, target_version)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))
    }

    pub async fn commit(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> Result<RevisionRecord, CatalogError> {
        match self
            .backend
            .commit_revision(model_id, target_version)
            .await
            .map_err(|error| CatalogError::Backend(error.to_string()))?
        {
            CommitOutcome::Committed(record) | CommitOutcome::AlreadyCommitted(record) => {
                Ok(record)
            }
            CommitOutcome::NotFound => Err(CatalogError::RevisionNotFound {
                model_id: model_id.to_string(),
                target_version: target_version.to_string(),
            }),
            CommitOutcome::InvalidState(record) => Err(CatalogError::InvalidLifecycle {
                model_id: model_id.to_string(),
                target_version: target_version.to_string(),
                state: record.state,
            }),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::revision::S3Object;

    fn launch_manifest() -> RevisionManifest {
        RevisionManifest {
            model_id: "model".to_string(),
            target_version: "0".to_string(),
            target_digest: "sha256:target-0".to_string(),
            format_digest: "sha256:format".to_string(),
            ..Default::default()
        }
    }

    fn target_manifest() -> RevisionManifest {
        RevisionManifest {
            model_id: "model".to_string(),
            target_version: "1".to_string(),
            base_version: Some("0".to_string()),
            base_digest: Some("sha256:target-0".to_string()),
            target_digest: "sha256:target-1".to_string(),
            format_digest: "sha256:format".to_string(),
            payload: Some(S3Object {
                bucket: "bucket".to_string(),
                key: "model/1/index.json".to_string(),
                object_version: None,
                checksum: "crc32c:01020304".to_string(),
            }),
        }
    }

    #[tokio::test]
    async fn immutable_publication_is_idempotent_and_rejects_conflicts() {
        let state = RevisionCatalogState::for_tests();
        let first = state.publish(target_manifest()).await.expect("publish");
        assert!(first.created);
        assert_eq!(first.record.state, RevisionState::Ready as i32);

        let retry = state.publish(target_manifest()).await.expect("retry");
        assert!(!retry.created);
        assert_eq!(retry.record, first.record);

        let mut conflicting = target_manifest();
        conflicting.target_digest = "sha256:different".to_string();
        assert!(matches!(
            state.publish(conflicting).await,
            Err(CatalogError::ManifestConflict { .. })
        ));
    }

    #[tokio::test]
    async fn commit_is_an_idempotent_ready_to_committed_transition() {
        let state = RevisionCatalogState::for_tests();
        state.publish(launch_manifest()).await.expect("publish");

        let committed = state.commit("model", "0").await.expect("commit");
        assert_eq!(committed.state, RevisionState::Committed as i32);
        assert_eq!(
            state.commit("model", "0").await.expect("idempotent"),
            committed
        );
        assert!(matches!(
            state.commit("model", "missing").await,
            Err(CatalogError::RevisionNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn commit_rejects_an_unrecognized_lifecycle_state() {
        let backend = Arc::new(super::super::backend::testing::TestRevisionCatalogBackend::new());
        backend.insert(RevisionRecord {
            manifest: Some(launch_manifest()),
            state: 99,
        });
        let state = RevisionCatalogState::with_backend(backend);

        assert!(matches!(
            state.commit("model", "0").await,
            Err(CatalogError::InvalidLifecycle { state: 99, .. })
        ));
    }
}
