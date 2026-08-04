// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use modelexpress_common::grpc::revision::revision_catalog_service_server::RevisionCatalogService;
use modelexpress_common::grpc::revision::{
    CommitVersionRequest, CommitVersionResponse, GetRecoveryCandidatesRequest,
    GetRecoveryCandidatesResponse, GetRevisionRequest, GetRevisionResponse,
    ListReadyRevisionsRequest, ListReadyRevisionsResponse, PublicationMode, PublishRevisionRequest,
    PublishRevisionResponse, ReceiverRevisionState, RecoveryCandidate, RecoveryCandidateKind,
    RevisionRecord, RevisionSummary, UpdateReceiverStateRequest, UpdateReceiverStateResponse,
};
use sha2::{Digest, Sha256};
use tonic::{Request, Response, Status};

use super::state::{CatalogError, RevisionCatalogState};

const DEFAULT_PAGE_SIZE: usize = 100;
const MAX_PAGE_SIZE: usize = 1000;
const READY_PAGE_TOKEN_PREFIX: &str = "mx-ready-page-v1:";
const RECOVERY_PAGE_TOKEN_PREFIX: &str = "mx-recovery-page-v1:";

#[derive(Debug, Clone, Copy)]
enum InputError {
    Missing(&'static str),
    InvalidPageToken,
}

impl From<InputError> for Status {
    fn from(error: InputError) -> Self {
        match error {
            InputError::Missing(field) => Status::invalid_argument(format!("{field} is required")),
            InputError::InvalidPageToken => Status::invalid_argument("invalid page_token"),
        }
    }
}

#[derive(Clone)]
pub struct RevisionCatalogServiceImpl {
    state: Arc<RevisionCatalogState>,
}

impl RevisionCatalogServiceImpl {
    #[must_use]
    pub fn new(state: Arc<RevisionCatalogState>) -> Self {
        Self { state }
    }
}

fn now_unix_ms() -> u64 {
    chrono::Utc::now().timestamp_millis().max(0) as u64
}

fn require_text(value: &str, field: &'static str) -> Result<(), InputError> {
    if value.is_empty() {
        Err(InputError::Missing(field))
    } else {
        Ok(())
    }
}

fn page_size(limit: u32) -> usize {
    if limit == 0 {
        DEFAULT_PAGE_SIZE
    } else {
        (limit as usize).min(MAX_PAGE_SIZE)
    }
}

fn model_token(model_id: &str) -> String {
    Sha256::digest(model_id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn decode_ready_page_token(
    token: Option<&str>,
    model_id: &str,
) -> Result<Option<String>, InputError> {
    let Some(token) = token.filter(|token| !token.is_empty()) else {
        return Ok(None);
    };
    let decoded = URL_SAFE_NO_PAD
        .decode(token)
        .map_err(|_| InputError::InvalidPageToken)?;
    let text = std::str::from_utf8(&decoded).map_err(|_| InputError::InvalidPageToken)?;
    let suffix = text
        .strip_prefix(READY_PAGE_TOKEN_PREFIX)
        .ok_or(InputError::InvalidPageToken)?;
    let (stored_model, version) = suffix.split_once(':').ok_or(InputError::InvalidPageToken)?;
    if stored_model != model_token(model_id) || version.is_empty() {
        return Err(InputError::InvalidPageToken);
    }
    Ok(Some(version.to_string()))
}

fn paginate_ready(
    mut values: Vec<RevisionSummary>,
    model_id: &str,
    token: Option<&str>,
    limit: u32,
) -> Result<(Vec<RevisionSummary>, String), InputError> {
    values.sort_by(|left, right| left.version.cmp(&right.version));
    let after = decode_ready_page_token(token, model_id)?;
    let start = after.as_ref().map_or(0, |after| {
        values.partition_point(|summary| summary.version <= *after)
    });
    let end = start.saturating_add(page_size(limit)).min(values.len());
    let page = values[start..end].to_vec();
    let next = if end < values.len() {
        let version = &page.last().ok_or(InputError::InvalidPageToken)?.version;
        URL_SAFE_NO_PAD.encode(format!(
            "{READY_PAGE_TOKEN_PREFIX}{}:{version}",
            model_token(model_id)
        ))
    } else {
        String::new()
    };
    Ok((page, next))
}

fn recovery_query_scope(
    model_id: &str,
    installed_version: Option<&str>,
    target_version: &str,
    max_delta_replay_length: Option<u32>,
) -> String {
    fn update_field(hasher: &mut Sha256, value: &[u8]) {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value);
    }

    let mut hasher = Sha256::new();
    update_field(&mut hasher, model_id.as_bytes());
    match installed_version {
        Some(version) => {
            hasher.update([1]);
            update_field(&mut hasher, version.as_bytes());
        }
        None => hasher.update([0]),
    }
    update_field(&mut hasher, target_version.as_bytes());
    match max_delta_replay_length {
        Some(limit) => {
            hasher.update([1]);
            hasher.update(limit.to_be_bytes());
        }
        None => hasher.update([0]),
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn recovery_candidate_key(candidate: &RecoveryCandidate) -> Vec<u8> {
    let mut key = candidate.kind.to_be_bytes().to_vec();
    for revision in &candidate.revisions {
        let version = revision
            .manifest
            .as_ref()
            .map_or(&[][..], |manifest| manifest.version.as_bytes());
        key.extend_from_slice(&(version.len() as u64).to_be_bytes());
        key.extend_from_slice(version);
    }
    key
}

fn recovery_candidate_key_is_valid(key: &[u8]) -> bool {
    let Some(kind_bytes) = key.get(..4).and_then(|bytes| bytes.try_into().ok()) else {
        return false;
    };
    let kind = i32::from_be_bytes(kind_bytes);
    match RecoveryCandidateKind::try_from(kind) {
        Ok(RecoveryCandidateKind::Unspecified) | Err(_) => return false,
        Ok(_) => {}
    }

    let mut offset = 4usize;
    let mut versions = 0usize;
    while offset < key.len() {
        let Some(length_end) = offset.checked_add(8) else {
            return false;
        };
        let Some(length_bytes) = key
            .get(offset..length_end)
            .and_then(|bytes| bytes.try_into().ok())
        else {
            return false;
        };
        let Ok(length) = usize::try_from(u64::from_be_bytes(length_bytes)) else {
            return false;
        };
        if length == 0 {
            return false;
        }
        let Some(version_end) = length_end.checked_add(length) else {
            return false;
        };
        let Some(version) = key.get(length_end..version_end) else {
            return false;
        };
        if std::str::from_utf8(version).is_err() {
            return false;
        }
        versions = versions.saturating_add(1);
        offset = version_end;
    }
    versions > 0
}

fn decode_recovery_page_token(
    token: Option<&str>,
    scope: &str,
) -> Result<Option<Vec<u8>>, InputError> {
    let Some(token) = token.filter(|token| !token.is_empty()) else {
        return Ok(None);
    };
    let decoded = URL_SAFE_NO_PAD
        .decode(token)
        .map_err(|_| InputError::InvalidPageToken)?;
    let text = std::str::from_utf8(&decoded).map_err(|_| InputError::InvalidPageToken)?;
    let suffix = text
        .strip_prefix(RECOVERY_PAGE_TOKEN_PREFIX)
        .ok_or(InputError::InvalidPageToken)?;
    let (stored_scope, encoded_cursor) =
        suffix.split_once(':').ok_or(InputError::InvalidPageToken)?;
    if stored_scope != scope || encoded_cursor.is_empty() {
        return Err(InputError::InvalidPageToken);
    }
    let cursor = URL_SAFE_NO_PAD
        .decode(encoded_cursor)
        .map_err(|_| InputError::InvalidPageToken)?;
    if !recovery_candidate_key_is_valid(&cursor) {
        return Err(InputError::InvalidPageToken);
    }
    Ok(Some(cursor))
}

fn paginate_recovery(
    candidates: Vec<RecoveryCandidate>,
    request: &GetRecoveryCandidatesRequest,
) -> Result<(Vec<RecoveryCandidate>, String), InputError> {
    let scope = recovery_query_scope(
        &request.model_id,
        request.installed_version.as_deref(),
        &request.target_version,
        request.max_delta_replay_length,
    );
    let after = decode_recovery_page_token(request.page_token.as_deref(), &scope)?;
    let mut keyed: Vec<_> = candidates
        .into_iter()
        .map(|candidate| (recovery_candidate_key(&candidate), candidate))
        .collect();
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    let start = after
        .as_ref()
        .map_or(0, |after| keyed.partition_point(|(key, _)| key <= after));
    let end = start
        .saturating_add(page_size(request.limit))
        .min(keyed.len());
    let page: Vec<_> = keyed[start..end]
        .iter()
        .map(|(_, candidate)| candidate.clone())
        .collect();
    let next = if end < keyed.len() {
        let cursor = &keyed
            .get(end.saturating_sub(1))
            .ok_or(InputError::InvalidPageToken)?
            .0;
        URL_SAFE_NO_PAD.encode(format!(
            "{RECOVERY_PAGE_TOKEN_PREFIX}{scope}:{}",
            URL_SAFE_NO_PAD.encode(cursor)
        ))
    } else {
        String::new()
    };
    Ok((page, next))
}

fn map_catalog_error(error: CatalogError) -> Status {
    match error {
        CatalogError::InvalidManifest(_) | CatalogError::InvalidReceiverState => {
            Status::invalid_argument(error.to_string())
        }
        CatalogError::ManifestConflict { .. } => Status::already_exists(error.to_string()),
        CatalogError::RevisionNotFound { .. } => Status::not_found(error.to_string()),
        CatalogError::InvalidLifecycle { .. } => Status::failed_precondition(error.to_string()),
        CatalogError::Backend(detail) => {
            tracing::error!(error = %detail, "revision catalog backend failure");
            Status::internal("revision catalog backend failure")
        }
    }
}

fn summary(record: &RevisionRecord) -> Option<RevisionSummary> {
    let manifest = record.manifest.as_ref()?;
    Some(RevisionSummary {
        model_id: manifest.model_id.clone(),
        version: manifest.version.clone(),
        state: record.state,
        ready_at_unix_ms: record.created_at_unix_ms,
    })
}

#[tonic::async_trait]
impl RevisionCatalogService for RevisionCatalogServiceImpl {
    async fn publish_revision(
        &self,
        request: Request<PublishRevisionRequest>,
    ) -> Result<Response<PublishRevisionResponse>, Status> {
        let request = request.into_inner();
        require_text(&request.publisher_id, "publisher_id")?;
        // BLOCK/ASYNC govern the publisher's post-READY waiting policy. The catalog RPC
        // always performs the same atomic publication and only validates the wire value.
        if let Some(mode) = request.publication_mode {
            PublicationMode::try_from(mode)
                .map_err(|_| Status::invalid_argument("invalid publication_mode"))?;
        }
        let manifest = request
            .manifest
            .ok_or_else(|| Status::invalid_argument("manifest is required"))?;
        let result = self
            .state
            .publish(manifest, now_unix_ms())
            .await
            .map_err(map_catalog_error)?;
        Ok(Response::new(PublishRevisionResponse {
            revision: Some(result.record),
            created: result.created,
        }))
    }

    async fn get_revision(
        &self,
        request: Request<GetRevisionRequest>,
    ) -> Result<Response<GetRevisionResponse>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        require_text(&request.version, "version")?;
        let revision = self
            .state
            .get(&request.model_id, &request.version)
            .await
            .map_err(map_catalog_error)?
            .ok_or_else(|| Status::not_found("revision was not found"))?;
        Ok(Response::new(GetRevisionResponse {
            revision: Some(revision),
        }))
    }

    async fn list_ready_revisions(
        &self,
        request: Request<ListReadyRevisionsRequest>,
    ) -> Result<Response<ListReadyRevisionsResponse>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        let revisions: Vec<_> = self
            .state
            .list_visible(&request.model_id)
            .await
            .map_err(map_catalog_error)?
            .iter()
            .filter_map(summary)
            .collect();
        let (revisions, next_page_token) = paginate_ready(
            revisions,
            &request.model_id,
            request.page_token.as_deref(),
            request.limit,
        )?;
        Ok(Response::new(ListReadyRevisionsResponse {
            revisions,
            next_page_token,
        }))
    }

    async fn get_recovery_candidates(
        &self,
        request: Request<GetRecoveryCandidatesRequest>,
    ) -> Result<Response<GetRecoveryCandidatesResponse>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        require_text(&request.target_version, "target_version")?;
        if let Some(installed_version) = request.installed_version.as_deref() {
            require_text(installed_version, "installed_version")?;
        }
        let candidates = self
            .state
            .recovery_candidates(
                &request.model_id,
                request.installed_version.as_deref(),
                &request.target_version,
                request.max_delta_replay_length,
            )
            .await
            .map_err(map_catalog_error)?;
        let (candidates, next_page_token) = paginate_recovery(candidates, &request)?;
        Ok(Response::new(GetRecoveryCandidatesResponse {
            candidates,
            next_page_token,
        }))
    }

    async fn update_receiver_state(
        &self,
        request: Request<UpdateReceiverStateRequest>,
    ) -> Result<Response<UpdateReceiverStateResponse>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        require_text(&request.version, "version")?;
        require_text(&request.receiver_id, "receiver_id")?;
        let state = ReceiverRevisionState::try_from(request.state)
            .map_err(|_| Status::invalid_argument("invalid receiver state"))?;
        let receiver = self
            .state
            .update_receiver_state(
                &request.model_id,
                &request.version,
                &request.receiver_id,
                state,
                request.installed_version,
                request.detail,
                now_unix_ms(),
            )
            .await
            .map_err(map_catalog_error)?;
        Ok(Response::new(UpdateReceiverStateResponse {
            receiver: Some(receiver),
        }))
    }

    async fn commit_version(
        &self,
        request: Request<CommitVersionRequest>,
    ) -> Result<Response<CommitVersionResponse>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        require_text(&request.version, "version")?;
        let revision = self
            .state
            .commit(&request.model_id, &request.version, now_unix_ms())
            .await
            .map_err(map_catalog_error)?;
        Ok(Response::new(CommitVersionResponse {
            revision: Some(revision),
        }))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use modelexpress_common::grpc::revision::revision_catalog_service_server::RevisionCatalogService;
    use modelexpress_common::grpc::revision::{
        ChangeState, CommitVersionRequest, DeltaLocation, DeltaTransferMethod,
        GetRecoveryCandidatesRequest, GetRevisionRequest, ListReadyRevisionsRequest,
        PublishRevisionRequest, RankDelta, ReceiverRevisionState, RevisionLifecycleState,
        RevisionManifest, RevisionRank, S3Location, UpdateReceiverStateRequest, delta_location,
    };
    use tonic::{Code, Request};

    use super::*;
    use crate::revision::state::RevisionCatalogState;

    fn manifest(version: &str, base_version: &str, target_digest: &str) -> RevisionManifest {
        RevisionManifest {
            model_id: "model".to_string(),
            version: version.to_string(),
            base_version: Some(base_version.to_string()),
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

    fn candidate(version: &str) -> RecoveryCandidate {
        RecoveryCandidate {
            kind: RecoveryCandidateKind::DirectDelta as i32,
            revisions: vec![RevisionRecord {
                manifest: Some(manifest(version, "v0", &format!("target-{version}"))),
                ..Default::default()
            }],
        }
    }

    fn service() -> RevisionCatalogServiceImpl {
        RevisionCatalogServiceImpl::new(Arc::new(RevisionCatalogState::for_tests()))
    }

    #[tokio::test]
    async fn publish_get_report_commit_lifecycle_is_independent() {
        let service = service();
        let published = service
            .publish_revision(Request::new(PublishRevisionRequest {
                manifest: Some(manifest("v1", "v0", "target-1")),
                publisher_id: "trainer".to_string(),
                publication_mode: None,
            }))
            .await
            .expect("publish")
            .into_inner();
        assert!(published.created);
        assert_eq!(
            published.revision.as_ref().map(|record| record.state),
            Some(RevisionLifecycleState::Ready as i32)
        );

        let report = service
            .update_receiver_state(Request::new(UpdateReceiverStateRequest {
                model_id: "model".to_string(),
                version: "v1".to_string(),
                receiver_id: "rollout-0".to_string(),
                state: ReceiverRevisionState::Verified as i32,
                installed_version: Some("v1".to_string()),
                detail: "verified".to_string(),
            }))
            .await
            .expect("report")
            .into_inner();
        assert_eq!(
            report.receiver.as_ref().map(|receiver| receiver.state),
            Some(ReceiverRevisionState::Verified as i32)
        );

        let before_commit = service
            .get_revision(Request::new(GetRevisionRequest {
                model_id: "model".to_string(),
                version: "v1".to_string(),
            }))
            .await
            .expect("get")
            .into_inner();
        assert_eq!(
            before_commit.revision.as_ref().map(|record| record.state),
            Some(RevisionLifecycleState::Ready as i32)
        );

        let committed = service
            .commit_version(Request::new(CommitVersionRequest {
                model_id: "model".to_string(),
                version: "v1".to_string(),
            }))
            .await
            .expect("commit")
            .into_inner();
        assert_eq!(
            committed.revision.as_ref().map(|record| record.state),
            Some(RevisionLifecycleState::Committed as i32)
        );
    }

    #[tokio::test]
    async fn list_ready_uses_opaque_pagination_tokens() {
        let service = service();
        for (version, base, target) in [
            ("v1", "v0", "target-1"),
            ("v2", "v1", "target-2"),
            ("v3", "v2", "target-3"),
        ] {
            service
                .publish_revision(Request::new(PublishRevisionRequest {
                    manifest: Some(manifest(version, base, target)),
                    publisher_id: "trainer".to_string(),
                    publication_mode: None,
                }))
                .await
                .expect("publish");
        }

        let first = service
            .list_ready_revisions(Request::new(ListReadyRevisionsRequest {
                model_id: "model".to_string(),
                page_token: None,
                limit: 2,
            }))
            .await
            .expect("first page")
            .into_inner();
        assert_eq!(
            first
                .revisions
                .iter()
                .map(|revision| revision.version.as_str())
                .collect::<Vec<_>>(),
            vec!["v1", "v2"]
        );
        assert!(!first.next_page_token.is_empty());
        assert_ne!(first.next_page_token, "2");

        let cross_model = service
            .list_ready_revisions(Request::new(ListReadyRevisionsRequest {
                model_id: "other".to_string(),
                page_token: Some(first.next_page_token.clone()),
                limit: 2,
            }))
            .await
            .expect_err("token is model scoped");
        assert_eq!(cross_model.code(), Code::InvalidArgument);

        service
            .publish_revision(Request::new(PublishRevisionRequest {
                manifest: Some(manifest("v0", "root", "target-0")),
                publisher_id: "trainer".to_string(),
                publication_mode: None,
            }))
            .await
            .expect("insert before cursor");

        let second = service
            .list_ready_revisions(Request::new(ListReadyRevisionsRequest {
                model_id: "model".to_string(),
                page_token: Some(first.next_page_token),
                limit: 2,
            }))
            .await
            .expect("second page")
            .into_inner();
        assert_eq!(second.revisions.len(), 1);
        assert_eq!(second.revisions[0].version, "v3");
        assert!(second.next_page_token.is_empty());
    }

    #[test]
    fn recovery_pagination_is_query_scoped_and_cursor_stable() {
        assert_ne!(
            recovery_query_scope("model", None, "v3", Some(2)),
            recovery_query_scope("model", Some(""), "v3", Some(2))
        );
        let request = GetRecoveryCandidatesRequest {
            model_id: "model".to_string(),
            installed_version: Some("v0".to_string()),
            target_version: "v3".to_string(),
            max_delta_replay_length: Some(2),
            page_token: None,
            limit: 1,
        };
        let (first, token) = paginate_recovery(vec![candidate("v2"), candidate("v3")], &request)
            .expect("first page");
        assert_eq!(
            first[0].revisions[0]
                .manifest
                .as_ref()
                .expect("manifest")
                .version,
            "v2"
        );
        assert!(!token.is_empty());

        let mut wrong_query = request.clone();
        wrong_query.model_id = "other".to_string();
        wrong_query.page_token = Some(token.clone());
        assert!(paginate_recovery(vec![candidate("v2"), candidate("v3")], &wrong_query).is_err());

        let mut wrong_installed = request.clone();
        wrong_installed.installed_version = Some("other-base".to_string());
        wrong_installed.page_token = Some(token.clone());
        assert!(
            paginate_recovery(vec![candidate("v2"), candidate("v3")], &wrong_installed).is_err()
        );

        let mut wrong_target = request.clone();
        wrong_target.target_version = "other-target".to_string();
        wrong_target.page_token = Some(token.clone());
        assert!(paginate_recovery(vec![candidate("v2"), candidate("v3")], &wrong_target).is_err());

        let scope = recovery_query_scope("model", Some("v0"), "v3", Some(2));
        let malformed = URL_SAFE_NO_PAD.encode(format!(
            "{RECOVERY_PAGE_TOKEN_PREFIX}{scope}:{}",
            URL_SAFE_NO_PAD.encode([1, 2, 3])
        ));
        let mut malformed_request = request.clone();
        malformed_request.page_token = Some(malformed);
        assert!(
            paginate_recovery(vec![candidate("v2"), candidate("v3")], &malformed_request).is_err()
        );

        let mut removed_request = request.clone();
        removed_request.page_token = Some(token.clone());
        let (after_removal, _) = paginate_recovery(vec![candidate("v3")], &removed_request)
            .expect("cursor remains valid after its candidate is removed");
        assert_eq!(
            after_removal[0].revisions[0]
                .manifest
                .as_ref()
                .expect("manifest")
                .version,
            "v3"
        );

        let mut changed_limit = request.clone();
        changed_limit.max_delta_replay_length = Some(3);
        changed_limit.page_token = Some(token.clone());
        assert!(paginate_recovery(vec![candidate("v2"), candidate("v3")], &changed_limit).is_err());

        let mut second_request = request;
        second_request.page_token = Some(token);
        let (second, next) = paginate_recovery(
            vec![candidate("v1"), candidate("v2"), candidate("v3")],
            &second_request,
        )
        .expect("second page after insertion before cursor");
        assert_eq!(
            second[0].revisions[0]
                .manifest
                .as_ref()
                .expect("manifest")
                .version,
            "v3"
        );
        assert!(next.is_empty());
    }

    #[test]
    fn backend_errors_are_sanitized_at_the_grpc_boundary() {
        let status = map_catalog_error(CatalogError::Backend(
            "redis://secret-host:6379 key mx:revision:abc".to_string(),
        ));
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "revision catalog backend failure");
    }

    #[tokio::test]
    async fn invalid_and_missing_inputs_use_grpc_status_codes() {
        let service = service();
        let missing_manifest = service
            .publish_revision(Request::new(PublishRevisionRequest {
                manifest: None,
                publisher_id: "trainer".to_string(),
                publication_mode: None,
            }))
            .await
            .expect_err("missing manifest");
        assert_eq!(missing_manifest.code(), Code::InvalidArgument);

        let not_found = service
            .get_revision(Request::new(GetRevisionRequest {
                model_id: "model".to_string(),
                version: "missing".to_string(),
            }))
            .await
            .expect_err("missing revision");
        assert_eq!(not_found.code(), Code::NotFound);

        let bad_token = service
            .list_ready_revisions(Request::new(ListReadyRevisionsRequest {
                model_id: "model".to_string(),
                page_token: Some("not-a-token".to_string()),
                limit: 10,
            }))
            .await
            .expect_err("bad token");
        assert_eq!(bad_token.code(), Code::InvalidArgument);
    }
}
