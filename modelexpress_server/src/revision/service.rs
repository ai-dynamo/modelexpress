// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use modelexpress_common::grpc::revision::revision_catalog_service_server::RevisionCatalogService;
use modelexpress_common::grpc::revision::{
    CommitRevisionRequest, GetRevisionRequest, PublishRevisionRequest, RevisionRecord,
};
use tonic::{Request, Response, Status};

use super::state::{CatalogError, RevisionCatalogState};

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

#[derive(Debug, Clone, Copy)]
enum InputError {
    Missing(&'static str),
}

impl From<InputError> for Status {
    fn from(error: InputError) -> Self {
        match error {
            InputError::Missing(field) => Status::invalid_argument(format!("{field} is required")),
        }
    }
}

fn require_text(value: &str, field: &'static str) -> Result<(), InputError> {
    if value.trim().is_empty() {
        Err(InputError::Missing(field))
    } else {
        Ok(())
    }
}

fn map_catalog_error(error: CatalogError) -> Status {
    match error {
        CatalogError::InvalidManifest(message) => Status::invalid_argument(message),
        CatalogError::ManifestConflict { .. } => Status::already_exists(error.to_string()),
        CatalogError::RevisionNotFound { .. } => Status::not_found(error.to_string()),
        CatalogError::InvalidLifecycle { .. } => Status::failed_precondition(error.to_string()),
        CatalogError::Backend(message) => {
            tracing::error!(error = %message, "revision catalog backend failure");
            Status::internal("revision catalog backend failure")
        }
    }
}

#[tonic::async_trait]
impl RevisionCatalogService for RevisionCatalogServiceImpl {
    async fn publish_revision(
        &self,
        request: Request<PublishRevisionRequest>,
    ) -> Result<Response<RevisionRecord>, Status> {
        let manifest = request
            .into_inner()
            .manifest
            .ok_or_else(|| Status::invalid_argument("manifest is required"))?;
        let published = self
            .state
            .publish(manifest)
            .await
            .map_err(map_catalog_error)?;
        Ok(Response::new(published.record))
    }

    async fn get_revision(
        &self,
        request: Request<GetRevisionRequest>,
    ) -> Result<Response<RevisionRecord>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        require_text(&request.target_version, "target_version")?;
        let record = self
            .state
            .get(&request.model_id, &request.target_version)
            .await
            .map_err(map_catalog_error)?
            .ok_or_else(|| {
                Status::not_found(format!(
                    "revision '{}/{}' was not found",
                    request.model_id, request.target_version
                ))
            })?;
        Ok(Response::new(record))
    }

    async fn commit_revision(
        &self,
        request: Request<CommitRevisionRequest>,
    ) -> Result<Response<RevisionRecord>, Status> {
        let request = request.into_inner();
        require_text(&request.model_id, "model_id")?;
        require_text(&request.target_version, "target_version")?;
        let record = self
            .state
            .commit(&request.model_id, &request.target_version)
            .await
            .map_err(map_catalog_error)?;
        Ok(Response::new(record))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::revision::{RevisionManifest, RevisionState};

    fn launch_manifest() -> RevisionManifest {
        RevisionManifest {
            model_id: "model".to_string(),
            target_version: "0".to_string(),
            target_digest: "sha256:target-0".to_string(),
            format_digest: "sha256:format".to_string(),
            ..Default::default()
        }
    }

    fn service() -> RevisionCatalogServiceImpl {
        RevisionCatalogServiceImpl::new(Arc::new(RevisionCatalogState::for_tests()))
    }

    #[tokio::test]
    async fn publish_get_and_commit_one_exact_revision() {
        let service = service();
        let published = service
            .publish_revision(Request::new(PublishRevisionRequest {
                manifest: Some(launch_manifest()),
            }))
            .await
            .expect("publish")
            .into_inner();
        assert_eq!(published.state, RevisionState::Ready as i32);

        let fetched = service
            .get_revision(Request::new(GetRevisionRequest {
                model_id: "model".to_string(),
                target_version: "0".to_string(),
            }))
            .await
            .expect("get")
            .into_inner();
        assert_eq!(fetched, published);

        let committed = service
            .commit_revision(Request::new(CommitRevisionRequest {
                model_id: "model".to_string(),
                target_version: "0".to_string(),
            }))
            .await
            .expect("commit")
            .into_inner();
        assert_eq!(committed.state, RevisionState::Committed as i32);
    }

    #[test]
    fn backend_errors_are_not_exposed_to_clients() {
        let status = map_catalog_error(CatalogError::Backend(
            "redis://user:secret@internal-host:6379 failed".to_string(),
        ));

        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(status.message(), "revision catalog backend failure");
        assert!(!status.message().contains("secret"));
    }

    #[tokio::test]
    async fn malformed_requests_are_rejected_at_the_service_boundary() {
        let service = service();
        let missing_manifest = service
            .publish_revision(Request::new(PublishRevisionRequest { manifest: None }))
            .await
            .expect_err("missing manifest");
        assert_eq!(missing_manifest.code(), tonic::Code::InvalidArgument);

        let missing_version = service
            .get_revision(Request::new(GetRevisionRequest {
                model_id: "model".to_string(),
                target_version: String::new(),
            }))
            .await
            .expect_err("missing target version");
        assert_eq!(missing_version.code(), tonic::Code::InvalidArgument);

        let whitespace_model = service
            .get_revision(Request::new(GetRevisionRequest {
                model_id: "   ".to_string(),
                target_version: "0".to_string(),
            }))
            .await
            .expect_err("whitespace-only model id");
        assert_eq!(whitespace_model.code(), tonic::Code::InvalidArgument);
    }
}
