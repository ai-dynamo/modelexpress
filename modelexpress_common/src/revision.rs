// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

use crate::grpc::revision::RevisionManifest;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RevisionManifestValidationError {
    #[error("model_id is required")]
    MissingModelId,
    #[error("target_version is required")]
    MissingTargetVersion,
    #[error("format_digest is required")]
    MissingFormatDigest,
    #[error("target_digest is required")]
    MissingTargetDigest,
    #[error("launch revision must not have an exact base")]
    UnexpectedLaunchBase,
    #[error("launch revision must not have a payload")]
    UnexpectedLaunchPayload,
    #[error("base_version is required")]
    MissingBaseVersion,
    #[error("base_digest is required")]
    MissingBaseDigest,
    #[error("S3 payload is required")]
    MissingPayload,
    #[error("S3 payload bucket is required")]
    MissingPayloadBucket,
    #[error("S3 payload key is required")]
    MissingPayloadKey,
    #[error("S3 payload checksum is required")]
    MissingPayloadChecksum,
    #[error("S3 payload object_version must be non-empty when present")]
    EmptyObjectVersion,
}

pub fn validate_revision_manifest(
    manifest: &RevisionManifest,
) -> Result<(), RevisionManifestValidationError> {
    require_text(
        &manifest.model_id,
        RevisionManifestValidationError::MissingModelId,
    )?;
    require_text(
        &manifest.target_version,
        RevisionManifestValidationError::MissingTargetVersion,
    )?;
    require_text(
        &manifest.format_digest,
        RevisionManifestValidationError::MissingFormatDigest,
    )?;
    require_text(
        &manifest.target_digest,
        RevisionManifestValidationError::MissingTargetDigest,
    )?;

    if manifest.target_version == "0" {
        if manifest.base_version.is_some() || manifest.base_digest.is_some() {
            return Err(RevisionManifestValidationError::UnexpectedLaunchBase);
        }
        if manifest.payload.is_some() {
            return Err(RevisionManifestValidationError::UnexpectedLaunchPayload);
        }
        return Ok(());
    }

    require_optional_text(
        manifest.base_version.as_deref(),
        RevisionManifestValidationError::MissingBaseVersion,
    )?;
    require_optional_text(
        manifest.base_digest.as_deref(),
        RevisionManifestValidationError::MissingBaseDigest,
    )?;
    let payload = manifest
        .payload
        .as_ref()
        .ok_or(RevisionManifestValidationError::MissingPayload)?;
    require_text(
        &payload.bucket,
        RevisionManifestValidationError::MissingPayloadBucket,
    )?;
    require_text(
        &payload.key,
        RevisionManifestValidationError::MissingPayloadKey,
    )?;
    require_text(
        &payload.checksum,
        RevisionManifestValidationError::MissingPayloadChecksum,
    )?;
    if payload
        .object_version
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(RevisionManifestValidationError::EmptyObjectVersion);
    }
    Ok(())
}

fn require_optional_text(
    value: Option<&str>,
    error: RevisionManifestValidationError,
) -> Result<(), RevisionManifestValidationError> {
    match value {
        Some(value) => require_text(value, error),
        None => Err(error),
    }
}

fn require_text(
    value: &str,
    error: RevisionManifestValidationError,
) -> Result<(), RevisionManifestValidationError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpc::revision::S3Object;

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

    #[test]
    fn launch_anchor_has_no_base_or_payload() {
        assert_eq!(validate_revision_manifest(&launch_manifest()), Ok(()));

        let mut with_base = launch_manifest();
        with_base.base_version = Some("previous".to_string());
        assert_eq!(
            validate_revision_manifest(&with_base),
            Err(RevisionManifestValidationError::UnexpectedLaunchBase)
        );

        let mut with_payload = launch_manifest();
        with_payload.payload = target_manifest().payload;
        assert_eq!(
            validate_revision_manifest(&with_payload),
            Err(RevisionManifestValidationError::UnexpectedLaunchPayload)
        );
    }

    #[test]
    fn required_text_rejects_whitespace_only_values() {
        let mut manifest = target_manifest();
        manifest.model_id = "   ".to_string();
        assert_eq!(
            validate_revision_manifest(&manifest),
            Err(RevisionManifestValidationError::MissingModelId)
        );

        let mut manifest = target_manifest();
        match manifest.payload.as_mut() {
            Some(payload) => payload.checksum = "\t".to_string(),
            None => panic!("target manifest must have a payload"),
        }
        assert_eq!(
            validate_revision_manifest(&manifest),
            Err(RevisionManifestValidationError::MissingPayloadChecksum)
        );

        let mut manifest = target_manifest();
        match manifest.payload.as_mut() {
            Some(payload) => payload.object_version = Some("   ".to_string()),
            None => panic!("target manifest must have a payload"),
        }
        assert_eq!(
            validate_revision_manifest(&manifest),
            Err(RevisionManifestValidationError::EmptyObjectVersion)
        );
    }

    #[test]
    fn later_revision_requires_exact_base_and_s3_payload() {
        assert_eq!(validate_revision_manifest(&target_manifest()), Ok(()));

        let mut missing_base = target_manifest();
        missing_base.base_version = None;
        assert_eq!(
            validate_revision_manifest(&missing_base),
            Err(RevisionManifestValidationError::MissingBaseVersion)
        );

        let mut missing_payload = target_manifest();
        missing_payload.payload = None;
        assert_eq!(
            validate_revision_manifest(&missing_payload),
            Err(RevisionManifestValidationError::MissingPayload)
        );
    }
}
