// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use thiserror::Error;

use crate::grpc::revision::{
    ChangeState, DeltaDescriptor, DeltaLocation, DeltaTransferMethod, RankDelta, RevisionManifest,
    RevisionRank, TensorShard, delta_location::Transport,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RevisionManifestValidationError {
    #[error("model_id is required")]
    MissingModelId,
    #[error("version is required")]
    MissingVersion,
    #[error("format_digest is required")]
    MissingFormatDigest,
    #[error("target_digest is required")]
    MissingTargetDigest,
    #[error("transfer_method is required and must be supported")]
    InvalidTransferMethod,
    #[error("at least one rank is required")]
    MissingRanks,
    #[error("trainer rank {rank} appears more than once")]
    DuplicateTrainerRank { rank: u32 },
    #[error("rank {rank} requires producer_id")]
    MissingProducerId { rank: u32 },
    #[error("rank {rank} requires source_layout_digest")]
    MissingSourceLayoutDigest { rank: u32 },
    #[error("base_version is required")]
    MissingBaseVersion,
    #[error("base_digest is required")]
    MissingBaseDigest,
    #[error("base_version and base_digest must be absent")]
    UnexpectedBase,
    #[error("delta_method is required")]
    MissingDeltaMethod,
    #[error("compression_algorithm is required")]
    MissingCompressionAlgorithm,
    #[error("delta_method and compression_algorithm must be both present or both absent")]
    IncompleteDeltaConfiguration,
    #[error("CANONICAL requires exactly one rank")]
    InvalidCanonicalRankCount,
    #[error("CANONICAL requires trainer_rank=0")]
    InvalidCanonicalTrainerRank,
    #[error("rank {rank} requires delta")]
    MissingRankDelta { rank: u32 },
    #[error("rank {rank} must not contain delta")]
    UnexpectedRankDelta { rank: u32 },
    #[error("rank {rank} must not contain shards")]
    UnexpectedShards { rank: u32 },
    #[error("rank {rank} requires at least one shard")]
    MissingShards { rank: u32 },
    #[error("rank {rank} delta has invalid change_state")]
    InvalidDeltaChangeState { rank: u32 },
    #[error("rank {rank} CLEAN delta must not contain byte references")]
    CleanDeltaHasByteReference { rank: u32 },
    #[error("rank {rank} DIRTY delta requires checksum")]
    MissingDeltaChecksum { rank: u32 },
    #[error("rank {rank} delta checksum is invalid")]
    InvalidDeltaChecksum { rank: u32 },
    #[error("rank {rank} DIRTY delta requires location")]
    MissingDeltaLocation { rank: u32 },
    #[error("rank {rank} delta location is invalid for the transfer method")]
    InvalidDeltaLocation { rank: u32 },
    #[error("rank {rank} DIRTY delta requires delta_descriptor")]
    MissingDeltaDescriptor { rank: u32 },
    #[error("rank {rank} delta descriptor is invalid")]
    InvalidDeltaDescriptor { rank: u32 },
    #[error("rank {rank} DIRTY delta has conflicting byte references")]
    ConflictingDeltaReferences { rank: u32 },
    #[error("rank {rank} shard {shard} has invalid change_state")]
    InvalidShardChangeState { rank: u32, shard: usize },
    #[error("rank {rank} shard {shard} must be DIRTY without delta configuration")]
    ShardMustBeDirty { rank: u32, shard: usize },
    #[error("rank {rank} shard {shard} requires tensor_descriptor")]
    MissingTensorDescriptor { rank: u32, shard: usize },
    #[error("rank {rank} shard {shard} has invalid tensor_descriptor")]
    InvalidTensorDescriptor { rank: u32, shard: usize },
    #[error("rank {rank} shard {shard} requires tensor_region")]
    MissingTensorRegion { rank: u32, shard: usize },
    #[error("rank {rank} shard {shard} has invalid tensor_region")]
    InvalidTensorRegion { rank: u32, shard: usize },
    #[error("rank {rank} CLEAN shard {shard} must not contain address/device_id")]
    CleanShardHasTransferReference { rank: u32, shard: usize },
    #[error("rank {rank} DIRTY shard {shard} requires address/device_id")]
    DirtyShardMissingTransferReference { rank: u32, shard: usize },
}

pub fn validate_revision_manifest(
    manifest: &RevisionManifest,
) -> Result<(), RevisionManifestValidationError> {
    require_text(
        &manifest.model_id,
        RevisionManifestValidationError::MissingModelId,
    )?;
    require_text(
        &manifest.version,
        RevisionManifestValidationError::MissingVersion,
    )?;
    require_text(
        &manifest.format_digest,
        RevisionManifestValidationError::MissingFormatDigest,
    )?;
    require_text(
        &manifest.target_digest,
        RevisionManifestValidationError::MissingTargetDigest,
    )?;

    let method = DeltaTransferMethod::try_from(manifest.transfer_method)
        .map_err(|_| RevisionManifestValidationError::InvalidTransferMethod)?;
    if method == DeltaTransferMethod::Unspecified {
        return Err(RevisionManifestValidationError::InvalidTransferMethod);
    }
    if manifest.ranks.is_empty() {
        return Err(RevisionManifestValidationError::MissingRanks);
    }

    validate_unique_rank_metadata(&manifest.ranks)?;

    match method {
        DeltaTransferMethod::Canonical => validate_canonical(manifest),
        DeltaTransferMethod::RankLocal => validate_rank_local(manifest),
        DeltaTransferMethod::P2pCpuRank => validate_p2p_cpu_rank(manifest),
        DeltaTransferMethod::P2pGpuShard => validate_p2p_gpu_shard(manifest),
        DeltaTransferMethod::Unspecified => {
            Err(RevisionManifestValidationError::InvalidTransferMethod)
        }
    }
}

fn validate_unique_rank_metadata(
    ranks: &[RevisionRank],
) -> Result<(), RevisionManifestValidationError> {
    let mut seen = HashSet::with_capacity(ranks.len());
    for rank in ranks {
        if !seen.insert(rank.trainer_rank) {
            return Err(RevisionManifestValidationError::DuplicateTrainerRank {
                rank: rank.trainer_rank,
            });
        }
        require_text(
            &rank.producer_id,
            RevisionManifestValidationError::MissingProducerId {
                rank: rank.trainer_rank,
            },
        )?;
        require_text(
            &rank.source_layout_digest,
            RevisionManifestValidationError::MissingSourceLayoutDigest {
                rank: rank.trainer_rank,
            },
        )?;
    }
    Ok(())
}

fn validate_canonical(manifest: &RevisionManifest) -> Result<(), RevisionManifestValidationError> {
    require_exact_base(manifest)?;
    require_delta_configuration(manifest)?;
    if manifest.ranks.len() != 1 {
        return Err(RevisionManifestValidationError::InvalidCanonicalRankCount);
    }
    let rank = &manifest.ranks[0];
    if rank.trainer_rank != 0 {
        return Err(RevisionManifestValidationError::InvalidCanonicalTrainerRank);
    }
    validate_cpu_rank(rank, CpuDeltaReference::CanonicalS3)
}

fn validate_rank_local(manifest: &RevisionManifest) -> Result<(), RevisionManifestValidationError> {
    require_exact_base(manifest)?;
    require_delta_configuration(manifest)?;
    for rank in &manifest.ranks {
        validate_cpu_rank(rank, CpuDeltaReference::Location)?;
    }
    Ok(())
}

fn validate_p2p_cpu_rank(
    manifest: &RevisionManifest,
) -> Result<(), RevisionManifestValidationError> {
    require_exact_base(manifest)?;
    require_delta_configuration(manifest)?;
    for rank in &manifest.ranks {
        validate_cpu_rank(rank, CpuDeltaReference::Descriptor)?;
    }
    Ok(())
}

fn validate_p2p_gpu_shard(
    manifest: &RevisionManifest,
) -> Result<(), RevisionManifestValidationError> {
    let delta_configured = optional_delta_configuration(manifest)?;
    if delta_configured {
        require_exact_base(manifest)?;
    } else if manifest.base_version.is_some() || manifest.base_digest.is_some() {
        return Err(RevisionManifestValidationError::UnexpectedBase);
    }

    for rank in &manifest.ranks {
        if rank.delta.is_some() {
            return Err(RevisionManifestValidationError::UnexpectedRankDelta {
                rank: rank.trainer_rank,
            });
        }
        if rank.shards.is_empty() {
            return Err(RevisionManifestValidationError::MissingShards {
                rank: rank.trainer_rank,
            });
        }
        for (index, shard) in rank.shards.iter().enumerate() {
            validate_tensor_shard(rank.trainer_rank, index, shard, delta_configured)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CpuDeltaReference {
    CanonicalS3,
    Location,
    Descriptor,
}

fn validate_cpu_rank(
    rank: &RevisionRank,
    required_reference: CpuDeltaReference,
) -> Result<(), RevisionManifestValidationError> {
    if !rank.shards.is_empty() {
        return Err(RevisionManifestValidationError::UnexpectedShards {
            rank: rank.trainer_rank,
        });
    }
    let delta = rank
        .delta
        .as_ref()
        .ok_or(RevisionManifestValidationError::MissingRankDelta {
            rank: rank.trainer_rank,
        })?;
    validate_rank_delta(rank.trainer_rank, delta, required_reference)
}

fn validate_rank_delta(
    rank: u32,
    delta: &RankDelta,
    required_reference: CpuDeltaReference,
) -> Result<(), RevisionManifestValidationError> {
    let state = ChangeState::try_from(delta.change_state)
        .map_err(|_| RevisionManifestValidationError::InvalidDeltaChangeState { rank })?;
    match state {
        ChangeState::Unspecified => {
            Err(RevisionManifestValidationError::InvalidDeltaChangeState { rank })
        }
        ChangeState::Clean => {
            if delta.checksum.is_some()
                || delta.location.is_some()
                || delta.delta_descriptor.is_some()
            {
                Err(RevisionManifestValidationError::CleanDeltaHasByteReference { rank })
            } else {
                Ok(())
            }
        }
        ChangeState::Dirty => {
            let checksum = delta
                .checksum
                .as_deref()
                .ok_or(RevisionManifestValidationError::MissingDeltaChecksum { rank })?;
            if !is_crc32c(checksum) {
                return Err(RevisionManifestValidationError::InvalidDeltaChecksum { rank });
            }
            match required_reference {
                CpuDeltaReference::CanonicalS3 => {
                    let location = require_location_without_descriptor(rank, delta)?;
                    if !matches!(location.transport, Some(Transport::S3(_)))
                        || !valid_delta_location(location)
                    {
                        return Err(RevisionManifestValidationError::InvalidDeltaLocation { rank });
                    }
                }
                CpuDeltaReference::Location => {
                    let location = require_location_without_descriptor(rank, delta)?;
                    if !valid_delta_location(location) {
                        return Err(RevisionManifestValidationError::InvalidDeltaLocation { rank });
                    }
                }
                CpuDeltaReference::Descriptor => {
                    if delta.location.is_some() && delta.delta_descriptor.is_some() {
                        return Err(
                            RevisionManifestValidationError::ConflictingDeltaReferences { rank },
                        );
                    }
                    if delta.location.is_some() {
                        return Err(RevisionManifestValidationError::InvalidDeltaLocation { rank });
                    }
                    let descriptor = delta
                        .delta_descriptor
                        .as_ref()
                        .ok_or(RevisionManifestValidationError::MissingDeltaDescriptor { rank })?;
                    if !valid_delta_descriptor(descriptor) {
                        return Err(RevisionManifestValidationError::InvalidDeltaDescriptor {
                            rank,
                        });
                    }
                }
            }
            Ok(())
        }
    }
}

fn require_location_without_descriptor(
    rank: u32,
    delta: &RankDelta,
) -> Result<&DeltaLocation, RevisionManifestValidationError> {
    if delta.location.is_some() && delta.delta_descriptor.is_some() {
        return Err(RevisionManifestValidationError::ConflictingDeltaReferences { rank });
    }
    if delta.delta_descriptor.is_some() {
        return Err(RevisionManifestValidationError::InvalidDeltaDescriptor { rank });
    }
    delta
        .location
        .as_ref()
        .ok_or(RevisionManifestValidationError::MissingDeltaLocation { rank })
}

fn valid_delta_location(location: &DeltaLocation) -> bool {
    match location.transport.as_ref() {
        Some(Transport::S3(location)) => {
            !location.bucket.trim().is_empty() && !location.key.trim().is_empty()
        }
        Some(Transport::Zeromq(location)) => {
            !location.endpoint.trim().is_empty() && !location.payload_id.trim().is_empty()
        }
        Some(Transport::Filesystem(location)) => !location.path.trim().is_empty(),
        None => false,
    }
}

fn valid_delta_descriptor(descriptor: &DeltaDescriptor) -> bool {
    descriptor.address != 0 && descriptor.length != 0 && !descriptor.dtype.trim().is_empty()
}

fn validate_tensor_shard(
    rank: u32,
    index: usize,
    shard: &TensorShard,
    delta_configured: bool,
) -> Result<(), RevisionManifestValidationError> {
    let state = ChangeState::try_from(shard.change_state).map_err(|_| {
        RevisionManifestValidationError::InvalidShardChangeState { rank, shard: index }
    })?;
    if state == ChangeState::Unspecified {
        return Err(RevisionManifestValidationError::InvalidShardChangeState {
            rank,
            shard: index,
        });
    }
    if !delta_configured && state != ChangeState::Dirty {
        return Err(RevisionManifestValidationError::ShardMustBeDirty { rank, shard: index });
    }

    let descriptor = shard
        .tensor_descriptor
        .as_ref()
        .ok_or(RevisionManifestValidationError::MissingTensorDescriptor { rank, shard: index })?;
    if descriptor.tensor_name.trim().is_empty()
        || descriptor.dtype.trim().is_empty()
        || descriptor.byte_size == 0
    {
        return Err(RevisionManifestValidationError::InvalidTensorDescriptor {
            rank,
            shard: index,
        });
    }

    match state {
        ChangeState::Clean => {
            if descriptor.address.is_some() || descriptor.device_id.is_some() {
                return Err(
                    RevisionManifestValidationError::CleanShardHasTransferReference {
                        rank,
                        shard: index,
                    },
                );
            }
        }
        ChangeState::Dirty => {
            if descriptor.address.is_none() || descriptor.device_id.is_none() {
                return Err(
                    RevisionManifestValidationError::DirtyShardMissingTransferReference {
                        rank,
                        shard: index,
                    },
                );
            }
        }
        ChangeState::Unspecified => unreachable!("unspecified state rejected above"),
    }

    let region = shard
        .tensor_region
        .as_ref()
        .ok_or(RevisionManifestValidationError::MissingTensorRegion { rank, shard: index })?;
    let dimensions = region.full_shape.len();
    if dimensions == 0
        || region.global_offset.len() != dimensions
        || region.region_shape.len() != dimensions
        || region.target_digest.trim().is_empty()
        || region
            .full_shape
            .iter()
            .zip(&region.global_offset)
            .zip(&region.region_shape)
            .any(|((&full, &offset), &shape)| {
                shape == 0 || offset.checked_add(shape).is_none_or(|end| end > full)
            })
    {
        return Err(RevisionManifestValidationError::InvalidTensorRegion { rank, shard: index });
    }
    Ok(())
}

fn require_exact_base(manifest: &RevisionManifest) -> Result<(), RevisionManifestValidationError> {
    require_optional_text(
        manifest.base_version.as_deref(),
        RevisionManifestValidationError::MissingBaseVersion,
    )?;
    require_optional_text(
        manifest.base_digest.as_deref(),
        RevisionManifestValidationError::MissingBaseDigest,
    )
}

fn require_delta_configuration(
    manifest: &RevisionManifest,
) -> Result<(), RevisionManifestValidationError> {
    require_optional_text(
        manifest.delta_method.as_deref(),
        RevisionManifestValidationError::MissingDeltaMethod,
    )?;
    require_optional_text(
        manifest.compression_algorithm.as_deref(),
        RevisionManifestValidationError::MissingCompressionAlgorithm,
    )
}

fn optional_delta_configuration(
    manifest: &RevisionManifest,
) -> Result<bool, RevisionManifestValidationError> {
    match (
        manifest.delta_method.as_deref(),
        manifest.compression_algorithm.as_deref(),
    ) {
        (None, None) => Ok(false),
        (Some(method), Some(compression)) => {
            require_text(method, RevisionManifestValidationError::MissingDeltaMethod)?;
            require_text(
                compression,
                RevisionManifestValidationError::MissingCompressionAlgorithm,
            )?;
            Ok(true)
        }
        _ => Err(RevisionManifestValidationError::IncompleteDeltaConfiguration),
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

fn require_optional_text(
    value: Option<&str>,
    error: RevisionManifestValidationError,
) -> Result<(), RevisionManifestValidationError> {
    match value {
        Some(value) => require_text(value, error),
        None => Err(error),
    }
}

fn is_crc32c(value: &str) -> bool {
    value.len() == 8
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpc::revision::{
        FilesystemLocation, PublicationMode, ReceiverRevisionState, RevisionLifecycleState,
        RevisionRank, S3Location, TensorDescriptor, TensorRegion, ZeroMqLocation,
    };

    fn s3_location() -> DeltaLocation {
        DeltaLocation {
            transport: Some(Transport::S3(S3Location {
                bucket: "bucket".to_string(),
                key: "models/policy/versions/1/canonical/index.json".to_string(),
                object_version: Some("version-id".to_string()),
            })),
        }
    }

    fn clean_delta() -> RankDelta {
        RankDelta {
            change_state: ChangeState::Clean as i32,
            checksum: None,
            location: None,
            delta_descriptor: None,
        }
    }

    fn dirty_location_delta(location: DeltaLocation) -> RankDelta {
        RankDelta {
            change_state: ChangeState::Dirty as i32,
            checksum: Some("a1b2c3d4".to_string()),
            location: Some(location),
            delta_descriptor: None,
        }
    }

    fn rank(rank: u32, delta: Option<RankDelta>, shards: Vec<TensorShard>) -> RevisionRank {
        RevisionRank {
            trainer_rank: rank,
            producer_id: format!("producer-{rank}"),
            source_layout_digest: "sha256:layout".to_string(),
            delta,
            shards,
        }
    }

    fn manifest(method: DeltaTransferMethod, ranks: Vec<RevisionRank>) -> RevisionManifest {
        RevisionManifest {
            model_id: "model".to_string(),
            version: "1".to_string(),
            base_version: Some("0".to_string()),
            transfer_method: method as i32,
            delta_method: Some("xor".to_string()),
            compression_algorithm: Some("zstd".to_string()),
            format_digest: "sha256:format".to_string(),
            base_digest: Some("sha256:target-0".to_string()),
            target_digest: "sha256:target-1".to_string(),
            ranks,
        }
    }

    fn tensor_shard(state: ChangeState) -> TensorShard {
        let dirty = state == ChangeState::Dirty;
        TensorShard {
            change_state: state as i32,
            tensor_descriptor: Some(TensorDescriptor {
                tensor_name: "model.layers.0.weight".to_string(),
                dtype: "bfloat16".to_string(),
                byte_size: 32,
                address: dirty.then_some(0x1000),
                device_id: dirty.then_some(0),
            }),
            tensor_region: Some(TensorRegion {
                full_shape: vec![4, 4],
                global_offset: vec![0, 0],
                region_shape: vec![4, 4],
                target_digest: "sha256:shard-target".to_string(),
            }),
        }
    }

    #[test]
    fn canonical_accepts_exactly_one_rank_zero_delta() {
        let dirty = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(0, Some(dirty_location_delta(s3_location())), vec![])],
        );
        assert_eq!(validate_revision_manifest(&dirty), Ok(()));

        let clean = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(0, Some(clean_delta()), vec![])],
        );
        assert_eq!(validate_revision_manifest(&clean), Ok(()));
    }

    #[test]
    fn canonical_rejects_wrong_rank_shape_and_non_s3_location() {
        let two_ranks = manifest(
            DeltaTransferMethod::Canonical,
            vec![
                rank(0, Some(clean_delta()), vec![]),
                rank(1, Some(clean_delta()), vec![]),
            ],
        );
        assert_eq!(
            validate_revision_manifest(&two_ranks),
            Err(RevisionManifestValidationError::InvalidCanonicalRankCount)
        );

        let wrong_rank = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(1, Some(clean_delta()), vec![])],
        );
        assert_eq!(
            validate_revision_manifest(&wrong_rank),
            Err(RevisionManifestValidationError::InvalidCanonicalTrainerRank)
        );

        let filesystem = DeltaLocation {
            transport: Some(Transport::Filesystem(FilesystemLocation {
                path: "/shared/delta".to_string(),
            })),
        };
        let wrong_location = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(0, Some(dirty_location_delta(filesystem)), vec![])],
        );
        assert_eq!(
            validate_revision_manifest(&wrong_location),
            Err(RevisionManifestValidationError::InvalidDeltaLocation { rank: 0 })
        );
    }

    #[test]
    fn cpu_delta_methods_enforce_method_shaped_references() {
        let rank_local = manifest(
            DeltaTransferMethod::RankLocal,
            vec![rank(
                0,
                Some(dirty_location_delta(DeltaLocation {
                    transport: Some(Transport::Zeromq(ZeroMqLocation {
                        endpoint: "tcp://trainer:5555".to_string(),
                        payload_id: "payload-1".to_string(),
                    })),
                })),
                vec![],
            )],
        );
        assert_eq!(validate_revision_manifest(&rank_local), Ok(()));

        let cpu_direct = manifest(
            DeltaTransferMethod::P2pCpuRank,
            vec![rank(
                0,
                Some(RankDelta {
                    change_state: ChangeState::Dirty as i32,
                    checksum: Some("a1b2c3d4".to_string()),
                    location: None,
                    delta_descriptor: Some(DeltaDescriptor {
                        address: 0x1000,
                        length: 128,
                        dtype: "uint8".to_string(),
                    }),
                }),
                vec![],
            )],
        );
        assert_eq!(validate_revision_manifest(&cpu_direct), Ok(()));
    }

    #[test]
    fn clean_delta_omits_checksum_and_transfer_references() {
        let mut invalid = clean_delta();
        invalid.checksum = Some("a1b2c3d4".to_string());
        let manifest = manifest(
            DeltaTransferMethod::RankLocal,
            vec![rank(0, Some(invalid), vec![])],
        );
        assert_eq!(
            validate_revision_manifest(&manifest),
            Err(RevisionManifestValidationError::CleanDeltaHasByteReference { rank: 0 })
        );
    }

    #[test]
    fn dirty_delta_requires_valid_checksum() {
        let mut invalid = dirty_location_delta(s3_location());
        invalid.checksum = Some("ABC".to_string());
        let manifest = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(0, Some(invalid), vec![])],
        );
        assert_eq!(
            validate_revision_manifest(&manifest),
            Err(RevisionManifestValidationError::InvalidDeltaChecksum { rank: 0 })
        );
    }

    #[test]
    fn gpu_shard_method_supports_detected_and_no_base_full_transfer() {
        let detected = manifest(
            DeltaTransferMethod::P2pGpuShard,
            vec![rank(
                0,
                None,
                vec![
                    tensor_shard(ChangeState::Clean),
                    tensor_shard(ChangeState::Dirty),
                ],
            )],
        );
        assert_eq!(validate_revision_manifest(&detected), Ok(()));

        let mut no_base = manifest(
            DeltaTransferMethod::P2pGpuShard,
            vec![rank(0, None, vec![tensor_shard(ChangeState::Dirty)])],
        );
        no_base.base_version = None;
        no_base.base_digest = None;
        no_base.delta_method = None;
        no_base.compression_algorithm = None;
        assert_eq!(validate_revision_manifest(&no_base), Ok(()));
    }

    #[test]
    fn gpu_no_base_transfer_requires_every_shard_dirty() {
        let mut manifest = manifest(
            DeltaTransferMethod::P2pGpuShard,
            vec![rank(0, None, vec![tensor_shard(ChangeState::Clean)])],
        );
        manifest.base_version = None;
        manifest.base_digest = None;
        manifest.delta_method = None;
        manifest.compression_algorithm = None;
        assert_eq!(
            validate_revision_manifest(&manifest),
            Err(RevisionManifestValidationError::ShardMustBeDirty { rank: 0, shard: 0 })
        );
    }

    #[test]
    fn gpu_clean_and_dirty_shards_enforce_transfer_reference_presence() {
        let mut clean_with_address = tensor_shard(ChangeState::Clean);
        let Some(descriptor) = clean_with_address.tensor_descriptor.as_mut() else {
            unreachable!("test helper always creates a tensor descriptor");
        };
        descriptor.address = Some(0x1000);
        let invalid_clean = manifest(
            DeltaTransferMethod::P2pGpuShard,
            vec![rank(0, None, vec![clean_with_address])],
        );
        assert_eq!(
            validate_revision_manifest(&invalid_clean),
            Err(
                RevisionManifestValidationError::CleanShardHasTransferReference {
                    rank: 0,
                    shard: 0,
                }
            )
        );

        let mut dirty_without_address = tensor_shard(ChangeState::Dirty);
        let Some(descriptor) = dirty_without_address.tensor_descriptor.as_mut() else {
            unreachable!("test helper always creates a tensor descriptor");
        };
        descriptor.address = None;
        let invalid_dirty = manifest(
            DeltaTransferMethod::P2pGpuShard,
            vec![rank(0, None, vec![dirty_without_address])],
        );
        assert_eq!(
            validate_revision_manifest(&invalid_dirty),
            Err(
                RevisionManifestValidationError::DirtyShardMissingTransferReference {
                    rank: 0,
                    shard: 0,
                }
            )
        );
    }

    #[test]
    fn exact_base_and_delta_configuration_are_required_for_cpu_methods() {
        let mut candidate = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(0, Some(clean_delta()), vec![])],
        );
        candidate.base_version = None;
        assert_eq!(
            validate_revision_manifest(&candidate),
            Err(RevisionManifestValidationError::MissingBaseVersion)
        );

        candidate = manifest(
            DeltaTransferMethod::Canonical,
            vec![rank(0, Some(clean_delta()), vec![])],
        );
        candidate.compression_algorithm = None;
        assert_eq!(
            validate_revision_manifest(&candidate),
            Err(RevisionManifestValidationError::MissingCompressionAlgorithm)
        );
    }

    #[test]
    fn manifest_rejects_unspecified_transfer_and_duplicate_ranks() {
        let unspecified = manifest(
            DeltaTransferMethod::Unspecified,
            vec![rank(0, Some(clean_delta()), vec![])],
        );
        assert_eq!(
            validate_revision_manifest(&unspecified),
            Err(RevisionManifestValidationError::InvalidTransferMethod)
        );

        let duplicate = manifest(
            DeltaTransferMethod::RankLocal,
            vec![
                rank(0, Some(clean_delta()), vec![]),
                rank(0, Some(clean_delta()), vec![]),
            ],
        );
        assert_eq!(
            validate_revision_manifest(&duplicate),
            Err(RevisionManifestValidationError::DuplicateTrainerRank { rank: 0 })
        );
    }

    #[test]
    fn lifecycle_and_receiver_states_use_safe_unspecified_defaults() {
        assert_eq!(RevisionLifecycleState::Unspecified as i32, 0);
        assert_eq!(RevisionLifecycleState::Ready as i32, 1);
        assert_eq!(RevisionLifecycleState::Committed as i32, 2);
        assert!(RevisionLifecycleState::try_from(3).is_err());

        assert_eq!(ReceiverRevisionState::Unspecified as i32, 0);
        assert_eq!(ReceiverRevisionState::BytesReceived as i32, 1);
        assert_eq!(ReceiverRevisionState::Verified as i32, 2);
        assert_eq!(ReceiverRevisionState::Failed as i32, 3);
        assert_eq!(ReceiverRevisionState::Poisoned as i32, 4);
        assert!(ReceiverRevisionState::try_from(5).is_err());
    }

    #[test]
    fn publication_modes_contain_only_block_and_async() {
        assert_eq!(PublicationMode::Block as i32, 0);
        assert_eq!(PublicationMode::Async as i32, 1);
        assert!(PublicationMode::try_from(2).is_err());
    }
}
