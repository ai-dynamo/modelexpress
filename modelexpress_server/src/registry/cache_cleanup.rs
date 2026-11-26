// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure survivor-aware cache cleanup planning shared by `DeleteModel`, LRU eviction,
//! and the Kubernetes finalizer controller.
//!
//! Every caller feeds the current live registry records into [`plan_cleanup_for_entry`]
//! or [`plan_cleanup_for_model`] and executes the resulting [`CleanupPlan`] via
//! [`execute_cleanup_plan`]. Putting all validation, canonicalization, overlap checks,
//! and Hugging Face blob-reclaim rules in one pure module ensures all three delete paths
//! enforce identical invariants.

use std::path::Path;

use modelexpress_common::download::{canonical_model_name, get_provider};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use modelexpress_common::providers::HuggingFaceProvider;
use tracing::{debug, warn};

use crate::registry::backend::ModelRecord;
use crate::registry::entry_key::EntryKey;

/// Action to take on the shared cache filesystem for a deleted registry entry or model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupPlan {
    /// Target model name or revision failed provider validation; never touch disk.
    InvalidTarget { reason: String },
    /// Surviving live registry entries still protect these files; delete nothing on disk.
    Keep { reason: &'static str },
    /// Delete the entire model cache directory for `(provider, model_name)`.
    DeleteWholeModel {
        provider: ModelProvider,
        model_name: String,
    },
    /// Delete a single Hugging Face snapshot and stale refs pointing to it, then reclaim
    /// unreferenced blobs (and remove the repository directory if no snapshots remain).
    ///
    /// Only produced when no surviving entry of the same model is `DOWNLOADING`.
    DeleteHfRevisionReclaimBlobs {
        model_name: String,
        revision: String,
    },
    /// Delete a single Hugging Face snapshot and stale refs pointing to it while leaving
    /// `blobs/` and the repository root untouched because a sibling entry of the same
    /// model is currently `DOWNLOADING`.
    ///
    /// Deferred blobs are reclaimed on a later cleanup for that model when no sibling is
    /// downloading.
    DeleteHfRevisionKeepBlobs {
        model_name: String,
        revision: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixRelation {
    Same,
    OtherIsAncestor,
    OtherIsDescendant,
    Disjoint,
}

fn validate_target_revision(provider: ModelProvider, revision: Option<&str>) -> Result<(), String> {
    let Some(revision) = revision else {
        return Ok(());
    };
    if provider == ModelProvider::HuggingFace {
        HuggingFaceProvider::validate_snapshot_revision(revision).map_err(|e| e.to_string())
    } else {
        Err(format!(
            "provider {:?} does not support revisions, refusing target revision '{revision}'",
            provider
        ))
    }
}

fn split_object_storage_uri(provider: ModelProvider, canonical: &str) -> Option<(&str, Vec<&str>)> {
    let stripped = match provider {
        ModelProvider::S3 => canonical.strip_prefix("s3://")?,
        ModelProvider::Gcs => canonical.strip_prefix("gs://")?,
        ModelProvider::HuggingFace | ModelProvider::Ngc => return None,
    };
    let mut parts = stripped.split('/').filter(|part| !part.is_empty());
    let bucket = parts.next()?;
    let prefix: Vec<&str> = parts.collect();
    Some((bucket, prefix))
}

fn object_storage_prefix_relation(
    provider: ModelProvider,
    target_canonical: &str,
    other_canonical: &str,
) -> PrefixRelation {
    if target_canonical == other_canonical {
        return PrefixRelation::Same;
    }
    let (Some((target_bucket, target_prefix)), Some((other_bucket, other_prefix))) = (
        split_object_storage_uri(provider, target_canonical),
        split_object_storage_uri(provider, other_canonical),
    ) else {
        return PrefixRelation::Disjoint;
    };
    if target_bucket != other_bucket {
        return PrefixRelation::Disjoint;
    }
    if other_prefix.len() < target_prefix.len() && target_prefix.starts_with(&other_prefix) {
        PrefixRelation::OtherIsAncestor
    } else if target_prefix.len() < other_prefix.len() && other_prefix.starts_with(&target_prefix) {
        PrefixRelation::OtherIsDescendant
    } else {
        PrefixRelation::Disjoint
    }
}

/// Returns `true` when `record` belongs to `(provider, canonical_target)` after parsing
/// its `EntryKey` and canonicalizing the stored model name.
///
/// Unparseable stored names for `provider` are skipped with a warning and never match.
#[must_use]
pub fn record_belongs_to_model(
    record: &ModelRecord,
    provider: ModelProvider,
    canonical_target: &str,
) -> bool {
    if record.provider != provider {
        return false;
    }
    let parsed = EntryKey::parse(&record.model_name);
    match canonical_model_name(&parsed.model_name, provider) {
        Ok(canonical) => canonical == canonical_target,
        Err(err) => {
            warn!(
                "Skipping stored registry entry '{}' for provider {:?}: failed to canonicalize \
                 model name '{}': {err}",
                record.model_name, provider, parsed.model_name
            );
            false
        }
    }
}

/// Plan filesystem cleanup for removing a single registry entry (`target_entry_key`).
///
/// `live_records` must contain only the live registry records that remain after the
/// target entry has been deleted or marked terminating. If a live record with the same
/// `(provider, model_name)` still exists (for example, a migrated provider-scoped CR
/// after legacy CR deletion, or a concurrent re-claim), it is treated as a survivor and
/// protects cached files.
///
/// Validation of the target model name and revision always runs first: an invalid stored
/// record produces [`CleanupPlan::InvalidTarget`] and never falls back to whole-model
/// deletion, even when no surviving entries remain.
#[must_use]
pub fn plan_cleanup_for_entry(
    provider: ModelProvider,
    target_entry_key: &str,
    live_records: &[ModelRecord],
) -> CleanupPlan {
    let target_key = EntryKey::parse(target_entry_key);
    let canonical_target = match canonical_model_name(&target_key.model_name, provider) {
        Ok(name) => name,
        Err(err) => {
            return CleanupPlan::InvalidTarget {
                reason: format!("invalid model name '{}': {err}", target_key.model_name),
            };
        }
    };
    if let Err(reason) = validate_target_revision(provider, target_key.revision.as_deref()) {
        return CleanupPlan::InvalidTarget { reason };
    }

    let mut has_blocking_ancestor = false;
    let mut has_downloading_descendant = false;
    let mut same_model_survivors: Vec<(EntryKey, ModelStatus)> = Vec::new();

    for record in live_records {
        if record.provider != provider {
            continue;
        }
        let other_key = EntryKey::parse(&record.model_name);
        let other_canonical = match canonical_model_name(&other_key.model_name, provider) {
            Ok(name) => name,
            Err(err) => {
                warn!(
                    "Ignoring malformed live registry entry '{}' for provider {:?} during \
                     cleanup planning: {err}",
                    record.model_name, provider
                );
                continue;
            }
        };

        if matches!(provider, ModelProvider::S3 | ModelProvider::Gcs) {
            match object_storage_prefix_relation(provider, &canonical_target, &other_canonical) {
                PrefixRelation::OtherIsAncestor => {
                    has_blocking_ancestor = true;
                }
                PrefixRelation::OtherIsDescendant => {
                    if record.status == ModelStatus::DOWNLOADING {
                        has_downloading_descendant = true;
                    }
                }
                PrefixRelation::Same | PrefixRelation::Disjoint => {}
            }
        }

        if other_canonical == canonical_target {
            same_model_survivors.push((other_key, record.status));
        }
    }

    if has_blocking_ancestor {
        return CleanupPlan::Keep {
            reason: "live ancestor entry shares storage prefix",
        };
    }
    if has_downloading_descendant {
        return CleanupPlan::Keep {
            reason: "downloading descendant entry shares storage prefix",
        };
    }
    if same_model_survivors.is_empty() {
        return CleanupPlan::DeleteWholeModel {
            provider,
            model_name: canonical_target,
        };
    }

    // An unpinned entry (revision == None) covers the entire model directory, so it
    // shares files with every surviving entry of that model (including NGC/S3/GCS full
    // and metadata-only entries that share one directory).
    let Some(target_revision) = target_key.revision else {
        return CleanupPlan::Keep {
            reason: "unpinned entry shares files with surviving entries",
        };
    };

    let snapshot_still_referenced = same_model_survivors.iter().any(|(other, _)| {
        other.revision.is_none() || other.revision.as_deref() == Some(target_revision.as_str())
    });
    if snapshot_still_referenced {
        return CleanupPlan::Keep {
            reason: "surviving entry references the same snapshot",
        };
    }

    let sibling_downloading = same_model_survivors
        .iter()
        .any(|(_, status)| *status == ModelStatus::DOWNLOADING);
    if sibling_downloading {
        CleanupPlan::DeleteHfRevisionKeepBlobs {
            model_name: canonical_target,
            revision: target_revision,
        }
    } else {
        CleanupPlan::DeleteHfRevisionReclaimBlobs {
            model_name: canonical_target,
            revision: target_revision,
        }
    }
}

/// Plan filesystem cleanup for removing all entries of `(provider, raw_model_name)`
/// (used by `DeleteModel` after deleting matching records, or when retrying with 0 records).
///
/// `remaining_live_records` must contain the live registry records that remain after this
/// model's entries were deleted.
#[must_use]
pub fn plan_cleanup_for_model(
    provider: ModelProvider,
    raw_model_name: &str,
    remaining_live_records: &[ModelRecord],
) -> CleanupPlan {
    let canonical_target = match canonical_model_name(raw_model_name, provider) {
        Ok(name) => name,
        Err(err) => {
            return CleanupPlan::InvalidTarget {
                reason: format!("invalid model name '{raw_model_name}': {err}"),
            };
        }
    };

    for record in remaining_live_records {
        if record.provider != provider {
            continue;
        }
        let other_key = EntryKey::parse(&record.model_name);
        let other_canonical = match canonical_model_name(&other_key.model_name, provider) {
            Ok(name) => name,
            Err(err) => {
                warn!(
                    "Ignoring malformed live registry entry '{}' for provider {:?} during \
                     whole-model cleanup planning: {err}",
                    record.model_name, provider
                );
                continue;
            }
        };

        if other_canonical == canonical_target {
            return CleanupPlan::Keep {
                reason: "live entry for this model still exists",
            };
        }

        if matches!(provider, ModelProvider::S3 | ModelProvider::Gcs) {
            match object_storage_prefix_relation(provider, &canonical_target, &other_canonical) {
                PrefixRelation::OtherIsAncestor => {
                    return CleanupPlan::Keep {
                        reason: "live ancestor entry shares storage prefix",
                    };
                }
                PrefixRelation::OtherIsDescendant if record.status == ModelStatus::DOWNLOADING => {
                    return CleanupPlan::Keep {
                        reason: "downloading descendant entry shares storage prefix",
                    };
                }
                PrefixRelation::Same
                | PrefixRelation::OtherIsDescendant
                | PrefixRelation::Disjoint => {}
            }
        }
    }

    CleanupPlan::DeleteWholeModel {
        provider,
        model_name: canonical_target,
    }
}

/// Execute a [`CleanupPlan`] against `cache_dir`.
///
/// Runs synchronous filesystem removals on Tokio's blocking thread pool so
/// deleting large model directories does not stall async worker threads.
/// Returns `Ok(true)` if disk cleanup was executed, or `Ok(false)` if files were kept
/// (`Keep`) or refused (`InvalidTarget`).
pub async fn execute_cleanup_plan(plan: &CleanupPlan, cache_dir: &Path) -> anyhow::Result<bool> {
    match plan {
        CleanupPlan::InvalidTarget { reason } => {
            warn!("Refusing cache file cleanup for invalid registry entry: {reason}");
            Ok(false)
        }
        CleanupPlan::Keep { reason } => {
            debug!("Keeping cached files: {reason}");
            Ok(false)
        }
        CleanupPlan::DeleteWholeModel { .. }
        | CleanupPlan::DeleteHfRevisionReclaimBlobs { .. }
        | CleanupPlan::DeleteHfRevisionKeepBlobs { .. } => {
            let plan = plan.clone();
            let cache_dir = cache_dir.to_path_buf();
            let handle = tokio::runtime::Handle::current();
            tokio::task::spawn_blocking(move || match plan {
                CleanupPlan::DeleteWholeModel {
                    provider,
                    model_name,
                } => handle.block_on(get_provider(provider).delete_model(&model_name, cache_dir)),
                CleanupPlan::DeleteHfRevisionReclaimBlobs {
                    model_name,
                    revision,
                } => handle.block_on(
                    get_provider(ModelProvider::HuggingFace).delete_model_revision(
                        &model_name,
                        cache_dir,
                        Some(&revision),
                    ),
                ),
                CleanupPlan::DeleteHfRevisionKeepBlobs {
                    model_name,
                    revision,
                } => HuggingFaceProvider::delete_snapshot_keeping_blobs(
                    &model_name,
                    &cache_dir,
                    &revision,
                ),
                CleanupPlan::InvalidTarget { .. } | CleanupPlan::Keep { .. } => Ok(()),
            })
            .await
            .map_err(|err| anyhow::anyhow!("cache cleanup task failed: {err}"))??;
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;

    fn record(model_name: &str, provider: ModelProvider, status: ModelStatus) -> ModelRecord {
        let now = Utc::now();
        ModelRecord {
            model_name: model_name.to_string(),
            provider,
            status,
            created_at: now,
            last_used_at: now,
            message: None,
        }
    }

    #[test]
    fn hf_single_revision_with_no_survivors_deletes_whole_model() {
        let target = EntryKey::new("org/model", Some("abc1234".to_string()), false).to_string();
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &target, &[]),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::HuggingFace,
                model_name: "org/model".to_string(),
            }
        );

        // If a live record with the exact same entry key still exists (e.g. legacy-to-scoped
        // CR migration or a concurrent re-claim), keep files on disk.
        let same_key_survivor = vec![record(
            &target,
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
        )];
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &target, &same_key_survivor),
            CleanupPlan::Keep { .. }
        ));
    }

    #[test]
    fn hf_single_revision_keeps_when_same_revision_or_unpinned_survivor_exists() {
        let target = EntryKey::new("org/model", Some("abc1234".to_string()), false).to_string();
        let meta_same_rev =
            EntryKey::new("org/model", Some("abc1234".to_string()), true).to_string();
        let live = vec![record(
            &meta_same_rev,
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
        )];
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &target, &live),
            CleanupPlan::Keep { .. }
        ));

        let unpinned = vec![record(
            "org/model",
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
        )];
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &target, &unpinned),
            CleanupPlan::Keep { .. }
        ));
    }

    #[test]
    fn hf_variant_1_skips_blob_reclaim_while_sibling_is_downloading() {
        let rev1 = EntryKey::new("org/model", Some("rev1".to_string()), false).to_string();
        let rev2 = EntryKey::new("org/model", Some("rev2".to_string()), false).to_string();

        let live_downloading = vec![record(
            &rev2,
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADING,
        )];
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &rev1, &live_downloading),
            CleanupPlan::DeleteHfRevisionKeepBlobs {
                model_name: "org/model".to_string(),
                revision: "rev1".to_string(),
            }
        );

        let live_downloaded = vec![record(
            &rev2,
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
        )];
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &rev1, &live_downloaded),
            CleanupPlan::DeleteHfRevisionReclaimBlobs {
                model_name: "org/model".to_string(),
                revision: "rev1".to_string(),
            }
        );

        let live_error = vec![record(
            &rev2,
            ModelProvider::HuggingFace,
            ModelStatus::ERROR,
        )];
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &rev1, &live_error),
            CleanupPlan::DeleteHfRevisionReclaimBlobs {
                model_name: "org/model".to_string(),
                revision: "rev1".to_string(),
            }
        );
    }

    #[test]
    fn odo_1_unpinned_deleted_entry_keeps_when_any_survivor_exists_across_providers() {
        // Hugging Face: unpinned full entry deleted while pinned revision survives
        let hf_pinned = EntryKey::new("org/model", Some("rev1".to_string()), false).to_string();
        assert!(matches!(
            plan_cleanup_for_entry(
                ModelProvider::HuggingFace,
                "org/model",
                &[record(
                    &hf_pinned,
                    ModelProvider::HuggingFace,
                    ModelStatus::DOWNLOADED
                )],
            ),
            CleanupPlan::Keep { .. }
        ));

        // NGC: full-weight deleted while metadata-only survives (sharing one directory)
        let ngc_meta = EntryKey::new("ngc://nim/nvidia/llama/1", None, true).to_string();
        assert!(matches!(
            plan_cleanup_for_entry(
                ModelProvider::Ngc,
                "nim/nvidia/llama:1",
                &[record(
                    &ngc_meta,
                    ModelProvider::Ngc,
                    ModelStatus::DOWNLOADED
                )],
            ),
            CleanupPlan::Keep { .. }
        ));
    }

    #[test]
    fn odo_2_canonicalizes_stored_ngc_spellings_for_survivors_and_matching() {
        let canonical =
            canonical_model_name("nim/org/mistral/2", ModelProvider::Ngc).expect("canonical NGC");
        let raw_record = record(
            "nim/org/mistral/2",
            ModelProvider::Ngc,
            ModelStatus::DOWNLOADED,
        );
        let uri_record = record(
            "ngc://nim/org/mistral/2",
            ModelProvider::Ngc,
            ModelStatus::DOWNLOADED,
        );
        let colon_meta = record(
            &EntryKey::new("nim/org/mistral:2", None, true).to_string(),
            ModelProvider::Ngc,
            ModelStatus::DOWNLOADED,
        );
        let explicit_type = record(
            "nim/org/models/mistral/2",
            ModelProvider::Ngc,
            ModelStatus::DOWNLOADED,
        );

        assert!(record_belongs_to_model(
            &raw_record,
            ModelProvider::Ngc,
            &canonical
        ));
        assert!(record_belongs_to_model(
            &uri_record,
            ModelProvider::Ngc,
            &canonical
        ));
        assert!(record_belongs_to_model(
            &colon_meta,
            ModelProvider::Ngc,
            &canonical
        ));
        assert!(record_belongs_to_model(
            &explicit_type,
            ModelProvider::Ngc,
            &canonical
        ));

        // Deleting raw spelling keeps files when canonical URI metadata-only entry survives.
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::Ngc, "nim/org/mistral/2", &[colon_meta]),
            CleanupPlan::Keep { .. }
        ));
    }

    #[test]
    fn invalid_target_name_or_revision_never_falls_back_to_whole_model_delete() {
        // Jensen ^43 cases 5b/5c: crafted traversal revision with zero survivors
        let crafted_rev =
            EntryKey::new("org/model", Some("../../../outside".to_string()), false).to_string();
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, &crafted_rev, &[]),
            CleanupPlan::InvalidTarget { .. }
        ));

        // Crafted Hugging Face double-dash segment
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, "e2e-bystander--model", &[]),
            CleanupPlan::InvalidTarget { .. }
        ));

        // Crafted revision on a provider without revisions
        let ngc_rev =
            EntryKey::new("nvidia/llama/1", Some("../outside".to_string()), false).to_string();
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::Ngc, &ngc_rev, &[]),
            CleanupPlan::InvalidTarget { .. }
        ));

        // Whole-model delete with invalid name
        assert!(matches!(
            plan_cleanup_for_model(ModelProvider::HuggingFace, "e2e-bystander--model", &[]),
            CleanupPlan::InvalidTarget { .. }
        ));
    }

    #[test]
    fn unrelated_malformed_live_records_are_skipped_without_blocking_valid_delete() {
        let live = vec![
            record(
                "e2e-bystander--model",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
            ),
            record(
                "org/other",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
            ),
        ];
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, "org/valid", &live),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::HuggingFace,
                model_name: "org/valid".to_string(),
            }
        );
    }

    #[test]
    fn cross_provider_records_do_not_interfere() {
        let ngc_record = record("org/model/1", ModelProvider::Ngc, ModelStatus::DOWNLOADED);
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::HuggingFace, "org/model", &[ngc_record]),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::HuggingFace,
                model_name: "org/model".to_string(),
            }
        );
    }

    #[cfg(feature = "s3")]
    #[test]
    fn s3_overlap_rules_protect_ancestor_and_downloading_descendant_only() {
        let parent = "s3://bucket/a";
        let child = "s3://bucket/a/b";
        let sibling_prefix = "s3://bucket/ab";

        // Delete child `a/b` while parent `a` is live (any status) -> Keep
        for status in [
            ModelStatus::DOWNLOADED,
            ModelStatus::DOWNLOADING,
            ModelStatus::ERROR,
        ] {
            let live = vec![record(parent, ModelProvider::S3, status)];
            assert!(
                matches!(
                    plan_cleanup_for_entry(ModelProvider::S3, child, &live),
                    CleanupPlan::Keep { .. }
                ),
                "child delete must Keep when ancestor has status {status:?}"
            );
            assert!(matches!(
                plan_cleanup_for_model(ModelProvider::S3, child, &live),
                CleanupPlan::Keep { .. }
            ));
        }

        // Delete parent `a` while child `a/b` is DOWNLOADING -> Keep
        let downloading_child = vec![record(child, ModelProvider::S3, ModelStatus::DOWNLOADING)];
        assert!(matches!(
            plan_cleanup_for_entry(ModelProvider::S3, parent, &downloading_child),
            CleanupPlan::Keep { .. }
        ));
        assert!(matches!(
            plan_cleanup_for_model(ModelProvider::S3, parent, &downloading_child),
            CleanupPlan::Keep { .. }
        ));

        // Delete parent `a` while child `a/b` is DOWNLOADED -> DeleteWholeModel (S3 walk keeps a/b)
        let downloaded_child = vec![record(child, ModelProvider::S3, ModelStatus::DOWNLOADED)];
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::S3, parent, &downloaded_child),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::S3,
                model_name: parent.to_string(),
            }
        );
        assert_eq!(
            plan_cleanup_for_model(ModelProvider::S3, parent, &downloaded_child),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::S3,
                model_name: parent.to_string(),
            }
        );

        // Segment boundary: `s3://bucket/ab` downloading must NOT block deleting `s3://bucket/a`
        let downloading_sibling = vec![record(
            sibling_prefix,
            ModelProvider::S3,
            ModelStatus::DOWNLOADING,
        )];
        assert_eq!(
            plan_cleanup_for_entry(ModelProvider::S3, parent, &downloading_sibling),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::S3,
                model_name: parent.to_string(),
            }
        );

        // Odo 1: metadata-only + full-weight for same S3 model -> Keep
        let s3_meta = EntryKey::new(parent, None, true).to_string();
        assert!(matches!(
            plan_cleanup_for_entry(
                ModelProvider::S3,
                parent,
                &[record(&s3_meta, ModelProvider::S3, ModelStatus::DOWNLOADED)],
            ),
            CleanupPlan::Keep { .. }
        ));
    }

    #[cfg(feature = "gcs")]
    #[test]
    fn gcs_overlap_rules_and_odo_1_hold() {
        let parent = "gs://bucket/a";
        let child = "gs://bucket/a/b";

        // Live ancestor -> Keep
        assert!(matches!(
            plan_cleanup_for_entry(
                ModelProvider::Gcs,
                child,
                &[record(parent, ModelProvider::Gcs, ModelStatus::DOWNLOADED)],
            ),
            CleanupPlan::Keep { .. }
        ));

        // DOWNLOADING descendant -> Keep
        assert!(matches!(
            plan_cleanup_for_entry(
                ModelProvider::Gcs,
                parent,
                &[record(child, ModelProvider::Gcs, ModelStatus::DOWNLOADING)],
            ),
            CleanupPlan::Keep { .. }
        ));

        // DOWNLOADED descendant -> DeleteWholeModel (GCS is_removable protects overlap)
        assert_eq!(
            plan_cleanup_for_entry(
                ModelProvider::Gcs,
                parent,
                &[record(child, ModelProvider::Gcs, ModelStatus::DOWNLOADED)],
            ),
            CleanupPlan::DeleteWholeModel {
                provider: ModelProvider::Gcs,
                model_name: parent.to_string(),
            }
        );

        // Odo 1: metadata-only + full-weight for same GCS model -> Keep
        let gcs_meta = EntryKey::new(parent, None, true).to_string();
        assert!(matches!(
            plan_cleanup_for_entry(
                ModelProvider::Gcs,
                parent,
                &[record(
                    &gcs_meta,
                    ModelProvider::Gcs,
                    ModelStatus::DOWNLOADED
                )],
            ),
            CleanupPlan::Keep { .. }
        ));
    }

    #[tokio::test]
    async fn execute_cleanup_plan_refuses_invalid_and_executes_hf_variant_1() {
        let temp_dir = TempDir::new().expect("temp dir");
        let outside = temp_dir.path().join("outside");
        std::fs::create_dir_all(&outside).expect("create outside");
        std::fs::write(outside.join("keep.txt"), b"keep").expect("write outside");

        let cache_dir = temp_dir.path().join("cache");
        let repo_root = cache_dir.join("models--org--model");
        std::fs::create_dir_all(repo_root.join("snapshots/rev1")).expect("create rev1");
        std::fs::create_dir_all(repo_root.join("blobs")).expect("create blobs");
        std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
        std::fs::write(repo_root.join("snapshots/rev1/config.json"), b"{}").expect("write cfg");
        std::fs::write(repo_root.join("blobs/in-flight.sync.part"), b"part").expect("write part");
        std::fs::write(repo_root.join("refs/main"), "rev1").expect("write ref");

        // Invalid target does nothing and returns Ok(false).
        let invalid_plan = plan_cleanup_for_entry(
            ModelProvider::HuggingFace,
            &EntryKey::new("org/model", Some("../../../outside".to_string()), false).to_string(),
            &[],
        );
        assert!(
            !execute_cleanup_plan(&invalid_plan, &cache_dir)
                .await
                .expect("invalid plan should not error")
        );
        assert!(outside.join("keep.txt").exists());
        assert!(repo_root.join("snapshots/rev1/config.json").exists());

        // HF Variant 1: rev1 deleted while rev2 is DOWNLOADING removes rev1 + stale ref, keeps blobs.
        let rev1_key = EntryKey::new("org/model", Some("rev1".to_string()), false).to_string();
        let rev2_key = EntryKey::new("org/model", Some("rev2".to_string()), false).to_string();
        let plan = plan_cleanup_for_entry(
            ModelProvider::HuggingFace,
            &rev1_key,
            &[record(
                &rev2_key,
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADING,
            )],
        );
        assert!(
            execute_cleanup_plan(&plan, &cache_dir)
                .await
                .expect("variant 1 cleanup should succeed")
        );
        assert!(!repo_root.join("snapshots/rev1").exists());
        assert!(!repo_root.join("refs/main").exists());
        assert!(repo_root.join("blobs/in-flight.sync.part").exists());
    }
}
