// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached files that belong to a registry entry.
//!
//! Deleting a registry entry deletes the files it covers, whatever deleted it: LRU
//! eviction, the `DeleteModel` RPC, or, on the Kubernetes backend, a direct delete of
//! the `ModelCacheEntry` through the Kubernetes API. Entries can share files (the
//! revisions of one model share blobs, and an entry with no revision covers every
//! snapshot), so [`deletable_scope`] decides what is safe to remove given the entries
//! that remain.

use std::path::{Component, Path};

use modelexpress_common::download::{canonical_model_name, get_provider};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use tracing::{debug, warn};

use super::backend::ModelRecord;
use super::entry_key::EntryKey;

/// What deleting one registry entry may remove from the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileScope<'a> {
    /// A remaining entry references this entry's files, or can come to reference them.
    Keep,
    Revision(&'a str),
    WholeModel,
}

/// A registry entry that outlives the one being deleted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Survivor {
    pub key: EntryKey,
    pub downloading: bool,
}

impl From<&ModelRecord> for Survivor {
    fn from(record: &ModelRecord) -> Self {
        Self {
            key: record_key(&record.model_name, record.provider),
            downloading: record.status == ModelStatus::DOWNLOADING,
        }
    }
}

/// The entry key of a registry record, with the model name in its provider's canonical
/// spelling. A record written before its provider had a canonical spelling can name the
/// same cache directory as another record does, and the file rules must see the two as
/// one model.
pub fn record_key(record_name: &str, provider: ModelProvider) -> EntryKey {
    let mut key = EntryKey::parse(record_name);
    if let Ok(canonical) = canonical_model_name(&key.model_name, provider) {
        key.model_name = canonical;
    }
    key
}

/// Entries of other models are ignored.
pub fn deletable_scope<'a>(entry: &'a EntryKey, survivors: &[Survivor]) -> FileScope<'a> {
    let mut same_model = survivors
        .iter()
        .filter(|other| other.key.model_name == entry.model_name)
        .peekable();
    if same_model.peek().is_none() {
        // Nothing else references this model. Drop the whole cache entry rather than
        // just this revision, so snapshots left behind by an earlier shared-file
        // decision cannot outlive the last record pointing at them.
        return FileScope::WholeModel;
    }

    // An entry with no revision covers every snapshot of its model, so it shares
    // files with all of them. That is how a record written before revisions existed
    // behaves: deleting it must not delete a snapshot a revision-scoped entry uses,
    // and deleting a revision-scoped entry must not delete files the legacy entry covers.
    let Some(revision) = entry.revision.as_deref() else {
        return FileScope::Keep;
    };
    let shares_files = |other: &Survivor| {
        // A download in flight holds partial blobs that no snapshot links yet, so a
        // revision delete would reclaim them as unreferenced.
        other.downloading || other.key.revision.as_deref().is_none_or(|r| r == revision)
    };
    if same_model.any(shares_files) {
        return FileScope::Keep;
    }
    FileScope::Revision(revision)
}

pub async fn delete_files(
    provider: ModelProvider,
    model_name: &str,
    cache_dir: &Path,
    scope: FileScope<'_>,
) -> anyhow::Result<()> {
    let revision = match scope {
        FileScope::Keep => {
            debug!(
                "Keeping files for '{model_name}': another registry entry still references them"
            );
            return Ok(());
        }
        FileScope::Revision(revision) => {
            if !is_plain_path_component(revision) {
                // The server only writes commit identifiers into a key. Any other shape
                // comes from a record it did not write, covers no files of its own, and
                // would address a directory outside the model's snapshots.
                warn!(
                    "Not deleting files for '{model_name}': revision {revision:?} is not a \
                     commit identifier"
                );
                return Ok(());
            }
            Some(revision)
        }
        FileScope::WholeModel => None,
    };
    if let Err(e) = canonical_model_name(model_name, provider) {
        // The server refuses such a name in every request, so the entry has no files
        // of its own. The name can give the cache directory of a different model.
        warn!("Not deleting files for '{model_name}': {e:#}");
        return Ok(());
    }
    get_provider(provider)
        .delete_model_revision(model_name, cache_dir.to_path_buf(), revision)
        .await
}

fn is_plain_path_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(only)), None) if only == name
    )
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn key(revision: Option<&str>) -> EntryKey {
        EntryKey::new("org/model", revision.map(String::from), false)
    }

    fn survivor(key: EntryKey) -> Survivor {
        Survivor {
            key,
            downloading: false,
        }
    }

    #[test]
    fn last_entry_of_a_model_deletes_the_whole_model() {
        let entry = key(Some("abc"));
        let other_model = survivor(EntryKey::new("org/other", None, false));
        assert_eq!(
            deletable_scope(&entry, &[other_model]),
            FileScope::WholeModel
        );
        assert_eq!(deletable_scope(&entry, &[]), FileScope::WholeModel);
    }

    #[test]
    fn other_revision_survivor_deletes_only_this_revision() {
        let entry = key(Some("abc"));
        assert_eq!(
            deletable_scope(&entry, &[survivor(key(Some("def")))]),
            FileScope::Revision("abc")
        );
    }

    #[test]
    fn shared_snapshot_is_kept() {
        let entry = key(Some("abc"));
        // Same revision under a different weight mode.
        let metadata_only = EntryKey::new("org/model", Some("abc".to_string()), true);
        assert_eq!(
            deletable_scope(&entry, &[survivor(metadata_only)]),
            FileScope::Keep
        );
        // An unpinned survivor covers every snapshot.
        assert_eq!(
            deletable_scope(&entry, &[survivor(key(None))]),
            FileScope::Keep
        );
        // An unpinned entry shares files with every survivor.
        assert_eq!(
            deletable_scope(&key(None), &[survivor(key(Some("def")))]),
            FileScope::Keep
        );
    }

    #[test]
    fn a_download_in_flight_for_another_revision_keeps_the_files() {
        let entry = key(Some("abc"));
        let downloading = Survivor {
            key: key(Some("def")),
            downloading: true,
        };
        assert_eq!(
            deletable_scope(&entry, std::slice::from_ref(&downloading)),
            FileScope::Keep
        );
        // A second, finished revision does not change that.
        assert_eq!(
            deletable_scope(&entry, &[survivor(key(Some("ghi"))), downloading]),
            FileScope::Keep
        );
    }

    #[test]
    fn record_keys_use_the_canonical_model_name() {
        let alias = record_key("nvidia/model/1", ModelProvider::Ngc);
        let canonical = record_key("ngc://nvidia/model/1", ModelProvider::Ngc);
        assert_eq!(alias, canonical);

        // A name its provider rejects stays as written.
        let invalid = record_key("////", ModelProvider::Ngc);
        assert_eq!(invalid.model_name, "////");

        let pinned = record_key(
            &EntryKey::new("org/model", Some("abc".to_string()), true).to_string(),
            ModelProvider::HuggingFace,
        );
        assert_eq!(
            pinned,
            EntryKey::new("org/model", Some("abc".to_string()), true)
        );
    }

    #[tokio::test]
    async fn keep_scope_touches_nothing() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let marker = dir.path().join("models--org--model");
        std::fs::create_dir_all(&marker).expect("mkdir");
        delete_files(
            ModelProvider::HuggingFace,
            "org/model",
            dir.path(),
            FileScope::Keep,
        )
        .await
        .expect("keep");
        assert!(marker.exists());
    }

    #[tokio::test]
    async fn whole_model_scope_removes_the_repository() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let repo = dir.path().join("models--org--model");
        std::fs::create_dir_all(repo.join("snapshots").join("abc")).expect("mkdir");
        delete_files(
            ModelProvider::HuggingFace,
            "org/model",
            dir.path(),
            FileScope::WholeModel,
        )
        .await
        .expect("delete");
        assert!(!repo.exists());
        delete_files(
            ModelProvider::HuggingFace,
            "org/model",
            dir.path(),
            FileScope::WholeModel,
        )
        .await
        .expect("idempotent delete");
    }

    /// `org--model` is not a repository id, and its cache directory is the one of
    /// `org/model`. An entry with that name must not take the files of the other model.
    #[tokio::test]
    async fn a_name_its_provider_rejects_deletes_nothing() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let snapshot = dir.path().join("models--org--model/snapshots/abc");
        std::fs::create_dir_all(&snapshot).expect("mkdir");

        for scope in [FileScope::WholeModel, FileScope::Revision("abc")] {
            delete_files(ModelProvider::HuggingFace, "org--model", dir.path(), scope)
                .await
                .expect("an invalid name is not an error");
            assert!(
                snapshot.exists(),
                "{scope:?} removed the files of org/model"
            );
        }
    }

    /// A revision is joined into the snapshot path, so one that is not a plain
    /// directory name would move the delete out of the model's own snapshots.
    #[tokio::test]
    async fn a_revision_that_is_not_a_commit_identifier_deletes_nothing() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cache = dir.path().join("cache");
        let outside = dir.path().join("outside");
        let snapshot = cache.join("models--org--model/snapshots/abc");
        let neighbor = cache.join("models--org--other");
        for path in [&snapshot, &neighbor, &outside] {
            std::fs::create_dir_all(path).expect("mkdir");
        }

        let absolute = outside.to_string_lossy().into_owned();
        for revision in ["../..", "..", ".", "abc/..", "abc/", "", absolute.as_str()] {
            delete_files(
                ModelProvider::HuggingFace,
                "org/model",
                &cache,
                FileScope::Revision(revision),
            )
            .await
            .expect("an invalid revision is not an error");
            for path in [&snapshot, &neighbor, &outside] {
                assert!(path.exists(), "{revision:?} removed {}", path.display());
            }
        }
    }
}
