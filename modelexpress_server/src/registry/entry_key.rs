// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identity of a single registry entry.
//!
//! Registry backends key every record, lease, and claim on one string. A model name
//! alone is not a sufficient key:
//!
//! - two revisions of the same model must not share a download lease, or a request for
//!   one revision would coalesce onto a download of the other;
//! - a metadata-only download (`ignore_weights`) must not satisfy a later full-weight
//!   request, which would hand back a snapshot with no weights in it.
//!
//! [`EntryKey`] folds those three components into the single string the backends want
//! and parses it back, so cache eviction can still recover the model name and revision
//! it needs in order to delete the right files.
//!
//! An unpinned, full-weight entry encodes to the bare model name, so providers without
//! a revision concept keep the keys they have always used.

use std::fmt::{Display, Formatter};

/// Separates the model name from its resolved revision.
const REVISION_SEPARATOR: &str = "@rev:";
/// Marks an entry whose download skipped weight files.
const METADATA_SUFFIX: &str = "#metadata";

/// The components a registry key is built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryKey {
    /// Provider-canonical model name.
    pub model_name: String,
    /// Immutable revision the request resolved to, when the provider has one.
    pub revision: Option<String>,
    /// Whether the entry covers a weightless (config/tokenizer only) download.
    pub metadata_only: bool,
}

impl EntryKey {
    pub fn new(
        model_name: impl Into<String>,
        revision: Option<String>,
        metadata_only: bool,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            revision,
            metadata_only,
        }
    }

    /// Recover the components of an encoded key.
    ///
    /// A key that carries neither marker parses back to an unpinned, full-weight entry,
    /// which is how records written before revisions existed are read.
    pub fn parse(key: &str) -> Self {
        let (body, metadata_only) = match key.strip_suffix(METADATA_SUFFIX) {
            Some(body) => (body, true),
            None => (key, false),
        };

        match body.rsplit_once(REVISION_SEPARATOR) {
            Some((model_name, revision)) if !model_name.is_empty() && !revision.is_empty() => {
                Self::new(model_name, Some(revision.to_string()), metadata_only)
            }
            _ => Self::new(body, None, metadata_only),
        }
    }

    /// True when this entry belongs to `model_name`, whatever its revision or weight mode.
    /// Drives "forget everything about this model" operations such as `model clear`.
    pub fn belongs_to(&self, model_name: &str) -> bool {
        self.model_name == model_name
    }
}

impl Display for EntryKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.model_name)?;
        if let Some(revision) = &self.revision {
            write!(f, "{REVISION_SEPARATOR}{revision}")?;
        }
        if self.metadata_only {
            f.write_str(METADATA_SUFFIX)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(key: EntryKey) {
        assert_eq!(EntryKey::parse(&key.to_string()), key);
    }

    #[test]
    fn unpinned_full_weight_entry_encodes_to_the_bare_model_name() {
        let key = EntryKey::new("google-t5/t5-small", None, false);
        assert_eq!(key.to_string(), "google-t5/t5-small");
        roundtrip(key);
    }

    #[test]
    fn every_component_survives_a_roundtrip() {
        roundtrip(EntryKey::new(
            "org/model",
            Some("abc123".to_string()),
            false,
        ));
        roundtrip(EntryKey::new("org/model", None, true));
        roundtrip(EntryKey::new("org/model", Some("abc123".to_string()), true));
        roundtrip(EntryKey::new("gs://bucket/org/model/rev-1", None, false));
    }

    #[test]
    fn two_revisions_of_one_model_get_distinct_keys() {
        let first = EntryKey::new("org/model", Some("abc123".to_string()), false).to_string();
        let second = EntryKey::new("org/model", Some("def456".to_string()), false).to_string();
        assert_ne!(first, second);
    }

    #[test]
    fn metadata_only_and_full_weight_entries_do_not_collide() {
        let revision = Some("abc123".to_string());
        assert_ne!(
            EntryKey::new("org/model", revision.clone(), true).to_string(),
            EntryKey::new("org/model", revision, false).to_string()
        );
    }

    #[test]
    fn a_legacy_key_parses_as_unpinned_and_full_weight() {
        let parsed = EntryKey::parse("meta-llama/Llama-3.1-70B");
        assert_eq!(parsed.model_name, "meta-llama/Llama-3.1-70B");
        assert!(parsed.revision.is_none());
        assert!(!parsed.metadata_only);
    }

    #[test]
    fn belongs_to_ignores_revision_and_weight_mode() {
        let key = EntryKey::new("org/model", Some("abc123".to_string()), true);
        assert!(key.belongs_to("org/model"));
        assert!(!key.belongs_to("org/other"));
    }
}
