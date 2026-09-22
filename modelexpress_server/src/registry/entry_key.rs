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
//!
//! # Encoding
//!
//! Anything else encodes as `mx1:<revision>:<flags>:<model_name>`. The model name goes
//! last because it is the only field with no character restrictions — a GCS object path
//! accepts almost any byte, so a name like `gs://bucket/models/foo:m:bar` must not be
//! able to impersonate the other fields. Putting it last means nothing after it needs
//! parsing, and the round trip is exact for any name.
//!
//! The revision field does have to avoid `:`, which holds because it is always a commit
//! identifier: Git forbids `:` in ref names, and resolved revisions are commit SHAs.

use std::fmt::{Display, Formatter};

/// Marks a key that carries a revision or weight mode. Versioned so the format can
/// change without misreading old keys.
const STRUCTURED_PREFIX: &str = "mx1:";
/// Weight-mode field values.
const METADATA_FLAG: &str = "m";
const FULL_WEIGHT_FLAG: &str = "-";

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

    /// Whether the key needs the structured form.
    ///
    /// A model name that happens to start with the prefix is encoded structurally even
    /// when it carries no revision, so it can never be mistaken for a key we wrote.
    fn needs_structured_form(&self) -> bool {
        self.revision.is_some()
            || self.metadata_only
            || self.model_name.starts_with(STRUCTURED_PREFIX)
    }

    /// Recover the components of an encoded key.
    ///
    /// A key without the prefix parses back to an unpinned, full-weight entry, which is
    /// how records written before revisions existed are read.
    pub fn parse(key: &str) -> Self {
        let Some(fields) = key.strip_prefix(STRUCTURED_PREFIX) else {
            return Self::new(key, None, false);
        };

        // `splitn(3)` leaves the model name — the only field that may contain `:` —
        // untouched in the final part.
        let mut parts = fields.splitn(3, ':');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(revision), Some(flags), Some(model_name)) => Self::new(
                model_name,
                (!revision.is_empty()).then(|| revision.to_string()),
                flags == METADATA_FLAG,
            ),
            // Not a key this module produced; treat it as a plain model name.
            _ => Self::new(key, None, false),
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
        if !self.needs_structured_form() {
            return f.write_str(&self.model_name);
        }

        let flags = if self.metadata_only {
            METADATA_FLAG
        } else {
            FULL_WEIGHT_FLAG
        };
        write!(
            f,
            "{STRUCTURED_PREFIX}{}:{flags}:{}",
            self.revision.as_deref().unwrap_or(""),
            self.model_name
        )
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

    /// A GCS object path accepts almost any byte, so a model name is free to contain
    /// whatever the encoding uses as structure. Misreading one would evict a different
    /// model's files, so every shape has to round-trip exactly.
    #[test]
    fn a_model_name_cannot_impersonate_the_encoding() {
        for name in [
            "gs://bucket/models/foo:m:bar",
            "gs://bucket/models/foo#metadata",
            "gs://bucket/models/foo@rev:abc123",
            "mx1:abc123:m:gs://bucket/models/foo",
            "mx1:",
            ":::",
        ] {
            roundtrip(EntryKey::new(name, None, false));
            roundtrip(EntryKey::new(name, None, true));
            roundtrip(EntryKey::new(name, Some("abc123".to_string()), false));
        }
    }

    #[test]
    fn a_name_that_looks_like_a_key_stays_distinct_from_the_key_it_mimics() {
        // The literal name and the entry whose encoding it copies must not collide.
        let mimic = EntryKey::new("mx1:abc123:m:org/model", None, false);
        let real = EntryKey::new("org/model", Some("abc123".to_string()), true);
        assert_ne!(mimic.to_string(), real.to_string());
        assert_eq!(EntryKey::parse(&mimic.to_string()), mimic);
        assert_eq!(EntryKey::parse(&real.to_string()), real);
    }

    #[test]
    fn a_model_name_with_a_colon_keeps_its_revision_and_weight_mode() {
        let key = EntryKey::new("ngc/org/model:1.0", Some("abc123".to_string()), true);
        let parsed = EntryKey::parse(&key.to_string());
        assert_eq!(parsed.model_name, "ngc/org/model:1.0");
        assert_eq!(parsed.revision.as_deref(), Some("abc123"));
        assert!(parsed.metadata_only);
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
