// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

//! Static checks over the Helm chart templates.
//!
//! Every namespaced resource the chart renders must stamp the release namespace
//! onto `metadata.namespace`, so that `helm template | kubectl apply -f -` places
//! the whole release in one namespace rather than splitting it across the
//! rendered namespace and the caller's current context.
//!
//! These operate on the template source rather than on `helm template` output so
//! they run under plain `cargo test` without a helm binary, and so they also
//! cover templates gated off by default (an unrendered template is exactly how
//! the `ingress.yaml` gap went unnoticed).

use std::fs;
use std::path::{Path, PathBuf};

/// Kinds that live outside any namespace. Anything not listed here is treated as
/// namespaced, so a template introducing an unrecognized kind fails until someone
/// classifies it. Failing closed is the point: the alternative, an allowlist of
/// namespaced kinds, silently skips whatever it does not know about.
const CLUSTER_SCOPED_KINDS: &[&str] = &[
    "APIService",
    "ClusterRole",
    "ClusterRoleBinding",
    "CustomResourceDefinition",
    "IngressClass",
    "MutatingWebhookConfiguration",
    "Namespace",
    "PersistentVolume",
    "PriorityClass",
    "StorageClass",
    "ValidatingWebhookConfiguration",
];

/// `metadata.namespace` sits at two-space indent. Matching on the indent keeps
/// this from being satisfied by a deeper `namespace:` such as the one under a
/// RoleBinding's `subjects`.
const METADATA_NAMESPACE_LINE: &str = "  namespace: {{ .Release.Namespace }}";

fn templates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace-tests has a parent directory")
        .join("helm/templates")
}

fn yaml_templates(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.is_dir() {
            found.extend(yaml_templates(&path));
        } else if path.extension().is_some_and(|ext| ext == "yaml") {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Split a template into YAML documents and return the top-level `kind` of each
/// alongside its text. Requiring `kind:` at column zero skips the nested `kind:`
/// keys under `roleRef` and `subjects`.
fn documents_with_kind(contents: &str) -> Vec<(String, String)> {
    contents
        .split('\n')
        .fold(vec![String::new()], |mut docs, line| {
            if line.trim_end() == "---" {
                docs.push(String::new());
            } else {
                let current = docs.last_mut().expect("fold seeds one document");
                current.push_str(line);
                current.push('\n');
            }
            docs
        })
        .into_iter()
        .filter_map(|doc| {
            let kind = doc
                .lines()
                .find_map(|line| line.strip_prefix("kind: "))?
                .trim()
                .to_string();
            Some((kind, doc))
        })
        .collect()
}

#[test]
fn namespaced_resources_stamp_the_release_namespace() {
    let dir = templates_dir();
    let mut checked = 0;
    let mut missing = Vec::new();

    for path in yaml_templates(&dir) {
        let contents =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for (kind, doc) in documents_with_kind(&contents) {
            if CLUSTER_SCOPED_KINDS.contains(&kind.as_str()) {
                continue;
            }
            checked += 1;
            if !doc.lines().any(|line| line == METADATA_NAMESPACE_LINE) {
                missing.push(format!("{} ({kind})", path.display()));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "namespaced resources missing `{METADATA_NAMESPACE_LINE}` in their metadata block: {missing:#?}\n\
         If a listed kind is actually cluster-scoped, add it to CLUSTER_SCOPED_KINDS instead."
    );
    assert!(
        checked > 0,
        "no namespaced resources found under {} - the template scan is broken",
        dir.display()
    );
}

#[test]
fn cluster_scoped_resources_omit_the_release_namespace() {
    for path in yaml_templates(&templates_dir()) {
        let contents =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for (kind, doc) in documents_with_kind(&contents) {
            if !CLUSTER_SCOPED_KINDS.contains(&kind.as_str()) {
                continue;
            }
            assert!(
                !doc.lines().any(|line| line == METADATA_NAMESPACE_LINE),
                "{} ({kind}) is cluster-scoped but sets metadata.namespace; the API server rejects that",
                path.display()
            );
        }
    }
}
