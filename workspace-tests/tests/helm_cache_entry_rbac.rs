// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every RBAC file that grants the server access to `modelcacheentries` grants the same
//! verbs on it, and each file grants what the server needs to delete the cached files of
//! an entry: the finalizer controller watches the entries and patches their metadata,
//! and the claim revocation patches their status. (Extra verbs on the status
//! subresource are allowed: Helm grants both resources in one rule.)

#![allow(clippy::expect_used, clippy::arithmetic_side_effects)]

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

const FILES: &[&str] = &[
    "helm/templates/rbac.yaml",
    "examples/p2p_transfer_k8s/server/kubernetes_backend/rbac-modelmetadata.yaml",
    "examples/dynamo_p2p_transfer_k8s/vllm/rbac-modelmetadata.yaml",
    "ci/k8s/server/rbac-modelmetadata.yaml",
    "workspace-tests/tests/k8s_multinode_tests.rs",
];

const ENTRIES: &str = "modelcacheentries";
const STATUS: &str = "modelcacheentries/status";

const ENTRY_VERBS: &[&str] = &[
    "get", "list", "watch", "create", "update", "patch", "delete",
];
const STATUS_VERBS: &[&str] = &["get", "update", "patch"];

const KNOWN_VERBS: &[&str] = &[
    "get",
    "list",
    "watch",
    "create",
    "update",
    "patch",
    "delete",
    "deletecollection",
    "*",
];

/// Words of a YAML or Rust file with comments, quotes and list punctuation removed.
fn words(text: &str, rust: bool) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = if rust {
            line.split("//").next().unwrap_or_default()
        } else {
            line.split('#').next().unwrap_or_default()
        };
        for word in line.split(|c: char| {
            c.is_whitespace() || matches!(c, '[' | ']' | '{' | '}' | ',' | '"' | '\'' | ':')
        }) {
            let word = word.trim_start_matches('-');
            if !word.is_empty() {
                out.push(word.to_string());
            }
        }
    }
    out
}

/// Verbs for each cache-entry resource in one file. A rule is the words from
/// `resources` up to `verbs`, then the verbs that follow.
fn cache_entry_verbs(text: &str, rust: bool) -> BTreeMap<String, BTreeSet<String>> {
    let words = words(text, rust);
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut i = 0;
    while i < words.len() {
        if words[i] != "resources" {
            i += 1;
            continue;
        }
        let mut resources = Vec::new();
        i += 1;
        while i < words.len() && words[i] != "verbs" && words[i] != "resources" {
            resources.push(words[i].clone());
            i += 1;
        }
        if i >= words.len() || words[i] != "verbs" {
            continue;
        }
        i += 1;
        let mut verbs = BTreeSet::new();
        while i < words.len() && KNOWN_VERBS.contains(&words[i].as_str()) {
            verbs.insert(words[i].clone());
            i += 1;
        }
        for resource in resources {
            if resource == ENTRIES || resource == STATUS {
                out.entry(resource)
                    .or_default()
                    .extend(verbs.iter().cloned());
            }
        }
    }
    out
}

#[test]
fn rbac_files_agree_on_cache_entry_verbs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace-tests has a parent directory");
    let mut problems = Vec::new();
    let mut seen: Vec<(String, BTreeMap<String, BTreeSet<String>>)> = Vec::new();
    for file in FILES {
        let text =
            std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("read {file}: {e}"));
        let verbs = cache_entry_verbs(&text, file.ends_with(".rs"));
        for (resource, needed) in [(ENTRIES, ENTRY_VERBS), (STATUS, STATUS_VERBS)] {
            let granted = verbs.get(resource).cloned().unwrap_or_default();
            let missing: Vec<_> = needed
                .iter()
                .filter(|v| !granted.contains(**v) && !granted.contains("*"))
                .collect();
            if !missing.is_empty() {
                problems.push(format!("{file}: {resource} lacks {missing:?}"));
            }
        }
        seen.push(((*file).to_string(), verbs));
    }
    if let Some((first, reference)) = seen.first() {
        for (file, verbs) in &seen[1..] {
            if verbs.get(ENTRIES) != reference.get(ENTRIES) {
                problems.push(format!(
                    "{file} grants {:?} on {ENTRIES}, but {first} grants {:?}",
                    verbs.get(ENTRIES),
                    reference.get(ENTRIES)
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn the_parser_reads_the_three_formats() {
    let flow = "rules:\n  - apiGroups: [\"x\"]\n    resources: [\"modelcacheentries\", \"modelcacheentries/status\"]\n    verbs: [\"get\", \"watch\"]\n";
    let block = "  - apiGroups:\n      - x\n    resources:\n      - modelcacheentries\n    # a comment with verbs\n    verbs:\n      - get\n      - watch\n";
    let rust =
        "\"resources\": [\n \"modelcacheentries\", // note\n],\n\"verbs\": [\"get\", \"watch\"],";
    for (text, rust_file) in [(flow, false), (block, false), (rust, true)] {
        let verbs = cache_entry_verbs(text, rust_file);
        let expected: BTreeSet<String> = ["get", "watch"].iter().map(|s| s.to_string()).collect();
        assert_eq!(verbs.get(ENTRIES), Some(&expected), "format: {text}");
    }
}
