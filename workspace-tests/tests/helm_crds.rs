// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

//! Static checks for the CRDs packaged with the Helm chart.

use std::fs;
use std::path::Path;

fn repo_file(path: &str) -> String {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace-tests has a parent directory");
    let path = repo_root.join(path);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn helm_and_standalone_crd_manifests_stay_in_sync() {
    let helm_crds = repo_file("helm/crds/modelexpress.nvidia.com.yaml");
    let standalone_crds = repo_file("examples/crds.yaml");

    assert_eq!(
        helm_crds, standalone_crds,
        "update the Helm and standalone CRD manifests together"
    );
    assert_eq!(
        helm_crds
            .lines()
            .filter(|line| *line == "kind: CustomResourceDefinition")
            .count(),
        2,
        "the Helm manifest must contain both ModelExpress CRDs"
    );
    assert!(helm_crds.contains("name: modelmetadatas.modelexpress.nvidia.com"));
    assert!(helm_crds.contains("name: modelcacheentries.modelexpress.nvidia.com"));
}
