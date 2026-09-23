// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

//! Static checks guarding against MODEL_EXPRESS_SERVER_PORT drifting away from
//! `service.port`.
//!
//! `service.port` drives the gRPC listener's containerPort, the Service port
//! and the probes. Before this fix, `MODEL_EXPRESS_SERVER_PORT` (the env var
//! the server's `--port` clap arg actually reads) was a fourth, independent
//! literal in `values.yaml`, so overriding only one of the two desynced the
//! port Kubernetes advertises from the port the server actually binds.
//!
//! These operate on the template/value source rather than on `helm template`
//! output so they run under plain `cargo test` without a helm binary.

use std::fs;
use std::path::{Path, PathBuf};

fn helm_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace-tests has a parent directory")
        .join("helm")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn server_port_env_var_is_not_a_literal_in_any_values_file() {
    let dir = helm_dir();
    let mut checked = 0;
    let mut offenders = Vec::new();

    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        let is_values_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("values.yaml") || n.ends_with("values.yml"));
        if !is_values_file {
            continue;
        }
        checked += 1;
        let contents = read(&path);
        if contents.contains("MODEL_EXPRESS_SERVER_PORT:") {
            offenders.push(path.display().to_string());
        }
    }

    assert!(
        offenders.is_empty(),
        "MODEL_EXPRESS_SERVER_PORT must not be set as a literal env value; \
         service.port is the single source of truth: {offenders:#?}"
    );
    assert!(
        checked > 0,
        "no values files found under {} - the scan is broken",
        dir.display()
    );
}

#[test]
fn deployment_template_derives_server_port_from_service_port() {
    let contents = read(&helm_dir().join("templates/deployment.yaml"));

    assert!(
        contents.contains("$serverPort := .Values.service.port"),
        "deployment.yaml should derive a $serverPort variable from .Values.service.port"
    );
    assert!(
        contents.contains("containerPort: {{ $serverPort }}"),
        "the http containerPort should reuse $serverPort rather than a separate literal"
    );
    assert!(
        contents.contains(
            "- name: MODEL_EXPRESS_SERVER_PORT\n              value: {{ $serverPort | quote }}"
        ),
        "the MODEL_EXPRESS_SERVER_PORT env value should be derived from $serverPort"
    );
}

#[test]
fn deployment_template_rejects_direct_server_port_env_override() {
    let contents = read(&helm_dir().join("templates/deployment.yaml"));

    assert!(
        contents.contains(r#"hasKey .Values.env "MODEL_EXPRESS_SERVER_PORT""#)
            && contents
                .contains("fail \"Set service.port rather than MODEL_EXPRESS_SERVER_PORT directly"),
        "deployment.yaml should fail the render when MODEL_EXPRESS_SERVER_PORT is set directly via env or extraEnv"
    );
}
