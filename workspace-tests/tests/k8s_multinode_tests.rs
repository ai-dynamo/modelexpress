// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-node Kubernetes integration tests for ModelExpress.
//!
//! These tests verify cross-node gRPC file transfers by deploying server pods
//! on one node and client jobs on a different node, ensuring files transfer
//! correctly when there is no shared storage.
//!
//! All tests are `#[ignore]` and require:
//! - A multi-node Kubernetes cluster (at least 2 nodes)
//! - The `modelexpress:multinode-test` image available on all nodes
//! - KUBECONFIG pointing at the target cluster (or valid ~/.kube/config)
//!
//! Run via: `./test_multinode_k8s.sh` (builds image + runs tests)
//! Or directly: `cargo test --test k8s_multinode_tests -- --ignored`

use anyhow::{Context, Result, bail};
use k8s_openapi::api::{
    apps::v1::Deployment,
    batch::v1::Job,
    core::v1::{Namespace, Node, Pod, Service},
};
use kube::{
    Client,
    api::{Api, AttachParams, DeleteParams, ListParams, PostParams},
    config::{KubeConfigOptions, Kubeconfig},
};
use std::collections::BTreeSet;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::time::timeout;

const IMAGE: &str = "modelexpress:multinode-test";
const TEST_MODEL: &str = "hf-internal-testing/tiny-random-gpt2";
const TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a kube client, explicitly honoring `KUBECONFIG` when set.
///
/// `Client::try_default()` is supposed to respect the env var, but cargo test
/// subprocesses don't always inherit it reliably (e.g. through nested shells).
/// This helper reads the var and loads the kubeconfig from the
/// specified path, falling back to the default discovery chain otherwise.
async fn kube_client() -> Result<Client> {
    if let Ok(path) = std::env::var("KUBECONFIG") {
        let kubeconfig = Kubeconfig::read_from(&path)
            .with_context(|| format!("failed to read kubeconfig from {path}"))?;
        let config =
            kube::Config::from_custom_kubeconfig(kubeconfig, &KubeConfigOptions::default())
                .await
                .with_context(|| format!("failed to build client config from {path}"))?;
        Ok(Client::try_from(config)?)
    } else {
        Ok(Client::try_default().await?)
    }
}

/// Pick two different schedulable nodes from the cluster.
async fn select_nodes(client: &Client) -> Result<(String, String)> {
    let nodes: Api<Node> = Api::all(client.clone());
    let node_list = nodes.list(&ListParams::default()).await?;

    let schedulable: Vec<String> = node_list
        .items
        .iter()
        .filter(|n| {
            // Skip nodes marked unschedulable
            if n.spec
                .as_ref()
                .and_then(|s| s.unschedulable)
                .unwrap_or(false)
            {
                return false;
            }
            // Skip nodes with NoSchedule taints
            if n.spec
                .as_ref()
                .and_then(|s| s.taints.as_ref())
                .is_some_and(|taints| taints.iter().any(|t| t.effect == "NoSchedule"))
            {
                return false;
            }
            // Skip nodes that aren't Ready
            n.status
                .as_ref()
                .and_then(|s| s.conditions.as_ref())
                .is_some_and(|conds| {
                    conds
                        .iter()
                        .any(|c| c.type_ == "Ready" && c.status == "True")
                })
        })
        .filter_map(|n| n.metadata.name.clone())
        .collect();

    if schedulable.len() < 2 {
        bail!(
            "need at least 2 schedulable nodes, found {}",
            schedulable.len()
        );
    }

    Ok((schedulable[0].clone(), schedulable[1].clone()))
}

/// Create a namespace, returning a guard that deletes it on drop.
async fn create_namespace(client: &Client, name: &str) -> Result<NamespaceGuard> {
    let ns_api: Api<Namespace> = Api::all(client.clone());
    let ns: Namespace = serde_json::from_value(serde_json::json!({
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": { "name": name }
    }))?;

    // Delete if leftover from a previous failed run, wait until fully gone
    let _ = ns_api.delete(name, &DeleteParams::default()).await;
    while ns_api.get(name).await.is_ok() {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    ns_api.create(&PostParams::default(), &ns).await?;

    Ok(NamespaceGuard {
        client: client.clone(),
        name: name.to_string(),
    })
}

struct NamespaceGuard {
    client: Client,
    name: String,
}

impl Drop for NamespaceGuard {
    fn drop(&mut self) {
        let client = self.client.clone();
        let name = self.name.clone();
        // Best-effort cleanup - fire and forget
        tokio::spawn(async move {
            let ns_api: Api<Namespace> = Api::all(client);
            let _ = ns_api.delete(&name, &DeleteParams::default()).await;
        });
    }
}

/// Deploy a ModelExpress server on a specific node. Returns the service name.
async fn deploy_server(
    client: &Client,
    namespace: &str,
    release: &str,
    node: &str,
) -> Result<String> {
    let svc_name = format!("{release}-svc");

    // Deployment
    let dep_api: Api<Deployment> = Api::namespaced(client.clone(), namespace);
    let dep: Deployment = serde_json::from_value(serde_json::json!({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": { "name": release, "labels": { "app": release } },
        "spec": {
            "replicas": 1,
            "selector": { "matchLabels": { "app": release } },
            "template": {
                "metadata": { "labels": { "app": release } },
                "spec": {
                    "nodeName": node,
                    "containers": [{
                        "name": "server",
                        "image": IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["./modelexpress-server"],
                        "ports": [{ "containerPort": 8001 }],
                        "env": [
                            { "name": "MODEL_EXPRESS_SERVER_PORT", "value": "8001" },
                            { "name": "MODEL_EXPRESS_LOG_LEVEL", "value": "debug" },
                            { "name": "MODEL_EXPRESS_DATABASE_PATH", "value": "/tmp/models.db" },
                            { "name": "MODEL_EXPRESS_CACHE_DIRECTORY", "value": "/root" },
                            { "name": "MX_METADATA_BACKEND", "value": "kubernetes" },
                            { "name": "HOME", "value": "/root" },
                        ],
                        "readinessProbe": {
                            "tcpSocket": { "port": 8001 },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5,
                        },
                        "volumeMounts": [{ "name": "cache", "mountPath": "/root" }],
                    }],
                    "volumes": [{ "name": "cache", "emptyDir": { "sizeLimit": "4Gi" } }],
                }
            }
        }
    }))?;
    dep_api.create(&PostParams::default(), &dep).await?;

    // Service
    let svc_api: Api<Service> = Api::namespaced(client.clone(), namespace);
    let svc: Service = serde_json::from_value(serde_json::json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": { "name": &svc_name },
        "spec": {
            "selector": { "app": release },
            "ports": [{ "port": 8001, "targetPort": 8001 }],
        }
    }))?;
    svc_api.create(&PostParams::default(), &svc).await?;

    // Wait for ready
    wait_for_ready_pod(client, namespace, release).await?;

    Ok(svc_name)
}

/// Wait until at least one pod with the given app label is ready.
async fn wait_for_ready_pod(client: &Client, namespace: &str, app: &str) -> Result<()> {
    timeout(TIMEOUT, async {
        let pods: Api<Pod> = Api::namespaced(client.clone(), namespace);
        let lp = ListParams::default().labels(&format!("app={app}"));
        loop {
            let pod_list = pods.list(&lp).await?;
            let ready = pod_list.items.iter().any(|p| {
                p.status
                    .as_ref()
                    .and_then(|s| s.conditions.as_ref())
                    .map(|conds| {
                        conds
                            .iter()
                            .any(|c| c.type_ == "Ready" && c.status == "True")
                    })
                    .unwrap_or(false)
            });
            if ready {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    })
    .await
    .context(format!("timeout waiting for pod app={app}"))?
}

/// Submit a client download job pinned to a node. Returns the job name.
async fn submit_client_job(
    client: &Client,
    namespace: &str,
    job_name: &str,
    node: &str,
    service: &str,
    extra_cli_args: &str,
) -> Result<String> {
    let jobs: Api<Job> = Api::namespaced(client.clone(), namespace);

    let cli_cmd = format!(
        r#"
        /app/modelexpress-cli \
            --no-shared-storage \
            {extra_cli_args} \
            --endpoint "http://{service}:8001" \
            model download "{TEST_MODEL}"
        DOWNLOAD_RESULT=$?

        FILE_COUNT=$(find /cache -type f 2>/dev/null | wc -l)
        TOTAL_BYTES=$(du -sb /cache 2>/dev/null | cut -f1 || echo 0)

        echo "FILES=$FILE_COUNT"
        echo "BYTES=$TOTAL_BYTES"

        if [ $DOWNLOAD_RESULT -eq 0 ] && [ "$FILE_COUNT" -gt 0 ] && [ "$TOTAL_BYTES" -gt 0 ]; then
            echo "RESULT=SUCCESS"
            exit 0
        else
            echo "RESULT=FAILED"
            exit 1
        fi
        "#
    );

    let job: Job = serde_json::from_value(serde_json::json!({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": { "name": job_name },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "nodeName": node,
                    "containers": [{
                        "name": "client",
                        "image": IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/sh", "-c"],
                        "args": [cli_cmd],
                        "env": [
                            { "name": "MODEL_EXPRESS_CACHE_DIRECTORY", "value": "/cache" },
                            { "name": "HOME", "value": "/cache" },
                        ],
                        "volumeMounts": [{ "name": "cache", "mountPath": "/cache" }],
                    }],
                    "volumes": [{ "name": "cache", "emptyDir": { "sizeLimit": "2Gi" } }],
                }
            }
        }
    }))?;

    jobs.create(&PostParams::default(), &job).await?;
    Ok(job_name.to_string())
}

/// Wait for a job to complete. Returns true if succeeded, false if failed.
async fn wait_for_job(client: &Client, namespace: &str, job_name: &str) -> Result<bool> {
    timeout(TIMEOUT, async {
        let jobs: Api<Job> = Api::namespaced(client.clone(), namespace);
        loop {
            let job = jobs.get(job_name).await?;
            if let Some(status) = &job.status
                && let Some(conditions) = &status.conditions
            {
                for c in conditions {
                    if c.type_ == "Complete" && c.status == "True" {
                        return Ok(true);
                    }
                    if c.type_ == "Failed" && c.status == "True" {
                        return Ok(false);
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    })
    .await
    .context(format!("timeout waiting for job {job_name}"))?
}

/// Get logs from a job's pod.
async fn get_job_logs(client: &Client, namespace: &str, job_name: &str) -> Result<String> {
    let pods: Api<Pod> = Api::namespaced(client.clone(), namespace);
    let lp = ListParams::default().labels(&format!("job-name={job_name}"));
    let pod_list = pods.list(&lp).await?;
    let pod_name = pod_list
        .items
        .first()
        .and_then(|p| p.metadata.name.as_ref())
        .context("no pod found for job")?;

    Ok(pods.logs(pod_name, &Default::default()).await?)
}

/// Parse "FILES=N" and "BYTES=N" from job logs.
fn parse_job_results(logs: &str) -> (u64, u64) {
    let mut files = 0u64;
    let mut bytes = 0u64;
    for line in logs.lines() {
        if let Some(v) = line.strip_prefix("FILES=") {
            files = v.trim().parse().unwrap_or(0);
        }
        if let Some(v) = line.strip_prefix("BYTES=") {
            bytes = v.trim().parse().unwrap_or(0);
        }
    }
    (files, bytes)
}

/// Collect content hashes from a pod's cache directory via exec.
/// Skips .lock files and refs/ directory (HuggingFace metadata).
async fn collect_hashes_via_exec(
    client: &Client,
    namespace: &str,
    app_label: &str,
    cache_path: &str,
) -> Result<BTreeSet<String>> {
    let pods: Api<Pod> = Api::namespaced(client.clone(), namespace);
    let lp = ListParams::default().labels(&format!("app={app_label}"));
    let pod_list = pods.list(&lp).await?;
    let pod_name = pod_list
        .items
        .first()
        .and_then(|p| p.metadata.name.as_ref())
        .context("no pod found")?;

    let cmd = format!(
        "find {cache_path} -type f ! -name '*.lock' ! -path '*/refs/*' -exec md5sum {{}} \\; | awk '{{print $1}}' | sort -u"
    );

    let ap = AttachParams::default().container("server").stderr(false);

    let mut attached = pods.exec(pod_name, vec!["sh", "-c", &cmd], &ap).await?;

    let mut stdout_bytes = Vec::new();
    if let Some(mut stdout) = attached.stdout() {
        stdout.read_to_end(&mut stdout_bytes).await?;
    }
    attached.join().await?;

    let output = String::from_utf8_lossy(&stdout_bytes);
    let hashes: BTreeSet<String> = output
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    Ok(hashes)
}

/// Submit a client job that downloads and outputs checksums.
async fn submit_checksum_job(
    client: &Client,
    namespace: &str,
    job_name: &str,
    node: &str,
    service: &str,
) -> Result<String> {
    let jobs: Api<Job> = Api::namespaced(client.clone(), namespace);

    let script = format!(
        r#"
        /app/modelexpress-cli \
            --no-shared-storage \
            --endpoint "http://{service}:8001" \
            model download "{TEST_MODEL}"

        if [ $? -ne 0 ]; then echo "RESULT=FAILED"; exit 1; fi

        echo "CHECKSUMS_START"
        find /cache -type f ! -name '*.lock' ! -path '*/refs/*' -exec md5sum {{}} \; | awk '{{print $1}}' | sort -u
        echo "CHECKSUMS_END"
        echo "RESULT=SUCCESS"
        "#
    );

    let job: Job = serde_json::from_value(serde_json::json!({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": { "name": job_name },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "nodeName": node,
                    "containers": [{
                        "name": "client",
                        "image": IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/sh", "-c"],
                        "args": [script],
                        "env": [
                            { "name": "MODEL_EXPRESS_CACHE_DIRECTORY", "value": "/cache" },
                            { "name": "HOME", "value": "/cache" },
                        ],
                        "volumeMounts": [{ "name": "cache", "mountPath": "/cache" }],
                    }],
                    "volumes": [{ "name": "cache", "emptyDir": { "sizeLimit": "2Gi" } }],
                }
            }
        }
    }))?;

    jobs.create(&PostParams::default(), &job).await?;
    Ok(job_name.to_string())
}

/// Parse checksums from job logs (between CHECKSUMS_START and CHECKSUMS_END).
fn parse_checksums(logs: &str) -> BTreeSet<String> {
    let mut in_block = false;
    let mut hashes = BTreeSet::new();
    for line in logs.lines() {
        if line.contains("CHECKSUMS_START") {
            in_block = true;
            continue;
        }
        if line.contains("CHECKSUMS_END") {
            break;
        }
        if in_block {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                hashes.insert(trimmed.to_string());
            }
        }
    }
    hashes
}

/// Count files in a pod's directory via exec.
async fn count_files_in_pod(
    client: &Client,
    namespace: &str,
    app_label: &str,
    path: &str,
) -> Result<u64> {
    let pods: Api<Pod> = Api::namespaced(client.clone(), namespace);
    let lp = ListParams::default().labels(&format!("app={app_label}"));
    let pod_list = pods.list(&lp).await?;
    let pod_name = pod_list
        .items
        .first()
        .and_then(|p| p.metadata.name.as_ref())
        .context("no pod found")?;

    let cmd = format!("find {path} -type f | wc -l");
    let ap = AttachParams::default().container("server").stderr(false);

    let mut attached = pods.exec(pod_name, vec!["sh", "-c", &cmd], &ap).await?;

    let mut stdout_bytes = Vec::new();
    if let Some(mut stdout) = attached.stdout() {
        stdout.read_to_end(&mut stdout_bytes).await?;
    }
    attached.join().await?;

    let output = String::from_utf8_lossy(&stdout_bytes);
    Ok(output.trim().parse().unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Basic cross-node gRPC transfer with default chunk size.
#[tokio::test]
#[ignore = "requires multi-node k8s cluster with modelexpress:multinode-test image"]
async fn cross_node_transfer_default_chunks() -> Result<()> {
    let client = kube_client().await?;
    let (server_node, client_node) = select_nodes(&client).await?;
    let ns = "mx-test-default";
    let _guard = create_namespace(&client, ns).await?;

    let svc = deploy_server(&client, ns, "mx-server", &server_node).await?;
    submit_client_job(&client, ns, "dl-default", &client_node, &svc, "").await?;

    let success = wait_for_job(&client, ns, "dl-default").await?;
    let logs = get_job_logs(&client, ns, "dl-default").await?;
    let (files, bytes) = parse_job_results(&logs);

    assert!(success, "job failed:\n{logs}");
    assert!(files > 0, "expected files > 0, got {files}");
    assert!(bytes > 0, "expected bytes > 0, got {bytes}");

    Ok(())
}

/// Cross-node gRPC transfer with small 64KB chunks.
#[tokio::test]
#[ignore = "requires multi-node k8s cluster with modelexpress:multinode-test image"]
async fn cross_node_transfer_small_chunks() -> Result<()> {
    let client = kube_client().await?;
    let (server_node, client_node) = select_nodes(&client).await?;
    let ns = "mx-test-small";
    let _guard = create_namespace(&client, ns).await?;

    let svc = deploy_server(&client, ns, "mx-server", &server_node).await?;
    submit_client_job(
        &client,
        ns,
        "dl-small",
        &client_node,
        &svc,
        "--transfer-chunk-size 65536",
    )
    .await?;

    let success = wait_for_job(&client, ns, "dl-small").await?;
    let logs = get_job_logs(&client, ns, "dl-small").await?;
    let (files, bytes) = parse_job_results(&logs);

    assert!(success, "job failed:\n{logs}");
    assert!(files > 0, "expected files > 0, got {files}");
    assert!(bytes > 0, "expected bytes > 0, got {bytes}");

    Ok(())
}

/// Cross-node gRPC transfer with large 4MB chunks.
#[tokio::test]
#[ignore = "requires multi-node k8s cluster with modelexpress:multinode-test image"]
async fn cross_node_transfer_large_chunks() -> Result<()> {
    let client = kube_client().await?;
    let (server_node, client_node) = select_nodes(&client).await?;
    let ns = "mx-test-large";
    let _guard = create_namespace(&client, ns).await?;

    let svc = deploy_server(&client, ns, "mx-server", &server_node).await?;
    submit_client_job(
        &client,
        ns,
        "dl-large",
        &client_node,
        &svc,
        "--transfer-chunk-size 4194304",
    )
    .await?;

    let success = wait_for_job(&client, ns, "dl-large").await?;
    let logs = get_job_logs(&client, ns, "dl-large").await?;
    let (files, bytes) = parse_job_results(&logs);

    assert!(success, "job failed:\n{logs}");
    assert!(files > 0, "expected files > 0, got {files}");
    assert!(bytes > 0, "expected bytes > 0, got {bytes}");

    Ok(())
}

/// Transfer integrity: compare md5 content hashes between server cache
/// and client-received files. The file layouts differ (server uses HuggingFace
/// blob storage, client gets resolved snapshots), but content must match.
#[tokio::test]
#[ignore = "requires multi-node k8s cluster with modelexpress:multinode-test image"]
async fn cross_node_transfer_integrity() -> Result<()> {
    let client = kube_client().await?;
    let (server_node, client_node) = select_nodes(&client).await?;
    let ns = "mx-test-integrity";
    let _guard = create_namespace(&client, ns).await?;

    let svc = deploy_server(&client, ns, "mx-server", &server_node).await?;

    // Client downloads via gRPC and outputs checksums
    submit_checksum_job(&client, ns, "dl-checksums", &client_node, &svc).await?;
    let success = wait_for_job(&client, ns, "dl-checksums").await?;
    assert!(success, "checksum download job failed");

    let logs = get_job_logs(&client, ns, "dl-checksums").await?;
    let client_hashes = parse_checksums(&logs);

    // Server-side checksums via exec (skip .lock and refs/)
    let server_hashes = collect_hashes_via_exec(&client, ns, "mx-server", "/root").await?;

    assert!(!client_hashes.is_empty(), "client produced no checksums");
    assert!(!server_hashes.is_empty(), "server produced no checksums");
    assert_eq!(
        server_hashes,
        client_hashes,
        "content hash mismatch:\n  server-only: {:?}\n  client-only: {:?}",
        server_hashes.difference(&client_hashes).collect::<Vec<_>>(),
        client_hashes.difference(&server_hashes).collect::<Vec<_>>(),
    );

    Ok(())
}

/// Multi-replica: two servers on different nodes, each with independent caches.
/// Both must download and cache the model independently.
#[tokio::test]
#[ignore = "requires multi-node k8s cluster with modelexpress:multinode-test image"]
async fn multi_replica_independent_caches() -> Result<()> {
    let client = kube_client().await?;
    let (node_a, node_b) = select_nodes(&client).await?;
    let ns = "mx-test-replica";
    let _guard = create_namespace(&client, ns).await?;

    // Deploy two servers on different nodes
    let svc_a = deploy_server(&client, ns, "mx-a", &node_a).await?;
    let svc_b = deploy_server(&client, ns, "mx-b", &node_b).await?;

    // Client on node_b downloads from server on node_a
    submit_client_job(&client, ns, "dl-from-a", &node_b, &svc_a, "").await?;
    // Client on node_a downloads from server on node_b
    submit_client_job(&client, ns, "dl-from-b", &node_a, &svc_b, "").await?;

    let success_a = wait_for_job(&client, ns, "dl-from-a").await?;
    let success_b = wait_for_job(&client, ns, "dl-from-b").await?;

    assert!(success_a, "download from server A failed");
    assert!(success_b, "download from server B failed");

    // Verify both servers cached independently
    let count_a = count_files_in_pod(&client, ns, "mx-a", "/root").await?;
    let count_b = count_files_in_pod(&client, ns, "mx-b", "/root").await?;

    assert!(
        count_a > 0,
        "server A should have cached files, got {count_a}"
    );
    assert!(
        count_b > 0,
        "server B should have cached files, got {count_b}"
    );

    Ok(())
}
