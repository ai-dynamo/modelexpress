// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the Redis-backed Refit gRPC service.
//!
//! Run with a Redis 7 server:
//!
//! ```sh
//! REDIS_URL=redis://localhost:6379 cargo test -p model-express-workspace-tests \
//!     --test refit_service_redis -- --include-ignored
//! ```

#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::num::NonZeroU16;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use modelexpress_common::grpc::refit::{
    CreateTrainerMeshRequest, CreateWeightVersionRequest, CreateWeightVersionShardRequest,
    DeleteTrainerMeshRequest, DeleteVersionLeaseRequest, DeleteWeightVersionRequest,
    DeleteWeightVersionShardRequest, GetTrainerMeshRequest, GetWeightVersionRequest,
    GetWeightVersionShardManifestRequest, GetWeightVersionShardManifestResponse,
    ListWeightVersionShardsRequest, ListWeightVersionsRequest, ObjectStorageSource,
    ObjectStorageType, RegisterVersionLeaseRequest, RegisterWorkerRequest, TrainerTensorsMetadata,
    UpdateTrainerMeshRequest, UpdateWeightVersionStateRequest, WeightPayloadFormat, WeightVersion,
    WeightVersionShard, WeightVersionState, WorkerRegistration, WorkerRole,
    refit_service_client::RefitServiceClient,
    refit_worker_service_server::{RefitWorkerService, RefitWorkerServiceServer},
};
use modelexpress_server::backend_config::BackendConfig;
use modelexpress_server::config::ServerConfig;
use modelexpress_server::run_server;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic_health::pb::{
    HealthCheckRequest, health_check_response::ServingStatus, health_client::HealthClient,
};

type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

struct BoundWorker {
    manifest: Vec<u8>,
    calls: Arc<AtomicUsize>,
    activity: Option<Arc<BindingActivity>>,
}

#[derive(Default)]
struct BindingActivity {
    active: AtomicUsize,
    peak: AtomicUsize,
}

#[tonic::async_trait]
impl RefitWorkerService for BoundWorker {
    async fn get_weight_version_shard_manifest(
        &self,
        _request: tonic::Request<GetWeightVersionShardManifestRequest>,
    ) -> Result<tonic::Response<GetWeightVersionShardManifestResponse>, tonic::Status> {
        use sha2::{Digest, Sha256};
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(activity) = &self.activity {
            let active = activity
                .active
                .fetch_add(1, Ordering::SeqCst)
                .saturating_add(1);
            activity.peak.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(50)).await;
            activity.active.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(tonic::Response::new(
            GetWeightVersionShardManifestResponse {
                manifest: self.manifest.clone(),
                manifest_digest: format!("{:x}", Sha256::digest(&self.manifest)),
            },
        ))
    }
}

async fn bound_worker() -> (
    TrainerTensorsMetadata,
    oneshot::Sender<()>,
    JoinHandle<ServerResult>,
) {
    bound_worker_shard(0, 4).await
}

async fn bound_worker_shard(
    offset: u64,
    length: u64,
) -> (
    TrainerTensorsMetadata,
    oneshot::Sender<()>,
    JoinHandle<ServerResult>,
) {
    bound_worker_counted(offset, length, Arc::new(AtomicUsize::new(0))).await
}

async fn bound_worker_counted(
    offset: u64,
    length: u64,
    calls: Arc<AtomicUsize>,
) -> (
    TrainerTensorsMetadata,
    oneshot::Sender<()>,
    JoinHandle<ServerResult>,
) {
    bound_worker_observed(offset, length, calls, None).await
}

async fn bound_worker_observed(
    offset: u64,
    length: u64,
    calls: Arc<AtomicUsize>,
    activity: Option<Arc<BindingActivity>>,
) -> (
    TrainerTensorsMetadata,
    oneshot::Sender<()>,
    JoinHandle<ServerResult>,
) {
    use sha2::{Digest, Sha256};
    let manifest = serde_json::to_vec(&serde_json::json!({"tensors": [{
        "name": "weight", "dtype": "torch.bfloat16", "elsize": 2,
        "full_shape": [4], "shards": [{"shard_offset": [offset], "shape": [length]}],
    }]}))
    .expect("encode bound coverage");
    let logical_shard_id = format!("{:x}", Sha256::digest(&manifest));
    let port = free_port();
    let (tx, rx) = oneshot::channel();
    let worker = BoundWorker {
        manifest,
        calls,
        activity,
    };
    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RefitWorkerServiceServer::new(worker))
            .serve_with_shutdown(([127, 0, 0, 1], port).into(), async {
                let _ = rx.await;
            })
            .await?;
        Ok(())
    });
    for _ in 0..100 {
        if tonic::transport::Endpoint::new(format!("http://127.0.0.1:{port}"))
            .expect("endpoint")
            .connect()
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    (
        TrainerTensorsMetadata {
            logical_shard_id,
            metadata_endpoint: format!("127.0.0.1:{port}"),
        },
        tx,
        handle,
    )
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn unique_id(tag: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    format!("refit-test-{tag}-{nanos}")
}

fn start_server(port: u16, redis_url: &str) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let mut config = ServerConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = NonZeroU16::new(port).expect("port is non-zero");
    config.cache.eviction.enabled = false;

    let backend = BackendConfig::Redis {
        url: redis_url.to_string(),
    };
    let (tx, rx) = oneshot::channel();
    let handle = tokio::spawn(run_server(config, backend, async move {
        let _ = rx.await;
    }));
    (tx, handle)
}

async fn connect(port: u16) -> RefitServiceClient<tonic::transport::Channel> {
    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if let Ok(client) = RefitServiceClient::connect(endpoint.clone()).await {
            return client;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server on port {port} never became reachable");
}

async fn stop(tx: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    let _ = tx.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server did not stop")
        .expect("server task panicked")
        .expect("server failed");
}

fn trainer(worker_id: &str) -> RegisterWorkerRequest {
    worker(worker_id, WorkerRole::Trainer, 60)
}

fn mesh_trainer(worker_id: &str, metadata: &TrainerTensorsMetadata) -> RegisterWorkerRequest {
    let mut request = trainer(worker_id);
    request
        .worker
        .as_mut()
        .expect("trainer registration")
        .refit_endpoint = metadata.metadata_endpoint.clone();
    request
}

fn mesh_shard(
    version_id: &str,
    worker_id: &str,
    metadata: &TrainerTensorsMetadata,
) -> WeightVersionShard {
    let mut publication = shard(version_id, &metadata.logical_shard_id, worker_id);
    publication.manifest_endpoint = metadata.metadata_endpoint.clone();
    publication.manifest_digest = metadata.logical_shard_id.clone();
    publication.tensor_count = 1;
    publication.total_bytes = 8;
    publication
}

async fn fresh_mesh_version(
    client: &mut RefitServiceClient<tonic::transport::Channel>,
    mesh_id: &str,
) -> WeightVersion {
    client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("new-mesh-generation"),
            trainer_mesh_id: Some(mesh_id.to_string()),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            state: WeightVersionState::Staging.into(),
            ..Default::default()
        })
        .await
        .expect("create version for current mesh generation")
        .into_inner()
        .version
        .expect("version")
}

async fn reject_stale_publication(
    client: &mut RefitServiceClient<tonic::transport::Channel>,
    publication: WeightVersionShard,
) {
    let uid = publication.version_id.clone();
    let before = client
        .get_weight_version(GetWeightVersionRequest { uid: uid.clone() })
        .await
        .expect("version before stale publication")
        .into_inner()
        .version;
    let shards = client
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: uid.clone(),
        })
        .await
        .expect("publications before stale attempt")
        .into_inner()
        .shards;
    let failure = client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(publication),
        })
        .await
        .expect_err("new mesh cannot publish an old version");
    assert_eq!(failure.code(), tonic::Code::FailedPrecondition);
    assert!(failure.message().contains("generation"));
    assert_eq!(
        client
            .get_weight_version(GetWeightVersionRequest { uid: uid.clone() })
            .await
            .expect("read immutable version")
            .into_inner()
            .version,
        before
    );
    assert_eq!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest { version_id: uid })
            .await
            .expect("publications unchanged")
            .into_inner()
            .shards,
        shards
    );
}

fn worker(worker_id: &str, role: WorkerRole, ttl_seconds: u32) -> RegisterWorkerRequest {
    RegisterWorkerRequest {
        worker: Some(WorkerRegistration {
            worker_id: worker_id.to_string(),
            role: role.into(),
            model_name: "test/model".to_string(),
            expires_at_unix_ms: 0,
            refit_endpoint: String::new(),
        }),
        ttl_seconds,
    }
}

fn shard(version_id: &str, logical_shard_id: &str, worker_id: &str) -> WeightVersionShard {
    WeightVersionShard {
        version_id: version_id.to_string(),
        logical_shard_id: logical_shard_id.to_string(),
        worker_id: worker_id.to_string(),
        tensor_count: 10,
        total_bytes: 1024,
        manifest_digest: format!("digest-{logical_shard_id}"),
        manifest_endpoint: format!("{worker_id}:9000"),
    }
}

fn s3_source(uri: &str) -> ObjectStorageSource {
    ObjectStorageSource {
        uri: uri.to_string(),
        storage_type: ObjectStorageType::S3.into(),
    }
}

async fn update_state(
    client: &mut RefitServiceClient<tonic::transport::Channel>,
    uid: &str,
    state: WeightVersionState,
) -> Result<modelexpress_common::grpc::refit::UpdateWeightVersionStateResponse, Box<tonic::Status>>
{
    client
        .update_weight_version_state(UpdateWeightVersionStateRequest {
            uid: uid.to_string(),
            state: state.into(),
        })
        .await
        .map(tonic::Response::into_inner)
        .map_err(Box::new)
}

async fn mesh_for_workers(
    client: &mut RefitServiceClient<tonic::transport::Channel>,
    workers: HashMap<String, TrainerTensorsMetadata>,
) -> String {
    client
        .create_trainer_mesh(CreateTrainerMeshRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("trainer-mesh"),
            workers,
        })
        .await
        .expect("create trainer mesh")
        .into_inner()
        .mesh
        .expect("mesh in response")
        .mesh_id
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn trainer_mesh_membership_is_shared_and_generation_checked() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port_a = free_port();
    let port_b = free_port();
    let (stop_a, server_a) = start_server(port_a, &redis_url);
    let (stop_b, server_b) = start_server(port_b, &redis_url);
    let mut client_a = connect(port_a).await;
    let mut client_b = connect(port_b).await;
    let old_worker = unique_id("mesh-worker-old");
    let new_worker = unique_id("mesh-worker-new");
    let (metadata, stop_worker, worker_server) = bound_worker().await;
    client_a
        .register_worker(mesh_trainer(&old_worker, &metadata))
        .await
        .expect("register initial trainer");
    client_a
        .register_worker(mesh_trainer(&new_worker, &metadata))
        .await
        .expect("register replacement trainer");
    let request = CreateTrainerMeshRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("mesh-request"),
        workers: HashMap::from([(old_worker, metadata.clone())]),
    };
    let mesh = client_a
        .create_trainer_mesh(request.clone())
        .await
        .expect("create mesh")
        .into_inner()
        .mesh
        .expect("mesh in response");
    assert_eq!(mesh.generation, 1);
    let repeated = client_b
        .create_trainer_mesh(request.clone())
        .await
        .expect("idempotent create")
        .into_inner()
        .mesh
        .expect("mesh in response");
    assert_eq!(repeated, mesh);
    let unchanged = client_b
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: mesh.generation,
            workers: request.workers.clone(),
        })
        .await
        .expect("unchanged membership")
        .into_inner()
        .mesh
        .expect("mesh in response");
    assert_eq!(unchanged, mesh);
    let mut conflicting = request.clone();
    conflicting.workers = HashMap::from([(new_worker.clone(), metadata.clone())]);
    assert_eq!(
        client_a
            .create_trainer_mesh(conflicting)
            .await
            .expect_err("conflicting idempotency key")
            .code(),
        tonic::Code::AlreadyExists
    );
    let updated = client_b
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: mesh.generation,
            workers: HashMap::from([(new_worker.clone(), metadata.clone())]),
        })
        .await
        .expect("replace worker")
        .into_inner()
        .mesh
        .expect("mesh in response");
    assert_eq!(updated.generation, 2);
    let observed = client_a
        .get_trainer_mesh(GetTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
        })
        .await
        .expect("read updated mesh")
        .into_inner()
        .mesh
        .expect("mesh in response");
    assert_eq!(observed, updated);
    let retried = client_a
        .create_trainer_mesh(request.clone())
        .await
        .expect("creation retry after membership update")
        .into_inner()
        .mesh
        .expect("mesh in response");
    assert_eq!(retried, updated);
    let stale = client_a
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: 1,
            workers: request.workers.clone(),
        })
        .await
        .expect_err("stale generation must fail");
    assert_eq!(stale.code(), tonic::Code::FailedPrecondition);
    let unregistered = client_a
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: updated.generation,
            workers: HashMap::from([(unique_id("unregistered"), metadata)]),
        })
        .await
        .expect_err("unregistered replacement must fail");
    assert_eq!(unregistered.code(), tonic::Code::FailedPrecondition);
    client_a
        .delete_trainer_mesh(DeleteTrainerMeshRequest {
            mesh_id: mesh.mesh_id,
        })
        .await
        .expect("delete unused mesh");
    stop(stop_worker, worker_server).await;
    stop(stop_a, server_a).await;
    stop(stop_b, server_b).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn mesh_linked_versions_publish_and_discover_declared_trainers() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let trainer_id = unique_id("mesh-trainer");
    let (metadata, stop_worker, worker_server) = bound_worker().await;
    let logical_shard_id = metadata.logical_shard_id.clone();
    client
        .register_worker(mesh_trainer(&trainer_id, &metadata))
        .await
        .expect("register mesh trainer");
    let mesh = client
        .create_trainer_mesh(CreateTrainerMeshRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("linked-mesh"),
            workers: HashMap::from([(trainer_id.clone(), metadata.clone())]),
        })
        .await
        .expect("create mesh")
        .into_inner()
        .mesh
        .expect("mesh in response");
    let request = CreateWeightVersionRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("linked-version"),
        payload_format: WeightPayloadFormat::FullTensor.into(),
        base_version_id: None,
        object_storage: None,
        state: WeightVersionState::Staging.into(),
        uid: None,
        trainer_mesh_id: Some(mesh.mesh_id.clone()),
        version_number: Some(7),
    };
    let mut missing_mesh = request.clone();
    missing_mesh.idempotency_key = unique_id("missing-mesh");
    missing_mesh.trainer_mesh_id = Some(unique_id("absent-mesh"));
    assert_eq!(
        client
            .create_weight_version(missing_mesh)
            .await
            .expect_err("mesh must exist")
            .code(),
        tonic::Code::NotFound
    );
    let mut mismatched_model = request.clone();
    mismatched_model.model_name = "other/model".to_string();
    assert_eq!(
        client
            .create_weight_version(mismatched_model)
            .await
            .expect_err("mesh model must match")
            .code(),
        tonic::Code::FailedPrecondition
    );
    let version = client
        .create_weight_version(request.clone())
        .await
        .expect("create linked version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(
        version.trainer_mesh_id.as_deref(),
        Some(mesh.mesh_id.as_str())
    );
    assert_eq!(version.version_number, Some(7));
    assert_eq!(
        client
            .delete_trainer_mesh(DeleteTrainerMeshRequest {
                mesh_id: format!("versions:{}", mesh.mesh_id),
            })
            .await
            .expect_err("caller IDs cannot alias mesh version-reference sets")
            .code(),
        tonic::Code::NotFound
    );
    for (model_name, trainer_mesh_id) in [
        ("other/model", Some(mesh.mesh_id.clone())),
        ("test/model", Some(unique_id("absent-list-mesh"))),
    ] {
        assert!(
            client
                .list_weight_versions(ListWeightVersionsRequest {
                    model_name: model_name.to_string(),
                    trainer_mesh_id,
                })
                .await
                .expect("list without matching versions")
                .into_inner()
                .versions
                .is_empty()
        );
    }
    assert_eq!(
        client
            .list_weight_versions(ListWeightVersionsRequest {
                model_name: "test/model".to_string(),
                trainer_mesh_id: Some(mesh.mesh_id.clone()),
            })
            .await
            .expect("list linked version")
            .into_inner()
            .versions,
        vec![version.clone()]
    );
    assert_eq!(
        client
            .create_weight_version(request)
            .await
            .expect("idempotent linked creation")
            .into_inner()
            .version,
        Some(version.clone())
    );
    assert_eq!(
        client
            .delete_trainer_mesh(DeleteTrainerMeshRequest {
                mesh_id: mesh.mesh_id.clone(),
            })
            .await
            .expect_err("active version retains mesh")
            .code(),
        tonic::Code::FailedPrecondition
    );
    assert!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: version.uid.clone(),
            })
            .await
            .expect("empty mesh-backed publication")
            .into_inner()
            .shards
            .is_empty()
    );
    assert_eq!(
        client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(shard(&version.uid, &logical_shard_id, "legacy-worker")),
            })
            .await
            .expect_err("nonmember publication must fail closed")
            .code(),
        tonic::Code::FailedPrecondition
    );
    let published = mesh_shard(&version.uid, &trainer_id, &metadata);
    let ready = client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(published.clone()),
        })
        .await
        .expect("mesh trainer publishes")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(ready.state, WeightVersionState::Ready as i32);
    assert_eq!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: version.uid.clone(),
            })
            .await
            .expect("discover mesh publication")
            .into_inner()
            .shards,
        vec![published]
    );
    let replacement_id = unique_id("mesh-replacement");
    client
        .register_worker(mesh_trainer(&replacement_id, &metadata))
        .await
        .expect("register replacement trainer");
    client
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: mesh.generation,
            workers: HashMap::from([(replacement_id.clone(), metadata.clone())]),
        })
        .await
        .expect("replace mesh trainer");
    assert!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: version.uid.clone(),
            })
            .await
            .expect("exclude replaced trainer")
            .into_inner()
            .shards
            .is_empty()
    );
    assert_eq!(
        client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(shard(&version.uid, &logical_shard_id, &trainer_id)),
            })
            .await
            .expect_err("replaced trainer is fenced")
            .code(),
        tonic::Code::FailedPrecondition
    );
    reject_stale_publication(
        &mut client,
        mesh_shard(&version.uid, &replacement_id, &metadata),
    )
    .await;
    let fresh = fresh_mesh_version(&mut client, &mesh.mesh_id).await;
    assert_eq!(
        fresh.trainer_mesh_generation,
        mesh.generation
            .checked_add(1)
            .expect("generation increment")
    );
    let replacement = mesh_shard(&fresh.uid, &replacement_id, &metadata);
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(replacement.clone()),
        })
        .await
        .expect("replacement publishes");
    assert_eq!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: fresh.uid.clone(),
            })
            .await
            .expect("discover replacement")
            .into_inner()
            .shards,
        vec![replacement]
    );
    assert_eq!(
        update_state(&mut client, &fresh.uid, WeightVersionState::Ready)
            .await
            .expect_err("manual readiness must fail")
            .code(),
        tonic::Code::FailedPrecondition
    );
    client
        .delete_weight_version_shard(DeleteWeightVersionShardRequest {
            version_id: fresh.uid.clone(),
            logical_shard_id: logical_shard_id.clone(),
            worker_id: replacement_id,
        })
        .await
        .expect("release replacement before retirement");
    client
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: fresh.uid.clone(),
        })
        .await
        .expect("retire new generation version");
    client
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: version.uid.clone(),
        })
        .await
        .expect("retire linked version");
    assert_eq!(
        client
            .delete_weight_version_shard(DeleteWeightVersionShardRequest {
                version_id: version.uid.clone(),
                logical_shard_id: logical_shard_id.clone(),
                worker_id: trainer_id,
            })
            .await
            .expect_err("mesh update already retired the original publication")
            .code(),
        tonic::Code::NotFound
    );
    client
        .delete_trainer_mesh(DeleteTrainerMeshRequest {
            mesh_id: mesh.mesh_id,
        })
        .await
        .expect("retired version releases mesh");
    stop(shutdown, server).await;
    stop(stop_worker, worker_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn staged_mesh_readiness_requires_current_publications() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let (first, stop_first, first_server) = bound_worker_shard(0, 2).await;
    let (second, stop_second, second_server) = bound_worker_shard(2, 2).await;
    for scenario in ["released", "replaced", "expired", "replica"] {
        let first_id = unique_id(scenario);
        let second_id = unique_id("second-shard");
        let mut registration = mesh_trainer(&first_id, &first);
        if scenario == "expired" {
            registration.ttl_seconds = 1;
        }
        client
            .register_worker(registration)
            .await
            .expect("register first trainer");
        client
            .register_worker(mesh_trainer(&second_id, &second))
            .await
            .expect("register second trainer");
        let mut workers = HashMap::from([
            (first_id.clone(), first.clone()),
            (second_id.clone(), second.clone()),
        ]);
        let replica_id = unique_id("first-shard-replica");
        if scenario == "replica" {
            client
                .register_worker(mesh_trainer(&replica_id, &first))
                .await
                .expect("register replica");
            workers.insert(replica_id.clone(), first.clone());
        }
        let mesh = client
            .create_trainer_mesh(CreateTrainerMeshRequest {
                model_name: "test/model".to_string(),
                idempotency_key: unique_id("staged-mesh"),
                workers,
            })
            .await
            .expect("create complete two-shard mesh")
            .into_inner()
            .mesh
            .expect("mesh");
        let mut version = client
            .create_weight_version(CreateWeightVersionRequest {
                model_name: "test/model".to_string(),
                idempotency_key: unique_id("staged-mesh-version"),
                trainer_mesh_id: Some(mesh.mesh_id.clone()),
                payload_format: WeightPayloadFormat::FullTensor.into(),
                state: WeightVersionState::Staging.into(),
                ..Default::default()
            })
            .await
            .expect("create version")
            .into_inner()
            .version
            .expect("version");
        let mut first_publication = mesh_shard(&version.uid, &first_id, &first);
        first_publication.total_bytes = 4;
        client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(first_publication.clone()),
            })
            .await
            .expect("publish first shard");
        if scenario == "replica" {
            let mut publication = first_publication.clone();
            publication.worker_id = replica_id;
            let pending = client
                .create_weight_version_shard(CreateWeightVersionShardRequest {
                    shard: Some(publication),
                })
                .await
                .expect("publish replica")
                .into_inner()
                .version
                .expect("version");
            assert_eq!(pending.state, WeightVersionState::Staging as i32);
        }
        let mut current_first = first_id.clone();
        match scenario {
            "released" | "replica" => {
                client
                    .delete_weight_version_shard(DeleteWeightVersionShardRequest {
                        version_id: version.uid.clone(),
                        logical_shard_id: first.logical_shard_id.clone(),
                        worker_id: first_id.clone(),
                    })
                    .await
                    .expect("release first shard while staging");
            }
            "replaced" => {
                current_first = unique_id("replacement-shard");
                client
                    .register_worker(mesh_trainer(&current_first, &first))
                    .await
                    .expect("register replacement");
                client
                    .update_trainer_mesh(UpdateTrainerMeshRequest {
                        mesh_id: mesh.mesh_id.clone(),
                        expected_generation: mesh.generation,
                        workers: HashMap::from([
                            (current_first.clone(), first.clone()),
                            (second_id.clone(), second.clone()),
                        ]),
                    })
                    .await
                    .expect("replace first trainer");
                let mut stale = mesh_shard(&version.uid, &second_id, &second);
                stale.total_bytes = 4;
                reject_stale_publication(&mut client, stale).await;
                assert_eq!(version.trainer_mesh_generation, mesh.generation);
                version = fresh_mesh_version(&mut client, &mesh.mesh_id).await;
                assert_eq!(
                    version.trainer_mesh_generation,
                    mesh.generation
                        .checked_add(1)
                        .expect("generation increment")
                );
                first_publication.version_id = version.uid.clone();
            }
            "expired" => tokio::time::sleep(Duration::from_millis(1200)).await,
            _ => unreachable!("fixed test scenarios"),
        }
        let mut second_publication = mesh_shard(&version.uid, &second_id, &second);
        second_publication.total_bytes = 4;
        let pending = client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(second_publication),
            })
            .await
            .expect("publish second shard")
            .into_inner()
            .version
            .expect("version");
        let expected_state = if scenario == "replica" {
            WeightVersionState::Ready
        } else {
            WeightVersionState::Staging
        };
        assert_eq!(pending.state, expected_state as i32, "{scenario}");
        let discovered = client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: version.uid.clone(),
            })
            .await
            .expect("discover current publications")
            .into_inner()
            .shards;
        assert_eq!(
            discovered.len(),
            if scenario == "replica" { 2 } else { 1 },
            "{scenario}"
        );
        client
            .register_worker(mesh_trainer(&current_first, &first))
            .await
            .expect("renew current trainer");
        first_publication.worker_id = current_first;
        let ready = client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(first_publication),
            })
            .await
            .expect("publish current first shard")
            .into_inner()
            .version
            .expect("version");
        assert_eq!(ready.state, WeightVersionState::Ready as i32, "{scenario}");
    }
    stop(shutdown, server).await;
    stop(stop_first, first_server).await;
    stop(stop_second, second_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn mesh_endpoint_rebinding_retires_old_publication_after_readers_drain() {
    mesh_worker_rebinding_retires_publication(false).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn removed_mesh_worker_publications_retire_after_readers_drain() {
    mesh_worker_rebinding_retires_publication(true).await;
}

async fn mesh_worker_rebinding_retires_publication(replace_worker_id: bool) {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let (old, stop_old, old_server) = bound_worker().await;
    let (new, stop_new, new_server) = bound_worker().await;
    assert_eq!(old.logical_shard_id, new.logical_shard_id);
    let trainer_id = unique_id("rebound-trainer");
    let replacement_id = if replace_worker_id {
        unique_id("replacement-trainer")
    } else {
        trainer_id.clone()
    };
    let mut registration = mesh_trainer(&trainer_id, &old);
    registration.ttl_seconds = 1;
    client
        .register_worker(registration)
        .await
        .expect("register old endpoint");
    let mesh = client
        .create_trainer_mesh(CreateTrainerMeshRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("rebound-mesh"),
            workers: HashMap::from([(trainer_id.clone(), old.clone())]),
        })
        .await
        .expect("create mesh")
        .into_inner()
        .mesh
        .expect("mesh");
    let version = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("rebound-version"),
            trainer_mesh_id: Some(mesh.mesh_id.clone()),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            state: WeightVersionState::Staging.into(),
            ..Default::default()
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version");
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(mesh_shard(&version.uid, &trainer_id, &old)),
        })
        .await
        .expect("publish old endpoint");
    let generator_id = unique_id("rebound-reader");
    client
        .register_worker(worker(&generator_id, WorkerRole::Generator, 60))
        .await
        .expect("register reader");
    let lease = client
        .register_version_lease(RegisterVersionLeaseRequest {
            version_id: version.uid.clone(),
            worker_id: generator_id,
            ttl_seconds: 60,
        })
        .await
        .expect("acquire reader lease")
        .into_inner()
        .lease
        .expect("lease");
    tokio::time::sleep(Duration::from_millis(1200)).await;
    client
        .register_worker(mesh_trainer(&replacement_id, &new))
        .await
        .expect("register replacement endpoint");
    let update = UpdateTrainerMeshRequest {
        mesh_id: mesh.mesh_id.clone(),
        expected_generation: mesh.generation,
        workers: HashMap::from([(replacement_id.clone(), new.clone())]),
    };
    assert_eq!(
        client
            .update_trainer_mesh(update.clone())
            .await
            .expect_err("active reader prevents retirement")
            .code(),
        tonic::Code::FailedPrecondition
    );
    let unchanged = client
        .get_trainer_mesh(GetTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
        })
        .await
        .expect("read unchanged mesh")
        .into_inner()
        .mesh
        .expect("mesh");
    assert_eq!(unchanged, mesh);
    client
        .delete_version_lease(DeleteVersionLeaseRequest {
            version_id: version.uid.clone(),
            lease_id: lease.lease_id,
            worker_id: lease.worker_id,
        })
        .await
        .expect("drain reader");
    client
        .update_trainer_mesh(update)
        .await
        .expect("rebind after drain");
    assert!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: version.uid.clone(),
            })
            .await
            .expect("old publication fenced")
            .into_inner()
            .shards
            .is_empty()
    );
    assert_eq!(
        client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(mesh_shard(&version.uid, &trainer_id, &old)),
            })
            .await
            .expect_err("old endpoint cannot republish")
            .code(),
        tonic::Code::FailedPrecondition
    );
    reject_stale_publication(&mut client, mesh_shard(&version.uid, &replacement_id, &new)).await;
    let fresh = fresh_mesh_version(&mut client, &mesh.mesh_id).await;
    assert_eq!(
        fresh.trainer_mesh_generation,
        mesh.generation
            .checked_add(1)
            .expect("generation increment")
    );
    let publication = mesh_shard(&fresh.uid, &replacement_id, &new);
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(publication.clone()),
        })
        .await
        .expect("replacement publishes without conflict");
    assert_eq!(
        client
            .list_weight_version_shards(ListWeightVersionShardsRequest {
                version_id: fresh.uid.clone()
            })
            .await
            .expect("new endpoint discovered")
            .into_inner()
            .shards,
        vec![publication]
    );
    client
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: version.uid.clone(),
        })
        .await
        .expect("release version");
    client
        .delete_weight_version_shard(DeleteWeightVersionShardRequest {
            version_id: fresh.uid.clone(),
            logical_shard_id: new.logical_shard_id,
            worker_id: replacement_id,
        })
        .await
        .expect("retire replacement publication");
    client
        .delete_weight_version(DeleteWeightVersionRequest { uid: fresh.uid })
        .await
        .expect("retire fresh generation version");
    client
        .delete_trainer_mesh(DeleteTrainerMeshRequest {
            mesh_id: mesh.mesh_id,
        })
        .await
        .expect("removed publications do not prevent mesh deletion");
    stop(shutdown, server).await;
    stop(stop_old, old_server).await;
    stop(stop_new, new_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn version_becomes_ready_across_server_replicas() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port_a = free_port();
    let port_b = free_port();
    let (stop_a, server_a) = start_server(port_a, &redis_url);
    let (stop_b, server_b) = start_server(port_b, &redis_url);
    let mut client_a = connect(port_a).await;
    let mut client_b = connect(port_b).await;
    let health_channel =
        tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port_a}"))
            .expect("valid health endpoint")
            .connect()
            .await
            .expect("connect health client");
    let mut health = HealthClient::new(health_channel);
    let health_status = health
        .check(HealthCheckRequest {
            service: "model_express.refit.RefitService".to_string(),
        })
        .await
        .expect("check Refit service health")
        .into_inner();
    assert_eq!(health_status.status, i32::from(ServingStatus::Serving));

    let worker_a = unique_id("worker-a");
    let worker_b = unique_id("worker-b");
    let (metadata_a, stop_worker_a, worker_server_a) = bound_worker_shard(0, 2).await;
    let (metadata_b, stop_worker_b, worker_server_b) = bound_worker_shard(2, 2).await;
    client_a
        .register_worker(mesh_trainer(&worker_a, &metadata_a))
        .await
        .expect("register trainer A");
    client_b
        .register_worker(mesh_trainer(&worker_b, &metadata_b))
        .await
        .expect("register trainer B");

    let mesh_id = mesh_for_workers(
        &mut client_a,
        HashMap::from([
            (worker_a.clone(), metadata_a.clone()),
            (worker_b.clone(), metadata_b.clone()),
        ]),
    )
    .await;
    let create_request = CreateWeightVersionRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("publish"),
        payload_format: WeightPayloadFormat::FullTensor.into(),
        base_version_id: None,
        object_storage: None,
        state: WeightVersionState::Staging.into(),
        uid: None,
        trainer_mesh_id: Some(mesh_id),
        version_number: None,
    };
    let create_a = client_a.create_weight_version(create_request.clone());
    let create_b = client_b.create_weight_version(create_request);
    let (version_a, version_b) = tokio::join!(create_a, create_b);
    let version = version_a
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");
    let repeated = version_b
        .expect("concurrent create through second server")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(version.uid.len(), 8);
    assert_eq!(version.state, i32::from(WeightVersionState::Staging));
    assert_eq!(repeated, version);

    let observed = client_b
        .get_weight_version(GetWeightVersionRequest {
            uid: version.uid.clone(),
        })
        .await
        .expect("second server reads version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(observed, version);

    let manual_ready = update_state(&mut client_a, &version.uid, WeightVersionState::Ready)
        .await
        .expect_err("worker-sharded readiness comes only from source coverage");
    assert_eq!(manual_ready.code(), tonic::Code::FailedPrecondition);

    let unexpected_source_slot = client_a
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(shard(&version.uid, "publisher:global-rank:9", &worker_a)),
        })
        .await
        .expect_err("publication must cover a required logical shard");
    assert_eq!(
        unexpected_source_slot.code(),
        tonic::Code::FailedPrecondition
    );

    let mut shard_a = mesh_shard(&version.uid, &worker_a, &metadata_a);
    shard_a.total_bytes = 4;
    let mut shard_b = mesh_shard(&version.uid, &worker_b, &metadata_b);
    shard_b.total_bytes = 4;
    let publish_a = client_a.create_weight_version_shard(CreateWeightVersionShardRequest {
        shard: Some(shard_a.clone()),
    });
    let publish_b = client_b.create_weight_version_shard(CreateWeightVersionShardRequest {
        shard: Some(shard_b),
    });
    let (published_a, published_b) = tokio::join!(publish_a, publish_b);
    let states = [published_a, published_b].map(|result| {
        result
            .expect("concurrent shard publication")
            .into_inner()
            .version
            .expect("version in response")
            .state
    });
    assert!(
        states.contains(&i32::from(WeightVersionState::Ready)),
        "the publication completing logical coverage must observe READY"
    );

    client_a
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(shard_a.clone()),
        })
        .await
        .expect("byte-identical repeated shard publication is idempotent");
    let mut conflicting_shard = shard_a;
    conflicting_shard.total_bytes = 2048;
    let conflict = client_a
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(conflicting_shard),
        })
        .await
        .expect_err("the same worker and logical shard cannot publish different metadata");
    assert_eq!(conflict.code(), tonic::Code::AlreadyExists);

    let ready = client_b
        .get_weight_version(GetWeightVersionRequest {
            uid: version.uid.clone(),
        })
        .await
        .expect("second server observes READY")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(ready.state, i32::from(WeightVersionState::Ready));

    let shards = client_a
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: version.uid,
        })
        .await
        .expect("list shards")
        .into_inner()
        .shards;
    let mut logical_shards = vec![metadata_a.logical_shard_id, metadata_b.logical_shard_id];
    logical_shards.sort();
    assert_eq!(
        shards
            .iter()
            .map(|shard| shard.logical_shard_id.clone())
            .collect::<Vec<_>>(),
        logical_shards
    );

    stop(stop_a, server_a).await;
    stop(stop_b, server_b).await;
    stop(stop_worker_a, worker_server_a).await;
    stop(stop_worker_b, worker_server_b).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn caller_selected_version_uid_is_unique_and_idempotency_bound() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let requested_uid = unique_id("caller-version");
    let request = CreateWeightVersionRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("caller-version-request"),
        payload_format: WeightPayloadFormat::FullTensor.into(),
        base_version_id: None,
        object_storage: Some(s3_source(
            "s3://weights/run/policy/caller-version/model.safetensors.index.json",
        )),
        state: WeightVersionState::Staging.into(),
        uid: Some(requested_uid.clone()),
        trainer_mesh_id: None,
        version_number: None,
    };

    let mut blank_uid = request.clone();
    blank_uid.uid = Some(" \t".to_string());
    blank_uid.idempotency_key = unique_id("blank-caller-version-request");
    let invalid = client
        .create_weight_version(blank_uid)
        .await
        .expect_err("a present caller UID must not be blank");
    assert_eq!(invalid.code(), tonic::Code::InvalidArgument);

    let created = client
        .create_weight_version(request.clone())
        .await
        .expect("create version with caller UID")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(created.uid, requested_uid);

    let repeated = client
        .create_weight_version(request.clone())
        .await
        .expect("an identical request is idempotent")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(repeated, created);

    let mut same_uid_different_key = request.clone();
    same_uid_different_key.idempotency_key = unique_id("different-request");
    let uid_conflict = client
        .create_weight_version(same_uid_different_key)
        .await
        .expect_err("an existing caller UID rejects a different request");
    assert_eq!(uid_conflict.code(), tonic::Code::AlreadyExists);

    let mut different_uid_same_key = request;
    different_uid_same_key.uid = Some(unique_id("different-caller-version"));
    let idempotency_conflict = client
        .create_weight_version(different_uid_same_key)
        .await
        .expect_err("idempotency cannot return a different requested UID");
    assert_eq!(idempotency_conflict.code(), tonic::Code::AlreadyExists);

    stop(shutdown, server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn caller_selected_version_uids_do_not_collide_with_derived_keys() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let worker_id = unique_id("key-boundary-worker");
    let (metadata, stop_worker, worker_server) = bound_worker().await;
    client
        .register_worker(mesh_trainer(&worker_id, &metadata))
        .await
        .expect("register trainer");

    let mesh_id = mesh_for_workers(
        &mut client,
        HashMap::from([(worker_id.clone(), metadata.clone())]),
    )
    .await;
    let version_uid = unique_id("key-boundary-version");

    let request = CreateWeightVersionRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("key-boundary-request"),
        payload_format: WeightPayloadFormat::FullTensor.into(),
        base_version_id: None,
        object_storage: None,
        state: WeightVersionState::Staging.into(),
        uid: Some(version_uid.clone()),
        trainer_mesh_id: Some(mesh_id),
        version_number: None,
    };
    client
        .create_weight_version(request.clone())
        .await
        .expect("create base version");

    let mut suffix_request = request;
    suffix_request.uid = Some(format!("{version_uid}:shards"));
    suffix_request.idempotency_key = unique_id("key-boundary-suffix-request");
    client
        .create_weight_version(suffix_request)
        .await
        .expect("create version whose UID ends with the shard-key suffix");

    let published = mesh_shard(&version_uid, &worker_id, &metadata);
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(published.clone()),
        })
        .await
        .expect("publish shard");
    let shards = client
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: version_uid,
        })
        .await
        .expect("list shards")
        .into_inner()
        .shards;
    assert_eq!(shards, vec![published]);

    stop(shutdown, server).await;
    stop(stop_worker, worker_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn s3_versions_support_staged_and_direct_ready_creation() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let uri = "s3://weights/run/policy/v42/model.safetensors.index.json";
    let staged_request = CreateWeightVersionRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("staged-s3-version"),
        payload_format: WeightPayloadFormat::FullHfCheckpoint.into(),
        base_version_id: None,
        object_storage: Some(s3_source(uri)),
        state: WeightVersionState::Staging.into(),
        uid: None,
        trainer_mesh_id: None,
        version_number: None,
    };
    let staged = client
        .create_weight_version(staged_request.clone())
        .await
        .expect("create staged S3 version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(staged.state, i32::from(WeightVersionState::Staging));
    assert_eq!(
        staged.payload_format,
        i32::from(WeightPayloadFormat::FullHfCheckpoint)
    );
    assert_eq!(
        staged
            .object_storage
            .as_ref()
            .expect("object storage source")
            .uri,
        uri
    );

    let mut cancelled_request = staged_request.clone();
    cancelled_request.idempotency_key = unique_id("cancelled-s3-version");
    cancelled_request.object_storage = Some(s3_source(
        "s3://weights/run/policy/v43/model.safetensors.index.json",
    ));
    let cancelled = client
        .create_weight_version(cancelled_request.clone())
        .await
        .expect("create S3 version to cancel")
        .into_inner()
        .version
        .expect("version in response");
    let cancelled = client
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: cancelled.uid.clone(),
        })
        .await
        .expect("cancel staged S3 version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(cancelled.state, i32::from(WeightVersionState::Releasing));
    let repeated_cancel = client
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: cancelled.uid.clone(),
        })
        .await
        .expect("repeated cancellation is idempotent")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(repeated_cancel, cancelled);
    let cancelled_ready = update_state(&mut client, &cancelled.uid, WeightVersionState::Ready)
        .await
        .expect_err("cancelled version cannot become READY");
    assert_eq!(cancelled_ready.code(), tonic::Code::FailedPrecondition);
    let repeated_cancel_create = client
        .create_weight_version(cancelled_request)
        .await
        .expect("idempotent create returns the cancelled version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(repeated_cancel_create, cancelled);

    let ready = update_state(&mut client, &staged.uid, WeightVersionState::Ready)
        .await
        .expect("mark S3 version ready")
        .version
        .expect("updated version");
    assert_eq!(ready.state, i32::from(WeightVersionState::Ready));
    let repeated = update_state(&mut client, &staged.uid, WeightVersionState::Ready)
        .await
        .expect("repeated READY update is idempotent")
        .version
        .expect("updated version");
    assert_eq!(repeated, ready);
    let backward = update_state(&mut client, &staged.uid, WeightVersionState::Staging)
        .await
        .expect_err("READY cannot transition back to STAGING");
    assert_eq!(backward.code(), tonic::Code::FailedPrecondition);

    let releasing = update_state(&mut client, &staged.uid, WeightVersionState::Releasing)
        .await
        .expect("release S3 version")
        .version
        .expect("updated version");
    assert_eq!(releasing.state, i32::from(WeightVersionState::Releasing));
    update_state(&mut client, &staged.uid, WeightVersionState::Releasing)
        .await
        .expect("repeated RELEASING update is idempotent");
    let released_backward = update_state(&mut client, &staged.uid, WeightVersionState::Ready)
        .await
        .expect_err("RELEASING cannot transition back to READY");
    assert_eq!(released_backward.code(), tonic::Code::FailedPrecondition);

    let repeated_create = client
        .create_weight_version(staged_request.clone())
        .await
        .expect("idempotent create keeps the current state")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(repeated_create, releasing);
    let mut conflicting_create = staged_request;
    conflicting_create.state = WeightVersionState::Ready.into();
    let conflict = client
        .create_weight_version(conflicting_create)
        .await
        .expect_err("idempotency key cannot change the requested initial state");
    assert_eq!(conflict.code(), tonic::Code::AlreadyExists);

    let direct = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("ready-s3-version"),
            payload_format: WeightPayloadFormat::FullHfCheckpoint.into(),
            base_version_id: None,
            object_storage: Some(s3_source(
                "s3://weights/run/policy/v43/model.safetensors.index.json",
            )),
            state: WeightVersionState::Ready.into(),
            uid: None,
            trainer_mesh_id: None,
            version_number: None,
        })
        .await
        .expect("create directly ready S3 version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(direct.state, i32::from(WeightVersionState::Ready));
    let listed = client
        .list_weight_versions(ListWeightVersionsRequest {
            model_name: "test/model".to_string(),
            trainer_mesh_id: None,
        })
        .await
        .expect("list model versions without a mesh filter")
        .into_inner()
        .versions;
    for uid in [&staged.uid, &cancelled.uid, &direct.uid] {
        assert!(listed.iter().any(|version| &version.uid == uid));
    }
    assert!(listed.windows(2).all(|pair| {
        (&pair[0].created_at_unix_ms, &pair[0].uid) >= (&pair[1].created_at_unix_ms, &pair[1].uid)
    }));

    let shards = client
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: direct.uid,
        })
        .await
        .expect("S3 version has no worker shards")
        .into_inner()
        .shards;
    assert!(shards.is_empty());

    stop(shutdown, server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn replacement_worker_can_publish_the_same_source_slot() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;

    let original_worker_id = unique_id("original-worker");
    let replacement_worker_id = unique_id("replacement-worker");
    let (metadata, stop_worker, worker_server) = bound_worker().await;
    client
        .register_worker(mesh_trainer(&original_worker_id, &metadata))
        .await
        .expect("register original worker");
    client
        .register_worker(mesh_trainer(&replacement_worker_id, &metadata))
        .await
        .expect("register replacement worker");
    let mesh_id = mesh_for_workers(
        &mut client,
        HashMap::from([
            (original_worker_id.clone(), metadata.clone()),
            (replacement_worker_id.clone(), metadata.clone()),
        ]),
    )
    .await;
    let version = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("replacement-publish"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            object_storage: None,
            state: WeightVersionState::Staging.into(),
            uid: None,
            trainer_mesh_id: Some(mesh_id),
            version_number: None,
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");

    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(mesh_shard(&version.uid, &original_worker_id, &metadata)),
        })
        .await
        .expect("original worker publishes its manifest");
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(mesh_shard(&version.uid, &replacement_worker_id, &metadata)),
        })
        .await
        .expect("replacement may publish the same logical shard");

    let publications = client
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: version.uid,
        })
        .await
        .expect("list publications")
        .into_inner()
        .shards;
    assert_eq!(publications.len(), 2);
    assert!(
        publications
            .iter()
            .all(|publication| publication.logical_shard_id == metadata.logical_shard_id)
    );

    stop(shutdown, server).await;
    stop(stop_worker, worker_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn live_lease_protects_releasing_version_shards() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port_a = free_port();
    let port_b = free_port();
    let (stop_a, server_a) = start_server(port_a, &redis_url);
    let (stop_b, server_b) = start_server(port_b, &redis_url);
    let mut client_a = connect(port_a).await;
    let mut client_b = connect(port_b).await;

    let trainer_id = unique_id("lease-trainer");
    let generator_id = unique_id("lease-generator");
    let second_generator_id = unique_id("late-generator");
    let (metadata, stop_worker, worker_server) = bound_worker().await;
    client_a
        .register_worker(mesh_trainer(&trainer_id, &metadata))
        .await
        .expect("register trainer");
    client_a
        .register_worker(worker(&generator_id, WorkerRole::Generator, 60))
        .await
        .expect("register generator");
    client_b
        .register_worker(worker(&second_generator_id, WorkerRole::Generator, 60))
        .await
        .expect("register second generator");

    let mesh_id = mesh_for_workers(
        &mut client_a,
        HashMap::from([(trainer_id.clone(), metadata.clone())]),
    )
    .await;
    let version = client_a
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("lease-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            object_storage: None,
            state: WeightVersionState::Staging.into(),
            uid: None,
            trainer_mesh_id: Some(mesh_id.clone()),
            version_number: None,
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");
    let published_shard = mesh_shard(&version.uid, &trainer_id, &metadata);
    client_a
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(published_shard.clone()),
        })
        .await
        .expect("publish complete version");

    let register_lease = RegisterVersionLeaseRequest {
        version_id: version.uid.clone(),
        worker_id: generator_id.clone(),
        ttl_seconds: 60,
    };
    let lease = client_b
        .register_version_lease(register_lease.clone())
        .await
        .expect("register lease through second server")
        .into_inner()
        .lease
        .expect("lease in response");
    let repeated = client_a
        .register_version_lease(register_lease.clone())
        .await
        .expect("repeated registration renews the lease")
        .into_inner()
        .lease
        .expect("lease in response");
    assert_eq!(repeated.lease_id, lease.lease_id);

    let releasing = client_a
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: version.uid.clone(),
        })
        .await
        .expect("logically release version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(releasing.state, i32::from(WeightVersionState::Releasing));

    let late_lease = client_b
        .register_version_lease(RegisterVersionLeaseRequest {
            version_id: version.uid.clone(),
            worker_id: second_generator_id.clone(),
            ttl_seconds: 60,
        })
        .await
        .expect_err("a releasing version rejects new consumers");
    assert_eq!(late_lease.code(), tonic::Code::FailedPrecondition);

    client_b
        .register_version_lease(register_lease)
        .await
        .expect("an existing consumer may renew while finishing");

    let protected = client_a
        .delete_weight_version_shard(DeleteWeightVersionShardRequest {
            version_id: version.uid.clone(),
            logical_shard_id: published_shard.logical_shard_id.clone(),
            worker_id: trainer_id.clone(),
        })
        .await
        .expect_err("a live lease protects every source shard");
    assert_eq!(protected.code(), tonic::Code::FailedPrecondition);

    client_b
        .delete_version_lease(DeleteVersionLeaseRequest {
            version_id: version.uid.clone(),
            lease_id: lease.lease_id,
            worker_id: generator_id.clone(),
        })
        .await
        .expect("release consumer lease");
    let deleted = client_a
        .delete_weight_version_shard(DeleteWeightVersionShardRequest {
            version_id: version.uid.clone(),
            logical_shard_id: published_shard.logical_shard_id,
            worker_id: trainer_id.clone(),
        })
        .await
        .expect("source may evict after the final lease is gone")
        .into_inner();
    assert!(deleted.deleted);

    let remaining = client_a
        .list_weight_version_shards(ListWeightVersionShardsRequest {
            version_id: version.uid,
        })
        .await
        .expect("list shards after eviction")
        .into_inner();
    assert!(remaining.shards.is_empty());

    let expiring_version = client_a
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("expiring-lease-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            object_storage: None,
            state: WeightVersionState::Staging.into(),
            uid: None,
            trainer_mesh_id: Some(mesh_id.clone()),
            version_number: None,
        })
        .await
        .expect("create version protected by an expiring lease")
        .into_inner()
        .version
        .expect("version in response");
    let expiring_shard = mesh_shard(&expiring_version.uid, &trainer_id, &metadata);
    client_a
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(expiring_shard.clone()),
        })
        .await
        .expect("publish version protected by an expiring lease");
    let expiring_lease = RegisterVersionLeaseRequest {
        version_id: expiring_version.uid.clone(),
        worker_id: generator_id.clone(),
        ttl_seconds: 1,
    };
    client_b
        .register_version_lease(expiring_lease.clone())
        .await
        .expect("register short consumer lease");
    client_a
        .delete_weight_version(DeleteWeightVersionRequest {
            uid: expiring_version.uid.clone(),
        })
        .await
        .expect("logically release version with short lease");
    tokio::time::sleep(Duration::from_millis(1_250)).await;
    let expired_re_registration = client_b
        .register_version_lease(expiring_lease)
        .await
        .expect_err("an expired lease cannot be re-registered after logical release");
    assert_eq!(
        expired_re_registration.code(),
        tonic::Code::FailedPrecondition
    );
    client_a
        .delete_weight_version_shard(DeleteWeightVersionShardRequest {
            version_id: expiring_version.uid,
            logical_shard_id: expiring_shard.logical_shard_id,
            worker_id: trainer_id.clone(),
        })
        .await
        .expect("expired lease no longer protects source shards");

    stop(stop_a, server_a).await;
    stop(stop_b, server_b).await;
    stop(stop_worker, worker_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn register_worker_refreshes_liveness_and_expires_without_renewal() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;

    let worker_id = unique_id("heartbeat-worker");
    let (metadata, stop_worker, worker_server) = bound_worker().await;
    let mut registration = mesh_trainer(&worker_id, &metadata);
    registration.ttl_seconds = 1;
    client
        .register_worker(registration.clone())
        .await
        .expect("initial registration");
    registration.ttl_seconds = 10;
    client
        .register_worker(registration)
        .await
        .expect("same registration refreshes its TTL");
    let mesh_id = mesh_for_workers(
        &mut client,
        HashMap::from([(worker_id.clone(), metadata.clone())]),
    )
    .await;
    tokio::time::sleep(Duration::from_secs(5)).await;

    let version = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("heartbeat-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            object_storage: None,
            state: WeightVersionState::Staging.into(),
            uid: None,
            trainer_mesh_id: Some(mesh_id.clone()),
            version_number: None,
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");
    let current_shard = mesh_shard(&version.uid, &worker_id, &metadata);
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(current_shard),
        })
        .await
        .expect("refreshed registration remains live past its original TTL");

    tokio::time::sleep(Duration::from_secs(6)).await;
    let expired_version = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("expired-worker-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            object_storage: None,
            state: WeightVersionState::Staging.into(),
            uid: None,
            trainer_mesh_id: Some(mesh_id.clone()),
            version_number: None,
        })
        .await
        .expect("create version after worker expiry")
        .into_inner()
        .version
        .expect("version in response");
    let expired = client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(mesh_shard(&expired_version.uid, &worker_id, &metadata)),
        })
        .await
        .expect_err("expired worker registration cannot publish a manifest");
    assert_eq!(expired.code(), tonic::Code::FailedPrecondition);

    stop(shutdown, server).await;
    stop(stop_worker, worker_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn validates_bindings_on_mesh_join_without_publication_callbacks() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let (metadata, stop_worker, worker_server) = bound_worker_counted(0, 4, calls.clone()).await;
    let trainer_id = unique_id("binding-trainer");
    client
        .register_worker(mesh_trainer(&trainer_id, &metadata))
        .await
        .expect("register trainer");
    let mut wrong = metadata.clone();
    wrong.logical_shard_id = "wrong-coverage".to_string();
    let invalid = CreateTrainerMeshRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("invalid-binding"),
        workers: HashMap::from([(trainer_id.clone(), wrong)]),
    };
    assert_eq!(
        client
            .create_trainer_mesh(invalid)
            .await
            .expect_err("mesh rejects invalid binding")
            .code(),
        tonic::Code::InvalidArgument
    );
    let creation = CreateTrainerMeshRequest {
        model_name: "test/model".to_string(),
        idempotency_key: unique_id("valid-binding"),
        workers: HashMap::from([(trainer_id.clone(), metadata.clone())]),
    };
    let mesh = client
        .create_trainer_mesh(creation.clone())
        .await
        .expect("mesh validates binding")
        .into_inner()
        .mesh
        .expect("mesh");
    let validated = calls.load(Ordering::SeqCst);
    assert_eq!(validated, 2);
    client
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id.clone(),
            expected_generation: mesh.generation,
            workers: mesh.workers.clone(),
        })
        .await
        .expect("unchanged binding needs no callback");
    assert_eq!(calls.load(Ordering::SeqCst), validated);
    stop(stop_worker, worker_server).await;
    for _ in 0..2 {
        let version = client
            .create_weight_version(CreateWeightVersionRequest {
                model_name: "test/model".to_string(),
                idempotency_key: unique_id("binding-version"),
                payload_format: WeightPayloadFormat::FullTensor.into(),
                state: WeightVersionState::Staging.into(),
                trainer_mesh_id: Some(mesh.mesh_id.clone()),
                ..Default::default()
            })
            .await
            .expect("create version")
            .into_inner()
            .version
            .expect("version");
        client
            .create_weight_version_shard(CreateWeightVersionShardRequest {
                shard: Some(mesh_shard(&version.uid, &trainer_id, &metadata)),
            })
            .await
            .expect("publication does not contact trainer manifest endpoint");
    }
    assert_eq!(calls.load(Ordering::SeqCst), validated);
    let replacement_calls = Arc::new(AtomicUsize::new(0));
    let (replacement, stop_replacement, replacement_server) =
        bound_worker_counted(0, 4, replacement_calls.clone()).await;
    let replacement_id = unique_id("replacement-binding");
    client
        .register_worker(mesh_trainer(&replacement_id, &replacement))
        .await
        .expect("register replacement");
    let mut invalid_replacement = replacement.clone();
    invalid_replacement.logical_shard_id = "wrong-coverage".to_string();
    assert_eq!(
        client
            .update_trainer_mesh(UpdateTrainerMeshRequest {
                mesh_id: mesh.mesh_id.clone(),
                expected_generation: mesh.generation,
                workers: HashMap::from([(replacement_id.clone(), invalid_replacement)]),
            })
            .await
            .expect_err("replacement binding must be valid")
            .code(),
        tonic::Code::InvalidArgument
    );
    let updated = client
        .update_trainer_mesh(UpdateTrainerMeshRequest {
            mesh_id: mesh.mesh_id,
            expected_generation: mesh.generation,
            workers: HashMap::from([(replacement_id, replacement)]),
        })
        .await
        .expect("validate replacement binding")
        .into_inner()
        .mesh
        .expect("updated mesh");
    let retried = client
        .create_trainer_mesh(creation.clone())
        .await
        .expect("creation retry does not contact retired trainer")
        .into_inner()
        .mesh
        .expect("retried mesh");
    assert_eq!(retried, updated);
    assert_eq!(calls.load(Ordering::SeqCst), validated);
    let mut conflict = creation;
    conflict
        .workers
        .values_mut()
        .next()
        .expect("original worker")
        .logical_shard_id = "different-binding".to_string();
    assert_eq!(
        client
            .create_trainer_mesh(conflict)
            .await
            .expect_err("idempotency conflicts do not contact retired trainers")
            .code(),
        tonic::Code::AlreadyExists
    );

    assert_eq!(replacement_calls.load(Ordering::SeqCst), 2);
    stop(stop_replacement, replacement_server).await;
    stop(shutdown, server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn rejects_invalid_mesh_registrations_before_binding_callbacks() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let (metadata, stop_worker, worker_server) = bound_worker_counted(0, 4, calls.clone()).await;
    let trainer_id = unique_id("preflight-trainer");
    client
        .register_worker(mesh_trainer(&trainer_id, &metadata))
        .await
        .expect("register trainer");
    let mesh = client
        .create_trainer_mesh(CreateTrainerMeshRequest {
            model_name: "test/model".to_string(),
            idempotency_key: unique_id("preflight-mesh"),
            workers: HashMap::from([(trainer_id.clone(), metadata.clone())]),
        })
        .await
        .expect("create mesh")
        .into_inner()
        .mesh
        .expect("mesh");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    for scenario in ["missing", "model", "role", "endpoint"] {
        let invalid_id = unique_id(scenario);
        if scenario != "missing" {
            let mut registration = mesh_trainer(&invalid_id, &metadata);
            let worker = registration.worker.as_mut().expect("worker");
            match scenario {
                "model" => worker.model_name = "other/model".to_string(),
                "role" => worker.role = WorkerRole::Generator.into(),
                "endpoint" => worker.refit_endpoint = "127.0.0.1:1".to_string(),
                _ => unreachable!(),
            }
            client
                .register_worker(registration)
                .await
                .expect("register invalid member");
        }
        let workers = HashMap::from([
            (trainer_id.clone(), metadata.clone()),
            (invalid_id, metadata.clone()),
        ]);
        let creation = client
            .create_trainer_mesh(CreateTrainerMeshRequest {
                model_name: "test/model".to_string(),
                idempotency_key: unique_id("preflight-rejected"),
                workers: workers.clone(),
            })
            .await
            .expect_err("reject invalid registration before fetching bindings");
        assert_eq!(
            creation.code(),
            tonic::Code::FailedPrecondition,
            "{scenario}"
        );
        let update = client
            .update_trainer_mesh(UpdateTrainerMeshRequest {
                mesh_id: mesh.mesh_id.clone(),
                expected_generation: mesh.generation,
                workers,
            })
            .await
            .expect_err("reject invalid replacement before fetching bindings");
        assert_eq!(update.code(), tonic::Code::FailedPrecondition, "{scenario}");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "{scenario}");
    }
    stop(shutdown, server).await;
    stop(stop_worker, worker_server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn validates_every_mesh_worker_with_bounded_concurrency() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (shutdown, server) = start_server(port, &redis_url);
    let mut client = connect(port).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let activity = Arc::new(BindingActivity::default());
    let (metadata, stop_worker, worker_server) =
        bound_worker_observed(0, 4, calls.clone(), Some(activity.clone())).await;
    let mut mesh = None;
    for attempt in 0..2 {
        let mut workers = HashMap::new();
        for _ in 0..64 {
            let worker_id = unique_id("concurrent-binding");
            client
                .register_worker(mesh_trainer(&worker_id, &metadata))
                .await
                .expect("register trainer");
            workers.insert(worker_id, metadata.clone());
        }
        activity.peak.store(0, Ordering::SeqCst);
        mesh = Some(if let Some(current) = mesh {
            let current: modelexpress_common::grpc::refit::TrainerMesh = current;
            client
                .update_trainer_mesh(UpdateTrainerMeshRequest {
                    mesh_id: current.mesh_id,
                    expected_generation: current.generation,
                    workers,
                })
                .await
                .expect("validate replacement workers concurrently")
                .into_inner()
                .mesh
                .expect("mesh")
        } else {
            client
                .create_trainer_mesh(CreateTrainerMeshRequest {
                    model_name: "test/model".to_string(),
                    idempotency_key: unique_id("concurrent-mesh"),
                    workers,
                })
                .await
                .expect("validate new workers concurrently")
                .into_inner()
                .mesh
                .expect("mesh")
        });
        assert_eq!(calls.load(Ordering::SeqCst), 64 * (attempt + 1));
        let peak = activity.peak.load(Ordering::SeqCst);
        assert!(peak > 1, "callbacks were serialized");
        assert!(
            peak <= 32,
            "callback concurrency exceeded the bound: {peak}"
        );
    }
    stop(shutdown, server).await;
    stop(stop_worker, worker_server).await;
}
