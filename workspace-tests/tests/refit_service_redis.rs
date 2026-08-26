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

use std::num::NonZeroU16;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use modelexpress_common::grpc::refit::{
    CreateWeightVersionRequest, CreateWeightVersionShardRequest, DeleteVersionLeaseRequest,
    DeleteWeightVersionRequest, DeleteWeightVersionShardRequest, GetWeightVersionRequest,
    ListWeightVersionShardsRequest, ObjectStorageSource, ObjectStorageType,
    RegisterVersionLeaseRequest, RegisterWorkerRequest, UpdateWeightVersionStateRequest,
    WeightPayloadFormat, WeightVersionShard, WeightVersionState, WorkerRegistration, WorkerRole,
    refit_service_client::RefitServiceClient,
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

fn worker(worker_id: &str, role: WorkerRole, ttl_seconds: u32) -> RegisterWorkerRequest {
    RegisterWorkerRequest {
        worker: Some(WorkerRegistration {
            worker_id: worker_id.to_string(),
            role: role.into(),
            model_name: "test/model".to_string(),
            expires_at_unix_ms: 0,
        }),
        ttl_seconds,
    }
}

fn shard(version_id: &str, source_slot_id: &str, worker_id: &str) -> WeightVersionShard {
    WeightVersionShard {
        version_id: version_id.to_string(),
        source_slot_id: source_slot_id.to_string(),
        worker_id: worker_id.to_string(),
        tensor_count: 10,
        total_bytes: 1024,
        manifest_digest: format!("digest-{source_slot_id}"),
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
    client_a
        .register_worker(trainer(&worker_a))
        .await
        .expect("register trainer A");
    client_b
        .register_worker(trainer(&worker_b))
        .await
        .expect("register trainer B");

    let create_request = CreateWeightVersionRequest {
        model_name: "test/model".to_string(),
        version_number: Some(42),
        idempotency_key: unique_id("publish"),
        payload_format: WeightPayloadFormat::FullTensor.into(),
        base_version_id: None,
        expected_source_slots: vec![
            "publisher:global-rank:0".to_string(),
            "publisher:global-rank:1".to_string(),
        ],
        object_storage: None,
        state: WeightVersionState::Staging.into(),
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
        .expect_err("publication must cover a required source slot");
    assert_eq!(unexpected_source_slot.code(), tonic::Code::InvalidArgument);

    let shard_a = shard(&version.uid, "publisher:global-rank:0", &worker_a);
    let publish_a = client_a.create_weight_version_shard(CreateWeightVersionShardRequest {
        shard: Some(shard_a.clone()),
    });
    let publish_b = client_b.create_weight_version_shard(CreateWeightVersionShardRequest {
        shard: Some(shard(&version.uid, "publisher:global-rank:1", &worker_b)),
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
        .expect_err("the same worker and source slot cannot publish different metadata");
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
    assert_eq!(
        shards
            .iter()
            .map(|shard| shard.source_slot_id.as_str())
            .collect::<Vec<_>>(),
        ["publisher:global-rank:0", "publisher:global-rank:1"]
    );

    stop(stop_a, server_a).await;
    stop(stop_b, server_b).await;
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
        version_number: Some(42),
        idempotency_key: unique_id("staged-s3-version"),
        payload_format: WeightPayloadFormat::FullTensor.into(),
        base_version_id: None,
        expected_source_slots: Vec::new(),
        object_storage: Some(s3_source(uri)),
        state: WeightVersionState::Staging.into(),
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
        staged
            .object_storage
            .as_ref()
            .expect("object storage source")
            .uri,
        uri
    );
    assert!(staged.expected_source_slots.is_empty());

    let mut cancelled_request = staged_request.clone();
    cancelled_request.version_number = Some(43);
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
            version_number: Some(43),
            idempotency_key: unique_id("ready-s3-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            expected_source_slots: Vec::new(),
            object_storage: Some(s3_source(
                "s3://weights/run/policy/v43/model.safetensors.index.json",
            )),
            state: WeightVersionState::Ready.into(),
        })
        .await
        .expect("create directly ready S3 version")
        .into_inner()
        .version
        .expect("version in response");
    assert_eq!(direct.state, i32::from(WeightVersionState::Ready));

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
    client
        .register_worker(trainer(&original_worker_id))
        .await
        .expect("register original worker");
    client
        .register_worker(trainer(&replacement_worker_id))
        .await
        .expect("register replacement worker");
    let version = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            version_number: None,
            idempotency_key: unique_id("replacement-publish"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            expected_source_slots: vec!["publisher:global-rank:0".to_string()],
            object_storage: None,
            state: WeightVersionState::Staging.into(),
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");

    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(shard(
                &version.uid,
                "publisher:global-rank:0",
                &original_worker_id,
            )),
        })
        .await
        .expect("original worker publishes its manifest");
    client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(shard(
                &version.uid,
                "publisher:global-rank:0",
                &replacement_worker_id,
            )),
        })
        .await
        .expect("replacement may publish the same source slot");

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
            .all(|publication| publication.source_slot_id == "publisher:global-rank:0")
    );

    stop(shutdown, server).await;
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
    client_a
        .register_worker(trainer(&trainer_id))
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

    let version = client_a
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            version_number: Some(43),
            idempotency_key: unique_id("lease-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            expected_source_slots: vec!["publisher:global-rank:0".to_string()],
            object_storage: None,
            state: WeightVersionState::Staging.into(),
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");
    let published_shard = shard(&version.uid, "publisher:global-rank:0", &trainer_id);
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
            source_slot_id: published_shard.source_slot_id.clone(),
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
            source_slot_id: published_shard.source_slot_id,
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
            version_number: Some(44),
            idempotency_key: unique_id("expiring-lease-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            expected_source_slots: vec!["publisher:global-rank:0".to_string()],
            object_storage: None,
            state: WeightVersionState::Staging.into(),
        })
        .await
        .expect("create version protected by an expiring lease")
        .into_inner()
        .version
        .expect("version in response");
    let expiring_shard = shard(
        &expiring_version.uid,
        "publisher:global-rank:0",
        &trainer_id,
    );
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
            source_slot_id: expiring_shard.source_slot_id,
            worker_id: trainer_id.clone(),
        })
        .await
        .expect("expired lease no longer protects source shards");

    stop(stop_a, server_a).await;
    stop(stop_b, server_b).await;
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
    client
        .register_worker(worker(&worker_id, WorkerRole::Trainer, 1))
        .await
        .expect("initial registration");
    client
        .register_worker(worker(&worker_id, WorkerRole::Trainer, 10))
        .await
        .expect("same registration refreshes its TTL");
    tokio::time::sleep(Duration::from_secs(5)).await;

    let version = client
        .create_weight_version(CreateWeightVersionRequest {
            model_name: "test/model".to_string(),
            version_number: None,
            idempotency_key: unique_id("heartbeat-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            expected_source_slots: vec!["publisher:global-rank:0".to_string()],
            object_storage: None,
            state: WeightVersionState::Staging.into(),
        })
        .await
        .expect("create version")
        .into_inner()
        .version
        .expect("version in response");
    let current_shard = shard(&version.uid, "publisher:global-rank:0", &worker_id);
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
            version_number: None,
            idempotency_key: unique_id("expired-worker-version"),
            payload_format: WeightPayloadFormat::FullTensor.into(),
            base_version_id: None,
            expected_source_slots: vec!["publisher:global-rank:0".to_string()],
            object_storage: None,
            state: WeightVersionState::Staging.into(),
        })
        .await
        .expect("create version after worker expiry")
        .into_inner()
        .version
        .expect("version in response");
    let expired = client
        .create_weight_version_shard(CreateWeightVersionShardRequest {
            shard: Some(shard(
                &expired_version.uid,
                "publisher:global-rank:0",
                &worker_id,
            )),
        })
        .await
        .expect_err("expired worker registration cannot publish a manifest");
    assert_eq!(expired.code(), tonic::Code::FailedPrecondition);

    stop(shutdown, server).await;
}
