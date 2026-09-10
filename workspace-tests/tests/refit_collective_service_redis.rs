// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end invariants for the Redis-backed collective refit control plane.
//!
//! Run with a Redis 7 server:
//!
//! ```sh
//! REDIS_URL=redis://localhost:6379 cargo test -p model-express-workspace-tests \
//!     --test refit_collective_service_redis -- --include-ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

use std::num::NonZeroU16;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use modelexpress_common::grpc::refit::{
    RegisterWorkerRequest, WorkerRegistration, WorkerRole, refit_service_client::RefitServiceClient,
};
use modelexpress_common::grpc::refit_collective::{
    CollectiveGroup, CollectiveGroupSpec, CollectiveGroupState, CollectiveRole, CollectiveTransfer,
    CollectiveTransferState, CreateCollectiveTransferRequest, DeleteCollectiveTransferRequest,
    GetCollectiveGroupRequest, JoinCollectiveGroupRequest, PublishGroupBootstrapRequest,
    ReportCollectiveTransferRequest, refit_collective_service_client::RefitCollectiveServiceClient,
};
use modelexpress_server::backend_config::BackendConfig;
use modelexpress_server::config::ServerConfig;
use modelexpress_server::run_server;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

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
    format!("refit-collective-test-{tag}-{nanos}")
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

async fn connect(
    port: u16,
) -> (
    RefitServiceClient<Channel>,
    RefitCollectiveServiceClient<Channel>,
) {
    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if let Ok(channel) = Channel::from_shared(endpoint.clone())
            .expect("valid endpoint")
            .connect()
            .await
        {
            return (
                RefitServiceClient::new(channel.clone()),
                RefitCollectiveServiceClient::new(channel),
            );
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

fn spec(model_name: &str, trainers: &[&str], generators: &[&str]) -> CollectiveGroupSpec {
    CollectiveGroupSpec {
        model_name: model_name.to_string(),
        expected_trainer_slots: trainers.iter().map(|slot| (*slot).to_string()).collect(),
        expected_generator_slots: generators.iter().map(|slot| (*slot).to_string()).collect(),
        source_partition_count: 1,
    }
}

async fn register(
    client: &mut RefitServiceClient<Channel>,
    model_name: &str,
    worker_id: &str,
    role: WorkerRole,
    ttl_seconds: u32,
) {
    client
        .register_worker(RegisterWorkerRequest {
            worker: Some(WorkerRegistration {
                worker_id: worker_id.to_string(),
                role: role.into(),
                model_name: model_name.to_string(),
                endpoint: format!("{worker_id}:9000"),
                expires_at_unix_ms: 0,
            }),
            ttl_seconds,
        })
        .await
        .expect("register worker");
}

fn join_request(
    spec: &CollectiveGroupSpec,
    slot_id: &str,
    worker_id: &str,
    role: CollectiveRole,
    index_in_role: u32,
) -> JoinCollectiveGroupRequest {
    JoinCollectiveGroupRequest {
        spec: Some(spec.clone()),
        slot_id: slot_id.to_string(),
        worker_id: worker_id.to_string(),
        role: role.into(),
        index_in_role,
        source_partition: (role == CollectiveRole::Trainer).then_some(0),
        plan_digest: "plan-digest".to_string(),
        plan_source: None,
    }
}

async fn join(
    client: &mut RefitCollectiveServiceClient<Channel>,
    request: JoinCollectiveGroupRequest,
) -> modelexpress_common::grpc::refit_collective::CollectiveGroupMembership {
    client
        .join_collective_group(request)
        .await
        .expect("join collective group")
        .into_inner()
}

async fn publish(
    client: &mut RefitCollectiveServiceClient<Channel>,
    group_id: &str,
    epoch: u64,
    lane_id: u32,
    worker_id: &str,
    fill: u8,
) -> CollectiveGroup {
    client
        .publish_group_bootstrap(PublishGroupBootstrapRequest {
            group_id: group_id.to_string(),
            epoch,
            lane_id,
            worker_id: worker_id.to_string(),
            nccl_unique_id: vec![fill; 128],
        })
        .await
        .expect("publish lane bootstrap")
        .into_inner()
}

async fn create_transfer(
    client: &mut RefitCollectiveServiceClient<Channel>,
    spec: &CollectiveGroupSpec,
    version_id: &str,
    idempotency_key: &str,
) -> CollectiveTransfer {
    client
        .create_collective_transfer(CreateCollectiveTransferRequest {
            spec: Some(spec.clone()),
            version_id: version_id.to_string(),
            idempotency_key: idempotency_key.to_string(),
        })
        .await
        .expect("create collective transfer")
        .into_inner()
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn admission_bootstrap_epoch_and_transfer_fences_hold() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (stop_server, server) = start_server(port, &redis_url);
    let (mut refit, mut collective) = connect(port).await;

    let model = unique_id("lifecycle");
    let group_spec = spec(&model, &["t0", "t1"], &["g0"]);
    let worker_t0 = unique_id("worker-t0");
    let worker_t1 = unique_id("worker-t1");
    let worker_g0 = unique_id("worker-g0");

    let unregistered = collective
        .join_collective_group(join_request(
            &group_spec,
            "t0",
            &worker_t0,
            CollectiveRole::Trainer,
            0,
        ))
        .await
        .expect_err("unregistered generation must not be admitted");
    assert_eq!(unregistered.code(), tonic::Code::FailedPrecondition);

    register(&mut refit, &model, &worker_t0, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &worker_t1, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &worker_g0, WorkerRole::Generator, 60).await;

    let t0 = join(
        &mut collective,
        join_request(&group_spec, "t0", &worker_t0, CollectiveRole::Trainer, 0),
    )
    .await;
    assert_eq!(t0.epoch, 1);

    let changed_assignment = collective
        .join_collective_group(join_request(
            &group_spec,
            "t0",
            &worker_t0,
            CollectiveRole::Trainer,
            1,
        ))
        .await
        .expect_err("a stable slot must not move to another rank");
    assert_eq!(changed_assignment.code(), tonic::Code::AlreadyExists);

    let unexpected = collective
        .join_collective_group(join_request(
            &group_spec,
            "not-declared",
            &worker_t1,
            CollectiveRole::Trainer,
            1,
        ))
        .await
        .expect_err("undeclared slot must not count toward readiness");
    assert_eq!(unexpected.code(), tonic::Code::InvalidArgument);

    let duplicate_rank = collective
        .join_collective_group(join_request(
            &group_spec,
            "t1",
            &worker_t1,
            CollectiveRole::Trainer,
            0,
        ))
        .await
        .expect_err("two slots must not own one NCCL rank");
    assert_eq!(duplicate_rank.code(), tonic::Code::AlreadyExists);

    join(
        &mut collective,
        join_request(&group_spec, "t1", &worker_t1, CollectiveRole::Trainer, 1),
    )
    .await;
    join(
        &mut collective,
        join_request(&group_spec, "g0", &worker_g0, CollectiveRole::Generator, 0),
    )
    .await;

    let pending = create_transfer(
        &mut collective,
        &group_spec,
        "pending-version",
        &unique_id("pending-operation"),
    )
    .await;
    let premature_report = collective
        .report_collective_transfer(ReportCollectiveTransferRequest {
            operation_id: pending.operation_id,
            group_id: pending.group_id,
            epoch: pending.epoch,
            worker_id: worker_t0.clone(),
            succeeded: true,
            message: String::new(),
        })
        .await
        .expect_err("PENDING cannot advance until the group is READY");
    assert_eq!(premature_report.code(), tonic::Code::FailedPrecondition);

    let non_leader = collective
        .publish_group_bootstrap(PublishGroupBootstrapRequest {
            group_id: t0.group_id.clone(),
            epoch: 1,
            lane_id: 0,
            worker_id: worker_g0.clone(),
            nccl_unique_id: vec![1; 128],
        })
        .await
        .expect_err("only rank zero may publish a lane bootstrap");
    assert_eq!(non_leader.code(), tonic::Code::FailedPrecondition);

    publish(&mut collective, &t0.group_id, 1, 0, &worker_t0, 1).await;
    let ready = publish(&mut collective, &t0.group_id, 1, 1, &worker_t0, 2).await;
    assert_eq!(ready.state, i32::from(CollectiveGroupState::Ready));

    publish(&mut collective, &t0.group_id, 1, 0, &worker_t0, 1).await;
    let conflicting_bootstrap = collective
        .publish_group_bootstrap(PublishGroupBootstrapRequest {
            group_id: t0.group_id.clone(),
            epoch: 1,
            lane_id: 0,
            worker_id: worker_t0.clone(),
            nccl_unique_id: vec![9; 128],
        })
        .await
        .expect_err("a lane identifier is immutable within one epoch");
    assert_eq!(conflicting_bootstrap.code(), tonic::Code::AlreadyExists);

    let replacement_t1 = unique_id("worker-t1-replacement");
    register(&mut refit, &model, &replacement_t1, WorkerRole::Trainer, 60).await;
    let replacement = join(
        &mut collective,
        join_request(
            &group_spec,
            "t1",
            &replacement_t1,
            CollectiveRole::Trainer,
            1,
        ),
    )
    .await;
    assert_eq!(replacement.epoch, 2);
    assert_eq!(replacement.state, i32::from(CollectiveGroupState::Forming));

    // A leader that has not acknowledged the replacement epoch cannot publish
    // new communicator metadata on behalf of its still-stale peers.
    let stale_leader = collective
        .publish_group_bootstrap(PublishGroupBootstrapRequest {
            group_id: t0.group_id.clone(),
            epoch: 2,
            lane_id: 0,
            worker_id: worker_t0.clone(),
            nccl_unique_id: vec![3; 128],
        })
        .await
        .expect_err("leader must rejoin the current epoch before publishing");
    assert_eq!(stale_leader.code(), tonic::Code::FailedPrecondition);
    join(
        &mut collective,
        join_request(&group_spec, "t0", &worker_t0, CollectiveRole::Trainer, 0),
    )
    .await;
    join(
        &mut collective,
        join_request(&group_spec, "g0", &worker_g0, CollectiveRole::Generator, 0),
    )
    .await;
    publish(&mut collective, &t0.group_id, 2, 0, &worker_t0, 3).await;
    publish(&mut collective, &t0.group_id, 2, 1, &worker_t0, 4).await;
    let epoch_two = collective
        .get_collective_group(GetCollectiveGroupRequest {
            group_id: t0.group_id.clone(),
        })
        .await
        .expect("read epoch-two group")
        .into_inner();
    assert_eq!(epoch_two.state, i32::from(CollectiveGroupState::Ready));

    let idem = unique_id("operation");
    let transfer = create_transfer(&mut collective, &group_spec, "version-1", &idem).await;
    let replay = create_transfer(&mut collective, &group_spec, "version-1", &idem).await;
    assert_eq!(replay.operation_id, transfer.operation_id);

    let conflicting_replay = collective
        .create_collective_transfer(CreateCollectiveTransferRequest {
            spec: Some(group_spec.clone()),
            version_id: "version-2".to_string(),
            idempotency_key: idem.clone(),
        })
        .await
        .expect_err("idempotency replay with a different payload must fail");
    assert_eq!(conflicting_replay.code(), tonic::Code::AlreadyExists);

    let premature_delete = collective
        .delete_collective_transfer(DeleteCollectiveTransferRequest {
            operation_id: transfer.operation_id.clone(),
        })
        .await
        .expect_err("a running-capable operation must retain its fence");
    assert_eq!(premature_delete.code(), tonic::Code::FailedPrecondition);

    let wrong_operation_epoch = collective
        .report_collective_transfer(ReportCollectiveTransferRequest {
            operation_id: transfer.operation_id.clone(),
            group_id: transfer.group_id.clone(),
            epoch: transfer.epoch + 1,
            worker_id: worker_t0.clone(),
            succeeded: true,
            message: String::new(),
        })
        .await
        .expect_err("the operation's stored epoch is authoritative");
    assert_eq!(
        wrong_operation_epoch.code(),
        tonic::Code::FailedPrecondition
    );

    for worker_id in [&worker_t0, &replacement_t1, &worker_g0] {
        collective
            .report_collective_transfer(ReportCollectiveTransferRequest {
                operation_id: transfer.operation_id.clone(),
                group_id: transfer.group_id.clone(),
                epoch: transfer.epoch,
                worker_id: worker_id.clone(),
                succeeded: true,
                message: String::new(),
            })
            .await
            .expect("report successful participant");
    }
    let late_failure = collective
        .report_collective_transfer(ReportCollectiveTransferRequest {
            operation_id: transfer.operation_id.clone(),
            group_id: transfer.group_id.clone(),
            epoch: transfer.epoch,
            worker_id: worker_t0.clone(),
            succeeded: false,
            message: "late failure".to_string(),
        })
        .await
        .expect("terminal replay")
        .into_inner();
    assert_eq!(
        late_failure.state,
        i32::from(CollectiveTransferState::Complete)
    );

    collective
        .delete_collective_transfer(DeleteCollectiveTransferRequest {
            operation_id: transfer.operation_id.clone(),
        })
        .await
        .expect("delete terminal transfer");
    let recreated = create_transfer(&mut collective, &group_spec, "version-1", &idem).await;
    assert_ne!(recreated.operation_id, transfer.operation_id);

    let failed_idem = unique_id("failed-operation");
    let failed = create_transfer(&mut collective, &group_spec, "version-3", &failed_idem).await;
    let failed = collective
        .report_collective_transfer(ReportCollectiveTransferRequest {
            operation_id: failed.operation_id,
            group_id: failed.group_id,
            epoch: failed.epoch,
            worker_id: worker_g0,
            succeeded: false,
            message: "collective timeout".to_string(),
        })
        .await
        .expect("report failed collective")
        .into_inner();
    assert_eq!(failed.state, i32::from(CollectiveTransferState::Failed));
    let after_failure = collective
        .get_collective_group(GetCollectiveGroupRequest {
            group_id: t0.group_id,
        })
        .await
        .expect("read fenced group")
        .into_inner();
    assert_eq!(after_failure.epoch, 3);
    assert_eq!(
        after_failure.state,
        i32::from(CollectiveGroupState::Forming)
    );
    assert!(
        after_failure
            .lanes
            .iter()
            .all(|lane| lane.nccl_unique_id.is_empty())
    );

    stop(stop_server, server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn full_cohort_replacement_converges_on_one_new_epoch() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (stop_server, server) = start_server(port, &redis_url);
    let (mut refit, mut collective) = connect(port).await;

    let model = unique_id("cohort-restart");
    let group_spec = spec(&model, &["t0", "t1"], &["g0"]);
    let old_t0 = unique_id("old-t0");
    let old_t1 = unique_id("old-t1");
    let old_g0 = unique_id("old-g0");
    register(&mut refit, &model, &old_t0, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &old_t1, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &old_g0, WorkerRole::Generator, 60).await;

    let initial = join(
        &mut collective,
        join_request(&group_spec, "t0", &old_t0, CollectiveRole::Trainer, 0),
    )
    .await;
    join(
        &mut collective,
        join_request(&group_spec, "t1", &old_t1, CollectiveRole::Trainer, 1),
    )
    .await;
    join(
        &mut collective,
        join_request(&group_spec, "g0", &old_g0, CollectiveRole::Generator, 0),
    )
    .await;
    publish(&mut collective, &initial.group_id, 1, 0, &old_t0, 10).await;
    let initial_ready = publish(&mut collective, &initial.group_id, 1, 1, &old_t0, 11).await;
    assert_eq!(initial_ready.state, i32::from(CollectiveGroupState::Ready));

    let new_t0 = unique_id("new-t0");
    let new_t1 = unique_id("new-t1");
    let new_g0 = unique_id("new-g0");
    register(&mut refit, &model, &new_t0, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &new_t1, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &new_g0, WorkerRole::Generator, 60).await;

    let replacement_t0 = join(
        &mut collective,
        join_request(&group_spec, "t0", &new_t0, CollectiveRole::Trainer, 0),
    )
    .await;
    let replacement_t1 = join(
        &mut collective,
        join_request(&group_spec, "t1", &new_t1, CollectiveRole::Trainer, 1),
    )
    .await;
    let replacement_g0 = join(
        &mut collective,
        join_request(&group_spec, "g0", &new_g0, CollectiveRole::Generator, 0),
    )
    .await;
    assert_eq!(replacement_t0.epoch, 2);
    assert_eq!(replacement_t1.epoch, 2);
    assert_eq!(replacement_g0.epoch, 2);

    publish(&mut collective, &initial.group_id, 2, 0, &new_t0, 12).await;
    let replacement_ready = publish(&mut collective, &initial.group_id, 2, 1, &new_t0, 13).await;
    assert_eq!(replacement_ready.epoch, 2);
    assert_eq!(
        replacement_ready.state,
        i32::from(CollectiveGroupState::Ready)
    );
    let broadcast = replacement_ready
        .lanes
        .last()
        .expect("broadcast lane contains the full cohort");
    let workers: Vec<&str> = broadcast
        .participants
        .iter()
        .map(|participant| participant.worker_id.as_str())
        .collect();
    assert_eq!(
        workers,
        vec![new_t0.as_str(), new_t1.as_str(), new_g0.as_str()]
    );

    let newer_t0 = unique_id("newer-t0");
    let newer_t1 = unique_id("newer-t1");
    register(&mut refit, &model, &newer_t0, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &newer_t1, WorkerRole::Trainer, 60).await;
    let third_epoch = join(
        &mut collective,
        join_request(&group_spec, "t0", &newer_t0, CollectiveRole::Trainer, 0),
    )
    .await;
    assert_eq!(third_epoch.epoch, 3);
    publish(&mut collective, &initial.group_id, 3, 0, &newer_t0, 14).await;
    let fourth_epoch = join(
        &mut collective,
        join_request(&group_spec, "t1", &newer_t1, CollectiveRole::Trainer, 1),
    )
    .await;
    assert_eq!(fourth_epoch.epoch, 4);
    let invalidated = collective
        .get_collective_group(GetCollectiveGroupRequest {
            group_id: initial.group_id,
        })
        .await
        .expect("read epoch after published lane was invalidated")
        .into_inner();
    assert!(
        invalidated
            .lanes
            .iter()
            .all(|lane| lane.nccl_unique_id.is_empty())
    );

    stop(stop_server, server).await;
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn expired_registration_revokes_ready_membership() {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let port = free_port();
    let (stop_server, server) = start_server(port, &redis_url);
    let (mut refit, mut collective) = connect(port).await;

    let model = unique_id("expiry");
    let group_spec = spec(&model, &["t0"], &["g0"]);
    let trainer = unique_id("live-trainer");
    let generator = unique_id("expiring-generator");
    register(&mut refit, &model, &trainer, WorkerRole::Trainer, 60).await;
    register(&mut refit, &model, &generator, WorkerRole::Generator, 1).await;

    let membership = join(
        &mut collective,
        join_request(&group_spec, "t0", &trainer, CollectiveRole::Trainer, 0),
    )
    .await;
    join(
        &mut collective,
        join_request(&group_spec, "g0", &generator, CollectiveRole::Generator, 0),
    )
    .await;
    publish(&mut collective, &membership.group_id, 1, 0, &trainer, 5).await;
    let ready = publish(&mut collective, &membership.group_id, 1, 1, &trainer, 6).await;
    assert_eq!(ready.state, i32::from(CollectiveGroupState::Ready));

    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let refreshed = collective
        .get_collective_group(GetCollectiveGroupRequest {
            group_id: membership.group_id,
        })
        .await
        .expect("refresh group after registration expiry")
        .into_inner();
    assert_eq!(refreshed.epoch, 2);
    assert_eq!(refreshed.state, i32::from(CollectiveGroupState::Forming));
    assert!(
        refreshed
            .lanes
            .iter()
            .all(|lane| lane.nccl_unique_id.is_empty())
    );
    let participant_slots: Vec<&str> = refreshed
        .lanes
        .last()
        .expect("broadcast lane")
        .participants
        .iter()
        .map(|participant| participant.slot_id.as_str())
        .collect();
    assert_eq!(participant_slots, vec!["t0"]);

    stop(stop_server, server).await;
}
