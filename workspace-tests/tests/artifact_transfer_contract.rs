// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

use modelexpress_common::{
    artifact_manifest::ArtifactManifest,
    grpc::p2p::worker_metadata::SourcePayload,
    grpc::p2p::{
        ArtifactSourceMetadata, BackendFramework, MxSourceType, SourceIdentity, TensorDescriptor,
        WorkerMetadata,
    },
};
use modelexpress_server::p2p::backend::WorkerRecord;
use modelexpress_server::p2p::k8s_types::{ArtifactSourceStatus, ModelMetadataSpec, WorkerStatus};
use std::fs;

fn base_identity() -> SourceIdentity {
    SourceIdentity {
        mx_version: "0.5.0".to_string(),
        mx_source_type: MxSourceType::Weights as i32,
        model_name: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        backend_framework: BackendFramework::Vllm as i32,
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        expert_parallel_size: 0,
        dtype: "bfloat16".to_string(),
        quantization: String::new(),
        extra_parameters: Default::default(),
        revision: "abc123".to_string(),
        backend_framework_version: String::new(),
        torch_version: String::new(),
        cuda_version: String::new(),
        triton_version: String::new(),
        gpu_arch: String::new(),
        compile_config_digest: String::new(),
    }
}

#[test]
fn artifact_source_identity_is_separate_from_weights() {
    let weights = base_identity();
    let mut compile_cache = weights.clone();
    compile_cache.mx_source_type = MxSourceType::TorchCompileCache as i32;
    compile_cache.backend_framework_version = "vllm-0.10.0".to_string();
    compile_cache.torch_version = "2.8.0+cu128".to_string();
    compile_cache.cuda_version = "12.8".to_string();
    compile_cache.triton_version = "3.4.0".to_string();
    compile_cache.gpu_arch = "SM90".to_string();
    compile_cache.compile_config_digest = "compile-config-a".to_string();

    let mut other_compile_cache = compile_cache.clone();
    other_compile_cache.compile_config_digest = "compile-config-b".to_string();

    assert_ne!(
        modelexpress_server::p2p::source_identity::compute_mx_source_id(&weights),
        modelexpress_server::p2p::source_identity::compute_mx_source_id(&compile_cache),
        "artifact source types must not collide with model weight sources"
    );
    assert_ne!(
        modelexpress_server::p2p::source_identity::compute_mx_source_id(&compile_cache),
        modelexpress_server::p2p::source_identity::compute_mx_source_id(&other_compile_cache),
        "artifact compatibility fields must affect artifact source identity"
    );
}

#[test]
#[allow(deprecated)]
fn artifact_payload_does_not_publish_weight_tensors() {
    let worker = WorkerMetadata {
        worker_rank: 0,
        tensors: vec![TensorDescriptor {
            name: "legacy.weight".to_string(),
            addr: 0x1000,
            size: 4096,
            device_id: 0,
            dtype: "bfloat16".to_string(),
        }],
        source_payload: Some(SourcePayload::ArtifactSource(ArtifactSourceMetadata {
            artifact_id: "artifact-manifest".to_string(),
            total_size: 1_099_511_627_776,
            file_count: 42,
            chunk_count: 4096,
        })),
        ..Default::default()
    };

    let record = WorkerRecord::from(worker);

    assert!(
        record.tensors.is_empty(),
        "artifact sources must not be interpreted as tensor sources"
    );
    let artifact = record
        .artifact_source
        .as_ref()
        .expect("artifact payload should be preserved");
    assert_eq!(artifact.artifact_id, "artifact-manifest");
    assert_eq!(artifact.total_size, 1_099_511_627_776);
    assert_eq!(artifact.file_count, 42);
    assert_eq!(artifact.chunk_count, 4096);

    let back: WorkerMetadata = record.into();
    assert!(back.tensors.is_empty());
    assert!(matches!(
        back.source_payload,
        Some(SourcePayload::ArtifactSource(ref artifact))
            if artifact.artifact_id == "artifact-manifest"
                && artifact.total_size == 1_099_511_627_776
                && artifact.file_count == 42
                && artifact.chunk_count == 4096
    ));
}

#[test]
fn sealed_artifact_manifest_derives_discovery_summary() {
    let temp_dir = tempfile::TempDir::new().expect("create temp dir");
    fs::create_dir(temp_dir.path().join("torchinductor")).expect("create artifact dir");
    fs::write(
        temp_dir.path().join("torchinductor/fxgraph"),
        b"compiled-graph",
    )
    .expect("write compiled graph");
    fs::write(temp_dir.path().join("triton.cubin"), b"cubin").expect("write cubin");

    let sealed = ArtifactManifest::from_directory(
        temp_dir.path(),
        8,
        MxSourceType::TorchCompileCache as i32,
    )
    .expect("build artifact manifest")
    .seal()
    .expect("seal artifact manifest");
    let metadata = sealed.source_metadata().expect("derive source metadata");

    assert_eq!(metadata.artifact_id, sealed.artifact_id);
    assert_eq!(metadata.total_size, 19);
    assert_eq!(metadata.file_count, 2);
    assert_eq!(metadata.chunk_count, 3);
}

#[test]
#[allow(deprecated)]
fn tensor_payload_also_populates_legacy_tensors_for_old_readers() {
    let worker = WorkerMetadata {
        worker_rank: 0,
        source_payload: Some(SourcePayload::TensorSource(
            modelexpress_common::grpc::p2p::TensorSourceMetadata {
                tensors: vec![TensorDescriptor {
                    name: "model.layers.0.weight".to_string(),
                    addr: 0x1000,
                    size: 4096,
                    device_id: 0,
                    dtype: "bfloat16".to_string(),
                }],
            },
        )),
        ..Default::default()
    };

    let record = WorkerRecord::from(worker);
    let back: WorkerMetadata = record.into();

    assert_eq!(back.tensors.len(), 1);
    assert_eq!(back.tensors[0].name, "model.layers.0.weight");
    assert!(matches!(
        back.source_payload,
        Some(SourcePayload::TensorSource(ref tensor_source))
            if tensor_source.tensors.len() == 1
                && tensor_source.tensors[0].name == "model.layers.0.weight"
    ));
}

#[test]
fn k8s_metadata_contract_carries_artifact_source_type_and_summary() {
    assert_eq!(
        ModelMetadataSpec::source_type_name_from_proto(MxSourceType::TorchCompileCache as i32),
        "torch_compile_cache"
    );
    assert_eq!(
        ModelMetadataSpec::source_type_name_from_proto(MxSourceType::DeepGemmCache as i32),
        "deep_gemm_cache"
    );

    let worker = WorkerStatus {
        worker_rank: 0,
        tensor_count: 0,
        tensor_config_map: None,
        artifact_source: Some(ArtifactSourceStatus {
            artifact_id: "artifact-manifest".to_string(),
            total_size: 1_099_511_627_776,
            file_count: 42,
            chunk_count: 4096,
        }),
        ..Default::default()
    };
    let json = serde_json::to_value(&worker).expect("serialize worker status");

    assert_eq!(json["tensorCount"], 0);
    assert!(json["tensorConfigMap"].is_null());
    assert_eq!(json["artifactSource"]["artifactId"], "artifact-manifest");
    assert_eq!(json["artifactSource"]["totalSize"], 1_099_511_627_776_u64);
    assert_eq!(json["artifactSource"]["fileCount"], 42);
    assert_eq!(json["artifactSource"]["chunkCount"], 4096);
}
