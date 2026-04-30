// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::arithmetic_side_effects
)]

use criterion::{Criterion, criterion_group, criterion_main};
use modelexpress_common::models::{ModelProvider, ModelStatus, Status};
use std::hint::black_box;

// A RegistryBackend bench belongs in its own testcontainers-based harness so it can
// drive a live Redis or Kubernetes CRD instance — tracked as a follow-up.

fn benchmark_serialization(c: &mut Criterion) {
    c.bench_function("status_serialization", |b| {
        let status = Status {
            version: "1.0.0".to_string(),
            status: "ok".to_string(),
            uptime: 3600,
        };

        b.iter(|| {
            let serialized = serde_json::to_string(black_box(&status)).unwrap();
            let _deserialized: Status = serde_json::from_str(black_box(&serialized)).unwrap();
        });
    });

    c.bench_function("model_status_conversion", |b| {
        let model_status = ModelStatus::DOWNLOADED;

        b.iter(|| {
            let grpc_status: modelexpress_common::grpc::model::ModelStatus =
                black_box(model_status).into();
            let _back_to_model: ModelStatus = grpc_status.into();
        });
    });
}

fn benchmark_model_provider_operations(c: &mut Criterion) {
    c.bench_function("provider_default", |b| {
        b.iter(|| {
            let _provider = black_box(ModelProvider::default());
        });
    });

    c.bench_function("provider_conversion", |b| {
        let provider = ModelProvider::HuggingFace;

        b.iter(|| {
            let grpc_provider: modelexpress_common::grpc::model::ModelProvider =
                black_box(provider).into();
            let _back_to_model: ModelProvider = grpc_provider.into();
        });
    });
}

criterion_group!(
    benches,
    benchmark_serialization,
    benchmark_model_provider_operations
);
criterion_main!(benches);
