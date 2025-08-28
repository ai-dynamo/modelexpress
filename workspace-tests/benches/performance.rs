// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::arithmetic_side_effects
)]

use criterion::{Criterion, criterion_group, criterion_main};
use modelexpress_common::models::{ModelProvider, ModelStatus, Status};
use model_express_server::database::ModelDatabase;
use std::hint::black_box;
use tempfile::TempDir;

fn benchmark_database_operations(c: &mut Criterion) {
    c.bench_function("database_set_status", |b| {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("bench_models.db");
        let db = ModelDatabase::new(db_path.to_str().unwrap()).unwrap();

        let mut counter = 0;
        b.iter(|| {
            let model_name = format!("benchmark-model-{counter}");
            counter += 1;
            db.set_status(
                black_box(&model_name),
                black_box(ModelProvider::HuggingFace),
                black_box(ModelStatus::DOWNLOADED),
                black_box(Some("Benchmark test".to_string())),
            )
            .unwrap();
        });
    });

    c.bench_function("database_get_status", |b| {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("bench_models.db");
        let db = ModelDatabase::new(db_path.to_str().unwrap()).unwrap();

        // Pre-populate with some data
        for i in 0..1000 {
            let model_name = format!("benchmark-model-{i}");
            db.set_status(
                &model_name,
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .unwrap();
        }

        let mut counter = 0;
        b.iter(|| {
            let model_name = format!("benchmark-model-{}", counter % 1000);
            counter += 1;
            db.get_status(black_box(&model_name)).unwrap();
        });
    });

    c.bench_function("database_get_models_by_last_used", |b| {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("bench_models.db");
        let db = ModelDatabase::new(db_path.to_str().unwrap()).unwrap();

        // Pre-populate with data
        for i in 0..100 {
            let model_name = format!("benchmark-model-{i}");
            db.set_status(
                &model_name,
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .unwrap();
        }

        b.iter(|| {
            db.get_models_by_last_used(black_box(Some(10))).unwrap();
        });
    });
}

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
    benchmark_database_operations,
    benchmark_serialization,
    benchmark_model_provider_operations
);
criterion_main!(benches);
