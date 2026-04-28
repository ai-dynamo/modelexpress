// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the Redis registry backend.
//!
//! All tests are `#[ignore]` by default and require a live Redis reachable via
//! `REDIS_URL` (defaulting to `redis://localhost:6379`). Run with:
//!
//! ```sh
//! docker run --rm -d -p 6379:6379 redis:7-alpine
//! REDIS_URL=redis://localhost:6379 cargo test -p model-express-workspace-tests \
//!     --test registry_backend_redis -- --include-ignored
//! ```
//!
//! Each test uses a unique key prefix so runs are isolated without needing FLUSHDB.

#![allow(clippy::expect_used, clippy::arithmetic_side_effects)]

use modelexpress_common::models::{ModelProvider, ModelStatus};
use modelexpress_server::registry::backend::{
    ClaimOutcome, RegistryBackend, redis::RedisRegistryBackend,
};
use std::sync::Arc;

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string())
}

/// Unique model-name prefix per test so concurrent test runs don't collide.
fn unique_name(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_nanos();
    format!("mx-test/{tag}-{nanos}")
}

async fn fresh_backend() -> RedisRegistryBackend {
    let backend = RedisRegistryBackend::new(&redis_url());
    backend
        .connect()
        .await
        .expect("connect to Redis at REDIS_URL (is docker running?)");
    backend
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn claim_then_set_then_delete_roundtrip() {
    let backend = fresh_backend().await;
    let name = unique_name("roundtrip");

    // Missing model: no status.
    assert_eq!(backend.get_status(&name).await.expect("get_status"), None);

    // First claim wins and marks DOWNLOADING.
    let claimed = backend
        .try_claim_for_download(&name, ModelProvider::HuggingFace)
        .await
        .expect("claim");
    assert_eq!(claimed, ClaimOutcome::Claimed);

    // Second claim observes the existing record without mutation.
    let re_claim = backend
        .try_claim_for_download(&name, ModelProvider::HuggingFace)
        .await
        .expect("re-claim");
    assert_eq!(
        re_claim,
        ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)
    );

    // Full record is populated.
    let rec = backend
        .get_model_record(&name)
        .await
        .expect("get_record")
        .expect("record present");
    assert_eq!(rec.model_name, name);
    assert_eq!(rec.provider, ModelProvider::HuggingFace);
    assert_eq!(rec.status, ModelStatus::DOWNLOADING);
    assert!(rec.message.is_some());
    let original_created = rec.created_at;

    // set_status flips to DOWNLOADED and preserves created_at.
    backend
        .set_status(
            &name,
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
            Some("done".into()),
        )
        .await
        .expect("set_status");
    let after = backend
        .get_model_record(&name)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(after.status, ModelStatus::DOWNLOADED);
    assert_eq!(after.created_at, original_created);
    assert_eq!(after.message.as_deref(), Some("done"));

    // delete_model removes the key.
    backend.delete_model(&name).await.expect("delete");
    assert_eq!(backend.get_status(&name).await.expect("get"), None);
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn concurrent_claims_yield_single_winner() {
    let backend = Arc::new(fresh_backend().await);
    let name = unique_name("concurrent");

    let mut handles = Vec::new();
    for _ in 0..8 {
        let b = backend.clone();
        let n = name.clone();
        handles.push(tokio::spawn(async move {
            b.try_claim_for_download(&n, ModelProvider::HuggingFace)
                .await
        }));
    }
    // Exactly one `Claimed`; the rest see `AlreadyExists(DOWNLOADING)`. This is the
    // assertion that guarantees multi-replica servers never double-download.
    let mut winners = 0;
    let mut observers = 0;
    for h in handles {
        let outcome = h.await.expect("spawn join").expect("claim result");
        match outcome {
            ClaimOutcome::Claimed => winners += 1,
            ClaimOutcome::AlreadyExists(s) => {
                assert_eq!(s, ModelStatus::DOWNLOADING);
                observers += 1;
            }
        }
    }
    assert_eq!(winners, 1, "exactly one replica must claim");
    assert_eq!(observers, 7, "the other seven must observe");

    // Only one record exists.
    let rec = backend
        .get_model_record(&name)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(rec.status, ModelStatus::DOWNLOADING);

    backend.delete_model(&name).await.expect("cleanup");
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn touch_updates_last_used_at() {
    let backend = fresh_backend().await;
    let name = unique_name("touch");

    backend
        .try_claim_for_download(&name, ModelProvider::HuggingFace)
        .await
        .expect("initial claim");
    let first = backend
        .get_model_record(&name)
        .await
        .expect("get_model_record")
        .expect("record present after claim")
        .last_used_at;

    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    backend.touch_model(&name).await.expect("touch_model");

    let second = backend
        .get_model_record(&name)
        .await
        .expect("get_model_record after touch")
        .expect("record present after touch")
        .last_used_at;
    assert!(
        second > first,
        "touch_model should bump last_used_at: {first} -> {second}"
    );

    backend.delete_model(&name).await.expect("delete_model");
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn touch_missing_model_is_noop() {
    let backend = fresh_backend().await;
    let name = unique_name("touch-missing");

    // Should succeed without creating a malformed record.
    backend.touch_model(&name).await.expect("touch missing");
    assert_eq!(
        backend.get_status(&name).await.expect("get_status"),
        None,
        "touching a missing model must not create a record"
    );
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn get_models_by_last_used_returns_sorted_slice() {
    // Dedicated DB so the limit assertion doesn't race with other ignored tests.
    let backend = RedisRegistryBackend::new(&isolated_db_url(14));
    backend.connect().await.expect("connect to isolated DB");
    flushdb(14);

    let names = ["a", "b", "c"];
    for n in &names {
        backend
            .set_status(n, ModelProvider::HuggingFace, ModelStatus::DOWNLOADED, None)
            .await
            .expect("set_status seeds a record");
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    }

    let all = backend
        .get_models_by_last_used(None)
        .await
        .expect("get_models_by_last_used (all)");
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].model_name, "a"); // oldest
    assert_eq!(all[2].model_name, "c"); // newest

    let limited = backend
        .get_models_by_last_used(Some(2))
        .await
        .expect("get_models_by_last_used (limited)");
    assert_eq!(limited.len(), 2);
    assert_eq!(limited[0].model_name, "a");
    assert_eq!(limited[1].model_name, "b");

    flushdb(14);
}

/// Build a REDIS_URL that targets a dedicated logical DB index, so tests that assert
/// absolute counts don't race with other ignored tests writing to DB 0.
fn isolated_db_url(db: u32) -> String {
    let base = redis_url();
    let trimmed = base.trim_end_matches('/');
    match trimmed.rsplit_once('/') {
        Some((prefix, tail)) if tail.parse::<u32>().is_ok() => format!("{prefix}/{db}"),
        _ => format!("{trimmed}/{db}"),
    }
}

/// FLUSHDB the isolated DB via a blocking redis client (the registry backend doesn't
/// expose a flush primitive by design).
fn flushdb(db: u32) {
    let mut conn = redis::Client::open(isolated_db_url(db).as_str())
        .expect("open Redis client for FLUSHDB")
        .get_connection()
        .expect("open sync connection for FLUSHDB");
    redis::cmd("FLUSHDB")
        .query::<()>(&mut conn)
        .expect("FLUSHDB on isolated DB");
}

#[tokio::test]
#[ignore = "requires a live Redis at REDIS_URL"]
async fn get_status_counts_reflects_stored_records() {
    let backend = RedisRegistryBackend::new(&isolated_db_url(15));
    backend.connect().await.expect("connect to isolated DB");
    flushdb(15);

    let (d0, ok0, e0) = backend.get_status_counts().await.expect("initial counts");
    assert_eq!((d0, ok0, e0), (0, 0, 0));

    backend
        .set_status(
            "m-downloaded",
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
            None,
        )
        .await
        .expect("set DOWNLOADED");
    backend
        .set_status(
            "m-error",
            ModelProvider::HuggingFace,
            ModelStatus::ERROR,
            None,
        )
        .await
        .expect("set ERROR");
    backend
        .try_claim_for_download("m-downloading", ModelProvider::HuggingFace)
        .await
        .expect("claim DOWNLOADING");

    let (d1, ok1, e1) = backend
        .get_status_counts()
        .await
        .expect("counts after seeding");
    assert_eq!((d1, ok1, e1), (1, 1, 1));

    flushdb(15);
}
