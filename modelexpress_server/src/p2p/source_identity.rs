// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source identity hashing for content-addressed metadata keys.
//!
//! Computes a 16-char hex `mx_source_id` from a `SourceIdentity` proto by
//! normalizing all fields and taking the first 16 chars of SHA256.

use modelexpress_common::grpc::p2p::SourceIdentity;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Compute the `mx_source_id` for a `SourceIdentity`.
///
/// Normalizes the identity (lowercased strings, sorted map keys) then hashes
/// with SHA256. Returns the first 16 hex characters of the digest.
pub fn compute_mx_source_id(identity: &SourceIdentity) -> String {
    let canonical = canonical_json(identity);
    let digest = Sha256::digest(canonical.as_bytes());
    format!("{:x}", digest)[..16].to_string()
}

/// Validate that a `SourceIdentity` has required fields set.
pub fn validate_identity(identity: &SourceIdentity) -> Result<(), String> {
    if identity.model_name.is_empty() {
        return Err("identity.model_name is required".to_string());
    }
    Ok(())
}

fn canonical_json(identity: &SourceIdentity) -> String {
    // Normalize extra_parameters deterministically:
    // 1. sort by original key (String::cmp = byte order, matches Python's
    //    default sorted() on str keys for ASCII content)
    // 2. lowercase both keys and values
    // 3. on case-colliding keys, keep the first value we see
    // HashMap iteration order is non-deterministic in Rust, so relying on
    // .collect() into BTreeMap would let a case-colliding key's surviving
    // value flip run-to-run. Explicit sort-then-dedup keeps cross-language
    // hashes stable with the Python implementation.
    let mut items: Vec<(&String, &String)> = identity.extra_parameters.iter().collect();
    items.sort_by(|a, b| a.0.cmp(b.0));
    let mut sorted_extra: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in items {
        let lk = k.to_lowercase();
        sorted_extra.entry(lk).or_insert_with(|| v.to_lowercase());
    }

    serde_json::json!({
        "mx_version": identity.mx_version.to_lowercase(),
        "mx_source_type": identity.mx_source_type,
        "model_name": identity.model_name.to_lowercase(),
        "backend_framework": identity.backend_framework,
        "tensor_parallel_size": identity.tensor_parallel_size,
        "pipeline_parallel_size": identity.pipeline_parallel_size,
        "expert_parallel_size": identity.expert_parallel_size,
        "dtype": identity.dtype.to_lowercase(),
        "quantization": identity.quantization.to_lowercase(),
        "extra_parameters": sorted_extra,
        "revision": identity.revision.to_lowercase(),
    })
    .to_string()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn base_identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: 0, // Weights (default)
            model_name: "deepseek-ai/DeepSeek-V3".to_string(),
            backend_framework: 1, // vllm
            tensor_parallel_size: 8,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
            revision: String::new(),
        }
    }

    #[test]
    fn test_id_is_16_chars() {
        let id = compute_mx_source_id(&base_identity());
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_deterministic() {
        let id1 = compute_mx_source_id(&base_identity());
        let id2 = compute_mx_source_id(&base_identity());
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_case_insensitive() {
        let mut upper = base_identity();
        upper.model_name = "DEEPSEEK-AI/DEEPSEEK-V3".to_string();
        upper.dtype = "BFLOAT16".to_string();
        assert_eq!(
            compute_mx_source_id(&base_identity()),
            compute_mx_source_id(&upper)
        );
    }

    #[test]
    fn test_different_tp_gives_different_id() {
        let mut tp4 = base_identity();
        tp4.tensor_parallel_size = 4;
        assert_ne!(
            compute_mx_source_id(&base_identity()),
            compute_mx_source_id(&tp4)
        );
    }

    #[test]
    fn test_different_dtype_gives_different_id() {
        let mut fp8 = base_identity();
        fp8.dtype = "float8_e4m3fn".to_string();
        assert_ne!(
            compute_mx_source_id(&base_identity()),
            compute_mx_source_id(&fp8)
        );
    }

    #[test]
    fn test_different_source_type_gives_different_id() {
        let mut lora = base_identity();
        lora.mx_source_type = 1; // LoRA
        assert_ne!(
            compute_mx_source_id(&base_identity()),
            compute_mx_source_id(&lora)
        );
    }

    #[test]
    fn test_extra_parameters_sorted() {
        let mut a = base_identity();
        a.extra_parameters
            .insert("z_key".to_string(), "val".to_string());
        a.extra_parameters
            .insert("a_key".to_string(), "val".to_string());

        let mut b = base_identity();
        b.extra_parameters
            .insert("a_key".to_string(), "val".to_string());
        b.extra_parameters
            .insert("z_key".to_string(), "val".to_string());

        assert_eq!(compute_mx_source_id(&a), compute_mx_source_id(&b));
    }

    #[test]
    fn test_different_revision_gives_different_id() {
        let mut pinned = base_identity();
        pinned.revision = "abc123def4567890".to_string();
        assert_ne!(
            compute_mx_source_id(&base_identity()),
            compute_mx_source_id(&pinned)
        );
    }

    #[test]
    fn test_revision_case_insensitive() {
        let mut upper = base_identity();
        upper.revision = "ABC123DEF4567890".to_string();
        let mut lower = base_identity();
        lower.revision = "abc123def4567890".to_string();
        assert_eq!(compute_mx_source_id(&upper), compute_mx_source_id(&lower));
    }

    // Cross-checked against modelexpress_client/python/tests/test_source_id.py.
    // If either side's canonical JSON encoding or hashing scheme changes,
    // both of these asserts diverge from their Python counterparts and
    // the mismatch is caught in CI.
    #[test]
    fn test_python_cross_check_base_identity() {
        assert_eq!(compute_mx_source_id(&base_identity()), "b0c2c67edeaefc20");
    }

    #[test]
    fn test_python_cross_check_with_revision() {
        let mut pinned = base_identity();
        pinned.revision = "abc123def4567890".to_string();
        assert_eq!(compute_mx_source_id(&pinned), "40704b34e4b7deaa");
    }

    #[test]
    fn test_python_cross_check_case_colliding_extra() {
        // Case-colliding keys (Foo vs foo) with different values. The
        // deterministic normalization rule: sort original keys (String::cmp
        // byte order puts "Foo" before "foo"), lowercase, keep the first
        // value. "Foo"="a" survives over "foo"="b" regardless of insertion
        // order into the proto's HashMap. Matches Python
        // test_source_id.py::test_case_colliding_extra_parameters_are_deterministic.
        let mut id = base_identity();
        id.extra_parameters
            .insert("Foo".to_string(), "a".to_string());
        id.extra_parameters
            .insert("foo".to_string(), "b".to_string());
        assert_eq!(compute_mx_source_id(&id), "bd9ea6c70d83fef1");
    }

    #[test]
    fn test_validate_requires_model_name() {
        let mut id = base_identity();
        id.model_name = String::new();
        assert!(validate_identity(&id).is_err());
    }

    #[test]
    fn test_validate_passes() {
        assert!(validate_identity(&base_identity()).is_ok());
    }
}
