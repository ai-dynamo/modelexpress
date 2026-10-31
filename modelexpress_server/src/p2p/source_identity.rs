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

/// Maximum accepted length of a `worker_id`, in bytes.
///
/// A `worker_id` reaches Kubernetes on two paths with different limits: it is
/// interpolated into the `mx-source-{source_id}-{worker_id}` object name, which
/// is a DNS-1123 subdomain capped at 253, and it is written verbatim as the
/// `modelexpress.nvidia.com/mx-worker-id` label value, which is capped at 63.
/// The label value is the binding constraint.
const WORKER_ID_MAX_LEN: usize = 63;

/// Validate a client-supplied `worker_id` before it reaches a backend.
///
/// The value is interpolated into a Kubernetes object name and written as a
/// label value, so it is constrained to the intersection of what both accept:
/// lowercase alphanumerics, `-` and `.`, with alphanumeric boundaries. Object
/// names are DNS-1123 and reject uppercase, which is why the lowercase rule is
/// not merely stylistic. The documented producer is a UUID, which satisfies it.
///
/// This rejects rather than sanitizes, unlike the model-name path. Rewriting
/// the value would change which object name a previously-published worker maps
/// to, and the CR name is reconstructed from `worker_id` on later operations,
/// so a lossy transform here would strand records rather than protect them.
///
/// The length check is on bytes rather than characters on purpose: Kubernetes
/// counts bytes, and non-ASCII input is rejected by the charset check anyway.
pub fn validate_worker_id(worker_id: &str) -> Result<(), String> {
    if worker_id.is_empty() {
        return Err("worker_id is required".to_string());
    }
    if worker_id.len() > WORKER_ID_MAX_LEN {
        return Err(format!(
            "worker_id must be at most {} bytes, got {}",
            WORKER_ID_MAX_LEN,
            worker_id.len()
        ));
    }
    if let Some(bad) = worker_id
        .chars()
        .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-' || *c == '.'))
    {
        return Err(format!(
            "worker_id may only contain lowercase alphanumerics, '-' and '.', found {bad:?}"
        ));
    }
    let bytes = worker_id.as_bytes();
    let boundary_ok =
        |b: Option<&u8>| matches!(b, Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit());
    if !boundary_ok(bytes.first()) || !boundary_ok(bytes.last()) {
        return Err(
            "worker_id must start and end with a lowercase alphanumeric character".to_string(),
        );
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

    let mut payload = serde_json::json!({
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
    });

    if let Some(object) = payload.as_object_mut() {
        insert_non_empty_lowercase(
            object,
            "backend_framework_version",
            &identity.backend_framework_version,
        );
        insert_non_empty_lowercase(object, "torch_version", &identity.torch_version);
        insert_non_empty_lowercase(object, "cuda_version", &identity.cuda_version);
        insert_non_empty_lowercase(object, "triton_version", &identity.triton_version);
        insert_non_empty_lowercase(object, "gpu_arch", &identity.gpu_arch);
        insert_non_empty_lowercase(
            object,
            "compile_config_digest",
            &identity.compile_config_digest,
        );
    }

    payload.to_string()
}

fn insert_non_empty_lowercase(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: &str,
) {
    if !value.is_empty() {
        object.insert(
            key.to_string(),
            serde_json::Value::String(value.to_lowercase()),
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn base_identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.8.0".to_string(),
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
            backend_framework_version: String::new(),
            torch_version: String::new(),
            cuda_version: String::new(),
            triton_version: String::new(),
            gpu_arch: String::new(),
            compile_config_digest: String::new(),
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
    fn test_empty_artifact_fields_preserve_existing_id() {
        assert_eq!(compute_mx_source_id(&base_identity()), "e0d1bc7459cb4cde");
    }

    #[test]
    fn test_artifact_compatibility_fields_affect_id() {
        let mut artifact = base_identity();
        artifact.mx_source_type =
            modelexpress_common::grpc::p2p::MxSourceType::TorchCompileCache as i32;
        artifact.backend_framework_version = "0.10.0".to_string();
        artifact.torch_version = "2.8.0+cu128".to_string();
        artifact.cuda_version = "12.8".to_string();
        artifact.triton_version = "3.4.0".to_string();
        artifact.gpu_arch = "SM90".to_string();
        artifact.compile_config_digest = "abc123".to_string();

        let mut different_torch = artifact.clone();
        different_torch.torch_version = "2.9.0+cu128".to_string();

        assert_ne!(
            compute_mx_source_id(&artifact),
            compute_mx_source_id(&different_torch)
        );
    }

    #[test]
    fn test_artifact_compatibility_fields_are_case_insensitive() {
        let mut upper = base_identity();
        upper.mx_source_type =
            modelexpress_common::grpc::p2p::MxSourceType::TorchCompileCache as i32;
        upper.backend_framework_version = "VLLM-0.10.0".to_string();
        upper.gpu_arch = "SM90".to_string();

        let mut lower = base_identity();
        lower.mx_source_type =
            modelexpress_common::grpc::p2p::MxSourceType::TorchCompileCache as i32;
        lower.backend_framework_version = "vllm-0.10.0".to_string();
        lower.gpu_arch = "sm90".to_string();

        assert_eq!(compute_mx_source_id(&upper), compute_mx_source_id(&lower));
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
        assert_eq!(compute_mx_source_id(&base_identity()), "e0d1bc7459cb4cde");
    }

    #[test]
    fn test_python_cross_check_with_revision() {
        let mut pinned = base_identity();
        pinned.revision = "abc123def4567890".to_string();
        assert_eq!(compute_mx_source_id(&pinned), "96db7e109daf9e53");
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
        assert_eq!(compute_mx_source_id(&id), "f4eeab03e859b088");
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

    #[test]
    fn worker_id_accepts_a_uuid() {
        // The shape the proto documents as the producer.
        assert!(validate_worker_id("3f2504e0-4f89-41d3-9a0c-0305e82c3301").is_ok());
    }

    #[test]
    fn worker_id_accepts_dotted_and_numeric_forms() {
        assert!(validate_worker_id("worker-0").is_ok());
        assert!(validate_worker_id("rank0.replica1").is_ok());
        assert!(validate_worker_id("0").is_ok());
    }

    #[test]
    fn worker_id_rejects_empty() {
        assert!(validate_worker_id("").is_err());
    }

    #[test]
    fn worker_id_rejects_over_label_value_budget() {
        // 63 bytes is the Kubernetes label-value cap and is accepted; 64 is not.
        let at_cap = "a".repeat(WORKER_ID_MAX_LEN);
        assert!(validate_worker_id(&at_cap).is_ok());
        let over_cap = "a".repeat(WORKER_ID_MAX_LEN + 1);
        assert!(validate_worker_id(&over_cap).is_err());
    }

    #[test]
    fn worker_id_rejects_k8s_name_injection() {
        // Each of these is currently interpolated raw into an object name and a
        // label value. A '/' escapes the name component entirely.
        for bad in [
            "a/../b", "a/b", "a b", "a\nb", "a:b", "a=b", "a,b", "a%2Fb", "a\"b", "a$b",
        ] {
            assert!(
                validate_worker_id(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn worker_id_rejects_uppercase() {
        // Valid as a label value, invalid as a DNS-1123 object name, so the CR
        // create would fail at the API server rather than here.
        assert!(validate_worker_id("Worker-0").is_err());
        assert!(validate_worker_id("ABC").is_err());
    }

    #[test]
    fn worker_id_rejects_non_alphanumeric_boundaries() {
        assert!(validate_worker_id("-abc").is_err());
        assert!(validate_worker_id("abc-").is_err());
        assert!(validate_worker_id(".abc").is_err());
        assert!(validate_worker_id("abc.").is_err());
        assert!(validate_worker_id("-").is_err());
    }

    #[test]
    fn worker_id_rejects_non_ascii() {
        assert!(validate_worker_id("wörker").is_err());
        assert!(validate_worker_id("worker\u{200b}0").is_err());
    }
}
