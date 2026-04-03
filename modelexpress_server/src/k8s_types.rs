// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD types for ModelMetadata.
//!
//! These types define the ModelMetadata CustomResourceDefinition used as an
//! alternative to Redis for storing P2P metadata.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// ModelMetadata spec - the desired state
#[derive(CustomResource, Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[kube(
    group = "modelexpress.nvidia.com",
    version = "v1alpha1",
    kind = "ModelMetadata",
    plural = "modelmetadatas",
    shortname = "mxmeta",
    namespaced,
    status = "ModelMetadataStatus"
)]
pub struct ModelMetadataSpec {
    /// Full model name (e.g., deepseek-ai/DeepSeek-V3)
    #[serde(rename = "modelName")]
    pub model_name: String,
}

/// ModelMetadata status - the observed state
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
pub struct ModelMetadataStatus {
    /// Single worker NIXL metadata and readiness state (one CR per worker)
    #[serde(default)]
    pub worker: Option<WorkerStatus>,

    /// Conditions for ModelMetadata lifecycle
    #[serde(default)]
    pub conditions: Vec<Condition>,

    /// Generation observed by the controller
    #[serde(rename = "observedGeneration", default)]
    pub observed_generation: i64,

    /// Timestamp when first worker published
    #[serde(rename = "publishedAt", default)]
    pub published_at: Option<String>,
}

/// Per-worker status
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
pub struct WorkerStatus {
    /// Worker rank (0-indexed)
    #[serde(rename = "workerRank")]
    pub worker_rank: i32,

    /// Backend type discriminator ("nixl", "transfer_engine", "none")
    #[serde(rename = "backendType", default)]
    pub backend_type: Option<String>,

    /// Base64-encoded NIXL agent metadata blob
    #[serde(rename = "nixlMetadata", default)]
    pub nixl_metadata: String,

    /// Mooncake TransferEngine session ID
    #[serde(rename = "transferEngineSessionId", default)]
    pub transfer_engine_session_id: Option<String>,

    /// Number of tensors registered by this worker
    #[serde(rename = "tensorCount", default)]
    pub tensor_count: i32,

    /// Name of ConfigMap containing tensor descriptors
    #[serde(rename = "tensorConfigMap", default)]
    pub tensor_config_map: Option<String>,

    /// Worker lifecycle status (Initializing, Ready, Stale)
    #[serde(default)]
    pub status: String,

    /// Timestamp of last status update (RFC3339)
    #[serde(rename = "updatedAt", default)]
    pub updated_at: Option<String>,

    /// P2P: NIXL listen thread endpoint (host:port)
    #[serde(rename = "metadataEndpoint", default)]
    pub metadata_endpoint: String,

    /// P2P: NIXL agent name
    #[serde(rename = "agentName", default)]
    pub agent_name: String,

    /// P2P: Worker gRPC endpoint for tensor manifest (host:port)
    #[serde(rename = "workerGrpcEndpoint", default)]
    pub worker_grpc_endpoint: String,

    /// CUDA allocation end addresses for pool-based transfer coalescing
    #[serde(rename = "allocEnds", default)]
    pub alloc_ends: Vec<u64>,
}

impl WorkerStatus {
    /// Convert a `SourceStatus` proto enum value (i32) to the CRD status string.
    pub fn status_name_from_proto(status: i32) -> String {
        match status {
            0 => "Unknown",
            1 => "Initializing",
            2 => "Ready",
            3 => "Stale",
            _ => "Unknown",
        }
        .to_string()
    }

    /// Convert a CRD status string back to the `SourceStatus` proto enum value (i32).
    pub fn status_proto_from_name(name: &str) -> i32 {
        match name {
            "Initializing" => 1,
            "Ready" => 2,
            "Stale" => 3,
            _ => 0,
        }
    }
}

/// Standard Kubernetes condition
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
pub struct Condition {
    /// Condition type
    #[serde(rename = "type")]
    pub type_: String,

    /// Status: True, False, Unknown
    pub status: String,

    /// Machine-readable reason for condition
    #[serde(default)]
    pub reason: Option<String>,

    /// Human-readable message
    #[serde(default)]
    pub message: Option<String>,

    /// Timestamp of last transition
    #[serde(rename = "lastTransitionTime", default)]
    pub last_transition_time: Option<String>,
}

/// Tensor descriptor stored in ConfigMap
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorDescriptorJson {
    pub name: String,
    /// Serialized as string to avoid precision loss
    pub addr: String,
    /// Serialized as string to avoid precision loss
    pub size: String,
    pub device_id: u32,
    pub dtype: String,
}

/// Sanitize model name to be a valid Kubernetes resource name
/// e.g., "deepseek-ai/DeepSeek-V3" -> "deepseek-ai-deepseek-v3"
pub fn sanitize_model_name(model_name: &str) -> String {
    model_name
        .to_lowercase()
        .replace(['/', '_'], "-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '.')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_status_roundtrip() {
        for (proto, name) in [
            (0, "Unknown"),
            (1, "Initializing"),
            (2, "Ready"),
            (3, "Stale"),
        ] {
            assert_eq!(WorkerStatus::status_name_from_proto(proto), name);
            assert_eq!(WorkerStatus::status_proto_from_name(name), proto);
        }
    }

    /// Regression test: proto status 0 (SOURCE_STATUS_UNKNOWN) must survive a
    /// write-to-CRD -> read-from-CRD roundtrip. Before the fix, status_proto_from_name
    /// returned None for "Unknown", causing get_metadata to hard-error on any worker
    /// that hadn't received an explicit UpdateStatus call after PublishMetadata.
    #[test]
    fn test_status_unknown_roundtrip() {
        let written = WorkerStatus::status_name_from_proto(0);
        assert_eq!(written, "Unknown");
        let read_back = WorkerStatus::status_proto_from_name(&written);
        assert_eq!(
            read_back, 0,
            "Unknown status must roundtrip to proto value 0"
        );
    }

    #[test]
    fn test_status_name_from_proto_unknown() {
        assert_eq!(WorkerStatus::status_name_from_proto(99), "Unknown");
        assert_eq!(WorkerStatus::status_name_from_proto(4), "Unknown");
    }

    #[test]
    fn test_status_proto_from_name_unknown() {
        assert_eq!(WorkerStatus::status_proto_from_name("Unknown"), 0);
        assert_eq!(WorkerStatus::status_proto_from_name(""), 0);
        assert_eq!(WorkerStatus::status_proto_from_name("ready"), 0);
    }

    #[test]
    fn test_sanitize_model_name() {
        assert_eq!(
            sanitize_model_name("deepseek-ai/DeepSeek-V3"),
            "deepseek-ai-deepseek-v3"
        );
        assert_eq!(
            sanitize_model_name("meta-llama/Llama-3.1-70B"),
            "meta-llama-llama-3.1-70b"
        );
        assert_eq!(sanitize_model_name("simple-model"), "simple-model");
    }

    #[test]
    fn test_sanitize_model_name_special_chars() {
        assert_eq!(sanitize_model_name("Llama@3.1+8B"), "llama3.18b");
        assert_eq!(sanitize_model_name("model with spaces"), "modelwithspaces");
        assert_eq!(
            sanitize_model_name("org_name/model_v2"),
            "org-name-model-v2"
        );
    }

    #[test]
    fn test_sanitize_model_name_edge_cases() {
        assert_eq!(sanitize_model_name(""), "");
        assert_eq!(sanitize_model_name("///"), "");
        assert_eq!(sanitize_model_name("---"), "");
        assert_eq!(sanitize_model_name("-model-"), "model");
    }

    #[test]
    fn test_tensor_descriptor_json_roundtrip() {
        let original = TensorDescriptorJson {
            name: "model.layers.0.weight".to_string(),
            addr: "139948187451390".to_string(),
            size: "134217728".to_string(),
            device_id: 0,
            dtype: "bfloat16".to_string(),
        };

        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: TensorDescriptorJson = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.addr, original.addr);
        assert_eq!(parsed.size, original.size);
        assert_eq!(parsed.device_id, original.device_id);
        assert_eq!(parsed.dtype, original.dtype);

        let addr: u64 = parsed.addr.parse().expect("addr should parse as u64");
        assert_eq!(addr, 139948187451390);
        let size: u64 = parsed.size.parse().expect("size should parse as u64");
        assert_eq!(size, 134217728);
    }

    #[test]
    fn test_tensor_descriptor_json_large_values() {
        let desc = TensorDescriptorJson {
            name: "test".to_string(),
            addr: u64::MAX.to_string(),
            size: u64::MAX.to_string(),
            device_id: 7,
            dtype: "float16".to_string(),
        };

        let json = serde_json::to_string(&desc).expect("serialize");
        let parsed: TensorDescriptorJson = serde_json::from_str(&json).expect("deserialize");

        let addr: u64 = parsed.addr.parse().expect("max u64 addr should parse");
        assert_eq!(addr, u64::MAX);
    }
}
