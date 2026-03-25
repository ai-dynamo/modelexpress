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
    /// Per-worker P2P transfer metadata and lifecycle state
    #[serde(default)]
    pub workers: Vec<WorkerStatus>,

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

    /// Endpoint (host:port) where this worker's NIXL listen thread serves metadata.
    #[serde(rename = "metadataEndpoint", default)]
    pub metadata_endpoint: Option<String>,

    /// NIXL agent name for this worker
    #[serde(rename = "agentName", default)]
    pub agent_name: Option<String>,

    /// Mooncake TransferEngine session ID
    #[serde(rename = "transferEngineSessionId", default)]
    pub transfer_engine_session_id: Option<String>,

    /// Endpoint (host:port) for this worker's gRPC server (WorkerService)
    #[serde(rename = "workerGrpcEndpoint", default)]
    pub worker_grpc_endpoint: Option<String>,

    /// Worker lifecycle status (Initializing, Ready, Stale)
    #[serde(default)]
    pub status: String,

    /// Timestamp of last status update (RFC3339)
    #[serde(rename = "updatedAt", default)]
    pub updated_at: Option<String>,
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
}
