// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes CRD types for the model registry.
//!
//! Defines `ModelCacheEntry` — a custom resource that holds model-download lifecycle
//! state (phase + timestamps + message) for one cached model. This is the registry
//! analogue of the P2P `ModelMetadata` CRD, kept in a separate Kind so cardinality,
//! lifecycle, and RBAC scopes stay independent.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// ModelCacheEntry spec - desired state. One CR per cached model.
#[derive(CustomResource, Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[kube(
    group = "modelexpress.nvidia.com",
    version = "v1alpha1",
    kind = "ModelCacheEntry",
    plural = "modelcacheentries",
    shortname = "mxcache",
    namespaced,
    status = "ModelCacheEntryStatus"
)]
pub struct ModelCacheEntrySpec {
    /// Full model name (e.g., `meta-llama/Llama-3.1-70B`). Preserved in spec so
    /// operators can recover the original name even when the CR's metadata.name is a
    /// sanitized/hashed form.
    #[serde(rename = "modelName")]
    pub model_name: String,

    /// Provider string — `"HuggingFace"` or `"Ngc"`.
    pub provider: String,
}

/// ModelCacheEntry status - observed state.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
pub struct ModelCacheEntryStatus {
    /// One of `"Downloading"`, `"Downloaded"`, `"Error"`. Empty string on freshly-created
    /// records that haven't yet received a status patch.
    #[serde(default)]
    pub phase: String,

    /// RFC3339 timestamp of first write. Omitted until the first status patch.
    #[serde(rename = "createdAt", default)]
    pub created_at: Option<String>,

    /// RFC3339 timestamp of most recent status write or touch.
    #[serde(rename = "lastUsedAt", default)]
    pub last_used_at: Option<String>,

    /// Optional human-readable message (download progress, error reason).
    #[serde(default)]
    pub message: Option<String>,
}

/// Phase strings used by the CRD. Match `ModelStatus` one-for-one.
pub mod phase {
    pub const DOWNLOADING: &str = "Downloading";
    pub const DOWNLOADED: &str = "Downloaded";
    pub const ERROR: &str = "Error";
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn spec_roundtrips_through_json() {
        let spec = ModelCacheEntrySpec {
            model_name: "meta-llama/Llama-3.1-70B".to_string(),
            provider: "HuggingFace".to_string(),
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        assert!(json.contains("\"modelName\":\"meta-llama/Llama-3.1-70B\""));
        let back: ModelCacheEntrySpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_name, spec.model_name);
        assert_eq!(back.provider, spec.provider);
    }

    #[test]
    fn status_default_is_empty() {
        let status = ModelCacheEntryStatus::default();
        assert_eq!(status.phase, "");
        assert!(status.created_at.is_none());
        assert!(status.last_used_at.is_none());
        assert!(status.message.is_none());
    }
}
