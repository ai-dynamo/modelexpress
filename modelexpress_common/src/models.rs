// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request model for client -> server communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    pub action: String,
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

/// Status model for server health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Status {
    pub version: String,
    pub status: String,
    pub uptime: u64,
}

/// Status of a model download
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    /// Model is currently being downloaded
    DOWNLOADING,
    /// Model has been successfully downloaded
    DOWNLOADED,
    /// Model download failed with an error
    ERROR,
}

/// Supported model providers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum ModelProvider {
    /// Hugging Face model hub
    #[default]
    HuggingFace,
}

/// Response for model status request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusResponse {
    pub model_name: String,
    pub status: ModelStatus,
    pub provider: ModelProvider,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_model_status_serialization() {
        let status = ModelStatus::DOWNLOADING;
        let serialized = serde_json::to_string(&status).expect("Failed to serialize ModelStatus");
        let deserialized: ModelStatus =
            serde_json::from_str(&serialized).expect("Failed to deserialize ModelStatus");
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_model_provider_serialization() {
        let provider = ModelProvider::HuggingFace;
        let serialized =
            serde_json::to_string(&provider).expect("Failed to serialize ModelProvider");
        let deserialized: ModelProvider =
            serde_json::from_str(&serialized).expect("Failed to deserialize ModelProvider");
        assert_eq!(provider, deserialized);
    }

    #[test]
    fn test_model_provider_default() {
        let provider = ModelProvider::default();
        assert_eq!(provider, ModelProvider::HuggingFace);
    }

    #[test]
    fn test_request_serialization() {
        let mut payload = HashMap::new();
        payload.insert("key".to_string(), json!("value"));

        let request = Request {
            id: "test-id".to_string(),
            action: "test-action".to_string(),
            payload: Some(payload),
        };

        let serialized = serde_json::to_string(&request).expect("Failed to serialize Request");
        let deserialized: Request =
            serde_json::from_str(&serialized).expect("Failed to deserialize Request");

        assert_eq!(request.id, deserialized.id);
        assert_eq!(request.action, deserialized.action);
        assert!(request.payload.is_some());
        assert!(deserialized.payload.is_some());
    }

    #[test]
    fn test_status_serialization() {
        let status = Status {
            version: "1.0.0".to_string(),
            status: "ok".to_string(),
            uptime: 3600,
        };

        let serialized = serde_json::to_string(&status).expect("Failed to serialize Status");
        let deserialized: Status =
            serde_json::from_str(&serialized).expect("Failed to deserialize Status");

        assert_eq!(status.version, deserialized.version);
        assert_eq!(status.status, deserialized.status);
        assert_eq!(status.uptime, deserialized.uptime);
    }

    #[test]
    fn test_model_status_response_serialization() {
        let response = ModelStatusResponse {
            model_name: "test-model".to_string(),
            status: ModelStatus::DOWNLOADED,
            provider: ModelProvider::HuggingFace,
        };

        let serialized =
            serde_json::to_string(&response).expect("Failed to serialize ModelStatusResponse");
        let deserialized: ModelStatusResponse =
            serde_json::from_str(&serialized).expect("Failed to deserialize ModelStatusResponse");

        assert_eq!(response.model_name, deserialized.model_name);
        assert_eq!(response.status, deserialized.status);
        assert_eq!(response.provider, deserialized.provider);
    }

    #[test]
    fn test_model_status_all_variants() {
        assert_eq!(ModelStatus::DOWNLOADING, ModelStatus::DOWNLOADING);
        assert_eq!(ModelStatus::DOWNLOADED, ModelStatus::DOWNLOADED);
        assert_eq!(ModelStatus::ERROR, ModelStatus::ERROR);

        assert_ne!(ModelStatus::DOWNLOADING, ModelStatus::DOWNLOADED);
        assert_ne!(ModelStatus::DOWNLOADED, ModelStatus::ERROR);
        assert_ne!(ModelStatus::ERROR, ModelStatus::DOWNLOADING);
    }
}
