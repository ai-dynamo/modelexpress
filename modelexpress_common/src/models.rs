// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{ValueEnum, builder::PossibleValue};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ModelProvider {
    /// Hugging Face model hub
    #[default]
    HuggingFace,
    /// NVIDIA NGC catalog
    Ngc,
}

impl ModelProvider {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HuggingFace => "hugging-face",
            Self::Ngc => "ngc",
        }
    }
}

impl Display for ModelProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ValueEnum for ModelProvider {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::HuggingFace, Self::Ngc]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(self.as_str()))
    }
}

/// Controls which weight file formats to download.
///
/// When set to `Auto` (the default), safetensors files are preferred over other
/// formats, and sharded vs consolidated duplicates within the same format are
/// deduplicated. Other variants restrict downloads to a single format, or
/// disable filtering entirely (`All`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum WeightFormat {
    /// Smart defaults: prefer safetensors, deduplicate sharded vs consolidated
    #[default]
    Auto,
    /// Only download safetensors files
    Safetensors,
    /// Only download pytorch bin files
    Pytorch,
    /// Download all weight formats (current/legacy behavior)
    All,
}

impl WeightFormat {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safetensors => "safetensors",
            Self::Pytorch => "pytorch",
            Self::All => "all",
        }
    }
}

impl Display for WeightFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for WeightFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "safetensors" => Ok(Self::Safetensors),
            "pytorch" => Ok(Self::Pytorch),
            "all" => Ok(Self::All),
            _ => Err(format!(
                "invalid weight format '{s}': expected one of auto, safetensors, pytorch, all"
            )),
        }
    }
}

impl ValueEnum for WeightFormat {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Auto, Self::Safetensors, Self::Pytorch, Self::All]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(PossibleValue::new(self.as_str()))
    }
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
    fn test_model_provider_display() {
        assert_eq!(ModelProvider::HuggingFace.to_string(), "hugging-face");
        assert_eq!(ModelProvider::Ngc.to_string(), "ngc");
    }

    #[test]
    fn test_model_provider_value_enum_matches_display() {
        for provider in [ModelProvider::HuggingFace, ModelProvider::Ngc] {
            let parsed = ModelProvider::from_str(provider.as_str(), false)
                .expect("Failed to parse ModelProvider from clap value");
            assert_eq!(parsed, provider);
        }
    }

    #[test]
    fn test_model_provider_ngc_serialization() {
        let provider = ModelProvider::Ngc;
        let serialized =
            serde_json::to_string(&provider).expect("Failed to serialize ModelProvider");
        let deserialized: ModelProvider =
            serde_json::from_str(&serialized).expect("Failed to deserialize ModelProvider");
        assert_eq!(provider, deserialized);
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

    #[test]
    fn test_weight_format_default() {
        assert_eq!(WeightFormat::default(), WeightFormat::Auto);
    }

    #[test]
    fn test_weight_format_display() {
        assert_eq!(WeightFormat::Auto.to_string(), "auto");
        assert_eq!(WeightFormat::Safetensors.to_string(), "safetensors");
        assert_eq!(WeightFormat::Pytorch.to_string(), "pytorch");
        assert_eq!(WeightFormat::All.to_string(), "all");
    }

    #[test]
    fn test_weight_format_from_str() {
        assert_eq!(
            "auto".parse::<WeightFormat>().expect("parse auto"),
            WeightFormat::Auto
        );
        assert_eq!(
            "safetensors"
                .parse::<WeightFormat>()
                .expect("parse safetensors"),
            WeightFormat::Safetensors
        );
        assert_eq!(
            "pytorch".parse::<WeightFormat>().expect("parse pytorch"),
            WeightFormat::Pytorch
        );
        assert_eq!(
            "all".parse::<WeightFormat>().expect("parse all"),
            WeightFormat::All
        );
        assert_eq!(
            "AUTO".parse::<WeightFormat>().expect("parse AUTO"),
            WeightFormat::Auto
        );
        assert!("invalid".parse::<WeightFormat>().is_err());
    }

    #[test]
    fn test_weight_format_serialization() {
        let format = WeightFormat::Safetensors;
        let serialized = serde_json::to_string(&format).expect("Failed to serialize WeightFormat");
        let deserialized: WeightFormat =
            serde_json::from_str(&serialized).expect("Failed to deserialize WeightFormat");
        assert_eq!(format, deserialized);
    }

    #[test]
    fn test_weight_format_value_enum_matches_display() {
        for format in [
            WeightFormat::Auto,
            WeightFormat::Safetensors,
            WeightFormat::Pytorch,
            WeightFormat::All,
        ] {
            let parsed = <WeightFormat as ValueEnum>::from_str(format.as_str(), false)
                .expect("Failed to parse WeightFormat from clap value");
            assert_eq!(parsed, format);
        }
    }
}
