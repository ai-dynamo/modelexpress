// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::model_name::ModelName;
use serde::{Deserialize, Serialize};

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheManifest {
    pub version: u32,
    pub model: String,
    pub files: Vec<ManifestEntry>,
}

impl CacheManifest {
    pub fn new(model: &ModelName, mut files: Vec<ManifestEntry>) -> Self {
        files.sort_by(|left, right| left.path.cmp(&right.path));
        Self {
            version: MANIFEST_VERSION,
            model: model.to_string(),
            files,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub path: String,
    pub size: u64,
    pub crc32c: String,
    pub generation: Option<u64>,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::model_name::{BucketName, ModelName};
    use super::*;

    #[test]
    fn test_cache_manifest_new_sorts_files_and_sets_metadata() {
        let model = ModelName::new(
            BucketName::parse("test-bucket").expect("Expected bucket parse"),
            "org/model/rev",
        )
        .expect("Expected model name");

        let manifest = CacheManifest::new(
            &model,
            vec![
                ManifestEntry {
                    path: "weights/model.bin".to_string(),
                    size: 7,
                    crc32c: "00000001".to_string(),
                    generation: Some(2),
                },
                ManifestEntry {
                    path: "config.json".to_string(),
                    size: 2,
                    crc32c: "00000002".to_string(),
                    generation: Some(1),
                },
            ],
        );

        assert_eq!(manifest.version, MANIFEST_VERSION);
        assert_eq!(manifest.model, "gs://test-bucket/org/model/rev");
        assert_eq!(
            manifest
                .files
                .iter()
                .map(|entry| entry.path.as_str())
                .collect::<Vec<_>>(),
            vec!["config.json", "weights/model.bin"]
        );
    }
}
