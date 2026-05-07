// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::OciProvider;
use crate::providers::ModelProviderTrait;
use anyhow::Result;
use std::fmt;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactPath {
    path: PathBuf,
    raw: String,
}

impl ArtifactPath {
    pub fn from_title(title: &str) -> Result<Self> {
        if title.is_empty() {
            anyhow::bail!("OCI layer title annotation must not be empty");
        }

        Self::from_relative_path(Path::new(title), &format!("OCI layer title '{title}'"))
    }

    pub fn from_relative_path(path: &Path, description: &str) -> Result<Self> {
        let mut relative_path = PathBuf::new();
        let mut parts = Vec::new();

        for component in path.components() {
            match component {
                Component::Normal(part) => {
                    let Some(part) = part.to_str() else {
                        anyhow::bail!("{description} contains non-UTF-8 path data");
                    };
                    if part.contains('\\') {
                        anyhow::bail!("{description} must use forward-slash relative paths");
                    }
                    relative_path.push(part);
                    parts.push(part.to_owned());
                }
                Component::CurDir => {
                    anyhow::bail!("{description} must not contain '.' path components");
                }
                Component::ParentDir => {
                    anyhow::bail!("{description} must not contain '..' path components");
                }
                Component::RootDir | Component::Prefix(_) => {
                    anyhow::bail!("{description} must be a relative path");
                }
            }
        }

        if parts.is_empty() {
            anyhow::bail!("{description} must name a path");
        }

        Ok(Self {
            path: relative_path,
            raw: parts.join("/"),
        })
    }

    pub fn as_path(&self) -> &Path {
        &self.path
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn is_skipped(&self, ignore_weights: bool) -> bool {
        OciProvider::is_ignored(self.as_str())
            || OciProvider::is_image(self.as_path())
            || (ignore_weights && OciProvider::is_weight_file(self.as_str()))
    }
}

impl fmt::Display for ArtifactPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_title_path_rejects_unsafe_paths() {
        for title in [
            "",
            ".",
            "..",
            "../model.bin",
            "/model.bin",
            "a/../model.bin",
        ] {
            assert!(
                ArtifactPath::from_title(title).is_err(),
                "title should be rejected: {title}"
            );
        }

        let artifact_path =
            ArtifactPath::from_title("nested/config.json").expect("safe path should pass");
        assert_eq!(artifact_path.as_path(), Path::new("nested/config.json"));
        assert_eq!(artifact_path.as_str(), "nested/config.json");
    }
}
