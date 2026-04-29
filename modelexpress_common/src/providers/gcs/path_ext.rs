// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use crc32c::Crc32cReader;
use std::fs;
use std::io;
use std::path::{Component, Path};

pub trait PathExt {
    fn calculate_file_crc32c(&self) -> Result<u32>;
    fn is_safe_relative(&self) -> bool;
}

impl PathExt for Path {
    fn calculate_file_crc32c(&self) -> Result<u32> {
        let file = fs::File::open(self).with_context(|| {
            format!(
                "Failed to open '{}' for CRC32C verification",
                self.display()
            )
        })?;
        let mut reader = Crc32cReader::new(file);
        io::copy(&mut reader, &mut io::sink()).with_context(|| {
            format!(
                "Failed to read '{}' for CRC32C verification",
                self.display()
            )
        })?;
        Ok(reader.crc32c())
    }

    fn is_safe_relative(&self) -> bool {
        if self.is_absolute() {
            return false;
        }

        self.components()
            .all(|component| matches!(component, Component::Normal(_)))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_is_safe_relative_path_cases() {
        for (path, expected) in [
            ("tokenizer.json", true),
            ("weights/model.bin", true),
            (".", false),
            ("..", false),
            ("weights/../model.bin", false),
            ("/tmp/model.bin", false),
        ] {
            assert_eq!(Path::new(path).is_safe_relative(), expected, "path={path}");
        }
    }

    #[test]
    fn test_calculate_file_crc32c() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("weights.bin");
        fs::write(&path, b"weights").expect("Failed to write checksum payload");

        assert_eq!(
            path.as_path()
                .calculate_file_crc32c()
                .expect("Expected checksum calculation"),
            crc32c::crc32c(b"weights")
        );
    }
}
