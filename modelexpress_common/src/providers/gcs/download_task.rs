// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::path_ext::PathExt;
use anyhow::{Context, Result};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

const LOCKS_DIR_NAME: &str = "locks";
const PARTS_DIR_NAME: &str = "parts";

#[derive(Debug, Clone)]
pub struct DownloadTask {
    pub object_name: String,
    pub relative_path: PathBuf,
    pub destination_path: PathBuf,
    pub internal_dir: PathBuf,
    pub expected_size: Option<u64>,
    pub generation: Option<i64>,
    pub expected_crc32c: Option<u32>,
}

impl DownloadTask {
    fn sidecar_path(&self, subdir: &str, suffix: &str) -> PathBuf {
        let mut suffixed_name = self
            .relative_path
            .file_name()
            .map(|value| value.to_os_string())
            .unwrap_or_else(|| OsStr::new("download").to_os_string());
        suffixed_name.push(suffix);

        let suffixed_relative_path = self.relative_path.with_file_name(suffixed_name);
        self.internal_dir.join(subdir).join(suffixed_relative_path)
    }

    pub fn lock_path(&self) -> PathBuf {
        self.sidecar_path(LOCKS_DIR_NAME, ".lock")
    }

    pub fn temp_path(&self) -> PathBuf {
        self.sidecar_path(PARTS_DIR_NAME, ".part")
    }

    async fn verify_file_crc32c(&self, path: &Path) -> Result<()> {
        let Some(expected_crc32c) = self.expected_crc32c else {
            anyhow::bail!(
                "Cannot verify file '{}' for '{}': remote CRC32C is unavailable",
                path.display(),
                self.object_name
            );
        };

        let verify_path = path.to_path_buf();
        let actual_crc32c =
            tokio::task::spawn_blocking(move || verify_path.calculate_file_crc32c())
                .await
                .with_context(|| {
                    format!("CRC32C verification task panicked for '{}'", path.display())
                })??;

        if actual_crc32c != expected_crc32c {
            anyhow::bail!(
                "CRC32C mismatch for file '{}' for '{}': expected {:08x}, got {:08x}",
                path.display(),
                self.object_name,
                expected_crc32c,
                actual_crc32c
            );
        }

        Ok(())
    }

    pub async fn has_cached_destination_file(&self) -> Result<bool> {
        let metadata = match tokio::fs::metadata(&self.destination_path).await {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Failed to inspect destination '{}': {}",
                    self.destination_path.display(),
                    err
                ));
            }
        };

        if !metadata.is_file() {
            return Ok(false);
        }

        Ok(true)
    }

    pub async fn remove_temp_file_if_present(&self, temp_path: &Path) -> Result<()> {
        match tokio::fs::remove_file(temp_path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(anyhow::anyhow!(
                "Failed to remove stale temp file '{}': {}",
                temp_path.display(),
                err
            )),
        }
    }

    pub async fn promote_verified_temp_file(&self, temp_path: &Path) -> Result<()> {
        if let Err(err) = self.verify_file_crc32c(temp_path).await {
            if let Err(remove_err) = tokio::fs::remove_file(temp_path).await {
                return Err(anyhow::anyhow!(
                    "{}; also failed to discard unverified temp file '{}': {}",
                    err,
                    temp_path.display(),
                    remove_err
                ));
            }
            return Err(err);
        }

        tokio::fs::rename(temp_path, &self.destination_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to move '{}' to '{}'",
                    temp_path.display(),
                    self.destination_path.display()
                )
            })?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::model_dir::INTERNAL_METADATA_DIR_NAME;
    use super::*;
    use std::{fs, path::Path};
    use tempfile::TempDir;

    fn test_download_task(
        root: &Path,
        relative_path: &str,
        expected_crc32c: Option<u32>,
    ) -> DownloadTask {
        let relative_path = PathBuf::from(relative_path);
        DownloadTask {
            object_name: format!("org/model/rev/{}", relative_path.display()),
            relative_path: relative_path.clone(),
            destination_path: root.join(&relative_path),
            internal_dir: root.join(INTERNAL_METADATA_DIR_NAME),
            expected_size: None,
            generation: Some(42),
            expected_crc32c,
        }
    }

    #[test]
    fn test_download_task_sidecar_paths() {
        for (relative_path, expected_temp, expected_lock) in [
            (
                "weights/model.bin",
                "/tmp/cache/model/.mx/parts/weights/model.bin.part",
                "/tmp/cache/model/.mx/locks/weights/model.bin.lock",
            ),
            (
                "weights/model.mlir",
                "/tmp/cache/model/.mx/parts/weights/model.mlir.part",
                "/tmp/cache/model/.mx/locks/weights/model.mlir.lock",
            ),
            (
                "weights/model",
                "/tmp/cache/model/.mx/parts/weights/model.part",
                "/tmp/cache/model/.mx/locks/weights/model.lock",
            ),
        ] {
            let task = test_download_task(Path::new("/tmp/cache/model"), relative_path, None);
            assert_eq!(task.temp_path(), PathBuf::from(expected_temp));
            assert_eq!(task.lock_path(), PathBuf::from(expected_lock));
        }
    }

    #[tokio::test]
    async fn test_download_task_file_verification_cases() {
        struct PromoteCase {
            name: &'static str,
            temp_payload: &'static [u8],
            expected_crc32c: Option<u32>,
            expect_error_substr: Option<&'static str>,
        }

        for case in [
            PromoteCase {
                name: "match",
                temp_payload: b"weights",
                expected_crc32c: Some(crc32c::crc32c(b"weights")),
                expect_error_substr: None,
            },
            PromoteCase {
                name: "mismatch",
                temp_payload: b"weights",
                expected_crc32c: Some(crc32c::crc32c(b"different")),
                expect_error_substr: Some("CRC32C mismatch"),
            },
            PromoteCase {
                name: "missing_crc",
                temp_payload: b"weights",
                expected_crc32c: None,
                expect_error_substr: Some("remote CRC32C is unavailable"),
            },
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let temp_path = temp_dir.path().join("weights.bin.tmp");
            fs::write(&temp_path, case.temp_payload).expect("Failed to write temp file");

            let task = test_download_task(temp_dir.path(), "weights.bin", case.expected_crc32c);
            let destination_path = task.destination_path.clone();

            match case.expect_error_substr {
                None => {
                    task.promote_verified_temp_file(&temp_path)
                        .await
                        .expect("Expected checksum verification");
                    assert!(!temp_path.exists(), "scenario={}", case.name);
                    assert_eq!(
                        fs::read(&destination_path).expect("Expected promoted destination file"),
                        case.temp_payload,
                        "scenario={}",
                        case.name
                    );
                }
                Some(error_substr) => {
                    let err = task
                        .promote_verified_temp_file(&temp_path)
                        .await
                        .expect_err("Expected promote failure");
                    assert!(
                        err.to_string().contains(error_substr),
                        "scenario={} err={err}",
                        case.name
                    );
                    assert!(!temp_path.exists(), "scenario={}", case.name);
                    assert!(!destination_path.exists(), "scenario={}", case.name);
                }
            }
        }

        for (name, setup, expected_match) in [
            ("file", "file", true),
            ("missing", "missing", false),
            ("directory", "directory", false),
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let task = test_download_task(
                temp_dir.path(),
                "weights.bin",
                Some(crc32c::crc32c(b"weights")),
            );

            match setup {
                "file" => {
                    fs::write(&task.destination_path, b"weights")
                        .expect("Failed to create destination file");
                }
                "missing" => {}
                "directory" => {
                    fs::create_dir_all(&task.destination_path)
                        .expect("Failed to create destination directory");
                }
                _ => unreachable!("unexpected setup"),
            }

            assert_eq!(
                task.has_cached_destination_file()
                    .await
                    .expect("Expected destination inspection"),
                expected_match,
                "scenario={name}"
            );
        }
    }
}
