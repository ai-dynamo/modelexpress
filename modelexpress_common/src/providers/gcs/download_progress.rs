// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::download_task::DownloadTask;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use tracing::info;

const PROGRESS_LOG_INTERVAL_BYTES: u64 = 128 * 1024 * 1024;

pub struct DownloadProgress {
    total_files: usize,
    total_bytes: Option<u64>,
    completed_files: AtomicUsize,
    downloaded_bytes: AtomicU64,
    next_log_bytes: AtomicU64,
    started_at: Instant,
}

impl DownloadProgress {
    pub fn new(tasks: &[DownloadTask]) -> Self {
        let mut total_bytes = Some(0u64);
        for task in tasks {
            total_bytes = match (total_bytes, task.expected_size) {
                (Some(acc), Some(size)) => acc.checked_add(size),
                _ => None,
            };
        }

        Self {
            total_files: tasks.len(),
            total_bytes,
            completed_files: AtomicUsize::new(0),
            downloaded_bytes: AtomicU64::new(0),
            next_log_bytes: AtomicU64::new(PROGRESS_LOG_INTERVAL_BYTES),
            started_at: Instant::now(),
        }
    }

    pub fn log_start(&self, max_parallel_downloads: usize) {
        let total_size = self
            .total_bytes
            .map(Self::format_bytes)
            .unwrap_or_else(|| "unknown".to_string());
        info!(
            "Starting GCS parallel download: {} files ({} total) with {} workers",
            self.total_files, total_size, max_parallel_downloads
        );
    }

    pub fn record_downloaded_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }

        let previous = self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
        let current = previous.saturating_add(bytes);
        self.maybe_log_threshold_progress(current);
    }

    pub fn mark_file_completed(&self, object_name: &str) {
        let previous = self.completed_files.fetch_add(1, Ordering::Relaxed);
        let completed = previous.saturating_add(1);
        self.log_progress(&format!(
            "Completed '{}' ({}/{})",
            object_name, completed, self.total_files
        ));
    }

    pub fn log_finish(&self) {
        let elapsed = self.started_at.elapsed();
        self.log_progress(&format!(
            "Finished GCS download in {:.1}s",
            elapsed.as_secs_f64()
        ));
    }

    fn maybe_log_threshold_progress(&self, current_bytes: u64) {
        let mut threshold = self.next_log_bytes.load(Ordering::Relaxed);
        while current_bytes >= threshold {
            match self.next_log_bytes.compare_exchange(
                threshold,
                threshold.saturating_add(PROGRESS_LOG_INTERVAL_BYTES),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.log_progress("GCS download progress");
                    break;
                }
                Err(observed) => threshold = observed,
            }
        }
    }

    fn log_progress(&self, prefix: &str) {
        let completed = self.completed_files.load(Ordering::Relaxed);
        let downloaded = self.downloaded_bytes.load(Ordering::Relaxed);
        match self.total_bytes {
            Some(total) if total > 0 => {
                let percent = ((downloaded as f64) * 100.0 / (total as f64)).min(100.0);
                info!(
                    "{}: files {}/{}, bytes {}/{} ({:.1}%)",
                    prefix,
                    completed,
                    self.total_files,
                    Self::format_bytes(downloaded),
                    Self::format_bytes(total),
                    percent
                );
            }
            Some(_) | None => {
                info!(
                    "{}: files {}/{}, bytes {}",
                    prefix,
                    completed,
                    self.total_files,
                    Self::format_bytes(downloaded)
                );
            }
        }
    }

    fn format_bytes(bytes: u64) -> String {
        const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
        let mut value = bytes as f64;
        let mut unit_idx = 0usize;
        let last_unit_idx = UNITS.len().saturating_sub(1);
        while value >= 1024.0 && unit_idx < last_unit_idx {
            value /= 1024.0;
            unit_idx = unit_idx.saturating_add(1);
        }
        format!("{value:.1} {}", UNITS[unit_idx])
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::{
        download_task::DownloadTask, model_dir::INTERNAL_METADATA_DIR_NAME, model_name::BucketName,
        path_ext::PathExt,
    };
    use super::*;
    use std::{fs, path::PathBuf, sync::atomic::Ordering};
    use tempfile::TempDir;

    #[test]
    fn test_progress_and_crc_helpers() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let checksum_path = temp_dir.path().join("weights.bin");
        fs::write(&checksum_path, b"weights").expect("Failed to write checksum payload");
        assert_eq!(
            checksum_path
                .as_path()
                .calculate_file_crc32c()
                .expect("Expected checksum calculation"),
            crc32c::crc32c(b"weights")
        );

        let tasks = vec![
            DownloadTask {
                object_name: "org/model/rev123/tokenizer.json".to_string(),
                relative_path: PathBuf::from("tokenizer.json"),
                destination_path: temp_dir.path().join("tokenizer.json"),
                internal_dir: temp_dir.path().join(INTERNAL_METADATA_DIR_NAME),
                expected_size: Some(2),
                generation: Some(51),
                expected_crc32c: Some(crc32c::crc32c(b"{}")),
            },
            DownloadTask {
                object_name: "org/model/rev123/weights/model.bin".to_string(),
                relative_path: PathBuf::from("weights/model.bin"),
                destination_path: temp_dir.path().join("weights/model.bin"),
                internal_dir: temp_dir.path().join(INTERNAL_METADATA_DIR_NAME),
                expected_size: Some(7),
                generation: Some(52),
                expected_crc32c: Some(crc32c::crc32c(b"weights")),
            },
        ];
        let progress = DownloadProgress::new(&tasks);
        assert_eq!(progress.total_bytes, Some(9));
        progress.record_downloaded_bytes(2);
        progress.mark_file_completed("tokenizer.json");
        assert_eq!(progress.completed_files.load(Ordering::Relaxed), 1);
        assert_eq!(progress.downloaded_bytes.load(Ordering::Relaxed), 2);
        progress.log_start(8);
        progress.log_finish();

        assert_eq!(DownloadProgress::format_bytes(0), "0.0 B");
        assert_eq!(DownloadProgress::format_bytes(1536), "1.5 KiB");
        assert_eq!(
            BucketName::parse("gs://bucket-name/")
                .expect("Expected bucket parse")
                .resource_name(),
            "projects/_/buckets/bucket-name"
        );
    }
}
