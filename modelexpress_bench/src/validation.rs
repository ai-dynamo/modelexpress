// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-chunk validation that mirrors the production client's checks.
//!
//! When the benchmark runs in `--strict` mode this is invoked for every
//! incoming `FileChunk` so the measured CPU cost matches what production
//! would actually pay. In non-strict mode the client skips this entirely
//! and only counts bytes; the difference between the two reveals how much
//! the validation layer is contributing per chunk.

use anyhow::{Result, anyhow};
use modelexpress_common::grpc::model::FileChunk;

#[derive(Debug, Default)]
pub struct StrictValidator {
    current_file: Option<FileState>,
    files_complete: u64,
    saw_final_chunk: bool,
}

#[derive(Debug)]
struct FileState {
    relative_path: String,
    expected_size: u64,
    bytes_seen: u64,
    saw_last_chunk: bool,
}

impl StrictValidator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn files_complete(&self) -> u64 {
        self.files_complete
    }

    pub fn saw_final_chunk(&self) -> bool {
        self.saw_final_chunk
    }

    /// Mirror of `stream_model_files_from_server` invariants in the client.
    /// Returns Ok on a valid chunk and a descriptive error on any violation.
    pub fn observe(&mut self, chunk: &FileChunk) -> Result<()> {
        if self.saw_final_chunk {
            return Err(anyhow!(
                "received chunk after final stream marker: {:?}",
                chunk.relative_path
            ));
        }
        if chunk.offset > chunk.total_size {
            return Err(anyhow!(
                "chunk offset {} exceeds total_size {} on {:?}",
                chunk.offset,
                chunk.total_size,
                chunk.relative_path
            ));
        }

        let need_new_file = match &self.current_file {
            None => true,
            Some(state) => state.relative_path != chunk.relative_path,
        };

        if need_new_file {
            if let Some(prev) = self.current_file.take() {
                if !prev.saw_last_chunk || prev.bytes_seen != prev.expected_size {
                    return Err(anyhow!(
                        "previous file {:?} ended incomplete: saw_last_chunk={}, bytes={}, expected={}",
                        prev.relative_path,
                        prev.saw_last_chunk,
                        prev.bytes_seen,
                        prev.expected_size
                    ));
                }
                self.files_complete = self.files_complete.saturating_add(1);
            }
            if chunk.offset != 0 {
                return Err(anyhow!(
                    "first chunk for {:?} has offset {}, expected 0",
                    chunk.relative_path,
                    chunk.offset
                ));
            }
            self.current_file = Some(FileState {
                relative_path: chunk.relative_path.clone(),
                expected_size: chunk.total_size,
                bytes_seen: 0,
                saw_last_chunk: false,
            });
        }

        let state = self
            .current_file
            .as_mut()
            .ok_or_else(|| anyhow!("validator state lost current file"))?;

        if state.saw_last_chunk {
            return Err(anyhow!(
                "received extra chunk for completed file {:?}",
                chunk.relative_path
            ));
        }
        if chunk.total_size != state.expected_size {
            return Err(anyhow!(
                "inconsistent total_size for {:?}: expected {}, got {}",
                chunk.relative_path,
                state.expected_size,
                chunk.total_size
            ));
        }
        if chunk.offset != state.bytes_seen {
            return Err(anyhow!(
                "out-of-order offset {} for {:?}, expected {}",
                chunk.offset,
                chunk.relative_path,
                state.bytes_seen
            ));
        }
        let chunk_len = chunk.data.len() as u64;
        let next = state.bytes_seen.saturating_add(chunk_len);
        if next > state.expected_size {
            return Err(anyhow!(
                "chunk for {:?} exceeds advertised size {}",
                chunk.relative_path,
                state.expected_size
            ));
        }
        state.bytes_seen = next;

        if chunk.is_last_chunk {
            if state.bytes_seen != state.expected_size {
                return Err(anyhow!(
                    "is_last_chunk on {:?} but {} of {} bytes received",
                    chunk.relative_path,
                    state.bytes_seen,
                    state.expected_size
                ));
            }
            state.saw_last_chunk = true;
        } else if state.bytes_seen == state.expected_size {
            return Err(anyhow!(
                "non-final chunk completed file {:?}",
                chunk.relative_path
            ));
        }

        if chunk.is_last_file && chunk.is_last_chunk {
            self.saw_final_chunk = true;
        }
        Ok(())
    }

    /// Final invariants after the stream closes.
    pub fn finish(mut self) -> Result<u64> {
        if let Some(prev) = self.current_file.take() {
            if !prev.saw_last_chunk || prev.bytes_seen != prev.expected_size {
                return Err(anyhow!(
                    "stream ended mid-file {:?}: saw_last_chunk={}, bytes={}, expected={}",
                    prev.relative_path,
                    prev.saw_last_chunk,
                    prev.bytes_seen,
                    prev.expected_size
                ));
            }
            self.files_complete = self.files_complete.saturating_add(1);
        }
        if !self.saw_final_chunk {
            return Err(anyhow!("stream ended before final-file/final-chunk marker"));
        }
        Ok(self.files_complete)
    }
}
