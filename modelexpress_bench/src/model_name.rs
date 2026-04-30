// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoding helpers for the synthetic `model_name` field.
//!
//! `ModelFilesRequest` has no `total_bytes` field, so we smuggle the desired
//! synthetic payload size and file-count through the existing `model_name`
//! string. The encoding is unambiguous and round-trips through the proto
//! without any other change.
//!
//! Format: `bench:<bytes>[:<files>]`
//!
//! Examples:
//! - `bench:1073741824` -> 1 file of 1 GiB
//! - `bench:1073741824:8` -> 8 files of 1 GiB each (total 8 GiB)

use anyhow::{Context, Result, anyhow};

/// Parsed synthetic-model spec carried inside `ModelFilesRequest.model_name`.
#[derive(Debug, Clone, Copy)]
pub struct BenchSpec {
    pub bytes_per_file: u64,
    pub file_count: u64,
}

impl BenchSpec {
    pub fn new(bytes_per_file: u64, file_count: u64) -> Self {
        Self {
            bytes_per_file,
            file_count,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.bytes_per_file.saturating_mul(self.file_count)
    }

    pub fn encode(&self) -> String {
        format!("bench:{}:{}", self.bytes_per_file, self.file_count)
    }

    pub fn parse(name: &str) -> Result<Self> {
        let rest = name
            .strip_prefix("bench:")
            .ok_or_else(|| anyhow!("model_name does not start with 'bench:': {name:?}"))?;
        let mut parts = rest.split(':');
        let bytes = parts
            .next()
            .ok_or_else(|| anyhow!("missing byte count in {name:?}"))?
            .parse::<u64>()
            .with_context(|| format!("parsing byte count from {name:?}"))?;
        let files = match parts.next() {
            Some(s) => s
                .parse::<u64>()
                .with_context(|| format!("parsing file count from {name:?}"))?,
            None => 1,
        };
        if files == 0 {
            return Err(anyhow!("file count must be >= 1"));
        }
        if parts.next().is_some() {
            return Err(anyhow!("unexpected trailing component in {name:?}"));
        }
        Ok(Self::new(bytes, files))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_default_files() {
        let spec = BenchSpec::new(1024 * 1024, 1);
        let encoded = spec.encode();
        let parsed = BenchSpec::parse(&encoded).expect("round trip");
        assert_eq!(parsed.bytes_per_file, spec.bytes_per_file);
        assert_eq!(parsed.file_count, spec.file_count);
    }

    #[test]
    fn round_trip_multi_files() {
        let spec = BenchSpec::new(2 * 1024 * 1024 * 1024, 8);
        let parsed = BenchSpec::parse(&spec.encode()).expect("round trip");
        assert_eq!(parsed.total_bytes(), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn rejects_missing_prefix() {
        assert!(BenchSpec::parse("1024").is_err());
    }

    #[test]
    fn rejects_zero_files() {
        assert!(BenchSpec::parse("bench:1024:0").is_err());
    }

    #[test]
    fn rejects_trailing_components() {
        assert!(BenchSpec::parse("bench:1024:8:extra").is_err());
    }
}
