// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::path::ArtifactPath;
use anyhow::{Context, Result};
use std::{
    collections::HashSet,
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    Tar,
    TarZstd,
}

impl ArchiveFormat {
    pub fn is_archive_media_type(media_type: &str) -> bool {
        let media_type = media_type.to_ascii_lowercase();

        media_type == "application/x-tar"
            || media_type.ends_with(".tar")
            || media_type.ends_with("+tar")
            || media_type.contains(".tar+")
            || media_type.contains("+tar+")
            || media_type.contains("-tar+")
    }

    pub fn from_media_type(media_type: &str) -> Result<Option<Self>> {
        let media_type = media_type.to_ascii_lowercase();

        if media_type == "application/x-tar"
            || media_type.ends_with(".tar")
            || media_type.ends_with("+tar")
        {
            return Ok(Some(Self::Tar));
        }

        if media_type.ends_with(".tar+zstd")
            || media_type.ends_with("+tar+zstd")
            || media_type.ends_with("-tar+zstd")
        {
            return Ok(Some(Self::TarZstd));
        }

        if media_type.contains(".tar+")
            || media_type.contains("+tar+")
            || media_type.contains("-tar+")
        {
            anyhow::bail!(
                "OCI archive layer media type '{media_type}' is not supported; supported archive formats are tar and tar+zstd"
            );
        }

        Ok(None)
    }

    pub fn extract_blob(self, blob_path: &Path, output_root: &Path) -> Result<Vec<String>> {
        let file = fs::File::open(blob_path)
            .with_context(|| format!("Failed to open OCI archive blob {blob_path:?}"))?;
        let reader = BufReader::new(file);

        match self {
            Self::Tar => TarExtractor::new(output_root).extract(reader),
            Self::TarZstd => {
                let decoder = zstd::stream::read::Decoder::new(reader)
                    .with_context(|| format!("Failed to create zstd decoder for {blob_path:?}"))?;
                TarExtractor::new(output_root).extract(decoder)
            }
        }
    }
}

struct TarExtractor<'a> {
    output_root: &'a Path,
    files: Vec<String>,
    seen_paths: HashSet<ArtifactPath>,
}

impl<'a> TarExtractor<'a> {
    fn new(output_root: &'a Path) -> Self {
        Self {
            output_root,
            files: Vec::new(),
            seen_paths: HashSet::new(),
        }
    }

    fn extract<R: Read>(mut self, reader: R) -> Result<Vec<String>> {
        let mut archive = tar::Archive::new(reader);

        for entry in archive
            .entries()
            .context("Failed to read OCI tar entries")?
        {
            self.extract_entry(entry.context("Failed to read OCI tar entry")?)?;
        }

        Ok(self.files)
    }

    fn extract_entry<R: Read>(&mut self, mut entry: tar::Entry<'_, R>) -> Result<()> {
        let entry_type = entry.header().entry_type();

        if entry_type.is_dir() {
            return Ok(());
        }

        if !entry_type.is_file() {
            anyhow::bail!(
                "OCI archive entry '{}' has unsupported type {:?}; only regular files are supported",
                Self::entry_path_for_error(&entry),
                entry_type
            );
        }

        let relative_path = Self::member_path(&entry)?;

        if relative_path.is_skipped(false) {
            tracing::debug!("Skipping OCI archive file: {relative_path}");
            return Ok(());
        }

        self.ensure_unique(&relative_path)?;
        let output_path = self.create_output_path(&relative_path)?;
        let mut output = Self::create_output_file(&output_path)?;

        std::io::copy(&mut entry, &mut output)
            .with_context(|| format!("Failed to extract OCI archive file '{relative_path}'"))?;
        output
            .sync_all()
            .with_context(|| format!("Failed to sync OCI archive output file {output_path:?}"))?;

        self.files.push(relative_path.to_string());
        Ok(())
    }

    fn member_path<R: Read>(entry: &tar::Entry<'_, R>) -> Result<ArtifactPath> {
        let path = entry.path().context("Failed to read OCI tar entry path")?;
        ArtifactPath::from_relative_path(
            path.as_ref(),
            &format!("OCI archive member '{}'", path.display()),
        )
    }

    fn entry_path_for_error<R: Read>(entry: &tar::Entry<'_, R>) -> String {
        entry.path().map_or_else(
            |_| "<unknown>".to_string(),
            |path| path.display().to_string(),
        )
    }

    fn ensure_unique(&mut self, relative_path: &ArtifactPath) -> Result<()> {
        if !self.seen_paths.insert(relative_path.clone()) {
            anyhow::bail!("Duplicate OCI archive file path '{relative_path}'");
        }
        Ok(())
    }

    fn create_output_path(&self, relative_path: &ArtifactPath) -> Result<PathBuf> {
        let output_path = self.output_root.join(relative_path.as_path());
        if output_path.exists() {
            anyhow::bail!("Duplicate OCI artifact file path '{relative_path}'");
        }

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create OCI archive output directory {parent:?}")
            })?;
        }

        Ok(output_path)
    }

    fn create_output_file(output_path: &Path) -> Result<fs::File> {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)
            .with_context(|| format!("Failed to create OCI archive output file {output_path:?}"))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::{fs, io::Write, path::PathBuf};
    use tempfile::TempDir;

    fn tar_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut bytes);
            for (path, contents) in entries {
                let mut header = tar::Header::new_gnu();
                header.set_size(contents.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder
                    .append_data(&mut header, path, *contents)
                    .expect("append tar entry");
            }
            builder.finish().expect("finish tar");
        }
        bytes
    }

    fn write_tar_octal(field: &mut [u8], value: u64) {
        let width = field.len().checked_sub(1).expect("tar field has room");
        let encoded = format!("{value:0width$o}\0");
        field.copy_from_slice(encoded.as_bytes());
    }

    fn unsafe_tar_bytes(path: &str, contents: &[u8]) -> Vec<u8> {
        let mut header = [0_u8; 512];
        header[..path.len()].copy_from_slice(path.as_bytes());
        write_tar_octal(&mut header[100..108], 0o644);
        write_tar_octal(&mut header[108..116], 0);
        write_tar_octal(&mut header[116..124], 0);
        write_tar_octal(&mut header[124..136], contents.len() as u64);
        write_tar_octal(&mut header[136..148], 0);
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let checksum: u64 = header.iter().map(|byte| u64::from(*byte)).sum();
        let checksum = format!("{checksum:06o}\0 ");
        header[148..156].copy_from_slice(checksum.as_bytes());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(contents);
        let padding = match contents.len() % 512 {
            0 => 0,
            remainder => 512usize
                .checked_sub(remainder)
                .expect("tar padding remainder is smaller than block size"),
        };
        bytes.extend(std::iter::repeat_n(0, padding));
        bytes.extend_from_slice(&[0_u8; 1024]);
        bytes
    }

    fn write_blob(dir: &TempDir, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join("blob");
        let mut file = fs::File::create(&path).expect("create blob");
        file.write_all(bytes).expect("write blob");
        path
    }

    #[test]
    fn test_archive_format_accepts_x_tar_zstd_media_type() {
        assert_eq!(
            ArchiveFormat::from_media_type("application/x-tar+zstd")
                .expect("media type should parse"),
            Some(ArchiveFormat::TarZstd)
        );
    }

    #[test]
    fn test_extract_tar_archive_applies_ignore_rules() {
        let dir = TempDir::new().expect("temp dir");
        let output = dir.path().join("out");
        fs::create_dir_all(&output).expect("create output");
        let tar = tar_bytes(&[
            ("program.0.gas", b"gas"),
            ("README.md", b"readme"),
            (".hidden", b"hidden"),
            ("diagram.png", b"image"),
        ]);
        let blob = write_blob(&dir, &tar);

        let files = ArchiveFormat::Tar
            .extract_blob(&blob, &output)
            .expect("extract archive");

        assert_eq!(files, vec!["program.0.gas".to_string()]);
        assert_eq!(
            fs::read(output.join("program.0.gas")).expect("read gas"),
            b"gas"
        );
        assert!(!output.join("README.md").exists());
        assert!(!output.join(".hidden").exists());
        assert!(!output.join("diagram.png").exists());
    }

    #[test]
    fn test_extract_zstd_tar_archive() {
        let dir = TempDir::new().expect("temp dir");
        let output = dir.path().join("out");
        fs::create_dir_all(&output).expect("create output");
        let tar = tar_bytes(&[("config.json", br#"{"ok":true}"#)]);
        let compressed = zstd::stream::encode_all(tar.as_slice(), 3).expect("compress tar");
        let blob = write_blob(&dir, &compressed);

        let files = ArchiveFormat::TarZstd
            .extract_blob(&blob, &output)
            .expect("extract archive");

        assert_eq!(files, vec!["config.json".to_string()]);
        assert_eq!(
            fs::read(output.join("config.json")).expect("read config"),
            br#"{"ok":true}"#
        );
    }

    #[test]
    fn test_extract_tar_archive_rejects_unsafe_paths() {
        let dir = TempDir::new().expect("temp dir");
        let output = dir.path().join("out");
        fs::create_dir_all(&output).expect("create output");
        let tar = unsafe_tar_bytes("../escape", b"bad");
        let blob = write_blob(&dir, &tar);

        let err = ArchiveFormat::Tar
            .extract_blob(&blob, &output)
            .expect_err("unsafe archive path should fail");

        assert!(err.to_string().contains(".."));
        assert!(!dir.path().join("escape").exists());
    }
}
