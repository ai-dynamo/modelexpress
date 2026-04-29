// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    GcsProvider,
    cache_manifest::{CacheManifest, MANIFEST_VERSION},
    downloader::Downloader,
    lock_file::LockFile,
    model_name::{BucketName, CACHE_ROOT_DIR_NAME, ModelName},
    path_ext::PathExt,
};
use crate::providers::ModelProviderTrait;
use anyhow::{Context, Result};
use std::{
    collections::HashSet,
    ffi::OsStr,
    fs, io,
    path::{Component, Path, PathBuf},
};
use tracing::warn;

pub const INTERNAL_METADATA_DIR_NAME: &str = ".mx";
const MANIFEST_FILE_NAME: &str = "manifest.json";
const MANIFEST_LOCK_FILE_NAME: &str = "manifest.lock";

#[derive(Debug, Clone, Copy)]
pub struct ModelDir<'a> {
    cache_dir: &'a Path,
    pub model_dir: &'a Path,
}

impl<'a> ModelDir<'a> {
    pub fn new(cache_dir: &'a Path, model_dir: &'a Path) -> Self {
        Self {
            cache_dir,
            model_dir,
        }
    }

    fn metadata_dir(&self) -> PathBuf {
        self.model_dir.join(INTERNAL_METADATA_DIR_NAME)
    }

    fn manifest_path(&self) -> PathBuf {
        self.metadata_dir().join(MANIFEST_FILE_NAME)
    }

    fn manifest_temp_path(&self) -> PathBuf {
        self.metadata_dir()
            .join(format!("{MANIFEST_FILE_NAME}.tmp"))
    }

    fn manifest_lock_path(&self) -> PathBuf {
        self.metadata_dir().join(MANIFEST_LOCK_FILE_NAME)
    }

    pub fn has_manifest_file(&self) -> bool {
        self.manifest_path().is_file()
    }

    pub fn write_manifest(&self, manifest: &CacheManifest) -> Result<()> {
        let manifest_path = self.manifest_path();
        let temp_path = self.manifest_temp_path();
        fs::create_dir_all(self.metadata_dir()).with_context(|| {
            format!(
                "Failed to create GCS metadata directory '{}'",
                self.metadata_dir().display()
            )
        })?;

        let mut payload = serde_json::to_vec_pretty(manifest).with_context(|| {
            format!(
                "Failed to serialize manifest for '{}'",
                self.model_dir.display()
            )
        })?;
        payload.push(b'\n');

        fs::write(&temp_path, payload)
            .with_context(|| format!("Failed to write manifest '{}'", temp_path.display()))?;
        if let Err(err) = fs::rename(&temp_path, &manifest_path).with_context(|| {
            format!(
                "Failed to move manifest '{}' into '{}'",
                temp_path.display(),
                manifest_path.display()
            )
        }) {
            let _ = fs::remove_file(&temp_path);
            return Err(err);
        }

        Ok(())
    }

    pub async fn ensure_manifest(
        &self,
        model: &ModelName,
        downloader: &Downloader<'_, impl google_cloud_storage::stub::Storage + 'static>,
    ) -> Result<CacheManifest> {
        if let Some(manifest) = self.load_manifest()? {
            return Ok(manifest);
        }

        LockFile::with_exclusive(&self.manifest_lock_path(), || async move {
            if let Some(manifest) = self.load_manifest()? {
                return Ok(manifest);
            }

            let manifest = CacheManifest::new(model, downloader.collect_manifest_entries().await?);
            self.write_manifest(&manifest)?;
            Ok(manifest)
        })
        .await
    }

    fn load_manifest(&self) -> Result<Option<CacheManifest>> {
        let manifest_path = self.manifest_path();
        if !manifest_path.is_file() {
            return Ok(None);
        }

        let contents = fs::read(&manifest_path)
            .with_context(|| format!("Failed to read manifest '{}'", manifest_path.display()));
        let manifest = contents.and_then(|contents| {
            serde_json::from_slice::<CacheManifest>(&contents)
                .with_context(|| format!("Failed to parse manifest '{}'", manifest_path.display()))
        });
        let Some(manifest) = (match manifest {
            Ok(manifest) => Some(manifest),
            Err(err) => {
                warn!(
                    "Ignoring unreadable GCS manifest '{}': {}",
                    manifest_path.display(),
                    err
                );
                return Ok(None);
            }
        }) else {
            return Ok(None);
        };

        let expected_model = self.model_name()?.to_string();
        let mut seen_paths = HashSet::new();
        let manifest_is_valid = manifest.version == MANIFEST_VERSION
            && manifest.model == expected_model
            && !manifest.files.is_empty()
            && manifest.files.iter().all(|entry| {
                let relative_path = Path::new(&entry.path);
                !entry.path.is_empty()
                    && relative_path.is_safe_relative()
                    && !Self::is_internal_artifact(relative_path)
                    && u32::from_str_radix(&entry.crc32c, 16).is_ok()
                    && seen_paths.insert(entry.path.clone())
            });

        if !manifest_is_valid {
            warn!(
                "Ignoring invalid GCS manifest '{}'",
                manifest_path.display(),
            );
            return Ok(None);
        }

        Ok(Some(manifest))
    }

    pub fn model_name(&self) -> Result<ModelName> {
        let gcs_root = self.cache_dir.join(CACHE_ROOT_DIR_NAME);
        let relative = self.model_dir.strip_prefix(&gcs_root).with_context(|| {
            format!(
                "GCS model directory '{}' is outside cache directory '{}'",
                self.model_dir.display(),
                gcs_root.display()
            )
        })?;
        let mut components = relative.components();
        let bucket_component = components.next().ok_or_else(|| {
            anyhow::anyhow!(
                "GCS model directory '{}' is missing bucket and object path",
                self.model_dir.display()
            )
        })?;
        let bucket_name = match bucket_component {
            Component::Normal(component) => component.to_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "GCS bucket component in '{}' is not valid UTF-8",
                    self.model_dir.display()
                )
            })?,
            _ => {
                anyhow::bail!(
                    "GCS model directory '{}' has invalid bucket component",
                    self.model_dir.display()
                )
            }
        };
        let bucket = BucketName::parse(bucket_name)?;

        let mut object_prefix_components = Vec::new();
        for component in components {
            let component = match component {
                Component::Normal(component) => component.to_str().ok_or_else(|| {
                    anyhow::anyhow!(
                        "GCS object path component in '{}' is not valid UTF-8",
                        self.model_dir.display()
                    )
                })?,
                _ => {
                    anyhow::bail!(
                        "GCS model directory '{}' has invalid object path component",
                        self.model_dir.display()
                    )
                }
            };
            object_prefix_components.push(component);
        }

        if object_prefix_components.is_empty() {
            anyhow::bail!(
                "GCS model directory '{}' is missing object path",
                self.model_dir.display()
            );
        }

        ModelName::new(bucket, &object_prefix_components.join("/"))
    }

    pub fn parse_relative_object_path(&self, relative_key: &str) -> Result<PathBuf> {
        if relative_key.split('/').any(str::is_empty) {
            anyhow::bail!(
                "Unsafe object path '{relative_key}' for model '{}': empty path segments are not allowed",
                self.model_name()?
            );
        }

        let path = Path::new(relative_key);
        if !path.is_safe_relative() {
            anyhow::bail!(
                "Unsafe object path '{relative_key}' for model '{}'",
                self.model_name()?
            );
        }
        if Self::is_internal_artifact(path) {
            anyhow::bail!(
                "GCS model '{}' contains reserved internal path '{}'",
                self.model_name()?,
                relative_key
            );
        }

        Ok(path.to_path_buf())
    }

    pub fn cache_satisfies_request(&self, ignore_weights: bool) -> Result<bool> {
        let Some(manifest) = self.load_manifest()? else {
            return Ok(false);
        };
        let mut matched_entry = false;
        for entry in &manifest.files {
            if ignore_weights && GcsProvider::is_weight_file(&entry.path) {
                continue;
            }

            matched_entry = true;
            let file_path = self.model_dir.join(&entry.path);
            let metadata = match fs::metadata(&file_path) {
                Ok(metadata) => metadata,
                Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "Failed to inspect manifest file '{}': {}",
                        file_path.display(),
                        err
                    ));
                }
            };
            if !metadata.is_file() {
                return Ok(false);
            }
        }
        Ok(matched_entry)
    }

    fn find_overlap(&self) -> Result<Option<(&'static str, ModelName)>> {
        let model = self.model_name()?;
        let bucket_dir = model.bucket_dir(self.cache_dir);
        let mut current = self.model_dir.parent();
        while let Some(ancestor) = current {
            if !ancestor.starts_with(&bucket_dir) {
                break;
            }

            let ancestor_model_dir = ModelDir::new(self.cache_dir, ancestor);
            if ancestor_model_dir.has_manifest_file() {
                return Ok(Some(("ancestor", ancestor_model_dir.model_name()?)));
            }

            if ancestor == bucket_dir {
                break;
            }
            current = ancestor.parent();
        }

        if !self.model_dir.is_dir() {
            return Ok(None);
        }

        let mut pending = vec![self.model_dir.to_path_buf()];
        while let Some(current_dir) = pending.pop() {
            for entry in fs::read_dir(&current_dir)
                .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let child_model_dir = ModelDir::new(self.cache_dir, &path);
                if child_model_dir.has_manifest_file() {
                    return Ok(Some(("descendant", child_model_dir.model_name()?)));
                }

                pending.push(path);
            }
        }

        Ok(None)
    }

    pub fn is_removable(&self) -> Result<bool> {
        if self.has_manifest_file() {
            self.model_name()?;
            return Ok(true);
        }

        if !self.model_dir.exists() {
            return Ok(false);
        }

        Ok(self.find_overlap()?.is_none())
    }

    pub fn remove(&self) -> Result<()> {
        if !self.is_removable()? {
            anyhow::bail!(
                "GCS model directory '{}' is not removable",
                self.model_dir.display()
            );
        }

        if self.model_dir.is_dir() {
            fs::remove_dir_all(self.model_dir).with_context(|| {
                format!(
                    "Failed to remove cached GCS model directory '{}'",
                    self.model_dir.display()
                )
            })?;
        } else {
            fs::remove_file(self.model_dir).with_context(|| {
                format!(
                    "Failed to remove cached GCS model file '{}'",
                    self.model_dir.display()
                )
            })?;
        }

        Ok(())
    }

    pub fn ensure_available(&self) -> Result<()> {
        let model = self.model_name()?;

        if let Some((kind, overlap)) = self.find_overlap()? {
            anyhow::bail!(
                "GCS model '{}' overlaps cached {} model '{}'",
                model,
                kind,
                overlap
            );
        }

        Ok(())
    }

    fn is_internal_artifact(relative: &Path) -> bool {
        matches!(
            relative.components().next(),
            Some(Component::Normal(component)) if component == OsStr::new(INTERNAL_METADATA_DIR_NAME)
        )
    }

    pub fn size(&self) -> Result<u64> {
        let mut size = 0u64;
        let mut pending = vec![self.model_dir.to_path_buf()];

        while let Some(current_dir) = pending.pop() {
            for entry in fs::read_dir(&current_dir)
                .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                let relative = path
                    .strip_prefix(self.model_dir)
                    .with_context(|| format!("Failed to make '{}' relative", path.display()))?;

                if Self::is_internal_artifact(relative) {
                    continue;
                }

                if path.is_file() {
                    size = size.saturating_add(fs::metadata(&path)?.len());
                } else if path.is_dir() {
                    pending.push(path);
                }
            }
        }

        Ok(size)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::{
        GcsProvider,
        cache_manifest::{CacheManifest, MANIFEST_VERSION},
        model_name::{BucketName, ModelName},
        path_ext::PathExt,
        test_support::{
            expected_internal_dir, expected_model_dir, manifest_entry, write_cached_model,
            write_manifest_with_payloads,
        },
    };
    use super::*;
    use crate::providers::ModelProviderTrait;
    use std::{
        fs,
        path::{Path, PathBuf},
    };
    use tempfile::TempDir;

    #[test]
    fn test_load_manifest_rejects_invalid_entries() {
        let model_name = "gs://test-bucket/org/model/rev-1";
        for (scenario, files) in [
            (
                "duplicate_paths",
                serde_json::json!([
                    {
                        "path": "tokenizer.json",
                        "size": 2,
                        "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                        "generation": 11
                    },
                    {
                        "path": "tokenizer.json",
                        "size": 7,
                        "crc32c": format!("{:08x}", crc32c::crc32c(b"weights")),
                        "generation": 12
                    }
                ]),
            ),
            (
                "reserved_internal_path",
                serde_json::json!([{
                    "path": ".mx/manifest.json",
                    "size": 2,
                    "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                    "generation": 13
                }]),
            ),
            (
                "invalid_crc",
                serde_json::json!([{
                    "path": "tokenizer.json",
                    "size": 2,
                    "crc32c": "not-hex",
                    "generation": 14
                }]),
            ),
            (
                "wrong_model_name",
                serde_json::json!([{
                    "path": "tokenizer.json",
                    "size": 2,
                    "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                    "generation": 15
                }]),
            ),
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            let internal_dir = expected_internal_dir(temp_dir.path(), model_name);
            fs::create_dir_all(&internal_dir).expect("Failed to create metadata dir");

            let manifest = serde_json::json!({
                "version": MANIFEST_VERSION,
                "model": if scenario == "wrong_model_name" {
                    "gs://other-bucket/org/model/rev-1"
                } else {
                    model_name
                },
                "files": files
            });
            fs::write(
                internal_dir.join(MANIFEST_FILE_NAME),
                serde_json::to_vec(&manifest).expect("Expected manifest serialization"),
            )
            .expect("Failed to write manifest");

            let model_dir = ModelDir::new(temp_dir.path(), &model_dir);
            assert!(
                model_dir
                    .load_manifest()
                    .expect("Expected manifest read to succeed")
                    .is_none(),
                "scenario={scenario}"
            );
        }
    }

    #[test]
    fn test_write_manifest_scenarios() {
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model = ModelName::parse(model_name).expect("Expected model parse");
        let cases = [
            (
                "create",
                None,
                CacheManifest::new(&model, vec![manifest_entry("tokenizer.json", b"{}")]),
            ),
            (
                "replace",
                Some(CacheManifest::new(
                    &model,
                    vec![manifest_entry("tokenizer.json", b"{}")],
                )),
                CacheManifest::new(&model, vec![manifest_entry("config.json", br#"{"a":1}"#)]),
            ),
            (
                "expand",
                Some(CacheManifest::new(
                    &model,
                    vec![manifest_entry("tokenizer.json", b"{}")],
                )),
                CacheManifest::new(
                    &model,
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
            ),
        ];

        for (name, initial_manifest, final_manifest) in cases {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            let model_dir_state = ModelDir::new(temp_dir.path(), &model_dir);

            if let Some(initial_manifest) = initial_manifest {
                model_dir_state
                    .write_manifest(&initial_manifest)
                    .expect("Expected initial manifest write");
            }

            model_dir_state
                .write_manifest(&final_manifest)
                .expect("Expected manifest write");

            let parsed: CacheManifest = serde_json::from_slice(
                &fs::read(model_dir_state.manifest_path()).expect("Expected manifest contents"),
            )
            .expect("Expected manifest JSON");
            assert_eq!(parsed, final_manifest, "scenario={name}");
            assert!(
                !model_dir_state.manifest_temp_path().exists(),
                "scenario={name}"
            );
        }
    }

    #[test]
    fn test_mx_path_layout_tables() {
        assert_eq!(GcsProvider.provider_name(), "GCS");

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir);

        for (path, expected) in [
            (
                model_dir.manifest_path(),
                expected_internal_dir(temp_dir.path(), model_name).join(MANIFEST_FILE_NAME),
            ),
            (
                model_dir.manifest_temp_path(),
                expected_internal_dir(temp_dir.path(), model_name).join("manifest.json.tmp"),
            ),
            (
                model_dir.manifest_lock_path(),
                expected_internal_dir(temp_dir.path(), model_name).join(MANIFEST_LOCK_FILE_NAME),
            ),
        ] {
            assert_eq!(path, expected);
        }

        for (path, expected_reserved) in [
            (".mx/manifest.json", true),
            (".mx/parts/weights/model.bin.part", true),
            ("weights/model.bin", false),
        ] {
            assert_eq!(
                ModelDir::is_internal_artifact(Path::new(path)),
                expected_reserved,
                "path={path}"
            );
        }

        let root = temp_dir.path().join("size-case");
        fs::create_dir_all(root.join("nested")).expect("Failed to create nested dirs");
        fs::create_dir_all(root.join(".mx/parts/nested")).expect("Failed to create parts dir");
        fs::create_dir_all(root.join(".mx/locks/nested")).expect("Failed to create locks dir");
        fs::write(root.join("nested/model.bin"), b"weights").expect("Failed to create model file");
        fs::write(root.join(".mx/parts/nested/model.bin.part"), b"partial")
            .expect("Failed to create temp file");
        fs::write(root.join(".mx/locks/nested/model.bin.lock"), b"lock")
            .expect("Failed to create lock file");
        assert_eq!(
            (ModelDir::new(&root, &root))
                .size()
                .expect("Expected model dir size"),
            7
        );
    }

    #[test]
    fn test_model_name_tables_cover_normalization_paths_and_cache_roundtrip() {
        let valid_cases = [
            (
                "gs://testbucket/dev/bake/qwen/rev123",
                "gs://testbucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/testbucket/dev/bake/qwen/rev123"),
            ),
            (
                "gs://testbucket/dev/bake/qwen/rev123/",
                "gs://testbucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/testbucket/dev/bake/qwen/rev123"),
            ),
            (
                "gs://sourcebucket/dev/bake/qwen/rev123",
                "gs://sourcebucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/sourcebucket/dev/bake/qwen/rev123"),
            ),
            (
                "gs://sourcebucket/dev/bake/qwen/rev123/",
                "gs://sourcebucket/dev/bake/qwen/rev123",
                PathBuf::from("/tmp/cache/gcs/sourcebucket/dev/bake/qwen/rev123"),
            ),
        ];

        for (input, canonical, expected_model_dir) in valid_cases {
            let model = ModelName::parse(input).expect("Expected model parsing");
            assert_eq!(model.to_string(), canonical, "input={input}");
            assert_eq!(
                model.model_dir(Path::new("/tmp/cache")),
                expected_model_dir,
                "input={input}"
            );
        }

        for invalid in [
            "",
            "gs://bucket-only",
            "gs://bucket-only/",
            "dev/bake/qwen/rev123",
        ] {
            assert!(ModelName::parse(invalid).is_err(), "input={invalid}");
        }

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        write_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");
        let model = ModelDir::new(temp_dir.path(), &model_dir)
            .model_name()
            .expect("Expected cached model name");
        assert_eq!(model.to_string(), model_name);
    }

    #[test]
    fn test_bucket_name_parse_table() {
        let valid_cases = [("gs://example-bucket/", "example-bucket")];
        for (input, expected) in valid_cases {
            assert_eq!(
                BucketName::parse(input)
                    .expect("Expected bucket normalization")
                    .to_string(),
                expected,
                "input={input}"
            );
        }

        for invalid in ["example-bucket/path", "..", ".", "gs://../"] {
            assert!(BucketName::parse(invalid).is_err(), "input={invalid}");
        }
    }

    #[test]
    fn test_relative_object_path_tables_cover_safety_downloadability_and_weight_matching() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = expected_model_dir(temp_dir.path(), "gs://test-bucket/org/model/rev-1");
        let model_dir = ModelDir::new(temp_dir.path(), &model_path);
        let relative_path_cases = [
            ("README.md", false, true, true, false),
            ("nested/README.md", false, true, true, false),
            (".gitattributes", false, true, true, false),
            ("image.png", false, true, true, false),
            ("nested/image.webp", false, true, true, false),
            ("config.json", true, true, true, false),
            ("tokenizer.json", true, true, true, false),
            ("foo.mlir", true, true, true, false),
            ("compile_summary.json", true, true, true, false),
            ("manifest.json", true, true, true, false),
            ("nested/preset.json", true, true, true, false),
            ("model.safetensors", true, false, true, true),
            ("weights/model.bin", true, false, true, true),
            ("weights/model.iop", true, false, true, true),
            ("weights/model.gas", true, false, true, true),
            ("weights/model.h5", true, false, true, true),
            ("weights/model.msgpack", true, false, true, true),
            ("weights/model.ckpt.index", true, false, true, true),
        ];

        for (
            relative_path,
            expected_downloadable,
            expected_metadata_match,
            expected_safe,
            expected_weight,
        ) in relative_path_cases
        {
            let relative = model_dir
                .parse_relative_object_path(relative_path)
                .expect("Expected valid relative object path");
            assert_eq!(
                relative.is_safe_relative(),
                expected_safe,
                "safe check for {relative_path}"
            );
            assert_eq!(
                GcsProvider::is_downloadable_path(&relative),
                expected_downloadable,
                "downloadable check for {relative_path}"
            );
            assert_eq!(
                GcsProvider::matches_request_path(&relative, true),
                expected_metadata_match,
                "metadata match for {relative_path}"
            );
            assert!(
                GcsProvider::matches_request_path(&relative, false),
                "full request match for {relative_path}"
            );
            assert_eq!(
                GcsProvider::is_weight_file(relative.to_string_lossy().as_ref()),
                expected_weight,
                "weight check for {relative_path}"
            );
        }

        for path in ["weights/model.safetensors", "tokenizer.json"] {
            assert!(
                Path::new(path).is_safe_relative(),
                "expected safe relative path: {path}"
            );
        }
        for path in ["../model.safetensors", "/etc/passwd"] {
            assert!(
                !Path::new(path).is_safe_relative(),
                "expected unsafe relative path: {path}"
            );
        }
    }

    #[test]
    fn test_relative_path_validation_tables() {
        fn candidate<'a>(object_name: &'a str, list_prefix: &str) -> Option<&'a str> {
            object_name
                .strip_prefix(list_prefix)
                .filter(|suffix| !suffix.is_empty() && !suffix.ends_with('/'))
        }

        assert!(candidate("other/model/file.bin", "org/model/rev123/").is_none());
        assert!(candidate("org/model/rev123/", "org/model/rev123/").is_none());
        assert!(candidate("org/model/rev123/subdir/", "org/model/rev123/").is_none());
        assert_eq!(
            candidate("org/model/rev123/tokenizer.json", "org/model/rev123/"),
            Some("tokenizer.json")
        );
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = expected_model_dir(temp_dir.path(), "gs://test-bucket/org/model/rev-1");
        let model_dir = ModelDir::new(temp_dir.path(), &model_path);
        for invalid in ["../secret.bin", "a//b.bin", "/a/b.bin", "a/b//c.bin"] {
            assert!(
                model_dir.parse_relative_object_path(invalid).is_err(),
                "input={invalid}"
            );
        }

        for (path, expected_valid) in [
            ("weights/model.bin", true),
            ("", false),
            ("/etc/passwd", false),
            ("../escape", false),
        ] {
            let path = Path::new(path);
            let is_valid_relative = !path.as_os_str().is_empty() && path.is_safe_relative();
            assert_eq!(is_valid_relative, expected_valid, "path={:?}", path);
        }
    }

    #[test]
    fn test_request_is_satisfied_scenarios() {
        for scenario in ["metadata_only_payload", "invalid_manifest"] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_name = match scenario {
                "metadata_only_payload" => "gs://bucket/foo/bar/baz",
                "invalid_manifest" => "gs://bucket/foo/bar/bad",
                _ => unreachable!("unexpected scenario"),
            };
            let model_dir = expected_model_dir(temp_dir.path(), model_name);

            match scenario {
                "metadata_only_payload" => write_manifest_with_payloads(
                    temp_dir.path(),
                    model_name,
                    &[("tokenizer.json", b"{}")],
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
                "invalid_manifest" => {
                    let internal_dir = expected_internal_dir(temp_dir.path(), model_name);
                    fs::create_dir_all(&internal_dir).expect("Failed to create metadata dir");
                    let invalid_manifest = serde_json::json!({
                        "version": MANIFEST_VERSION + 1,
                        "model": model_name,
                        "files": [{
                            "path": "tokenizer.json",
                            "size": 2,
                            "crc32c": format!("{:08x}", crc32c::crc32c(b"{}")),
                            "generation": 42
                        }]
                    });
                    fs::write(
                        internal_dir.join(MANIFEST_FILE_NAME),
                        serde_json::to_vec(&invalid_manifest)
                            .expect("Expected manifest serialization"),
                    )
                    .expect("Failed to write invalid manifest");
                    fs::create_dir_all(&model_dir).expect("Failed to create model dir");
                    fs::write(model_dir.join("tokenizer.json"), b"{}")
                        .expect("Failed to write payload");
                }
                _ => unreachable!("unexpected scenario"),
            }

            let model_dir = ModelDir::new(temp_dir.path(), &model_dir);
            match scenario {
                "metadata_only_payload" => {
                    assert!(
                        model_dir
                            .cache_satisfies_request(true)
                            .expect("Expected metadata request to be satisfied")
                    );
                    assert!(
                        !model_dir
                            .cache_satisfies_request(false)
                            .expect("Expected full request to remain incomplete")
                    );
                }
                "invalid_manifest" => {
                    assert!(
                        !model_dir
                            .cache_satisfies_request(false)
                            .expect("Expected invalid manifest to be ignored")
                    );
                }
                _ => unreachable!("unexpected scenario"),
            }
        }
    }

    #[test]
    fn test_ensure_model_dir_available_rejects_ancestor_descendant_overlap() {
        for (cached_model, requested_model) in [
            ("gs://bucket/foo/bar/baz", "gs://bucket/foo/bar"),
            ("gs://bucket/foo/.mx/bar", "gs://bucket/foo"),
        ] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            write_cached_model(temp_dir.path(), cached_model, "tokenizer.json", b"{}");

            let model = ModelName::parse(requested_model).expect("Expected model parse");
            let model_dir = model.model_dir(temp_dir.path());

            let err = (ModelDir::new(temp_dir.path(), &model_dir))
                .ensure_available()
                .expect_err("Expected overlap rejection");

            assert!(
                err.to_string().contains("overlaps cached descendant model"),
                "cached_model={cached_model} requested_model={requested_model} err={err}"
            );
        }
    }
}
