// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    GcsProvider,
    cache_manifest::{CacheManifest, ManifestEntry},
    download_progress::DownloadProgress,
    download_task::DownloadTask,
    lock_file::LockFile,
    model_dir::{INTERNAL_METADATA_DIR_NAME, ModelDir},
    model_name::{BucketName, ModelName},
};
use anyhow::{Context, Result};
use futures::StreamExt;
use futures::stream;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model::{ListObjectsRequest, Object};
use google_cloud_storage::model_ext::ReadRange;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tracing::warn;

const MAX_GCS_LIST_RESULTS: i32 = 1000;
const MAX_PARALLEL_DOWNLOADS: usize = 8;

pub struct Downloader<'a, S = google_cloud_storage::stub::DefaultStorage>
where
    S: google_cloud_storage::stub::Storage + 'static,
{
    storage: &'a Storage<S>,
    control: &'a StorageControl,
    bucket: &'a BucketName,
    model: &'a ModelName,
    model_dir: ModelDir<'a>,
    bucket_resource_name: String,
    list_prefix: String,
}

impl<'a, S> Downloader<'a, S>
where
    S: google_cloud_storage::stub::Storage + 'static,
{
    pub fn new(
        storage: &'a Storage<S>,
        control: &'a StorageControl,
        model: &'a ModelName,
        model_dir: ModelDir<'a>,
    ) -> Self {
        Self {
            storage,
            control,
            bucket: &model.bucket,
            model,
            model_dir,
            bucket_resource_name: model.bucket.resource_name(),
            list_prefix: format!("{}/", model.object_prefix),
        }
    }

    fn build_list_request(&self, page_token: Option<String>) -> ListObjectsRequest {
        let mut request = ListObjectsRequest::new()
            .set_parent(self.bucket_resource_name.clone())
            .set_prefix(self.list_prefix.clone())
            .set_page_size(MAX_GCS_LIST_RESULTS);

        if let Some(page_token) = page_token {
            request = request.set_page_token(page_token);
        }

        request
    }

    fn ensure_parent_directory(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
        }
        Ok(())
    }

    fn candidate_relative_key<'b>(&self, object: &'b Object) -> Option<&'b str> {
        object
            .name
            .strip_prefix(&self.list_prefix)
            .filter(|suffix| !suffix.is_empty() && !suffix.ends_with('/'))
    }

    fn build_download_task(
        &self,
        relative_path: PathBuf,
        expected_size: u64,
        generation: Option<u64>,
        expected_crc32c: u32,
    ) -> Result<DownloadTask> {
        let destination_path = self.model_dir.model_dir.join(&relative_path);

        Ok(DownloadTask {
            object_name: format!("{}/{}", self.model.object_prefix, relative_path.display()),
            relative_path,
            destination_path,
            internal_dir: self.model_dir.model_dir.join(INTERNAL_METADATA_DIR_NAME),
            expected_size: Some(expected_size),
            generation: generation.and_then(|value| i64::try_from(value).ok()),
            expected_crc32c: Some(expected_crc32c),
        })
    }

    fn manifest_entry_if_downloadable(&self, object: &Object) -> Result<Option<ManifestEntry>> {
        let Some(relative_key) = self.candidate_relative_key(object) else {
            return Ok(None);
        };
        let relative_path = self.model_dir.parse_relative_object_path(relative_key)?;
        if !GcsProvider::is_downloadable_path(&relative_path) {
            return Ok(None);
        }

        let expected_size = u64::try_from(object.size).with_context(|| {
            format!(
                "Object '{}' for model '{}' has an invalid size '{}'",
                object.name, self.model, object.size
            )
        })?;
        let expected_crc32c = object
            .checksums
            .as_ref()
            .and_then(|checksums| checksums.crc32c)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Cannot record manifest entry for '{}': remote CRC32C is unavailable",
                    object.name
                )
            })?;

        Ok(Some(ManifestEntry {
            path: relative_path.to_string_lossy().into_owned(),
            size: expected_size,
            crc32c: format!("{expected_crc32c:08x}"),
            generation: (object.generation > 0)
                .then_some(object.generation)
                .and_then(|generation| u64::try_from(generation).ok()),
        }))
    }

    pub async fn collect_manifest_entries(&self) -> Result<Vec<ManifestEntry>> {
        let mut page_token: Option<String> = None;
        let mut entries = Vec::new();

        loop {
            let response = self
                .control
                .list_objects()
                .with_request(self.build_list_request(page_token.clone()))
                .send()
                .await
                .with_context(|| {
                    format!(
                        "Failed to list objects in gs://{}/{}",
                        self.bucket, self.model.object_prefix
                    )
                })?;

            for object in response.objects {
                if let Some(entry) = self.manifest_entry_if_downloadable(&object)? {
                    entries.push(entry);
                }
            }

            if response.next_page_token.is_empty() {
                break;
            }
            page_token = Some(response.next_page_token);
        }

        Ok(entries)
    }

    pub fn download_tasks_from_manifest(
        &self,
        manifest: &CacheManifest,
        ignore_weights: bool,
    ) -> Result<Vec<DownloadTask>> {
        let mut tasks = Vec::new();
        for entry in &manifest.files {
            let relative_path = self.model_dir.parse_relative_object_path(&entry.path)?;
            if !GcsProvider::matches_request_path(&relative_path, ignore_weights) {
                continue;
            }

            let expected_crc32c = u32::from_str_radix(&entry.crc32c, 16).with_context(|| {
                format!(
                    "Manifest for '{}' contains invalid CRC32C '{}'",
                    self.model, entry.crc32c
                )
            })?;

            tasks.push(self.build_download_task(
                relative_path,
                entry.size,
                entry.generation,
                expected_crc32c,
            )?);
        }

        Ok(tasks)
    }

    pub async fn download_tasks_in_parallel(&self, tasks: &[DownloadTask]) -> Result<()> {
        let task_count = tasks.len();
        if task_count == 0 {
            return Ok(());
        }
        let progress = Arc::new(DownloadProgress::new(tasks));
        progress.log_start(MAX_PARALLEL_DOWNLOADS);

        let download_stream = stream::iter(tasks.iter().cloned().map(|task| {
            let progress = Arc::clone(&progress);
            async move {
                self.download_task_once(&task, progress.as_ref()).await?;
                progress.mark_file_completed(&task.object_name);
                Ok::<(), anyhow::Error>(())
            }
        }))
        .buffer_unordered(MAX_PARALLEL_DOWNLOADS);

        tokio::pin!(download_stream);

        while let Some(result) = download_stream.next().await {
            result?;
        }
        progress.log_finish();

        Ok(())
    }

    async fn download_task_once(
        &self,
        task: &DownloadTask,
        progress: &DownloadProgress,
    ) -> Result<()> {
        Self::ensure_parent_directory(&task.destination_path)?;
        let temp_path = task.temp_path();
        Self::ensure_parent_directory(&temp_path)?;

        LockFile::with_exclusive(&task.lock_path(), || async move {
            let Some(resume_offset) = self.prepare_task_download(task).await? else {
                return Ok(());
            };
            self.stream_task_to_temp(task, resume_offset, progress)
                .await?;
            self.finalize_task_download(task).await
        })
        .await
    }

    async fn prepare_task_download(&self, task: &DownloadTask) -> Result<Option<u64>> {
        let temp_path = task.temp_path();
        if task.has_cached_destination_file().await? {
            task.remove_temp_file_if_present(&temp_path).await?;
            return Ok(None);
        }

        let mut resume_offset = match tokio::fs::metadata(&temp_path).await {
            Ok(metadata) => metadata.len(),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0,
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Failed to inspect '{}': {}",
                    temp_path.display(),
                    err
                ));
            }
        };
        if let Some(expected_size) = task.expected_size {
            if resume_offset > expected_size {
                tokio::fs::remove_file(&temp_path).await.with_context(|| {
                    format!(
                        "Failed to clear oversized temp file '{}'",
                        temp_path.display()
                    )
                })?;
                resume_offset = 0;
            } else if resume_offset == expected_size && resume_offset > 0 {
                match task.promote_verified_temp_file(&temp_path).await {
                    Ok(()) => return Ok(None),
                    Err(err) => {
                        warn!("{err}");
                        resume_offset = 0;
                    }
                }
            }
        }

        if resume_offset > 0 && task.generation.is_none() {
            warn!(
                "Discarding partial temp file '{}' for '{}' because remote generation is unavailable",
                temp_path.display(),
                task.object_name
            );
            tokio::fs::remove_file(&temp_path).await.with_context(|| {
                format!(
                    "Failed to discard temp file '{}' without a known remote generation",
                    temp_path.display()
                )
            })?;
            resume_offset = 0;
        }

        Ok(Some(resume_offset))
    }

    async fn stream_task_to_temp(
        &self,
        task: &DownloadTask,
        resume_offset: u64,
        progress: &DownloadProgress,
    ) -> Result<()> {
        let temp_path = task.temp_path();
        let bucket = self.bucket;
        let read_range = if resume_offset == 0 {
            ReadRange::all()
        } else {
            ReadRange::offset(resume_offset)
        };

        let mut request = self
            .storage
            .read_object(self.bucket_resource_name.clone(), task.object_name.clone())
            .set_read_range(read_range);
        if let Some(generation) = task.generation {
            request = request.set_generation(generation);
        }

        let mut stream = request.send().await.with_context(|| {
            format!(
                "Failed to start download for gs://{bucket}/{}",
                task.object_name
            )
        })?;

        let mut file = if resume_offset == 0 {
            tokio::fs::File::create(&temp_path)
                .await
                .with_context(|| format!("Failed to create '{}'", temp_path.display()))?
        } else {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&temp_path)
                .await
                .with_context(|| {
                    format!(
                        "Failed to open '{}' for resume at byte {}",
                        temp_path.display(),
                        resume_offset
                    )
                })?
        };

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| {
                format!("Failed while streaming gs://{bucket}/{}", task.object_name)
            })?;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write '{}'", temp_path.display()))?;
            progress.record_downloaded_bytes(chunk.len() as u64);
        }

        file.flush()
            .await
            .with_context(|| format!("Failed to flush '{}'", temp_path.display()))?;
        drop(file);

        Ok(())
    }

    async fn finalize_task_download(&self, task: &DownloadTask) -> Result<()> {
        let temp_path = task.temp_path();
        if let Some(expected_size) = task.expected_size {
            let final_size = tokio::fs::metadata(&temp_path)
                .await
                .with_context(|| {
                    format!(
                        "Failed to inspect completed temp file '{}'",
                        temp_path.display()
                    )
                })?
                .len();
            if final_size != expected_size {
                anyhow::bail!(
                    "Incomplete download for '{}': expected {} bytes, got {} bytes",
                    task.object_name,
                    expected_size,
                    final_size
                );
            }
        }

        task.promote_verified_temp_file(&temp_path).await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::{
        cache_manifest::{CacheManifest, ManifestEntry},
        model_dir::ModelDir,
        model_name::ModelName,
        test_support::{
            TestStorageControlStub, TestStorageStub, gcs_object, manifest_entry_with_generation,
        },
    };
    use super::*;
    use google_cloud_storage::{
        client::{Storage, StorageControl},
        model::ListObjectsResponse,
    };
    use std::{fs, path::PathBuf};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_downloader_collect_manifest_entries_paginates_and_filters_downloadables() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let object_prefix = model.object_prefix.clone();

        let storage_stub = TestStorageStub::new(Vec::<(String, String)>::new());
        let storage = Storage::from_stub(storage_stub);
        let control_stub = TestStorageControlStub::new([
            ListObjectsResponse::new()
                .set_objects([
                    gcs_object(&format!("{object_prefix}/tokenizer.json"), b"{}", 11),
                    gcs_object(&format!("{object_prefix}/README.md"), b"readme", 12),
                    gcs_object(&format!("{object_prefix}/image.png"), b"png", 13),
                    gcs_object(&format!("{object_prefix}/"), b"", 14),
                    gcs_object("other/prefix/file.bin", b"other", 15),
                ])
                .set_next_page_token("page-2"),
            ListObjectsResponse::new().set_objects([gcs_object(
                &format!("{object_prefix}/weights/model.bin"),
                b"weights",
                16,
            )]),
        ]);
        let control = StorageControl::from_stub(control_stub.clone());
        let downloader = Downloader::new(&storage, &control, &model, model_dir);

        let mut entries = downloader
            .collect_manifest_entries()
            .await
            .expect("Expected manifest entry collection");
        entries.sort_by(|left, right| left.path.cmp(&right.path));

        assert_eq!(
            entries,
            vec![
                manifest_entry_with_generation("tokenizer.json", b"{}", 11),
                manifest_entry_with_generation("weights/model.bin", b"weights", 16),
            ]
        );

        let requests = control_stub.list_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].parent, model.bucket.resource_name());
        assert_eq!(requests[0].prefix, format!("{object_prefix}/"));
        assert_eq!(requests[0].page_size, MAX_GCS_LIST_RESULTS);
        assert!(requests[0].page_token.is_empty());
        assert_eq!(requests[1].page_token, "page-2");
    }

    #[test]
    fn test_download_tasks_from_manifest_filters_weights_and_validates_crc() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let storage = Storage::from_stub(TestStorageStub::new(Vec::<(String, String)>::new()));
        let control = StorageControl::from_stub(TestStorageControlStub::new(Vec::<
            ListObjectsResponse,
        >::new()));
        let downloader = Downloader::new(&storage, &control, &model, model_dir);

        let manifest = CacheManifest::new(
            &model,
            vec![
                manifest_entry_with_generation("tokenizer.json", b"{}", 21),
                manifest_entry_with_generation("weights/model.bin", b"weights", 22),
            ],
        );

        let metadata_only = downloader
            .download_tasks_from_manifest(&manifest, true)
            .expect("Expected task generation");
        assert_eq!(metadata_only.len(), 1);
        assert_eq!(
            metadata_only[0].relative_path,
            PathBuf::from("tokenizer.json")
        );
        assert_eq!(
            metadata_only[0].destination_path,
            model_dir_path.join("tokenizer.json")
        );
        assert_eq!(metadata_only[0].generation, Some(21));

        let full_request = downloader
            .download_tasks_from_manifest(&manifest, false)
            .expect("Expected full task generation");
        assert_eq!(full_request.len(), 2);
        assert_eq!(
            full_request[1].relative_path,
            PathBuf::from("weights/model.bin")
        );
        assert_eq!(
            full_request[1].object_name,
            format!("{}/weights/model.bin", model.object_prefix)
        );

        let invalid_crc_manifest = CacheManifest::new(
            &model,
            vec![ManifestEntry {
                path: "tokenizer.json".to_string(),
                size: 2,
                crc32c: "not-hex".to_string(),
                generation: Some(23),
            }],
        );
        assert!(
            downloader
                .download_tasks_from_manifest(&invalid_crc_manifest, false)
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_download_tasks_in_parallel_resumes_partial_downloads_and_skips_cached_files() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let tokenizer_object = format!("{}/tokenizer.json", model.object_prefix);
        let weights_object = format!("{}/weights/model.bin", model.object_prefix);

        let storage_stub = TestStorageStub::new([(weights_object.clone(), "weights".to_string())]);
        let storage = Storage::from_stub(storage_stub.clone());
        let control = StorageControl::from_stub(TestStorageControlStub::new(Vec::<
            ListObjectsResponse,
        >::new()));
        let downloader = Downloader::new(&storage, &control, &model, model_dir);
        let manifest = CacheManifest::new(
            &model,
            vec![
                manifest_entry_with_generation("tokenizer.json", b"{}", 31),
                manifest_entry_with_generation("weights/model.bin", b"weights", 32),
            ],
        );
        let tasks = downloader
            .download_tasks_from_manifest(&manifest, false)
            .expect("Expected download tasks");

        let cached_task = tasks
            .iter()
            .find(|task| task.object_name == tokenizer_object)
            .expect("Expected cached tokenizer task");
        if let Some(parent) = cached_task.destination_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create cached destination dir");
        }
        fs::write(&cached_task.destination_path, b"{}")
            .expect("Failed to write cached destination");
        if let Some(parent) = cached_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create cached temp dir");
        }
        fs::write(cached_task.temp_path(), b"stale").expect("Failed to write stale temp");

        let resumed_task = tasks
            .iter()
            .find(|task| task.object_name == weights_object)
            .expect("Expected resumable weights task");
        if let Some(parent) = resumed_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create resume temp dir");
        }
        fs::write(resumed_task.temp_path(), b"we").expect("Failed to write partial temp");

        downloader
            .download_tasks_in_parallel(&tasks)
            .await
            .expect("Expected parallel download");

        assert_eq!(
            fs::read(&cached_task.destination_path).expect("Expected cached destination payload"),
            b"{}"
        );
        assert!(
            !cached_task.temp_path().exists(),
            "Expected stale temp file to be removed"
        );
        assert_eq!(
            fs::read(&resumed_task.destination_path).expect("Expected resumed destination payload"),
            b"weights"
        );
        assert!(
            !resumed_task.temp_path().exists(),
            "Expected resumed temp file to be promoted"
        );

        let requests = storage_stub.read_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].bucket, model.bucket.resource_name());
        assert_eq!(requests[0].object, weights_object);
        assert_eq!(requests[0].read_offset, 2);
        assert_eq!(requests[0].generation, 32);
    }

    #[tokio::test]
    async fn test_prepare_and_finalize_task_download_edge_cases() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model =
            ModelName::parse("gs://test-bucket/org/model/rev123").expect("Expected model parsing");
        let model_dir_path = model.model_dir(temp_dir.path());
        let model_dir = ModelDir::new(temp_dir.path(), &model_dir_path);
        let storage = Storage::from_stub(TestStorageStub::new(Vec::<(String, String)>::new()));
        let control = StorageControl::from_stub(TestStorageControlStub::new(Vec::<
            ListObjectsResponse,
        >::new()));
        let downloader = Downloader::new(&storage, &control, &model, model_dir);

        let mismatch_task = downloader
            .build_download_task(
                PathBuf::from("weights/checksum.bin"),
                7,
                Some(41),
                crc32c::crc32c(b"different"),
            )
            .expect("Expected mismatch task");
        if let Some(parent) = mismatch_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create mismatch temp dir");
        }
        fs::write(mismatch_task.temp_path(), b"weights").expect("Failed to write mismatch temp");
        assert_eq!(
            downloader
                .prepare_task_download(&mismatch_task)
                .await
                .expect("Expected mismatch temp handling"),
            Some(0)
        );
        assert!(
            !mismatch_task.temp_path().exists(),
            "Expected mismatched temp file to be discarded"
        );

        let oversized_task = downloader
            .build_download_task(
                PathBuf::from("weights/oversized.bin"),
                3,
                Some(42),
                crc32c::crc32c(b"abc"),
            )
            .expect("Expected oversized task");
        if let Some(parent) = oversized_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create oversized temp dir");
        }
        fs::write(oversized_task.temp_path(), b"toolong").expect("Failed to write oversized temp");
        assert_eq!(
            downloader
                .prepare_task_download(&oversized_task)
                .await
                .expect("Expected oversized temp handling"),
            Some(0)
        );
        assert!(
            !oversized_task.temp_path().exists(),
            "Expected oversized temp file to be discarded"
        );

        let generationless_task = DownloadTask {
            generation: None,
            ..downloader
                .build_download_task(
                    PathBuf::from("weights/generationless.bin"),
                    7,
                    Some(43),
                    crc32c::crc32c(b"weights"),
                )
                .expect("Expected generationless task")
        };
        if let Some(parent) = generationless_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create generationless temp dir");
        }
        fs::write(generationless_task.temp_path(), b"we")
            .expect("Failed to write generationless temp");
        assert_eq!(
            downloader
                .prepare_task_download(&generationless_task)
                .await
                .expect("Expected generationless temp handling"),
            Some(0)
        );
        assert!(
            !generationless_task.temp_path().exists(),
            "Expected generationless temp file to be discarded"
        );

        let incomplete_task = downloader
            .build_download_task(
                PathBuf::from("weights/incomplete.bin"),
                7,
                Some(44),
                crc32c::crc32c(b"weights"),
            )
            .expect("Expected incomplete task");
        if let Some(parent) = incomplete_task.temp_path().parent() {
            fs::create_dir_all(parent).expect("Failed to create incomplete temp dir");
        }
        fs::write(incomplete_task.temp_path(), b"w").expect("Failed to write incomplete temp");
        let err = downloader
            .finalize_task_download(&incomplete_task)
            .await
            .expect_err("Expected finalize to reject incomplete temp");
        assert!(err.to_string().contains("Incomplete download"));
    }
}
