// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::providers::ModelProviderTrait;
use anyhow::{Context, Result};
use google_cloud_storage::client::{Storage, StorageControl};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

mod cache_manifest;
mod download_progress;
mod download_task;
mod downloader;
mod lock_file;
mod model_dir;
mod model_name;
mod path_ext;
mod provider_cache;

use downloader::Downloader;
use model_dir::ModelDir;
use model_name::ModelName;

pub use provider_cache::GcsProviderCache;

fn ensure_crypto_provider() -> Result<()> {
    if rustls::crypto::CryptoProvider::get_default().is_some() {
        return Ok(());
    }

    match rustls::crypto::ring::default_provider().install_default() {
        Ok(()) => Ok(()),
        Err(_) if rustls::crypto::CryptoProvider::get_default().is_some() => Ok(()),
        Err(_) => anyhow::bail!("Failed to install rustls ring CryptoProvider for GCS"),
    }
}

pub struct GcsProvider;

impl GcsProvider {
    fn is_downloadable_path(path: &Path) -> bool {
        !Self::is_ignored(path.to_string_lossy().as_ref()) && !Self::is_image(path)
    }

    fn matches_request_path(path: &Path, ignore_weights: bool) -> bool {
        !ignore_weights || !Self::is_weight_file(path.to_string_lossy().as_ref())
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for GcsProvider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        let cache_dir = cache_dir
            .ok_or_else(|| anyhow::anyhow!("GCS download requires cache_dir to be provided"))?;
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory: {}", cache_dir.display())
        })?;

        let model = ModelName::parse(model_name)?;

        let model_dir = model.model_dir(&cache_dir);
        let current_model_dir = ModelDir::new(&cache_dir, &model_dir);
        if current_model_dir.cache_satisfies_request(ignore_weights)? {
            info!(
                "Using cached GCS model '{}' from '{}'",
                model,
                model_dir.display()
            );
            return Ok(model_dir);
        }

        (ModelDir::new(&cache_dir, &model_dir)).ensure_available()?;

        ensure_crypto_provider()?;
        let storage = Storage::builder()
            .build()
            .await
            .context("Failed to initialize Google Cloud Storage data client")?;
        let control = StorageControl::builder()
            .build()
            .await
            .context("Failed to initialize Google Cloud Storage control client")?;
        let downloader = Downloader::new(&storage, &control, &model, current_model_dir);
        let manifest = current_model_dir
            .ensure_manifest(&model, &downloader)
            .await?;

        if current_model_dir.cache_satisfies_request(ignore_weights)? {
            info!(
                "Using cached GCS model '{}' from '{}'",
                model,
                model_dir.display()
            );
            return Ok(model_dir);
        }

        let download_tasks = downloader.download_tasks_from_manifest(&manifest, ignore_weights)?;

        if download_tasks.is_empty() {
            anyhow::bail!("No downloadable files found in {}", model);
        }

        downloader
            .download_tasks_in_parallel(&download_tasks)
            .await?;

        if !current_model_dir.cache_satisfies_request(ignore_weights)? {
            anyhow::bail!(
                "Downloaded GCS model '{}' is still incomplete for the requested file set",
                model
            );
        }

        info!(
            "Downloaded {} files for model '{}'",
            download_tasks.len(),
            model
        );

        Ok(model_dir)
    }

    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        let model_dir_state = ModelDir::new(&cache_dir, &model_dir);

        if !model_dir_state.is_removable()? {
            info!(
                "GCS model '{}' not found in cache, skipping delete",
                model_name
            );
            return Ok(());
        }

        model_dir_state.remove()?;
        info!(
            "Deleted cached GCS model '{}' from '{}'",
            model_name,
            model_dir.display()
        );
        Ok(())
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        let model_dir_state = ModelDir::new(&cache_dir, &model_dir);
        if !model_dir_state.cache_satisfies_request(false)? {
            anyhow::bail!("GCS model '{model_name}' not found in cache");
        }

        Ok(model_dir)
    }

    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        Ok(ModelName::parse(model_name)?.to_string())
    }

    fn provider_name(&self) -> &'static str {
        "GCS"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod test_support {
    use super::{
        cache_manifest::{CacheManifest, ManifestEntry},
        model_dir::{INTERNAL_METADATA_DIR_NAME, ModelDir},
        model_name::ModelName,
    };
    use google_cloud_gax::{options::RequestOptions as ControlRequestOptions, response::Response};
    use google_cloud_storage::{
        model::{
            ListObjectsRequest, ListObjectsResponse, Object, ObjectChecksums, ReadObjectRequest,
        },
        model_ext::ObjectHighlights,
        read_object::ReadObjectResponse,
        request_options::RequestOptions as StorageRequestOptions,
    };
    use std::{
        collections::{HashMap, VecDeque},
        fs,
        future::Future,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
    };

    pub fn expected_model_dir(cache_dir: &Path, model_name: &str) -> PathBuf {
        ModelName::parse(model_name)
            .expect("Expected model parsing")
            .model_dir(cache_dir)
    }

    pub fn expected_internal_dir(cache_dir: &Path, model_name: &str) -> PathBuf {
        expected_model_dir(cache_dir, model_name).join(INTERNAL_METADATA_DIR_NAME)
    }

    pub fn manifest_entry_with_generation(
        relative_path: &str,
        contents: &[u8],
        generation: u64,
    ) -> ManifestEntry {
        ManifestEntry {
            path: relative_path.to_string(),
            size: contents.len() as u64,
            crc32c: format!("{:08x}", crc32c::crc32c(contents)),
            generation: Some(generation),
        }
    }

    pub fn manifest_entry(relative_path: &str, contents: &[u8]) -> ManifestEntry {
        manifest_entry_with_generation(relative_path, contents, 42)
    }

    pub fn gcs_object(object_name: &str, contents: &[u8], generation: i64) -> Object {
        Object::new()
            .set_name(object_name)
            .set_size(contents.len() as i64)
            .set_generation(generation)
            .set_checksums(ObjectChecksums::new().set_crc32c(crc32c::crc32c(contents)))
    }

    #[derive(Debug, Clone)]
    pub struct TestStorageStub {
        objects: Arc<HashMap<String, String>>,
        requests: Arc<Mutex<Vec<ReadObjectRequest>>>,
    }

    impl TestStorageStub {
        pub fn new<I, K, V>(objects: I) -> Self
        where
            I: IntoIterator<Item = (K, V)>,
            K: Into<String>,
            V: Into<String>,
        {
            Self {
                objects: Arc::new(
                    objects
                        .into_iter()
                        .map(|(name, contents)| (name.into(), contents.into()))
                        .collect(),
                ),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn read_requests(&self) -> Vec<ReadObjectRequest> {
            self.requests
                .lock()
                .expect("Failed to lock read requests")
                .clone()
        }
    }

    impl google_cloud_storage::stub::Storage for TestStorageStub {
        fn read_object(
            &self,
            req: ReadObjectRequest,
            _options: StorageRequestOptions,
        ) -> impl Future<Output = google_cloud_storage::Result<ReadObjectResponse>> + Send {
            let objects = Arc::clone(&self.objects);
            let requests = Arc::clone(&self.requests);
            async move {
                requests
                    .lock()
                    .expect("Failed to lock read requests")
                    .push(req.clone());

                let payload = objects
                    .get(&req.object)
                    .expect("Expected test object payload")
                    .clone();
                let start = usize::try_from(req.read_offset.max(0))
                    .expect("Expected non-negative read offset");
                let resumed = payload
                    .get(start..)
                    .expect("Expected resume offset to be within payload")
                    .to_string();

                Ok(ReadObjectResponse::from_source(
                    ObjectHighlights::default(),
                    resumed,
                ))
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct TestStorageControlStub {
        responses: Arc<Mutex<VecDeque<Response<ListObjectsResponse>>>>,
        requests: Arc<Mutex<Vec<ListObjectsRequest>>>,
    }

    impl TestStorageControlStub {
        pub fn new<I>(responses: I) -> Self
        where
            I: IntoIterator<Item = ListObjectsResponse>,
        {
            Self {
                responses: Arc::new(Mutex::new(
                    responses.into_iter().map(Response::from).collect(),
                )),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn list_requests(&self) -> Vec<ListObjectsRequest> {
            self.requests
                .lock()
                .expect("Failed to lock list requests")
                .clone()
        }
    }

    impl google_cloud_storage::stub::StorageControl for TestStorageControlStub {
        fn list_objects(
            &self,
            req: ListObjectsRequest,
            _options: ControlRequestOptions,
        ) -> impl Future<Output = google_cloud_storage::Result<Response<ListObjectsResponse>>> + Send
        {
            let responses = Arc::clone(&self.responses);
            let requests = Arc::clone(&self.requests);
            async move {
                requests
                    .lock()
                    .expect("Failed to lock list requests")
                    .push(req);
                Ok(responses
                    .lock()
                    .expect("Failed to lock list responses")
                    .pop_front()
                    .expect("Expected test list_objects response"))
            }
        }
    }

    fn write_payload(cache_dir: &Path, model_name: &str, relative_path: &str, contents: &[u8]) {
        let model_dir = expected_model_dir(cache_dir, model_name);
        let payload_path = model_dir.join(relative_path);
        if let Some(parent) = payload_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create model payload directory");
        }
        fs::write(&payload_path, contents).expect("Failed to write model payload");
    }

    fn write_manifest(cache_dir: &Path, model_name: &str, entries: Vec<ManifestEntry>) {
        let model = ModelName::parse(model_name).expect("Expected model name parse");
        let model_dir = expected_model_dir(cache_dir, model_name);
        (ModelDir::new(cache_dir, &model_dir))
            .write_manifest(&CacheManifest::new(&model, entries))
            .expect("Failed to write manifest");
    }

    pub fn write_cached_model(
        cache_dir: &Path,
        model_name: &str,
        relative_path: &str,
        contents: &[u8],
    ) {
        write_payload(cache_dir, model_name, relative_path, contents);
        write_manifest(
            cache_dir,
            model_name,
            vec![manifest_entry(relative_path, contents)],
        );
    }

    pub fn write_manifest_with_payloads(
        cache_dir: &Path,
        model_name: &str,
        payloads: &[(&str, &[u8])],
        manifest_entries: Vec<ManifestEntry>,
    ) {
        for (relative_path, contents) in payloads {
            write_payload(cache_dir, model_name, relative_path, contents);
        }
        write_manifest(cache_dir, model_name, manifest_entries);
    }

    pub fn write_incomplete_cached_model(
        cache_dir: &Path,
        model_name: &str,
        relative_path: &str,
        contents: &[u8],
    ) {
        let model_dir = expected_model_dir(cache_dir, model_name);
        let payload_path = model_dir.join(relative_path);
        if let Some(parent) = payload_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create model payload directory");
        }
        fs::write(&payload_path, contents).expect("Failed to write model payload");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::test_support::{
        expected_model_dir, manifest_entry, write_cached_model, write_incomplete_cached_model,
        write_manifest_with_payloads,
    };
    use super::*;
    use crate::providers::ModelProviderTrait;
    use tempfile::TempDir;

    #[test]
    fn test_download_model_validation_cases() {
        let provider = GcsProvider;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime");

        for (name, model_name, cache_dir, expected_error) in [
            (
                "missing_cache_dir",
                "gs://test-bucket/test/model/rev-1",
                None,
                "requires cache_dir",
            ),
            (
                "relative_path",
                "org/model/rev-2",
                Some(TempDir::new().expect("Failed to create temp dir")),
                "full gs://<bucket>/<path> URL",
            ),
        ] {
            let result = runtime.block_on(provider.download_model(
                model_name,
                cache_dir.as_ref().map(|dir| dir.path().to_path_buf()),
                false,
            ));
            assert!(
                result
                    .expect_err("Expected validation failure")
                    .to_string()
                    .contains(expected_error),
                "scenario={name}"
            );
        }
    }

    #[test]
    fn test_download_model_cached_cases() {
        let provider = GcsProvider;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime");
        let cases = [
            (
                "full",
                "gs://test-bucket/org/model/rev-1",
                false,
                vec![("weights/model.bin", b"weights".as_slice())],
                vec![manifest_entry("weights/model.bin", b"weights")],
            ),
            (
                "metadata_subset",
                "gs://test-bucket/org/model/rev-meta",
                true,
                vec![("tokenizer.json", b"{}".as_slice())],
                vec![
                    manifest_entry("tokenizer.json", b"{}"),
                    manifest_entry("weights/model.bin", b"weights"),
                ],
            ),
        ];

        for (name, model_name, ignore_weights, payloads, manifest_entries) in cases {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            write_manifest_with_payloads(temp_dir.path(), model_name, &payloads, manifest_entries);

            let result = runtime
                .block_on(provider.download_model(
                    model_name,
                    Some(temp_dir.path().to_path_buf()),
                    ignore_weights,
                ))
                .expect("Expected cached model reuse");

            assert_eq!(result, model_dir, "scenario={name}");
        }
    }

    #[tokio::test]
    async fn test_get_model_path_scenarios() {
        let provider = GcsProvider;
        for scenario in ["missing", "partial_manifest", "complete"] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_name = match scenario {
                "missing" => "gs://test-bucket/org/model/rev-3",
                "partial_manifest" => "gs://test-bucket/org/model/rev-partial",
                "complete" => "gs://test-bucket/org/model/rev-5",
                _ => unreachable!("unexpected scenario"),
            };

            match scenario {
                "partial_manifest" => write_manifest_with_payloads(
                    temp_dir.path(),
                    model_name,
                    &[("tokenizer.json", b"{}")],
                    vec![
                        manifest_entry("tokenizer.json", b"{}"),
                        manifest_entry("weights/model.bin", b"weights"),
                    ],
                ),
                "complete" => {
                    write_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");
                }
                "missing" => {}
                _ => unreachable!("unexpected scenario"),
            }

            let result = provider
                .get_model_path(model_name, temp_dir.path().to_path_buf())
                .await;
            match scenario {
                "complete" => assert_eq!(
                    result.expect("Expected model path"),
                    expected_model_dir(temp_dir.path(), model_name)
                ),
                _ => assert!(
                    result
                        .expect_err("Expected cache miss")
                        .to_string()
                        .contains("not found in cache")
                ),
            }
        }
    }

    #[tokio::test]
    async fn test_delete_model_scenarios() {
        let provider = GcsProvider;
        for scenario in ["complete", "incomplete"] {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let model_name = "gs://test-bucket/org/model/rev-1";
            let model_dir = expected_model_dir(temp_dir.path(), model_name);
            match scenario {
                "complete" => {
                    write_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");
                }
                "incomplete" => {
                    write_incomplete_cached_model(
                        temp_dir.path(),
                        model_name,
                        "tokenizer.json",
                        b"{}",
                    );
                }
                _ => unreachable!("unexpected scenario"),
            }
            provider
                .delete_model(model_name, temp_dir.path().to_path_buf())
                .await
                .expect("Expected successful delete");
            assert!(!model_dir.exists());
        }
    }
}
