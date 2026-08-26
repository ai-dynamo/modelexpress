// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::metrics::grpc::RpcOutcome;
use crate::metrics::registry::{DownloadMetrics, RegistryMetrics, StatusLabel};
use crate::registry::backend::ClaimOutcome;
use crate::registry::entry_key::EntryKey;
use crate::registry::state::RegistryManager;
use modelexpress_common::{
    cache::{CacheConfig, resolve_model_path},
    constants, download,
    grpc::{
        api::{ApiRequest, ApiResponse, api_service_server::ApiService},
        health::{HealthRequest, HealthResponse, health_service_server::HealthService},
        model::{
            DeleteModelRequest, DeleteModelResponse, FileChunk, ModelDownloadRequest,
            ModelFileInfo, ModelFileList, ModelFileSelector, ModelFilesRequest,
            ModelProvider as GrpcModelProvider, ModelStatusUpdate,
            model_service_server::ModelService,
        },
    },
    models::{ModelProvider, ModelStatus},
    providers::is_weight_file,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};
use tokio::io::AsyncReadExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

static START_TIME: std::sync::OnceLock<SystemTime> = std::sync::OnceLock::new();

/// Ceiling on resolving a revision with the provider. This is a metadata lookup, not a
/// download, so it should complete quickly or not at all.
const REVISION_RESOLVE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Get the configured cache directory for model downloads
fn get_server_cache_dir() -> Option<std::path::PathBuf> {
    // Try to get cache configuration
    if let Ok(config) = CacheConfig::discover() {
        Some(config.local_path)
    } else {
        // Fall back to environment variable
        modelexpress_common::envs::hf_hub_cache()
    }
}

/// Returns true if the model's files are present in the given cache directory. Used to
/// guard against stale `DOWNLOADED` registry records that point at a cache entry which no
/// longer exists on disk (e.g. left behind by a client-side `model clear`). When no cache
/// directory is configured we cannot verify, so we assume the files are present to preserve
/// existing behavior rather than loop re-downloading forever.
async fn model_files_present(
    cache_dir: Option<std::path::PathBuf>,
    model_name: &str,
    provider: ModelProvider,
    revision: Option<&str>,
) -> bool {
    let Some(cache_dir) = cache_dir else {
        return true;
    };
    download::get_provider(provider)
        .get_model_path_revision(model_name, cache_dir, revision)
        .await
        .is_ok()
}

/// A model download request reduced to the identity the registry and the provider need.
///
/// `revision` is always the *resolved*, immutable revision (a commit SHA for Hugging
/// Face), never the branch or tag the caller typed, so cache and lease identity cannot
/// drift when a tag moves.
#[derive(Debug, Clone)]
pub struct DownloadTarget {
    pub model_name: String,
    pub provider: ModelProvider,
    pub revision: Option<String>,
    pub ignore_weights: bool,
}

impl DownloadTarget {
    /// The single string every registry backend keys this download on.
    fn entry_key(&self) -> String {
        EntryKey::new(
            self.model_name.clone(),
            self.revision.clone(),
            self.ignore_weights,
        )
        .to_string()
    }
}

/// Resolve a requested revision to the immutable revision the download will use.
///
/// A revision the caller pinned explicitly must resolve or the request fails: silently
/// serving the default revision instead would hand back a different model than asked
/// for. For an unpinned request we fall back to whatever snapshot is already cached,
/// so a Hub outage still serves models the server has downloaded before.
#[allow(clippy::result_large_err)] // tonic::Status grew past Clippy's Rust 1.98 threshold.
async fn resolve_target_revision(
    model_name: &str,
    provider: ModelProvider,
    cache_dir: Option<PathBuf>,
    requested: Option<&str>,
) -> Result<Option<String>, Status> {
    let provider_impl = download::get_provider(provider);

    // Pinning a revision on a provider that has none is a malformed request, not a
    // revision that happens to be missing.
    if requested.is_some() && !provider_impl.supports_revisions() {
        return Err(Status::invalid_argument(format!(
            "Provider '{}' does not support pinned revisions",
            provider_impl.provider_name()
        )));
    }

    // Resolution runs before the download lease is claimed, so an unresponsive provider
    // would otherwise hold the RPC open indefinitely without any download in progress.
    let resolve = provider_impl.resolve_revision(model_name, cache_dir.clone(), requested);
    let error = match tokio::time::timeout(REVISION_RESOLVE_TIMEOUT, resolve).await {
        Ok(Ok(resolved)) => return Ok(resolved),
        Ok(Err(e)) => e,
        Err(_) => anyhow::anyhow!("timed out after {}s", REVISION_RESOLVE_TIMEOUT.as_secs()),
    };

    if requested.is_some() {
        return Err(Status::not_found(format!(
            "Failed to resolve revision for model '{model_name}': {error:#}"
        )));
    }

    warn!(
        "Failed to resolve the default revision for '{model_name}': {error:#}; \
         falling back to the cached snapshot"
    );

    match provider_impl
        .get_model_path_revision(model_name, cache_dir.unwrap_or_default(), None)
        .await
        .ok()
        .and_then(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(String::from)
        }) {
        Some(cached) => Ok(Some(cached)),
        None => Err(Status::not_found(format!(
            "Failed to resolve revision for model '{model_name}': {error:#}"
        ))),
    }
}

/// Health service implementation
#[derive(Debug, Default)]
pub struct HealthServiceImpl;

#[tonic::async_trait]
impl HealthService for HealthServiceImpl {
    async fn get_health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let start_time = START_TIME.get_or_init(SystemTime::now);
        let uptime = SystemTime::now()
            .duration_since(*start_time)
            .unwrap_or_default()
            .as_secs();

        let response = HealthResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            status: "ok".to_string(),
            uptime,
        };

        Ok(Response::new(response))
    }
}

/// API service implementation
#[derive(Debug, Default)]
pub struct ApiServiceImpl;

/// Return `body` with the handler's own verdict attached.
///
/// See [`crate::metrics::grpc`]. Mirrors the helper in `p2p/service.rs`; kept
/// local to each module so a handler's outcome is stated next to the handler.
#[allow(clippy::result_large_err)] // Returns the handlers' own tonic::Status result type.
fn tagged<T>(body: T, outcome: RpcOutcome) -> Result<Response<T>, Status> {
    let mut response = Response::new(body);
    response.extensions_mut().insert(outcome);
    Ok(response)
}

#[tonic::async_trait]
impl ApiService for ApiServiceImpl {
    async fn send_request(
        &self,
        request: Request<ApiRequest>,
    ) -> Result<Response<ApiResponse>, Status> {
        let api_request = request.into_inner();
        info!("Received gRPC request: {:?}", api_request);

        // Process the request based on the action
        if api_request.action.as_str() == "ping" {
            info!("Processing ping request");
            let response_data = serde_json::json!({ "message": "pong" });
            let data_bytes = serde_json::to_vec(&response_data)
                .map_err(|e| Status::internal(format!("Serialization error: {e}")))?;

            tagged(
                ApiResponse {
                    success: true,
                    data: Some(data_bytes),
                    error: None,
                },
                RpcOutcome::Ok,
            )
        } else {
            error!("Unknown action: {}", api_request.action);
            // In band, like the P2P handlers: `Ok` on the wire with
            // `success: false` in the body, so the outcome has to be stated
            // rather than inferred from the status code.
            tagged(
                ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Unknown action: {}", api_request.action)),
                },
                RpcOutcome::InvalidArgument,
            )
        }
    }
}

/// Model service implementation
#[derive(Clone)]
pub struct ModelServiceImpl {
    tracker: Arc<ModelDownloadTracker>,
}

impl ModelServiceImpl {
    /// Each server owns its tracker, so multiple servers can run in one process.
    pub fn new(tracker: Arc<ModelDownloadTracker>) -> Self {
        Self { tracker }
    }
}

/// Helper function to collect all files in a model directory recursively
fn collect_model_files(
    base_path: &Path,
    current_path: &Path,
    file_selector: Option<&ModelFileSelector>,
    ignore_weights: bool,
) -> Vec<(PathBuf, u64)> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(current_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Ok(metadata) = std::fs::metadata(&path) {
                    // Get relative path from base_path
                    if let Ok(relative) = path.strip_prefix(base_path) {
                        // Validate that the relative path does not contain any '..' components or is absolute
                        let mut is_safe = true;
                        for comp in relative.components() {
                            use std::path::Component;
                            match comp {
                                Component::ParentDir
                                | Component::RootDir
                                | Component::Prefix(_) => {
                                    is_safe = false;
                                    break;
                                }
                                _ => {}
                            }
                        }
                        if !is_safe {
                            tracing::warn!(
                                "Skipping potentially unsafe file path: {:?} (relative: {:?})",
                                path,
                                relative
                            );
                        } else if ignore_weights
                            && is_weight_file(relative.to_string_lossy().as_ref())
                        {
                            // Weight file skipped because ignore_weights is set.
                        } else if file_selector.is_none_or(|selector| {
                            selector
                                .paths
                                .iter()
                                .any(|selector_path| Path::new(selector_path) == relative)
                        }) {
                            files.push((relative.to_path_buf(), metadata.len()));
                        }
                    }
                }
            } else if path.is_dir() {
                files.extend(collect_model_files(
                    base_path,
                    &path,
                    file_selector,
                    ignore_weights,
                ));
            }
        }
    }

    files
}

fn ensure_selected_files_exist(
    files: &[(PathBuf, u64)],
    file_selector: Option<&ModelFileSelector>,
) -> Result<(), String> {
    let Some(selector) = file_selector else {
        return Ok(());
    };

    if let Some(missing_path) = selector.paths.iter().find(|selector_path| {
        !files
            .iter()
            .any(|(path, _)| Path::new(selector_path) == path.as_path())
    }) {
        Err(format!(
            "Selected file not found in model directory: {missing_path}"
        ))
    } else {
        Ok(())
    }
}

#[tonic::async_trait]
impl ModelService for ModelServiceImpl {
    type EnsureModelDownloadedStream = ReceiverStream<Result<ModelStatusUpdate, Status>>;
    type StreamModelFilesStream = ReceiverStream<Result<FileChunk, Status>>;

    async fn ensure_model_downloaded(
        &self,
        request: Request<ModelDownloadRequest>,
    ) -> Result<Response<Self::EnsureModelDownloadedStream>, Status> {
        info!("Starting model download stream");
        let model_request = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        // Convert gRPC provider to our enum
        let grpc_provider = GrpcModelProvider::try_from(model_request.provider).map_err(|_| {
            Status::invalid_argument(format!(
                "Invalid provider value: {}",
                model_request.provider
            ))
        })?;
        let provider = ModelProvider::from(grpc_provider);
        let model_name = download::canonical_model_name(&model_request.model_name, provider)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let ignore_weights = model_request.ignore_weights;

        // Resolve before claiming: the registry keys the download lease on the resolved
        // revision, so two revisions of one model never coalesce onto a single claim.
        let revision = resolve_target_revision(
            &model_name,
            provider,
            get_server_cache_dir(),
            model_request.revision.as_deref(),
        )
        .await?;

        let target = DownloadTarget {
            model_name: model_name.clone(),
            provider,
            revision: revision.clone(),
            ignore_weights,
        };

        // Spawn a task to handle the streaming download updates
        let tracker = self.tracker.clone();
        tokio::spawn(async move {
            // Run the full claim + wait + retry flow. `ensure_model_downloaded` sends
            // its own initial status update (based on the `ClaimOutcome` returned by the
            // registry), so we don't do a pre-check here — a pre-check would either
            // duplicate that update or, worse, emit `status=ERROR` on a model we're
            // about to retry and trip the client-lib's terminal-error bailout before
            // the retry completion broadcast arrives.
            let final_status = tracker.ensure_model_downloaded(&target, &tx).await;

            // Send final status update
            let final_update = ModelStatusUpdate {
                model_name: model_name.clone(),
                status: modelexpress_common::grpc::model::ModelStatus::from(final_status) as i32,
                message: match final_status {
                    ModelStatus::DOWNLOADED => {
                        Some("Model download completed successfully".to_string())
                    }
                    ModelStatus::ERROR => Some("Model download failed".to_string()),
                    ModelStatus::DOWNLOADING => Some("Download still in progress".to_string()),
                },
                provider: grpc_provider as i32,
                resolved_revision: revision,
            };

            let _ = tx.send(Ok(final_update)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn stream_model_files(
        &self,
        request: Request<ModelFilesRequest>,
    ) -> Result<Response<Self::StreamModelFilesStream>, Status> {
        let files_request = request.into_inner();
        let chunk_size = if files_request.chunk_size == 0 {
            constants::DEFAULT_TRANSFER_CHUNK_SIZE
        } else {
            files_request.chunk_size as usize
        };

        // Convert gRPC provider to our enum
        let grpc_provider = GrpcModelProvider::try_from(files_request.provider).map_err(|_| {
            Status::invalid_argument(format!(
                "Invalid provider value: {}",
                files_request.provider
            ))
        })?;
        let provider = ModelProvider::from(grpc_provider);
        let model_name = download::canonical_model_name(&files_request.model_name, provider)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let provider_impl = download::get_provider(provider);

        info!(
            "Starting file stream for model: {} with chunk size: {} bytes",
            model_name, chunk_size
        );

        // Get the cache directory
        let cache_dir = get_server_cache_dir()
            .ok_or_else(|| Status::internal("Server cache directory not configured"))?;

        // Get the model path using the provider from the request. A requested revision
        // selects that exact snapshot instead of whichever one is newest on disk.
        let model_path = provider_impl
            .get_model_path_revision(
                &model_name,
                cache_dir.clone(),
                files_request.revision.as_deref(),
            )
            .await
            .map_err(|e| Status::not_found(format!("Model not found: {e}")))?;

        debug!("Model path resolved to: {:?}", model_path);

        let commit_hash = if provider == ModelProvider::HuggingFace {
            model_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(String::from)
        } else {
            None
        };

        if provider == ModelProvider::HuggingFace && commit_hash.is_none() {
            return Err(Status::internal(
                "Resolved Hugging Face model path did not contain a revision",
            ));
        }

        let expected_model_path =
            resolve_model_path(&cache_dir, provider, &model_name, commit_hash.as_deref()).map_err(
                |e| Status::internal(format!("Failed to resolve expected cache layout: {e}")),
            )?;

        if model_path != expected_model_path {
            error!(
                "Resolved model path '{}' does not match expected cache layout '{}' for model '{}'",
                model_path.display(),
                expected_model_path.display(),
                model_name
            );
            return Err(Status::internal(
                "Resolved model path does not match expected cache layout",
            ));
        }

        // Collect all files to stream
        let files = collect_model_files(
            &model_path,
            &model_path,
            files_request.file_selector.as_ref(),
            files_request.ignore_weights,
        );
        ensure_selected_files_exist(&files, files_request.file_selector.as_ref())
            .map_err(Status::not_found)?;

        if files.is_empty() {
            return Err(Status::not_found("No files found in model directory"));
        }

        let total_files = files.len();
        info!(
            "Found {} files to stream for model {}",
            total_files, model_name
        );

        let (tx, rx) = tokio::sync::mpsc::channel(16);

        // Spawn a task to stream files
        tokio::spawn(async move {
            // Allocate buffer once and reuse across all files
            let mut buffer = vec![0u8; chunk_size];
            let mut is_first_chunk = true;

            for (file_idx, (relative_path, total_size)) in files.iter().enumerate() {
                let file_path = model_path.join(relative_path);
                let is_last_file = file_idx == total_files.saturating_sub(1);

                debug!("Streaming file: {:?} ({} bytes)", relative_path, total_size);

                // Open the file
                let file = match tokio::fs::File::open(&file_path).await {
                    Ok(f) => f,
                    Err(e) => {
                        error!("Failed to open file {:?}: {}", file_path, e);
                        let _ = tx
                            .send(Err(Status::internal(format!("Failed to open file: {e}"))))
                            .await;
                        return;
                    }
                };

                let mut reader = tokio::io::BufReader::new(file);
                let mut offset: u64 = 0;

                if *total_size == 0 {
                    let first_chunk = std::mem::replace(&mut is_first_chunk, false);
                    let chunk = FileChunk {
                        relative_path: relative_path.to_string_lossy().to_string(),
                        data: Vec::new(),
                        offset: 0,
                        total_size: 0,
                        is_last_chunk: true,
                        is_last_file,
                        commit_hash: if first_chunk {
                            commit_hash.clone()
                        } else {
                            None
                        },
                    };

                    if tx.send(Ok(chunk)).await.is_err() {
                        debug!("Client disconnected during file stream");
                        return;
                    }

                    continue;
                }

                loop {
                    let bytes_read = match reader.read(&mut buffer).await {
                        Ok(0) => break, // EOF
                        Ok(n) => n,
                        Err(e) => {
                            error!("Failed to read file {:?}: {}", file_path, e);
                            let _ = tx
                                .send(Err(Status::internal(format!("Failed to read file: {e}"))))
                                .await;
                            return;
                        }
                    };

                    let is_last_chunk = offset.saturating_add(bytes_read as u64) >= *total_size;

                    let first_chunk = std::mem::replace(&mut is_first_chunk, false);

                    let chunk = FileChunk {
                        relative_path: relative_path.to_string_lossy().to_string(),
                        data: buffer[..bytes_read].to_vec(),
                        offset,
                        total_size: *total_size,
                        is_last_chunk,
                        is_last_file: is_last_file && is_last_chunk,
                        commit_hash: if first_chunk {
                            commit_hash.clone()
                        } else {
                            None
                        },
                    };

                    if tx.send(Ok(chunk)).await.is_err() {
                        debug!("Client disconnected during file stream");
                        return;
                    }

                    offset = offset.saturating_add(bytes_read as u64);
                }
            }

            info!("File streaming completed for model");
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn list_model_files(
        &self,
        request: Request<ModelFilesRequest>,
    ) -> Result<Response<ModelFileList>, Status> {
        let files_request = request.into_inner();

        // Convert gRPC provider to our enum
        let grpc_provider = GrpcModelProvider::try_from(files_request.provider).map_err(|_| {
            Status::invalid_argument(format!(
                "Invalid provider value: {}",
                files_request.provider
            ))
        })?;
        let provider = ModelProvider::from(grpc_provider);
        let model_name = download::canonical_model_name(&files_request.model_name, provider)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let provider_impl = download::get_provider(provider);

        info!("Listing files for model: {}", model_name);

        // Get the cache directory
        let cache_dir = get_server_cache_dir()
            .ok_or_else(|| Status::internal("Server cache directory not configured"))?;

        // Get the model path using the provider from the request
        let model_path = provider_impl
            .get_model_path_revision(&model_name, cache_dir, files_request.revision.as_deref())
            .await
            .map_err(|e| Status::not_found(format!("Model not found: {e}")))?;

        // Collect all files
        let files = collect_model_files(
            &model_path,
            &model_path,
            files_request.file_selector.as_ref(),
            files_request.ignore_weights,
        );
        ensure_selected_files_exist(&files, files_request.file_selector.as_ref())
            .map_err(Status::not_found)?;

        let file_infos: Vec<ModelFileInfo> = files
            .iter()
            .map(|(path, size)| ModelFileInfo {
                relative_path: path.to_string_lossy().to_string(),
                size: *size,
            })
            .collect();

        let total_size: u64 = files.iter().map(|(_, size)| size).sum();

        Ok(Response::new(ModelFileList {
            model_name,
            files: file_infos,
            total_size,
        }))
    }

    async fn delete_model(
        &self,
        request: Request<DeleteModelRequest>,
    ) -> Result<Response<DeleteModelResponse>, Status> {
        let delete_request = request.into_inner();

        let grpc_provider = GrpcModelProvider::try_from(delete_request.provider).map_err(|_| {
            Status::invalid_argument(format!(
                "Invalid provider value: {}",
                delete_request.provider
            ))
        })?;
        let provider = ModelProvider::from(grpc_provider);
        let model_name = download::canonical_model_name(&delete_request.model_name, provider)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        // A model can hold several entries at once (one per revision, and separate ones
        // for metadata-only downloads). `model clear` means "forget this model", so drop
        // all of them rather than only the unpinned full-weight entry.
        let tracker = self.tracker.clone();
        let outcome = tracker.delete_model_entries(&model_name).await;
        let removed = outcome.removed;
        info!("Deleted {removed} registry record(s) for model '{model_name}'");

        // The response stays `success: true` -- the fallback delete was still
        // attempted and callers depend on that shape. But a registry outage and a
        // model with no entries both return zero records here, so without the tag
        // the RPC would be recorded as a plain success during an outage. That is
        // the exact failure this layer exists to prevent.
        tagged(
            DeleteModelResponse {
                success: true,
                message: Some(format!(
                    "Model '{model_name}' removed from registry ({removed} record(s))"
                )),
            },
            if outcome.degraded {
                RpcOutcome::BackendError
            } else {
                RpcOutcome::Ok
            },
        )
    }
}

/// What clearing a model's registry entries actually achieved.
///
/// `removed` alone cannot distinguish "this model had no entries" from "the
/// registry was unreachable and only the fallback key was tried" -- both are
/// zero.
pub struct DeleteEntriesOutcome {
    /// How many records were removed.
    pub removed: usize,
    /// The registry listing failed, so `removed` is a floor rather than a count.
    pub degraded: bool,
}

/// Type alias for the complex waiting channels type
type WaitingChannels =
    Arc<Mutex<HashMap<String, Vec<tokio::sync::mpsc::Sender<Result<ModelStatusUpdate, Status>>>>>>;

/// What a status update is routed by and what it reports.
///
/// Waiters are registered under the registry entry key, which carries the revision and
/// weight mode, while the update itself reports the plain model name the caller asked
/// for plus the revision it resolved to.
struct UpdateIdentity<'a> {
    entry_key: &'a str,
    model_name: &'a str,
    provider: ModelProvider,
    resolved_revision: Option<&'a str>,
}

impl UpdateIdentity<'_> {
    fn status_update(&self, status: ModelStatus, message: Option<String>) -> ModelStatusUpdate {
        ModelStatusUpdate {
            model_name: self.model_name.to_string(),
            status: modelexpress_common::grpc::model::ModelStatus::from(status) as i32,
            message,
            provider: GrpcModelProvider::from(self.provider) as i32,
            resolved_revision: self.resolved_revision.map(String::from),
        }
    }
}

/// Tracks the status of model downloads through the distributed registry backend.
#[derive(Clone)]
pub struct ModelDownloadTracker {
    /// Distributed registry (Redis today, K8s CRDs in a follow-up).
    registry: Arc<RegistryManager>,
    /// Maps model names to list of channels waiting for updates on this server replica.
    waiting_channels: WaitingChannels,
    download_metrics: DownloadMetrics,
    registry_metrics: RegistryMetrics,
}

const DOWNLOAD_LEASE_DURATION: std::time::Duration = std::time::Duration::from_secs(30);
const DOWNLOAD_HEARTBEAT_INTERVAL: tokio::time::Duration = tokio::time::Duration::from_secs(10);

impl ModelDownloadTracker {
    pub fn new(
        registry: Arc<RegistryManager>,
        download_metrics: DownloadMetrics,
        registry_metrics: RegistryMetrics,
    ) -> Self {
        Self {
            registry,
            waiting_channels: Arc::new(Mutex::new(HashMap::new())),
            download_metrics,
            registry_metrics,
        }
    }

    /// Number of models with callers currently waiting on a download.
    ///
    /// This map is never evicted wholesale, so its size is the early warning for
    /// the leak that would eventually OOM the server. A poisoned lock reports the
    /// recovered length rather than failing: a metric read must not be able to
    /// take down the download path.
    pub fn waiting_count(&self) -> usize {
        match self.waiting_channels.lock() {
            Ok(waiting) => waiting.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    async fn touch_and_log(&self, model_name: &str) {
        if let Err(e) = self.registry.touch_model(model_name).await {
            error!("Failed to touch model {model_name}: {e}");
        }
    }

    /// Gets the status of a model from the registry, bumping `last_used_at` on hit.
    /// Returns None on lookup failure (error logged) or unknown model.
    pub async fn get_status(&self, model_name: &str) -> Option<ModelStatus> {
        match self.registry.get_status(model_name).await {
            Ok(Some(status)) => {
                self.touch_and_log(model_name).await;
                Some(status)
            }
            Ok(None) => None,
            Err(e) => {
                error!("Failed to get model status from registry: {e}");
                None
            }
        }
    }

    /// Sets the status of a model and notifies all waiting channels on this replica.
    pub async fn set_status_and_notify(
        &self,
        model_name: String,
        status: ModelStatus,
        provider: ModelProvider,
        message: Option<String>,
    ) {
        if let Err(e) = self
            .registry
            .set_status(&model_name, provider, status, message.clone())
            .await
        {
            error!("Failed to update model status in registry: {e}");
            return;
        }
        self.notify_waiters(
            UpdateIdentity {
                entry_key: &model_name,
                model_name: &model_name,
                provider,
                resolved_revision: None,
            },
            status,
            message,
        );
    }

    fn notify_waiters(
        &self,
        identity: UpdateIdentity<'_>,
        status: ModelStatus,
        message: Option<String>,
    ) {
        let mut waiting = match self.waiting_channels.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                error!("Waiting channels mutex is poisoned, recovering");
                poisoned.into_inner()
            }
        };
        if let Some(channels) = waiting.get(identity.entry_key) {
            let update = identity.status_update(status, message);
            for channel in channels {
                let _ = channel.try_send(Ok(update.clone()));
            }
            if status == ModelStatus::DOWNLOADED || status == ModelStatus::ERROR {
                waiting.remove(identity.entry_key);
            }
        }
    }

    /// Adds a channel that wants updates on a specific model (server-replica-local).
    pub fn add_waiting_channel(
        &self,
        model_name: &str,
        tx: tokio::sync::mpsc::Sender<Result<ModelStatusUpdate, Status>>,
    ) {
        let mut waiting = match self.waiting_channels.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                error!("Waiting channels mutex is poisoned, recovering");
                poisoned.into_inner()
            }
        };
        waiting.entry(model_name.to_string()).or_default().push(tx);
    }

    /// Deletes a model record from the registry and clears local waiters.
    /// Returns whether the registry record was actually removed.
    ///
    /// The error is still swallowed -- callers proceed to notify waiters either
    /// way -- but the outcome is reported, because a caller that records the
    /// removal as a lifecycle event must not do so when nothing was removed.
    pub async fn delete_status(&self, model_name: &str) -> bool {
        let deleted = match self.registry.delete_model(model_name).await {
            Ok(()) => true,
            Err(e) => {
                error!("Failed to delete model from registry: {e}");
                false
            }
        };
        let mut waiting = match self.waiting_channels.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                error!("Waiting channels mutex is poisoned, recovering");
                poisoned.into_inner()
            }
        };
        waiting.remove(model_name);
        deleted
    }

    /// Deletes every registry entry belonging to `model_name`, whatever revision or
    /// weight mode each entry covers.
    ///
    /// The registry-listing failure is swallowed on purpose: the fallback still
    /// clears the common case, which is better than refusing outright. It is
    /// reported in the return value so the caller can say so, because otherwise a
    /// backend outage and a model with no entries are the same `0`.
    pub async fn delete_model_entries(&self, model_name: &str) -> DeleteEntriesOutcome {
        let records = match self.registry.get_models_by_last_used(None).await {
            Ok(records) => records,
            Err(e) => {
                error!("Failed to list registry records for '{model_name}': {e}");
                // Fall back to the unpinned full-weight key so the common case still
                // clears rather than failing outright.
                self.delete_status(model_name).await;
                return DeleteEntriesOutcome {
                    removed: 0,
                    degraded: true,
                };
            }
        };

        let mut removed: usize = 0;
        for record in records {
            if EntryKey::parse(&record.model_name).belongs_to(model_name) {
                // Deleting a record mid-download is a real exit from DOWNLOADING,
                // and the only one the claim lifecycle cannot see: once the key
                // is gone, finish_download_claim finds nothing to fence and
                // returns false, so it records no departure. Without this the
                // arrival booked by the claim is never matched and the flow
                // accounting stays permanently ahead. The status is already in
                // hand here, so this costs no extra read.
                // Only once the record is really gone. delete_model can fail and
                // is swallowed, and recording first would book a departure for an
                // entry still sitting in DOWNLOADING -- turning a backend outage
                // into a permanently understated flow count.
                let deleted = self.delete_status(&record.model_name).await;
                if deleted && record.status == ModelStatus::DOWNLOADING {
                    self.registry_metrics
                        .record_transition(StatusLabel::Downloading, StatusLabel::Absent);
                }
                removed = removed.saturating_add(1);
            }
        }
        DeleteEntriesOutcome {
            removed,
            degraded: false,
        }
    }

    /// Spawn a background task that actually downloads the model, updating the tracker on
    /// success or failure. Extracted here so the claim and retry paths share the code.
    fn spawn_download_task(&self, target: DownloadTarget, retry: bool, claim_id: String) {
        let tracker = self.clone();
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let entry_key = target.entry_key();
            let DownloadTarget {
                model_name,
                provider,
                revision,
                ignore_weights,
            } = target;
            let cache_dir = get_server_cache_dir();
            let download = download::download_model_revision(
                &model_name,
                provider,
                cache_dir,
                ignore_weights,
                revision.as_deref(),
            );
            tokio::pin!(download);
            let start = tokio::time::Instant::now()
                .checked_add(DOWNLOAD_HEARTBEAT_INTERVAL)
                .unwrap_or_else(tokio::time::Instant::now);
            let mut heartbeat = tokio::time::interval_at(start, DOWNLOAD_HEARTBEAT_INTERVAL);
            let result = loop {
                tokio::select! {
                    result = &mut download => break result,
                    _ = heartbeat.tick() => {
                        match tracker
                            .registry
                            .refresh_download_claim(
                                &entry_key,
                                provider,
                                &claim_id,
                                DOWNLOAD_LEASE_DURATION,
                            )
                            .await
                        {
                            Ok(true) => {}
                            Ok(false) => {
                                warn!(
                                    "Stopping download for {model_name}: lease ownership was lost"
                                );
                                return;
                            }
                            Err(e) => {
                                warn!("Failed to refresh download lease for {model_name}: {e}");
                            }
                        }
                    }
                }
            };

            let (status, message) = match result {
                Ok(_path) => (
                    ModelStatus::DOWNLOADED,
                    Some("Model download completed successfully".to_string()),
                ),
                Err(e) => {
                    if retry {
                        error!("Failed to download model {model_name} on retry: {e}");
                    } else {
                        error!("Failed to download model {model_name}: {e}");
                    }
                    let msg = if retry {
                        format!("Download failed on retry: {e}")
                    } else {
                        format!("Download failed: {e}")
                    };
                    (ModelStatus::ERROR, Some(msg))
                }
            };

            // Timed from the top of the task, so this is the whole download
            // rather than the registry write below it. Observed before the fence
            // check: the bytes were fetched regardless of whether this replica
            // still owns the right to publish the result.
            tracker
                .download_metrics
                .observe(StatusLabel::from(status), started.elapsed().as_secs_f64());

            match tracker
                .registry
                .finish_download_claim(&entry_key, provider, &claim_id, status, message.clone())
                .await
            {
                Ok(true) => {
                    tracker.notify_waiters(
                        UpdateIdentity {
                            entry_key: &entry_key,
                            model_name: &model_name,
                            provider,
                            resolved_revision: revision.as_deref(),
                        },
                        status,
                        message,
                    );
                }
                Ok(false) => {
                    warn!("Ignoring completion for {model_name}: lease ownership was lost");
                }
                Err(e) => {
                    error!("Failed to finish download claim for {model_name}: {e}");
                }
            }
        });
    }

    /// Initiates a download for a model and streams status updates.
    ///
    /// All registry operations key on the target's entry key rather than its model name,
    /// so a second revision (or a full-weight request behind a metadata-only one) claims
    /// its own lease instead of coalescing onto an unrelated download.
    pub async fn ensure_model_downloaded(
        &self,
        target: &DownloadTarget,
        tx: &tokio::sync::mpsc::Sender<Result<ModelStatusUpdate, Status>>,
    ) -> ModelStatus {
        let entry_key = target.entry_key();
        let entry_key = entry_key.as_str();
        let model_name = target.model_name.as_str();
        let provider = target.provider;
        let identity = UpdateIdentity {
            entry_key,
            model_name,
            provider,
            resolved_revision: target.revision.as_deref(),
        };

        // Atomically try to claim this model for download. The `ClaimOutcome` tells us
        // whether THIS replica won the claim or is observing someone else's claim —
        // status alone (`DOWNLOADING`) can't distinguish those cases across replicas.
        // A claim may report an existing `DOWNLOADED` record whose files no longer exist
        // on disk (e.g. after a client-side `model clear` that only removed local files).
        // When that happens we drop the stale record and re-claim once, so the download
        // path runs instead of returning a false success. Bounded to two attempts to
        // avoid looping if the delete or a concurrent re-claim keeps the record around.
        const MAX_CLAIM_ATTEMPTS: usize = 2;
        let mut claim_id = uuid::Uuid::new_v4().to_string();
        let mut attempt: usize = 0;
        let (status, is_owner) = loop {
            attempt = attempt.saturating_add(1);
            match self
                .registry
                .try_claim_for_download(entry_key, provider, &claim_id, DOWNLOAD_LEASE_DURATION)
                .await
            {
                // Both outcomes mean this replica owns the download; they differ
                // only in cost, which the metrics layer records separately.
                Ok(ClaimOutcome::Claimed | ClaimOutcome::TookOver) => {
                    break (ModelStatus::DOWNLOADING, true);
                }
                Ok(ClaimOutcome::AlreadyExists(existing)) => {
                    if existing == ModelStatus::DOWNLOADED
                        && attempt < MAX_CLAIM_ATTEMPTS
                        && !model_files_present(
                            get_server_cache_dir(),
                            model_name,
                            provider,
                            target.revision.as_deref(),
                        )
                        .await
                    {
                        error!(
                            "Registry reports model '{model_name}' as DOWNLOADED but its files \
                             are missing from the cache; clearing the stale record and \
                             re-downloading"
                        );
                        self.delete_status(entry_key).await;
                        continue;
                    }
                    if existing == ModelStatus::DOWNLOADED {
                        // Returning an existing downloaded model is a cache hit for LRU purposes.
                        self.touch_and_log(entry_key).await;
                    }
                    break (existing, false);
                }
                Err(e) => {
                    error!("Failed to claim model for download: {e}");
                    let error_update = identity.status_update(
                        ModelStatus::ERROR,
                        Some("Registry error occurred".to_string()),
                    );
                    let _ = tx.send(Ok(error_update)).await;
                    return ModelStatus::ERROR;
                }
            }
        };

        // If we observed a previous ERROR, attempt the ERROR -> DOWNLOADING CAS up front.
        // Only the CAS winner spawns the retry download; observers fall through to the
        // wait loop. Doing this *before* the initial stream update keeps the reported
        // status honest: after this block, the record is DOWNLOADING (the record may
        // briefly have been ERROR, but the client should wait, not bail).
        let (effective_status, is_retry_owner) = if status == ModelStatus::ERROR {
            let won = match self
                .registry
                .try_reset_error_for_retry(entry_key, provider, &claim_id, DOWNLOAD_LEASE_DURATION)
                .await
            {
                Ok(won) => won,
                Err(e) => {
                    error!("Failed to CAS status for retry: {e}");
                    let _ = tx
                        .send(Ok(identity.status_update(
                            ModelStatus::ERROR,
                            Some("Registry error occurred during retry".to_string()),
                        )))
                        .await;
                    return ModelStatus::ERROR;
                }
            };
            (ModelStatus::DOWNLOADING, won)
        } else {
            (status, false)
        };

        let update = identity.status_update(
            effective_status,
            match (status, effective_status) {
                (_, ModelStatus::DOWNLOADED) => Some("Model already downloaded".to_string()),
                (ModelStatus::ERROR, _) => Some("Previous download failed, retrying".to_string()),
                (_, ModelStatus::DOWNLOADING) => Some("Model download in progress".to_string()),
                // effective can never be ERROR: ERROR observations are CAS'd above.
                (_, ModelStatus::ERROR) => Some("Download error".to_string()),
            },
        );
        let _ = tx.send(Ok(update)).await;

        if effective_status == ModelStatus::DOWNLOADING {
            // Every caller is a waiter — whether we own the download or not, we still
            // need a channel so the completion broadcast reaches this stream.
            self.add_waiting_channel(entry_key, tx.clone());

            // Spawn the download only on the replica that won the claim (fresh
            // download) or won the ERROR-retry CAS. Everyone else waits.
            if is_owner || is_retry_owner {
                let retry = status == ModelStatus::ERROR;
                self.spawn_download_task(target.clone(), retry, claim_id);
                claim_id = uuid::Uuid::new_v4().to_string();
            }

            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                match self
                    .registry
                    .try_claim_for_download(entry_key, provider, &claim_id, DOWNLOAD_LEASE_DURATION)
                    .await
                {
                    // A taken-over lease must be re-driven exactly like a fresh
                    // claim. Letting it fall through instead would leave the entry
                    // wedged in DOWNLOADING with no owner running.
                    Ok(ClaimOutcome::Claimed | ClaimOutcome::TookOver) => {
                        self.spawn_download_task(target.clone(), false, claim_id);
                        claim_id = uuid::Uuid::new_v4().to_string();
                    }
                    Ok(ClaimOutcome::AlreadyExists(current_status))
                        if current_status != ModelStatus::DOWNLOADING =>
                    {
                        return current_status;
                    }
                    Ok(ClaimOutcome::AlreadyExists(_)) => {}
                    Err(e) => warn!("Failed to poll download lease for {model_name}: {e}"),
                }
            }
        }

        effective_status
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::{api::ApiRequest, health::HealthRequest};
    use modelexpress_common::test_support::{EnvVarGuard, acquire_env_mutex};
    use tempfile::TempDir;
    use tokio_stream::StreamExt;
    use tonic::Request;

    #[tokio::test]
    async fn test_health_service() {
        let service = HealthServiceImpl;
        let request = Request::new(HealthRequest {});

        let response = service.get_health(request).await;
        assert!(response.is_ok());

        let health_response = response.expect("Health response should be ok").into_inner();
        assert_eq!(health_response.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(health_response.status, "ok");
        // uptime is u64, always >= 0, so just verify it exists
        let _uptime = health_response.uptime;
    }

    /// The unknown-action path is an in-band failure and must not read as `ok`.
    #[tokio::test]
    async fn unknown_action_is_tagged_invalid_argument() {
        let service = ApiServiceImpl;
        let response = service
            .send_request(Request::new(ApiRequest {
                id: "test-id".to_string(),
                action: "not-a-real-action".to_string(),
                payload: None,
            }))
            .await
            .expect("the RPC itself succeeds");

        assert_eq!(
            response.extensions().get::<RpcOutcome>(),
            Some(&RpcOutcome::InvalidArgument)
        );
        assert!(!response.into_inner().success);
    }

    #[tokio::test]
    async fn test_api_service_ping() {
        let service = ApiServiceImpl;
        let request = Request::new(ApiRequest {
            id: "test-id".to_string(),
            action: "ping".to_string(),
            payload: None,
        });

        let response = service.send_request(request).await;
        assert!(response.is_ok());

        let api_response = response.expect("API response should be ok").into_inner();
        assert!(api_response.success);
        assert!(api_response.data.is_some());
        assert!(api_response.error.is_none());

        // Check that the response data contains "pong"
        let data_bytes = api_response.data.expect("Data should be present");
        let data: serde_json::Value =
            serde_json::from_slice(&data_bytes).expect("Data should be valid JSON");
        assert_eq!(data["message"], "pong");
    }

    #[tokio::test]
    async fn test_api_service_unknown_action() {
        let service = ApiServiceImpl;
        let request = Request::new(ApiRequest {
            id: "test-id".to_string(),
            action: "unknown-action".to_string(),
            payload: None,
        });

        let response = service.send_request(request).await;
        assert!(response.is_ok());

        let api_response = response.expect("API response should be ok").into_inner();
        assert!(!api_response.success);
        assert!(api_response.data.is_none());
        assert!(api_response.error.is_some());

        let error_message = api_response.error.expect("Error should be present");
        assert!(error_message.contains("Unknown action"));
    }

    // Tracker tests exercise the ModelDownloadTracker's interaction with a mocked
    // RegistryBackend. The full backend semantics (claim atomicity, LRU ordering, etc.) are
    // covered by the per-backend unit tests in modelexpress_server::registry and by the
    // testcontainers-based integration tests.
    fn tracker_with_mock(
        mock: crate::registry::backend::MockRegistryBackend,
    ) -> ModelDownloadTracker {
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(mock)));
        ModelDownloadTracker::new(
            registry,
            DownloadMetrics::register(&mut crate::metrics::new_registry()),
            RegistryMetrics::register(&mut crate::metrics::new_registry()),
        )
    }

    /// As [`tracker_with_mock`], but hands back the Prometheus registry the metric
    /// families were registered into.
    ///
    /// [`tracker_with_mock`] registers into a temporary registry that is dropped on
    /// the spot, so nothing recorded through it can ever be encoded. Any metric
    /// assertion against a tracker built that way would be vacuous by construction.
    fn tracker_with_mock_and_registry(
        mock: crate::registry::backend::MockRegistryBackend,
    ) -> (ModelDownloadTracker, prometheus_client::registry::Registry) {
        let mut metrics_registry = crate::metrics::new_registry();
        let download_metrics = DownloadMetrics::register(&mut metrics_registry);
        let registry_metrics = RegistryMetrics::register(&mut metrics_registry);
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(mock)));
        (
            ModelDownloadTracker::new(registry, download_metrics, registry_metrics),
            metrics_registry,
        )
    }

    fn encoded_metrics(registry: &prometheus_client::registry::Registry) -> String {
        crate::metrics::encode_text(registry).unwrap_or_else(|_| String::from("<encode failed>"))
    }

    /// The `downloading -> absent` series, or 0 when it was never created.
    fn departures_from_downloading(encoded: &str) -> i64 {
        encoded
            .lines()
            .find_map(|line| {
                line.strip_prefix(
                    r#"mx_registry_status_transitions_total{from="downloading",to="absent"} "#,
                )
            })
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_default()
    }

    fn record_with_status(
        model_name: &str,
        status: ModelStatus,
    ) -> crate::registry::backend::ModelRecord {
        let now = chrono::Utc::now();
        crate::registry::backend::ModelRecord {
            model_name: model_name.to_string(),
            provider: ModelProvider::HuggingFace,
            status,
            created_at: now,
            last_used_at: now,
            message: None,
        }
    }

    /// Unpinned, full-weight target: its entry key is the bare model name.
    fn test_target(model_name: &str) -> DownloadTarget {
        DownloadTarget {
            model_name: model_name.to_string(),
            provider: ModelProvider::HuggingFace,
            revision: None,
            ignore_weights: false,
        }
    }

    #[tokio::test]
    async fn test_tracker_get_status_missing_returns_none() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_status().once().returning(|_| Ok(None));
        // touch is NOT called when status is missing
        let tracker = tracker_with_mock(mock);
        assert!(tracker.get_status("unknown").await.is_none());
    }

    #[tokio::test]
    async fn test_tracker_get_status_hit_bumps_last_used_at() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_status()
            .once()
            .returning(|_| Ok(Some(ModelStatus::DOWNLOADED)));
        mock.expect_touch_model().once().returning(|_| Ok(()));
        let tracker = tracker_with_mock(mock);
        assert_eq!(tracker.get_status("m").await, Some(ModelStatus::DOWNLOADED));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_tracker_downloaded_cache_hit_bumps_last_used_at() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");
        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_try_claim_for_download()
            .with(
                mockall::predicate::eq("test/model"),
                mockall::predicate::eq(ModelProvider::HuggingFace),
                mockall::predicate::always(),
                mockall::predicate::always(),
            )
            .once()
            .returning(|_, _, _, _| Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED)));
        mock.expect_touch_model()
            .with(mockall::predicate::eq("test/model"))
            .once()
            .returning(|_| Ok(()));
        let tracker = tracker_with_mock(mock);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);

        assert_eq!(
            tracker
                .ensure_model_downloaded(&test_target("test/model"), &tx)
                .await,
            ModelStatus::DOWNLOADED
        );
    }

    /// Claim, touch, and report a cache hit for `target`, asserting the registry was
    /// keyed on the target's entry key rather than on the bare model name.
    async fn assert_claims_on_entry_key(target: DownloadTarget) {
        let expected_key = target.entry_key();
        assert_ne!(
            expected_key, target.model_name,
            "This target must not reduce to the bare model name, or the test proves nothing"
        );
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_try_claim_for_download()
            .with(
                mockall::predicate::eq(expected_key.clone()),
                mockall::predicate::eq(ModelProvider::HuggingFace),
                mockall::predicate::always(),
                mockall::predicate::always(),
            )
            .once()
            .returning(|_, _, _, _| Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED)));
        mock.expect_touch_model()
            .with(mockall::predicate::eq(expected_key))
            .once()
            .returning(|_| Ok(()));

        let tracker = tracker_with_mock(mock);
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);

        assert_eq!(
            tracker.ensure_model_downloaded(&target, &tx).await,
            ModelStatus::DOWNLOADED
        );

        // Clients see the model name they asked for, not the internal registry key.
        let update = rx
            .recv()
            .await
            .expect("Expected a status update")
            .expect("Expected an ok status update");
        assert_eq!(update.model_name, target.model_name);
        assert_eq!(update.resolved_revision, target.revision);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_tracker_scopes_the_lease_to_the_resolved_revision() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");
        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        assert_claims_on_entry_key(DownloadTarget {
            revision: Some("abc123".to_string()),
            ..test_target("test/model")
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_tracker_separates_metadata_only_from_full_weight_downloads() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");
        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        // A metadata-only entry must not be the same registry record as a full-weight
        // one, or a weightless snapshot would satisfy a request that needs weights.
        assert_claims_on_entry_key(DownloadTarget {
            ignore_weights: true,
            ..test_target("test/model")
        })
        .await;
    }

    #[tokio::test]
    async fn test_tracker_set_status_notifies_waiting_channel() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_set_status()
            .once()
            .returning(|_, _, _, _| Ok(()));
        let tracker = tracker_with_mock(mock);

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        tracker.add_waiting_channel("m", tx);

        tracker
            .set_status_and_notify(
                "m".to_string(),
                ModelStatus::DOWNLOADED,
                ModelProvider::HuggingFace,
                Some("done".to_string()),
            )
            .await;

        let update = rx.recv().await.expect("waiter should receive update");
        let update = update.expect("notify should send Ok");
        assert_eq!(update.model_name, "m");
        assert_eq!(
            update.status,
            modelexpress_common::grpc::model::ModelStatus::Downloaded as i32
        );
        assert_eq!(update.message.as_deref(), Some("done"));

        // Terminal status removes waiters.
        let waiters = tracker
            .waiting_channels
            .lock()
            .expect("waiters lock")
            .get("m")
            .map_or(0, std::vec::Vec::len);
        assert_eq!(waiters, 0);
    }

    #[tokio::test]
    async fn test_tracker_delete_status_clears_backend_and_waiters() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_delete_model().once().returning(|_| Ok(()));
        let tracker = tracker_with_mock(mock);

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("m", tx);
        tracker.delete_status("m").await;

        let waiters = tracker
            .waiting_channels
            .lock()
            .expect("waiters lock")
            .contains_key("m");
        assert!(!waiters);
    }

    /// The bool `delete_status` returns is the sole gate on the `downloading ->
    /// absent` departure, and no caller is forced to read it -- there is no
    /// `#[must_use]` and two of the three call sites discard it. So the contract
    /// is pinned here directly rather than only through the delete-path tests
    /// below, which would still pass if the two arms were swapped in step.
    #[tokio::test]
    async fn delete_status_reports_whether_the_backend_removed_the_record() {
        let mut removed = crate::registry::backend::MockRegistryBackend::new();
        removed.expect_delete_model().once().returning(|_| Ok(()));
        assert!(
            tracker_with_mock(removed).delete_status("m").await,
            "a successful delete must report true"
        );

        let mut failed = crate::registry::backend::MockRegistryBackend::new();
        failed
            .expect_delete_model()
            .once()
            .returning(|_| Err("backend unreachable".into()));
        assert!(
            !tracker_with_mock(failed).delete_status("m").await,
            "the error is still swallowed, but it must not report a removal"
        );
    }

    /// `model clear` on an entry that is still downloading is the third exit from
    /// `DOWNLOADING`, and the only one the claim lifecycle cannot observe for
    /// itself: once the record is gone `finish_download_claim` finds nothing to
    /// fence and records no departure, so the deleting side has to book it.
    /// Without this the arrival booked by the claim is never matched and the flow
    /// accounting stays permanently ahead by one per deleted download.
    #[tokio::test]
    async fn deleting_a_downloading_entry_books_a_departure() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .once()
            .returning(|_| Ok(vec![record_with_status("m", ModelStatus::DOWNLOADING)]));
        mock.expect_delete_model().once().returning(|_| Ok(()));

        let (tracker, metrics_registry) = tracker_with_mock_and_registry(mock);
        let outcome = tracker.delete_model_entries("m").await;
        assert_eq!(outcome.removed, 1);
        assert!(!outcome.degraded);

        let encoded = encoded_metrics(&metrics_registry);
        assert_eq!(
            departures_from_downloading(&encoded),
            1,
            "a deleted download must not stay counted as in flight: {encoded}"
        );
    }

    /// `delete_model` failing is swallowed so waiters are still notified, which is
    /// exactly why the departure has to be gated on the outcome. Booking it anyway
    /// would report a departure for an entry still sitting in `DOWNLOADING` and
    /// leave the flow count permanently short during a backend outage.
    #[tokio::test]
    async fn a_failed_delete_books_no_departure() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .once()
            .returning(|_| Ok(vec![record_with_status("m", ModelStatus::DOWNLOADING)]));
        mock.expect_delete_model()
            .once()
            .returning(|_| Err("backend unreachable".into()));

        let (tracker, metrics_registry) = tracker_with_mock_and_registry(mock);
        tracker.delete_model_entries("m").await;

        let encoded = encoded_metrics(&metrics_registry);
        assert_eq!(
            departures_from_downloading(&encoded),
            0,
            "a swallowed delete failure must not book a departure: {encoded}"
        );
    }

    /// Ordinary cache eviction clears `DOWNLOADED` entries. Those never occupied
    /// the in-flight level, so counting them as departures would drive the
    /// derivation negative on the most common delete there is.
    #[tokio::test]
    async fn deleting_a_downloaded_entry_books_no_departure() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .once()
            .returning(|_| Ok(vec![record_with_status("m", ModelStatus::DOWNLOADED)]));
        mock.expect_delete_model().once().returning(|_| Ok(()));

        let (tracker, metrics_registry) = tracker_with_mock_and_registry(mock);
        let outcome = tracker.delete_model_entries("m").await;
        assert_eq!(outcome.removed, 1, "the record is still removed");

        let encoded = encoded_metrics(&metrics_registry);
        assert_eq!(
            departures_from_downloading(&encoded),
            0,
            "a completed model was never in flight: {encoded}"
        );
    }

    /// `waiting_count` is the sole input to `mx_state_entries{map="download_waiters"}`,
    /// the gauge registered as "the OOM early warning". Stuck-at-zero is the failure
    /// mode that matters: the map leaks, the server heads for OOM, and the metric
    /// built to warn about it reads flat the whole way. The refresh task's own test
    /// substitutes a stub closure, so nothing else executes this function.
    #[tokio::test]
    async fn waiting_count_tracks_the_waiter_map() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_delete_model().once().returning(|_| Ok(()));
        let tracker = tracker_with_mock(mock);
        assert_eq!(tracker.waiting_count(), 0);

        let (tx_a, _rx_a) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("a", tx_a);
        let (tx_b, _rx_b) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("b", tx_b);
        // Keyed by model, so a second waiter on "a" does not add an entry.
        let (tx_a2, _rx_a2) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("a", tx_a2);
        assert_eq!(tracker.waiting_count(), 2);

        tracker.delete_status("a").await;
        assert_eq!(tracker.waiting_count(), 1);
    }

    /// The download histogram registers and exports whether or not anything ever
    /// observes into it, so an empty series is the deceptive failure mode: the
    /// scrape looks healthy and simply reports zero downloads. This drives the
    /// real task to its terminal status and checks a sample landed.
    ///
    /// `finish_download_claim` returns `Ok(false)` on purpose -- this replica was
    /// fenced. The bytes were still fetched, so the observation has to happen
    /// before the fence check; returning `Ok(true)` here would leave that ordering
    /// unpinned.
    ///
    /// No network: the NGC provider rejects a model name that is not
    /// `org/name/version` before it opens a client or touches the filesystem, so
    /// the download future resolves to `Err` immediately and the task takes the
    /// `ERROR` outcome.
    #[tokio::test]
    async fn a_finished_download_is_observed_into_the_histogram_even_when_fenced() {
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let done_tx = Mutex::new(Some(done_tx));

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_finish_download_claim()
            .once()
            .returning(move |_, _, _, status, _| {
                assert_eq!(status, ModelStatus::ERROR);
                if let Ok(mut slot) = done_tx.lock()
                    && let Some(tx) = slot.take()
                {
                    let _ = tx.send(());
                }
                // Fenced: someone else owns the claim now.
                Ok(false)
            });

        let (tracker, metrics_registry) = tracker_with_mock_and_registry(mock);
        tracker.spawn_download_task(
            DownloadTarget {
                model_name: "not-a-valid-ngc-artifact".to_string(),
                provider: ModelProvider::Ngc,
                revision: None,
                ignore_weights: false,
            },
            false,
            "claim-1".to_string(),
        );
        done_rx.await.expect("the download task reaches the finish");

        let encoded = encoded_metrics(&metrics_registry);
        assert!(
            encoded.contains(r#"mx_download_seconds_count{outcome="error"} 1"#),
            "a failed download must be observed under its terminal status: {encoded}"
        );
    }

    #[tokio::test]
    async fn test_tracker_error_status_clears_waiters() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_set_status()
            .once()
            .returning(|_, _, _, _| Ok(()));
        let tracker = tracker_with_mock(mock);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("m", tx);
        tracker
            .set_status_and_notify(
                "m".to_string(),
                ModelStatus::ERROR,
                ModelProvider::HuggingFace,
                Some("fail".to_string()),
            )
            .await;
        let waiters = tracker
            .waiting_channels
            .lock()
            .expect("waiters lock")
            .get("m")
            .map_or(0, std::vec::Vec::len);
        assert_eq!(waiters, 0, "ERROR is terminal, waiters must be cleared");
    }

    #[tokio::test]
    async fn test_tracker_downloading_status_keeps_waiters() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_set_status()
            .once()
            .returning(|_, _, _, _| Ok(()));
        let tracker = tracker_with_mock(mock);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("m", tx);
        tracker
            .set_status_and_notify(
                "m".to_string(),
                ModelStatus::DOWNLOADING,
                ModelProvider::HuggingFace,
                None,
            )
            .await;
        let waiters = tracker
            .waiting_channels
            .lock()
            .expect("waiters lock")
            .get("m")
            .map_or(0, std::vec::Vec::len);
        assert_eq!(
            waiters, 1,
            "DOWNLOADING is non-terminal, waiter must remain"
        );
    }

    #[tokio::test]
    async fn test_tracker_set_status_swallows_backend_error() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_set_status()
            .once()
            .returning(|_, _, _, _| Err("redis down".into()));
        let tracker = tracker_with_mock(mock);
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        tracker.add_waiting_channel("m", tx);
        // Error is logged but set_status_and_notify returns ()
        tracker
            .set_status_and_notify(
                "m".to_string(),
                ModelStatus::DOWNLOADED,
                ModelProvider::HuggingFace,
                None,
            )
            .await;
        // Nothing should be notified on the channel because set_status failed early.
        assert!(
            rx.try_recv().is_err(),
            "waiter shouldn't receive on backend error"
        );
    }

    /// Model service for the file-serving tests, which don't touch the tracker. A
    /// no-expectation mock backend keeps them off the `memory-backend` feature.
    fn test_model_service() -> ModelServiceImpl {
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(
            crate::registry::backend::MockRegistryBackend::new(),
        )));
        ModelServiceImpl::new(Arc::new(ModelDownloadTracker::new(
            registry,
            DownloadMetrics::register(&mut crate::metrics::new_registry()),
            RegistryMetrics::register(&mut crate::metrics::new_registry()),
        )))
    }

    #[test]
    fn test_collect_model_files_empty_dir() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let files = collect_model_files(temp_dir.path(), temp_dir.path(), None, false);
        assert!(files.is_empty());
    }

    #[test]
    fn test_collect_model_files_with_files() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create some test files
        let file1_path = temp_dir.path().join("config.json");
        std::fs::write(&file1_path, r#"{"test": "data"}"#).expect("Failed to write file1");

        let file2_path = temp_dir.path().join("model.bin");
        std::fs::write(&file2_path, vec![0u8; 100]).expect("Failed to write file2");

        let files = collect_model_files(temp_dir.path(), temp_dir.path(), None, false);

        assert_eq!(files.len(), 2);

        // Check file sizes
        let total_size: u64 = files.iter().map(|(_, size)| size).sum();
        assert!(total_size > 0);

        // Check that relative paths are correct
        let paths: Vec<_> = files
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();
        assert!(paths.contains(&"config.json".to_string()));
        assert!(paths.contains(&"model.bin".to_string()));
    }

    #[test]
    fn test_collect_model_files_ignore_weights() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Mix of non-weight and weight files, including a nested weight shard.
        std::fs::write(temp_dir.path().join("config.json"), "{}").expect("write config");
        std::fs::write(temp_dir.path().join("tokenizer.json"), "{}").expect("write tokenizer");
        std::fs::write(temp_dir.path().join("model.safetensors"), vec![0u8; 10])
            .expect("write safetensors");
        std::fs::write(temp_dir.path().join("pytorch_model.bin"), vec![0u8; 10])
            .expect("write bin");
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).expect("create subdir");
        std::fs::write(
            subdir.join("model-00001-of-00002.safetensors"),
            vec![0u8; 10],
        )
        .expect("write nested shard");

        // ignore_weights = false streams everything.
        let all = collect_model_files(temp_dir.path(), temp_dir.path(), None, false);
        assert_eq!(all.len(), 5);

        // ignore_weights = true keeps only the non-weight files (config + tokenizer),
        // dropping weight shards at any depth.
        let no_weights = collect_model_files(temp_dir.path(), temp_dir.path(), None, true);
        let names: Vec<String> = no_weights
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();
        assert_eq!(
            no_weights.len(),
            2,
            "expected only non-weight files: {names:?}"
        );
        assert!(names.contains(&"config.json".to_string()));
        assert!(names.contains(&"tokenizer.json".to_string()));
        assert!(
            names
                .iter()
                .all(|n| !n.ends_with(".safetensors") && !n.ends_with(".bin")),
            "weight files must be excluded: {names:?}"
        );
    }

    #[test]
    fn test_collect_model_files_nested() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create nested directory structure
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).expect("Failed to create subdir");

        let file1_path = temp_dir.path().join("root_file.txt");
        std::fs::write(&file1_path, "root content").expect("Failed to write file1");

        let file2_path = subdir.join("nested_file.txt");
        std::fs::write(&file2_path, "nested content").expect("Failed to write file2");

        let files = collect_model_files(temp_dir.path(), temp_dir.path(), None, false);

        assert_eq!(files.len(), 2);

        // Check that nested path is correct
        let paths: Vec<_> = files
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();
        assert!(paths.iter().any(|p| p.contains("nested_file")));
    }

    #[test]
    fn test_collect_model_files_with_selector_filters_exact_paths() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).expect("Failed to create subdir");
        std::fs::write(temp_dir.path().join("config.json"), "{}").expect("Failed to write config");
        std::fs::write(temp_dir.path().join("model.bin"), vec![0u8; 100])
            .expect("Failed to write model");
        std::fs::write(temp_dir.path().join("ignored.txt"), "ignore")
            .expect("Failed to write ignored");
        std::fs::write(subdir.join("nested.txt"), "nested").expect("Failed to write nested");

        let selector = ModelFileSelector {
            paths: vec!["config.json".to_string(), "subdir/nested.txt".to_string()],
        };
        let files = collect_model_files(temp_dir.path(), temp_dir.path(), Some(&selector), false);

        let mut paths: Vec<_> = files
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();
        paths.sort();
        assert_eq!(
            paths,
            vec!["config.json".to_string(), "subdir/nested.txt".to_string()]
        );
    }

    #[test]
    fn test_collect_model_files_with_selector_empty_and_nonmatching_paths() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("config.json"), "{}").expect("Failed to write config");

        let empty_selector = ModelFileSelector { paths: vec![] };
        assert!(
            collect_model_files(
                temp_dir.path(),
                temp_dir.path(),
                Some(&empty_selector),
                false
            )
            .is_empty()
        );

        let nonmatching_selector = ModelFileSelector {
            paths: vec!["missing.json".to_string(), "../config.json".to_string()],
        };
        assert!(
            collect_model_files(
                temp_dir.path(),
                temp_dir.path(),
                Some(&nonmatching_selector),
                false
            )
            .is_empty()
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_list_model_files_hf_honors_file_selector() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(model_dir.join("subdir")).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), br#"{"model":"test"}"#)
            .expect("Failed to write config");
        std::fs::write(model_dir.join("model.bin"), vec![0u8; 100]).expect("Failed to write model");
        std::fs::write(model_dir.join("subdir/nested.txt"), b"nested")
            .expect("Failed to write nested");

        let service = test_model_service();
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 0,
            file_selector: Some(ModelFileSelector {
                paths: vec!["config.json".to_string(), "subdir/nested.txt".to_string()],
            }),
            ignore_weights: false,
            revision: None,
        });

        let response = service
            .list_model_files(request)
            .await
            .expect("Expected file list")
            .into_inner();
        let mut paths: Vec<_> = response
            .files
            .iter()
            .map(|file| file.relative_path.clone())
            .collect();
        paths.sort();

        assert_eq!(
            paths,
            vec!["config.json".to_string(), "subdir/nested.txt".to_string()]
        );
        assert_eq!(
            response.total_size,
            br#"{"model":"test"}"#.len() as u64 + b"nested".len() as u64
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_model_files_present_reflects_disk_state() {
        let env_lock = acquire_env_mutex();
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path().to_path_buf();

        // No files on disk: a stale DOWNLOADED record must not be honored.
        assert!(
            !model_files_present(
                Some(cache_dir.clone()),
                "test/model",
                ModelProvider::HuggingFace,
                None
            )
            .await
        );

        // Once the snapshot exists, the cache hit is real.
        let model_dir = cache_dir.join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");
        assert!(
            model_files_present(
                Some(cache_dir),
                "test/model",
                ModelProvider::HuggingFace,
                None
            )
            .await
        );
    }

    #[tokio::test]
    async fn test_model_files_present_assumes_present_without_cache_dir() {
        // With no configured cache directory we cannot verify, so we must not force a
        // re-download loop: assume the files are present.
        assert!(model_files_present(None, "test/model", ModelProvider::HuggingFace, None).await);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_stream_model_files_hf_honors_file_selector() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), br#"{"model":"test"}"#)
            .expect("Failed to write config");
        std::fs::write(model_dir.join("model.bin"), vec![0u8; 100]).expect("Failed to write model");

        let service = test_model_service();
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
            file_selector: Some(ModelFileSelector {
                paths: vec!["config.json".to_string()],
            }),
            ignore_weights: false,
            revision: None,
        });

        let response = service
            .stream_model_files(request)
            .await
            .expect("Expected stream response");
        let chunks: Vec<_> = response
            .into_inner()
            .map(|chunk| chunk.expect("Expected chunk"))
            .collect()
            .await;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].relative_path, "config.json");
        assert_eq!(chunks[0].commit_hash.as_deref(), Some("abc123"));
        assert!(chunks[0].is_last_file);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_stream_model_files_serves_the_requested_revision() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        // Two revisions cached side by side. Without a requested revision the newest
        // snapshot wins, so asking for the older one proves the request is honored.
        let write_snapshot = |commit: &str, body: &[u8]| {
            let model_dir = temp_dir
                .path()
                .join("models--test--model/snapshots")
                .join(commit);
            std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
            std::fs::write(model_dir.join("config.json"), body).expect("Failed to write config");
        };
        write_snapshot("older111", b"old");
        // Snapshot ordering is by directory timestamp, so the two have to be distinct.
        tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;
        write_snapshot("newer222", b"new");

        let stream_revision = async |revision: Option<&str>| {
            let request = Request::new(ModelFilesRequest {
                model_name: "test/model".to_string(),
                provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
                chunk_size: 1024,
                file_selector: None,
                ignore_weights: false,
                revision: revision.map(String::from),
            });
            let chunks: Vec<_> = test_model_service()
                .stream_model_files(request)
                .await
                .expect("Expected stream response")
                .into_inner()
                .map(|chunk| chunk.expect("Expected chunk"))
                .collect()
                .await;
            assert_eq!(chunks.len(), 1);
            (chunks[0].commit_hash.clone(), chunks[0].data.clone())
        };

        assert_eq!(
            stream_revision(Some("older111")).await,
            (Some("older111".to_string()), b"old".to_vec())
        );
        assert_eq!(
            stream_revision(None).await,
            (Some("newer222".to_string()), b"new".to_vec()),
            "Without a revision the newest snapshot is served, so the pinned request above \
             really did select a different one"
        );
    }

    #[tokio::test]
    async fn test_ensure_model_downloaded_rejects_a_revision_on_a_provider_without_revisions() {
        let service = test_model_service();
        let request = Request::new(ModelDownloadRequest {
            model_name: "gs://test-bucket/org/model/rev-1".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::Gcs as i32,
            ignore_weights: false,
            revision: Some("v1.0".to_string()),
        });

        let status = service
            .ensure_model_downloaded(request)
            .await
            .map(|_| ())
            .expect_err("A provider without revisions must reject a pinned revision");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status
                .message()
                .contains("does not support pinned revisions"),
            "Unexpected message: {}",
            status.message()
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_stream_model_files_rejects_an_uncached_revision() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        let service = test_model_service();
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
            file_selector: None,
            ignore_weights: false,
            revision: Some("not-cached".to_string()),
        });

        let status = service
            .stream_model_files(request)
            .await
            .map(|_| ())
            .expect_err("An uncached revision must not fall back to another snapshot");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_stream_model_files_hf_returns_not_found_for_missing_selector_path() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), br#"{"model":"test"}"#)
            .expect("Failed to write config");

        let service = test_model_service();
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
            file_selector: Some(ModelFileSelector {
                paths: vec!["config.json".to_string(), "missing.json".to_string()],
            }),
            ignore_weights: false,
            revision: None,
        });

        let result = service.stream_model_files(request).await;
        let status = result.expect_err("Expected not found");
        assert_eq!(status.code(), tonic::Code::NotFound);
        assert_eq!(
            status.message(),
            "Selected file not found in model directory: missing.json"
        );
    }

    #[tokio::test]
    async fn test_list_model_files_not_found() {
        let service = test_model_service();

        let request = Request::new(ModelFilesRequest {
            model_name: "non-existent-model-12345".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 0,
            file_selector: None,
            ignore_weights: false,
            revision: None,
        });

        let result = service.list_model_files(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_stream_model_files_not_found() {
        let service = test_model_service();

        let request = Request::new(ModelFilesRequest {
            model_name: "non-existent-model-12345".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
            file_selector: None,
            ignore_weights: false,
            revision: None,
        });

        let result = service.stream_model_files(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_ensure_model_downloaded_rejects_invalid_provider() {
        let service = test_model_service();

        let request = Request::new(ModelDownloadRequest {
            model_name: "test/model".to_string(),
            provider: 99,
            ignore_weights: false,
            revision: None,
        });

        let result = service.ensure_model_downloaded(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("Invalid provider value"));
    }

    #[tokio::test]
    async fn test_list_model_files_rejects_invalid_provider() {
        let service = test_model_service();

        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: 99,
            chunk_size: 0,
            file_selector: None,
            ignore_weights: false,
            revision: None,
        });

        let result = service.list_model_files(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("Invalid provider value"));
    }

    #[tokio::test]
    async fn test_stream_model_files_rejects_invalid_provider() {
        let service = test_model_service();

        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: 99,
            chunk_size: 1024,
            file_selector: None,
            ignore_weights: false,
            revision: None,
        });

        let result = service.stream_model_files(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("Invalid provider value"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_stream_model_files_hf_first_chunk_includes_commit_hash() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), br#"{"model":"test"}"#)
            .expect("Failed to write model file");

        let service = test_model_service();
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
            file_selector: None,
            ignore_weights: false,
            revision: None,
        });

        let response = service
            .stream_model_files(request)
            .await
            .expect("Expected stream response");
        let mut stream = response.into_inner();
        let first_chunk = stream
            .next()
            .await
            .expect("Expected stream item")
            .expect("Expected first chunk");

        assert_eq!(first_chunk.relative_path, "config.json");
        assert_eq!(first_chunk.commit_hash.as_deref(), Some("abc123"));
        assert!(first_chunk.is_last_chunk);
        assert!(first_chunk.is_last_file);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_stream_model_files_hf_emits_chunk_for_zero_byte_file() {
        let env_lock = acquire_env_mutex();
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            temp_dir.path().to_str().expect("Expected temp dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("empty.bin"), []).expect("Failed to write empty file");

        let service = test_model_service();
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
            file_selector: None,
            ignore_weights: false,
            revision: None,
        });

        let response = service
            .stream_model_files(request)
            .await
            .expect("Expected stream response");
        let mut stream = response.into_inner();
        let first_chunk = stream
            .next()
            .await
            .expect("Expected stream item")
            .expect("Expected first chunk");

        assert_eq!(first_chunk.relative_path, "empty.bin");
        assert_eq!(first_chunk.total_size, 0);
        assert_eq!(first_chunk.data.len(), 0);
        assert_eq!(first_chunk.offset, 0);
        assert_eq!(first_chunk.commit_hash.as_deref(), Some("abc123"));
        assert!(first_chunk.is_last_chunk);
        assert!(first_chunk.is_last_file);
    }
}
