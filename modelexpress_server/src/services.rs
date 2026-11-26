// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::metrics::grpc::RpcOutcome;
use crate::metrics::registry::{DownloadMetrics, RegistryMetrics, StatusLabel};
use crate::registry::backend::ClaimOutcome;
use crate::registry::cache_cleanup::{
    execute_cleanup_plan, plan_cleanup_for_model, record_belongs_to_model,
};
use crate::registry::entry_key::EntryKey;
use crate::registry::state::RegistryManager;
use modelexpress_common::{
    cache::resolve_model_path,
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

/// Returns true if the model's files for the requested weight mode are present in the
/// given cache directory. Used to guard against stale `DOWNLOADED` registry records that
/// point at a cache entry which no longer exists on disk.
async fn model_files_present(
    cache_dir: &Path,
    model_name: &str,
    provider: ModelProvider,
    revision: Option<&str>,
    ignore_weights: bool,
) -> bool {
    match download::get_provider(provider)
        .has_cached_files(
            model_name,
            cache_dir.to_path_buf(),
            revision,
            ignore_weights,
        )
        .await
    {
        Ok(present) => present,
        Err(err) => {
            warn!("Failed to verify cached files for '{model_name}': {err}");
            false
        }
    }
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
            Some(self.tracker.cache_directory().to_path_buf()),
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
            // the retry completes.
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
        let cache_dir = self.tracker.cache_directory().to_path_buf();

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
        let cache_dir = self.tracker.cache_directory().to_path_buf();

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

        // Run the entire delete workflow (registry deletes and cache cleanup) inside a
        // detached task spawned before the first destructive registry call, so dropping
        // the gRPC request mid-delete cannot leave deleted records with orphaned files.
        let tracker = self.tracker.clone();
        let model_name_for_task = model_name.clone();
        let outcome = tokio::spawn(async move {
            tracker
                .delete_model_entries(provider, &model_name_for_task)
                .await
        })
        .await
        .map_err(|e| {
            error!("Delete task panicked for '{model_name}': {e}");
            Status::internal(format!("Delete task failed for '{model_name}'"))
        })?;

        tagged(
            DeleteModelResponse {
                success: outcome.success,
                message: Some(outcome.message),
            },
            outcome.rpc_outcome,
        )
    }
}

/// Outcome of clearing a model's registry entries and cached files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteModelOutcome {
    /// Whether the delete operation succeeded.
    pub success: bool,
    /// Human-readable status message returned in `DeleteModelResponse`.
    pub message: String,
    /// Metric classification for the gRPC response.
    pub rpc_outcome: RpcOutcome,
}

impl DeleteModelOutcome {
    fn backend_error(message: String) -> Self {
        Self {
            success: false,
            message,
            rpc_outcome: RpcOutcome::BackendError,
        }
    }

    fn cleanup_error(message: String) -> Self {
        Self {
            success: false,
            message,
            rpc_outcome: RpcOutcome::Error,
        }
    }
}

type WaitingModels = Arc<Mutex<HashMap<String, usize>>>;

fn lock_waiting_models(
    waiting_models: &WaitingModels,
) -> std::sync::MutexGuard<'_, HashMap<String, usize>> {
    match waiting_models.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            error!("Waiting models mutex is poisoned, recovering");
            poisoned.into_inner()
        }
    }
}

/// RAII guard tracking an active waiter for `mx_state_entries{map="download_waiters"}`.
struct WaitingModelGuard {
    waiting_models: WaitingModels,
    entry_key: String,
}

impl WaitingModelGuard {
    fn acquire(waiting_models: &WaitingModels, entry_key: &str) -> Self {
        let mut map = lock_waiting_models(waiting_models);
        let count = map.entry(entry_key.to_string()).or_insert(0);
        *count = count.saturating_add(1);
        Self {
            waiting_models: Arc::clone(waiting_models),
            entry_key: entry_key.to_string(),
        }
    }
}

impl Drop for WaitingModelGuard {
    fn drop(&mut self) {
        let mut map = lock_waiting_models(&self.waiting_models);
        if let Some(count) = map.get_mut(&self.entry_key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                map.remove(&self.entry_key);
            }
        }
    }
}

/// Metadata reported on a streamed `ModelStatusUpdate`.
struct UpdateIdentity<'a> {
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

/// Lease, poll, and fence timings for download ownership and deferred deletion.
#[derive(Clone, Copy, Debug)]
struct LeaseTimings {
    lease_duration: std::time::Duration,
    heartbeat_interval: std::time::Duration,
    self_fence_duration: std::time::Duration,
    deferred_delete_delay: std::time::Duration,
    poll_interval: std::time::Duration,
}

impl Default for LeaseTimings {
    fn default() -> Self {
        Self {
            lease_duration: DOWNLOAD_LEASE_DURATION,
            heartbeat_interval: DOWNLOAD_HEARTBEAT_INTERVAL,
            self_fence_duration: DOWNLOAD_SELF_FENCE_DURATION,
            deferred_delete_delay: DOWNLOAD_LEASE_DURATION,
            poll_interval: DOWNLOAD_POLL_INTERVAL,
        }
    }
}

#[cfg(test)]
type TestDownloadOverride = Arc<dyn Fn(&Path) -> Result<(), anyhow::Error> + Send + Sync>;

/// Tracks the status of model downloads through the distributed registry backend.
#[derive(Clone)]
pub struct ModelDownloadTracker {
    /// Distributed registry (Redis today, K8s CRDs in a follow-up).
    registry: Arc<RegistryManager>,
    /// Single resolved cache directory shared with file serving, deletion, and eviction.
    cache_directory: PathBuf,
    /// Active waiter counts keyed by registry entry key for `mx_state_entries`.
    waiting_models: WaitingModels,
    download_metrics: DownloadMetrics,
    registry_metrics: RegistryMetrics,
    lease_timings: LeaseTimings,
    #[cfg(test)]
    download_override: Option<TestDownloadOverride>,
}

pub(crate) const DOWNLOAD_LEASE_DURATION: std::time::Duration = std::time::Duration::from_secs(30);
const DOWNLOAD_HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
const DOWNLOAD_SELF_FENCE_DURATION: std::time::Duration = std::time::Duration::from_secs(25);
const DOWNLOAD_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

impl ModelDownloadTracker {
    pub fn new(
        registry: Arc<RegistryManager>,
        cache_directory: PathBuf,
        download_metrics: DownloadMetrics,
        registry_metrics: RegistryMetrics,
    ) -> Self {
        Self {
            registry,
            cache_directory,
            waiting_models: Arc::new(Mutex::new(HashMap::new())),
            download_metrics,
            registry_metrics,
            lease_timings: LeaseTimings::default(),
            #[cfg(test)]
            download_override: None,
        }
    }

    #[cfg(test)]
    fn with_lease_timings(mut self, lease_timings: LeaseTimings) -> Self {
        self.lease_timings = lease_timings;
        self
    }

    #[cfg(test)]
    fn with_download_override(mut self, download_override: TestDownloadOverride) -> Self {
        self.download_override = Some(download_override);
        self
    }

    /// Configured cache directory used by downloads, file serving, and deletion.
    #[must_use]
    pub fn cache_directory(&self) -> &Path {
        &self.cache_directory
    }

    /// Number of models with callers currently waiting on a download.
    ///
    /// This map is never evicted wholesale, so its size is the early warning for
    /// the leak that would eventually OOM the server. A poisoned lock reports the
    /// recovered length rather than failing: a metric read must not be able to
    /// take down the download path.
    pub fn waiting_count(&self) -> usize {
        lock_waiting_models(&self.waiting_models).len()
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

    /// Deletes every registry entry belonging to `(provider, model_name)` and
    /// removes its cached files when no surviving entry protects them.
    ///
    /// Fails closed on registry listing or record-deletion errors (never deleting
    /// files when registry state cannot be verified or updated). When any deleted
    /// record was `DOWNLOADING`, files are not removed during the RPC: non-Kubernetes
    /// backends spawn a detached cleanup task that waits one lease window and
    /// re-checks surviving records before removing files, while Kubernetes leaves
    /// the wait and retries to its finalizer controller.
    pub async fn delete_model_entries(
        &self,
        provider: ModelProvider,
        model_name: &str,
    ) -> DeleteModelOutcome {
        let records = match self.registry.get_models_by_last_used(None).await {
            Ok(records) => records,
            Err(err) => {
                error!("Failed to list registry records for '{model_name}': {err}");
                return DeleteModelOutcome::backend_error(format!(
                    "Failed to list registry records for '{model_name}': {err}"
                ));
            }
        };

        let matched: Vec<_> = records
            .into_iter()
            .filter(|record| record_belongs_to_model(record, provider, model_name))
            .collect();

        let mut removed: usize = 0;
        let mut had_downloading = false;
        for record in matched {
            match self.registry.delete_model(&record.model_name).await {
                Ok(()) => {
                    if record.status == ModelStatus::DOWNLOADING {
                        had_downloading = true;
                        self.registry_metrics
                            .record_transition(StatusLabel::Downloading, StatusLabel::Absent);
                    }
                    removed = removed.saturating_add(1);
                }
                Err(err) => {
                    error!(
                        "Failed to delete registry record '{}' for model '{model_name}': {err}",
                        record.model_name
                    );
                    return DeleteModelOutcome::backend_error(format!(
                        "Failed to delete registry record '{}' for model '{model_name}': {err}",
                        record.model_name
                    ));
                }
            }
        }

        info!("Deleted {removed} registry record(s) for model '{model_name}'");

        if had_downloading {
            if !self.registry.manages_deletion_cleanup() {
                let tracker = self.clone();
                let deferred_model = model_name.to_string();
                let delay = self.lease_timings.deferred_delete_delay;
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let live_records = match tracker.registry.get_models_by_last_used(None).await {
                        Ok(records) => records,
                        Err(err) => {
                            error!(
                                "Deferred cleanup for '{deferred_model}' failed to list live \
                                 registry records after lease window: {err}. Run 'model clear' \
                                 to reclaim leftover files."
                            );
                            return;
                        }
                    };
                    let plan = plan_cleanup_for_model(provider, &deferred_model, &live_records);
                    if let Err(err) = execute_cleanup_plan(&plan, &tracker.cache_directory).await {
                        error!(
                            "Deferred cleanup for '{deferred_model}' failed to remove cached \
                             files after lease window: {err}. Run 'model clear' to reclaim \
                             leftover files."
                        );
                    }
                });
            }

            return DeleteModelOutcome {
                success: true,
                message: format!(
                    "Removed {removed} registry record(s) for model '{model_name}'; a download \
                     was in flight and cached files will be removed after the lease window"
                ),
                rpc_outcome: RpcOutcome::Ok,
            };
        }

        let remaining_records = match self.registry.get_models_by_last_used(None).await {
            Ok(records) => records,
            Err(err) => {
                error!(
                    "Deleted {removed} registry record(s) for '{model_name}', but failed to list \
                     surviving records before file cleanup: {err}"
                );
                return DeleteModelOutcome::backend_error(format!(
                    "Failed to list surviving registry records for '{model_name}' after \
                     deleting {removed} record(s): {err}"
                ));
            }
        };

        let plan = plan_cleanup_for_model(provider, model_name, &remaining_records);
        if let Err(err) = execute_cleanup_plan(&plan, &self.cache_directory).await {
            error!(
                "Deleted {removed} registry record(s) for '{model_name}', but failed to remove \
                 cached files: {err}"
            );
            return DeleteModelOutcome::cleanup_error(format!(
                "Removed {removed} registry record(s) for '{model_name}', but failed to \
                 remove cached files: {err}"
            ));
        }

        DeleteModelOutcome {
            success: true,
            message: format!(
                "Model '{model_name}' removed from registry ({removed} record(s)) and cache"
            ),
            rpc_outcome: RpcOutcome::Ok,
        }
    }

    /// Drive `download` while renewing its registry lease and self-fencing before the
    /// lease expires.
    ///
    /// `claim_started_at` must be captured **before** sending the registry claim,
    /// takeover, ERROR-retry, or stale-reclaim request so a delayed reply or task
    /// scheduling stall cannot grant a fresh self-fence window after deletion cleanup.
    ///
    /// Returns `Some((result, lease_expiry))` if `download` completes before the
    /// self-fence deadline, or `None` if lease ownership is lost (`Ok(false)`), a
    /// renewal times out at the self-fence deadline, or `self_fence_duration` elapses
    /// without a successful renewal. Dropping the outer `download` future stops further
    /// polling on this task; it does not preempt detached subtasks (such as `hf-hub`
    /// chunk tasks) or already-running blocking filesystem calls.
    async fn run_download_with_lease_fence<F, T, E>(
        &self,
        entry_key: &str,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        claim_started_at: tokio::time::Instant,
        download: F,
    ) -> Option<(Result<T, E>, tokio::time::Instant)>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        let timings = self.lease_timings;
        let mut last_confirmed_at = claim_started_at;
        let mut lease_deadline = claim_started_at
            .checked_add(timings.self_fence_duration)
            .unwrap_or(claim_started_at);

        if tokio::time::Instant::now() >= lease_deadline {
            warn!(
                "Stopping download for {model_name}: claim already exceeded self-fence \
                 window ({:?}) before download started",
                timings.self_fence_duration
            );
            return None;
        }

        tokio::pin!(download);

        let fence_sleep = tokio::time::sleep_until(lease_deadline);
        tokio::pin!(fence_sleep);

        let first_tick = claim_started_at
            .checked_add(timings.heartbeat_interval)
            .unwrap_or(claim_started_at);
        let mut heartbeat = tokio::time::interval_at(first_tick, timings.heartbeat_interval);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                () = &mut fence_sleep => {
                    warn!(
                        "Stopping download for {model_name}: failed to renew download lease \
                         within {:?}",
                        timings.self_fence_duration
                    );
                    return None;
                }
                _ = heartbeat.tick() => {
                    let renewal_start = tokio::time::Instant::now();
                    let refresh = tokio::time::timeout_at(
                        lease_deadline,
                        self.registry.refresh_download_claim(
                            entry_key,
                            provider,
                            claim_id,
                            timings.lease_duration,
                        ),
                    )
                    .await;
                    match refresh {
                        Ok(Ok(true)) => {
                            last_confirmed_at = renewal_start;
                            if let Some(next_deadline) =
                                renewal_start.checked_add(timings.self_fence_duration)
                            {
                                lease_deadline = next_deadline;
                                fence_sleep.as_mut().reset(lease_deadline);
                            }
                        }
                        Ok(Ok(false)) => {
                            warn!(
                                "Stopping download for {model_name}: lease ownership was lost"
                            );
                            return None;
                        }
                        Ok(Err(e)) => {
                            warn!("Failed to refresh download lease for {model_name}: {e}");
                        }
                        Err(_) => {
                            warn!(
                                "Stopping download for {model_name}: lease renewal timed out at \
                                 self-fence deadline"
                            );
                            return None;
                        }
                    }
                }
                result = &mut download => {
                    let lease_expiry = last_confirmed_at
                        .checked_add(timings.lease_duration)
                        .unwrap_or(lease_deadline);
                    return Some((result, lease_expiry));
                }
            }
        }
    }

    /// Spawn a background task that downloads the model and finishes its registry claim
    /// on success or failure. Waiters observe the terminal status via registry polling;
    /// the background task does not push terminal notifications directly to streams.
    fn spawn_download_task(
        &self,
        target: DownloadTarget,
        retry: bool,
        claim_id: String,
        claim_started_at: tokio::time::Instant,
    ) {
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
            let cache_dir = tracker.cache_directory.clone();
            #[cfg(test)]
            let download_override = tracker.download_override.clone();
            let download = async {
                #[cfg(test)]
                if let Some(override_fn) = download_override {
                    return override_fn(&cache_dir);
                }
                download::download_model_revision(
                    &model_name,
                    provider,
                    Some(cache_dir),
                    ignore_weights,
                    revision.as_deref(),
                )
                .await
                .map(|_| ())
            };
            let Some((result, lease_expiry)) = tracker
                .run_download_with_lease_fence(
                    &entry_key,
                    &model_name,
                    provider,
                    &claim_id,
                    claim_started_at,
                    download,
                )
                .await
            else {
                return;
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

            match tokio::time::timeout_at(
                lease_expiry,
                tracker
                    .registry
                    .finish_download_claim(&entry_key, provider, &claim_id, status, message),
            )
            .await
            {
                Ok(Ok(true)) => {
                    debug!("Finished download claim for {model_name} with status {status:?}");
                }
                Ok(Ok(false)) => {
                    warn!("Ignoring completion for {model_name}: lease ownership was lost");
                }
                Ok(Err(e)) => {
                    error!("Failed to finish download claim for {model_name}: {e}");
                }
                Err(_) => {
                    warn!("Timed out finishing download claim for {model_name}: lease expired");
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
            model_name,
            provider,
            resolved_revision: target.revision.as_deref(),
        };

        // Atomically try to claim this model for download. The `ClaimOutcome` tells us
        // whether THIS replica won the claim or is observing someone else's claim —
        // status alone (`DOWNLOADING`) can't distinguish those cases across replicas.
        // A claim may report an existing `DOWNLOADED` record whose files no longer exist
        // on disk (e.g. after a client-side `model clear` that only removed local files).
        // When that happens we CAS `DOWNLOADED -> DOWNLOADING` in place instead of deleting
        // the record: on Kubernetes deleting a record starts finalizer cleanup and races
        // with re-create, and across replicas a delete can remove another replica's new
        // claim. Only the CAS winner spawns the download; losers re-check and join as
        // waiters. Bounded to avoid looping forever if the entry is terminating.
        const MAX_CLAIM_ATTEMPTS: usize = 3;
        let lease_duration = self.lease_timings.lease_duration;
        let mut claim_id = uuid::Uuid::new_v4().to_string();
        let mut attempt: usize = 0;
        let (status, initial_claim_started_at) = loop {
            attempt = attempt.saturating_add(1);
            let claim_started_at = tokio::time::Instant::now();
            match self
                .registry
                .try_claim_for_download(entry_key, provider, &claim_id, lease_duration)
                .await
            {
                // Both outcomes mean this replica owns the download; they differ
                // only in cost, which the metrics layer records separately.
                Ok(ClaimOutcome::Claimed | ClaimOutcome::TookOver) => {
                    break (ModelStatus::DOWNLOADING, Some(claim_started_at));
                }
                Ok(ClaimOutcome::AlreadyExists(existing)) => {
                    if existing == ModelStatus::DOWNLOADED {
                        if model_files_present(
                            &self.cache_directory,
                            model_name,
                            provider,
                            target.revision.as_deref(),
                            target.ignore_weights,
                        )
                        .await
                        {
                            // Returning an existing downloaded model is a cache hit for LRU purposes.
                            self.touch_and_log(entry_key).await;
                            break (ModelStatus::DOWNLOADED, None);
                        }

                        warn!(
                            "Registry reports model '{model_name}' as DOWNLOADED but its files \
                             are missing from the cache; reclaiming entry in place"
                        );
                        let reclaim_started_at = tokio::time::Instant::now();
                        match self
                            .registry
                            .try_reclaim_downloaded(entry_key, provider, &claim_id, lease_duration)
                            .await
                        {
                            Ok(true) => break (ModelStatus::DOWNLOADING, Some(reclaim_started_at)),
                            Ok(false) => {
                                if attempt < MAX_CLAIM_ATTEMPTS {
                                    continue;
                                }
                                error!(
                                    "Stale DOWNLOADED record for '{model_name}' could not be \
                                     reclaimed after {MAX_CLAIM_ATTEMPTS} attempts"
                                );
                                let _ = tx
                                    .send(Ok(identity.status_update(
                                        ModelStatus::ERROR,
                                        Some(
                                            "Model files are missing from cache and stale entry \
                                             could not be reclaimed"
                                                .to_string(),
                                        ),
                                    )))
                                    .await;
                                return ModelStatus::ERROR;
                            }
                            Err(e) => {
                                error!(
                                    "Failed to reclaim stale DOWNLOADED entry for '{model_name}': {e}"
                                );
                                let _ = tx
                                    .send(Ok(identity.status_update(
                                        ModelStatus::ERROR,
                                        Some(
                                            "Registry error occurred while reclaiming stale entry"
                                                .to_string(),
                                        ),
                                    )))
                                    .await;
                                return ModelStatus::ERROR;
                            }
                        }
                    }
                    break (existing, None);
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
        let (effective_status, retry_claim_started_at) = if status == ModelStatus::ERROR {
            let retry_started_at = tokio::time::Instant::now();
            let won = match self
                .registry
                .try_reset_error_for_retry(entry_key, provider, &claim_id, lease_duration)
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
            (
                ModelStatus::DOWNLOADING,
                if won { Some(retry_started_at) } else { None },
            )
        } else {
            (status, None)
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
            let _waiting_guard = WaitingModelGuard::acquire(&self.waiting_models, entry_key);

            // Spawn the download only on the replica that won the claim (fresh
            // download) or won the ERROR-retry CAS. Every request observes
            // completion via the registry poll loop below.
            if let Some(claim_started_at) = initial_claim_started_at.or(retry_claim_started_at) {
                let retry = status == ModelStatus::ERROR;
                self.spawn_download_task(target.clone(), retry, claim_id, claim_started_at);
                claim_id = uuid::Uuid::new_v4().to_string();
            }

            let mut wait_reclaim_attempts: usize = 0;
            loop {
                tokio::select! {
                    biased;
                    () = tx.closed() => {
                        debug!("Client disconnected while waiting for download of '{model_name}'");
                        return ModelStatus::DOWNLOADING;
                    }
                    () = tokio::time::sleep(self.lease_timings.poll_interval) => {}
                }

                let poll_claim_started_at = tokio::time::Instant::now();
                match self
                    .registry
                    .try_claim_for_download(entry_key, provider, &claim_id, lease_duration)
                    .await
                {
                    // A taken-over lease must be re-driven exactly like a fresh
                    // claim. Letting it fall through instead would leave the entry
                    // wedged in DOWNLOADING with no owner running.
                    Ok(ClaimOutcome::Claimed | ClaimOutcome::TookOver) => {
                        self.spawn_download_task(
                            target.clone(),
                            false,
                            claim_id,
                            poll_claim_started_at,
                        );
                        claim_id = uuid::Uuid::new_v4().to_string();
                    }
                    Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED)) => {
                        if model_files_present(
                            &self.cache_directory,
                            model_name,
                            provider,
                            target.revision.as_deref(),
                            target.ignore_weights,
                        )
                        .await
                        {
                            return ModelStatus::DOWNLOADED;
                        }

                        wait_reclaim_attempts = wait_reclaim_attempts.saturating_add(1);
                        if wait_reclaim_attempts > MAX_CLAIM_ATTEMPTS {
                            error!(
                                "Wait loop observed stale DOWNLOADED record for '{model_name}' \
                                 after {MAX_CLAIM_ATTEMPTS} reclaim attempts"
                            );
                            let _ = tx
                                .send(Ok(identity.status_update(
                                    ModelStatus::ERROR,
                                    Some(
                                        "Model files are missing from cache and stale entry \
                                         could not be reclaimed"
                                            .to_string(),
                                    ),
                                )))
                                .await;
                            return ModelStatus::ERROR;
                        }

                        warn!(
                            "Wait loop observed DOWNLOADED for '{model_name}' but its files are \
                             missing from the cache; reclaiming entry in place \
                             (attempt {wait_reclaim_attempts}/{MAX_CLAIM_ATTEMPTS})"
                        );
                        let wait_reclaim_started_at = tokio::time::Instant::now();
                        match self
                            .registry
                            .try_reclaim_downloaded(entry_key, provider, &claim_id, lease_duration)
                            .await
                        {
                            Ok(true) => {
                                self.spawn_download_task(
                                    target.clone(),
                                    false,
                                    claim_id,
                                    wait_reclaim_started_at,
                                );
                                claim_id = uuid::Uuid::new_v4().to_string();
                            }
                            Ok(false) => {
                                if wait_reclaim_attempts >= MAX_CLAIM_ATTEMPTS {
                                    error!(
                                        "Wait loop could not reclaim stale DOWNLOADED record for \
                                         '{model_name}' after {MAX_CLAIM_ATTEMPTS} attempts"
                                    );
                                    let _ = tx
                                        .send(Ok(identity.status_update(
                                            ModelStatus::ERROR,
                                            Some(
                                                "Model files are missing from cache and stale \
                                                 entry could not be reclaimed"
                                                    .to_string(),
                                            ),
                                        )))
                                        .await;
                                    return ModelStatus::ERROR;
                                }
                            }
                            Err(e) => {
                                error!(
                                    "Failed to reclaim stale DOWNLOADED entry for '{model_name}' \
                                     in wait loop: {e}"
                                );
                                let _ = tx
                                    .send(Ok(identity.status_update(
                                        ModelStatus::ERROR,
                                        Some(
                                            "Registry error occurred while reclaiming stale entry"
                                                .to_string(),
                                        ),
                                    )))
                                    .await;
                                return ModelStatus::ERROR;
                            }
                        }
                    }
                    Ok(ClaimOutcome::AlreadyExists(current_status))
                        if current_status != ModelStatus::DOWNLOADING =>
                    {
                        return current_status;
                    }
                    Ok(ClaimOutcome::AlreadyExists(_)) => {}
                    Err(e) => {
                        warn!("Failed to poll download lease for {model_name}: {e}");
                    }
                }
            }
        }

        effective_status
    }
}

#[cfg(test)]
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
    ) -> (ModelDownloadTracker, TempDir) {
        let cache_dir = TempDir::new().expect("Failed to create isolated test cache dir");
        let tracker = tracker_with_mock_and_cache_dir(mock, cache_dir.path().to_path_buf());
        (tracker, cache_dir)
    }

    fn tracker_with_mock_and_cache_dir(
        mock: crate::registry::backend::MockRegistryBackend,
        cache_dir: PathBuf,
    ) -> ModelDownloadTracker {
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(mock)));
        ModelDownloadTracker::new(
            registry,
            cache_dir,
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
    ) -> (
        ModelDownloadTracker,
        prometheus_client::registry::Registry,
        TempDir,
    ) {
        let mut metrics_registry = crate::metrics::new_registry();
        let download_metrics = DownloadMetrics::register(&mut metrics_registry);
        let registry_metrics = RegistryMetrics::register(&mut metrics_registry);
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(mock)));
        let cache_dir = TempDir::new().expect("Failed to create isolated test cache dir");
        (
            ModelDownloadTracker::new(
                registry,
                cache_dir.path().to_path_buf(),
                download_metrics,
                registry_metrics,
            ),
            metrics_registry,
            cache_dir,
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
        let (tracker, _cache_dir) = tracker_with_mock(mock);
        assert!(tracker.get_status("unknown").await.is_none());
    }

    #[tokio::test]
    async fn test_tracker_get_status_hit_bumps_last_used_at() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_status()
            .once()
            .returning(|_| Ok(Some(ModelStatus::DOWNLOADED)));
        mock.expect_touch_model().once().returning(|_| Ok(()));
        let (tracker, _cache_dir) = tracker_with_mock(mock);
        assert_eq!(tracker.get_status("m").await, Some(ModelStatus::DOWNLOADED));
    }

    #[tokio::test]
    async fn test_tracker_downloaded_cache_hit_bumps_last_used_at() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
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
        let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
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
    async fn assert_claims_on_entry_key(target: DownloadTarget, cache_dir: PathBuf) {
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

        let tracker = tracker_with_mock_and_cache_dir(mock, cache_dir);
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
    async fn test_tracker_scopes_the_lease_to_the_resolved_revision() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        assert_claims_on_entry_key(
            DownloadTarget {
                revision: Some("abc123".to_string()),
                ..test_target("test/model")
            },
            temp_dir.path().to_path_buf(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_tracker_separates_metadata_only_from_full_weight_downloads() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        // A metadata-only entry must not be the same registry record as a full-weight
        // one, or a weightless snapshot would satisfy a request that needs weights.
        assert_claims_on_entry_key(
            DownloadTarget {
                ignore_weights: true,
                ..test_target("test/model")
            },
            temp_dir.path().to_path_buf(),
        )
        .await;
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

        let (tracker, metrics_registry, _cache_dir) = tracker_with_mock_and_registry(mock);
        let outcome = tracker
            .delete_model_entries(ModelProvider::HuggingFace, "m")
            .await;
        assert!(outcome.success);
        assert!(
            outcome.message.contains("Removed 1 registry record(s)")
                && outcome.message.contains("lease window"),
            "unexpected message: {}",
            outcome.message
        );
        assert_eq!(outcome.rpc_outcome, RpcOutcome::Ok);

        let encoded = encoded_metrics(&metrics_registry);
        assert_eq!(
            departures_from_downloading(&encoded),
            1,
            "a deleted download must not stay counted as in flight: {encoded}"
        );
    }

    /// `delete_model` failure fails closed and must not book a departure or delete files.
    #[tokio::test]
    async fn a_failed_delete_books_no_departure() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .once()
            .returning(|_| Ok(vec![record_with_status("m", ModelStatus::DOWNLOADING)]));
        mock.expect_delete_model()
            .once()
            .returning(|_| Err("backend unreachable".into()));

        let (tracker, metrics_registry, _cache_dir) = tracker_with_mock_and_registry(mock);
        let outcome = tracker
            .delete_model_entries(ModelProvider::HuggingFace, "m")
            .await;
        assert!(!outcome.success);
        assert_eq!(outcome.rpc_outcome, RpcOutcome::BackendError);

        let encoded = encoded_metrics(&metrics_registry);
        assert_eq!(
            departures_from_downloading(&encoded),
            0,
            "a failed delete must not book a departure: {encoded}"
        );
    }

    /// Ordinary cache eviction clears `DOWNLOADED` entries. Those never occupied
    /// the in-flight level, so counting them as departures would drive the
    /// derivation negative on the most common delete there is.
    #[tokio::test]
    async fn deleting_a_downloaded_entry_books_no_departure() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        let list_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let list_calls_clone = Arc::clone(&list_calls);
        mock.expect_get_models_by_last_used()
            .times(2)
            .returning(move |_| {
                if list_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(vec![record_with_status("m", ModelStatus::DOWNLOADED)])
                } else {
                    Ok(Vec::new())
                }
            });
        mock.expect_delete_model().once().returning(|_| Ok(()));

        let (tracker, metrics_registry, _cache_dir) = tracker_with_mock_and_registry(mock);
        let outcome = tracker
            .delete_model_entries(ModelProvider::HuggingFace, "m")
            .await;
        assert!(outcome.success);
        assert!(
            outcome.message.contains("1 record(s)"),
            "unexpected message: {}",
            outcome.message
        );

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
        let mock = crate::registry::backend::MockRegistryBackend::new();
        let (tracker, _cache_dir) = tracker_with_mock(mock);
        assert_eq!(tracker.waiting_count(), 0);

        let waiter_a = WaitingModelGuard::acquire(&tracker.waiting_models, "a");
        let waiter_b = WaitingModelGuard::acquire(&tracker.waiting_models, "b");
        // Keyed by entry key, so a second waiter on "a" does not add an entry.
        let waiter_a2 = WaitingModelGuard::acquire(&tracker.waiting_models, "a");
        assert_eq!(tracker.waiting_count(), 2);

        drop(waiter_a);
        assert_eq!(tracker.waiting_count(), 2);

        drop(waiter_a2);
        assert_eq!(tracker.waiting_count(), 1);

        drop(waiter_b);
        assert_eq!(tracker.waiting_count(), 0);
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

        let (tracker, metrics_registry, _cache_dir) = tracker_with_mock_and_registry(mock);
        tracker.spawn_download_task(
            DownloadTarget {
                model_name: "not-a-valid-ngc-artifact".to_string(),
                provider: ModelProvider::Ngc,
                revision: None,
                ignore_weights: false,
            },
            false,
            "claim-1".to_string(),
            tokio::time::Instant::now(),
        );
        done_rx.await.expect("the download task reaches the finish");

        let encoded = encoded_metrics(&metrics_registry);
        assert!(
            encoded.contains(r#"mx_download_seconds_count{outcome="error"} 1"#),
            "a failed download must be observed under its terminal status: {encoded}"
        );
    }

    /// Model service for the file-serving tests, which don't touch the tracker. A
    /// no-expectation mock backend keeps them off the `memory-backend` feature.
    fn test_model_service() -> (ModelServiceImpl, TempDir) {
        let cache_dir = TempDir::new().expect("Failed to create isolated test cache dir");
        let service = test_model_service_with_cache_dir(cache_dir.path().to_path_buf());
        (service, cache_dir)
    }

    fn test_model_service_with_cache_dir(cache_dir: PathBuf) -> ModelServiceImpl {
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(
            crate::registry::backend::MockRegistryBackend::new(),
        )));
        ModelServiceImpl::new(Arc::new(ModelDownloadTracker::new(
            registry,
            cache_dir,
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

        let service = test_model_service_with_cache_dir(temp_dir.path().to_path_buf());
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
                &cache_dir,
                "test/model",
                ModelProvider::HuggingFace,
                None,
                false,
            )
            .await
        );

        // Once the snapshot exists, the cache hit is real.
        let model_dir = cache_dir.join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");
        assert!(
            model_files_present(
                &cache_dir,
                "test/model",
                ModelProvider::HuggingFace,
                None,
                false,
            )
            .await
        );
    }

    #[tokio::test]
    async fn test_tracker_stale_downloaded_reclaims_in_place_and_never_serves_missing_files() {
        // Case 1: Loser of stale reclaim sees another replica's DOWNLOADING on re-check,
        // joins as a waiter, and returns DOWNLOADED only after files are present on disk.
        {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let snapshot_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            let claim_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let claim_calls_clone = std::sync::Arc::clone(&claim_calls);
            mock.expect_try_claim_for_download()
                .times(3)
                .returning(move |_, _, _, _| {
                    match claim_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                        0 => Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED)),
                        1 => Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)),
                        _ => {
                            std::fs::create_dir_all(&snapshot_dir)
                                .expect("Failed to create snapshot dir");
                            std::fs::write(snapshot_dir.join("config.json"), b"{}")
                                .expect("Failed to write config");
                            Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED))
                        }
                    }
                });
            mock.expect_try_reclaim_downloaded()
                .once()
                .returning(|_, _, _, _| Ok(false));
            mock.expect_delete_model().times(0);

            let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let status = tracker
                .ensure_model_downloaded(&test_target("test/model"), &tx)
                .await;
            assert_eq!(status, ModelStatus::DOWNLOADED);
            let first = rx.recv().await.expect("initial update").expect("ok update");
            assert_eq!(
                first.status,
                modelexpress_common::grpc::model::ModelStatus::Downloading as i32
            );
        }

        // Case 2: If reclaim keeps returning false (e.g. terminating CR), never return
        // DOWNLOADED while files are missing; fail cleanly with ERROR and never delete.
        {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_try_claim_for_download()
                .times(3)
                .returning(|_, _, _, _| Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED)));
            mock.expect_try_reclaim_downloaded()
                .times(3)
                .returning(|_, _, _, _| Ok(false));
            mock.expect_delete_model().times(0);

            let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let status = tracker
                .ensure_model_downloaded(&test_target("test/model"), &tx)
                .await;
            assert_eq!(status, ModelStatus::ERROR);
            let update = rx.recv().await.expect("error update").expect("ok update");
            assert_eq!(
                update.status,
                modelexpress_common::grpc::model::ModelStatus::Error as i32
            );
        }

        // Case 3: Wait loop observes DOWNLOADED while cache stays empty; it must attempt
        // in-place reclaim and return bounded ERROR instead of a false DOWNLOADED hit.
        {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            let claim_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let claim_calls_clone = std::sync::Arc::clone(&claim_calls);
            mock.expect_try_claim_for_download()
                .times(4)
                .returning(move |_, _, _, _| {
                    match claim_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                        0 => Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)),
                        _ => Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED)),
                    }
                });
            mock.expect_try_reclaim_downloaded()
                .times(3)
                .returning(|_, _, _, _| Ok(false));
            mock.expect_delete_model().times(0);

            let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
            let (tx, mut rx) = tokio::sync::mpsc::channel(8);
            let status = tracker
                .ensure_model_downloaded(&test_target("test/model"), &tx)
                .await;
            assert_eq!(status, ModelStatus::ERROR);

            let first = rx.recv().await.expect("initial update").expect("ok update");
            assert_eq!(
                first.status,
                modelexpress_common::grpc::model::ModelStatus::Downloading as i32
            );
            let second = rx.recv().await.expect("error update").expect("ok update");
            assert_eq!(
                second.status,
                modelexpress_common::grpc::model::ModelStatus::Error as i32
            );
        }
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

        let service = test_model_service_with_cache_dir(temp_dir.path().to_path_buf());
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

        let cache_path = temp_dir.path().to_path_buf();
        let stream_revision = async |revision: Option<&str>| {
            let request = Request::new(ModelFilesRequest {
                model_name: "test/model".to_string(),
                provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
                chunk_size: 1024,
                file_selector: None,
                ignore_weights: false,
                revision: revision.map(String::from),
            });
            let chunks: Vec<_> = test_model_service_with_cache_dir(cache_path.clone())
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
        let (service, _cache_dir) = test_model_service();
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
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        let service = test_model_service_with_cache_dir(temp_dir.path().to_path_buf());
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
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), br#"{"model":"test"}"#)
            .expect("Failed to write config");

        let service = test_model_service_with_cache_dir(temp_dir.path().to_path_buf());
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
        let (service, _cache_dir) = test_model_service();

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
        let (service, _cache_dir) = test_model_service();

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
        let (service, _cache_dir) = test_model_service();

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
        let (service, _cache_dir) = test_model_service();

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
        let (service, _cache_dir) = test_model_service();

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
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), br#"{"model":"test"}"#)
            .expect("Failed to write model file");

        let service = test_model_service_with_cache_dir(temp_dir.path().to_path_buf());
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
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = temp_dir.path().join("models--test--model/snapshots/abc123");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("empty.bin"), []).expect("Failed to write empty file");

        let service = test_model_service_with_cache_dir(temp_dir.path().to_path_buf());
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

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_model_service_uses_injected_cache_directory_over_env() {
        let env_lock = acquire_env_mutex();
        let configured_dir = TempDir::new().expect("Failed to create configured dir");
        let env_dir = TempDir::new().expect("Failed to create env dir");
        let _cache_dir_guard = EnvVarGuard::set(
            &env_lock,
            "MODEL_EXPRESS_CACHE_DIRECTORY",
            env_dir.path().to_str().expect("Expected env dir path"),
        );
        let _offline_guard = EnvVarGuard::set(&env_lock, "HF_HUB_OFFLINE", "1");

        let model_dir = configured_dir
            .path()
            .join("models--org--injected/snapshots/deadbeef");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("Failed to write config");

        let tracker = tracker_with_mock_and_cache_dir(
            crate::registry::backend::MockRegistryBackend::new(),
            configured_dir.path().to_path_buf(),
        );
        assert_eq!(tracker.cache_directory(), configured_dir.path());

        let service = test_model_service_with_cache_dir(configured_dir.path().to_path_buf());
        let response = service
            .list_model_files(Request::new(ModelFilesRequest {
                model_name: "org/injected".to_string(),
                provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
                chunk_size: 1024,
                file_selector: None,
                ignore_weights: false,
                revision: None,
            }))
            .await
            .expect("list_model_files should read from injected cache_directory")
            .into_inner();
        assert_eq!(response.files.len(), 1);
        assert_eq!(response.files[0].relative_path, "config.json");
    }

    fn record_with_provider(
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
    ) -> crate::registry::backend::ModelRecord {
        let now = chrono::Utc::now();
        crate::registry::backend::ModelRecord {
            model_name: model_name.to_string(),
            provider,
            status,
            created_at: now,
            last_used_at: now,
            message: None,
        }
    }

    fn model_service_with_tracker(tracker: ModelDownloadTracker) -> ModelServiceImpl {
        ModelServiceImpl::new(Arc::new(tracker))
    }

    /// Minimal async test backend for tests that need true `.await` suspension
    /// inside `delete_model` or `refresh_download_claim`.
    struct AsyncTestBackend {
        records: Arc<Mutex<Vec<crate::registry::backend::ModelRecord>>>,
        delete_entered: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        delete_release: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
        hang_refresh: bool,
    }

    #[tonic::async_trait]
    impl crate::registry::backend::RegistryBackend for AsyncTestBackend {
        async fn connect(&self) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn get_status(
            &self,
            _model_name: &str,
        ) -> crate::registry::backend::RegistryResult<Option<ModelStatus>> {
            Ok(None)
        }

        async fn get_model_record(
            &self,
            _model_name: &str,
        ) -> crate::registry::backend::RegistryResult<Option<crate::registry::backend::ModelRecord>>
        {
            Ok(None)
        }

        async fn set_status(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _status: ModelStatus,
            _message: Option<String>,
        ) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn touch_model(
            &self,
            _model_name: &str,
        ) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn delete_model(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<()> {
            self.records
                .lock()
                .expect("lock records")
                .retain(|record| record.model_name != model_name);
            if let Some(tx) = self.delete_entered.lock().expect("lock").take() {
                let _ = tx.send(());
            }
            if let Some(rx) = self.delete_release.lock().await.take() {
                let _ = rx.await;
            }
            Ok(())
        }

        async fn get_models_by_last_used(
            &self,
            _limit: Option<u32>,
        ) -> crate::registry::backend::RegistryResult<Vec<crate::registry::backend::ModelRecord>>
        {
            Ok(self.records.lock().expect("lock records").clone())
        }

        async fn get_status_counts(
            &self,
        ) -> crate::registry::backend::RegistryResult<(u32, u32, u32)> {
            Ok((0, 0, 0))
        }

        async fn try_claim_for_download(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<ClaimOutcome> {
            Ok(ClaimOutcome::Claimed)
        }

        async fn try_reset_error_for_retry(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(false)
        }

        async fn try_reclaim_downloaded(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(false)
        }

        async fn refresh_download_claim(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            if self.hang_refresh {
                std::future::pending::<()>().await;
            }
            Ok(true)
        }

        async fn finish_download_claim(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _status: ModelStatus,
            _message: Option<String>,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(true)
        }
    }

    #[tokio::test]
    async fn test_delete_model_matches_raw_and_canonical_ngc_spellings_and_removes_files() {
        let temp_dir = TempDir::new().expect("create cache dir");
        let ngc_dir = temp_dir.path().join("ngc/nim/nvidia/models/llama/1");
        std::fs::create_dir_all(&ngc_dir).expect("create ngc dir");
        std::fs::write(ngc_dir.join("config.json"), b"{}").expect("write ngc config");

        let raw_key = "nim/nvidia/llama/1".to_string();
        let meta_uri_key = EntryKey::new("ngc://nim/nvidia/llama:1", None, true).to_string();
        let other_provider_key = "nim/nvidia/llama/1".to_string();

        let deleted_keys = Arc::new(Mutex::new(Vec::<String>::new()));
        let deleted_keys_clone = Arc::clone(&deleted_keys);
        let list_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let list_calls_clone = Arc::clone(&list_calls);

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        let raw_key_clone = raw_key.clone();
        let meta_uri_key_clone = meta_uri_key.clone();
        mock.expect_get_models_by_last_used()
            .times(2)
            .returning(move |_| {
                if list_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(vec![
                        record_with_provider(
                            &raw_key_clone,
                            ModelProvider::Ngc,
                            ModelStatus::DOWNLOADED,
                        ),
                        record_with_provider(
                            &meta_uri_key_clone,
                            ModelProvider::Ngc,
                            ModelStatus::DOWNLOADED,
                        ),
                        record_with_provider(
                            &other_provider_key,
                            ModelProvider::HuggingFace,
                            ModelStatus::DOWNLOADED,
                        ),
                    ])
                } else {
                    Ok(vec![record_with_provider(
                        "nim/nvidia/llama/1",
                        ModelProvider::HuggingFace,
                        ModelStatus::DOWNLOADED,
                    )])
                }
            });
        mock.expect_delete_model().times(2).returning(move |key| {
            deleted_keys_clone
                .lock()
                .expect("lock deleted_keys")
                .push(key.to_string());
            Ok(())
        });

        let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
        let service = model_service_with_tracker(tracker);
        let response = service
            .delete_model(Request::new(DeleteModelRequest {
                model_name: "ngc://nim/nvidia/llama:1".to_string(),
                provider: GrpcModelProvider::Ngc as i32,
            }))
            .await
            .expect("delete_model RPC should succeed");
        assert_eq!(
            response.extensions().get::<RpcOutcome>(),
            Some(&RpcOutcome::Ok)
        );
        assert!(response.into_inner().success);

        let mut deleted = deleted_keys.lock().expect("lock").clone();
        deleted.sort();
        let mut expected = vec![raw_key, meta_uri_key];
        expected.sort();
        assert_eq!(deleted, expected);
        assert!(
            !ngc_dir.exists(),
            "NGC model directory must be deleted after all spellings are removed"
        );
    }

    #[tokio::test]
    async fn test_delete_model_list_failure_deletes_nothing_and_returns_backend_error() {
        let temp_dir = TempDir::new().expect("create cache dir");
        let snapshot = temp_dir.path().join("models--org--model/snapshots/rev1");
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .once()
            .returning(|_| Err("redis down".into()));
        mock.expect_delete_model().times(0);

        let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
        let service = model_service_with_tracker(tracker);
        let response = service
            .delete_model(Request::new(DeleteModelRequest {
                model_name: "org/model".to_string(),
                provider: GrpcModelProvider::HuggingFace as i32,
            }))
            .await
            .expect("delete_model should return structured error response");
        assert_eq!(
            response.extensions().get::<RpcOutcome>(),
            Some(&RpcOutcome::BackendError)
        );
        assert!(!response.into_inner().success);
        assert!(
            snapshot.join("config.json").is_file(),
            "cached files must not be touched when registry listing fails"
        );
    }

    #[tokio::test]
    async fn test_delete_model_record_delete_failure_deletes_no_files() {
        let temp_dir = TempDir::new().expect("create cache dir");
        let snapshot = temp_dir.path().join("models--org--model/snapshots/rev1");
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used().once().returning(|_| {
            Ok(vec![record_with_provider(
                "org/model",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
            )])
        });
        mock.expect_delete_model()
            .once()
            .returning(|_| Err("delete failed".into()));

        let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
        let service = model_service_with_tracker(tracker);
        let response = service
            .delete_model(Request::new(DeleteModelRequest {
                model_name: "org/model".to_string(),
                provider: GrpcModelProvider::HuggingFace as i32,
            }))
            .await
            .expect("delete_model should return structured response");
        assert_eq!(
            response.extensions().get::<RpcOutcome>(),
            Some(&RpcOutcome::BackendError)
        );
        assert!(!response.into_inner().success);
        assert!(
            snapshot.join("config.json").is_file(),
            "cached files must remain intact when registry record deletion fails"
        );
    }

    #[tokio::test]
    async fn test_delete_model_client_drop_mid_registry_delete_still_removes_files() {
        let temp_dir = TempDir::new().expect("create cache dir");
        let repo_root = temp_dir.path().join("models--org--model");
        let snapshot = repo_root.join("snapshots/rev1");
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

        let (del_entered_tx, del_entered_rx) = tokio::sync::oneshot::channel::<()>();
        let (release_del_tx, release_del_rx) = tokio::sync::oneshot::channel::<()>();

        let backend = Arc::new(AsyncTestBackend {
            records: Arc::new(Mutex::new(vec![record_with_provider(
                "org/model",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
            )])),
            delete_entered: Arc::new(Mutex::new(Some(del_entered_tx))),
            delete_release: Arc::new(tokio::sync::Mutex::new(Some(release_del_rx))),
            hang_refresh: false,
        });
        let registry = Arc::new(RegistryManager::with_backend(backend));
        let mut reg = crate::metrics::new_registry();
        let tracker = ModelDownloadTracker::new(
            registry,
            temp_dir.path().to_path_buf(),
            DownloadMetrics::register(&mut reg),
            RegistryMetrics::register(&mut reg),
        );
        let service = model_service_with_tracker(tracker);

        // Start DeleteModel, wait until delete_model applies the delete and blocks,
        // then drop the RPC future before the response arrives.
        {
            let rpc_fut = service.delete_model(Request::new(DeleteModelRequest {
                model_name: "org/model".to_string(),
                provider: GrpcModelProvider::HuggingFace as i32,
            }));
            tokio::pin!(rpc_fut);
            tokio::select! {
                _ = &mut rpc_fut => panic!("RPC must stay blocked until barrier releases"),
                res = del_entered_rx => {
                    res.expect("delete_model should have started");
                }
            }
            // rpc_fut is dropped here while delete_model is blocked.
        }

        // Release the backend delete response; the detached task must finish deleting files.
        release_del_tx.send(()).expect("release delete barrier");

        let deadline = tokio::time::Instant::now()
            .checked_add(std::time::Duration::from_secs(2))
            .expect("deadline overflow");
        while repo_root.exists() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "detached delete task did not remove cached files after caller dropped RPC"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn test_delete_model_downloading_defers_cleanup_and_handles_reclaim_and_k8s() {
        let fast_timings = LeaseTimings {
            lease_duration: std::time::Duration::from_millis(80),
            heartbeat_interval: std::time::Duration::from_millis(20),
            self_fence_duration: std::time::Duration::from_millis(60),
            deferred_delete_delay: std::time::Duration::from_millis(60),
            ..LeaseTimings::default()
        };

        // 1. Redis/Memory: returns success immediately, keeps files during wait, then deletes them.
        {
            let temp_dir = TempDir::new().expect("create cache dir");
            let repo_root = temp_dir.path().join("models--org--inflight");
            let snapshot = repo_root.join("snapshots/rev1");
            std::fs::create_dir_all(&snapshot).expect("create snapshot");
            std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

            let list_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let list_calls_clone = Arc::clone(&list_calls);
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_get_models_by_last_used()
                .times(2)
                .returning(move |_| {
                    if list_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        Ok(vec![record_with_provider(
                            "org/inflight",
                            ModelProvider::HuggingFace,
                            ModelStatus::DOWNLOADING,
                        )])
                    } else {
                        Ok(Vec::new())
                    }
                });
            mock.expect_delete_model().once().returning(|_| Ok(()));

            let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf())
                .with_lease_timings(fast_timings);
            let outcome = tracker
                .delete_model_entries(ModelProvider::HuggingFace, "org/inflight")
                .await;
            assert!(outcome.success);
            assert!(
                outcome.message.contains("in flight") && outcome.message.contains("lease window"),
                "unexpected message: {}",
                outcome.message
            );
            assert!(
                repo_root.exists(),
                "files must not be removed synchronously while download was in flight"
            );

            let deadline = tokio::time::Instant::now()
                .checked_add(std::time::Duration::from_secs(2))
                .expect("deadline overflow");
            while repo_root.exists() {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "deferred cleanup task did not delete files after lease window"
                );
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }

        // 2. If a new live record appears during the lease wait, deferred cleanup keeps files.
        {
            let temp_dir = TempDir::new().expect("create cache dir");
            let repo_root = temp_dir.path().join("models--org--reclaimed");
            let snapshot = repo_root.join("snapshots/rev1");
            std::fs::create_dir_all(&snapshot).expect("create snapshot");
            std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

            let (relisted_tx, relisted_rx) = tokio::sync::oneshot::channel::<()>();
            let relisted_tx = Arc::new(Mutex::new(Some(relisted_tx)));
            let list_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let list_calls_clone = Arc::clone(&list_calls);
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_get_models_by_last_used()
                .times(2)
                .returning(move |_| {
                    if list_calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        Ok(vec![record_with_provider(
                            "org/reclaimed",
                            ModelProvider::HuggingFace,
                            ModelStatus::DOWNLOADING,
                        )])
                    } else {
                        if let Some(tx) = relisted_tx.lock().expect("lock").take() {
                            let _ = tx.send(());
                        }
                        Ok(vec![record_with_provider(
                            "org/reclaimed",
                            ModelProvider::HuggingFace,
                            ModelStatus::DOWNLOADING,
                        )])
                    }
                });
            mock.expect_delete_model().once().returning(|_| Ok(()));

            let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf())
                .with_lease_timings(fast_timings);
            let outcome = tracker
                .delete_model_entries(ModelProvider::HuggingFace, "org/reclaimed")
                .await;
            assert!(outcome.success);
            assert!(
                outcome.message.contains("lease window"),
                "unexpected message: {}",
                outcome.message
            );
            tokio::time::timeout(std::time::Duration::from_secs(2), relisted_rx)
                .await
                .expect("deferred cleanup should re-list after lease delay")
                .expect("relist signal");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            assert!(
                snapshot.join("config.json").is_file(),
                "deferred cleanup must preserve files when a new live record exists"
            );
        }

        // 3. Kubernetes backend (manages_deletion_cleanup == true) spawns no deferred task.
        {
            let temp_dir = TempDir::new().expect("create cache dir");
            let repo_root = temp_dir.path().join("models--org--k8s");
            let snapshot = repo_root.join("snapshots/rev1");
            std::fs::create_dir_all(&snapshot).expect("create snapshot");
            std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_get_models_by_last_used().once().returning(|_| {
                Ok(vec![record_with_provider(
                    "org/k8s",
                    ModelProvider::HuggingFace,
                    ModelStatus::DOWNLOADING,
                )])
            });
            mock.expect_delete_model().once().returning(|_| Ok(()));

            let registry = Arc::new(
                RegistryManager::with_backend(Arc::new(mock)).with_manages_deletion_cleanup(true),
            );
            let mut reg = crate::metrics::new_registry();
            let tracker = ModelDownloadTracker::new(
                registry,
                temp_dir.path().to_path_buf(),
                DownloadMetrics::register(&mut reg),
                RegistryMetrics::register(&mut reg),
            )
            .with_lease_timings(fast_timings);

            let outcome = tracker
                .delete_model_entries(ModelProvider::HuggingFace, "org/k8s")
                .await;
            assert!(outcome.success);
            assert!(
                outcome.message.contains("lease window"),
                "unexpected message: {}",
                outcome.message
            );
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
            assert!(
                snapshot.join("config.json").is_file(),
                "Kubernetes backend must leave deferred cleanup to the finalizer controller"
            );
        }
    }

    #[tokio::test]
    async fn test_delete_model_zero_records_still_clears_leftover_files() {
        let temp_dir = TempDir::new().expect("create cache dir");
        let repo_root = temp_dir.path().join("models--org--orphan");
        let snapshot = repo_root.join("snapshots/rev1");
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .times(2)
            .returning(|_| Ok(Vec::new()));
        mock.expect_delete_model().times(0);

        let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
        let outcome = tracker
            .delete_model_entries(ModelProvider::HuggingFace, "org/orphan")
            .await;
        assert!(outcome.success);
        assert!(
            outcome.message.contains("0 record(s)"),
            "unexpected message: {}",
            outcome.message
        );
        assert!(
            !repo_root.exists(),
            "zero-record DeleteModel retry must reclaim leftover cached files"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_delete_model_file_delete_error_returns_failure_outcome() {
        let temp_dir = TempDir::new().expect("create cache dir");
        let outside = TempDir::new().expect("create outside dir");
        let outside_model = outside.path().join("m");
        std::fs::create_dir_all(&outside_model).expect("create outside model");
        std::fs::write(outside_model.join("keep.txt"), b"keep").expect("write keep");

        let bucket_dir = temp_dir.path().join("s3/bkt");
        std::fs::create_dir_all(&bucket_dir).expect("create bucket dir");
        std::os::unix::fs::symlink(outside.path(), bucket_dir.join("link"))
            .expect("create ancestor symlink");

        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_get_models_by_last_used()
            .times(2)
            .returning(|_| Ok(Vec::new()));

        let tracker = tracker_with_mock_and_cache_dir(mock, temp_dir.path().to_path_buf());
        let service = model_service_with_tracker(tracker);
        let response = service
            .delete_model(Request::new(DeleteModelRequest {
                model_name: "s3://bkt/link/m".to_string(),
                provider: GrpcModelProvider::S3 as i32,
            }))
            .await
            .expect("delete_model should return structured error");
        assert_eq!(
            response.extensions().get::<RpcOutcome>(),
            Some(&RpcOutcome::Error)
        );
        assert!(!response.into_inner().success);
        assert!(
            outside_model.join("keep.txt").is_file(),
            "outside files must not be deleted through symlinked S3 ancestor"
        );
    }

    #[tokio::test]
    async fn test_writer_self_fence_aborts_on_failing_or_stalled_heartbeat_and_extends_on_renewal()
    {
        let timings = LeaseTimings {
            lease_duration: std::time::Duration::from_millis(120),
            heartbeat_interval: std::time::Duration::from_millis(25),
            self_fence_duration: std::time::Duration::from_millis(80),
            deferred_delete_delay: std::time::Duration::from_millis(120),
            ..LeaseTimings::default()
        };

        // 1. Failing heartbeats abort the download at self_fence_duration.
        {
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_refresh_download_claim()
                .returning(|_, _, _, _| Err("registry outage".into()));
            let (tracker, _cache_dir) = tracker_with_mock(mock);
            let tracker = tracker.with_lease_timings(timings);

            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                tracker.run_download_with_lease_fence(
                    "org/model",
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-1",
                    tokio::time::Instant::now(),
                    std::future::pending::<Result<(), anyhow::Error>>(),
                ),
            )
            .await
            .expect("self-fence should abort within timeout");
            assert!(
                outcome.is_none(),
                "repeated renewal failures must stop the download"
            );
        }

        // 2. Stalled/hanging refresh_download_claim aborts at self_fence_duration.
        {
            let temp_dir = TempDir::new().expect("create cache dir");
            let backend = Arc::new(AsyncTestBackend {
                records: Arc::new(Mutex::new(Vec::new())),
                delete_entered: Arc::new(Mutex::new(None)),
                delete_release: Arc::new(tokio::sync::Mutex::new(None)),
                hang_refresh: true,
            });
            let registry = Arc::new(RegistryManager::with_backend(backend));
            let mut reg = crate::metrics::new_registry();
            let tracker = ModelDownloadTracker::new(
                registry,
                temp_dir.path().to_path_buf(),
                DownloadMetrics::register(&mut reg),
                RegistryMetrics::register(&mut reg),
            )
            .with_lease_timings(timings);

            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                tracker.run_download_with_lease_fence(
                    "org/model",
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-2",
                    tokio::time::Instant::now(),
                    std::future::pending::<Result<(), anyhow::Error>>(),
                ),
            )
            .await
            .expect("stalled renewal must time out at self-fence deadline");
            assert!(
                outcome.is_none(),
                "stalled renewal must stop the download at self-fence deadline"
            );
        }

        // 3. Lost ownership (Ok(false)) stops the download on the first heartbeat.
        {
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_refresh_download_claim()
                .once()
                .returning(|_, _, _, _| Ok(false));
            let (tracker, _cache_dir) = tracker_with_mock(mock);
            let tracker = tracker.with_lease_timings(timings);

            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                tracker.run_download_with_lease_fence(
                    "org/model",
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-3",
                    tokio::time::Instant::now(),
                    std::future::pending::<Result<(), anyhow::Error>>(),
                ),
            )
            .await
            .expect("lost lease ownership should abort promptly");
            assert!(outcome.is_none());
        }

        // 4. Healthy renewals extend the deadline past initial self_fence_duration.
        {
            let renewals = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let renewals_clone = Arc::clone(&renewals);
            let mut mock = crate::registry::backend::MockRegistryBackend::new();
            mock.expect_refresh_download_claim()
                .returning(move |_, _, _, _| {
                    renewals_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(true)
                });
            let (tracker, _cache_dir) = tracker_with_mock(mock);
            let tracker = tracker.with_lease_timings(timings);

            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                tracker.run_download_with_lease_fence(
                    "org/model",
                    "org/model",
                    ModelProvider::HuggingFace,
                    "claim-4",
                    tokio::time::Instant::now(),
                    async {
                        tokio::time::sleep(std::time::Duration::from_millis(140)).await;
                        Ok::<&str, anyhow::Error>("done")
                    },
                ),
            )
            .await
            .expect("healthy download should finish");
            assert_eq!(outcome.and_then(|(res, _)| res.ok()), Some("done"));
            assert!(
                renewals.load(std::sync::atomic::Ordering::SeqCst) >= 3,
                "expected multiple lease renewals during extended download"
            );
        }
    }

    #[derive(Clone, Copy)]
    enum DelayedClaimMode {
        FreshClaim,
        Takeover,
        ErrorRetry,
        StaleDownloadedReclaim,
    }

    /// Async test backend that records a DOWNLOADING claim, notifies the test,
    /// and holds the claim reply until released.
    struct DelayedClaimBackend {
        mode: DelayedClaimMode,
        records: Arc<Mutex<Vec<crate::registry::backend::ModelRecord>>>,
        active_claim_id: Arc<Mutex<Option<String>>>,
        claim_acquired_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        claim_release_rx: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
        claim_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl DelayedClaimBackend {
        async fn hold_after_claim(
            &self,
            model_name: &str,
            provider: ModelProvider,
            claim_id: &str,
        ) {
            {
                let mut records = self.records.lock().expect("lock records");
                records.retain(|r| r.model_name != model_name);
                records.push(record_with_provider(
                    model_name,
                    provider,
                    ModelStatus::DOWNLOADING,
                ));
            }
            *self.active_claim_id.lock().expect("lock claim_id") = Some(claim_id.to_string());
            if let Some(tx) = self.claim_acquired_tx.lock().expect("lock acquired").take() {
                let _ = tx.send(());
            }
            if let Some(rx) = self.claim_release_rx.lock().await.take() {
                let _ = rx.await;
            }
        }
    }

    #[tonic::async_trait]
    impl crate::registry::backend::RegistryBackend for DelayedClaimBackend {
        async fn connect(&self) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn get_status(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<Option<ModelStatus>> {
            Ok(self
                .records
                .lock()
                .expect("lock records")
                .iter()
                .find(|r| r.model_name == model_name)
                .map(|r| r.status))
        }

        async fn get_model_record(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<Option<crate::registry::backend::ModelRecord>>
        {
            Ok(self
                .records
                .lock()
                .expect("lock records")
                .iter()
                .find(|r| r.model_name == model_name)
                .cloned())
        }

        async fn set_status(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _status: ModelStatus,
            _message: Option<String>,
        ) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn touch_model(
            &self,
            _model_name: &str,
        ) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn delete_model(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<()> {
            self.records
                .lock()
                .expect("lock records")
                .retain(|r| r.model_name != model_name);
            *self.active_claim_id.lock().expect("lock claim_id") = None;
            Ok(())
        }

        async fn get_models_by_last_used(
            &self,
            _limit: Option<u32>,
        ) -> crate::registry::backend::RegistryResult<Vec<crate::registry::backend::ModelRecord>>
        {
            Ok(self.records.lock().expect("lock records").clone())
        }

        async fn get_status_counts(
            &self,
        ) -> crate::registry::backend::RegistryResult<(u32, u32, u32)> {
            Ok((0, 0, 0))
        }

        async fn try_claim_for_download(
            &self,
            model_name: &str,
            provider: ModelProvider,
            claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<crate::registry::backend::ClaimOutcome>
        {
            let call = self
                .claim_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if call > 0 {
                return Ok(crate::registry::backend::ClaimOutcome::AlreadyExists(
                    ModelStatus::DOWNLOADING,
                ));
            }
            match self.mode {
                DelayedClaimMode::FreshClaim => {
                    self.hold_after_claim(model_name, provider, claim_id).await;
                    Ok(crate::registry::backend::ClaimOutcome::Claimed)
                }
                DelayedClaimMode::Takeover => {
                    self.hold_after_claim(model_name, provider, claim_id).await;
                    Ok(crate::registry::backend::ClaimOutcome::TookOver)
                }
                DelayedClaimMode::ErrorRetry => Ok(
                    crate::registry::backend::ClaimOutcome::AlreadyExists(ModelStatus::ERROR),
                ),
                DelayedClaimMode::StaleDownloadedReclaim => Ok(
                    crate::registry::backend::ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADED),
                ),
            }
        }

        async fn try_reset_error_for_retry(
            &self,
            model_name: &str,
            provider: ModelProvider,
            claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            self.hold_after_claim(model_name, provider, claim_id).await;
            Ok(true)
        }

        async fn try_reclaim_downloaded(
            &self,
            model_name: &str,
            provider: ModelProvider,
            claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            self.hold_after_claim(model_name, provider, claim_id).await;
            Ok(true)
        }

        async fn refresh_download_claim(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(self
                .active_claim_id
                .lock()
                .expect("lock claim_id")
                .as_deref()
                == Some(claim_id))
        }

        async fn finish_download_claim(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            claim_id: &str,
            _status: ModelStatus,
            _message: Option<String>,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(self
                .active_claim_id
                .lock()
                .expect("lock claim_id")
                .as_deref()
                == Some(claim_id))
        }
    }

    #[tokio::test]
    async fn test_delayed_claim_reply_released_after_delete_does_not_write() {
        let timings = LeaseTimings {
            lease_duration: std::time::Duration::from_millis(120),
            heartbeat_interval: std::time::Duration::from_millis(25),
            self_fence_duration: std::time::Duration::from_millis(80),
            deferred_delete_delay: std::time::Duration::from_millis(120),
            ..LeaseTimings::default()
        };

        let cases = [
            ("fresh claim", DelayedClaimMode::FreshClaim, 140u64),
            ("takeover", DelayedClaimMode::Takeover, 140u64),
            ("error retry", DelayedClaimMode::ErrorRetry, 140u64),
            (
                "stale downloaded reclaim",
                DelayedClaimMode::StaleDownloadedReclaim,
                140u64,
            ),
            // Also cover a delay between heartbeat_interval (25 ms) and
            // self_fence_duration (80 ms): the overdue heartbeat must observe
            // ownership loss before the provider writes.
            ("overdue heartbeat", DelayedClaimMode::FreshClaim, 45u64),
        ];

        for (label, mode, hold_ms) in cases {
            let temp_dir = TempDir::new().expect("create cache dir");
            let model_dir = temp_dir.path().join("s3/bucket/delayed-model");
            std::fs::create_dir_all(&model_dir).expect("create initial cached dir");
            std::fs::write(model_dir.join(".mx-model"), b"").expect("write marker");
            std::fs::write(model_dir.join("old.bin"), b"old").expect("write old file");

            let (acquired_tx, acquired_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();

            let backend = Arc::new(DelayedClaimBackend {
                mode,
                records: Arc::new(Mutex::new(Vec::new())),
                active_claim_id: Arc::new(Mutex::new(None)),
                claim_acquired_tx: Arc::new(Mutex::new(Some(acquired_tx))),
                claim_release_rx: Arc::new(tokio::sync::Mutex::new(Some(release_rx))),
                claim_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            });
            let registry = Arc::new(RegistryManager::with_backend(backend));
            let mut reg = crate::metrics::new_registry();

            let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let writes_clone = Arc::clone(&writes);
            let tracker = ModelDownloadTracker::new(
                registry,
                temp_dir.path().to_path_buf(),
                DownloadMetrics::register(&mut reg),
                RegistryMetrics::register(&mut reg),
            )
            .with_lease_timings(timings)
            .with_download_override(Arc::new(move |cache_dir| {
                writes_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let out_dir = cache_dir.join("s3/bucket/delayed-model");
                std::fs::create_dir_all(&out_dir)?;
                std::fs::write(out_dir.join("stale-write.bin"), b"bad")?;
                Ok(())
            }));

            let target = DownloadTarget {
                model_name: "s3://bucket/delayed-model".to_string(),
                provider: ModelProvider::S3,
                revision: None,
                ignore_weights: false,
            };
            let (tx, _rx) = tokio::sync::mpsc::channel(8);
            let claim_tracker = tracker.clone();
            let claim_target = target.clone();
            let claim_handle = tokio::spawn(async move {
                claim_tracker
                    .ensure_model_downloaded(&claim_target, &tx)
                    .await
            });

            // Wait until A acquired its claim in the backend and its reply is held.
            tokio::time::timeout(std::time::Duration::from_secs(1), acquired_rx)
                .await
                .expect("claim should reach backend")
                .expect("acquired signal");

            // B deletes the DOWNLOADING entry and triggers deferred cleanup.
            let delete_outcome = tracker
                .delete_model_entries(ModelProvider::S3, "s3://bucket/delayed-model")
                .await;
            assert!(delete_outcome.success, "{label}: delete must succeed");
            assert!(
                delete_outcome.message.contains("lease window"),
                "{label}: active download must defer cleanup: {}",
                delete_outcome.message
            );

            // Hold A's reply until after the configured delay (either past the
            // full deferred cleanup window or past the first heartbeat tick).
            tokio::time::sleep(std::time::Duration::from_millis(hold_ms)).await;
            let _ = release_tx.send(());

            // Give the spawned download task ample time to run if unfenced,
            // while staying well under the 500 ms wait-loop poll interval.
            tokio::time::sleep(std::time::Duration::from_millis(60)).await;
            claim_handle.abort();

            assert_eq!(
                writes.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "{label}: provider must not write after delayed claim reply is released"
            );
            assert!(
                !model_dir.join("stale-write.bin").exists(),
                "{label}: stale write file must not exist on disk"
            );
            if hold_ms >= 140 {
                assert!(
                    !model_dir.exists(),
                    "{label}: deferred cleanup should have removed the old model directory"
                );
            }
        }
    }

    /// In-memory test backend with optional hold points on a specific wait poll
    /// and on `finish_download_claim` (after applying the terminal state).
    #[derive(Default)]
    struct RelayTestBackend {
        records: Arc<Mutex<Vec<crate::registry::backend::ModelRecord>>>,
        active_claim_id: Arc<Mutex<Option<String>>>,
        claim_calls: Arc<std::sync::atomic::AtomicUsize>,
        hold_poll_call_index: Option<usize>,
        poll_entered_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        poll_release_rx: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
        finish_applied_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        finish_release_rx: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    }

    #[tonic::async_trait]
    impl crate::registry::backend::RegistryBackend for RelayTestBackend {
        async fn connect(&self) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn get_status(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<Option<ModelStatus>> {
            Ok(self
                .records
                .lock()
                .expect("records lock")
                .iter()
                .find(|r| r.model_name == model_name)
                .map(|r| r.status))
        }

        async fn get_model_record(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<Option<crate::registry::backend::ModelRecord>>
        {
            Ok(self
                .records
                .lock()
                .expect("records lock")
                .iter()
                .find(|r| r.model_name == model_name)
                .cloned())
        }

        async fn set_status(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _status: ModelStatus,
            _message: Option<String>,
        ) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn touch_model(
            &self,
            _model_name: &str,
        ) -> crate::registry::backend::RegistryResult<()> {
            Ok(())
        }

        async fn delete_model(
            &self,
            model_name: &str,
        ) -> crate::registry::backend::RegistryResult<()> {
            self.records
                .lock()
                .expect("records lock")
                .retain(|r| r.model_name != model_name);
            *self.active_claim_id.lock().expect("claim lock") = None;
            Ok(())
        }

        async fn get_models_by_last_used(
            &self,
            _limit: Option<u32>,
        ) -> crate::registry::backend::RegistryResult<Vec<crate::registry::backend::ModelRecord>>
        {
            Ok(self.records.lock().expect("records lock").clone())
        }

        async fn get_status_counts(
            &self,
        ) -> crate::registry::backend::RegistryResult<(u32, u32, u32)> {
            Ok((0, 0, 0))
        }

        async fn try_claim_for_download(
            &self,
            model_name: &str,
            provider: ModelProvider,
            claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<crate::registry::backend::ClaimOutcome>
        {
            let call = self
                .claim_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.hold_poll_call_index == Some(call) {
                if let Some(tx) = self.poll_entered_tx.lock().expect("poll tx").take() {
                    let _ = tx.send(());
                }
                if let Some(rx) = self.poll_release_rx.lock().await.take() {
                    let _ = rx.await;
                }
            }
            let mut records = self.records.lock().expect("records lock");
            if let Some(existing) = records.iter().find(|r| r.model_name == model_name) {
                return Ok(crate::registry::backend::ClaimOutcome::AlreadyExists(
                    existing.status,
                ));
            }
            records.push(record_with_provider(
                model_name,
                provider,
                ModelStatus::DOWNLOADING,
            ));
            *self.active_claim_id.lock().expect("claim lock") = Some(claim_id.to_string());
            Ok(crate::registry::backend::ClaimOutcome::Claimed)
        }

        async fn try_reset_error_for_retry(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(false)
        }

        async fn try_reclaim_downloaded(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            _claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(false)
        }

        async fn refresh_download_claim(
            &self,
            _model_name: &str,
            _provider: ModelProvider,
            claim_id: &str,
            _lease_duration: std::time::Duration,
        ) -> crate::registry::backend::RegistryResult<bool> {
            Ok(self.active_claim_id.lock().expect("claim lock").as_deref() == Some(claim_id))
        }

        async fn finish_download_claim(
            &self,
            model_name: &str,
            provider: ModelProvider,
            claim_id: &str,
            status: ModelStatus,
            _message: Option<String>,
        ) -> crate::registry::backend::RegistryResult<bool> {
            let applied = {
                let mut claim = self.active_claim_id.lock().expect("claim lock");
                if claim.as_deref() != Some(claim_id) {
                    false
                } else {
                    *claim = None;
                    let mut records = self.records.lock().expect("records lock");
                    records.retain(|r| r.model_name != model_name);
                    records.push(record_with_provider(model_name, provider, status));
                    true
                }
            };
            if applied {
                if let Some(tx) = self.finish_applied_tx.lock().expect("finish tx").take() {
                    let _ = tx.send(());
                }
                if let Some(rx) = self.finish_release_rx.lock().await.take() {
                    let _ = rx.await;
                }
            }
            Ok(applied)
        }
    }

    fn write_s3_model_files(
        cache_dir: &Path,
        sub_path: &str,
        content: &[u8],
    ) -> Result<(), anyhow::Error> {
        let model_dir = cache_dir.join("s3").join(sub_path);
        std::fs::create_dir_all(&model_dir)?;
        std::fs::write(model_dir.join(".mx-model"), b"ready")?;
        std::fs::write(model_dir.join("config.json"), content)?;
        Ok(())
    }

    /// Case 8f (S2): while a request's wait-loop poll is held in flight, the background
    /// download finishes in the registry. No premature completion is delivered while the
    /// poll is pending; once the poll resolves, the request returns DOWNLOADED and a
    /// subsequent delete removes records and files without triggering any reclaim.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_s2_held_poll_in_flight_does_not_reclaim_after_completion_and_delete() {
        let temp_dir = TempDir::new().expect("temp dir");
        let (poll_entered_tx, poll_entered_rx) = tokio::sync::oneshot::channel();
        let (poll_release_tx, poll_release_rx) = tokio::sync::oneshot::channel();
        let (finish_applied_tx, finish_applied_rx) = tokio::sync::oneshot::channel();
        let (allow_download_tx, allow_download_rx) = std::sync::mpsc::channel::<()>();
        let allow_download_rx = Arc::new(Mutex::new(allow_download_rx));

        let records = Arc::new(Mutex::new(Vec::new()));
        let backend = Arc::new(RelayTestBackend {
            records: Arc::clone(&records),
            hold_poll_call_index: Some(1),
            poll_entered_tx: Arc::new(Mutex::new(Some(poll_entered_tx))),
            poll_release_rx: Arc::new(tokio::sync::Mutex::new(Some(poll_release_rx))),
            finish_applied_tx: Arc::new(Mutex::new(Some(finish_applied_tx))),
            ..RelayTestBackend::default()
        });
        let registry = Arc::new(RegistryManager::with_backend(backend));
        let mut reg = crate::metrics::new_registry();
        let download_runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let download_runs_clone = Arc::clone(&download_runs);
        let tracker = ModelDownloadTracker::new(
            registry,
            temp_dir.path().to_path_buf(),
            DownloadMetrics::register(&mut reg),
            RegistryMetrics::register(&mut reg),
        )
        .with_lease_timings(LeaseTimings {
            poll_interval: std::time::Duration::from_millis(50),
            ..LeaseTimings::default()
        })
        .with_download_override(Arc::new(move |cache_dir| {
            download_runs_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Ok(rx) = allow_download_rx.lock() {
                let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
            }
            write_s3_model_files(cache_dir, "bucket/s2-8f", b"{}")
        }));

        let target = DownloadTarget {
            model_name: "s3://bucket/s2-8f".to_string(),
            provider: ModelProvider::S3,
            revision: None,
            ignore_weights: true,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let owner_tracker = tracker.clone();
        let owner_target = target.clone();
        let owner_handle = tokio::spawn(async move {
            owner_tracker
                .ensure_model_downloaded(&owner_target, &tx)
                .await
        });

        let initial = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("initial update timeout")
            .expect("stream open")
            .expect("ok update");
        assert_eq!(
            initial.status,
            modelexpress_common::grpc::model::ModelStatus::Downloading as i32
        );

        // Wait until the wait-loop poll enters the backend and is held in flight.
        tokio::time::timeout(std::time::Duration::from_secs(2), poll_entered_rx)
            .await
            .expect("poll should enter backend")
            .expect("poll entered");

        // Let the download finish and mark the registry DOWNLOADED while the poll is held.
        let _ = allow_download_tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(2), finish_applied_rx)
            .await
            .expect("download should finish in registry")
            .expect("finish applied");

        // While the poll is held, the request must still be waiting (no background push).
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        assert!(
            !owner_handle.is_finished(),
            "request must wait for its in-flight poll rather than exiting on a background push"
        );
        assert!(
            rx.try_recv().is_err(),
            "no premature terminal update while poll is in flight"
        );

        // Release the poll; the request observes DOWNLOADED and exits cleanly.
        let _ = poll_release_tx.send(());
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), owner_handle)
                .await
                .expect("owner completion timeout")
                .expect("owner join"),
            ModelStatus::DOWNLOADED
        );
        assert_eq!(tracker.waiting_count(), 0);

        // Delete immediately after the client observed DOWNLOADED; no second download
        // or leftover record may appear.
        let delete_outcome = tracker
            .delete_model_entries(ModelProvider::S3, "s3://bucket/s2-8f")
            .await;
        assert!(delete_outcome.success, "{}", delete_outcome.message);
        assert!(!temp_dir.path().join("s3/bucket/s2-8f").exists());

        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        assert_eq!(download_runs.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(records.lock().expect("records lock").is_empty());
    }

    /// Case 8h (S2): when `finish_download_claim` applies DOWNLOADED in the registry
    /// but its RPC reply stays held indefinitely, both the owner and a same-replica
    /// co-waiter still observe DOWNLOADED via registry polling and finish promptly.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_s2_held_completion_reply_does_not_block_owner_or_co_waiter() {
        let temp_dir = TempDir::new().expect("temp dir");
        let (finish_applied_tx, finish_applied_rx) = tokio::sync::oneshot::channel();
        let (mut finish_release_tx, finish_release_rx) = tokio::sync::oneshot::channel();
        let (download_started_tx, download_started_rx) = tokio::sync::oneshot::channel();
        let download_started_tx = Arc::new(Mutex::new(Some(download_started_tx)));
        let (allow_download_tx, allow_download_rx) = std::sync::mpsc::channel::<()>();
        let allow_download_rx = Arc::new(Mutex::new(allow_download_rx));

        let backend = Arc::new(RelayTestBackend {
            finish_applied_tx: Arc::new(Mutex::new(Some(finish_applied_tx))),
            finish_release_rx: Arc::new(tokio::sync::Mutex::new(Some(finish_release_rx))),
            ..RelayTestBackend::default()
        });
        let registry = Arc::new(RegistryManager::with_backend(backend));
        let mut reg = crate::metrics::new_registry();
        let tracker = ModelDownloadTracker::new(
            registry,
            temp_dir.path().to_path_buf(),
            DownloadMetrics::register(&mut reg),
            RegistryMetrics::register(&mut reg),
        )
        .with_lease_timings(LeaseTimings {
            lease_duration: std::time::Duration::from_millis(400),
            heartbeat_interval: std::time::Duration::from_millis(100),
            self_fence_duration: std::time::Duration::from_millis(300),
            deferred_delete_delay: std::time::Duration::from_millis(400),
            poll_interval: std::time::Duration::from_millis(50),
        })
        .with_download_override(Arc::new(move |cache_dir| {
            if let Ok(mut slot) = download_started_tx.lock()
                && let Some(tx) = slot.take()
            {
                let _ = tx.send(());
            }
            if let Ok(rx) = allow_download_rx.lock() {
                let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
            }
            write_s3_model_files(cache_dir, "bucket/s2-8h", b"{}")
        }));

        let target = DownloadTarget {
            model_name: "s3://bucket/s2-8h".to_string(),
            provider: ModelProvider::S3,
            revision: None,
            ignore_weights: true,
        };
        let (owner_tx, _owner_rx) = tokio::sync::mpsc::channel(8);
        let (waiter_tx, mut waiter_rx) = tokio::sync::mpsc::channel(8);

        let owner_tracker = tracker.clone();
        let owner_target = target.clone();
        let owner_handle = tokio::spawn(async move {
            owner_tracker
                .ensure_model_downloaded(&owner_target, &owner_tx)
                .await
        });

        tokio::time::timeout(std::time::Duration::from_secs(2), download_started_rx)
            .await
            .expect("download should start")
            .expect("started");

        // Join a local co-waiter while the download is still running.
        let waiter_tracker = tracker.clone();
        let waiter_target = target.clone();
        let waiter_handle = tokio::spawn(async move {
            waiter_tracker
                .ensure_model_downloaded(&waiter_target, &waiter_tx)
                .await
        });

        let waiter_initial =
            tokio::time::timeout(std::time::Duration::from_secs(2), waiter_rx.recv())
                .await
                .expect("waiter initial update timeout")
                .expect("waiter stream open")
                .expect("ok update");
        assert_eq!(
            waiter_initial.status,
            modelexpress_common::grpc::model::ModelStatus::Downloading as i32
        );

        // Complete the download; finish_download_claim applies DOWNLOADED in the
        // backend while its reply stays held (_finish_release_tx is never sent).
        let _ = allow_download_tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(2), finish_applied_rx)
            .await
            .expect("completion should apply in registry")
            .expect("applied");

        // Both owner and co-waiter must observe DOWNLOADED via polling without waiting
        // for the held completion reply.
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), owner_handle)
                .await
                .expect("owner must not hang on held completion reply")
                .expect("owner join"),
            ModelStatus::DOWNLOADED
        );
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), waiter_handle)
                .await
                .expect("co-waiter must not hang on held completion reply")
                .expect("waiter join"),
            ModelStatus::DOWNLOADED
        );
        assert_eq!(tracker.waiting_count(), 0);

        // The background worker's hung finish_download_claim future must be bounded by
        // lease_expiry and dropped once the lease expires.
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            finish_release_tx.closed(),
        )
        .await
        .expect("hung finish_download_claim should time out at lease_expiry");
    }

    /// Case 8i (S2): download A finishes and applies DOWNLOADED with its completion
    /// reply held; the entry and files are deleted; request B starts a new claim for
    /// the same model on the same replica; A's old completion reply is then released.
    /// B must remain in progress until B's own download writes files and completes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_s2_late_old_completion_after_delete_does_not_complete_new_claim() {
        let temp_dir = TempDir::new().expect("temp dir");
        let (finish_a_applied_tx, finish_a_applied_rx) = tokio::sync::oneshot::channel();
        let (finish_a_release_tx, finish_a_release_rx) = tokio::sync::oneshot::channel();
        let (b_started_tx, b_started_rx) = tokio::sync::oneshot::channel();
        let b_started_tx = Arc::new(Mutex::new(Some(b_started_tx)));
        let (allow_b_finish_tx, allow_b_finish_rx) = std::sync::mpsc::channel::<()>();
        let allow_b_finish_rx = Arc::new(Mutex::new(allow_b_finish_rx));

        let backend = Arc::new(RelayTestBackend {
            finish_applied_tx: Arc::new(Mutex::new(Some(finish_a_applied_tx))),
            finish_release_rx: Arc::new(tokio::sync::Mutex::new(Some(finish_a_release_rx))),
            ..RelayTestBackend::default()
        });
        let registry = Arc::new(RegistryManager::with_backend(backend));
        let mut reg = crate::metrics::new_registry();
        let download_runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let download_runs_clone = Arc::clone(&download_runs);
        let tracker = ModelDownloadTracker::new(
            registry,
            temp_dir.path().to_path_buf(),
            DownloadMetrics::register(&mut reg),
            RegistryMetrics::register(&mut reg),
        )
        .with_lease_timings(LeaseTimings {
            poll_interval: std::time::Duration::from_millis(50),
            ..LeaseTimings::default()
        })
        .with_download_override(Arc::new(move |cache_dir| {
            let run = download_runs_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if run == 0 {
                write_s3_model_files(cache_dir, "bucket/s2-8i", b"{\"gen\":\"A\"}")
            } else {
                if let Ok(mut slot) = b_started_tx.lock()
                    && let Some(tx) = slot.take()
                {
                    let _ = tx.send(());
                }
                if let Ok(rx) = allow_b_finish_rx.lock() {
                    let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
                }
                write_s3_model_files(cache_dir, "bucket/s2-8i", b"{\"gen\":\"B\"}")
            }
        }));

        let target = DownloadTarget {
            model_name: "s3://bucket/s2-8i".to_string(),
            provider: ModelProvider::S3,
            revision: None,
            ignore_weights: true,
        };

        // 1. Download A runs and applies DOWNLOADED, while its reply stays held.
        let (a_tx, _a_rx) = tokio::sync::mpsc::channel(8);
        let a_tracker = tracker.clone();
        let a_target = target.clone();
        let a_handle =
            tokio::spawn(async move { a_tracker.ensure_model_downloaded(&a_target, &a_tx).await });
        tokio::time::timeout(std::time::Duration::from_secs(2), finish_a_applied_rx)
            .await
            .expect("A should apply DOWNLOADED in registry")
            .expect("applied");
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), a_handle)
                .await
                .expect("A should observe DOWNLOADED")
                .expect("A join"),
            ModelStatus::DOWNLOADED
        );

        // 2. Delete A's entry and files while A's completion reply is still held.
        let delete_outcome = tracker
            .delete_model_entries(ModelProvider::S3, "s3://bucket/s2-8i")
            .await;
        assert!(delete_outcome.success, "{}", delete_outcome.message);
        assert!(!temp_dir.path().join("s3/bucket/s2-8i").exists());

        // 3. Start B on the same replica; wait until B is actively downloading.
        let (b_tx, mut b_rx) = tokio::sync::mpsc::channel(8);
        let b_tracker = tracker.clone();
        let b_target = target.clone();
        let b_handle =
            tokio::spawn(async move { b_tracker.ensure_model_downloaded(&b_target, &b_tx).await });
        let b_initial = tokio::time::timeout(std::time::Duration::from_secs(2), b_rx.recv())
            .await
            .expect("B initial update timeout")
            .expect("B stream open")
            .expect("ok update");
        assert_eq!(
            b_initial.status,
            modelexpress_common::grpc::model::ModelStatus::Downloading as i32
        );
        tokio::time::timeout(std::time::Duration::from_secs(2), b_started_rx)
            .await
            .expect("B download should start")
            .expect("B started");

        // 4. Release A's stale Ok(true) completion reply while B is still downloading.
        assert!(
            finish_a_release_tx.send(()).is_ok(),
            "A's finish_download_claim should still be awaiting its held Ok(true) reply"
        );

        // B must NOT complete prematurely while B's files are not yet on disk.
        tokio::time::sleep(std::time::Duration::from_millis(160)).await;
        assert!(
            !b_handle.is_finished(),
            "late completion reply from A must not complete B while B is downloading"
        );
        assert!(
            b_rx.try_recv().is_err(),
            "B must not receive a premature DOWNLOADED update from A's late reply"
        );
        assert!(!temp_dir.path().join("s3/bucket/s2-8i").exists());

        // 5. Let B finish writing its files; B now observes DOWNLOADED with B's files.
        let _ = allow_b_finish_tx.send(());
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), b_handle)
                .await
                .expect("B completion timeout")
                .expect("B join"),
            ModelStatus::DOWNLOADED
        );
        assert_eq!(
            std::fs::read_to_string(temp_dir.path().join("s3/bucket/s2-8i/config.json"))
                .expect("read B config"),
            "{\"gen\":\"B\"}"
        );
    }
}
