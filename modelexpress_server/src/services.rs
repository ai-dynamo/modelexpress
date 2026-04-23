// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::registry::backend::ClaimOutcome;
use crate::registry::state::RegistryManager;
use modelexpress_common::{
    cache::CacheConfig,
    constants, download,
    grpc::{
        api::{ApiRequest, ApiResponse, api_service_server::ApiService},
        health::{HealthRequest, HealthResponse, health_service_server::HealthService},
        model::{
            FileChunk, ModelDownloadRequest, ModelFileInfo, ModelFileList, ModelFilesRequest,
            ModelStatusUpdate, model_service_server::ModelService,
        },
    },
    models::{ModelProvider, ModelStatus},
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
use tracing::{debug, error, info};

static START_TIME: std::sync::OnceLock<SystemTime> = std::sync::OnceLock::new();

/// Get the configured cache directory for model downloads
fn get_server_cache_dir() -> Option<std::path::PathBuf> {
    // Try to get cache configuration
    if let Ok(config) = CacheConfig::discover() {
        Some(config.local_path)
    } else {
        // Fall back to environment variable
        std::env::var("HF_HUB_CACHE")
            .ok()
            .map(std::path::PathBuf::from)
    }
}

/// Convert gRPC provider to internal ModelProvider enum
///
/// Falls back to HuggingFace provider if the conversion fails or an invalid
/// provider value is provided. A warning is logged when fallback occurs.
fn convert_provider(grpc_provider: i32) -> ModelProvider {
    match modelexpress_common::grpc::model::ModelProvider::try_from(grpc_provider) {
        Ok(provider) => provider.into(),
        Err(_) => {
            tracing::warn!(
                "Invalid provider value {}, falling back to HuggingFace",
                grpc_provider
            );
            ModelProvider::HuggingFace
        }
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

            Ok(Response::new(ApiResponse {
                success: true,
                data: Some(data_bytes),
                error: None,
            }))
        } else {
            error!("Unknown action: {}", api_request.action);
            Ok(Response::new(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Unknown action: {}", api_request.action)),
            }))
        }
    }
}

/// Model service implementation
#[derive(Debug, Default)]
pub struct ModelServiceImpl;

/// Helper function to collect all files in a model directory recursively
fn collect_model_files(base_path: &Path, current_path: &Path) -> Vec<(PathBuf, u64)> {
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
                        if is_safe {
                            files.push((relative.to_path_buf(), metadata.len()));
                        } else {
                            tracing::warn!(
                                "Skipping potentially unsafe file path: {:?} (relative: {:?})",
                                path,
                                relative
                            );
                        }
                    }
                }
            } else if path.is_dir() {
                files.extend(collect_model_files(base_path, &path));
            }
        }
    }

    files
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
        let model_name = model_request.model_name.clone();

        // Convert gRPC provider to our enum
        let provider: ModelProvider =
            modelexpress_common::grpc::model::ModelProvider::try_from(model_request.provider)
                .unwrap_or(modelexpress_common::grpc::model::ModelProvider::HuggingFace)
                .into();
        let ignore_weights = model_request.ignore_weights;

        // Spawn a task to handle the streaming download updates
        tokio::spawn(async move {
            let Some(tracker) = model_tracker() else {
                let _ = tx
                    .send(Err(Status::unavailable(
                        "server startup incomplete: model tracker not initialized",
                    )))
                    .await;
                return;
            };
            // Run the full claim + wait + retry flow. `ensure_model_downloaded` sends
            // its own initial status update (based on the `ClaimOutcome` returned by the
            // registry), so we don't do a pre-check here — a pre-check would either
            // duplicate that update or, worse, emit `status=ERROR` on a model we're
            // about to retry and trip the client-lib's terminal-error bailout before
            // the retry completion broadcast arrives.
            let final_status = tracker
                .ensure_model_downloaded(&model_name, provider, &tx, ignore_weights)
                .await;

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
                provider: modelexpress_common::grpc::model::ModelProvider::from(provider) as i32,
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
        let model_name = files_request.model_name.clone();
        let chunk_size = if files_request.chunk_size == 0 {
            constants::DEFAULT_TRANSFER_CHUNK_SIZE
        } else {
            files_request.chunk_size as usize
        };

        // Convert gRPC provider to our enum
        let provider = convert_provider(files_request.provider);

        info!(
            "Starting file stream for model: {} with chunk size: {} bytes",
            model_name, chunk_size
        );

        // Get the cache directory
        let cache_dir = get_server_cache_dir()
            .ok_or_else(|| Status::internal("Server cache directory not configured"))?;

        // Get the model path using the provider from the request
        let provider_impl = download::get_provider(provider);
        let model_path = provider_impl
            .get_model_path(&model_name, cache_dir.clone())
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

        // Collect all files to stream
        let files = collect_model_files(&model_path, &model_path);

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
        let model_name = files_request.model_name.clone();

        // Convert gRPC provider to our enum
        let provider = convert_provider(files_request.provider);

        info!("Listing files for model: {}", model_name);

        // Get the cache directory
        let cache_dir = get_server_cache_dir()
            .ok_or_else(|| Status::internal("Server cache directory not configured"))?;

        // Get the model path using the provider from the request
        let provider_impl = download::get_provider(provider);
        let model_path = provider_impl
            .get_model_path(&model_name, cache_dir)
            .await
            .map_err(|e| Status::not_found(format!("Model not found: {e}")))?;

        // Collect all files
        let files = collect_model_files(&model_path, &model_path);

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
}

/// Type alias for the complex waiting channels type
type WaitingChannels =
    Arc<Mutex<HashMap<String, Vec<tokio::sync::mpsc::Sender<Result<ModelStatusUpdate, Status>>>>>>;

/// Tracks the status of model downloads through the distributed registry backend.
#[derive(Clone)]
pub struct ModelDownloadTracker {
    /// Distributed registry (Redis today, K8s CRDs in a follow-up).
    registry: Arc<RegistryManager>,
    /// Maps model names to list of channels waiting for updates on this server replica.
    waiting_channels: WaitingChannels,
}

impl ModelDownloadTracker {
    pub fn new(registry: Arc<RegistryManager>) -> Self {
        Self {
            registry,
            waiting_channels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Gets the status of a model from the registry, bumping `last_used_at` on hit.
    /// Returns None on lookup failure (error logged) or unknown model.
    pub async fn get_status(&self, model_name: &str) -> Option<ModelStatus> {
        match self.registry.get_status(model_name).await {
            Ok(Some(status)) => {
                if let Err(e) = self.registry.touch_model(model_name).await {
                    error!("Failed to touch model {model_name}: {e}");
                }
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

        let mut waiting = match self.waiting_channels.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                error!("Waiting channels mutex is poisoned, recovering");
                poisoned.into_inner()
            }
        };
        if let Some(channels) = waiting.get(&model_name) {
            let update = ModelStatusUpdate {
                model_name: model_name.clone(),
                status: modelexpress_common::grpc::model::ModelStatus::from(status) as i32,
                message,
                provider: modelexpress_common::grpc::model::ModelProvider::from(provider) as i32,
            };
            for channel in channels {
                let _ = channel.try_send(Ok(update.clone()));
            }
            if status == ModelStatus::DOWNLOADED || status == ModelStatus::ERROR {
                waiting.remove(&model_name);
            }
        }
    }

    /// Sets the status of a model (no message), notifying waiters.
    pub async fn set_status(
        &self,
        model_name: String,
        status: ModelStatus,
        provider: ModelProvider,
    ) {
        self.set_status_and_notify(model_name, status, provider, None)
            .await;
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
    pub async fn delete_status(&self, model_name: &str) {
        if let Err(e) = self.registry.delete_model(model_name).await {
            error!("Failed to delete model from registry: {e}");
        }
        let mut waiting = match self.waiting_channels.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                error!("Waiting channels mutex is poisoned, recovering");
                poisoned.into_inner()
            }
        };
        waiting.remove(model_name);
    }

    /// Spawn a background task that actually downloads the model, updating the tracker on
    /// success or failure. Extracted here so the claim and retry paths share the code.
    fn spawn_download_task(
        &self,
        model_name: String,
        provider: ModelProvider,
        ignore_weights: bool,
        retry: bool,
    ) {
        let tracker = self.clone();
        tokio::spawn(async move {
            let cache_dir = get_server_cache_dir();
            match download::download_model(&model_name, provider, cache_dir, ignore_weights).await {
                Ok(_path) => {
                    tracker
                        .set_status_and_notify(
                            model_name,
                            ModelStatus::DOWNLOADED,
                            provider,
                            Some("Model download completed successfully".to_string()),
                        )
                        .await;
                }
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
                    tracker
                        .set_status_and_notify(model_name, ModelStatus::ERROR, provider, Some(msg))
                        .await;
                }
            }
        });
    }

    /// Initiates a download for a model and streams status updates.
    pub async fn ensure_model_downloaded(
        &self,
        model_name: &str,
        provider: ModelProvider,
        tx: &tokio::sync::mpsc::Sender<Result<ModelStatusUpdate, Status>>,
        ignore_weights: bool,
    ) -> ModelStatus {
        // Atomically try to claim this model for download. The `ClaimOutcome` tells us
        // whether THIS replica won the claim or is observing someone else's claim —
        // status alone (`DOWNLOADING`) can't distinguish those cases across replicas.
        let (status, is_owner) = match self
            .registry
            .try_claim_for_download(model_name, provider)
            .await
        {
            Ok(ClaimOutcome::Claimed) => (ModelStatus::DOWNLOADING, true),
            Ok(ClaimOutcome::AlreadyExists(existing)) => (existing, false),
            Err(e) => {
                error!("Failed to claim model for download: {e}");
                let error_update = ModelStatusUpdate {
                    model_name: model_name.to_string(),
                    status: modelexpress_common::grpc::model::ModelStatus::from(ModelStatus::ERROR)
                        as i32,
                    message: Some("Registry error occurred".to_string()),
                    provider: modelexpress_common::grpc::model::ModelProvider::from(provider)
                        as i32,
                };
                let _ = tx.send(Ok(error_update)).await;
                return ModelStatus::ERROR;
            }
        };

        // If we observed a previous ERROR, attempt the ERROR→DOWNLOADING CAS up front.
        // Only the CAS winner spawns the retry download; observers fall through to the
        // wait loop. Doing this *before* the initial stream update keeps the reported
        // status honest: after this block, the record is DOWNLOADING (the record may
        // briefly have been ERROR, but the client should wait, not bail).
        let (effective_status, is_retry_owner) = if status == ModelStatus::ERROR {
            let won = match self
                .registry
                .try_reset_error_for_retry(model_name, provider)
                .await
            {
                Ok(won) => won,
                Err(e) => {
                    error!("Failed to CAS status for retry: {e}");
                    let _ = tx
                        .send(Ok(ModelStatusUpdate {
                            model_name: model_name.to_string(),
                            status: modelexpress_common::grpc::model::ModelStatus::from(
                                ModelStatus::ERROR,
                            ) as i32,
                            message: Some("Registry error occurred during retry".to_string()),
                            provider: modelexpress_common::grpc::model::ModelProvider::from(
                                provider,
                            ) as i32,
                        }))
                        .await;
                    return ModelStatus::ERROR;
                }
            };
            (ModelStatus::DOWNLOADING, won)
        } else {
            (status, false)
        };

        let update = ModelStatusUpdate {
            model_name: model_name.to_string(),
            status: modelexpress_common::grpc::model::ModelStatus::from(effective_status) as i32,
            message: match (status, effective_status) {
                (_, ModelStatus::DOWNLOADED) => Some("Model already downloaded".to_string()),
                (ModelStatus::ERROR, _) => Some("Previous download failed, retrying".to_string()),
                (_, ModelStatus::DOWNLOADING) => Some("Model download in progress".to_string()),
                // effective can never be ERROR: ERROR observations are CAS'd above.
                (_, ModelStatus::ERROR) => Some("Download error".to_string()),
            },
            provider: modelexpress_common::grpc::model::ModelProvider::from(provider) as i32,
        };
        let _ = tx.send(Ok(update)).await;

        if effective_status == ModelStatus::DOWNLOADING {
            // Every caller is a waiter — whether we own the download or not, we still
            // need a channel so the completion broadcast reaches this stream.
            self.add_waiting_channel(model_name, tx.clone());

            // Spawn the download only on the replica that won the claim (fresh
            // download) or won the ERROR-retry CAS. Everyone else waits.
            if is_owner || is_retry_owner {
                let retry = status == ModelStatus::ERROR;
                self.spawn_download_task(model_name.to_string(), provider, ignore_weights, retry);
            }

            // Wait for completion by polling the registry.
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if let Some(current_status) = self.get_status(model_name).await
                    && current_status != ModelStatus::DOWNLOADING
                {
                    return current_status;
                }
            }
        }

        effective_status
    }
}

/// Global model download tracker. Initialized by `main.rs` after the registry is
/// connected, then read via `model_tracker()` from gRPC service handlers.
static MODEL_TRACKER: std::sync::OnceLock<Arc<ModelDownloadTracker>> = std::sync::OnceLock::new();

/// Initialize the process-wide tracker. Called exactly once during server startup.
/// Returns an error if called twice — main.rs propagates it as a startup failure.
pub fn init_model_tracker(
    tracker: Arc<ModelDownloadTracker>,
) -> Result<Arc<ModelDownloadTracker>, &'static str> {
    MODEL_TRACKER
        .set(tracker.clone())
        .map(|()| tracker)
        .map_err(|_| "init_model_tracker called more than once")
}

/// Read the process-wide tracker. Returns `None` if `init_model_tracker` hasn't run yet,
/// letting gRPC handlers return `Status::unavailable` instead of crashing the server.
pub fn model_tracker() -> Option<&'static Arc<ModelDownloadTracker>> {
    MODEL_TRACKER.get()
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
        ModelDownloadTracker::new(registry)
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

    #[tokio::test]
    async fn test_tracker_set_status_delegates_without_message() {
        let mut mock = crate::registry::backend::MockRegistryBackend::new();
        mock.expect_set_status()
            .withf(|_, _, status, msg| *status == ModelStatus::DOWNLOADING && msg.is_none())
            .once()
            .returning(|_, _, _, _| Ok(()));
        let tracker = tracker_with_mock(mock);
        tracker
            .set_status(
                "m".to_string(),
                ModelStatus::DOWNLOADING,
                ModelProvider::HuggingFace,
            )
            .await;
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

    #[tokio::test]
    async fn test_model_tracker_uninitialized_returns_none() {
        // The process-wide MODEL_TRACKER hasn't been init'd in tests, so
        // model_tracker() returns None — the service layer uses this to respond with
        // Status::unavailable rather than panic.
        assert!(model_tracker().is_none());
    }

    #[test]
    fn test_collect_model_files_empty_dir() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let files = collect_model_files(temp_dir.path(), temp_dir.path());
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

        let files = collect_model_files(temp_dir.path(), temp_dir.path());

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
    fn test_collect_model_files_nested() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create nested directory structure
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).expect("Failed to create subdir");

        let file1_path = temp_dir.path().join("root_file.txt");
        std::fs::write(&file1_path, "root content").expect("Failed to write file1");

        let file2_path = subdir.join("nested_file.txt");
        std::fs::write(&file2_path, "nested content").expect("Failed to write file2");

        let files = collect_model_files(temp_dir.path(), temp_dir.path());

        assert_eq!(files.len(), 2);

        // Check that nested path is correct
        let paths: Vec<_> = files
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();
        assert!(paths.iter().any(|p| p.contains("nested_file")));
    }

    #[tokio::test]
    async fn test_list_model_files_not_found() {
        let service = ModelServiceImpl;

        let request = Request::new(ModelFilesRequest {
            model_name: "non-existent-model-12345".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 0,
        });

        let result = service.list_model_files(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_stream_model_files_not_found() {
        let service = ModelServiceImpl;

        let request = Request::new(ModelFilesRequest {
            model_name: "non-existent-model-12345".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
        });

        let result = service.stream_model_files(request).await;
        assert!(result.is_err());
        let status = result.expect_err("Should return error");
        assert_eq!(status.code(), tonic::Code::NotFound);
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

        let service = ModelServiceImpl;
        let request = Request::new(ModelFilesRequest {
            model_name: "test/model".to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            chunk_size: 1024,
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
}
