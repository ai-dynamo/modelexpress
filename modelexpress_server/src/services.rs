// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::database::ModelDatabase;
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
            // Check if the model is already downloaded
            if let Some(status) = MODEL_TRACKER.get_status(&model_name) {
                let update = ModelStatusUpdate {
                    model_name: model_name.clone(),
                    status: modelexpress_common::grpc::model::ModelStatus::from(status) as i32,
                    message: match status {
                        ModelStatus::DOWNLOADED => Some("Model already downloaded".to_string()),
                        ModelStatus::DOWNLOADING => Some("Model download in progress".to_string()),
                        ModelStatus::ERROR => {
                            Some("Previous download failed - retrying".to_string())
                        }
                    },
                    provider: modelexpress_common::grpc::model::ModelProvider::from(provider)
                        as i32,
                };

                if tx.send(Ok(update)).await.is_err() {
                    return; // Client disconnected
                }

                // If already downloaded, we're done
                if status == ModelStatus::DOWNLOADED {
                    return;
                }
            }

            // Start or monitor the download process
            let final_status = MODEL_TRACKER
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
            .get_model_path(&model_name, cache_dir)
            .await
            .map_err(|e| Status::not_found(format!("Model not found: {e}")))?;

        debug!("Model path resolved to: {:?}", model_path);

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

                    let chunk = FileChunk {
                        relative_path: relative_path.to_string_lossy().to_string(),
                        data: buffer[..bytes_read].to_vec(),
                        offset,
                        total_size: *total_size,
                        is_last_chunk,
                        is_last_file: is_last_file && is_last_chunk,
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

/// Tracks the status of model downloads using `SQLite` for persistence
#[derive(Debug, Clone)]
pub struct ModelDownloadTracker {
    /// `SQLite` database for persistent model status tracking
    database: ModelDatabase,
    /// Maps model names to list of channels waiting for updates
    waiting_channels: WaitingChannels,
}

impl Default for ModelDownloadTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDownloadTracker {
    #[must_use]
    pub fn new() -> Self {
        // Initialize database in the current directory
        let database = match ModelDatabase::new("./models.db") {
            Ok(db) => db,
            Err(e) => {
                error!("Critical error: Could not initialize model database at ./models.db: {e}");
                panic!("Critical error: Could not initialize model database at ./models.db");
            }
        };

        Self {
            database,
            waiting_channels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Gets the status of a model from the database
    /// If the model is not in the database, it returns None
    pub fn get_status(&self, model_name: &str) -> Option<ModelStatus> {
        match self.database.get_status(model_name) {
            Ok(status) => {
                // Update last_used_at when checking status
                if status.is_some() {
                    let _ = self.database.touch_model(model_name);
                }
                status
            }
            Err(e) => {
                error!("Failed to get model status from database: {}", e);
                None
            }
        }
    }

    /// Sets the status of a model and notifies all waiting channels
    pub fn set_status_and_notify(
        &self,
        model_name: String,
        status: ModelStatus,
        provider: ModelProvider,
        message: Option<String>,
    ) {
        // Update status in database
        if let Err(e) = self
            .database
            .set_status(&model_name, provider, status, message.clone())
        {
            error!("Failed to update model status in database: {}", e);
            return;
        }

        // Notify all waiting channels
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

            // If the model is downloaded or errored, remove all waiting channels
            if status == ModelStatus::DOWNLOADED || status == ModelStatus::ERROR {
                waiting.remove(&model_name);
            }
        }
    }

    /// Sets the status of a model
    pub fn set_status(&self, model_name: String, status: ModelStatus, provider: ModelProvider) {
        self.set_status_and_notify(model_name, status, provider, None);
    }

    /// Adds a channel to wait for updates on a specific model
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

    /// Deletes the status of a model from the database
    /// This is used when a model is removed from the tracker
    pub fn delete_status(&self, model_name: &str) {
        if let Err(e) = self.database.delete_model(model_name) {
            error!("Failed to delete model from database: {}", e);
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

    /// Initiates a download for a model and streams status updates
    pub async fn ensure_model_downloaded(
        &self,
        model_name: &str,
        provider: ModelProvider,
        tx: &tokio::sync::mpsc::Sender<Result<ModelStatusUpdate, Status>>,
        ignore_weights: bool,
    ) -> ModelStatus {
        // Atomically try to claim this model for download using compare-and-swap
        let status = match self.database.try_claim_for_download(model_name, provider) {
            Ok(status) => status,
            Err(e) => {
                error!("Failed to claim model for download: {}", e);
                // Send error and return
                let error_update = ModelStatusUpdate {
                    model_name: model_name.to_string(),
                    status: modelexpress_common::grpc::model::ModelStatus::from(ModelStatus::ERROR)
                        as i32,
                    message: Some("Database error occurred".to_string()),
                    provider: modelexpress_common::grpc::model::ModelProvider::from(provider)
                        as i32,
                };
                let _ = tx.send(Ok(error_update)).await;
                return ModelStatus::ERROR;
            }
        };

        // Send current status
        let update = ModelStatusUpdate {
            model_name: model_name.to_string(),
            status: modelexpress_common::grpc::model::ModelStatus::from(status) as i32,
            message: match status {
                ModelStatus::DOWNLOADED => Some("Model already downloaded".to_string()),
                ModelStatus::DOWNLOADING => Some("Model download in progress".to_string()),
                ModelStatus::ERROR => Some("Previous download failed - retrying".to_string()),
            },
            provider: modelexpress_common::grpc::model::ModelProvider::from(provider) as i32,
        };

        let _ = tx.send(Ok(update)).await;

        // If the model already existed and is downloading, add this channel to wait for updates
        if status == ModelStatus::DOWNLOADING {
            self.add_waiting_channel(model_name, tx.clone());

            // Check if we were the ones who just claimed it vs. it was already downloading
            // If we just claimed it, we need to start the actual download
            // We can determine this by checking if there are any waiting channels yet
            let should_start_download = {
                let waiting = match self.waiting_channels.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        error!("Waiting channels mutex is poisoned, recovering");
                        poisoned.into_inner()
                    }
                };
                waiting
                    .get(model_name)
                    .is_none_or(|channels| channels.len() <= 1)
            };

            if should_start_download {
                // We claimed the model, so we're responsible for downloading it
                let tracker = self.clone();
                let model_name_owned = model_name.to_string();

                // Perform the download in the background
                tokio::spawn(async move {
                    let cache_dir = get_server_cache_dir();
                    match download::download_model(
                        &model_name_owned,
                        provider,
                        cache_dir,
                        ignore_weights,
                    )
                    .await
                    {
                        Ok(_path) => {
                            // Download completed successfully
                            tracker.set_status_and_notify(
                                model_name_owned,
                                ModelStatus::DOWNLOADED,
                                provider,
                                Some("Model download completed successfully".to_string()),
                            );
                        }
                        Err(e) => {
                            // Download failed
                            error!("Failed to download model {model_name_owned}: {e}");
                            tracker.set_status_and_notify(
                                model_name_owned,
                                ModelStatus::ERROR,
                                provider,
                                Some(format!("Download failed: {e}")),
                            );
                        }
                    }
                });
            }

            // Wait for completion by monitoring the status
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if let Some(current_status) = self.get_status(model_name)
                    && current_status != ModelStatus::DOWNLOADING
                {
                    return current_status;
                }
            }
        } else if status == ModelStatus::ERROR {
            // If the model is in ERROR status, try to retry the download
            // First, reset the status to DOWNLOADING
            if let Err(e) = self.database.set_status(
                model_name,
                provider,
                ModelStatus::DOWNLOADING,
                Some("Retrying download...".to_string()),
            ) {
                error!("Failed to reset status for retry: {}", e);
                return ModelStatus::ERROR;
            }

            // Add this channel to wait for updates
            self.add_waiting_channel(model_name, tx.clone());

            // Start the download
            let tracker = self.clone();
            let model_name_owned = model_name.to_string();

            tokio::spawn(async move {
                let cache_dir = get_server_cache_dir();
                match download::download_model(
                    &model_name_owned,
                    provider,
                    cache_dir,
                    ignore_weights,
                )
                .await
                {
                    Ok(_path) => {
                        // Download completed successfully
                        tracker.set_status_and_notify(
                            model_name_owned,
                            ModelStatus::DOWNLOADED,
                            provider,
                            Some("Model download completed successfully".to_string()),
                        );
                    }
                    Err(e) => {
                        // Download failed again
                        error!("Failed to download model {model_name_owned} on retry: {e}");
                        tracker.set_status_and_notify(
                            model_name_owned,
                            ModelStatus::ERROR,
                            provider,
                            Some(format!("Download failed on retry: {e}")),
                        );
                    }
                }
            });

            // Wait for completion by monitoring the status
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if let Some(current_status) = self.get_status(model_name)
                    && current_status != ModelStatus::DOWNLOADING
                {
                    return current_status;
                }
            }
        }

        status
    }
}

/// Global model download tracker
pub static MODEL_TRACKER: std::sync::LazyLock<ModelDownloadTracker> =
    std::sync::LazyLock::new(ModelDownloadTracker::new);

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::{
        api::ApiRequest, health::HealthRequest, model::ModelDownloadRequest,
    };
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

    #[test]
    fn test_model_download_tracker_new() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");
        let tracker = ModelDownloadTracker::new();

        // Test that we can get status for a non-existent model
        let status = tracker.get_status("non-existent-model");
        assert!(status.is_none());
    }

    #[test]
    fn test_model_download_tracker_set_and_get_status() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");
        let tracker = ModelDownloadTracker::new();

        // Use a unique model name based on current time to avoid conflicts
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos();
        let model_name = format!("test-model-{timestamp}");
        let provider = ModelProvider::HuggingFace;

        // Initially should return None
        assert!(tracker.get_status(&model_name).is_none());

        // Set status
        tracker.set_status(model_name.clone(), ModelStatus::DOWNLOADING, provider);

        // Should now return the status
        let status = tracker.get_status(&model_name);
        assert!(status.is_some());
        assert_eq!(
            status.expect("Status should be present"),
            ModelStatus::DOWNLOADING
        );

        // Cleanup
        tracker.delete_status(&model_name);
    }

    #[test]
    fn test_model_download_tracker_delete_status() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");
        let tracker = ModelDownloadTracker::new();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos();
        let model_name = format!("test-delete-model-{timestamp}");
        let provider = ModelProvider::HuggingFace;

        // Set status
        tracker.set_status(model_name.clone(), ModelStatus::DOWNLOADED, provider);
        assert!(tracker.get_status(&model_name).is_some());

        // Delete status
        tracker.delete_status(&model_name);
        assert!(tracker.get_status(&model_name).is_none());
    }

    #[tokio::test]
    async fn test_model_service_already_downloaded() {
        let service = ModelServiceImpl;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos();
        let model_name = format!("test-already-downloaded-model-{timestamp}");

        // Pre-populate the model as downloaded
        MODEL_TRACKER.set_status(
            model_name.clone(),
            ModelStatus::DOWNLOADED,
            ModelProvider::HuggingFace,
        );

        let request = Request::new(ModelDownloadRequest {
            model_name: model_name.clone(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            ignore_weights: false,
        });

        let response = service.ensure_model_downloaded(request).await;
        assert!(response.is_ok());

        let mut stream = response.expect("Response should be ok").into_inner();

        // Should get at least one update indicating it's already downloaded
        let update = stream.next().await;
        assert!(update.is_some());

        let update = update.expect("Update should be present");
        assert!(update.is_ok());

        let status_update = update.expect("Status update should be ok");
        assert_eq!(status_update.model_name, model_name);
        assert_eq!(
            status_update.status,
            modelexpress_common::grpc::model::ModelStatus::Downloaded as i32
        );

        // Cleanup
        MODEL_TRACKER.delete_status(&model_name);
    }

    #[test]
    fn test_model_download_tracker_set_status_and_notify() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");
        let tracker = ModelDownloadTracker::new();
        let model_name = "test-notify-model".to_string();
        let provider = ModelProvider::HuggingFace;

        // Test set_status_and_notify doesn't panic
        tracker.set_status_and_notify(
            model_name.clone(),
            ModelStatus::DOWNLOADED,
            provider,
            Some("Download completed".to_string()),
        );

        // Verify status was set
        let status = tracker.get_status(&model_name);
        assert!(status.is_some());
        assert_eq!(
            status.expect("Status should be present"),
            ModelStatus::DOWNLOADED
        );
    }

    #[test]
    fn test_waiting_channels_management() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");
        let tracker = ModelDownloadTracker::new();
        let model_name = "test-channels-model";

        let (tx, _rx) = tokio::sync::mpsc::channel(4);

        // Add a waiting channel
        tracker.add_waiting_channel(model_name, tx);

        // Verify the channel was added by checking internal state
        let waiting_count = {
            let waiting = match tracker.waiting_channels.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            waiting.get(model_name).map_or(0, std::vec::Vec::len)
        };
        assert_eq!(waiting_count, 1);

        // Clean up by setting final status
        tracker.set_status_and_notify(
            model_name.to_string(),
            ModelStatus::DOWNLOADED,
            ModelProvider::HuggingFace,
            None,
        );

        // Channels should be cleared for final statuses
        let waiting_count_after = {
            let waiting = match tracker.waiting_channels.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            waiting.get(model_name).map_or(0, std::vec::Vec::len)
        };
        assert_eq!(waiting_count_after, 0);
    }

    #[tokio::test]
    async fn test_model_service_stream_closes_properly() {
        let service = ModelServiceImpl;
        let model_name = "test-stream-model";

        let request = Request::new(ModelDownloadRequest {
            model_name: model_name.to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::HuggingFace as i32,
            ignore_weights: false,
        });

        let response = service.ensure_model_downloaded(request).await;
        assert!(response.is_ok());

        let mut stream = response.expect("Response should be ok").into_inner();

        // Read a few updates (may include initial status and progress)
        let mut update_count = 0;
        while let Some(update) = stream.next().await {
            assert!(update.is_ok());
            update_count += 1;

            // Prevent infinite loop in case of issues
            if update_count > 10 {
                break;
            }
        }

        assert!(update_count > 0);

        // Cleanup
        MODEL_TRACKER.delete_status(model_name);
    }

    #[tokio::test]
    async fn test_concurrent_model_download_no_race_condition() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");
        let tracker = ModelDownloadTracker::new();
        let model_name = "test-concurrent-model";
        let provider = ModelProvider::HuggingFace;

        // Test that the compare-and-swap mechanism works
        // First attempt should claim the model
        let status1 = tracker
            .database
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 1");
        assert_eq!(status1, ModelStatus::DOWNLOADING);

        // Second attempt should see it's already claimed
        let status2 = tracker
            .database
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 2");
        assert_eq!(status2, ModelStatus::DOWNLOADING);

        // Verify only one record exists
        let record = tracker
            .database
            .get_model_record(model_name)
            .expect("Failed to get model record")
            .expect("Record should exist");
        assert_eq!(record.status, ModelStatus::DOWNLOADING);

        // Cleanup
        tracker.delete_status(model_name);
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
}
