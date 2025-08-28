// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::database::ModelDatabase;
use modelexpress-common::{
    cache::CacheConfig,
    download,
    grpc::{
        api::{ApiRequest, ApiResponse, api_service_server::ApiService},
        health::{HealthRequest, HealthResponse, health_service_server::HealthService},
        model::{ModelDownloadRequest, ModelStatusUpdate, model_service_server::ModelService},
    },
    models::{ModelProvider, ModelStatus},
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::SystemTime,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{error, info};

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

#[tonic::async_trait]
impl ModelService for ModelServiceImpl {
    type EnsureModelDownloadedStream = ReceiverStream<Result<ModelStatusUpdate, Status>>;

    async fn ensure_model_downloaded(
        &self,
        request: Request<ModelDownloadRequest>,
    ) -> Result<Response<Self::EnsureModelDownloadedStream>, Status> {
        let model_request = request.into_inner();
        info!(
            "Starting model download stream for: {} from provider: {:?}",
            model_request.model_name, model_request.provider
        );

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let model_name = model_request.model_name.clone();

        // Convert gRPC provider to our enum
        let provider: ModelProvider =
            modelexpress-common::grpc::model::ModelProvider::try_from(model_request.provider)
                .unwrap_or(modelexpress-common::grpc::model::ModelProvider::HuggingFace)
                .into();

        // Spawn a task to handle the streaming download updates
        tokio::spawn(async move {
            // Check if the model is already downloaded
            if let Some(status) = MODEL_TRACKER.get_status(&model_name) {
                let update = ModelStatusUpdate {
                    model_name: model_name.clone(),
                    status: modelexpress-common::grpc::model::ModelStatus::from(status) as i32,
                    message: match status {
                        ModelStatus::DOWNLOADED => Some("Model already downloaded".to_string()),
                        ModelStatus::DOWNLOADING => Some("Model download in progress".to_string()),
                        ModelStatus::ERROR => Some("Previous download failed".to_string()),
                    },
                    provider: modelexpress-common::grpc::model::ModelProvider::from(provider)
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
                .ensure_model_downloaded(&model_name, provider, &tx)
                .await;

            // Send final status update
            let final_update = ModelStatusUpdate {
                model_name: model_name.clone(),
                status: modelexpress-common::grpc::model::ModelStatus::from(final_status) as i32,
                message: match final_status {
                    ModelStatus::DOWNLOADED => {
                        Some("Model download completed successfully".to_string())
                    }
                    ModelStatus::ERROR => Some("Model download failed".to_string()),
                    ModelStatus::DOWNLOADING => Some("Download still in progress".to_string()),
                },
                provider: modelexpress-common::grpc::model::ModelProvider::from(provider) as i32,
            };

            let _ = tx.send(Ok(final_update)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
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
                status: modelexpress-common::grpc::model::ModelStatus::from(status) as i32,
                message,
                provider: modelexpress-common::grpc::model::ModelProvider::from(provider) as i32,
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
    ) -> ModelStatus {
        // Atomically try to claim this model for download using compare-and-swap
        let status = match self.database.try_claim_for_download(model_name, provider) {
            Ok(status) => status,
            Err(e) => {
                error!("Failed to claim model for download: {}", e);
                // Send error and return
                let error_update = ModelStatusUpdate {
                    model_name: model_name.to_string(),
                    status: modelexpress-common::grpc::model::ModelStatus::from(ModelStatus::ERROR)
                        as i32,
                    message: Some("Database error occurred".to_string()),
                    provider: modelexpress-common::grpc::model::ModelProvider::from(provider)
                        as i32,
                };
                let _ = tx.send(Ok(error_update)).await;
                return ModelStatus::ERROR;
            }
        };

        // Send current status
        let update = ModelStatusUpdate {
            model_name: model_name.to_string(),
            status: modelexpress-common::grpc::model::ModelStatus::from(status) as i32,
            message: match status {
                ModelStatus::DOWNLOADED => Some("Model already downloaded".to_string()),
                ModelStatus::DOWNLOADING => Some("Model download in progress".to_string()),
                ModelStatus::ERROR => Some("Previous download failed".to_string()),
            },
            provider: modelexpress-common::grpc::model::ModelProvider::from(provider) as i32,
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
                    match download::download_model(&model_name_owned, provider, cache_dir).await {
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
                if let Some(current_status) = self.get_status(model_name) {
                    if current_status != ModelStatus::DOWNLOADING {
                        return current_status;
                    }
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
    use modelexpress-common::grpc::{
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
            provider: modelexpress-common::grpc::model::ModelProvider::HuggingFace as i32,
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
            modelexpress-common::grpc::model::ModelStatus::Downloaded as i32
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
            provider: modelexpress-common::grpc::model::ModelProvider::HuggingFace as i32,
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
}
