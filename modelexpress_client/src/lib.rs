// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use modelexpress_common::providers::ModelProviderTrait;
use modelexpress_common::{
    Result as CommonResult,
    cache::{CacheConfig, CacheStats},
    client_config::ClientConfig as Config,
    constants, download,
    grpc::{
        api::{ApiRequest, api_service_client::ApiServiceClient},
        health::{HealthRequest, health_service_client::HealthServiceClient},
        model::{
            ModelDownloadRequest, ModelFilesRequest, model_service_client::ModelServiceClient,
        },
    },
    models::{ModelStatus, Status},
    providers::huggingface::HuggingFaceProvider,
};
use std::collections::HashMap;
use std::path::{Component, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tonic::transport::Channel;
use tracing::debug;
use tracing::info;
#[cfg(test)]
use tracing::warn;
use uuid::Uuid;

// Re-export for public use
pub use modelexpress_common::client_config::{ClientArgs, ClientConfig};
pub use modelexpress_common::models::ModelProvider;

/// The main client for interacting with the `modelexpress_server` via gRPC
pub struct Client {
    health_client: HealthServiceClient<Channel>,
    api_client: ApiServiceClient<Channel>,
    model_client: ModelServiceClient<Channel>,
    cache_config: Option<CacheConfig>,
}

impl Client {
    /// Create a new client with the given configuration
    pub async fn new(config: Config) -> CommonResult<Self> {
        let endpoint = &config.connection.endpoint;
        let timeout = config
            .connection
            .timeout_secs
            .unwrap_or(constants::DEFAULT_TIMEOUT_SECS);

        let channel = tonic::transport::Endpoint::new(endpoint.clone())
            .map(|endpoint| endpoint.timeout(Duration::from_secs(timeout)))?
            .connect()
            .await?;

        let health_client = HealthServiceClient::new(channel.clone());
        let api_client = ApiServiceClient::new(channel.clone());
        let model_client = ModelServiceClient::new(channel);

        // Use the cache config from the client configuration
        let cache_config = Some(config.cache.clone());

        Ok(Self {
            health_client,
            api_client,
            model_client,
            cache_config,
        })
    }

    /// Create a new client with cache configuration
    pub async fn new_with_cache(config: Config, cache_config: CacheConfig) -> CommonResult<Self> {
        let mut client = Self::new(config).await?;
        client.cache_config = Some(cache_config);
        Ok(client)
    }

    /// Get cache configuration
    pub fn get_cache_config(&self) -> Option<&CacheConfig> {
        self.cache_config.as_ref()
    }

    /// Set cache configuration
    pub fn set_cache_config(&mut self, cache_config: CacheConfig) {
        self.cache_config = Some(cache_config);
    }

    /// List cached models
    pub fn list_cached_models(&self) -> CommonResult<CacheStats> {
        let cache_config = self.cache_config.as_ref().ok_or_else(|| {
            modelexpress_common::Error::Server("Cache not configured".to_string())
        })?;

        cache_config.get_cache_stats().map_err(|e| {
            modelexpress_common::Error::Server(format!("Failed to get cache stats: {e}")).into()
        })
    }

    /// Clear specific model from cache
    pub fn clear_cached_model(&self, model_name: &str) -> CommonResult<()> {
        let cache_config = self.cache_config.as_ref().ok_or_else(|| {
            modelexpress_common::Error::Server("Cache not configured".to_string())
        })?;

        cache_config.clear_model(model_name).map_err(|e| {
            modelexpress_common::Error::Server(format!("Failed to clear model: {e}")).into()
        })
    }

    /// Clear entire cache
    pub fn clear_all_cached_models(&self) -> CommonResult<()> {
        let cache_config = self.cache_config.as_ref().ok_or_else(|| {
            modelexpress_common::Error::Server("Cache not configured".to_string())
        })?;

        cache_config.clear_all().map_err(|e| {
            modelexpress_common::Error::Server(format!("Failed to clear cache: {e}")).into()
        })
    }

    /// Get the cache directory path, using client's cache config if available,
    /// otherwise discovering from environment variables and defaults.
    fn get_cache_dir(&self) -> PathBuf {
        // Check Hugging Face's HF_HUB_CACHE environment variable first,
        // as it takes precedence, and isn't specific to our client.
        use std::env;
        if let Ok(cache_path) = env::var("HF_HUB_CACHE") {
            return PathBuf::from(cache_path);
        }
        // Use client's cache config if available
        if let Some(cache_config) = &self.cache_config {
            return cache_config.local_path.clone();
        }

        // Fall back to discovery
        CacheConfig::discover()
            .map(|config| config.local_path)
            .unwrap_or_else(|_| CacheConfig::default().local_path)
    }

    pub async fn get_model_path(&self, model_name: &str) -> anyhow::Result<PathBuf> {
        let cache_dir = self.get_cache_dir();
        let model_path = HuggingFaceProvider
            .get_model_path(model_name, cache_dir)
            .await
            .map_err(|e| anyhow::anyhow!(format!("Failed to get model path: {e}")))?;

        debug!("Found model path at {:?}", model_path);
        Ok(model_path)
    }

    /// Pre-download model to cache
    pub async fn preload_model_to_cache(
        &mut self,
        model_name: &str,
        provider: ModelProvider,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        info!("Pre-loading model {} to cache", model_name);

        // First try to download via server
        match self
            .request_model_with_provider(model_name, provider, ignore_weights)
            .await
        {
            Ok(()) => {
                info!("Model {} pre-loaded successfully via server", model_name);
                Ok(())
            }
            Err(e) => {
                // Fallback to direct download
                info!(
                    "Server unavailable, pre-loading model {} directly. Error: {}",
                    model_name, e
                );
                Self::download_model_directly(model_name, provider, ignore_weights).await
            }
        }
    }

    /// Get the server health status
    pub async fn health_check(&mut self) -> CommonResult<Status> {
        let request = tonic::Request::new(HealthRequest {});

        let response = self.health_client.get_health(request).await?;
        let health_response = response.into_inner();

        Ok(health_response.into())
    }

    /// Send a request to the server
    pub async fn send_request<T: serde::de::DeserializeOwned>(
        &mut self,
        action: impl Into<String>,
        payload: Option<HashMap<String, serde_json::Value>>,
    ) -> CommonResult<T> {
        let request_id = Uuid::new_v4().to_string();

        let payload_bytes = if let Some(payload) = payload {
            Some(
                serde_json::to_vec(&payload)
                    .map_err(|e| modelexpress_common::Error::Serialization(e.to_string()))?,
            )
        } else {
            None
        };

        let grpc_request = tonic::Request::new(ApiRequest {
            id: request_id,
            action: action.into(),
            payload: payload_bytes,
        });

        let response = self.api_client.send_request(grpc_request).await?;
        let api_response = response.into_inner();

        if !api_response.success {
            return Err(modelexpress_common::Error::Server(
                api_response
                    .error
                    .unwrap_or_else(|| "Unknown server error".to_string()),
            )
            .into());
        }

        let data_bytes = api_response.data.ok_or_else(|| {
            modelexpress_common::Error::Server("Server returned success but no data".to_string())
        })?;

        let data: T = serde_json::from_slice(&data_bytes)
            .map_err(|e| modelexpress_common::Error::Serialization(e.to_string()))?;

        Ok(data)
    }

    /// Request a model from the server with a specific provider and automatic fallback
    /// This function will first try to use the server for streaming downloads.
    /// If the server is unavailable, it will fallback to downloading directly.
    /// When shared_storage is disabled, files will be streamed from server to client.
    pub async fn request_model_with_provider_and_fallback(
        &mut self,
        model_name: impl Into<String>,
        provider: ModelProvider,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        let model_name = model_name.into();

        // First try the server-based approach
        match self
            .request_model_with_provider(&model_name, provider, ignore_weights)
            .await
        {
            Ok(()) => {
                info!("Model {} downloaded successfully via server", model_name);

                // Check if we need to stream files from server (no shared storage)
                let needs_streaming = self
                    .cache_config
                    .as_ref()
                    .is_some_and(|c| !c.shared_storage);

                if needs_streaming {
                    info!(
                        "Shared storage disabled, streaming files from server for model {}",
                        model_name
                    );
                    self.stream_model_files_from_server(&model_name, provider)
                        .await?;
                }

                Ok(())
            }
            Err(e) => {
                // Check if it's a connection error (server not available)
                if let modelexpress_common::Error::Transport(_) = *e {
                    info!(
                        "Server unavailable, falling back to direct download for model: {}",
                        model_name
                    );

                    // Fallback to direct download
                    let cache_dir = CacheConfig::discover().ok().map(|config| config.local_path);
                    match download::download_model(&model_name, provider, cache_dir, ignore_weights).await {
                        Ok(_) => {
                            info!(
                                "Model {} downloaded successfully via direct download",
                                model_name
                            );
                            Ok(())
                        }
                        Err(download_err) => Err(modelexpress_common::Error::Server(format!(
                            "Both server and direct download failed. Server error: {e}. Download error: {download_err}"
                        )).into()),
                    }
                } else {
                    // For other types of errors, don't fallback
                    Err(e)
                }
            }
        }
    }

    /// Stream model files from the server to the local cache
    /// This is used when shared storage is disabled
    pub async fn stream_model_files_from_server(
        &mut self,
        model_name: &str,
        provider: ModelProvider,
    ) -> CommonResult<()> {
        let cache_config = self.cache_config.as_ref().ok_or_else(|| {
            modelexpress_common::Error::Server(
                "Cache configuration is required for file streaming. Please ensure cache_config is set.".to_string()
            )
        })?;

        let chunk_size = cache_config.transfer_chunk_size as u32;
        let local_cache_path = cache_config.local_path.clone();

        info!(
            "Streaming model {} files to {:?} with chunk size {} bytes",
            model_name, local_cache_path, chunk_size
        );

        let grpc_request = tonic::Request::new(ModelFilesRequest {
            model_name: model_name.to_string(),
            provider: modelexpress_common::grpc::model::ModelProvider::from(provider) as i32,
            chunk_size,
        });

        let mut stream = self
            .model_client
            .stream_model_files(grpc_request)
            .await?
            .into_inner();

        // Create a normalized model directory path (similar to HuggingFace cache structure)
        let normalized_name = model_name.replace('/', "--");
        let snapshots_dir = local_cache_path
            .join(format!("models--{}", normalized_name))
            .join("snapshots");

        // Model directory will be set after receiving the first chunk with commit hash
        let mut model_dir: Option<PathBuf> = None;
        let mut canonical_model_dir: Option<PathBuf> = None;

        let mut current_file: Option<(PathBuf, tokio::fs::File)> = None;
        let mut files_received: u64 = 0;
        let mut bytes_received: u64 = 0;

        while let Some(chunk_result) = stream.message().await? {
            // Extract commit hash from first chunk and create model directory
            if model_dir.is_none() {
                let commit_hash = chunk_result.commit_hash.as_ref().ok_or_else(|| {
                    modelexpress_common::Error::Server(
                        "Server did not provide commit hash in first chunk".to_string(),
                    )
                })?;

                let dir = snapshots_dir.join(commit_hash);
                std::fs::create_dir_all(&dir).map_err(|e| {
                    modelexpress_common::Error::Server(format!(
                        "Failed to create model directory: {e}"
                    ))
                })?;

                // Canonicalize model_dir once for efficiency and security
                let canonical_dir = dir.canonicalize().map_err(|e| {
                    modelexpress_common::Error::Server(format!(
                        "Failed to canonicalize model directory: {e}"
                    ))
                })?;

                model_dir = Some(dir);
                canonical_model_dir = Some(canonical_dir);
            }

            let model_dir_ref = model_dir.as_ref().ok_or_else(|| {
                modelexpress_common::Error::Server("Model directory not initialized".to_string())
            })?;
            let canonical_model_dir_ref = canonical_model_dir.as_ref().ok_or_else(|| {
                modelexpress_common::Error::Server(
                    "Canonical model directory not initialized".to_string(),
                )
            })?;

            let relative_path = PathBuf::from(&chunk_result.relative_path);

            // Validate that the relative path does not contain any '..' components or is absolute
            let is_safe = !relative_path.components().any(|c| {
                matches!(
                    c,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            });

            if !is_safe {
                return Err(Box::new(modelexpress_common::Error::Server(format!(
                    "Received potentially unsafe file path from server: {:?}",
                    chunk_result.relative_path
                ))));
            }

            let file_path = model_dir_ref.join(&relative_path);

            // Verify that the resolved path is still within model_dir
            // Create parent directory first if it doesn't exist to enable proper validation
            if let Some(parent) = file_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        modelexpress_common::Error::Server(format!(
                            "Failed to create parent directory: {e}"
                        ))
                    })?;
                }

                let canonical_parent = parent.canonicalize().map_err(|e| {
                    modelexpress_common::Error::Server(format!(
                        "Failed to canonicalize parent directory: {e}"
                    ))
                })?;

                if !canonical_parent.starts_with(canonical_model_dir_ref) {
                    return Err(Box::new(modelexpress_common::Error::Server(format!(
                        "Received file path that resolves outside model directory: {:?}",
                        chunk_result.relative_path
                    ))));
                }
            }

            // If this is a new file (different path or first chunk)
            let need_new_file = current_file
                .as_ref()
                .is_none_or(|(path, _)| path != &file_path);

            if need_new_file {
                // Close previous file if any
                if let Some((prev_path, file)) = current_file.take() {
                    drop(file);
                    debug!("Finished writing file: {:?}", prev_path);
                }

                // Create new file (parent directory was already created during validation)
                let file = tokio::fs::File::create(&file_path).await.map_err(|e| {
                    modelexpress_common::Error::Server(format!(
                        "Failed to create file {:?}: {e}",
                        file_path
                    ))
                })?;

                debug!(
                    "Starting to receive file: {:?} ({} bytes)",
                    relative_path, chunk_result.total_size
                );
                files_received = files_received.saturating_add(1);
                current_file = Some((file_path.clone(), file));
            }

            // Write chunk to file
            if let Some((_, ref mut file)) = current_file {
                file.write_all(&chunk_result.data).await.map_err(|e| {
                    modelexpress_common::Error::Server(format!("Failed to write to file: {e}"))
                })?;
                bytes_received = bytes_received.saturating_add(chunk_result.data.len() as u64);
            }

            // Check if we're done with all files
            if chunk_result.is_last_file && chunk_result.is_last_chunk {
                break;
            }
        }

        // Ensure the last file is properly closed
        if let Some((path, file)) = current_file.take() {
            drop(file);
            debug!("Finished writing final file: {:?}", path);
        }

        info!(
            "Streaming complete: received {} files ({} bytes) for model {}",
            files_received, bytes_received, model_name
        );

        Ok(())
    }

    /// Request a model from the server with a specific provider
    /// This function will wait until the model is downloaded using streaming updates
    pub async fn request_model_with_provider(
        &mut self,
        model_name: impl Into<String>,
        provider: ModelProvider,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        let model_name = model_name.into();
        info!(
            "Requesting model: {} from provider: {:?}",
            model_name, provider
        );

        let grpc_request = tonic::Request::new(ModelDownloadRequest {
            model_name: model_name.clone(),
            provider: modelexpress_common::grpc::model::ModelProvider::from(provider) as i32,
            ignore_weights,
        });

        let mut stream = self
            .model_client
            .ensure_model_downloaded(grpc_request)
            .await?
            .into_inner();

        // Process streaming updates until the download is complete
        while let Some(update_result) = stream.message().await? {
            let status: ModelStatus =
                modelexpress_common::grpc::model::ModelStatus::try_from(update_result.status)
                    .unwrap_or(modelexpress_common::grpc::model::ModelStatus::Error)
                    .into();

            // Log progress messages if available
            if let Some(message) = &update_result.message {
                info!("Model {}: {}", model_name, message);
            } else {
                info!("Model {} status: {:?}", model_name, status);
            }

            // Check if we're done
            match status {
                ModelStatus::DOWNLOADED => {
                    info!("Model {} is now available", model_name);
                    return Ok(());
                }
                ModelStatus::ERROR => {
                    let error_message = update_result
                        .message
                        .unwrap_or_else(|| "Unknown error occurred".to_string());
                    return Err(modelexpress_common::Error::Server(format!(
                        "Model download failed: {error_message}"
                    ))
                    .into());
                }
                ModelStatus::DOWNLOADING => {
                    // Continue processing updates
                    continue;
                }
            }
        }

        // If stream ended without DOWNLOADED status, treat as error
        Err(modelexpress_common::Error::Server(
            "Model download stream ended unexpectedly".to_string(),
        )
        .into())
    }

    /// Request a model from the server using the default provider (Hugging Face) with automatic fallback
    /// This function will first try to use the server, then fallback to direct download if needed
    pub async fn request_model(
        &mut self,
        model_name: impl Into<String>,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        self.request_model_with_provider_and_fallback(
            model_name,
            ModelProvider::default(),
            ignore_weights,
        )
        .await
    }

    /// Request a model from the server only (no fallback)
    /// This function will wait until the model is downloaded using streaming updates from the server
    pub async fn request_model_server_only(
        &mut self,
        model_name: impl Into<String>,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        self.request_model_with_provider(model_name, ModelProvider::default(), ignore_weights)
            .await
    }

    /// Request a model with automatic server fallback, creating client connection only if needed
    /// This function will try to download via server if possible, otherwise download directly
    pub async fn request_model_with_smart_fallback(
        model_name: impl Into<String>,
        provider: ModelProvider,
        config: Config,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        let model_name = model_name.into();

        // First try to create a client and use server-based download
        match Client::new(config.clone()).await {
            Ok(mut client) => {
                info!("Server connection established, downloading via server...");
                client
                    .request_model_with_provider_and_fallback(&model_name, provider, ignore_weights)
                    .await
            }
            Err(e) => {
                // If we can't even connect to the server, go straight to direct download
                info!("Cannot connect to server ({}), downloading directly...", e);
                Client::download_model_directly(&model_name, provider, ignore_weights).await
            }
        }
    }

    /// Download a model directly without using the server
    /// This bypasses the server entirely and downloads the model using the specified provider
    pub async fn download_model_directly(
        model_name: impl Into<String>,
        provider: ModelProvider,
        ignore_weights: bool,
    ) -> CommonResult<()> {
        let model_name = model_name.into();
        info!(
            "Downloading model {} directly using provider: {:?}",
            model_name, provider
        );

        // Try to get cache configuration, but don't fail if not available
        let cache_dir = CacheConfig::discover().ok().map(|config| config.local_path);

        download::download_model(&model_name, provider, cache_dir, ignore_weights)
            .await
            .map_err(|e| {
                modelexpress_common::Error::Server(format!("Direct download failed: {e}"))
            })?;

        info!("Model {} downloaded successfully", model_name);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::cache::CacheConfig;
    use std::collections::HashMap;

    fn create_test_client_config() -> ClientConfig {
        ClientConfig::for_testing("http://test-endpoint:1234")
    }

    #[test]
    fn test_client_config_creation() {
        let config = create_test_client_config();
        assert_eq!(config.connection.endpoint, "http://test-endpoint:1234");
    }

    #[test]
    fn test_client_config_default_timeout() {
        let config = create_test_client_config();
        assert!(config.connection.timeout_secs.is_some());
    }

    #[test]
    fn test_client_config_with_timeout() {
        let mut config = create_test_client_config();
        config.connection.timeout_secs = Some(60);
        assert_eq!(config.connection.timeout_secs, Some(60));
    }

    #[test]
    fn test_endpoint_override_priority() {
        // This test verifies that the configuration works correctly
        let config = ClientConfig::for_testing("http://command-line-endpoint:5678");

        // Test that the config has the correct endpoint
        assert_eq!(
            config.connection.endpoint,
            "http://command-line-endpoint:5678"
        );
    }

    #[test]
    fn test_cache_config_endpoint_override() {
        // Test that cache config can be created with a specific endpoint
        let mut cache_config = CacheConfig {
            local_path: std::path::PathBuf::from("/test/path"),
            server_endpoint: "http://original-endpoint:1234".to_string(),
            timeout_secs: None,
            shared_storage: true,
            transfer_chunk_size: modelexpress_common::constants::DEFAULT_TRANSFER_CHUNK_SIZE,
        };

        // Simulate endpoint override
        cache_config.server_endpoint = "http://override-endpoint:5678".to_string();

        assert_eq!(
            cache_config.server_endpoint,
            "http://override-endpoint:5678"
        );
    }

    // Note: Most client tests require a running server, so they would be integration tests
    // These unit tests focus on the configuration and setup logic

    #[tokio::test]
    async fn test_client_new_with_invalid_endpoint() {
        let config = ClientConfig::for_testing("invalid-endpoint");

        let result = Client::new(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_download_model_directly_invalid_model() {
        // This test may fail if network is available and HF is accessible
        // In a real test environment, you might want to mock the download function
        let result = Client::download_model_directly(
            "definitely-not-a-real-model-name-12345",
            ModelProvider::HuggingFace,
            false,
        )
        .await;

        // Should fail for a non-existent model
        assert!(result.is_err());
    }

    #[test]
    fn test_model_provider_enum() {
        let provider = ModelProvider::HuggingFace;
        assert_eq!(provider, ModelProvider::HuggingFace);

        let default_provider = ModelProvider::default();
        assert_eq!(default_provider, ModelProvider::HuggingFace);
    }

    #[test]
    fn test_serialization_payload_creation() {
        let mut payload = HashMap::new();
        payload.insert(
            "key1".to_string(),
            serde_json::Value::String("value1".to_string()),
        );
        payload.insert(
            "key2".to_string(),
            serde_json::Value::Number(serde_json::Number::from(42)),
        );

        let payload_bytes =
            serde_json::to_vec(&payload).expect("Serialization should not fail in test");
        let deserialized: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&payload_bytes)
                .expect("Deserialization should not fail in test");

        assert_eq!(payload, deserialized);
    }

    #[tokio::test]
    async fn test_request_model_with_smart_fallback_no_server() {
        // Test the smart fallback when server is not available
        let config = Config::for_testing("http://127.0.0.1:99999"); // Invalid port

        let result = Client::request_model_with_smart_fallback(
            "definitely-not-a-real-model-name-12345",
            ModelProvider::HuggingFace,
            config,
            false,
        )
        .await;

        // Should fail because the model doesn't exist, but we should get past the connection attempt
        assert!(result.is_err());
        // The error should indicate it tried to connect to the server but failed
        let error_msg = result.expect_err("Result should be an error").to_string();
        assert!(
            error_msg.contains("Direct download failed")
                || error_msg.contains("Cannot connect")
                || error_msg.contains("gRPC error")
                || error_msg.contains("h2 protocol error")
                || error_msg.contains("connection")
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod integration_tests {
    use super::*;
    use modelexpress_common::constants;
    use std::time::Duration;
    use tokio::time::timeout;

    // These tests require a running server and are more appropriate as integration tests
    // They are included here but should be run separately from unit tests

    async fn wait_for_server_startup() {
        // Give the server time to start up in CI environments
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    async fn try_create_client() -> Option<Client> {
        let config =
            Config::for_testing(format!("http://127.0.0.1:{}", constants::DEFAULT_GRPC_PORT));

        match timeout(Duration::from_secs(2), Client::new(config)).await {
            Ok(Ok(client)) => Some(client),
            _ => None,
        }
    }

    #[tokio::test]
    #[ignore = "Ignore by default since it requires a running server"]
    async fn test_integration_health_check() {
        wait_for_server_startup().await;

        if let Some(mut client) = try_create_client().await {
            let result = client.health_check().await;
            assert!(result.is_ok());

            let status = result.expect("Health check should succeed");
            assert!(!status.version.is_empty());
            assert_eq!(status.status, "ok");
            // uptime is u64, so it's always >= 0, just check it exists
            let _uptime = status.uptime;
        } else {
            info!("Skipping integration test - server not available");
        }
    }

    #[tokio::test]
    #[ignore = "Ignore by default since it requires a running server"]
    async fn test_integration_ping_request() {
        wait_for_server_startup().await;

        if let Some(mut client) = try_create_client().await {
            let result: Result<serde_json::Value, _> = client.send_request("ping", None).await;
            assert!(result.is_ok());

            let response = result.expect("Ping request should succeed");
            assert_eq!(response["message"], "pong");
        } else {
            info!("Skipping integration test - server not available");
        }
    }

    #[tokio::test]
    #[ignore = "Ignore by default since it requires a running server and network access"]
    async fn test_integration_model_request_tiny_model() {
        wait_for_server_startup().await;

        if let Some(mut client) = try_create_client().await {
            // Use a very small model for testing to avoid long download times
            // Note: This test might still take time and requires network access
            let result = client
                .request_model_server_only("sentence-transformers/all-MiniLM-L6-v2", false)
                .await;

            // We don't assert success here because it depends on network availability
            // In a real integration test environment, you might use a mock model
            match result {
                Ok(()) => info!("Model download successful"),
                Err(e) => warn!("Model download failed (expected in unit test env): {e}"),
            }
        } else {
            info!("Skipping integration test - server not available");
        }
    }
}
