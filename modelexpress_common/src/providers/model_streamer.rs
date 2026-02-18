// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::{GcsCredentials, S3Credentials, SidecarConfig};
use crate::providers::ModelProviderTrait;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Model Streamer provider implementation.
/// This provider communicates with a Python sidecar that uses the Model Streamer SDK
/// to download models from S3, GCS, or S3-compatible storage (like MinIO).
pub struct ModelStreamerProvider {
    client: reqwest::Client,
    config: SidecarConfig,
    s3_credentials: S3Credentials,
    gcs_credentials: GcsCredentials,
}

/// Request to download a model from the sidecar
#[derive(Debug, Serialize)]
struct DownloadRequest {
    model_path: String,
    cache_dir: String,
    ignore_weights: bool,
    s3_credentials: Option<S3CredentialsPayload>,
    gcs_credentials: Option<GcsCredentialsPayload>,
}

/// S3 credentials payload for sidecar requests
#[derive(Debug, Serialize)]
struct S3CredentialsPayload {
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    region: Option<String>,
    endpoint: Option<String>,
}

/// GCS credentials payload for sidecar requests
#[derive(Debug, Serialize)]
struct GcsCredentialsPayload {
    credentials_file: Option<String>,
    credentials_json: Option<String>,
}

/// Response from the sidecar download endpoint
#[derive(Debug, Deserialize)]
struct DownloadResponse {
    success: bool,
    local_path: Option<String>,
    files: Option<Vec<String>>,
    #[allow(dead_code)]
    total_size: Option<u64>,
    error: Option<String>,
    #[allow(dead_code)]
    error_code: Option<String>,
}

/// Response from the sidecar get model endpoint
#[derive(Debug, Deserialize)]
struct GetModelResponse {
    exists: bool,
    local_path: Option<String>,
    #[allow(dead_code)]
    files: Option<Vec<String>>,
    #[allow(dead_code)]
    total_size: Option<u64>,
    error: Option<String>,
}

/// Response from the sidecar delete endpoint
#[derive(Debug, Deserialize)]
struct DeleteResponse {
    success: bool,
    #[allow(dead_code)]
    deleted_path: Option<String>,
    error: Option<String>,
}

/// Response from the sidecar health endpoint
#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    #[allow(dead_code)]
    version: Option<String>,
}

impl ModelStreamerProvider {
    /// Create a new ModelStreamerProvider with the given configuration
    pub fn new(
        config: SidecarConfig,
        s3_credentials: S3Credentials,
        gcs_credentials: GcsCredentials,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .context("Failed to create HTTP client for Model Streamer sidecar")?;

        Ok(Self {
            client,
            config,
            s3_credentials,
            gcs_credentials,
        })
    }

    /// Create a new ModelStreamerProvider from environment variables
    pub fn from_env() -> Result<Self> {
        let config = SidecarConfig::from_env();
        let s3_credentials = S3Credentials::from_env();
        let gcs_credentials = GcsCredentials::from_env();

        Self::new(config, s3_credentials, gcs_credentials)
    }

    /// Check if the sidecar is healthy
    #[allow(dead_code)]
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.config.endpoint);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to connect to Model Streamer sidecar")?;

        if !response.status().is_success() {
            return Ok(false);
        }

        let health: HealthResponse = response
            .json()
            .await
            .context("Failed to parse health response")?;

        Ok(health.status == "healthy")
    }

    /// Encode a model URI as a path-safe identifier
    /// E.g., "s3://bucket/path/to/model" -> "s3/bucket/path/to/model"
    ///
    /// For s3+http/s3+https URIs (MinIO), the endpoint is stripped and the
    /// scheme is normalized to "s3" so the cache path matches the Python
    /// sidecar layout: s3+http://endpoint:port/bucket/path -> s3/bucket/path
    fn encode_model_id(model_path: &str) -> String {
        // s3+http://endpoint:port/bucket/path -> strip endpoint, keep bucket/path
        if let Some(rest) = model_path
            .strip_prefix("s3+http://")
            .or_else(|| model_path.strip_prefix("s3+https://"))
        {
            // rest = "endpoint:port/bucket/path" - skip the endpoint (first segment)
            if let Some(pos) = rest.find('/') {
                return format!("s3/{}", &rest[pos + 1..]);
            }
            return "s3/".to_string();
        }

        model_path
            .replace("s3://", "s3/")
            .replace("gs://", "gs/")
    }

    /// Build S3 credentials payload for the request
    fn build_s3_credentials(&self) -> Option<S3CredentialsPayload> {
        if self.s3_credentials.access_key_id.is_some()
            || self.s3_credentials.secret_access_key.is_some()
            || self.s3_credentials.region.is_some()
            || self.s3_credentials.endpoint.is_some()
        {
            Some(S3CredentialsPayload {
                access_key_id: self.s3_credentials.access_key_id.clone(),
                secret_access_key: self.s3_credentials.secret_access_key.clone(),
                region: self.s3_credentials.region.clone(),
                endpoint: self.s3_credentials.endpoint.clone(),
            })
        } else {
            None
        }
    }

    /// Build GCS credentials payload for the request
    fn build_gcs_credentials(&self) -> Option<GcsCredentialsPayload> {
        if self.gcs_credentials.credentials_file.is_some()
            || self.gcs_credentials.credentials_json.is_some()
        {
            Some(GcsCredentialsPayload {
                credentials_file: self.gcs_credentials.credentials_file.clone(),
                credentials_json: self.gcs_credentials.credentials_json.clone(),
            })
        } else {
            None
        }
    }
}

#[async_trait::async_trait]
impl ModelProviderTrait for ModelStreamerProvider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_path: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        info!("Downloading model from Model Streamer: {}", model_name);

        let cache_dir = cache_path
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| {
                std::env::var("MODEL_EXPRESS_CACHE_DIRECTORY")
                    .unwrap_or_else(|_| "/app/cache".to_string())
            });

        let request = DownloadRequest {
            model_path: model_name.to_string(),
            cache_dir: cache_dir.clone(),
            ignore_weights,
            s3_credentials: self.build_s3_credentials(),
            gcs_credentials: self.build_gcs_credentials(),
        };

        debug!("Sending download request to sidecar: {:?}", request);

        let url = format!("{}/api/v1/download", self.config.endpoint);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send download request to Model Streamer sidecar")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "Model Streamer sidecar returned error {}: {}",
                status,
                error_text
            );
        }

        let download_response: DownloadResponse = response
            .json()
            .await
            .context("Failed to parse download response from sidecar")?;

        if !download_response.success {
            let error_msg = download_response
                .error
                .unwrap_or_else(|| "Unknown error".to_string());
            anyhow::bail!("Failed to download model '{}': {}", model_name, error_msg);
        }

        let local_path = download_response.local_path.ok_or_else(|| {
            anyhow::anyhow!("Sidecar returned success but no local_path for model '{model_name}'")
        })?;

        if let Some(files) = &download_response.files {
            info!(
                "Downloaded {} files for model '{}' to '{}'",
                files.len(),
                model_name,
                local_path
            );
        } else {
            info!("Downloaded model '{}' to '{}'", model_name, local_path);
        }

        Ok(PathBuf::from(local_path))
    }

    async fn delete_model(&self, model_name: &str) -> Result<()> {
        info!("Deleting model from Model Streamer cache: {}", model_name);

        let model_id = Self::encode_model_id(model_name);
        let url = format!(
            "{}/api/v1/models/{}",
            self.config.endpoint,
            urlencoding::encode(&model_id)
        );

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .context("Failed to send delete request to Model Streamer sidecar")?;

        let status = response.status();

        // 404 is OK - model doesn't exist
        if status == reqwest::StatusCode::NOT_FOUND {
            info!(
                "Model '{}' not found in cache (already deleted)",
                model_name
            );
            return Ok(());
        }

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "Model Streamer sidecar returned error {} while deleting '{}': {}",
                status,
                model_name,
                error_text
            );
        }

        let delete_response: DeleteResponse = response
            .json()
            .await
            .context("Failed to parse delete response from sidecar")?;

        if !delete_response.success {
            let error_msg = delete_response
                .error
                .unwrap_or_else(|| "Unknown error".to_string());
            warn!("Failed to delete model '{}': {}", model_name, error_msg);
            anyhow::bail!("Failed to delete model '{}': {}", model_name, error_msg);
        }

        info!("Successfully deleted model '{}' from cache", model_name);
        Ok(())
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        debug!(
            "Getting model path for '{}' in cache '{}'",
            model_name,
            cache_dir.display()
        );

        let model_id = Self::encode_model_id(model_name);
        let url = format!(
            "{}/api/v1/models/{}?cache_dir={}",
            self.config.endpoint,
            urlencoding::encode(&model_id),
            urlencoding::encode(&cache_dir.to_string_lossy())
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send get model request to Model Streamer sidecar")?;

        let status = response.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!("Model '{}' not found in cache", model_name);
        }

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "Model Streamer sidecar returned error {} for '{}': {}",
                status,
                model_name,
                error_text
            );
        }

        let get_response: GetModelResponse = response
            .json()
            .await
            .context("Failed to parse get model response from sidecar")?;

        if !get_response.exists {
            let error_msg = get_response
                .error
                .unwrap_or_else(|| "Model not found".to_string());
            anyhow::bail!("Model '{}' not found: {}", model_name, error_msg);
        }

        let local_path = get_response.local_path.ok_or_else(|| {
            anyhow::anyhow!(
                "Sidecar returned exists=true but no local_path for model '{}'",
                model_name
            )
        })?;

        Ok(PathBuf::from(local_path))
    }

    fn provider_name(&self) -> &'static str {
        "Model Streamer"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let config = SidecarConfig::default();
        let provider =
            ModelStreamerProvider::new(config, S3Credentials::default(), GcsCredentials::default())
                .expect("Failed to create provider");
        assert_eq!(provider.provider_name(), "Model Streamer");
    }

    #[test]
    fn test_encode_model_id() {
        assert_eq!(
            ModelStreamerProvider::encode_model_id("s3://bucket/path/to/model"),
            "s3/bucket/path/to/model"
        );
        assert_eq!(
            ModelStreamerProvider::encode_model_id("gs://bucket/path/to/model"),
            "gs/bucket/path/to/model"
        );
        assert_eq!(
            ModelStreamerProvider::encode_model_id("s3+http://minio:9000/bucket/model"),
            "s3/bucket/model"
        );
        assert_eq!(
            ModelStreamerProvider::encode_model_id("s3+https://minio:9000/bucket/path/to/model"),
            "s3/bucket/path/to/model"
        );
    }

    #[test]
    fn test_build_s3_credentials_none() {
        let config = SidecarConfig::default();
        let provider =
            ModelStreamerProvider::new(config, S3Credentials::default(), GcsCredentials::default())
                .expect("Failed to create provider");
        assert!(provider.build_s3_credentials().is_none());
    }

    #[test]
    fn test_build_s3_credentials_some() {
        let config = SidecarConfig::default();
        let s3_creds = S3Credentials {
            access_key_id: Some("test_key".to_string()),
            secret_access_key: Some("test_secret".to_string()),
            region: Some("us-east-1".to_string()),
            endpoint: None,
        };
        let provider = ModelStreamerProvider::new(config, s3_creds, GcsCredentials::default())
            .expect("Failed to create provider");

        let creds = provider.build_s3_credentials();
        assert!(creds.is_some());
        let creds = creds.expect("Expected credentials");
        assert_eq!(creds.access_key_id, Some("test_key".to_string()));
        assert_eq!(creds.region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_build_gcs_credentials_none() {
        let config = SidecarConfig::default();
        let provider =
            ModelStreamerProvider::new(config, S3Credentials::default(), GcsCredentials::default())
                .expect("Failed to create provider");
        assert!(provider.build_gcs_credentials().is_none());
    }

    #[test]
    fn test_build_gcs_credentials_some() {
        let config = SidecarConfig::default();
        let gcs_creds = GcsCredentials {
            credentials_file: Some("/path/to/creds.json".to_string()),
            credentials_json: None,
        };
        let provider = ModelStreamerProvider::new(config, S3Credentials::default(), gcs_creds)
            .expect("Failed to create provider");

        let creds = provider.build_gcs_credentials();
        assert!(creds.is_some());
        let creds = creds.expect("Expected credentials");
        assert_eq!(
            creds.credentials_file,
            Some("/path/to/creds.json".to_string())
        );
    }

    #[test]
    fn test_is_weight_file() {
        assert!(ModelStreamerProvider::is_weight_file("model.safetensors"));
        assert!(ModelStreamerProvider::is_weight_file("weights.bin"));
        assert!(!ModelStreamerProvider::is_weight_file("config.json"));
    }

    #[test]
    fn test_is_ignored() {
        assert!(ModelStreamerProvider::is_ignored("README.md"));
        assert!(ModelStreamerProvider::is_ignored(".gitignore"));
        assert!(!ModelStreamerProvider::is_ignored("model.safetensors"));
    }
}
