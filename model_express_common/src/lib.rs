use serde::{Deserialize, Serialize};

pub mod cache;
pub mod download;
pub mod models;

// Generated gRPC code
#[allow(clippy::similar_names)]
#[allow(clippy::default_trait_access)]
#[allow(clippy::doc_markdown)]
#[allow(clippy::must_use_candidate)]
pub mod grpc {
    pub mod health {
        tonic::include_proto!("model_express.health");
    }
    pub mod api {
        tonic::include_proto!("model_express.api");
    }
    pub mod model {
        tonic::include_proto!("model_express.model");
    }
}

/// Defines the shared response format between server and client (legacy HTTP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

/// Common error types that both client and server can use
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Server returned error: {0}")]
    Server(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
}

// Implement From traits for Box<Error> to work with the Result<T> type
impl From<tonic::Status> for Box<Error> {
    fn from(err: tonic::Status) -> Self {
        Box::new(Error::Grpc(err))
    }
}

impl From<tonic::transport::Error> for Box<Error> {
    fn from(err: tonic::transport::Error) -> Self {
        Box::new(Error::Transport(err))
    }
}

/// Common result type for the project
pub type Result<T> = std::result::Result<T, Box<Error>>;

/// Constants shared between client and server
pub mod constants {
    pub const DEFAULT_PORT: u16 = 8000;
    pub const DEFAULT_GRPC_PORT: u16 = 8001;
    pub const API_VERSION: &str = "v1";
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
}

// Conversion utilities between gRPC and legacy models
impl From<&models::Status> for grpc::health::HealthResponse {
    fn from(status: &models::Status) -> Self {
        Self {
            version: status.version.clone(),
            status: status.status.clone(),
            uptime: status.uptime,
        }
    }
}

impl From<grpc::health::HealthResponse> for models::Status {
    fn from(response: grpc::health::HealthResponse) -> Self {
        Self {
            version: response.version,
            status: response.status,
            uptime: response.uptime,
        }
    }
}

impl From<models::ModelProvider> for grpc::model::ModelProvider {
    fn from(provider: models::ModelProvider) -> Self {
        match provider {
            models::ModelProvider::HuggingFace => grpc::model::ModelProvider::HuggingFace,
        }
    }
}

impl From<grpc::model::ModelProvider> for models::ModelProvider {
    fn from(provider: grpc::model::ModelProvider) -> Self {
        match provider {
            grpc::model::ModelProvider::HuggingFace => models::ModelProvider::HuggingFace,
        }
    }
}

impl From<models::ModelStatus> for grpc::model::ModelStatus {
    fn from(status: models::ModelStatus) -> Self {
        match status {
            models::ModelStatus::DOWNLOADING => grpc::model::ModelStatus::Downloading,
            models::ModelStatus::DOWNLOADED => grpc::model::ModelStatus::Downloaded,
            models::ModelStatus::ERROR => grpc::model::ModelStatus::Error,
        }
    }
}

impl From<grpc::model::ModelStatus> for models::ModelStatus {
    fn from(status: grpc::model::ModelStatus) -> Self {
        match status {
            grpc::model::ModelStatus::Downloading => models::ModelStatus::DOWNLOADING,
            grpc::model::ModelStatus::Downloaded => models::ModelStatus::DOWNLOADED,
            grpc::model::ModelStatus::Error => models::ModelStatus::ERROR,
        }
    }
}

impl From<&models::ModelStatusResponse> for grpc::model::ModelStatusUpdate {
    fn from(response: &models::ModelStatusResponse) -> Self {
        Self {
            model_name: response.model_name.clone(),
            status: grpc::model::ModelStatus::from(response.status) as i32,
            message: None,
            provider: grpc::model::ModelProvider::from(response.provider) as i32,
        }
    }
}

impl From<grpc::model::ModelStatusUpdate> for models::ModelStatusResponse {
    fn from(update: grpc::model::ModelStatusUpdate) -> Self {
        Self {
            model_name: update.model_name,
            status: grpc::model::ModelStatus::try_from(update.status)
                .unwrap_or(grpc::model::ModelStatus::Error)
                .into(),
            provider: grpc::model::ModelProvider::try_from(update.provider)
                .unwrap_or(grpc::model::ModelProvider::HuggingFace)
                .into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_conversion_from_models_to_grpc() {
        let status = models::Status {
            version: "1.0.0".to_string(),
            status: "ok".to_string(),
            uptime: 3600,
        };

        let grpc_response: grpc::health::HealthResponse = (&status).into();

        assert_eq!(grpc_response.version, status.version);
        assert_eq!(grpc_response.status, status.status);
        assert_eq!(grpc_response.uptime, status.uptime);
    }

    #[test]
    fn test_status_conversion_from_grpc_to_models() {
        let grpc_response = grpc::health::HealthResponse {
            version: "1.0.0".to_string(),
            status: "ok".to_string(),
            uptime: 3600,
        };

        let status: models::Status = grpc_response.into();

        assert_eq!(status.version, "1.0.0");
        assert_eq!(status.status, "ok");
        assert_eq!(status.uptime, 3600);
    }

    #[test]
    fn test_model_provider_conversion_both_ways() {
        let model_provider = models::ModelProvider::HuggingFace;
        let grpc_provider: grpc::model::ModelProvider = model_provider.into();
        let back_to_model: models::ModelProvider = grpc_provider.into();

        assert_eq!(model_provider, back_to_model);
    }

    #[test]
    fn test_model_status_conversion_both_ways() {
        let statuses = vec![
            models::ModelStatus::DOWNLOADING,
            models::ModelStatus::DOWNLOADED,
            models::ModelStatus::ERROR,
        ];

        for status in statuses {
            let grpc_status: grpc::model::ModelStatus = status.into();
            let back_to_model: models::ModelStatus = grpc_status.into();
            assert_eq!(status, back_to_model);
        }
    }

    #[test]
    fn test_model_status_response_conversion_from_models_to_grpc() {
        let response = models::ModelStatusResponse {
            model_name: "test-model".to_string(),
            status: models::ModelStatus::DOWNLOADED,
            provider: models::ModelProvider::HuggingFace,
        };

        let grpc_update: grpc::model::ModelStatusUpdate = (&response).into();

        assert_eq!(grpc_update.model_name, response.model_name);
        assert_eq!(
            grpc_update.status,
            grpc::model::ModelStatus::Downloaded as i32
        );
        assert_eq!(
            grpc_update.provider,
            grpc::model::ModelProvider::HuggingFace as i32
        );
        assert!(grpc_update.message.is_none());
    }

    #[test]
    fn test_model_status_response_conversion_from_grpc_to_models() {
        let grpc_update = grpc::model::ModelStatusUpdate {
            model_name: "test-model".to_string(),
            status: grpc::model::ModelStatus::Downloaded as i32,
            message: Some("Test message".to_string()),
            provider: grpc::model::ModelProvider::HuggingFace as i32,
        };

        let response: models::ModelStatusResponse = grpc_update.into();

        assert_eq!(response.model_name, "test-model");
        assert_eq!(response.status, models::ModelStatus::DOWNLOADED);
        assert_eq!(response.provider, models::ModelProvider::HuggingFace);
    }

    #[test]
    fn test_error_types() {
        let network_error = Error::Network("Connection failed".to_string());
        assert!(network_error.to_string().contains("Network error"));

        let server_error = Error::Server("Internal error".to_string());
        assert!(server_error.to_string().contains("Server returned error"));

        let serialization_error = Error::Serialization("JSON parse error".to_string());
        assert!(
            serialization_error
                .to_string()
                .contains("Serialization error")
        );
    }

    #[test]
    fn test_constants() {
        assert_eq!(constants::DEFAULT_PORT, 8000);
        assert_eq!(constants::DEFAULT_GRPC_PORT, 8001);
        assert_eq!(constants::API_VERSION, "v1");
        assert_eq!(constants::DEFAULT_TIMEOUT_SECS, 30);
    }

    #[test]
    fn test_response_creation() {
        let success_response = Response {
            success: true,
            data: Some("test data".to_string()),
            error: None,
        };

        assert!(success_response.success);
        assert!(success_response.data.is_some());
        assert!(success_response.error.is_none());

        let error_response: Response<String> = Response {
            success: false,
            data: None,
            error: Some("test error".to_string()),
        };

        assert!(!error_response.success);
        assert!(error_response.data.is_none());
        assert!(error_response.error.is_some());
    }
}
