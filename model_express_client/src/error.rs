// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// Client-specific error types
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("Failed to connect to server: {0}")]
    ConnectionFailed(String),

    #[error("Request timed out: {0}")]
    Timeout(String),

    #[error("Common error: {0}")]
    Common(#[from] model_express_common::Error),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_error_connection_failed() {
        let error = ClientError::ConnectionFailed("Connection refused".to_string());
        assert!(error.to_string().contains("Failed to connect to server"));
        assert!(error.to_string().contains("Connection refused"));
    }

    #[test]
    fn test_client_error_timeout() {
        let error = ClientError::Timeout("Request took too long".to_string());
        assert!(error.to_string().contains("Request timed out"));
        assert!(error.to_string().contains("Request took too long"));
    }

    #[test]
    fn test_client_error_config() {
        let error = ClientError::Config("Invalid endpoint format".to_string());
        assert!(error.to_string().contains("Invalid configuration"));
        assert!(error.to_string().contains("Invalid endpoint format"));
    }

    #[test]
    fn test_client_error_from_common_error() {
        let common_error = model_express_common::Error::Network("Network issue".to_string());
        let client_error: ClientError = common_error.into();

        match client_error {
            ClientError::Common(_) => {
                assert!(client_error.to_string().contains("Common error"));
            }
            _ => panic!("Expected Common error variant"),
        }
    }

    #[test]
    fn test_all_error_variants() {
        let errors = vec![
            ClientError::ConnectionFailed("test".to_string()),
            ClientError::Timeout("test".to_string()),
            ClientError::Config("test".to_string()),
            ClientError::Common(model_express_common::Error::Network("test".to_string())),
        ];

        for error in errors {
            // Each error should have a meaningful string representation
            let error_str = error.to_string();
            assert!(!error_str.is_empty());
            assert!(error_str.len() > 5); // Should be more than just "test"
        }
    }
}
