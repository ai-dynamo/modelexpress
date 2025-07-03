use model_express_common::constants;

/// Configuration for the client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// The gRPC endpoint of the server, e.g. "<http://localhost:8001>"
    pub grpc_endpoint: String,

    /// Timeout in seconds for requests (defaults to 30 if None)
    pub timeout_secs: Option<u64>,
}

impl ClientConfig {
    /// Create a new client configuration
    pub fn new(grpc_endpoint: impl Into<String>) -> Self {
        Self {
            grpc_endpoint: grpc_endpoint.into(),
            timeout_secs: None,
        }
    }

    /// Set a custom timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            grpc_endpoint: format!("http://localhost:{}", constants::DEFAULT_GRPC_PORT),
            timeout_secs: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_new() {
        let config = ClientConfig::new("http://example.com:8080");
        assert_eq!(config.grpc_endpoint, "http://example.com:8080");
        assert!(config.timeout_secs.is_none());
    }

    #[test]
    fn test_client_config_with_timeout() {
        let config = ClientConfig::new("http://example.com:8080").with_timeout(60);
        assert_eq!(config.grpc_endpoint, "http://example.com:8080");
        assert_eq!(config.timeout_secs, Some(60));
    }

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(
            config.grpc_endpoint,
            format!("http://localhost:{}", constants::DEFAULT_GRPC_PORT)
        );
        assert!(config.timeout_secs.is_none());
    }

    #[test]
    fn test_client_config_clone() {
        let config1 = ClientConfig::new("http://test.com").with_timeout(30);
        let config2 = config1.clone();

        assert_eq!(config1.grpc_endpoint, config2.grpc_endpoint);
        assert_eq!(config1.timeout_secs, config2.timeout_secs);
    }

    #[test]
    fn test_client_config_builder_pattern() {
        let config = ClientConfig::new("http://localhost:8001").with_timeout(45);

        assert_eq!(config.grpc_endpoint, "http://localhost:8001");
        assert_eq!(config.timeout_secs, Some(45));
    }
}
