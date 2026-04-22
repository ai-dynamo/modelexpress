// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared backend selection for distributed stores (Redis and Kubernetes CRDs).
//!
//! A single `BackendConfig` type and a single env var (`MX_METADATA_BACKEND`) drive both
//! the P2P metadata backend and the model registry backend. Deployments that need one
//! always need the other, so decoupling them would just be surface area without a use
//! case — the server crashes at startup if either can't connect.
//!
//! The trait implementations themselves live in `p2p::backend` and `registry::backend`.

/// Configuration for a distributed backend (Redis or Kubernetes CRDs).
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// Redis backend — persistent, horizontally scalable
    Redis { url: String },
    /// Kubernetes CRD backend — native K8s integration
    Kubernetes { namespace: String },
}

impl std::fmt::Display for BackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Redis { .. } => write!(f, "redis"),
            Self::Kubernetes { .. } => write!(f, "kubernetes"),
        }
    }
}

impl BackendConfig {
    /// Create backend config from `MX_METADATA_BACKEND`. Used by both the P2P state
    /// manager and the registry manager so they share a single env-var contract.
    ///
    /// Valid values: `redis`, `kubernetes` | `k8s` | `crd`.
    pub fn from_env() -> Result<Self, String> {
        let backend_type = std::env::var("MX_METADATA_BACKEND").unwrap_or_default();
        let redis_url = Self::redis_url_from_env();
        let k8s_namespace = Self::k8s_namespace_from_env();
        Self::from_type_str(
            "MX_METADATA_BACKEND",
            &backend_type,
            &redis_url,
            &k8s_namespace,
        )
    }

    /// Parse a backend type string into a config. Testable without env vars.
    ///
    /// `env_name` appears in the error message so the caller knows which variable was bad.
    pub fn from_type_str(
        env_name: &str,
        backend_type: &str,
        redis_url: &str,
        k8s_namespace: &str,
    ) -> Result<Self, String> {
        match backend_type.to_lowercase().as_str() {
            "redis" => Ok(Self::Redis {
                url: redis_url.to_string(),
            }),
            "kubernetes" | "k8s" | "crd" => Ok(Self::Kubernetes {
                namespace: k8s_namespace.to_string(),
            }),
            other => Err(format!(
                "{env_name}='{other}' is not valid. Use 'redis' or 'kubernetes'."
            )),
        }
    }

    pub fn redis_url_from_env() -> String {
        if let Ok(url) = std::env::var("REDIS_URL") {
            return url;
        }
        let host = std::env::var("MX_REDIS_HOST")
            .or_else(|_| std::env::var("REDIS_HOST"))
            .unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("MX_REDIS_PORT")
            .or_else(|_| std::env::var("REDIS_PORT"))
            .unwrap_or_else(|_| "6379".to_string());
        format!("redis://{host}:{port}")
    }

    fn k8s_namespace_from_env() -> String {
        std::env::var("MX_METADATA_NAMESPACE")
            .or_else(|_| std::env::var("POD_NAMESPACE"))
            .unwrap_or_else(|_| "default".to_string())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_redis_and_kubernetes_aliases() {
        let cfg = BackendConfig::from_type_str("X", "redis", "redis://h:1", "ns").expect("redis");
        assert!(matches!(cfg, BackendConfig::Redis { url } if url == "redis://h:1"));

        for alias in ["kubernetes", "k8s", "crd", "K8S", "Kubernetes"] {
            let cfg =
                BackendConfig::from_type_str("X", alias, "redis://h:1", "prod").expect("k8s alias");
            assert!(matches!(cfg, BackendConfig::Kubernetes { namespace } if namespace == "prod"));
        }
    }

    #[test]
    fn rejects_unknown_and_includes_env_name() {
        let err = BackendConfig::from_type_str("MX_WHATEVER", "memory", "", "")
            .expect_err("should reject");
        assert!(
            err.contains("MX_WHATEVER"),
            "error should name the env var: {err}"
        );
        assert!(
            err.contains("'memory'"),
            "error should echo bad value: {err}"
        );
    }
}
