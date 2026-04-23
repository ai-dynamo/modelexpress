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
    ///
    /// Errors if `MX_METADATA_BACKEND` is unset/invalid, or if the connection env for
    /// the selected backend is missing (no silent fallback to `localhost:6379` /
    /// `default` namespace, since those mask misconfig in production).
    pub fn from_env() -> Result<Self, String> {
        let backend_type = std::env::var("MX_METADATA_BACKEND").unwrap_or_default();
        match backend_type.to_lowercase().as_str() {
            "redis" => Ok(Self::Redis {
                url: Self::redis_url_from_env()?,
            }),
            "kubernetes" | "k8s" | "crd" => Ok(Self::Kubernetes {
                namespace: Self::k8s_namespace_from_env()?,
            }),
            other => Err(format!(
                "MX_METADATA_BACKEND='{other}' is not valid. Use 'redis' or 'kubernetes'."
            )),
        }
    }

    /// Parse a backend type string into a config. Testable without env vars — callers
    /// supply the connection strings directly.
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

    /// Return the Redis connection URL from env. Accepts either `REDIS_URL` directly, or
    /// `MX_REDIS_HOST` + `MX_REDIS_PORT` (with `REDIS_HOST` / `REDIS_PORT` fallbacks for
    /// compatibility with charts that predate the `MX_` prefix). Errors when neither the
    /// URL nor both host-and-port pieces are provided.
    pub fn redis_url_from_env() -> Result<String, String> {
        if let Ok(url) = std::env::var("REDIS_URL") {
            return Ok(url);
        }
        let host = std::env::var("MX_REDIS_HOST")
            .or_else(|_| std::env::var("REDIS_HOST"))
            .map_err(|_| {
                "MX_METADATA_BACKEND=redis requires REDIS_URL or MX_REDIS_HOST (alias \
                 REDIS_HOST) to be set."
                    .to_string()
            })?;
        let port = std::env::var("MX_REDIS_PORT")
            .or_else(|_| std::env::var("REDIS_PORT"))
            .map_err(|_| {
                "MX_METADATA_BACKEND=redis requires REDIS_URL or MX_REDIS_PORT (alias \
                 REDIS_PORT) to be set."
                    .to_string()
            })?;
        Ok(format!("redis://{host}:{port}"))
    }

    /// Return the Kubernetes namespace from env. The downward API exposes
    /// `POD_NAMESPACE` to in-cluster pods; `MX_METADATA_NAMESPACE` overrides it for
    /// out-of-cluster operators. Errors when neither is set so a typo in the chart
    /// can't silently land ModelCacheEntry CRs in the `default` namespace.
    fn k8s_namespace_from_env() -> Result<String, String> {
        std::env::var("MX_METADATA_NAMESPACE")
            .or_else(|_| std::env::var("POD_NAMESPACE"))
            .map_err(|_| {
                "MX_METADATA_BACKEND=kubernetes requires MX_METADATA_NAMESPACE or \
                 POD_NAMESPACE to be set."
                    .to_string()
            })
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

    #[test]
    fn rejects_empty_backend_type() {
        let err = BackendConfig::from_type_str("MX_METADATA_BACKEND", "", "", "")
            .expect_err("empty should reject");
        assert!(err.contains("''"), "error should echo empty value: {err}");
    }

    #[test]
    fn display_renders_backend_name() {
        let redis = BackendConfig::Redis {
            url: "redis://host:6379".to_string(),
        };
        assert_eq!(redis.to_string(), "redis");
        let k8s = BackendConfig::Kubernetes {
            namespace: "prod".to_string(),
        };
        assert_eq!(k8s.to_string(), "kubernetes");
    }

    /// The `from_env` / `redis_url_from_env` / `k8s_namespace_from_env` helpers read
    /// process-global env vars. Tests that mutate env have to acquire a mutex to stay
    /// serialized, since `cargo test` runs with multiple threads by default.
    use modelexpress_common::test_support::{EnvVarGuard, acquire_env_mutex};

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_reads_mx_metadata_backend() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis");
        let _g2 = EnvVarGuard::set(&lock, "REDIS_URL", "redis://myhost:7777");
        let cfg = BackendConfig::from_env().expect("from_env redis");
        assert!(matches!(cfg, BackendConfig::Redis { url } if url == "redis://myhost:7777"));
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_accepts_kubernetes_aliases() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "k8s");
        let _g2 = EnvVarGuard::set(&lock, "POD_NAMESPACE", "test-ns");
        let cfg = BackendConfig::from_env().expect("from_env k8s alias");
        assert!(matches!(cfg, BackendConfig::Kubernetes { namespace } if namespace == "test-ns"));
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_errors_when_backend_unset() {
        let lock = acquire_env_mutex();
        let _g = EnvVarGuard::remove(&lock, "MX_METADATA_BACKEND");
        let err = BackendConfig::from_env().expect_err("should reject missing backend");
        assert!(err.contains("MX_METADATA_BACKEND"));
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn redis_url_from_env_honors_explicit_url_over_host_port() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::set(&lock, "REDIS_URL", "redis://explicit:1234");
        let _g2 = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "other");
        let _g3 = EnvVarGuard::set(&lock, "MX_REDIS_PORT", "9999");
        assert_eq!(
            BackendConfig::redis_url_from_env().expect("REDIS_URL wins"),
            "redis://explicit:1234"
        );
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn redis_url_from_env_builds_from_host_port_when_url_missing() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::remove(&lock, "REDIS_URL");
        let _g2 = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "myhost");
        let _g3 = EnvVarGuard::set(&lock, "MX_REDIS_PORT", "6380");
        assert_eq!(
            BackendConfig::redis_url_from_env().expect("host+port build"),
            "redis://myhost:6380"
        );
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn redis_url_from_env_errors_when_host_and_port_missing() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::remove(&lock, "REDIS_URL");
        let _g2 = EnvVarGuard::remove(&lock, "MX_REDIS_HOST");
        let _g3 = EnvVarGuard::remove(&lock, "REDIS_HOST");
        let _g4 = EnvVarGuard::remove(&lock, "MX_REDIS_PORT");
        let _g5 = EnvVarGuard::remove(&lock, "REDIS_PORT");
        let err =
            BackendConfig::redis_url_from_env().expect_err("should error on missing Redis env");
        assert!(
            err.contains("REDIS_URL") && err.contains("MX_REDIS_HOST"),
            "error should name the required env vars: {err}"
        );
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn redis_url_from_env_errors_when_port_missing() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::remove(&lock, "REDIS_URL");
        let _g2 = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "myhost");
        let _g3 = EnvVarGuard::remove(&lock, "MX_REDIS_PORT");
        let _g4 = EnvVarGuard::remove(&lock, "REDIS_PORT");
        let err = BackendConfig::redis_url_from_env()
            .expect_err("should error when port is missing even with host set");
        assert!(
            err.contains("MX_REDIS_PORT"),
            "error should name the missing port env var: {err}"
        );
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_kubernetes_errors_when_namespace_unset() {
        let lock = acquire_env_mutex();
        let _g1 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "kubernetes");
        let _g2 = EnvVarGuard::remove(&lock, "MX_METADATA_NAMESPACE");
        let _g3 = EnvVarGuard::remove(&lock, "POD_NAMESPACE");
        let err = BackendConfig::from_env()
            .expect_err("kubernetes backend without namespace should reject");
        assert!(
            err.contains("MX_METADATA_NAMESPACE") || err.contains("POD_NAMESPACE"),
            "error should name the namespace env vars: {err}"
        );
    }
}
