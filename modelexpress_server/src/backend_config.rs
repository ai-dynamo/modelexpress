// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared backend selection for the metadata stores (Redis, Redis Cluster, Kubernetes CRDs,
//! and an in-memory backend behind the `memory-backend` feature for tests and local dev).
//!
//! A single `BackendConfig` type and env vars (`MX_METADATA_BACKEND` + optional `MX_REDIS_MODE`)
//! drive all metadata stores (model registry, P2P metadata, refit, and collective refit).
//!
//! The trait implementations themselves live in `p2p::backend`, `registry::backend`,
//! `refit::backend`, and `refit_collective::backend`.

use modelexpress_common::envs;

/// Redis deployment topology selected via [`envs::MX_REDIS_MODE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedisMode {
    /// Single instance or primary/replica managed via a single connection URL.
    #[default]
    Standalone,
    /// Multi-primary Redis Cluster discovered from one or more seed URLs.
    Cluster,
}

impl RedisMode {
    /// Parse an optional `MX_REDIS_MODE` string. Unset or whitespace-only defaults to [`Self::Standalone`].
    pub fn parse_opt(raw: Option<&str>) -> Result<Self, String> {
        let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(Self::Standalone);
        };
        match trimmed.to_ascii_lowercase().as_str() {
            "standalone" => Ok(Self::Standalone),
            "cluster" => Ok(Self::Cluster),
            other => Err(format!(
                "Unsupported {} value '{other}'. Supported values: standalone, cluster.",
                envs::MX_REDIS_MODE
            )),
        }
    }
}

/// Configuration for a distributed backend (Redis, Redis Cluster, or Kubernetes CRDs).
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// Standalone Redis backend (single instance or primary/replica).
    Redis { url: String },
    /// Redis Cluster backend (sharded multi-primary cluster with one or more seed URLs).
    RedisCluster { urls: Vec<String> },
    /// Kubernetes CRD backend — native K8s integration.
    Kubernetes { namespace: String },
    /// In-memory backend for tests and local dev (off in production).
    #[cfg(feature = "memory-backend")]
    Memory,
}

impl std::fmt::Display for BackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Redis { .. } => write!(f, "redis"),
            Self::RedisCluster { .. } => write!(f, "redis-cluster"),
            Self::Kubernetes { .. } => write!(f, "kubernetes"),
            #[cfg(feature = "memory-backend")]
            Self::Memory => write!(f, "memory"),
        }
    }
}

impl BackendConfig {
    /// Create backend config from `MX_METADATA_BACKEND` (and `MX_REDIS_MODE` when using Redis).
    /// Used by the registry, P2P, refit, and collective-refit managers so they share one contract.
    ///
    /// Valid values for `MX_METADATA_BACKEND`: `redis`, `redis-cluster`,
    /// `kubernetes` | `k8s` | `crd`, `memory`.
    /// When `MX_METADATA_BACKEND=redis`, `MX_REDIS_MODE` selects `standalone` (default) or `cluster`.
    ///
    /// Errors if `MX_METADATA_BACKEND` is unset/invalid, if `MX_REDIS_MODE` is invalid or conflicts
    /// with `redis-cluster`, or if the connection env for the selected backend is missing.
    pub fn from_env() -> Result<Self, String> {
        let backend_type = envs::metadata_backend();
        let redis_mode = envs::redis_mode();
        match backend_type.to_ascii_lowercase().as_str() {
            "redis" => match RedisMode::parse_opt(redis_mode.as_deref())? {
                RedisMode::Standalone => Ok(Self::Redis {
                    url: Self::redis_url_from_env()?,
                }),
                RedisMode::Cluster => Ok(Self::RedisCluster {
                    urls: Self::redis_cluster_urls_from_env()?,
                }),
            },
            "redis-cluster" => {
                if let Some(raw_mode) = redis_mode.as_deref()
                    && RedisMode::parse_opt(Some(raw_mode))? == RedisMode::Standalone
                {
                    return Err(format!(
                        "{}='redis-cluster' conflicts with {}='{}'. Remove {} or set it to 'cluster'.",
                        envs::MX_METADATA_BACKEND,
                        envs::MX_REDIS_MODE,
                        raw_mode.trim(),
                        envs::MX_REDIS_MODE,
                    ));
                }
                Ok(Self::RedisCluster {
                    urls: Self::redis_cluster_urls_from_env()?,
                })
            }
            "kubernetes" | "k8s" | "crd" => Ok(Self::Kubernetes {
                namespace: Self::k8s_namespace_from_env()?,
            }),
            #[cfg(feature = "memory-backend")]
            "memory" => Ok(Self::Memory),
            other => Err(format!(
                "MX_METADATA_BACKEND='{other}' is not valid. {}",
                Self::valid_backends_hint()
            )),
        }
    }

    fn valid_backends_hint() -> &'static str {
        if cfg!(feature = "memory-backend") {
            "Use 'redis', 'redis-cluster', 'kubernetes', or 'memory'."
        } else {
            "Use 'redis', 'redis-cluster', or 'kubernetes'."
        }
    }

    /// Parse a comma-separated list of Redis URLs into trimmed non-empty strings,
    /// preserving commas inside URL credentials and query values.
    pub fn parse_redis_urls(raw: &str) -> Result<Vec<String>, String> {
        let urls = crate::redis_conn::split_redis_urls(raw);
        if urls.is_empty() {
            return Err("Redis URL list cannot be empty.".to_string());
        }
        Ok(urls)
    }

    /// Reject comma-separated multi-node URL strings in standalone mode so cluster
    /// configurations aren't silently broken or misparsed.
    fn validate_standalone_url(raw: &str) -> Result<String, String> {
        let urls = Self::parse_redis_urls(raw)?;
        if urls.len() > 1 {
            return Err(format!(
                "Multiple comma-separated URLs found in {}/{} while {} is 'standalone'. Set {}=cluster (or {}=redis-cluster) to connect to a Redis Cluster.",
                envs::MX_REDIS_URL,
                envs::REDIS_URL,
                envs::MX_REDIS_MODE,
                envs::MX_REDIS_MODE,
                envs::MX_METADATA_BACKEND,
            ));
        }
        urls.into_iter()
            .next()
            .ok_or_else(|| "Redis URL cannot be empty.".to_string())
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
        Self::from_type_and_mode_str(env_name, backend_type, None, redis_url, k8s_namespace)
    }

    /// Parse backend type + optional Redis mode strings into a config without touching process env.
    pub fn from_type_and_mode_str(
        env_name: &str,
        backend_type: &str,
        redis_mode: Option<&str>,
        redis_url: &str,
        k8s_namespace: &str,
    ) -> Result<Self, String> {
        match backend_type.to_ascii_lowercase().as_str() {
            "redis" => match RedisMode::parse_opt(redis_mode)? {
                RedisMode::Standalone => Ok(Self::Redis {
                    url: Self::validate_standalone_url(redis_url)?,
                }),
                RedisMode::Cluster => Ok(Self::RedisCluster {
                    urls: Self::parse_redis_urls(redis_url)?,
                }),
            },
            "redis-cluster" => {
                if let Some(raw_mode) = redis_mode
                    && !raw_mode.trim().is_empty()
                    && RedisMode::parse_opt(Some(raw_mode))? == RedisMode::Standalone
                {
                    return Err(format!(
                        "{env_name}='redis-cluster' conflicts with {}='{}'. Remove {} or set it to 'cluster'.",
                        envs::MX_REDIS_MODE,
                        raw_mode.trim(),
                        envs::MX_REDIS_MODE,
                    ));
                }
                Ok(Self::RedisCluster {
                    urls: Self::parse_redis_urls(redis_url)?,
                })
            }
            "kubernetes" | "k8s" | "crd" => Ok(Self::Kubernetes {
                namespace: k8s_namespace.to_string(),
            }),
            #[cfg(feature = "memory-backend")]
            "memory" => Ok(Self::Memory),
            other => Err(format!(
                "{env_name}='{other}' is not valid. {}",
                Self::valid_backends_hint()
            )),
        }
    }

    /// Return the standalone Redis connection URL from env. Accepts either `REDIS_URL` directly,
    /// or `MX_REDIS_HOST` + `MX_REDIS_PORT` (with `REDIS_HOST` / `REDIS_PORT` fallbacks).
    /// Errors if multiple comma-separated URLs are passed while in standalone mode, or if neither
    /// the URL nor both host-and-port pieces are provided.
    pub fn redis_url_from_env() -> Result<String, String> {
        if let Some(url) = envs::redis_url() {
            return Self::validate_standalone_url(&url);
        }
        let host = envs::redis_host().ok_or_else(|| {
            format!(
                "MX_METADATA_BACKEND=redis requires {} or {} (alias {}) to be set.",
                envs::REDIS_URL,
                envs::MX_REDIS_HOST,
                envs::REDIS_HOST,
            )
        })?;
        let port = envs::redis_port().ok_or_else(|| {
            format!(
                "MX_METADATA_BACKEND=redis requires {} or {} (alias {}) to be set.",
                envs::REDIS_URL,
                envs::MX_REDIS_PORT,
                envs::REDIS_PORT,
            )
        })?;
        Ok(format!("redis://{host}:{port}"))
    }

    /// Return Redis Cluster seed URLs from env.
    ///
    /// Accepts comma-separated URLs in `MX_REDIS_URL` / `REDIS_URL`, or falls back to
    /// `MX_REDIS_HOST` / `REDIS_HOST` (either single host or comma-separated `host` / `host:port`
    /// entries) combined with `MX_REDIS_PORT` / `REDIS_PORT` when any host lacks an explicit port.
    pub fn redis_cluster_urls_from_env() -> Result<Vec<String>, String> {
        if let Some(raw_urls) = envs::redis_url() {
            return Self::parse_redis_urls(&raw_urls);
        }
        let raw_hosts = envs::redis_host().ok_or_else(|| {
            format!(
                "Redis Cluster backend requires {} (or {}) with comma-separated seed URLs, or {} (alias {}) + {} (alias {}).",
                envs::MX_REDIS_URL,
                envs::REDIS_URL,
                envs::MX_REDIS_HOST,
                envs::REDIS_HOST,
                envs::MX_REDIS_PORT,
                envs::REDIS_PORT,
            )
        })?;
        let port = envs::redis_port();
        let mut urls = Vec::new();
        for entry in raw_hosts
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if entry.starts_with("redis://") || entry.starts_with("rediss://") {
                urls.push(entry.to_string());
            } else if has_explicit_port(entry) {
                urls.push(format!("redis://{entry}"));
            } else {
                let p = port.as_deref().ok_or_else(|| {
                    format!(
                        "Redis Cluster host '{entry}' has no port and neither {} nor {} is set.",
                        envs::MX_REDIS_PORT,
                        envs::REDIS_PORT,
                    )
                })?;
                urls.push(format!("redis://{entry}:{p}"));
            }
        }
        if urls.is_empty() {
            return Err(format!(
                "Redis Cluster host list in {}/{} cannot be empty.",
                envs::MX_REDIS_HOST,
                envs::REDIS_HOST,
            ));
        }
        Ok(urls)
    }

    /// Return the Kubernetes namespace from env. The downward API exposes
    /// `POD_NAMESPACE` to in-cluster pods; `MX_METADATA_NAMESPACE` overrides it for
    /// out-of-cluster operators. Errors when neither is set so a typo in the chart
    /// can't silently land ModelCacheEntry CRs in the `default` namespace.
    fn k8s_namespace_from_env() -> Result<String, String> {
        envs::metadata_namespace().ok_or_else(|| {
            format!(
                "MX_METADATA_BACKEND=kubernetes requires {} or {} to be set.",
                envs::MX_METADATA_NAMESPACE,
                envs::POD_NAMESPACE,
            )
        })
    }
}

/// Returns `true` when `entry` ends with `:port` (handling bracketed IPv6 like `[::1]:6379`).
fn has_explicit_port(entry: &str) -> bool {
    if let Some(rest) = entry.strip_prefix('[') {
        rest.contains("]:")
    } else {
        entry
            .rsplit_once(':')
            .is_some_and(|(_, port)| !port.is_empty() && port.chars().all(|c| c.is_ascii_digit()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_redis_cluster_and_kubernetes_aliases() {
        let cfg = BackendConfig::from_type_str("X", "redis", "redis://h:1", "ns").expect("redis");
        assert!(matches!(cfg, BackendConfig::Redis { url } if url == "redis://h:1"));

        let cfg = BackendConfig::from_type_str(
            "X",
            "redis-cluster",
            "redis://n1:6379, redis://n2:6379 , redis://n3:6379",
            "ns",
        )
        .expect("redis-cluster");
        assert!(matches!(
            cfg,
            BackendConfig::RedisCluster { urls }
            if urls == vec![
                "redis://n1:6379",
                "redis://n2:6379",
                "redis://n3:6379",
            ]
        ));

        let cfg = BackendConfig::from_type_and_mode_str(
            "X",
            "redis",
            Some("cluster"),
            "redis://seed:6379",
            "ns",
        )
        .expect("redis + cluster mode");
        assert!(
            matches!(cfg, BackendConfig::RedisCluster { urls } if urls == vec!["redis://seed:6379"])
        );

        for alias in ["kubernetes", "k8s", "crd", "K8S", "Kubernetes"] {
            let cfg =
                BackendConfig::from_type_str("X", alias, "redis://h:1", "prod").expect("k8s alias");
            assert!(matches!(cfg, BackendConfig::Kubernetes { namespace } if namespace == "prod"));
        }
    }

    #[test]
    fn rejects_invalid_and_conflicting_redis_modes() {
        let err = BackendConfig::from_type_and_mode_str(
            "MX_METADATA_BACKEND",
            "redis",
            Some("sentinel"),
            "redis://h:1",
            "",
        )
        .expect_err("sentinel is unsupported");
        assert!(err.contains("MX_REDIS_MODE"), "{err}");
        assert!(err.contains("'sentinel'"), "{err}");

        let err = BackendConfig::from_type_and_mode_str(
            "MX_METADATA_BACKEND",
            "redis-cluster",
            Some("standalone"),
            "redis://h:1",
            "",
        )
        .expect_err("redis-cluster + standalone should conflict");
        assert!(err.contains("conflicts"), "{err}");

        let err = BackendConfig::from_type_str(
            "MX_METADATA_BACKEND",
            "redis",
            "redis://n1:6379,redis://n2:6379",
            "",
        )
        .expect_err("multi-url in standalone should reject");
        assert!(err.contains("MX_REDIS_MODE=cluster"), "{err}");
    }

    #[test]
    fn rejects_unknown_and_includes_env_name() {
        let err = BackendConfig::from_type_str("MX_WHATEVER", "sqlite", "", "")
            .expect_err("should reject");
        assert!(
            err.contains("MX_WHATEVER"),
            "error should name the env var: {err}"
        );
        assert!(
            err.contains("'sqlite'"),
            "error should echo bad value: {err}"
        );
    }

    #[cfg(feature = "memory-backend")]
    #[test]
    fn parses_memory_backend() {
        let cfg = BackendConfig::from_type_str("X", "memory", "", "").expect("memory");
        assert!(matches!(cfg, BackendConfig::Memory));
        assert_eq!(cfg.to_string(), "memory");
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
        let cluster = BackendConfig::RedisCluster {
            urls: vec!["redis://n1:6379".to_string(), "redis://n2:6379".to_string()],
        };
        assert_eq!(cluster.to_string(), "redis-cluster");
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
    fn from_env_reads_mx_metadata_backend_and_redis_mode() {
        let lock = acquire_env_mutex();
        let _m = EnvVarGuard::remove(&lock, "MX_REDIS_MODE");
        let _g1 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis");
        let _g2 = EnvVarGuard::set(&lock, "REDIS_URL", "redis://myhost:7777");
        let cfg = BackendConfig::from_env().expect("from_env redis");
        assert!(matches!(cfg, BackendConfig::Redis { url } if url == "redis://myhost:7777"));

        let _m = EnvVarGuard::set(&lock, "MX_REDIS_MODE", "cluster");
        let _g2 = EnvVarGuard::set(
            &lock,
            "REDIS_URL",
            "redis://node1:6379, redis://node2:6379, redis://node3:6379",
        );
        let cfg = BackendConfig::from_env().expect("from_env redis cluster via mode");
        assert!(matches!(
            cfg,
            BackendConfig::RedisCluster { urls }
            if urls == vec!["redis://node1:6379", "redis://node2:6379", "redis://node3:6379"]
        ));

        let _m = EnvVarGuard::remove(&lock, "MX_REDIS_MODE");
        let _g1 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis-cluster");
        let cfg = BackendConfig::from_env().expect("from_env redis-cluster alias");
        assert!(matches!(cfg, BackendConfig::RedisCluster { urls } if urls.len() == 3));
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_redis_cluster_supports_host_list_and_single_host() {
        let lock = acquire_env_mutex();
        let _g0 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis-cluster");
        let _u1 = EnvVarGuard::remove(&lock, "MX_REDIS_URL");
        let _u2 = EnvVarGuard::remove(&lock, "REDIS_URL");
        let _m = EnvVarGuard::remove(&lock, "MX_REDIS_MODE");

        let _h = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "redis-cluster");
        let _p = EnvVarGuard::set(&lock, "MX_REDIS_PORT", "7000");
        let cfg = BackendConfig::from_env().expect("cluster from single host+port");
        assert!(
            matches!(cfg, BackendConfig::RedisCluster { urls } if urls == vec!["redis://redis-cluster:7000"])
        );

        let _h = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "n1:7000, n2:7001, n3");
        let cfg = BackendConfig::from_env().expect("cluster from mixed host list");
        assert!(matches!(
            cfg,
            BackendConfig::RedisCluster { urls }
            if urls == vec!["redis://n1:7000", "redis://n2:7001", "redis://n3:7000"]
        ));
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_rejects_cluster_with_standalone_mode_conflict() {
        let lock = acquire_env_mutex();
        let _g0 = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis-cluster");
        let _m = EnvVarGuard::set(&lock, "MX_REDIS_MODE", "standalone");
        let _u = EnvVarGuard::set(&lock, "REDIS_URL", "redis://localhost:6379");
        let err = BackendConfig::from_env().expect_err("conflict should error");
        assert!(err.contains("conflicts"), "{err}");
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
        let _g2 = EnvVarGuard::remove(&lock, "MX_REDIS_URL");
        let _g3 = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "myhost");
        let _g4 = EnvVarGuard::set(&lock, "MX_REDIS_PORT", "6380");
        assert_eq!(
            BackendConfig::redis_url_from_env().expect("host+port build"),
            "redis://myhost:6380"
        );
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn redis_url_from_env_errors_when_host_and_port_missing() {
        let lock = acquire_env_mutex();
        let _g0 = EnvVarGuard::remove(&lock, "MX_REDIS_URL");
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
        let _g0 = EnvVarGuard::remove(&lock, "MX_REDIS_URL");
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

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn preserves_commas_in_credentials_for_standalone_and_cluster_urls() {
        let cfg = BackendConfig::from_type_str(
            "MX_METADATA_BACKEND",
            "redis",
            "redis://default:pa,ss@localhost:6379/0",
            "",
        )
        .expect("standalone URL with comma in password should parse");
        assert!(matches!(
            cfg,
            BackendConfig::Redis { url } if url == "redis://default:pa,ss@localhost:6379/0"
        ));

        let cluster = BackendConfig::from_type_str(
            "MX_METADATA_BACKEND",
            "redis-cluster",
            "redis://u:p,1@n1:7000, redis://u:p,2@n2:7001",
            "",
        )
        .expect("cluster URLs with commas in passwords should parse");
        assert!(matches!(
            cluster,
            BackendConfig::RedisCluster { urls }
            if urls == vec!["redis://u:p,1@n1:7000", "redis://u:p,2@n2:7001"]
        ));

        let lock = acquire_env_mutex();
        let _m = EnvVarGuard::remove(&lock, "MX_REDIS_MODE");
        let _b = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis");
        let _u = EnvVarGuard::set(&lock, "REDIS_URL", "redis://default:pa,ss@localhost:6379/0");
        let env_cfg = BackendConfig::from_env().expect("from_env standalone credential comma");
        assert!(matches!(
            env_cfg,
            BackendConfig::Redis { url } if url == "redis://default:pa,ss@localhost:6379/0"
        ));
    }

    #[test]
    #[allow(clippy::await_holding_lock)]
    fn from_env_validates_redis_mode_and_cluster_host_edge_cases() {
        assert_eq!(
            RedisMode::parse_opt(Some("   ")).expect("whitespace defaults to standalone"),
            RedisMode::Standalone
        );

        let lock = acquire_env_mutex();
        let _u1 = EnvVarGuard::remove(&lock, "MX_REDIS_URL");
        let _u2 = EnvVarGuard::remove(&lock, "REDIS_URL");
        let _h = EnvVarGuard::remove(&lock, "MX_REDIS_HOST");
        let _h2 = EnvVarGuard::remove(&lock, "REDIS_HOST");
        let _p = EnvVarGuard::remove(&lock, "MX_REDIS_PORT");
        let _p2 = EnvVarGuard::remove(&lock, "REDIS_PORT");

        let err = BackendConfig::redis_cluster_urls_from_env()
            .expect_err("cluster without URL or host must fail");
        assert!(err.contains("MX_REDIS_URL") && err.contains("MX_REDIS_HOST"));

        let _h = EnvVarGuard::set(&lock, "MX_REDIS_HOST", " , ");
        let err = BackendConfig::redis_cluster_urls_from_env()
            .expect_err("empty cluster host list must fail");
        assert!(err.contains("cannot be empty"));

        let _h = EnvVarGuard::set(&lock, "MX_REDIS_HOST", "node-without-port");
        let err = BackendConfig::redis_cluster_urls_from_env()
            .expect_err("portless cluster host without port env must fail");
        assert!(err.contains("has no port"));

        let _h = EnvVarGuard::set(
            &lock,
            "MX_REDIS_HOST",
            "rediss://secure:7000, [::1]:7001, [::1]",
        );
        let _p = EnvVarGuard::set(&lock, "MX_REDIS_PORT", "7002");
        let urls = BackendConfig::redis_cluster_urls_from_env().expect("ipv6 and scheme hosts");
        assert_eq!(
            urls,
            vec![
                "rediss://secure:7000",
                "redis://[::1]:7001",
                "redis://[::1]:7002",
            ]
        );

        let _b = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "redis-cluster");
        let _m = EnvVarGuard::set(&lock, "MX_REDIS_MODE", "cluster");
        let cfg = BackendConfig::from_env().expect("redis-cluster + cluster mode allowed");
        assert!(matches!(cfg, BackendConfig::RedisCluster { .. }));

        let _b = EnvVarGuard::set(&lock, "MX_METADATA_BACKEND", "bogus");
        let err = BackendConfig::from_env().expect_err("unknown backend must fail");
        assert!(err.contains("'bogus'"));
    }
}
