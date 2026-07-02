// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Centralized registry of every environment variable the workspace reads.
//!
//! Inspired by vLLM's `envs.py`: this module is the single source of truth for
//! env-var *names* (the `pub const` block) and provides typed *getters* that
//! encapsulate defaults, fallback chains, and parsing. Both ModelExpress-owned
//! variables (`MODEL_EXPRESS_*`, `MX_*`) and third-party variables the code
//! depends on (`HF_*`, `NGC_*`, `REDIS_*`, `POD_NAMESPACE`, `HOME`) live here.
//!
//! # Conventions
//! - Getters read `std::env` on **every** call and never cache. Some callers
//!   (and tests via [`crate::test_support::EnvVarGuard`]) mutate the process
//!   environment at runtime, so a cached value would go stale.
//! - The name constants are referenced directly from clap `#[arg(env = ...)]`
//!   attributes so the CLI and the getters can never drift apart.
//!
//! # Not covered here
//! - `CARGO_PKG_VERSION` is read at compile time via the `env!` macro in
//!   `modelexpress_server::services`; it is not a runtime variable.
//! - `HF_ENDPOINT` is read directly by the `hf_hub` crate (via
//!   `ApiBuilder::from_env`); ModelExpress only sets it in tests. Its name is
//!   registered below for reference.

use crate::Error;
use crate::constants;
use std::env;
use std::path::PathBuf;

// ── Config-loader prefix ────────────────────────────────────────────────────
/// Prefix consumed by the `config` crate's `Environment` source in
/// [`crate::config::load_layered_config`] (env vars like `MODEL_EXPRESS_*`
/// override matching config-file fields).
pub const MODEL_EXPRESS_PREFIX: &str = "MODEL_EXPRESS";

// ── ModelExpress-owned variables ────────────────────────────────────────────
/// Client server endpoint (`ClientArgs::endpoint`).
pub const MODEL_EXPRESS_ENDPOINT: &str = "MODEL_EXPRESS_ENDPOINT";
/// Client request timeout in seconds (`ClientArgs::timeout`).
pub const MODEL_EXPRESS_TIMEOUT: &str = "MODEL_EXPRESS_TIMEOUT";
/// Local model cache directory (client, server, and both providers).
pub const MODEL_EXPRESS_CACHE_DIRECTORY: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";
/// Log level (client and server).
pub const MODEL_EXPRESS_LOG_LEVEL: &str = "MODEL_EXPRESS_LOG_LEVEL";
/// Log output format (client and server).
pub const MODEL_EXPRESS_LOG_FORMAT: &str = "MODEL_EXPRESS_LOG_FORMAT";
/// Maximum connection/request retries (`ClientArgs::max_retries`).
pub const MODEL_EXPRESS_MAX_RETRIES: &str = "MODEL_EXPRESS_MAX_RETRIES";
/// Delay between retries in seconds (`ClientArgs::retry_delay`).
pub const MODEL_EXPRESS_RETRY_DELAY: &str = "MODEL_EXPRESS_RETRY_DELAY";
/// Disable shared-storage mode (`ClientArgs::no_shared_storage`).
pub const MODEL_EXPRESS_NO_SHARED_STORAGE: &str = "MODEL_EXPRESS_NO_SHARED_STORAGE";
/// File-transfer chunk size in bytes (`ClientArgs::transfer_chunk_size`).
pub const MODEL_EXPRESS_TRANSFER_CHUNK_SIZE: &str = "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE";
/// gRPC server listen port (`ServerArgs::port`).
pub const MODEL_EXPRESS_SERVER_PORT: &str = "MODEL_EXPRESS_SERVER_PORT";
/// Server host/bind address (`ServerArgs::host`).
pub const MODEL_EXPRESS_SERVER_HOST: &str = "MODEL_EXPRESS_SERVER_HOST";
/// Toggle the background cache-eviction sweeper (`ServerArgs::cache_eviction_enabled`).
pub const MODEL_EXPRESS_CACHE_EVICTION_ENABLED: &str = "MODEL_EXPRESS_CACHE_EVICTION_ENABLED";
/// Server endpoint used by the cache module's default-endpoint helper.
pub const MODEL_EXPRESS_SERVER_ENDPOINT: &str = "MODEL_EXPRESS_SERVER_ENDPOINT";

// ── HuggingFace ─────────────────────────────────────────────────────────────
/// HuggingFace Hub auth token.
pub const HF_TOKEN: &str = "HF_TOKEN";
/// HuggingFace Hub cache directory.
pub const HF_HUB_CACHE: &str = "HF_HUB_CACHE";
/// Enables HuggingFace offline mode.
pub const HF_HUB_OFFLINE: &str = "HF_HUB_OFFLINE";
/// HuggingFace Hub endpoint override. Read directly by the `hf_hub` crate;
/// registered here for reference (ModelExpress only sets it in tests).
pub const HF_ENDPOINT: &str = "HF_ENDPOINT";

// ── NGC ─────────────────────────────────────────────────────────────────────
/// Base URL for the NGC artifact/download API.
pub const NGC_API_ENDPOINT: &str = "NGC_API_ENDPOINT";
/// Base URL for the NGC authentication endpoint.
pub const NGC_AUTH_ENDPOINT: &str = "NGC_AUTH_ENDPOINT";
/// NGC API key.
pub const NGC_API_KEY: &str = "NGC_API_KEY";
/// Alternate NGC CLI API key.
pub const NGC_CLI_API_KEY: &str = "NGC_CLI_API_KEY";
/// Root directory used to locate the NGC CLI config file (`~/.ngc/config`).
pub const NGC_CLI_HOME: &str = "NGC_CLI_HOME";

/// Default NGC API base URL when [`NGC_API_ENDPOINT`] is unset.
pub const DEFAULT_NGC_API_BASE: &str = "https://api.ngc.nvidia.com";
/// Default NGC auth base URL when [`NGC_AUTH_ENDPOINT`] is unset.
pub const DEFAULT_NGC_AUTHN_BASE: &str = "https://authn.nvidia.com";

// ── Redis / metadata backend (server) ───────────────────────────────────────
/// Selects the metadata backend implementation (`redis`, `kubernetes`, `memory`).
pub const MX_METADATA_BACKEND: &str = "MX_METADATA_BACKEND";
/// Full Redis connection URL for the redis metadata backend.
pub const REDIS_URL: &str = "REDIS_URL";
/// Redis host (preferred) when building the URL from host + port.
pub const MX_REDIS_HOST: &str = "MX_REDIS_HOST";
/// Redis host alias for charts predating the `MX_` prefix.
pub const REDIS_HOST: &str = "REDIS_HOST";
/// Redis port (preferred) when building the URL from host + port.
pub const MX_REDIS_PORT: &str = "MX_REDIS_PORT";
/// Redis port alias for charts predating the `MX_` prefix.
pub const REDIS_PORT: &str = "REDIS_PORT";
/// Kubernetes namespace for ModelCacheEntry CRs (overrides [`POD_NAMESPACE`]).
pub const MX_METADATA_NAMESPACE: &str = "MX_METADATA_NAMESPACE";
/// Kubernetes namespace injected via the downward API for in-cluster pods.
pub const POD_NAMESPACE: &str = "POD_NAMESPACE";

// ── Reaper (server) ─────────────────────────────────────────────────────────
/// Interval (seconds) between reaper scans for stale/GC worker sweeps.
pub const MX_REAPER_SCAN_INTERVAL_SECS: &str = "MX_REAPER_SCAN_INTERVAL_SECS";
/// Age (seconds) after which an active worker's heartbeat is considered stale.
pub const MX_HEARTBEAT_TIMEOUT_SECS: &str = "MX_HEARTBEAT_TIMEOUT_SECS";
/// Age (seconds) after which a STALE worker is garbage-collected.
pub const MX_GC_TIMEOUT_SECS: &str = "MX_GC_TIMEOUT_SECS";

// ── P2P source selection (server) ─────────────────────────────────────────────
/// Sliding-window TTL (seconds) over which a source's recent `GetMetadata`
/// selections are counted into the `active_transfers` load estimate returned by
/// `ListSources`. Consumed by the client `load_aware` selector.
pub const MX_P2P_SOURCE_LOAD_TTL_SECS: &str = "MX_P2P_SOURCE_LOAD_TTL_SECS";

// ── System ──────────────────────────────────────────────────────────────────
/// Primary source for the user's home directory.
pub const HOME: &str = "HOME";
/// Windows fallback for the home directory when [`HOME`] is unset.
pub const USERPROFILE: &str = "USERPROFILE";
/// Path to a kubeconfig file (consumed by the k8s integration tests).
pub const KUBECONFIG: &str = "KUBECONFIG";

// ── Default reaper timings ───────────────────────────────────────────────────
const DEFAULT_REAPER_SCAN_INTERVAL_SECS: u64 = 30;
const DEFAULT_HEARTBEAT_TIMEOUT_SECS: u64 = 90;
const DEFAULT_GC_TIMEOUT_SECS: u64 = 3600;

// ── Default P2P source-load window ────────────────────────────────────────────
/// Roughly a small-model P2P transfer duration: long enough that a selected
/// source still counts while its target is pulling, short enough to decay once
/// the burst passes.
const DEFAULT_P2P_SOURCE_LOAD_TTL_SECS: u64 = 60;

// ── Getters ───────────────────────────────────────────────────────────────

/// Resolve the user's home directory: [`HOME`], then [`USERPROFILE`].
///
/// # Errors
/// Returns an error when neither variable is set.
pub fn home_dir() -> std::result::Result<String, Box<Error>> {
    env::var(HOME)
        .or_else(|_| env::var(USERPROFILE))
        .map_err(|e| Error::Generic(format!("Failed to get home directory: {e}")).into())
}

/// Home directory as a `PathBuf`, falling back to `.` when unresolved.
pub fn home_dir_or_cwd() -> PathBuf {
    PathBuf::from(home_dir().unwrap_or_else(|_| ".".to_string()))
}

/// Model cache directory override from [`MODEL_EXPRESS_CACHE_DIRECTORY`].
pub fn cache_directory() -> Option<PathBuf> {
    env::var(MODEL_EXPRESS_CACHE_DIRECTORY)
        .ok()
        .map(PathBuf::from)
}

/// Default server gRPC endpoint: [`MODEL_EXPRESS_SERVER_ENDPOINT`] or
/// `http://localhost:{DEFAULT_GRPC_PORT}`. Not normalized.
pub fn server_endpoint_or_default() -> String {
    env::var(MODEL_EXPRESS_SERVER_ENDPOINT)
        .unwrap_or_else(|_| format!("http://localhost:{}", constants::DEFAULT_GRPC_PORT))
}

/// HuggingFace Hub token from [`HF_TOKEN`].
pub fn hf_token() -> Option<String> {
    env::var(HF_TOKEN).ok()
}

/// HuggingFace Hub cache directory from [`HF_HUB_CACHE`].
pub fn hf_hub_cache() -> Option<PathBuf> {
    env::var(HF_HUB_CACHE).ok().map(PathBuf::from)
}

/// Whether HuggingFace offline mode is enabled via [`HF_HUB_OFFLINE`].
/// Enabled when the value is one of `1`, `ON`, `YES`, `TRUE` (case-insensitive).
pub fn hf_offline() -> bool {
    env::var(HF_HUB_OFFLINE)
        .map(|v| matches!(v.to_uppercase().as_str(), "1" | "ON" | "YES" | "TRUE"))
        .unwrap_or(false)
}

/// NGC API base URL: [`NGC_API_ENDPOINT`] or [`DEFAULT_NGC_API_BASE`].
pub fn ngc_api_base() -> String {
    env::var(NGC_API_ENDPOINT).unwrap_or_else(|_| DEFAULT_NGC_API_BASE.to_string())
}

/// NGC auth base URL: [`NGC_AUTH_ENDPOINT`] or [`DEFAULT_NGC_AUTHN_BASE`].
pub fn ngc_authn_base() -> String {
    env::var(NGC_AUTH_ENDPOINT).unwrap_or_else(|_| DEFAULT_NGC_AUTHN_BASE.to_string())
}

/// NGC API key from [`NGC_API_KEY`], then [`NGC_CLI_API_KEY`].
/// Returns the first non-empty, trimmed value found.
pub fn ngc_api_key() -> Option<String> {
    for var in [NGC_API_KEY, NGC_CLI_API_KEY] {
        if let Ok(v) = env::var(var) {
            let trimmed = v.trim().to_string();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

/// Root directory for the NGC CLI config from [`NGC_CLI_HOME`].
pub fn ngc_cli_home() -> Option<PathBuf> {
    env::var(NGC_CLI_HOME).ok().map(PathBuf::from)
}

/// Raw value of [`MX_METADATA_BACKEND`] (empty string when unset).
pub fn metadata_backend() -> String {
    env::var(MX_METADATA_BACKEND).unwrap_or_default()
}

/// Full Redis URL from [`REDIS_URL`].
pub fn redis_url() -> Option<String> {
    env::var(REDIS_URL).ok()
}

/// Redis host from [`MX_REDIS_HOST`], then [`REDIS_HOST`].
pub fn redis_host() -> Option<String> {
    env::var(MX_REDIS_HOST)
        .or_else(|_| env::var(REDIS_HOST))
        .ok()
}

/// Redis port from [`MX_REDIS_PORT`], then [`REDIS_PORT`].
pub fn redis_port() -> Option<String> {
    env::var(MX_REDIS_PORT)
        .or_else(|_| env::var(REDIS_PORT))
        .ok()
}

/// Kubernetes namespace from [`MX_METADATA_NAMESPACE`], then [`POD_NAMESPACE`].
pub fn metadata_namespace() -> Option<String> {
    env::var(MX_METADATA_NAMESPACE)
        .or_else(|_| env::var(POD_NAMESPACE))
        .ok()
}

/// Reaper scan interval in seconds ([`MX_REAPER_SCAN_INTERVAL_SECS`], default 30).
pub fn reaper_scan_interval_secs() -> u64 {
    env_u64(
        MX_REAPER_SCAN_INTERVAL_SECS,
        DEFAULT_REAPER_SCAN_INTERVAL_SECS,
    )
}

/// Heartbeat staleness timeout in seconds ([`MX_HEARTBEAT_TIMEOUT_SECS`], default 90).
pub fn heartbeat_timeout_secs() -> u64 {
    env_u64(MX_HEARTBEAT_TIMEOUT_SECS, DEFAULT_HEARTBEAT_TIMEOUT_SECS)
}

/// Garbage-collection timeout in seconds ([`MX_GC_TIMEOUT_SECS`], default 3600).
pub fn gc_timeout_secs() -> u64 {
    env_u64(MX_GC_TIMEOUT_SECS, DEFAULT_GC_TIMEOUT_SECS)
}

/// P2P source-load TTL window in seconds
/// ([`MX_P2P_SOURCE_LOAD_TTL_SECS`], default 60).
pub fn p2p_source_load_ttl_secs() -> u64 {
    env_u64(
        MX_P2P_SOURCE_LOAD_TTL_SECS,
        DEFAULT_P2P_SOURCE_LOAD_TTL_SECS,
    )
}

/// Read an environment variable as `u64`, falling back to `default`.
fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, acquire_env_mutex};

    #[test]
    fn name_constants_match_their_literals() {
        assert_eq!(MODEL_EXPRESS_PREFIX, "MODEL_EXPRESS");
        assert_eq!(MODEL_EXPRESS_ENDPOINT, "MODEL_EXPRESS_ENDPOINT");
        assert_eq!(MODEL_EXPRESS_TIMEOUT, "MODEL_EXPRESS_TIMEOUT");
        assert_eq!(
            MODEL_EXPRESS_CACHE_DIRECTORY,
            "MODEL_EXPRESS_CACHE_DIRECTORY"
        );
        assert_eq!(MODEL_EXPRESS_LOG_LEVEL, "MODEL_EXPRESS_LOG_LEVEL");
        assert_eq!(MODEL_EXPRESS_LOG_FORMAT, "MODEL_EXPRESS_LOG_FORMAT");
        assert_eq!(MODEL_EXPRESS_MAX_RETRIES, "MODEL_EXPRESS_MAX_RETRIES");
        assert_eq!(MODEL_EXPRESS_RETRY_DELAY, "MODEL_EXPRESS_RETRY_DELAY");
        assert_eq!(
            MODEL_EXPRESS_NO_SHARED_STORAGE,
            "MODEL_EXPRESS_NO_SHARED_STORAGE"
        );
        assert_eq!(
            MODEL_EXPRESS_TRANSFER_CHUNK_SIZE,
            "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE"
        );
        assert_eq!(MODEL_EXPRESS_SERVER_PORT, "MODEL_EXPRESS_SERVER_PORT");
        assert_eq!(MODEL_EXPRESS_SERVER_HOST, "MODEL_EXPRESS_SERVER_HOST");
        assert_eq!(
            MODEL_EXPRESS_CACHE_EVICTION_ENABLED,
            "MODEL_EXPRESS_CACHE_EVICTION_ENABLED"
        );
        assert_eq!(
            MODEL_EXPRESS_SERVER_ENDPOINT,
            "MODEL_EXPRESS_SERVER_ENDPOINT"
        );
        assert_eq!(HF_TOKEN, "HF_TOKEN");
        assert_eq!(HF_HUB_CACHE, "HF_HUB_CACHE");
        assert_eq!(HF_HUB_OFFLINE, "HF_HUB_OFFLINE");
        assert_eq!(HF_ENDPOINT, "HF_ENDPOINT");
        assert_eq!(NGC_API_ENDPOINT, "NGC_API_ENDPOINT");
        assert_eq!(NGC_AUTH_ENDPOINT, "NGC_AUTH_ENDPOINT");
        assert_eq!(NGC_API_KEY, "NGC_API_KEY");
        assert_eq!(NGC_CLI_API_KEY, "NGC_CLI_API_KEY");
        assert_eq!(NGC_CLI_HOME, "NGC_CLI_HOME");
        assert_eq!(MX_METADATA_BACKEND, "MX_METADATA_BACKEND");
        assert_eq!(REDIS_URL, "REDIS_URL");
        assert_eq!(MX_REDIS_HOST, "MX_REDIS_HOST");
        assert_eq!(REDIS_HOST, "REDIS_HOST");
        assert_eq!(MX_REDIS_PORT, "MX_REDIS_PORT");
        assert_eq!(REDIS_PORT, "REDIS_PORT");
        assert_eq!(MX_METADATA_NAMESPACE, "MX_METADATA_NAMESPACE");
        assert_eq!(POD_NAMESPACE, "POD_NAMESPACE");
        assert_eq!(MX_REAPER_SCAN_INTERVAL_SECS, "MX_REAPER_SCAN_INTERVAL_SECS");
        assert_eq!(MX_HEARTBEAT_TIMEOUT_SECS, "MX_HEARTBEAT_TIMEOUT_SECS");
        assert_eq!(MX_GC_TIMEOUT_SECS, "MX_GC_TIMEOUT_SECS");
        assert_eq!(MX_P2P_SOURCE_LOAD_TTL_SECS, "MX_P2P_SOURCE_LOAD_TTL_SECS");
        assert_eq!(HOME, "HOME");
        assert_eq!(USERPROFILE, "USERPROFILE");
        assert_eq!(KUBECONFIG, "KUBECONFIG");
    }

    #[test]
    fn hf_offline_parses_truthy_values() {
        let lock = acquire_env_mutex();
        for truthy in ["1", "on", "YES", "true", "True"] {
            let _g = EnvVarGuard::set(&lock, HF_HUB_OFFLINE, truthy);
            assert!(hf_offline(), "expected {truthy} to enable offline mode");
        }
        for falsey in ["0", "off", "no", "maybe"] {
            let _g = EnvVarGuard::set(&lock, HF_HUB_OFFLINE, falsey);
            assert!(!hf_offline(), "expected {falsey} to disable offline mode");
        }
        let _g = EnvVarGuard::remove(&lock, HF_HUB_OFFLINE);
        assert!(!hf_offline(), "unset should disable offline mode");
    }

    #[test]
    fn ngc_bases_default_then_override() {
        let lock = acquire_env_mutex();
        let _api = EnvVarGuard::remove(&lock, NGC_API_ENDPOINT);
        let _authn = EnvVarGuard::remove(&lock, NGC_AUTH_ENDPOINT);
        assert_eq!(ngc_api_base(), DEFAULT_NGC_API_BASE);
        assert_eq!(ngc_authn_base(), DEFAULT_NGC_AUTHN_BASE);

        let _api = EnvVarGuard::set(&lock, NGC_API_ENDPOINT, "https://api.example.com");
        let _authn = EnvVarGuard::set(&lock, NGC_AUTH_ENDPOINT, "https://authn.example.com");
        assert_eq!(ngc_api_base(), "https://api.example.com");
        assert_eq!(ngc_authn_base(), "https://authn.example.com");
    }

    #[test]
    fn ngc_api_key_prefers_primary_then_falls_back() {
        let lock = acquire_env_mutex();
        let _p = EnvVarGuard::set(&lock, NGC_API_KEY, "  primary  ");
        let _s = EnvVarGuard::set(&lock, NGC_CLI_API_KEY, "secondary");
        assert_eq!(ngc_api_key().as_deref(), Some("primary"));

        let _p = EnvVarGuard::remove(&lock, NGC_API_KEY);
        assert_eq!(ngc_api_key().as_deref(), Some("secondary"));

        let _s = EnvVarGuard::remove(&lock, NGC_CLI_API_KEY);
        assert_eq!(ngc_api_key(), None);
    }

    #[test]
    fn redis_and_namespace_fallbacks() {
        let lock = acquire_env_mutex();
        let _h1 = EnvVarGuard::remove(&lock, MX_REDIS_HOST);
        let _h2 = EnvVarGuard::set(&lock, REDIS_HOST, "legacy-host");
        assert_eq!(redis_host().as_deref(), Some("legacy-host"));
        let _h1 = EnvVarGuard::set(&lock, MX_REDIS_HOST, "mx-host");
        assert_eq!(redis_host().as_deref(), Some("mx-host"));

        let _n1 = EnvVarGuard::remove(&lock, MX_METADATA_NAMESPACE);
        let _n2 = EnvVarGuard::set(&lock, POD_NAMESPACE, "pod-ns");
        assert_eq!(metadata_namespace().as_deref(), Some("pod-ns"));
    }

    #[test]
    fn reaper_getters_default_parse_and_fallback() {
        let lock = acquire_env_mutex();
        let _g = EnvVarGuard::remove(&lock, MX_REAPER_SCAN_INTERVAL_SECS);
        assert_eq!(
            reaper_scan_interval_secs(),
            DEFAULT_REAPER_SCAN_INTERVAL_SECS
        );

        let _g = EnvVarGuard::set(&lock, MX_HEARTBEAT_TIMEOUT_SECS, "120");
        assert_eq!(heartbeat_timeout_secs(), 120);

        let _g = EnvVarGuard::set(&lock, MX_GC_TIMEOUT_SECS, "not-a-number");
        assert_eq!(gc_timeout_secs(), DEFAULT_GC_TIMEOUT_SECS);
    }
}
