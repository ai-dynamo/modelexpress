// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Redis connection abstraction supporting both standalone Redis and Redis Cluster.
//!
//! All four Redis-backed stores (model registry, P2P metadata, refit, and collective refit)
//! share this module for:
//! - Connecting to either a standalone `ConnectionManager` or a cluster `ClusterConnection`.
//! - Fast, actionable startup topology verification (`INFO cluster` on standalone connections;
//!   explicit cluster-disabled diagnostics when cluster mode points at a standalone server).
//! - Multi-primary `SCAN` (`scan_match`) iterating every active primary in cluster mode.
//! - Cross-slot-safe batch hash reads (`hgetall_many` and `hget_many`).
//! - Safe URL credential redaction for logs and error messages.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::future::try_join_all;
use modelexpress_common::envs;
use redis::aio::{ConnectionLike, ConnectionManager};
use redis::cluster::ClusterClient;
use redis::cluster_async::ClusterConnection;
use redis::cluster_routing::{RoutingInfo, SingleNodeRoutingInfo};
use redis::{
    Cmd, ConnectionAddr, ErrorKind, FromRedisValue, IntoConnectionInfo, Pipeline, RedisError,
    RedisFuture, Value,
};

/// Redis connection configuration for any metadata backend store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedisEndpointConfig {
    /// Single-instance or primary/replica Redis addressed by one URL.
    Standalone(String),
    /// Multi-primary Redis Cluster discovered from one or more seed URLs.
    Cluster(Vec<String>),
}

impl RedisEndpointConfig {
    /// Returns `true` when configured for Redis Cluster.
    #[must_use]
    pub fn is_cluster(&self) -> bool {
        matches!(self, Self::Cluster(_))
    }

    /// Canonical backend label for logs and metrics.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Standalone(_) => "redis",
            Self::Cluster(_) => "redis-cluster",
        }
    }

    /// Credential-redacted endpoint description suitable for logs and errors.
    #[must_use]
    pub fn redacted_display(&self) -> String {
        match self {
            Self::Standalone(url) => redact_url(url),
            Self::Cluster(urls) => redact_urls(urls),
        }
    }

    /// Open a connection and validate that the server topology matches this configuration.
    ///
    /// - In `Standalone` mode, verifies `INFO cluster` does not report `cluster_enabled:1`,
    ///   failing fast with an actionable error instead of hitting `MOVED` redirections mid-run.
    /// - In `Cluster` mode, opens a `ClusterConnection` from the configured seed URLs (with
    ///   failover-friendly retry settings) and verifies the target supports cluster mode and
    ///   answers `PING`.
    pub async fn connect(&self) -> Result<RedisConnection, String> {
        match self {
            Self::Standalone(url) => connect_standalone(url).await,
            Self::Cluster(urls) => connect_cluster(urls).await,
        }
    }
}

/// Active async Redis connection, unified over standalone and cluster transports.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // Avoids heap-allocating on every per-RPC ConnectionManager::clone().
pub enum RedisConnection {
    Standalone(ConnectionManager),
    Cluster {
        conn: ClusterConnection,
        seed_nodes: Arc<[(String, u16)]>,
    },
}

impl ConnectionLike for RedisConnection {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, Value> {
        match self {
            Self::Standalone(conn) => conn.req_packed_command(cmd),
            Self::Cluster { conn, .. } => conn.req_packed_command(cmd),
        }
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        cmd: &'a Pipeline,
        offset: usize,
        count: usize,
    ) -> RedisFuture<'a, Vec<Value>> {
        match self {
            Self::Standalone(conn) => conn.req_packed_commands(cmd, offset, count),
            Self::Cluster { conn, .. } => conn.req_packed_commands(cmd, offset, count),
        }
    }

    fn get_db(&self) -> i64 {
        match self {
            Self::Standalone(conn) => conn.get_db(),
            Self::Cluster { conn, .. } => conn.get_db(),
        }
    }
}

impl RedisConnection {
    /// Returns `true` if backed by a `ClusterConnection`.
    #[must_use]
    pub fn is_cluster(&self) -> bool {
        matches!(self, Self::Cluster { .. })
    }

    /// Execute `SCAN cursor MATCH pattern COUNT count` across all primaries and return
    /// deduplicated matching keys.
    ///
    /// On a standalone connection this loops a single `SCAN` cursor to `0`.
    /// On a Redis Cluster connection `SCAN` cannot be routed generically because plain `SCAN`
    /// only visits one random node; instead this discovers every primary node via `CLUSTER SLOTS`
    /// and drives each primary's cursor independently to `0`.
    pub async fn scan_match(
        &mut self,
        pattern: &str,
        count: usize,
    ) -> Result<Vec<String>, RedisError> {
        match self {
            Self::Standalone(conn) => scan_single_node(conn, pattern, count).await,
            Self::Cluster { conn, seed_nodes } => {
                scan_all_cluster_masters(conn, seed_nodes, pattern, count).await
            }
        }
    }

    /// Execute `HGETALL` for a slice of keys that may reside in different hash slots.
    ///
    /// On standalone Redis this uses a single pipeline round-trip.
    /// On Redis Cluster, `ClusterConnection` rejects multi-slot pipelines with `CROSSSLOT`,
    /// so requests are fanned out concurrently over cloned multiplexed cluster handles.
    pub async fn hgetall_many<T>(&mut self, keys: &[String]) -> Result<Vec<T>, RedisError>
    where
        T: FromRedisValue + Send,
    {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Standalone(conn) => {
                let mut pipe = redis::pipe();
                for key in keys {
                    pipe.hgetall(key);
                }
                pipe.query_async(conn).await
            }
            Self::Cluster { conn, .. } => {
                let futs = keys.iter().map(|key| {
                    let mut c = conn.clone();
                    let k = key.clone();
                    async move { redis::cmd("HGETALL").arg(&k).query_async::<T>(&mut c).await }
                });
                try_join_all(futs).await
            }
        }
    }

    /// Execute `HGET key field` for a slice of keys that may reside in different hash slots.
    pub async fn hget_many(
        &mut self,
        keys: &[String],
        field: &str,
    ) -> Result<Vec<Option<String>>, RedisError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Standalone(conn) => {
                let mut pipe = redis::pipe();
                for key in keys {
                    pipe.hget(key, field);
                }
                pipe.query_async(conn).await
            }
            Self::Cluster { conn, .. } => {
                let futs = keys.iter().map(|key| {
                    let mut c = conn.clone();
                    let k = key.clone();
                    let f = field.to_string();
                    async move {
                        redis::cmd("HGET")
                            .arg(&k)
                            .arg(&f)
                            .query_async::<Option<String>>(&mut c)
                            .await
                    }
                });
                try_join_all(futs).await
            }
        }
    }
}

fn starts_with_url_scheme(s: &str) -> bool {
    let trimmed = s.trim_start();
    let Some((scheme, _)) = trimmed.split_once("://") else {
        return false;
    };
    let mut chars = scheme.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

fn next_url_split_index(remaining: &str) -> Option<usize> {
    if !starts_with_url_scheme(remaining) {
        return remaining.find(',');
    }
    let scheme_pos = remaining.find("://")?;
    let after_scheme_idx = scheme_pos.saturating_add(3);
    let after_scheme = &remaining[after_scheme_idx..];
    let path_start = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let authority_region = &after_scheme[..path_start];
    let host_start_in_after = authority_region
        .rfind('@')
        .map_or(0, |i| i.saturating_add(1));

    if let Some(rel_comma) = after_scheme[host_start_in_after..path_start].find(',') {
        return Some(
            after_scheme_idx
                .saturating_add(host_start_in_after)
                .saturating_add(rel_comma),
        );
    }

    let suffix = &after_scheme[path_start..];
    let query_or_frag_pos = suffix.find(['?', '#']);
    let scheme_lower = remaining[..scheme_pos].trim_start().to_ascii_lowercase();
    for (rel_comma, _) in suffix.match_indices(',') {
        let abs_comma = after_scheme_idx
            .saturating_add(path_start)
            .saturating_add(rel_comma);
        let after_comma = &remaining[abs_comma.saturating_add(1)..];
        let before_query_or_frag = query_or_frag_pos.is_none_or(|qp| rel_comma < qp);
        if starts_with_url_scheme(after_comma)
            || after_comma.trim().is_empty()
            || (before_query_or_frag
                && (scheme_lower == "redis"
                    || scheme_lower == "rediss"
                    || after_comma.trim_start().starts_with('/')))
        {
            return Some(abs_comma);
        }
    }
    None
}

/// Split comma-separated Redis endpoint URLs while preserving RFC 3986 commas
/// inside URL userinfo credentials and query parameters.
pub(crate) fn split_redis_urls(raw: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let mut remaining = raw.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
    while !remaining.is_empty() {
        let (segment, rest) = match next_url_split_index(remaining) {
            Some(idx) => (&remaining[..idx], &remaining[idx.saturating_add(1)..]),
            None => (remaining, ""),
        };
        let trimmed = segment.trim();
        if !trimmed.is_empty() {
            urls.push(trimmed.to_string());
        }
        remaining = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
    }
    urls
}

async fn connect_standalone(url: &str) -> Result<RedisConnection, String> {
    let urls = split_redis_urls(url);
    if urls.len() > 1 {
        return Err(format!(
            "Multiple comma-separated Redis URLs provided ({}), but standalone Redis mode is active. Set MX_REDIS_MODE=cluster (or MX_METADATA_BACKEND=redis-cluster).",
            redact_urls(&urls)
        ));
    }

    let redacted = redact_url(url);
    let client = redis::Client::open(url)
        .map_err(|e| format!("Failed to open Redis client ({redacted}): {e}"))?;
    let mut conn = ConnectionManager::new(client)
        .await
        .map_err(|e| format!("Failed to connect to Redis ({redacted}): {e}"))?;

    match redis::cmd("INFO")
        .arg("cluster")
        .query_async::<String>(&mut conn)
        .await
    {
        Ok(info) => {
            if is_cluster_enabled_in_info(&info) {
                return Err(format!(
                    "Redis endpoint {redacted} has cluster mode enabled (cluster_enabled:1), but ModelExpress is configured for standalone Redis. Set MX_REDIS_MODE=cluster (or MX_METADATA_BACKEND=redis-cluster)."
                ));
            }
        }
        Err(e) if is_info_probe_unsupported_or_denied(&e) => {
            tracing::warn!(
                "Skipping standalone Redis topology check for {redacted} because INFO cluster is not permitted or unsupported ({e}); continuing with PING"
            );
        }
        Err(e) => {
            return Err(format!(
                "Failed to inspect Redis topology via INFO cluster ({redacted}): {e}"
            ));
        }
    }

    let _: String = redis::cmd("PING")
        .query_async(&mut conn)
        .await
        .map_err(|e| format!("Failed to PING Redis ({redacted}): {e}"))?;

    Ok(RedisConnection::Standalone(conn))
}

fn extract_seed_nodes(urls: &[String]) -> Vec<(String, u16)> {
    let mut seeds = Vec::new();
    for url in urls {
        if let Ok(info) = url.as_str().into_connection_info() {
            let entry = match info.addr {
                ConnectionAddr::Tcp(host, port) | ConnectionAddr::TcpTls { host, port, .. } => {
                    Some((host, port))
                }
                ConnectionAddr::Unix(_) => None,
            };
            if let Some(pair) = entry
                && !seeds.contains(&pair)
            {
                seeds.push(pair);
            }
        }
    }
    seeds
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClusterClientSettings {
    retries: u32,
    min_retry_wait_ms: u64,
    max_retry_wait_ms: u64,
    connection_timeout_ms: u64,
    response_timeout_ms: u64,
}

impl ClusterClientSettings {
    fn from_env() -> Self {
        let min_retry_wait_ms = envs::redis_cluster_min_retry_wait_ms();
        let max_retry_wait_ms = envs::redis_cluster_max_retry_wait_ms().max(min_retry_wait_ms);
        Self {
            retries: envs::redis_cluster_retries(),
            min_retry_wait_ms,
            max_retry_wait_ms,
            connection_timeout_ms: envs::redis_connection_timeout_ms(),
            response_timeout_ms: envs::redis_response_timeout_ms(),
        }
    }
}

async fn connect_cluster(urls: &[String]) -> Result<RedisConnection, String> {
    if urls.is_empty() {
        return Err("Redis Cluster backend requires at least one seed URL.".to_string());
    }
    let redacted = redact_urls(urls);
    let seed_nodes = Arc::from(extract_seed_nodes(urls));
    let settings = ClusterClientSettings::from_env();
    let client = ClusterClient::builder(urls.iter().map(String::as_str))
        .retries(settings.retries)
        .min_retry_wait(settings.min_retry_wait_ms)
        .max_retry_wait(settings.max_retry_wait_ms)
        .retry_wait_formula(settings.min_retry_wait_ms, 2)
        .connection_timeout(Duration::from_millis(settings.connection_timeout_ms))
        .response_timeout(Duration::from_millis(settings.response_timeout_ms))
        .build()
        .map_err(|e| format!("Failed to configure Redis Cluster client ({redacted}): {e}"))?;

    let mut conn = client.get_async_connection().await.map_err(|e| {
        if looks_like_standalone_error(&e) {
            format!(
                "Redis endpoint ({redacted}) does not have cluster mode enabled ({e}), but ModelExpress is configured for Redis Cluster. Unset MX_REDIS_MODE or set MX_REDIS_MODE=standalone."
            )
        } else {
            format!("Failed to connect to Redis Cluster ({redacted}): {e}")
        }
    })?;

    let _: String = redis::cmd("PING")
        .query_async(&mut conn)
        .await
        .map_err(|e| format!("Failed to PING Redis Cluster ({redacted}): {e}"))?;

    Ok(RedisConnection::Cluster { conn, seed_nodes })
}

fn is_cluster_enabled_in_info(info: &str) -> bool {
    info.lines().any(|line| {
        let trimmed = line.trim();
        trimmed == "cluster_enabled:1"
    })
}

fn is_info_probe_unsupported_or_denied(err: &RedisError) -> bool {
    if err.is_io_error()
        || err.is_timeout()
        || err.is_connection_dropped()
        || err.is_connection_refusal()
        || matches!(
            err.kind(),
            ErrorKind::AuthenticationFailed
                | ErrorKind::BusyLoadingError
                | ErrorKind::MasterDown
                | ErrorKind::TryAgain
        )
    {
        return false;
    }
    if matches!(
        err.code(),
        Some("NOAUTH" | "WRONGPASS" | "LOADING" | "MASTERDOWN" | "TRYAGAIN")
    ) {
        return false;
    }
    let msg = err.to_string().to_ascii_lowercase();
    !(msg.contains("noauth")
        || msg.contains("wrongpass")
        || msg.contains("authentication")
        || msg.contains("invalid username-password"))
}

fn looks_like_standalone_error(err: &RedisError) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("cluster support disabled")
        || msg.contains("cluster mode is not enabled")
        || msg.contains("unknown command 'cluster'")
        || msg.contains("unknown command `cluster`")
}

async fn scan_single_node(
    conn: &mut ConnectionManager,
    pattern: &str,
    count: usize,
) -> Result<Vec<String>, RedisError> {
    let mut keys = HashSet::new();
    let mut cursor: u64 = 0;
    loop {
        let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(pattern)
            .arg("COUNT")
            .arg(count)
            .query_async(conn)
            .await?;
        keys.extend(batch);
        cursor = next_cursor;
        if cursor == 0 {
            break;
        }
    }
    Ok(keys.into_iter().collect())
}

const CLUSTER_SCAN_MAX_ATTEMPTS: usize = 3;

trait ClusterNodeRouter {
    fn route<'a>(
        &'a mut self,
        cmd: &'a Cmd,
        routing: SingleNodeRoutingInfo,
    ) -> RedisFuture<'a, Value>;
}

impl ClusterNodeRouter for ClusterConnection {
    fn route<'a>(
        &'a mut self,
        cmd: &'a Cmd,
        routing: SingleNodeRoutingInfo,
    ) -> RedisFuture<'a, Value> {
        Box::pin(self.route_command(cmd, RoutingInfo::SingleNode(routing)))
    }
}

fn is_retryable_cluster_scan_error(err: &RedisError) -> bool {
    err.is_io_error()
        || err.is_timeout()
        || err.is_connection_dropped()
        || err.is_connection_refusal()
        || err.is_cluster_error()
        || matches!(
            err.kind(),
            ErrorKind::ClusterConnectionNotFound
                | ErrorKind::MasterDown
                | ErrorKind::TryAgain
                | ErrorKind::BusyLoadingError
                | ErrorKind::ReadOnly
                | ErrorKind::Moved
                | ErrorKind::Ask
        )
}

async fn discover_cluster_masters_with<R: ClusterNodeRouter + Send>(
    router: &mut R,
    seed_nodes: &[(String, u16)],
) -> Result<Vec<(String, u16)>, RedisError> {
    let slots_cmd = redis::cmd("CLUSTER").arg("SLOTS").to_owned();
    let mut slots_reply: Option<(Value, Option<&str>)> = None;
    for (host, port) in seed_nodes {
        if let Ok(val) = router
            .route(
                &slots_cmd,
                SingleNodeRoutingInfo::ByAddress {
                    host: host.clone(),
                    port: *port,
                },
            )
            .await
        {
            slots_reply = Some((val, Some(host.as_str())));
            break;
        }
    }
    let (slots_val, fallback_host) = match slots_reply {
        Some(pair) => pair,
        None => {
            let val = router
                .route(&slots_cmd, SingleNodeRoutingInfo::Random)
                .await?;
            (val, seed_nodes.first().map(|(h, _)| h.as_str()))
        }
    };
    let masters = parse_cluster_slot_masters(&slots_val, fallback_host)?;
    if masters.is_empty() {
        return Err(RedisError::from((
            ErrorKind::ClusterConnectionNotFound,
            "CLUSTER SLOTS returned no primary nodes",
        )));
    }
    Ok(masters)
}

async fn scan_single_master_with<R: ClusterNodeRouter + Send>(
    router: &mut R,
    host: &str,
    port: u16,
    pattern: &str,
    count: usize,
    keys: &mut HashSet<String>,
) -> Result<(), RedisError> {
    let mut cursor: u64 = 0;
    loop {
        let mut cmd = redis::cmd("SCAN");
        cmd.arg(cursor)
            .arg("MATCH")
            .arg(pattern)
            .arg("COUNT")
            .arg(count);
        let val = router
            .route(
                &cmd,
                SingleNodeRoutingInfo::ByAddress {
                    host: host.to_string(),
                    port,
                },
            )
            .await?;
        let (next_cursor, batch): (u64, Vec<String>) = FromRedisValue::from_redis_value(&val)?;
        keys.extend(batch);
        cursor = next_cursor;
        if cursor == 0 {
            break;
        }
    }
    Ok(())
}

async fn scan_cluster_masters_with<R: ClusterNodeRouter + Send>(
    router: &mut R,
    seed_nodes: &[(String, u16)],
    pattern: &str,
    count: usize,
) -> Result<Vec<String>, RedisError> {
    for attempt in 0..CLUSTER_SCAN_MAX_ATTEMPTS {
        let next_attempt = attempt.saturating_add(1);
        let has_next_attempt = next_attempt < CLUSTER_SCAN_MAX_ATTEMPTS;
        let backoff =
            Duration::from_millis(50u64.saturating_mul(u64::try_from(next_attempt).unwrap_or(1)));

        let initial_masters = match discover_cluster_masters_with(router, seed_nodes).await {
            Ok(masters) => masters,
            Err(err) if has_next_attempt && is_retryable_cluster_scan_error(&err) => {
                tracing::debug!(
                    "Cluster SCAN topology discovery failed on attempt {attempt}: {err}; retrying"
                );
                tokio::time::sleep(backoff).await;
                continue;
            }
            Err(err) => return Err(err),
        };

        let mut keys = HashSet::new();
        let mut scan_failed = false;
        for (host, port) in &initial_masters {
            if let Err(err) =
                scan_single_master_with(router, host, *port, pattern, count, &mut keys).await
            {
                if has_next_attempt && is_retryable_cluster_scan_error(&err) {
                    tracing::debug!(
                        "Cluster SCAN on {host}:{port} failed mid-scan ({err}); refreshing CLUSTER SLOTS and retrying"
                    );
                    scan_failed = true;
                    break;
                }
                return Err(err);
            }
        }
        if scan_failed {
            tokio::time::sleep(backoff).await;
            continue;
        }

        let post_masters = match discover_cluster_masters_with(router, seed_nodes).await {
            Ok(masters) => masters,
            Err(err) if has_next_attempt && is_retryable_cluster_scan_error(&err) => {
                tracing::debug!(
                    "Post-SCAN CLUSTER SLOTS verification failed ({err}); refreshing and retrying"
                );
                tokio::time::sleep(backoff).await;
                continue;
            }
            Err(err) => return Err(err),
        };

        if post_masters == initial_masters {
            return Ok(keys.into_iter().collect());
        }

        if has_next_attempt {
            tracing::debug!(
                "Cluster primary topology changed mid-SCAN; retrying against updated topology"
            );
            tokio::time::sleep(backoff).await;
            continue;
        }

        let scanned: HashSet<(String, u16)> = initial_masters.into_iter().collect();
        for (host, port) in post_masters {
            if !scanned.contains(&(host.clone(), port)) {
                scan_single_master_with(router, &host, port, pattern, count, &mut keys).await?;
            }
        }
        return Ok(keys.into_iter().collect());
    }

    Err(RedisError::from((
        ErrorKind::ClusterConnectionNotFound,
        "Cluster SCAN exhausted retry attempts during topology changes",
    )))
}

async fn scan_all_cluster_masters(
    conn: &mut ClusterConnection,
    seed_nodes: &[(String, u16)],
    pattern: &str,
    count: usize,
) -> Result<Vec<String>, RedisError> {
    scan_cluster_masters_with(conn, seed_nodes, pattern, count).await
}

fn resolve_slot_host(
    host_val: &Value,
    fallback_host: Option<&str>,
) -> Result<Option<String>, RedisError> {
    let resolved_fallback = || {
        fallback_host
            .filter(|h| !h.is_empty())
            .map(ToString::to_string)
            .ok_or_else(|| {
                RedisError::from((
                    ErrorKind::TypeError,
                    "CLUSTER SLOTS returned empty or NULL primary host without responding host fallback",
                ))
            })
    };
    match host_val {
        Value::BulkString(bytes) => {
            let raw = String::from_utf8_lossy(bytes);
            if raw == "?" {
                Ok(None)
            } else if raw.is_empty() {
                resolved_fallback().map(Some)
            } else {
                Ok(Some(raw.into_owned()))
            }
        }
        Value::SimpleString(raw) => {
            if raw == "?" {
                Ok(None)
            } else if raw.is_empty() {
                resolved_fallback().map(Some)
            } else {
                Ok(Some(raw.clone()))
            }
        }
        Value::Nil => resolved_fallback().map(Some),
        _ => Err(RedisError::from((
            ErrorKind::TypeError,
            "Invalid primary host in CLUSTER SLOTS",
        ))),
    }
}

/// Extract unique primary `(host, port)` endpoints from a `CLUSTER SLOTS` response.
fn parse_cluster_slot_masters(
    value: &Value,
    fallback_host: Option<&str>,
) -> Result<Vec<(String, u16)>, RedisError> {
    let Value::Array(ranges) = value else {
        return Err(RedisError::from((
            ErrorKind::TypeError,
            "Expected array from CLUSTER SLOTS",
        )));
    };

    let mut masters = BTreeSet::new();
    for range in ranges {
        let Value::Array(items) = range else {
            return Err(RedisError::from((
                ErrorKind::TypeError,
                "Expected slot range array in CLUSTER SLOTS",
            )));
        };
        let Some(Value::Array(primary)) = items.get(2) else {
            continue;
        };
        let (Some(host_val), Some(port_val)) = (primary.first(), primary.get(1)) else {
            continue;
        };
        let Some(host) = resolve_slot_host(host_val, fallback_host)? else {
            continue;
        };
        let port_i64: i64 = FromRedisValue::from_redis_value(port_val)?;
        let Ok(port) = u16::try_from(port_i64) else {
            return Err(RedisError::from((
                ErrorKind::TypeError,
                "Invalid port number in CLUSTER SLOTS",
            )));
        };
        if !host.is_empty() && port > 0 {
            masters.insert((host, port));
        }
    }
    Ok(masters.into_iter().collect())
}

fn redact_single_url(url: &str) -> String {
    let Some(scheme_sep) = url.find("://") else {
        return redact_authority(url);
    };
    let prefix_end = scheme_sep.saturating_add(3);
    let (scheme, rest) = url.split_at(prefix_end);
    format!("{scheme}{}", redact_authority(rest))
}

/// Redact credentials (`user:password@`, `:password@`, and query credentials such
/// as `?pass=...` / `?user=...`) from a single Redis URL or comma-separated URL list.
#[must_use]
pub fn redact_url(url: &str) -> String {
    let urls = split_redis_urls(url);
    if urls.len() > 1 {
        return urls
            .iter()
            .map(|u| redact_single_url(u))
            .collect::<Vec<_>>()
            .join(", ");
    }
    redact_single_url(url)
}

fn redact_suffix(suffix: &str) -> String {
    let Some((path, query_and_frag)) = suffix.split_once('?') else {
        return suffix.to_string();
    };
    let (query, frag) = match query_and_frag.split_once('#') {
        Some((q, f)) => (q, Some(f)),
        None => (query_and_frag, None),
    };
    let redacted_query = query
        .split('&')
        .map(|pair| {
            if let Some((key, _)) = pair.split_once('=')
                && matches!(
                    key.to_ascii_lowercase().as_str(),
                    "pass" | "password" | "user" | "username"
                )
            {
                format!("{key}=<redacted>")
            } else {
                pair.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("&");
    match frag {
        Some(f) => format!("{path}?{redacted_query}#{f}"),
        None => format!("{path}?{redacted_query}"),
    }
}

fn redact_authority(rest: &str) -> String {
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, suffix) = rest.split_at(authority_end);
    let redacted_suffix = redact_suffix(suffix);
    if let Some((_, host)) = authority.rsplit_once('@') {
        format!("<redacted>@{host}{redacted_suffix}")
    } else {
        format!("{authority}{redacted_suffix}")
    }
}

/// Redact credentials across a slice of Redis seed URLs and join with `, `.
#[must_use]
pub fn redact_urls(urls: &[String]) -> String {
    urls.iter()
        .map(|u| redact_url(u))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    async fn read_resp_args<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
    ) -> Option<Vec<String>> {
        let mut line = String::new();
        if reader.read_line(&mut line).await.ok()? == 0 {
            return None;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if let Some(count_str) = trimmed.strip_prefix('*') {
            let count: usize = count_str.parse().ok()?;
            let mut args = Vec::with_capacity(count);
            for _ in 0..count {
                let mut len_line = String::new();
                if reader.read_line(&mut len_line).await.ok()? == 0 {
                    return None;
                }
                let len: isize = len_line
                    .trim_end_matches(['\r', '\n'])
                    .strip_prefix('$')?
                    .parse()
                    .ok()?;
                if len < 0 {
                    args.push(String::new());
                    continue;
                }
                let len_usize = usize::try_from(len).ok()?;
                let mut buf = vec![0u8; len_usize.saturating_add(2)];
                reader.read_exact(&mut buf).await.ok()?;
                buf.truncate(len_usize);
                args.push(String::from_utf8_lossy(&buf).into_owned());
            }
            Some(args)
        } else {
            Some(
                trimmed
                    .split_whitespace()
                    .map(ToString::to_string)
                    .collect(),
            )
        }
    }

    async fn spawn_mock_resp<F>(
        handler: F,
    ) -> (u16, oneshot::Sender<()>, tokio::task::JoinHandle<()>)
    where
        F: Fn(&[String]) -> Vec<u8> + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock resp");
        let port = listener.local_addr().expect("mock addr").port();
        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        let handler = Arc::new(handler);
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { break };
                        let h = Arc::clone(&handler);
                        tokio::spawn(async move {
                            let (read_half, mut write_half) = stream.into_split();
                            let mut reader = BufReader::new(read_half);
                            while let Some(args) = read_resp_args(&mut reader).await {
                                let reply = h(&args);
                                if write_half.write_all(&reply).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }
                }
            }
        });
        (port, stop_tx, handle)
    }

    #[test]
    fn redacts_single_and_cluster_urls() {
        assert_eq!(
            redact_url("redis://default:s3cr3t@node1:6379/0"),
            "redis://<redacted>@node1:6379/0"
        );
        assert_eq!(
            redact_url("rediss://:pass@cluster.internal:6379"),
            "rediss://<redacted>@cluster.internal:6379"
        );
        assert_eq!(
            redact_url("redis://localhost:6379"),
            "redis://localhost:6379"
        );
        assert_eq!(
            redact_url("unix:///run/redis.sock?pass=example-secret"),
            "unix:///run/redis.sock?pass=<redacted>"
        );
        assert_eq!(
            redact_url(
                "redis+unix:///run/redis.sock?db=2&user=alice&pass=s3cr3t&protocol=resp3#frag"
            ),
            "redis+unix:///run/redis.sock?db=2&user=<redacted>&pass=<redacted>&protocol=resp3#frag"
        );
        assert_eq!(
            redact_urls(&[
                "redis://user:p1@n1:7000".to_string(),
                "redis://user:p2@n2:7001".to_string(),
            ]),
            "redis://<redacted>@n1:7000, redis://<redacted>@n2:7001"
        );
        let standalone = RedisEndpointConfig::Standalone("redis://u:sec@h:6379".to_string());
        assert!(!standalone.is_cluster());
        assert_eq!(standalone.backend_name(), "redis");
        assert_eq!(standalone.redacted_display(), "redis://<redacted>@h:6379");

        let cluster = RedisEndpointConfig::Cluster(vec![
            "redis://u:sec@n1:7000".to_string(),
            "redis://n2:7001".to_string(),
        ]);
        assert!(cluster.is_cluster());
        assert_eq!(cluster.backend_name(), "redis-cluster");
        assert_eq!(
            cluster.redacted_display(),
            "redis://<redacted>@n1:7000, redis://n2:7001"
        );
    }

    #[test]
    fn splits_redis_urls_preserving_credential_and_query_commas() {
        assert_eq!(
            split_redis_urls("redis://default:pa,ss@localhost:6379/0"),
            vec!["redis://default:pa,ss@localhost:6379/0"]
        );
        assert_eq!(
            split_redis_urls("redis://default:p@ss,word@localhost:6379/0"),
            vec!["redis://default:p@ss,word@localhost:6379/0"]
        );
        assert_eq!(
            split_redis_urls("redis://default:p,1@n1:7000/0, redis://default:p,2@n2:7001/0"),
            vec![
                "redis://default:p,1@n1:7000/0",
                "redis://default:p,2@n2:7001/0",
            ]
        );
        assert_eq!(
            split_redis_urls("unix:///run/redis.sock?pass=pa,ss&db=0"),
            vec!["unix:///run/redis.sock?pass=pa,ss&db=0"]
        );
        assert_eq!(
            split_redis_urls("unix:///run/a.sock?pass=p,1,unix:///run/b.sock?pass=p,2"),
            vec!["unix:///run/a.sock?pass=p,1", "unix:///run/b.sock?pass=p,2",]
        );
        assert_eq!(
            split_redis_urls("redis://n1:6379/0,n2:6379"),
            vec!["redis://n1:6379/0", "n2:6379"]
        );
        assert!(split_redis_urls(" ,  , ").is_empty());
    }

    #[test]
    fn detects_cluster_enabled_in_info_output() {
        assert!(is_cluster_enabled_in_info(
            "# Cluster\r\ncluster_enabled:1\r\n"
        ));
        assert!(!is_cluster_enabled_in_info(
            "# Cluster\r\ncluster_enabled:0\r\n"
        ));
        assert!(!is_cluster_enabled_in_info(""));
    }

    #[test]
    fn parses_unique_masters_from_cluster_slots_and_resolves_null_or_empty_host() {
        let slots = Value::Array(vec![
            Value::Array(vec![
                Value::Int(0),
                Value::Int(5460),
                Value::Array(vec![
                    Value::BulkString(b"127.0.0.1".to_vec()),
                    Value::Int(7000),
                    Value::BulkString(b"id1".to_vec()),
                ]),
                Value::Array(vec![
                    Value::BulkString(b"127.0.0.1".to_vec()),
                    Value::Int(7003),
                    Value::BulkString(b"id4".to_vec()),
                ]),
            ]),
            Value::Array(vec![
                Value::Int(5461),
                Value::Int(10922),
                Value::Array(vec![
                    Value::Nil,
                    Value::Int(7001),
                    Value::BulkString(b"id2".to_vec()),
                ]),
            ]),
            Value::Array(vec![
                Value::Int(10923),
                Value::Int(16383),
                Value::Array(vec![
                    Value::BulkString(b"".to_vec()),
                    Value::Int(7002),
                    Value::BulkString(b"id3".to_vec()),
                ]),
            ]),
            Value::Array(vec![
                Value::Int(0),
                Value::Int(100),
                Value::Array(vec![Value::BulkString(b"?".to_vec()), Value::Int(7009)]),
            ]),
        ]);

        let masters =
            parse_cluster_slot_masters(&slots, Some("10.0.0.5")).expect("valid CLUSTER SLOTS");
        assert_eq!(
            masters,
            vec![
                ("10.0.0.5".to_string(), 7001),
                ("10.0.0.5".to_string(), 7002),
                ("127.0.0.1".to_string(), 7000),
            ]
        );

        let missing_fallback = Value::Array(vec![Value::Array(vec![
            Value::Int(0),
            Value::Int(100),
            Value::Array(vec![Value::Nil, Value::Int(7000)]),
        ])]);
        assert!(parse_cluster_slot_masters(&missing_fallback, None).is_err());
        assert!(parse_cluster_slot_masters(&Value::Nil, None).is_err());
        let bad_range = Value::Array(vec![Value::Int(1)]);
        assert!(parse_cluster_slot_masters(&bad_range, None).is_err());
        let bad_port = Value::Array(vec![Value::Array(vec![
            Value::Int(0),
            Value::Int(100),
            Value::Array(vec![Value::BulkString(b"h".to_vec()), Value::Int(-1)]),
        ])]);
        assert!(parse_cluster_slot_masters(&bad_port, None).is_err());
    }

    #[test]
    fn extracts_deduplicated_seed_nodes() {
        let seeds = extract_seed_nodes(&[
            "redis://user:p,1@127.0.0.1:7000".to_string(),
            "rediss://127.0.0.1:7000".to_string(),
            "redis://node2:7001".to_string(),
            "unix:///tmp/redis.sock".to_string(),
            "not-a-url".to_string(),
        ]);
        assert_eq!(
            seeds,
            vec![("127.0.0.1".to_string(), 7000), ("node2".to_string(), 7001),]
        );
    }

    #[tokio::test]
    async fn standalone_connect_permits_acl_denied_info_and_handles_commands() {
        let (port, stop_tx, handle) = spawn_mock_resp(|args| {
            let cmd = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            match cmd.as_str() {
                "INFO" => {
                    b"-NOPERM this user has no permissions to run the 'info' command\r\n".to_vec()
                }
                "PING" => b"+PONG\r\n".to_vec(),
                "HELLO" => b"-ERR unknown command 'HELLO'\r\n".to_vec(),
                "SCAN" => {
                    let cursor = args.get(1).map(String::as_str).unwrap_or("0");
                    if cursor == "0" {
                        b"*2\r\n$2\r\n42\r\n*1\r\n$4\r\nmx:a\r\n".to_vec()
                    } else {
                        b"*2\r\n$1\r\n0\r\n*2\r\n$4\r\nmx:a\r\n$4\r\nmx:b\r\n".to_vec()
                    }
                }
                "HGETALL" => b"*2\r\n$1\r\nk\r\n$1\r\nv\r\n".to_vec(),
                "HGET" => b"$3\r\nval\r\n".to_vec(),
                _ => b"+OK\r\n".to_vec(),
            }
        })
        .await;

        let url = format!("redis://default:pa,ss@127.0.0.1:{port}/0");
        let mut conn = RedisEndpointConfig::Standalone(url)
            .connect()
            .await
            .expect("connect should succeed when INFO is NOPERM and credential has comma");
        assert!(!conn.is_cluster());
        assert_eq!(conn.get_db(), 0);

        let mut scanned = conn.scan_match("mx:*", 10).await.expect("scan_match");
        scanned.sort();
        assert_eq!(scanned, vec!["mx:a", "mx:b"]);

        let empty_all: Vec<Vec<(String, String)>> =
            conn.hgetall_many(&[]).await.expect("empty hgetall_many");
        assert!(empty_all.is_empty());
        let non_empty_all: Vec<Vec<(String, String)>> = conn
            .hgetall_many(&["k1".to_string()])
            .await
            .expect("hgetall_many");
        assert_eq!(
            non_empty_all,
            vec![vec![("k".to_string(), "v".to_string())]]
        );

        let empty_hget = conn.hget_many(&[], "f").await.expect("empty hget_many");
        assert!(empty_hget.is_empty());
        let non_empty_hget = conn
            .hget_many(&["k1".to_string()], "f")
            .await
            .expect("hget_many");
        assert_eq!(non_empty_hget, vec![Some("val".to_string())]);

        let _ = stop_tx.send(());
        let _ = handle.await;
    }

    #[tokio::test]
    async fn standalone_connect_rejects_cluster_info_auth_error_and_redacts_unix_query() {
        let (cluster_port, stop_a, h_a) = spawn_mock_resp(|args| {
            let cmd = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            if cmd == "INFO" {
                b"$19\r\ncluster_enabled:1\r\n\r\n".to_vec()
            } else if cmd == "PING" {
                b"+PONG\r\n".to_vec()
            } else {
                b"+OK\r\n".to_vec()
            }
        })
        .await;
        let cluster_err =
            RedisEndpointConfig::Standalone(format!("redis://127.0.0.1:{cluster_port}"))
                .connect()
                .await
                .err()
                .expect("standalone must reject cluster_enabled:1");
        assert!(cluster_err.contains("cluster_enabled:1"), "{cluster_err}");

        let (noauth_port, stop_b, h_b) = spawn_mock_resp(|args| {
            let cmd = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            if cmd == "INFO" {
                b"-NOAUTH Authentication required.\r\n".to_vec()
            } else {
                b"+OK\r\n".to_vec()
            }
        })
        .await;
        let auth_err = RedisEndpointConfig::Standalone(format!("redis://127.0.0.1:{noauth_port}"))
            .connect()
            .await
            .err()
            .expect("NOAUTH on INFO must fail");
        assert!(auth_err.contains("NOAUTH"), "{auth_err}");

        let multi_err = RedisEndpointConfig::Standalone(
            "redis://u:sec@n1:6379,redis://u:sec@n2:6379".to_string(),
        )
        .connect()
        .await
        .err()
        .expect("multi-url standalone must fail");
        assert!(multi_err.contains("<redacted>"), "{multi_err}");
        assert!(!multi_err.contains("sec"), "{multi_err}");

        let unix_invalid_err =
            RedisEndpointConfig::Standalone("unix://relative.sock?pass=topsecret".to_string())
                .connect()
                .await
                .err()
                .expect("relative unix url must fail");
        assert!(
            !unix_invalid_err.contains("topsecret"),
            "{unix_invalid_err}"
        );
        assert!(
            unix_invalid_err.contains("pass=<redacted>"),
            "{unix_invalid_err}"
        );

        #[cfg(unix)]
        {
            let temp_dir = tempfile::tempdir().expect("tempdir for unix socket");
            let sock_path = temp_dir.path().join("mx-test.sock");
            let listener = tokio::net::UnixListener::bind(&sock_path).expect("bind unix socket");
            let (stop_u, mut stop_u_rx) = oneshot::channel::<()>();
            let u_handle = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = &mut stop_u_rx => break,
                        accepted = listener.accept() => {
                            let Ok((stream, _)) = accepted else { break };
                            tokio::spawn(async move {
                                let (read_half, mut write_half) = stream.into_split();
                                let mut reader = BufReader::new(read_half);
                                while let Some(args) = read_resp_args(&mut reader).await {
                                    let cmd = args
                                        .first()
                                        .map(|s| s.to_ascii_uppercase())
                                        .unwrap_or_default();
                                    let reply: &[u8] = if cmd == "INFO" {
                                        b"-ERR simulated info error\r\n"
                                    } else if cmd == "PING" {
                                        b"-ERR simulated ping error\r\n"
                                    } else if cmd == "HELLO" {
                                        b"-ERR unknown command 'HELLO'\r\n"
                                    } else {
                                        b"+OK\r\n"
                                    };
                                    if write_half.write_all(reply).await.is_err() {
                                        break;
                                    }
                                }
                            });
                        }
                    }
                }
            });
            let unix_url = format!("unix://{}?pass=topsecret", sock_path.display());
            let unix_conn_err = RedisEndpointConfig::Standalone(unix_url)
                .connect()
                .await
                .err()
                .expect("simulated PING error on unix socket must fail");
            assert!(!unix_conn_err.contains("topsecret"), "{unix_conn_err}");
            assert!(unix_conn_err.contains("pass=<redacted>"), "{unix_conn_err}");
            let _ = stop_u.send(());
            let _ = u_handle.await;
        }

        let empty_cluster_err = RedisEndpointConfig::Cluster(Vec::new())
            .connect()
            .await
            .err()
            .expect("empty cluster seeds must fail");
        assert!(empty_cluster_err.contains("at least one seed URL"));

        let (standalone_port, stop_c, h_c) = spawn_mock_resp(|args| {
            let cmd = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            if cmd == "CLUSTER" {
                b"-ERR This instance has cluster support disabled\r\n".to_vec()
            } else if cmd == "PING" {
                b"+PONG\r\n".to_vec()
            } else {
                b"+OK\r\n".to_vec()
            }
        })
        .await;
        let mismatch_err =
            RedisEndpointConfig::Cluster(vec![format!("redis://127.0.0.1:{standalone_port}")])
                .connect()
                .await
                .err()
                .expect("cluster mode must reject standalone server");
        assert!(
            mismatch_err.contains("does not have cluster mode enabled"),
            "{mismatch_err}"
        );

        let _ = stop_a.send(());
        let _ = stop_b.send(());
        let _ = stop_c.send(());
        let _ = h_a.await;
        let _ = h_b.await;
        let _ = h_c.await;
    }

    #[tokio::test]
    async fn standalone_connect_permits_proxy_unsupported_info_and_rejects_wrongpass() {
        let (proxy_port, stop_p, h_p) = spawn_mock_resp(|args| {
            let cmd = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            match cmd.as_str() {
                "INFO" => b"-ERR unsupported command 'INFO'\r\n".to_vec(),
                "PING" => b"+PONG\r\n".to_vec(),
                "HELLO" => b"-ERR unknown command 'HELLO'\r\n".to_vec(),
                _ => b"+OK\r\n".to_vec(),
            }
        })
        .await;

        let conn = RedisEndpointConfig::Standalone(format!("redis://127.0.0.1:{proxy_port}"))
            .connect()
            .await
            .expect("proxy rejecting INFO must fall back to PING and connect");
        assert!(!conn.is_cluster());

        let (badpass_port, stop_w, h_w) = spawn_mock_resp(|args| {
            let cmd = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            if cmd == "INFO" {
                b"-WRONGPASS invalid username-password pair or user is disabled.\r\n".to_vec()
            } else {
                b"+OK\r\n".to_vec()
            }
        })
        .await;
        let wrongpass_err =
            RedisEndpointConfig::Standalone(format!("redis://127.0.0.1:{badpass_port}"))
                .connect()
                .await
                .err()
                .expect("WRONGPASS on INFO must still fail fast");
        assert!(wrongpass_err.contains("WRONGPASS"), "{wrongpass_err}");

        let _ = stop_p.send(());
        let _ = stop_w.send(());
        let _ = h_p.await;
        let _ = h_w.await;
    }

    #[test]
    fn cluster_client_settings_read_from_env() {
        use modelexpress_common::test_support::{EnvVarGuard, acquire_env_mutex};

        let lock = acquire_env_mutex();
        let _c = EnvVarGuard::set(&lock, envs::MX_REDIS_CONNECTION_TIMEOUT_MS, "2500");
        let _r = EnvVarGuard::set(&lock, envs::MX_REDIS_RESPONSE_TIMEOUT_MS, "8000");
        let _n = EnvVarGuard::set(&lock, envs::MX_REDIS_CLUSTER_RETRIES, "12");
        let _min = EnvVarGuard::set(&lock, envs::MX_REDIS_CLUSTER_MIN_RETRY_WAIT_MS, "200");
        let _max = EnvVarGuard::set(&lock, envs::MX_REDIS_CLUSTER_MAX_RETRY_WAIT_MS, "100");

        let settings = ClusterClientSettings::from_env();
        assert_eq!(settings.connection_timeout_ms, 2_500);
        assert_eq!(settings.response_timeout_ms, 8_000);
        assert_eq!(settings.retries, 12);
        assert_eq!(settings.min_retry_wait_ms, 200);
        assert_eq!(settings.max_retry_wait_ms, 200);
    }

    fn mock_slots_reply(masters: &[(&str, u16)]) -> Value {
        let count = u64::try_from(masters.len().max(1)).unwrap_or(1);
        let span = 16384u64.checked_div(count).unwrap_or(16384);
        let entries = masters
            .iter()
            .enumerate()
            .map(|(idx, (host, port))| {
                let idx_u64 = u64::try_from(idx).unwrap_or(0);
                let start = i64::try_from(idx_u64.saturating_mul(span)).unwrap_or(0);
                let end = if idx.saturating_add(1) == masters.len() {
                    16383
                } else {
                    i64::try_from(
                        idx_u64
                            .saturating_add(1)
                            .saturating_mul(span)
                            .saturating_sub(1),
                    )
                    .unwrap_or(0)
                };
                Value::Array(vec![
                    Value::Int(start),
                    Value::Int(end),
                    Value::Array(vec![
                        Value::BulkString(host.as_bytes().to_vec()),
                        Value::Int(i64::from(*port)),
                    ]),
                ])
            })
            .collect();
        Value::Array(entries)
    }

    fn mock_scan_reply(cursor: u64, keys: &[&str]) -> Value {
        Value::Array(vec![
            Value::BulkString(cursor.to_string().into_bytes()),
            Value::Array(
                keys.iter()
                    .map(|k| Value::BulkString(k.as_bytes().to_vec()))
                    .collect(),
            ),
        ])
    }

    type SlotsFn = Box<dyn Fn(usize) -> Result<Value, RedisError> + Send + Sync>;
    type ScanFn = Box<dyn Fn(&str, u16, &str) -> Result<Value, RedisError> + Send + Sync>;

    struct MockClusterRouter {
        slots_calls: std::sync::atomic::AtomicUsize,
        on_slots: SlotsFn,
        on_scan: ScanFn,
    }

    impl ClusterNodeRouter for MockClusterRouter {
        fn route<'a>(
            &'a mut self,
            cmd: &'a Cmd,
            routing: SingleNodeRoutingInfo,
        ) -> RedisFuture<'a, Value> {
            let args: Vec<String> = cmd
                .args_iter()
                .map(|arg| match arg {
                    redis::Arg::Simple(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                    redis::Arg::Cursor => "0".to_string(),
                })
                .collect();
            let op = args
                .first()
                .map(|s| s.to_ascii_uppercase())
                .unwrap_or_default();
            let res = if op == "CLUSTER" {
                let call = self
                    .slots_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                (self.on_slots)(call)
            } else {
                let (host, port) = match routing {
                    SingleNodeRoutingInfo::ByAddress { host, port } => (host, port),
                    _ => (String::new(), 0),
                };
                let cursor = args.get(1).map(String::as_str).unwrap_or("0");
                (self.on_scan)(&host, port, cursor)
            };
            Box::pin(async move { res })
        }
    }

    #[tokio::test]
    async fn cluster_scan_recovers_from_mid_scan_failover_and_topology_shift() {
        let seeds = vec![("127.0.0.1".to_string(), 7000)];

        // Case 1: primary 7001 fails mid-scan; CLUSTER SLOTS re-discovers promoted 7004.
        let mut failover_router = MockClusterRouter {
            slots_calls: std::sync::atomic::AtomicUsize::new(0),
            on_slots: Box::new(|call| {
                if call == 0 {
                    Ok(mock_slots_reply(&[
                        ("127.0.0.1", 7000),
                        ("127.0.0.1", 7001),
                    ]))
                } else {
                    Ok(mock_slots_reply(&[
                        ("127.0.0.1", 7000),
                        ("127.0.0.1", 7004),
                    ]))
                }
            }),
            on_scan: Box::new(|_host, port, _cursor| match port {
                7000 => Ok(mock_scan_reply(0, &["mx:k0"])),
                7001 => Err(RedisError::from((
                    ErrorKind::MasterDown,
                    "master is down during failover",
                ))),
                7004 => Ok(mock_scan_reply(0, &["mx:k1", "mx:k0"])),
                _ => Ok(mock_scan_reply(0, &[])),
            }),
        };
        let mut keys = scan_cluster_masters_with(&mut failover_router, &seeds, "mx:*", 100)
            .await
            .expect("mid-scan failover must recover via CLUSTER SLOTS refresh");
        keys.sort();
        assert_eq!(keys, vec!["mx:k0", "mx:k1"]);

        // Case 2: primary topology shifts between initial and post-scan CLUSTER SLOTS check.
        let mut shift_router = MockClusterRouter {
            slots_calls: std::sync::atomic::AtomicUsize::new(0),
            on_slots: Box::new(|call| {
                if call == 0 {
                    Ok(mock_slots_reply(&[
                        ("127.0.0.1", 7000),
                        ("127.0.0.1", 7001),
                    ]))
                } else {
                    Ok(mock_slots_reply(&[
                        ("127.0.0.1", 7000),
                        ("127.0.0.1", 7005),
                    ]))
                }
            }),
            on_scan: Box::new(|_host, port, _cursor| match port {
                7000 => Ok(mock_scan_reply(0, &["mx:a"])),
                7001 => Ok(mock_scan_reply(0, &[])),
                7005 => Ok(mock_scan_reply(0, &["mx:b"])),
                _ => Ok(mock_scan_reply(0, &[])),
            }),
        };
        let mut shifted_keys = scan_cluster_masters_with(&mut shift_router, &seeds, "mx:*", 100)
            .await
            .expect("topology shift mid-scan must retry and cover newly promoted master");
        shifted_keys.sort();
        assert_eq!(shifted_keys, vec!["mx:a", "mx:b"]);
    }
}
