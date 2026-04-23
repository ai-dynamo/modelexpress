// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis backend for the model registry.
//!
//! One Redis Hash per cached model at `mx:model:{model_name}`, with fields `provider`,
//! `status`, `created_at`, `last_used_at`, and optional `message`. LRU and status tallies
//! use on-demand `SCAN` + pipelined reads (no secondary indexes).
//!
//! [`CLAIM_LUA`] is the only Lua used: it combines the "claim if absent, else read status"
//! check + full field population into a single atomic EVAL so concurrent readers never
//! see a partially-written record.

use super::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use modelexpress_common::models::{ModelProvider, ModelStatus};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

const KEY_PREFIX: &str = "mx:model:";
const SCAN_PATTERN: &str = "mx:model:*";
const SCAN_BATCH: usize = 500;

/// Field names in the per-model hash.
mod fields {
    pub const STATUS: &str = "status";
    pub const PROVIDER: &str = "provider";
    pub const CREATED_AT: &str = "created_at";
    pub const LAST_USED_AT: &str = "last_used_at";
    pub const MESSAGE: &str = "message";
}

fn model_key(model_name: &str) -> String {
    format!("{KEY_PREFIX}{model_name}")
}

fn provider_str(p: ModelProvider) -> &'static str {
    match p {
        ModelProvider::HuggingFace => "HuggingFace",
        ModelProvider::Ngc => "Ngc",
    }
}

fn provider_from_str(s: &str) -> RegistryResult<ModelProvider> {
    match s {
        "HuggingFace" => Ok(ModelProvider::HuggingFace),
        "Ngc" => Ok(ModelProvider::Ngc),
        other => Err(format!("unknown provider in Redis record: {other:?}").into()),
    }
}

fn status_str(s: ModelStatus) -> &'static str {
    match s {
        ModelStatus::DOWNLOADING => "DOWNLOADING",
        ModelStatus::DOWNLOADED => "DOWNLOADED",
        ModelStatus::ERROR => "ERROR",
    }
}

fn status_from_str(s: &str) -> RegistryResult<ModelStatus> {
    match s {
        "DOWNLOADING" => Ok(ModelStatus::DOWNLOADING),
        "DOWNLOADED" => Ok(ModelStatus::DOWNLOADED),
        "ERROR" => Ok(ModelStatus::ERROR),
        other => Err(format!("unknown status in Redis record: {other:?}").into()),
    }
}

fn parse_rfc3339(s: &str, field: &str) -> RegistryResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| format!("invalid RFC3339 in field '{field}' ({s:?}): {e}").into())
}

/// Redaction helper for logging: strip userinfo (password, and user if present) from a
/// redis:// URL so secrets don't leak into logs.
fn redact_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let head_end = scheme_end.saturating_add(3);
    let (head, rest) = url.split_at(head_end); // head = "redis://"
    let Some(at_pos) = rest.find('@') else {
        return url.to_string(); // no userinfo
    };
    let (userinfo, tail) = rest.split_at(at_pos); // tail starts with '@'
    match userinfo.split_once(':') {
        Some((user, _pw)) => format!("{head}{user}:***{tail}"),
        None => format!("{head}***{tail}"), // user only, no password
    }
}

pub struct RedisRegistryBackend {
    redis: Arc<RwLock<Option<ConnectionManager>>>,
    redis_url: String,
}

impl RedisRegistryBackend {
    pub fn new(redis_url: &str) -> Self {
        Self {
            redis: Arc::new(RwLock::new(None)),
            redis_url: redis_url.to_string(),
        }
    }

    async fn get_conn(&self) -> RegistryResult<ConnectionManager> {
        {
            let guard = self.redis.read().await;
            if let Some(conn) = guard.as_ref() {
                return Ok(conn.clone());
            }
        }
        let mut guard = self.redis.write().await;
        if let Some(conn) = guard.as_ref() {
            return Ok(conn.clone());
        }
        let client = redis::Client::open(self.redis_url.as_str())?;
        let conn = ConnectionManager::new(client).await?;
        *guard = Some(conn.clone());
        Ok(conn)
    }

    /// Collect every `mx:model:*` key with a paged SCAN, deduplicating as we go.
    ///
    /// Redis `SCAN` can legitimately return the same key twice across cursor pages (for
    /// example when the hash table is resized mid-iteration), so callers that use the
    /// result length as a count would double-count. The dedup set keeps this honest.
    async fn scan_all_keys(&self, conn: &mut ConnectionManager) -> RegistryResult<Vec<String>> {
        use std::collections::HashSet;
        let mut cursor: u64 = 0;
        let mut keys: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        loop {
            let (next, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(SCAN_PATTERN)
                .arg("COUNT")
                .arg(SCAN_BATCH)
                .query_async(conn)
                .await?;
            for k in batch {
                if seen.insert(k.clone()) {
                    keys.push(k);
                }
            }
            if next == 0 {
                break;
            }
            cursor = next;
        }
        Ok(keys)
    }

    fn model_name_from_key(key: &str) -> Option<&str> {
        key.strip_prefix(KEY_PREFIX)
    }

    fn record_from_hash(
        model_name: &str,
        pairs: Vec<(String, String)>,
    ) -> RegistryResult<ModelRecord> {
        let mut map: std::collections::HashMap<String, String> = pairs.into_iter().collect();
        let take = |map: &mut std::collections::HashMap<String, String>, key: &str| {
            map.remove(key)
                .ok_or_else(|| format!("missing field '{key}' for {model_name}"))
        };
        Ok(ModelRecord {
            model_name: model_name.to_string(),
            provider: provider_from_str(&take(&mut map, fields::PROVIDER)?)?,
            status: status_from_str(&take(&mut map, fields::STATUS)?)?,
            created_at: parse_rfc3339(&take(&mut map, fields::CREATED_AT)?, fields::CREATED_AT)?,
            last_used_at: parse_rfc3339(
                &take(&mut map, fields::LAST_USED_AT)?,
                fields::LAST_USED_AT,
            )?,
            message: map.remove(fields::MESSAGE),
        })
    }
}

#[async_trait]
impl RegistryBackend for RedisRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        let client = redis::Client::open(self.redis_url.as_str())?;
        let conn = ConnectionManager::new(client).await?;
        let mut guard = self.redis.write().await;
        *guard = Some(conn);
        info!(
            "Registry: connected to Redis at {}",
            redact_url(&self.redis_url)
        );
        Ok(())
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        let mut conn = self.get_conn().await?;
        let value: Option<String> = conn.hget(model_key(model_name), fields::STATUS).await?;
        match value {
            Some(s) => Ok(Some(status_from_str(&s)?)),
            None => Ok(None),
        }
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        let mut conn = self.get_conn().await?;
        let pairs: Vec<(String, String)> = conn.hgetall(model_key(model_name)).await?;
        if pairs.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self::record_from_hash(model_name, pairs)?))
    }

    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        let mut conn = self.get_conn().await?;
        let now = Utc::now().to_rfc3339();
        let key = model_key(model_name);
        // A single EVAL so the status/provider/last_used_at/message/created_at updates
        // are atomic: concurrent get_model_record calls see either the pre- or post-
        // update record, never a half-written one (HGETALL never interleaves with the
        // script since Redis is single-threaded for scripts).
        let (msg_flag, msg_value) = match &message {
            Some(m) => ("1", m.as_str()),
            None => ("0", ""),
        };
        let _: () = redis::Script::new(SET_STATUS_LUA)
            .key(&key)
            .arg(status_str(status))
            .arg(provider_str(provider))
            .arg(&now)
            .arg(&now)
            .arg(msg_flag)
            .arg(msg_value)
            .invoke_async(&mut conn)
            .await?;
        Ok(())
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        let mut conn = self.get_conn().await?;
        let now = Utc::now().to_rfc3339();
        // HSET on a missing key would create a malformed record (last_used_at only).
        // Gate on EXISTS so touch is purely an update, never a create.
        let key = model_key(model_name);
        let exists: bool = conn.exists(&key).await?;
        if !exists {
            return Ok(());
        }
        let _: () = conn.hset(&key, fields::LAST_USED_AT, now).await?;
        Ok(())
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        let mut conn = self.get_conn().await?;
        let _: () = conn.del(model_key(model_name)).await?;
        Ok(())
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        let mut conn = self.get_conn().await?;
        let keys = self.scan_all_keys(&mut conn).await?;
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        // Pipeline the HGETALLs so we pay one network round-trip for all models.
        let mut pipe = redis::pipe();
        for k in &keys {
            pipe.hgetall(k);
        }
        let hashes: Vec<Vec<(String, String)>> = pipe.query_async(&mut conn).await?;
        let mut records: Vec<ModelRecord> = Vec::with_capacity(keys.len());
        for (key, pairs) in keys.iter().zip(hashes.into_iter()) {
            if pairs.is_empty() {
                // Deleted between SCAN and HGETALL; skip defensively.
                continue;
            }
            let Some(name) = Self::model_name_from_key(key) else {
                continue;
            };
            match Self::record_from_hash(name, pairs) {
                Ok(r) => records.push(r),
                Err(e) => tracing::warn!("Skipping malformed registry record at {}: {}", key, e),
            }
        }
        records.sort_by_key(|r| r.last_used_at);
        if let Some(n) = limit {
            records.truncate(n as usize);
        }
        Ok(records)
    }

    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        let mut conn = self.get_conn().await?;
        let keys = self.scan_all_keys(&mut conn).await?;
        if keys.is_empty() {
            return Ok((0, 0, 0));
        }
        let mut pipe = redis::pipe();
        for k in &keys {
            pipe.hget(k, fields::STATUS);
        }
        let statuses: Vec<Option<String>> = pipe.query_async(&mut conn).await?;
        let mut downloading = 0u32;
        let mut downloaded = 0u32;
        let mut error = 0u32;
        for s in statuses.into_iter().flatten() {
            match s.as_str() {
                "DOWNLOADING" => downloading = downloading.saturating_add(1),
                "DOWNLOADED" => downloaded = downloaded.saturating_add(1),
                "ERROR" => error = error.saturating_add(1),
                _ => {}
            }
        }
        Ok((downloading, downloaded, error))
    }

    async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<ClaimOutcome> {
        let mut conn = self.get_conn().await?;
        let key = model_key(model_name);
        let now = Utc::now().to_rfc3339();
        // Atomic claim + populate: a single EVAL so readers never see a partially-written
        // record. The script returns a sentinel string — `CLAIM_WON_SENTINEL` if we
        // created the record, or the existing status string if someone else already
        // claimed it. This distinction is what lets callers know which replica owns
        // the download (status alone can't — both claimer and observer see DOWNLOADING).
        let result: String = redis::Script::new(CLAIM_LUA)
            .key(&key)
            .arg(CLAIM_WON_SENTINEL)
            .arg(status_str(ModelStatus::DOWNLOADING))
            .arg(provider_str(provider))
            .arg(&now)
            .arg("Starting download...")
            .invoke_async(&mut conn)
            .await?;
        if result == CLAIM_WON_SENTINEL {
            Ok(ClaimOutcome::Claimed)
        } else {
            Ok(ClaimOutcome::AlreadyExists(status_from_str(&result)?))
        }
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<bool> {
        let mut conn = self.get_conn().await?;
        let key = model_key(model_name);
        let now = Utc::now().to_rfc3339();
        // Atomic CAS: flip status from ERROR to DOWNLOADING. Returns 1 on win, 0 on
        // miss. Only the winner spawns the retry download.
        let won: i32 = redis::Script::new(RETRY_CAS_LUA)
            .key(&key)
            .arg(status_str(ModelStatus::ERROR))
            .arg(status_str(ModelStatus::DOWNLOADING))
            .arg(provider_str(provider))
            .arg(&now)
            .arg("Retrying download...")
            .invoke_async(&mut conn)
            .await?;
        Ok(won == 1)
    }
}

/// Sentinel string returned by [`CLAIM_LUA`] when the caller won the claim. Picked so
/// it cannot be confused with any real `ModelStatus` string.
const CLAIM_WON_SENTINEL: &str = "__MX_CLAIM_WON__";

/// Atomic claim: if the hash has no `status` field, populate all fields in one shot and
/// return the win sentinel; otherwise return the existing `status` unchanged.
///
/// KEYS[1] = model key, ARGV = [win_sentinel, status, provider, now, message]
const CLAIM_LUA: &str = r#"
local existing = redis.call("HGET", KEYS[1], "status")
if existing then return existing end
redis.call("HSET", KEYS[1],
    "status", ARGV[2],
    "provider", ARGV[3],
    "created_at", ARGV[4],
    "last_used_at", ARGV[4],
    "message", ARGV[5])
return ARGV[1]
"#;

/// Atomic CAS: flip status from `from_status` to `to_status` (also refreshing
/// provider, last_used_at, message). Returns 1 on win, 0 on miss.
///
/// KEYS[1] = model key
/// ARGV    = [from_status, to_status, provider, last_used_at, message]
const RETRY_CAS_LUA: &str = r#"
local cur = redis.call("HGET", KEYS[1], "status")
if cur ~= ARGV[1] then return 0 end
redis.call("HSET", KEYS[1],
    "status", ARGV[2],
    "provider", ARGV[3],
    "last_used_at", ARGV[4],
    "message", ARGV[5])
return 1
"#;

/// Atomic `set_status`: overwrite status/provider/last_used_at, either set or clear
/// `message`, and only stamp `created_at` if it isn't already there (preserves the
/// original timestamp across status transitions).
///
/// KEYS[1] = model key
/// ARGV    = [status, provider, last_used_at, created_at_if_new, msg_flag, msg_value]
///    msg_flag = "1" → HSET message = msg_value
///    msg_flag = "0" → HDEL message (msg_value ignored)
const SET_STATUS_LUA: &str = r#"
redis.call("HSET", KEYS[1],
    "status", ARGV[1],
    "provider", ARGV[2],
    "last_used_at", ARGV[3])
if ARGV[5] == "1" then
    redis.call("HSET", KEYS[1], "message", ARGV[6])
else
    redis.call("HDEL", KEYS[1], "message")
end
redis.call("HSETNX", KEYS[1], "created_at", ARGV[4])
return 1
"#;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn provider_roundtrip() {
        for p in [ModelProvider::HuggingFace, ModelProvider::Ngc] {
            let s = provider_str(p);
            assert_eq!(provider_from_str(s).expect("roundtrip"), p);
        }
        assert!(provider_from_str("bogus").is_err());
    }

    #[test]
    fn status_roundtrip() {
        for s in [
            ModelStatus::DOWNLOADING,
            ModelStatus::DOWNLOADED,
            ModelStatus::ERROR,
        ] {
            assert_eq!(status_from_str(status_str(s)).expect("roundtrip"), s);
        }
        assert!(status_from_str("UNKNOWN").is_err());
    }

    #[test]
    fn model_key_and_parse() {
        let k = model_key("meta-llama/Llama-3.1-70B");
        assert_eq!(k, "mx:model:meta-llama/Llama-3.1-70B");
        assert_eq!(
            RedisRegistryBackend::model_name_from_key(&k),
            Some("meta-llama/Llama-3.1-70B")
        );
    }

    #[test]
    fn redact_url_strips_userinfo() {
        assert_eq!(
            redact_url("redis://user:secret@host:6379"),
            "redis://user:***@host:6379"
        );
        assert_eq!(redact_url("redis://host:6379"), "redis://host:6379");
        // User-only (no password): redact user too.
        assert_eq!(
            redact_url("redis://user@host:6379"),
            "redis://***@host:6379"
        );
        // Non-redis URL or malformed: pass through.
        assert_eq!(redact_url("not-a-url"), "not-a-url");
    }

    #[test]
    fn record_from_hash_builds_full_record() {
        let fields = vec![
            ("provider".to_string(), "HuggingFace".to_string()),
            ("status".to_string(), "DOWNLOADED".to_string()),
            (
                "created_at".to_string(),
                "2026-04-22T10:00:00+00:00".to_string(),
            ),
            (
                "last_used_at".to_string(),
                "2026-04-22T11:00:00+00:00".to_string(),
            ),
            ("message".to_string(), "ok".to_string()),
        ];
        let rec = RedisRegistryBackend::record_from_hash("foo/bar", fields).expect("parse");
        assert_eq!(rec.model_name, "foo/bar");
        assert_eq!(rec.provider, ModelProvider::HuggingFace);
        assert_eq!(rec.status, ModelStatus::DOWNLOADED);
        assert_eq!(rec.message.as_deref(), Some("ok"));
    }

    #[test]
    fn record_from_hash_rejects_missing_fields() {
        let fields = vec![("status".to_string(), "DOWNLOADED".to_string())];
        assert!(RedisRegistryBackend::record_from_hash("foo/bar", fields).is_err());
    }
}
