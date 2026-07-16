// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis-backed atomic storage for M2N bootstrap attempts.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::m2n_bootstrap::M2nBootstrapState;
use redis::aio::ConnectionManager;
use tokio::sync::RwLock;
use tracing::info;

use super::{
    BootstrapError, BootstrapKey, BootstrapRecord, BootstrapResult, M2nBootstrapBackend,
    PublishOutcome, TOMBSTONE_RETENTION_MS,
};

const PUBLISHED: i32 = M2nBootstrapState::Published as i32;
const ABORTED: i32 = M2nBootstrapState::Aborted as i32;
const EXPIRED: i32 = M2nBootstrapState::Expired as i32;

const FIELD_JOB_ID: &str = "job_id";
const FIELD_ATTEMPT_ID: &str = "attempt_id";
const FIELD_COHORT_ID: &str = "cohort_id";
const FIELD_UID: &str = "nccl_unique_id";
const FIELD_SOURCE_SIZE: &str = "source_world_size";
const FIELD_DESTINATION_SIZE: &str = "destination_world_size";
const FIELD_WORLD_SIZE: &str = "world_size";
const FIELD_ROSTER_DIGEST: &str = "roster_digest";
const FIELD_CONFIG_DIGEST: &str = "config_digest";
const FIELD_PUBLISHER: &str = "publisher_participant_id";
const FIELD_STATE: &str = "state";
const FIELD_EXPIRES_AT: &str = "expires_at_ms";
const FIELD_REASON: &str = "reason";
const FIELD_REVISION: &str = "revision";
const FIELD_FINGERPRINT: &str = "publication_fingerprint";
type RedisFieldPairs = Vec<(String, Vec<u8>)>;

pub struct RedisM2nBootstrapBackend {
    redis: Arc<RwLock<Option<ConnectionManager>>>,
    redis_url: String,
}

impl RedisM2nBootstrapBackend {
    #[must_use]
    pub fn new(redis_url: &str) -> Self {
        Self {
            redis: Arc::new(RwLock::new(None)),
            redis_url: redis_url.to_string(),
        }
    }

    async fn get_conn(&self) -> BootstrapResult<ConnectionManager> {
        if let Some(conn) = self.redis.read().await.as_ref() {
            return Ok(conn.clone());
        }

        let mut guard = self.redis.write().await;
        if let Some(conn) = guard.as_ref() {
            return Ok(conn.clone());
        }
        let client = redis::Client::open(self.redis_url.as_str())
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        let conn = ConnectionManager::new(client)
            .await
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        *guard = Some(conn.clone());
        Ok(conn)
    }

    fn redis_record_key(key: &BootstrapKey) -> String {
        format!("mx:m2n:{{{}}}:record", key.attempt_id)
    }

    fn redis_uid_key(key: &BootstrapKey) -> String {
        format!("mx:m2n:{{{}}}:uid", key.attempt_id)
    }

    async fn fetch_record(
        &self,
        record_key: &str,
        uid_key: &str,
    ) -> BootstrapResult<Option<BootstrapRecord>> {
        let mut conn = self.get_conn().await?;
        let (pairs, uid): (RedisFieldPairs, Option<Vec<u8>>) = redis::pipe()
            .atomic()
            .hgetall(record_key)
            .get(uid_key)
            .query_async(&mut conn)
            .await
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        if pairs.is_empty() {
            return Ok(None);
        }
        let mut fields: HashMap<String, Vec<u8>> = pairs.into_iter().collect();
        fields.insert(FIELD_UID.to_string(), uid.unwrap_or_default());
        parse_record(fields).map(Some)
    }
}

#[async_trait]
impl M2nBootstrapBackend for RedisM2nBootstrapBackend {
    async fn connect(&self) -> BootstrapResult<()> {
        let client = redis::Client::open(self.redis_url.as_str())
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        let conn = ConnectionManager::new(client)
            .await
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        *self.redis.write().await = Some(conn);
        info!("Connected M2N bootstrap state to Redis");
        Ok(())
    }

    async fn publish(
        &self,
        record: BootstrapRecord,
        now_ms: i64,
    ) -> BootstrapResult<PublishOutcome> {
        let record_key = Self::redis_record_key(&record.key);
        let uid_key = Self::redis_uid_key(&record.key);
        let uid_ttl_ms = uid_ttl_ms(record.expires_at_ms, now_ms)?;
        let retention_ms = retention_ms(record.expires_at_ms, now_ms)?;
        let mut conn = self.get_conn().await?;
        let code: i32 = redis::Script::new(PUBLISH_LUA)
            .key(&record_key)
            .key(&uid_key)
            .arg(now_ms)
            .arg(&record.publication_fingerprint)
            .arg(&record.key.job_id)
            .arg(&record.key.attempt_id)
            .arg(&record.key.cohort_id)
            .arg(&record.nccl_unique_id)
            .arg(record.source_world_size)
            .arg(record.destination_world_size)
            .arg(record.world_size)
            .arg(&record.roster_digest)
            .arg(&record.config_digest)
            .arg(&record.publisher_participant_id)
            .arg(PUBLISHED)
            .arg(record.expires_at_ms)
            .arg(record.revision)
            .arg(uid_ttl_ms)
            .arg(retention_ms)
            .arg(EXPIRED)
            .arg("bootstrap attempt expired")
            .invoke_async(&mut conn)
            .await
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;

        if code == 3 {
            return Err(BootstrapError::Conflict(format!(
                "attempt '{}' already exists with different or terminal state",
                record.key.attempt_id
            )));
        }
        let stored = self.get(&record.key, now_ms).await?.ok_or_else(|| {
            BootstrapError::Backend("bootstrap record disappeared after publication".to_string())
        })?;
        ensure_key(&record.key, &stored.key)?;

        Ok(PublishOutcome {
            record: stored,
            created: code == 1,
        })
    }

    async fn get(
        &self,
        key: &BootstrapKey,
        now_ms: i64,
    ) -> BootstrapResult<Option<BootstrapRecord>> {
        let record_key = Self::redis_record_key(key);
        let uid_key = Self::redis_uid_key(key);
        let mut conn = self.get_conn().await?;
        let exists: i32 = redis::Script::new(GET_LUA)
            .key(&record_key)
            .key(&uid_key)
            .arg(now_ms)
            .arg(PUBLISHED)
            .arg(EXPIRED)
            .arg("bootstrap attempt expired")
            .arg(TOMBSTONE_RETENTION_MS)
            .invoke_async(&mut conn)
            .await
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        if exists == 0 {
            return Ok(None);
        }
        let record = self.fetch_record(&record_key, &uid_key).await?;
        if let Some(record) = &record {
            ensure_key(key, &record.key)?;
        }
        Ok(record)
    }

    async fn abort(
        &self,
        key: &BootstrapKey,
        requested_by: &str,
        reason: &str,
        now_ms: i64,
    ) -> BootstrapResult<BootstrapRecord> {
        let record_key = Self::redis_record_key(key);
        let uid_key = Self::redis_uid_key(key);
        let mut conn = self.get_conn().await?;
        let code: i32 = redis::Script::new(ABORT_LUA)
            .key(&record_key)
            .key(&uid_key)
            .arg(&key.job_id)
            .arg(&key.attempt_id)
            .arg(&key.cohort_id)
            .arg(requested_by)
            .arg(reason)
            .arg(now_ms)
            .arg(PUBLISHED)
            .arg(ABORTED)
            .arg(TOMBSTONE_RETENTION_MS)
            .arg(EXPIRED)
            .arg("bootstrap attempt expired")
            .invoke_async(&mut conn)
            .await
            .map_err(|error| BootstrapError::Backend(error.to_string()))?;
        if code == -1 {
            return Err(BootstrapError::Conflict(format!(
                "attempt_id '{}' is already bound to another job or cohort",
                key.attempt_id
            )));
        }
        let record = self
            .fetch_record(&record_key, &uid_key)
            .await?
            .ok_or_else(|| {
                BootstrapError::Backend("bootstrap tombstone disappeared after abort".to_string())
            })?;
        ensure_key(key, &record.key)?;
        Ok(record)
    }
}

fn uid_ttl_ms(expires_at_ms: i64, now_ms: i64) -> BootstrapResult<u64> {
    let remaining = expires_at_ms.saturating_sub(now_ms).max(0);
    let remaining = u64::try_from(remaining).map_err(|_| {
        BootstrapError::Backend("bootstrap UID TTL cannot be represented".to_string())
    })?;
    if remaining == 0 {
        return Err(BootstrapError::Backend(
            "bootstrap UID TTL expired before publication".to_string(),
        ));
    }
    Ok(remaining)
}

fn retention_ms(expires_at_ms: i64, now_ms: i64) -> BootstrapResult<u64> {
    Ok(uid_ttl_ms(expires_at_ms, now_ms)?.saturating_add(TOMBSTONE_RETENTION_MS))
}

fn ensure_key(expected: &BootstrapKey, actual: &BootstrapKey) -> BootstrapResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(BootstrapError::Conflict(format!(
            "attempt_id '{}' is already bound to another job or cohort",
            expected.attempt_id
        )))
    }
}

fn parse_record(mut fields: HashMap<String, Vec<u8>>) -> BootstrapResult<BootstrapRecord> {
    let key = BootstrapKey {
        job_id: take_text(&mut fields, FIELD_JOB_ID)?,
        attempt_id: take_text(&mut fields, FIELD_ATTEMPT_ID)?,
        cohort_id: take_text(&mut fields, FIELD_COHORT_ID)?,
    };
    let state_value = take_number::<i32>(&mut fields, FIELD_STATE)?;
    let state = M2nBootstrapState::try_from(state_value).map_err(|_| {
        BootstrapError::InvalidStoredRecord(format!("invalid bootstrap state {state_value}"))
    })?;
    Ok(BootstrapRecord {
        key,
        nccl_unique_id: take_bytes(&mut fields, FIELD_UID)?,
        source_world_size: take_number(&mut fields, FIELD_SOURCE_SIZE)?,
        destination_world_size: take_number(&mut fields, FIELD_DESTINATION_SIZE)?,
        world_size: take_number(&mut fields, FIELD_WORLD_SIZE)?,
        roster_digest: take_bytes(&mut fields, FIELD_ROSTER_DIGEST)?,
        config_digest: take_bytes(&mut fields, FIELD_CONFIG_DIGEST)?,
        publisher_participant_id: take_text(&mut fields, FIELD_PUBLISHER)?,
        state,
        expires_at_ms: take_number(&mut fields, FIELD_EXPIRES_AT)?,
        reason: take_text(&mut fields, FIELD_REASON)?,
        revision: take_number(&mut fields, FIELD_REVISION)?,
        publication_fingerprint: take_text(&mut fields, FIELD_FINGERPRINT)?,
    })
}

fn take_bytes(fields: &mut HashMap<String, Vec<u8>>, name: &str) -> BootstrapResult<Vec<u8>> {
    fields
        .remove(name)
        .ok_or_else(|| BootstrapError::InvalidStoredRecord(format!("missing Redis field '{name}'")))
}

fn take_text(fields: &mut HashMap<String, Vec<u8>>, name: &str) -> BootstrapResult<String> {
    String::from_utf8(take_bytes(fields, name)?).map_err(|error| {
        BootstrapError::InvalidStoredRecord(format!(
            "invalid UTF-8 in Redis field '{name}': {error}"
        ))
    })
}

fn take_number<T>(fields: &mut HashMap<String, Vec<u8>>, name: &str) -> BootstrapResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    take_text(fields, name)?.parse().map_err(|error| {
        BootstrapError::InvalidStoredRecord(format!(
            "invalid numeric Redis field '{name}': {error}"
        ))
    })
}

const PUBLISH_LUA: &str = r#"
if redis.call("EXISTS", KEYS[1]) == 1 then
  local state = tonumber(redis.call("HGET", KEYS[1], "state"))
  local expires_at = tonumber(redis.call("HGET", KEYS[1], "expires_at_ms"))
  if state == tonumber(ARGV[13]) and tonumber(ARGV[1]) >= expires_at then
    local revision = tonumber(redis.call("HGET", KEYS[1], "revision")) + 1
    redis.call("DEL", KEYS[2])
    redis.call("HSET", KEYS[1],
      "state", ARGV[18],
      "reason", ARGV[19],
      "revision", revision)
    redis.call("PEXPIRE", KEYS[1], ARGV[17])
    return 3
  end
  local fingerprint = redis.call("HGET", KEYS[1], "publication_fingerprint")
  if state == tonumber(ARGV[13]) and fingerprint == ARGV[2]
      and redis.call("EXISTS", KEYS[2]) == 1 then
    return 2
  end
  return 3
end

redis.call("HSET", KEYS[1],
  "job_id", ARGV[3],
  "attempt_id", ARGV[4],
  "cohort_id", ARGV[5],
  "source_world_size", ARGV[7],
  "destination_world_size", ARGV[8],
  "world_size", ARGV[9],
  "roster_digest", ARGV[10],
  "config_digest", ARGV[11],
  "publisher_participant_id", ARGV[12],
  "state", ARGV[13],
  "expires_at_ms", ARGV[14],
  "reason", "",
  "revision", ARGV[15],
  "publication_fingerprint", ARGV[2])
redis.call("SET", KEYS[2], ARGV[6], "PX", ARGV[16])
redis.call("PEXPIRE", KEYS[1], ARGV[17])
return 1
"#;

const GET_LUA: &str = r#"
if redis.call("EXISTS", KEYS[1]) == 0 then return 0 end
local state = tonumber(redis.call("HGET", KEYS[1], "state"))
local expires_at = tonumber(redis.call("HGET", KEYS[1], "expires_at_ms"))
local uid_missing = redis.call("EXISTS", KEYS[2]) == 0
if state == tonumber(ARGV[2])
    and (tonumber(ARGV[1]) >= expires_at or uid_missing) then
  local revision = tonumber(redis.call("HGET", KEYS[1], "revision")) + 1
  redis.call("DEL", KEYS[2])
  redis.call("HSET", KEYS[1],
    "state", ARGV[3],
    "reason", ARGV[4],
    "revision", revision)
  redis.call("PEXPIRE", KEYS[1], ARGV[5])
end
return 1
"#;

const ABORT_LUA: &str = r#"
if redis.call("EXISTS", KEYS[1]) == 1 then
  if redis.call("HGET", KEYS[1], "job_id") ~= ARGV[1]
      or redis.call("HGET", KEYS[1], "attempt_id") ~= ARGV[2]
      or redis.call("HGET", KEYS[1], "cohort_id") ~= ARGV[3] then
    return -1
  end
  local state = tonumber(redis.call("HGET", KEYS[1], "state"))
  local expires_at = tonumber(redis.call("HGET", KEYS[1], "expires_at_ms"))
  if state == tonumber(ARGV[7]) and tonumber(ARGV[6]) >= expires_at then
    local revision = tonumber(redis.call("HGET", KEYS[1], "revision")) + 1
    redis.call("DEL", KEYS[2])
    redis.call("HSET", KEYS[1],
      "state", ARGV[10],
      "reason", ARGV[11],
      "revision", revision)
    redis.call("PEXPIRE", KEYS[1], ARGV[9])
    return 1
  end
  redis.call("DEL", KEYS[2])
  if state == tonumber(ARGV[7]) then
    local revision = tonumber(redis.call("HGET", KEYS[1], "revision")) + 1
    redis.call("HSET", KEYS[1],
      "state", ARGV[8],
      "reason", ARGV[5],
      "revision", revision)
  end
  redis.call("PEXPIRE", KEYS[1], ARGV[9])
  return 1
end

redis.call("DEL", KEYS[2])
redis.call("HSET", KEYS[1],
  "job_id", ARGV[1],
  "attempt_id", ARGV[2],
  "cohort_id", ARGV[3],
  "source_world_size", 0,
  "destination_world_size", 0,
  "world_size", 0,
  "roster_digest", "",
  "config_digest", "",
  "publisher_participant_id", ARGV[4],
  "state", ARGV[8],
  "expires_at_ms", ARGV[6],
  "reason", ARGV[5],
  "revision", 1,
  "publication_fingerprint", "")
redis.call("PEXPIRE", KEYS[1], ARGV[9])
return 1
"#;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use redis::AsyncCommands;

    #[test]
    fn parses_binary_record_fields() {
        let mut fields = HashMap::new();
        fields.insert(FIELD_JOB_ID.to_string(), b"job".to_vec());
        fields.insert(
            FIELD_ATTEMPT_ID.to_string(),
            b"123e4567-e89b-42d3-a456-426614174000".to_vec(),
        );
        fields.insert(FIELD_COHORT_ID.to_string(), b"cohort".to_vec());
        fields.insert(FIELD_UID.to_string(), vec![0, 255, 1]);
        fields.insert(FIELD_SOURCE_SIZE.to_string(), b"2".to_vec());
        fields.insert(FIELD_DESTINATION_SIZE.to_string(), b"4".to_vec());
        fields.insert(FIELD_WORLD_SIZE.to_string(), b"6".to_vec());
        fields.insert(FIELD_ROSTER_DIGEST.to_string(), vec![1; 32]);
        fields.insert(FIELD_CONFIG_DIGEST.to_string(), vec![2; 32]);
        fields.insert(FIELD_PUBLISHER.to_string(), b"source-0".to_vec());
        fields.insert(FIELD_STATE.to_string(), PUBLISHED.to_string().into_bytes());
        fields.insert(FIELD_EXPIRES_AT.to_string(), b"1000".to_vec());
        fields.insert(FIELD_REASON.to_string(), Vec::new());
        fields.insert(FIELD_REVISION.to_string(), b"1".to_vec());
        fields.insert(FIELD_FINGERPRINT.to_string(), b"fingerprint".to_vec());

        let record = parse_record(fields).expect("parse");
        assert_eq!(record.nccl_unique_id, vec![0, 255, 1]);
        assert_eq!(record.state, M2nBootstrapState::Published);
    }

    fn published_record(key: BootstrapKey, expires_at_ms: i64) -> BootstrapRecord {
        let mut record = BootstrapRecord {
            key,
            nccl_unique_id: vec![7; 128],
            source_world_size: 2,
            destination_world_size: 2,
            world_size: 4,
            roster_digest: vec![1; 32],
            config_digest: vec![2; 32],
            publisher_participant_id: "source-0".to_string(),
            state: M2nBootstrapState::Published,
            expires_at_ms,
            reason: String::new(),
            revision: 1,
            publication_fingerprint: String::new(),
        };
        record.publication_fingerprint = super::super::publication_fingerprint(&record);
        record
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn real_redis_publish_is_atomic_and_expiry_redacts_uid() {
        let Ok(redis_url) = std::env::var("MX_TEST_REDIS_URL") else {
            eprintln!("skipping real Redis test: MX_TEST_REDIS_URL is unset");
            return;
        };
        let backend = Arc::new(RedisM2nBootstrapBackend::new(&redis_url));
        backend
            .connect()
            .await
            .expect("connect to MX_TEST_REDIS_URL");

        let now_ms = chrono::Utc::now().timestamp_millis();
        let key = BootstrapKey {
            job_id: "m2n-bootstrap-test".to_string(),
            attempt_id: uuid::Uuid::new_v4().hyphenated().to_string(),
            cohort_id: "concurrency".to_string(),
        };
        let record = published_record(key.clone(), now_ms + 60_000);
        let mut tasks = Vec::new();
        for _ in 0..16 {
            let backend = Arc::clone(&backend);
            let record = record.clone();
            tasks.push(tokio::spawn(async move {
                backend.publish(record, now_ms).await
            }));
        }

        let mut created = 0;
        for task in tasks {
            let outcome = task.await.expect("publish task").expect("publish");
            if outcome.created {
                created += 1;
            }
        }
        assert_eq!(created, 1);

        let published = backend
            .get(&key, now_ms)
            .await
            .expect("get published")
            .expect("published record");
        assert_eq!(published.nccl_unique_id, vec![7; 128]);

        let expired = backend
            .get(&key, now_ms + 60_000)
            .await
            .expect("get expired")
            .expect("expired record");
        assert_eq!(expired.state, M2nBootstrapState::Expired);
        assert!(expired.nccl_unique_id.is_empty());

        let abort_key = BootstrapKey {
            job_id: "m2n-bootstrap-test".to_string(),
            attempt_id: uuid::Uuid::new_v4().hyphenated().to_string(),
            cohort_id: "abort-after-deadline".to_string(),
        };
        backend
            .publish(published_record(abort_key.clone(), now_ms + 60_000), now_ms)
            .await
            .expect("publish abort candidate");
        let expired_by_abort = backend
            .abort(&abort_key, "coordinator", "late abort", now_ms + 60_000)
            .await
            .expect("abort after deadline");
        assert_eq!(expired_by_abort.state, M2nBootstrapState::Expired);
        assert!(expired_by_abort.nccl_unique_id.is_empty());

        let record_key = RedisM2nBootstrapBackend::redis_record_key(&key);
        let uid_key = RedisM2nBootstrapBackend::redis_uid_key(&key);
        let mut conn = backend.get_conn().await.expect("Redis connection");
        let uid_exists: bool = conn.exists(&uid_key).await.expect("UID EXISTS");
        assert!(!uid_exists);
        let _: usize = conn.del(&record_key).await.expect("delete record");
        let _: usize = conn.del(&uid_key).await.expect("delete UID");
        let abort_record_key = RedisM2nBootstrapBackend::redis_record_key(&abort_key);
        let abort_uid_key = RedisM2nBootstrapBackend::redis_uid_key(&abort_key);
        let _: usize = conn
            .del(&abort_record_key)
            .await
            .expect("delete abort record");
        let _: usize = conn.del(&abort_uid_key).await.expect("delete abort UID");
    }
}
