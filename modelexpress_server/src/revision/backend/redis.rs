// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::revision::{RevisionRecord, RevisionState};
use prost::Message;
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use super::{CatalogResult, CommitOutcome, PublishReadyOutcome, RevisionCatalogBackend};

const PUBLISH_LUA: &str = r#"
local existing_manifest = redis.call('HGET', KEYS[1], 'manifest')
if existing_manifest then
    local existing_record = redis.call('HGET', KEYS[1], 'record')
    local existing_state = redis.call('HGET', KEYS[1], 'state') or ''
    if existing_manifest == ARGV[1] then
        return {2, existing_record, existing_state}
    end
    return {0, existing_record, existing_state}
end
redis.call('HSET', KEYS[1], 'manifest', ARGV[1], 'record', ARGV[2], 'state', ARGV[3])
return {1, ARGV[2], ARGV[3]}
"#;

const COMMIT_LUA: &str = r#"
local current = redis.call('HGET', KEYS[1], 'record')
if not current then return {0, '', ''} end
local state = redis.call('HGET', KEYS[1], 'state')
if not state then
    state = ARGV[3]
    redis.call('HSET', KEYS[1], 'state', state)
end
if state == ARGV[2] then return {2, current, state} end
if state ~= ARGV[1] then return {3, current, state or ''} end
redis.call('HSET', KEYS[1], 'state', ARGV[2])
return {1, current, ARGV[2]}
"#;

fn digest_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn revision_key(model_id: &str, target_version: &str) -> String {
    format!(
        "mx:revision-v0:{}",
        digest_hex(&format!("{model_id}\0{target_version}"))
    )
}

fn decode_record(bytes: &[u8]) -> CatalogResult<RevisionRecord> {
    RevisionRecord::decode(bytes).map_err(Into::into)
}

fn verify_record_identity(
    record: RevisionRecord,
    model_id: &str,
    target_version: &str,
) -> CatalogResult<RevisionRecord> {
    if record.manifest.as_ref().is_some_and(|manifest| {
        manifest.model_id == model_id && manifest.target_version == target_version
    }) {
        Ok(record)
    } else {
        Err(format!("corrupt revision identity at '{model_id}/{target_version}'").into())
    }
}

fn decode_stored_record(
    bytes: &[u8],
    state: Option<i32>,
    model_id: &str,
    target_version: &str,
) -> CatalogResult<RevisionRecord> {
    let mut record = verify_record_identity(decode_record(bytes)?, model_id, target_version)?;
    if let Some(state) = state {
        record.state = state;
    }
    Ok(record)
}

pub struct RedisRevisionCatalogBackend {
    redis: Arc<RwLock<Option<ConnectionManager>>>,
    redis_url: String,
}

impl RedisRevisionCatalogBackend {
    #[must_use]
    pub fn new(redis_url: &str) -> Self {
        Self {
            redis: Arc::new(RwLock::new(None)),
            redis_url: redis_url.to_string(),
        }
    }

    async fn connection(&self) -> CatalogResult<ConnectionManager> {
        {
            let guard = self.redis.read().await;
            if let Some(connection) = guard.as_ref() {
                return Ok(connection.clone());
            }
        }
        let mut guard = self.redis.write().await;
        if let Some(connection) = guard.as_ref() {
            return Ok(connection.clone());
        }
        let client = redis::Client::open(self.redis_url.as_str())?;
        let connection = ConnectionManager::new(client).await?;
        *guard = Some(connection.clone());
        Ok(connection)
    }
}

#[async_trait]
impl RevisionCatalogBackend for RedisRevisionCatalogBackend {
    async fn connect(&self) -> CatalogResult<()> {
        let mut connection = self.connection().await?;
        let response: String = redis::cmd("PING").query_async(&mut connection).await?;
        if response != "PONG" {
            return Err(format!("unexpected Redis PING response: {response}").into());
        }
        Ok(())
    }

    async fn publish_ready(&self, record: RevisionRecord) -> CatalogResult<PublishReadyOutcome> {
        let submitted_manifest = record
            .manifest
            .as_ref()
            .ok_or_else(|| "revision record is missing manifest".to_string())?
            .clone();
        let key = revision_key(
            &submitted_manifest.model_id,
            &submitted_manifest.target_version,
        );
        let manifest_bytes = submitted_manifest.encode_to_vec();
        let record_bytes = record.encode_to_vec();
        let mut connection = self.connection().await?;
        let (code, existing, existing_state): (i32, Vec<u8>, String) =
            redis::Script::new(PUBLISH_LUA)
                .key(key)
                .arg(manifest_bytes)
                .arg(record_bytes)
                .arg(RevisionState::Ready as i32)
                .invoke_async(&mut connection)
                .await?;
        match code {
            1 => Ok(PublishReadyOutcome::Created(record)),
            2 => {
                let existing = decode_stored_record(
                    &existing,
                    if existing_state.is_empty() {
                        None
                    } else {
                        Some(existing_state.parse()?)
                    },
                    &submitted_manifest.model_id,
                    &submitted_manifest.target_version,
                )?;
                if existing.manifest.as_ref() != Some(&submitted_manifest) {
                    return Err("corrupt Redis revision record/manifest pair".into());
                }
                Ok(PublishReadyOutcome::Existing(existing))
            }
            0 => Ok(PublishReadyOutcome::Conflict),
            other => Err(format!("unexpected Redis publish result: {other}").into()),
        }
    }

    async fn get_revision(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> CatalogResult<Option<RevisionRecord>> {
        let mut connection = self.connection().await?;
        let (bytes, state): (Option<Vec<u8>>, Option<i32>) = connection
            .hget(revision_key(model_id, target_version), ("record", "state"))
            .await?;
        match bytes {
            Some(bytes) => Ok(Some(decode_stored_record(
                &bytes,
                state,
                model_id,
                target_version,
            )?)),
            None => Ok(None),
        }
    }

    async fn commit_revision(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> CatalogResult<CommitOutcome> {
        let Some(current) = self.get_revision(model_id, target_version).await? else {
            return Ok(CommitOutcome::NotFound);
        };
        let current_state = current.state;
        let mut connection = self.connection().await?;
        let (code, stored, stored_state): (i32, Vec<u8>, String) = redis::Script::new(COMMIT_LUA)
            .key(revision_key(model_id, target_version))
            .arg(RevisionState::Ready as i32)
            .arg(RevisionState::Committed as i32)
            .arg(current_state)
            .invoke_async(&mut connection)
            .await?;
        let state = || -> CatalogResult<i32> { Ok(stored_state.parse()?) };
        match code {
            1 => Ok(CommitOutcome::Committed(decode_stored_record(
                &stored,
                Some(state()?),
                model_id,
                target_version,
            )?)),
            0 => Ok(CommitOutcome::NotFound),
            2 => Ok(CommitOutcome::AlreadyCommitted(decode_stored_record(
                &stored,
                Some(state()?),
                model_id,
                target_version,
            )?)),
            3 => Ok(CommitOutcome::InvalidState(decode_stored_record(
                &stored,
                Some(state()?),
                model_id,
                target_version,
            )?)),
            other => Err(format!("unexpected Redis commit result: {other}").into()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::revision::RevisionManifest;

    #[test]
    fn revision_keys_are_namespaced_and_bind_model_and_version() {
        let first = revision_key("model-a", "1");
        assert!(first.starts_with("mx:revision-v0:"));
        assert_ne!(first, revision_key("model-b", "1"));
        assert_ne!(first, revision_key("model-a", "2"));
    }

    #[test]
    fn stored_record_identity_must_match_the_lookup_key() {
        let record = RevisionRecord {
            manifest: Some(RevisionManifest {
                model_id: "model".to_string(),
                target_version: "1".to_string(),
                ..Default::default()
            }),
            state: RevisionState::Ready as i32,
        };
        assert!(verify_record_identity(record.clone(), "model", "1").is_ok());
        assert!(verify_record_identity(record, "model", "2").is_err());
    }

    #[tokio::test]
    #[ignore = "requires a live Redis at REDIS_URL"]
    async fn commit_does_not_depend_on_byte_exact_protobuf_reencoding() {
        let redis_url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let backend = RedisRevisionCatalogBackend::new(&redis_url);
        backend.connect().await.expect("connect to Redis");
        let model_id = format!("revision-cas-{}", std::process::id());
        let target_version = "1";
        let key = revision_key(&model_id, target_version);
        let manifest = RevisionManifest {
            model_id: model_id.clone(),
            target_version: target_version.to_string(),
            target_digest: "sha256:target".to_string(),
            format_digest: "sha256:format".to_string(),
            ..Default::default()
        };
        let record = RevisionRecord {
            manifest: Some(manifest.clone()),
            state: RevisionState::Ready as i32,
        };
        let mut stored_record = record.encode_to_vec();
        stored_record.extend_from_slice(&[0x78, 0x01]);
        let mut connection = backend.connection().await.expect("Redis connection");
        let _: () = redis::cmd("HSET")
            .arg(&key)
            .arg("manifest")
            .arg(manifest.encode_to_vec())
            .arg("record")
            .arg(stored_record.clone())
            .query_async(&mut connection)
            .await
            .expect("seed forward-compatible record");

        let outcome = backend
            .commit_revision(&model_id, target_version)
            .await
            .expect("commit");

        assert!(matches!(outcome, CommitOutcome::Committed(_)));
        let record_after_commit: Vec<u8> = redis::cmd("HGET")
            .arg(&key)
            .arg("record")
            .query_async(&mut connection)
            .await
            .expect("read immutable record bytes");
        assert_eq!(record_after_commit, stored_record);
        let fetched = backend
            .get_revision(&model_id, target_version)
            .await
            .expect("read committed revision")
            .expect("revision exists");
        assert_eq!(fetched.state, RevisionState::Committed as i32);
        let _: () = redis::cmd("DEL")
            .arg(key)
            .query_async(&mut connection)
            .await
            .expect("cleanup revision");
    }

    #[tokio::test]
    #[ignore = "requires a live Redis at REDIS_URL"]
    async fn redis_commit_preserves_all_lifecycle_outcomes() {
        let redis_url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let backend = RedisRevisionCatalogBackend::new(&redis_url);
        backend.connect().await.expect("connect to Redis");
        let model_id = format!("revision-lifecycle-{}", std::process::id());
        let target_version = "1";
        let key = revision_key(&model_id, target_version);
        let record = RevisionRecord {
            manifest: Some(RevisionManifest {
                model_id: model_id.clone(),
                target_version: target_version.to_string(),
                target_digest: "sha256:target".to_string(),
                format_digest: "sha256:format".to_string(),
                ..Default::default()
            }),
            state: RevisionState::Ready as i32,
        };

        assert!(matches!(
            backend
                .publish_ready(record.clone())
                .await
                .expect("publish"),
            PublishReadyOutcome::Created(_)
        ));
        assert!(matches!(
            backend
                .publish_ready(record.clone())
                .await
                .expect("idempotent publish"),
            PublishReadyOutcome::Existing(_)
        ));
        assert!(matches!(
            backend
                .commit_revision(&model_id, target_version)
                .await
                .expect("commit"),
            CommitOutcome::Committed(_)
        ));
        assert!(matches!(
            backend
                .commit_revision(&model_id, target_version)
                .await
                .expect("idempotent commit"),
            CommitOutcome::AlreadyCommitted(_)
        ));
        let PublishReadyOutcome::Existing(existing) = backend
            .publish_ready(record.clone())
            .await
            .expect("publish replay after commit")
        else {
            panic!("published revision should already exist");
        };
        assert_eq!(existing.state, RevisionState::Committed as i32);
        assert!(matches!(
            backend
                .commit_revision(&model_id, "missing")
                .await
                .expect("not found"),
            CommitOutcome::NotFound
        ));

        let mut connection = backend.connection().await.expect("Redis connection");
        let _: () = redis::cmd("HSET")
            .arg(&key)
            .arg("state")
            .arg(99)
            .query_async(&mut connection)
            .await
            .expect("seed invalid state");
        assert!(matches!(
            backend
                .commit_revision(&model_id, target_version)
                .await
                .expect("invalid state"),
            CommitOutcome::InvalidState(_)
        ));
        let _: () = redis::cmd("DEL")
            .arg(key)
            .query_async(&mut connection)
            .await
            .expect("cleanup revision");
    }
}
