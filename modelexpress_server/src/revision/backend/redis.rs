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
    if existing_manifest == ARGV[1] then
        return {2, existing_record}
    end
    return {0, existing_record}
end
redis.call('HSET', KEYS[1], 'manifest', ARGV[1], 'record', ARGV[2])
return {1, ARGV[2]}
"#;

const COMMIT_LUA: &str = r#"
local current = redis.call('HGET', KEYS[1], 'record')
if not current then return 0 end
if current ~= ARGV[1] then return 2 end
redis.call('HSET', KEYS[1], 'record', ARGV[2])
return 1
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
        let (code, existing): (i32, Vec<u8>) = redis::Script::new(PUBLISH_LUA)
            .key(key)
            .arg(manifest_bytes)
            .arg(record_bytes)
            .invoke_async(&mut connection)
            .await?;
        match code {
            1 => Ok(PublishReadyOutcome::Created(record)),
            2 => {
                let existing = verify_record_identity(
                    decode_record(&existing)?,
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
        let bytes: Option<Vec<u8>> = connection
            .hget(revision_key(model_id, target_version), "record")
            .await?;
        match bytes {
            Some(bytes) => Ok(Some(verify_record_identity(
                decode_record(&bytes)?,
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
        let key = revision_key(model_id, target_version);
        for _ in 0..8 {
            let Some(current) = self.get_revision(model_id, target_version).await? else {
                return Ok(CommitOutcome::NotFound);
            };
            match RevisionState::try_from(current.state).ok() {
                Some(RevisionState::Committed) => {
                    return Ok(CommitOutcome::AlreadyCommitted(current));
                }
                Some(RevisionState::Ready) => {}
                _ => return Ok(CommitOutcome::InvalidState(current)),
            }
            let mut updated = current.clone();
            updated.state = RevisionState::Committed as i32;
            let mut connection = self.connection().await?;
            let result: i32 = redis::Script::new(COMMIT_LUA)
                .key(&key)
                .arg(current.encode_to_vec())
                .arg(updated.encode_to_vec())
                .invoke_async(&mut connection)
                .await?;
            match result {
                1 => return Ok(CommitOutcome::Committed(updated)),
                0 => return Ok(CommitOutcome::NotFound),
                2 => continue,
                other => return Err(format!("unexpected Redis commit result: {other}").into()),
            }
        }
        Err("revision commit conflicted repeatedly".into())
    }
}

#[cfg(test)]
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
}
