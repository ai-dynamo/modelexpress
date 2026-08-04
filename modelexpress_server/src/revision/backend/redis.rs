// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::revision::{
    ReceiverStateRecord, RevisionLifecycleState, RevisionRecord,
};
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
redis.call('SADD', KEYS[2], KEYS[1])
return {1, ARGV[2]}
"#;

const COMMIT_LUA: &str = r#"
local current = redis.call('HGET', KEYS[1], 'record')
if not current then return 0 end
if current ~= ARGV[1] then return 2 end
redis.call('HSET', KEYS[1], 'record', ARGV[2])
return 1
"#;

const UPSERT_RECEIVER_LUA: &str = r#"
local existing_semantic = redis.call('HGET', KEYS[1], 'semantic')
if existing_semantic and existing_semantic == ARGV[1] then
    return redis.call('HGET', KEYS[1], 'record')
end
redis.call('HSET', KEYS[1], 'semantic', ARGV[1], 'record', ARGV[2])
redis.call('SADD', KEYS[2], KEYS[1])
return ARGV[2]
"#;

fn digest_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn revision_key(model_id: &str, version: &str) -> String {
    format!(
        "mx:revision:{}",
        digest_hex(&format!("{model_id}\0{version}"))
    )
}

fn revision_index_key(model_id: &str) -> String {
    format!("mx:revision-index:{}", digest_hex(model_id))
}

fn receiver_key(model_id: &str, version: &str, receiver_id: &str) -> String {
    format!(
        "mx:revision-receiver:{}",
        digest_hex(&format!("{model_id}\0{version}\0{receiver_id}"))
    )
}

fn receiver_index_key(model_id: &str, version: &str) -> String {
    format!(
        "mx:revision-receiver-index:{}",
        digest_hex(&format!("{model_id}\0{version}"))
    )
}

fn decode_record(bytes: &[u8]) -> CatalogResult<RevisionRecord> {
    RevisionRecord::decode(bytes).map_err(Into::into)
}

fn verify_record_identity(
    record: RevisionRecord,
    model_id: &str,
    version: &str,
) -> CatalogResult<RevisionRecord> {
    if record
        .manifest
        .as_ref()
        .is_some_and(|manifest| manifest.model_id == model_id && manifest.version == version)
    {
        Ok(record)
    } else {
        Err(format!("corrupt revision identity at '{model_id}/{version}'").into())
    }
}

fn verify_receiver_semantics(
    record: ReceiverStateRecord,
    expected: &ReceiverStateRecord,
) -> CatalogResult<ReceiverStateRecord> {
    if record.model_id == expected.model_id
        && record.version == expected.version
        && record.receiver_id == expected.receiver_id
        && record.state == expected.state
        && record.installed_version == expected.installed_version
        && record.detail == expected.detail
    {
        Ok(record)
    } else {
        Err("corrupt receiver state record".into())
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
        let key = revision_key(&submitted_manifest.model_id, &submitted_manifest.version);
        let manifest_bytes = submitted_manifest.encode_to_vec();
        let record_bytes = record.encode_to_vec();
        let mut connection = self.connection().await?;
        let (code, existing): (i32, Vec<u8>) = redis::Script::new(PUBLISH_LUA)
            .key(key)
            .key(revision_index_key(&submitted_manifest.model_id))
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
                    &submitted_manifest.version,
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
        version: &str,
    ) -> CatalogResult<Option<RevisionRecord>> {
        let mut connection = self.connection().await?;
        let bytes: Option<Vec<u8>> = connection
            .hget(revision_key(model_id, version), "record")
            .await?;
        match bytes {
            Some(bytes) => Ok(Some(verify_record_identity(
                decode_record(&bytes)?,
                model_id,
                version,
            )?)),
            None => Ok(None),
        }
    }

    async fn list_revisions(&self, model_id: &str) -> CatalogResult<Vec<RevisionRecord>> {
        let mut connection = self.connection().await?;
        let keys: Vec<String> = connection.smembers(revision_index_key(model_id)).await?;
        let mut records = Vec::with_capacity(keys.len());
        for key in keys {
            let bytes: Vec<u8> = connection
                .hget(key, "record")
                .await
                .map_err(|error| format!("corrupt revision index: {error}"))?;
            let record = decode_record(&bytes)?;
            if !record
                .manifest
                .as_ref()
                .is_some_and(|manifest| manifest.model_id == model_id)
            {
                return Err(format!("corrupt revision index for model '{model_id}'").into());
            }
            records.push(record);
        }
        records.sort_by(|left, right| {
            left.created_at_unix_ms
                .cmp(&right.created_at_unix_ms)
                .then_with(|| {
                    let left_version = left
                        .manifest
                        .as_ref()
                        .map_or("", |manifest| manifest.version.as_str());
                    let right_version = right
                        .manifest
                        .as_ref()
                        .map_or("", |manifest| manifest.version.as_str());
                    left_version.cmp(right_version)
                })
        });
        Ok(records)
    }

    async fn commit_revision(
        &self,
        model_id: &str,
        version: &str,
        changed_at_unix_ms: u64,
    ) -> CatalogResult<CommitOutcome> {
        let key = revision_key(model_id, version);
        for _ in 0..8 {
            let Some(current) = self.get_revision(model_id, version).await? else {
                return Ok(CommitOutcome::NotFound);
            };
            match RevisionLifecycleState::try_from(current.state).ok() {
                Some(RevisionLifecycleState::Committed) => {
                    return Ok(CommitOutcome::AlreadyCommitted(current));
                }
                Some(RevisionLifecycleState::Ready) => {}
                _ => return Ok(CommitOutcome::InvalidState(current)),
            }
            let mut updated = current.clone();
            updated.state = RevisionLifecycleState::Committed as i32;
            updated.state_changed_at_unix_ms = changed_at_unix_ms;
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

    async fn upsert_receiver_state(
        &self,
        record: ReceiverStateRecord,
    ) -> CatalogResult<ReceiverStateRecord> {
        let key = receiver_key(&record.model_id, &record.version, &record.receiver_id);
        let mut semantic = record.clone();
        semantic.observed_at_unix_ms = 0;
        let mut connection = self.connection().await?;
        let stored: Vec<u8> = redis::Script::new(UPSERT_RECEIVER_LUA)
            .key(key)
            .key(receiver_index_key(&record.model_id, &record.version))
            .arg(semantic.encode_to_vec())
            .arg(record.encode_to_vec())
            .invoke_async(&mut connection)
            .await?;
        let stored = ReceiverStateRecord::decode(stored.as_slice())?;
        verify_receiver_semantics(stored, &record)
    }

    async fn list_receiver_states(
        &self,
        model_id: &str,
        version: &str,
    ) -> CatalogResult<Vec<ReceiverStateRecord>> {
        let mut connection = self.connection().await?;
        let keys: Vec<String> = connection
            .smembers(receiver_index_key(model_id, version))
            .await?;
        let mut receivers = Vec::with_capacity(keys.len());
        for key in keys {
            let bytes: Vec<u8> = connection
                .hget(key, "record")
                .await
                .map_err(|error| format!("corrupt receiver index: {error}"))?;
            let record = ReceiverStateRecord::decode(bytes.as_slice())?;
            if record.model_id != model_id || record.version != version {
                return Err(format!("corrupt receiver index for '{model_id}/{version}'").into());
            }
            receivers.push(record);
        }
        receivers.sort_by(|left, right| left.receiver_id.cmp(&right.receiver_id));
        Ok(receivers)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use modelexpress_common::grpc::revision::{
        ChangeState, DeltaLocation, DeltaTransferMethod, RankDelta, RevisionManifest, RevisionRank,
        S3Location, delta_location,
    };

    use super::*;
    use crate::revision::state::RevisionCatalogState;

    #[test]
    fn redis_keys_are_namespaced_and_identity_bound() {
        let key = revision_key("model/a", "v1");
        assert!(key.starts_with("mx:revision:"));
        assert!(!key.contains("model/a"));
        assert_ne!(revision_key("a:b", "c"), revision_key("a", "b:c"));
        let index = revision_index_key("model/a");
        assert!(index.starts_with("mx:revision-index:"));
        assert!(!index.contains("model/a"));
    }

    #[test]
    fn redis_rejects_records_at_the_wrong_natural_identity() {
        let record = RevisionRecord {
            manifest: Some(RevisionManifest {
                model_id: "model".to_string(),
                version: "v1".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(verify_record_identity(record.clone(), "model", "v1").is_ok());
        assert!(verify_record_identity(record, "model", "v2").is_err());
    }

    #[tokio::test]
    #[ignore = "requires Redis at MX_TEST_REDIS_URL"]
    async fn redis_backend_satisfies_catalog_lifecycle_when_test_url_is_set() {
        let url = std::env::var("MX_TEST_REDIS_URL").expect("MX_TEST_REDIS_URL");
        let model_id = format!("test-model-{}", uuid::Uuid::new_v4());
        let manifest = RevisionManifest {
            model_id: model_id.clone(),
            version: "v1".to_string(),
            base_version: Some("v0".to_string()),
            transfer_method: DeltaTransferMethod::Canonical as i32,
            delta_method: Some("xor".to_string()),
            compression_algorithm: Some("zstd".to_string()),
            format_digest: "format".to_string(),
            base_digest: Some("base".to_string()),
            target_digest: "target".to_string(),
            ranks: vec![RevisionRank {
                trainer_rank: 0,
                producer_id: "producer".to_string(),
                source_layout_digest: "layout".to_string(),
                delta: Some(RankDelta {
                    change_state: ChangeState::Dirty as i32,
                    checksum: Some("deadbeef".to_string()),
                    location: Some(DeltaLocation {
                        transport: Some(delta_location::Transport::S3(S3Location {
                            bucket: "bucket".to_string(),
                            key: "root.json".to_string(),
                            object_version: Some("object-v1".to_string()),
                        })),
                    }),
                    delta_descriptor: None,
                }),
                shards: vec![],
            }],
        };
        let backend = Arc::new(RedisRevisionCatalogBackend::new(&url));
        let state = RevisionCatalogState::with_backend(backend);
        state.connect().await.expect("connect");

        let created = state.publish(manifest.clone(), 100).await.expect("publish");
        assert!(created.created);
        let retry = state.publish(manifest, 999).await.expect("retry");
        assert!(!retry.created);
        assert_eq!(retry.record, created.record);
        let committed = state.commit(&model_id, "v1", 200).await.expect("commit");
        assert_eq!(committed.state, RevisionLifecycleState::Committed as i32);
    }
}
