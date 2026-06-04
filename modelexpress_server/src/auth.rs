// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ServiceAccount `TokenReview` authentication plus an exact-match allowlist.

mod layer;
mod token;

pub use layer::AuthLayer;
pub use token::CallerIdentity;

use std::collections::HashSet;
use std::time::Duration;

use http::HeaderMap;
use kube::client::Client;
use moka::future::Cache;
use secrecy::ExposeSecret;
use sha2::{Digest, Sha256};

use crate::config::{SecurityConfig, ServiceAccountRef};
use token::{extract_bearer, review_token};

#[derive(Debug, thiserror::Error)]
pub enum Denial {
    #[error("missing or invalid service account token")]
    Unauthenticated,
    #[error("{0}")]
    PermissionDenied(String),
}

impl Denial {
    pub(crate) fn into_status(self) -> tonic::Status {
        match self {
            Self::Unauthenticated => {
                tonic::Status::unauthenticated("missing or invalid service account token")
            }
            Self::PermissionDenied(message) => tonic::Status::permission_denied(message),
        }
    }
}

pub struct AuthState {
    client: Client,
    audiences: Vec<String>,
    allowlist: HashSet<ServiceAccountRef>,
    token_cache: Cache<[u8; 32], CallerIdentity>,
    negative_cache: Cache<[u8; 32], ()>,
}

impl AuthState {
    #[must_use]
    pub fn new(client: Client, config: &SecurityConfig) -> Self {
        let ttl = Duration::from_secs(config.cache_ttl_secs);
        Self {
            client,
            audiences: config.token_audiences.clone(),
            allowlist: config.allowed_service_accounts.iter().cloned().collect(),
            token_cache: Cache::builder().time_to_live(ttl).build(),
            negative_cache: Cache::builder().time_to_live(ttl).build(),
        }
    }

    pub(crate) async fn verify(&self, headers: &HeaderMap) -> Result<CallerIdentity, Denial> {
        let token = extract_bearer(headers).ok_or(Denial::Unauthenticated)?;

        let mut hasher = Sha256::new();
        hasher.update(token.expose_secret().as_bytes());
        let key: [u8; 32] = hasher.finalize().into();

        if self.negative_cache.get(&key).await.is_some() {
            return Err(Denial::Unauthenticated);
        }

        let client = self.client.clone();
        let audiences = self.audiences.clone();
        let identity = match self
            .token_cache
            .try_get_with(key, async move {
                review_token(&client, token.expose_secret(), &audiences).await
            })
            .await
        {
            Ok(identity) => identity,
            Err(error) => {
                if error.is_token_rejection() {
                    self.negative_cache.insert(key, ()).await;
                }
                return Err(Denial::Unauthenticated);
            }
        };

        let caller = ServiceAccountRef {
            namespace: identity.namespace.clone(),
            service_account: identity.service_account.clone(),
        };
        if !self.allowlist.contains(&caller) {
            return Err(Denial::PermissionDenied(format!(
                "service account {}:{} is not in the allowlist",
                identity.namespace, identity.service_account
            )));
        }

        Ok(identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unauthenticated_maps_to_grpc_code() {
        let status = Denial::Unauthenticated.into_status();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn permission_denied_maps_to_grpc_code() {
        let status = Denial::PermissionDenied("nope".to_string()).into_status();
        assert_eq!(status.code(), tonic::Code::PermissionDenied);
        assert_eq!(status.message(), "nope");
    }
}
