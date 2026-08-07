// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ServiceAccount `TokenReview` authentication plus an exact-match allowlist.

mod layer;
mod token;

pub use layer::AuthLayer;
pub use token::CallerIdentity;

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use http::HeaderMap;
use kube::client::Client;
use moka::future::Cache;
use secrecy::ExposeSecret;
use secrecy::zeroize::Zeroizing;
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::time::timeout;
use tracing::warn;

use crate::config::{SecurityConfig, ServiceAccountRef};
use token::{TokenError, claimed_service_account, extract_bearer, review_token};

#[derive(Debug, thiserror::Error)]
pub enum Denial {
    #[error("missing or invalid service account token")]
    Unauthenticated,
    #[error("{0}")]
    PermissionDenied(String),
    #[error("authentication backend unavailable")]
    Unavailable,
}

impl Denial {
    pub(crate) fn into_status(self) -> tonic::Status {
        match self {
            Self::Unauthenticated => {
                tonic::Status::unauthenticated("missing or invalid service account token")
            }
            Self::PermissionDenied(message) => tonic::Status::permission_denied(message),
            Self::Unavailable => tonic::Status::unavailable("authentication backend unavailable"),
        }
    }
}

#[derive(Clone)]
struct TokenCacheKey(Zeroizing<String>);

impl TokenCacheKey {
    fn new(token: &str) -> Self {
        Self(Zeroizing::new(token.to_string()))
    }

    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for TokenCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TokenCacheKey([REDACTED])")
    }
}

impl PartialEq for TokenCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for TokenCacheKey {}

impl Hash for TokenCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

pub struct AuthState {
    client: Client,
    audiences: Vec<String>,
    allowlist: HashMap<ServiceAccountRef, Semaphore>,
    token_cache: Cache<TokenCacheKey, CallerIdentity>,
    negative_cache: Cache<TokenCacheKey, ()>,
    review_permits: Semaphore,
}

impl AuthState {
    const TOKEN_CACHE_MAX_ENTRIES: u64 = 10_000;
    const NEGATIVE_CACHE_MAX_ENTRIES: u64 = 1_000;
    const MAX_CONCURRENT_REVIEWS: usize = 8;
    const MAX_CONCURRENT_REVIEWS_PER_CALLER: usize = 4;
    const PERMIT_WAIT: Duration = Duration::from_millis(250);

    #[must_use]
    pub fn new(client: Client, config: &SecurityConfig) -> Self {
        let ttl = Duration::from_secs(config.cache_ttl_secs);
        Self {
            client,
            audiences: config.token_audiences.clone(),
            allowlist: config
                .allowed_service_accounts
                .iter()
                .map(|caller| {
                    (
                        caller.clone(),
                        Semaphore::new(Self::MAX_CONCURRENT_REVIEWS_PER_CALLER),
                    )
                })
                .collect(),
            token_cache: Cache::builder()
                .time_to_live(ttl)
                .max_capacity(Self::TOKEN_CACHE_MAX_ENTRIES)
                .build(),
            negative_cache: Cache::builder()
                .time_to_live(ttl)
                .max_capacity(Self::NEGATIVE_CACHE_MAX_ENTRIES)
                .build(),
            review_permits: Semaphore::new(Self::MAX_CONCURRENT_REVIEWS),
        }
    }

    async fn acquire(semaphore: &Semaphore) -> Result<SemaphorePermit<'_>, TokenError> {
        match timeout(Self::PERMIT_WAIT, semaphore.acquire()).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) | Err(_) => Err(TokenError::Saturated),
        }
    }

    pub(crate) async fn verify(&self, headers: &HeaderMap) -> Result<CallerIdentity, Denial> {
        let token = extract_bearer(headers).ok_or(Denial::Unauthenticated)?;
        let key = TokenCacheKey::new(token.expose_secret());

        if self.negative_cache.get(&key).await.is_some() {
            return Err(Denial::Unauthenticated);
        }

        let identity = match self
            .token_cache
            .try_get_with(key.clone(), async {
                let caller_permits = claimed_service_account(token.expose_secret())
                    .and_then(|claimed| self.allowlist.get(&claimed))
                    .ok_or(TokenError::Prefiltered)?;
                let _caller_permit = Self::acquire(caller_permits).await?;
                let _review_permit = Self::acquire(&self.review_permits).await?;
                review_token(&self.client, token.expose_secret(), &self.audiences).await
            })
            .await
        {
            Ok(identity) => identity,
            Err(error) => {
                if error.is_token_rejection() {
                    self.negative_cache.insert(key, ()).await;
                    return Err(Denial::Unauthenticated);
                }
                if !matches!(*error, TokenError::Saturated) {
                    warn!(error = %error, "TokenReview backend error");
                }
                return Err(Denial::Unavailable);
            }
        };

        let caller = ServiceAccountRef {
            namespace: identity.namespace.clone(),
            service_account: identity.service_account.clone(),
        };
        if !self.allowlist.contains_key(&caller) {
            return Err(Denial::PermissionDenied(format!(
                "service account {}:{} is not in the allowlist",
                identity.namespace, identity.service_account
            )));
        }

        Ok(identity)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
pub(crate) mod test_util {
    use std::collections::VecDeque;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use http::HeaderMap;
    use http::header::{AUTHORIZATION, CONTENT_TYPE};
    use k8s_openapi::api::authentication::v1::{TokenReview, TokenReviewStatus, UserInfo};
    use kube::client::{Body, Client};
    use tower::BoxError;

    use crate::config::{AuthMode, SecurityConfig, ServiceAccountRef};

    pub(crate) fn allowed_ref(namespace: &str, service_account: &str) -> ServiceAccountRef {
        ServiceAccountRef {
            namespace: namespace.to_string(),
            service_account: service_account.to_string(),
        }
    }

    pub(crate) fn security_config(allowed: Vec<ServiceAccountRef>) -> SecurityConfig {
        SecurityConfig {
            mode: Some(AuthMode::Enforce),
            token_audiences: vec!["modelexpress".to_string()],
            allowed_service_accounts: allowed,
            cache_ttl_secs: 60,
        }
    }

    pub(crate) fn sa_token(namespace: &str, service_account: &str, nonce: &str) -> String {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;

        let payload = serde_json::json!({
            "sub": format!("system:serviceaccount:{namespace}:{service_account}"),
            "jti": nonce,
        });
        let payload = serde_json::to_vec(&payload).expect("serialize claims");
        format!(
            "{}.{}.{}",
            URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#),
            URL_SAFE_NO_PAD.encode(payload),
            URL_SAFE_NO_PAD.encode(b"not-a-real-signature"),
        )
    }

    pub(crate) fn bearer_headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {token}")
                .parse()
                .expect("valid bearer header"),
        );
        headers
    }

    pub(crate) fn token_review_for(username: &str) -> TokenReview {
        TokenReview {
            status: Some(TokenReviewStatus {
                authenticated: Some(true),
                audiences: Some(vec!["modelexpress".to_string()]),
                user: Some(UserInfo {
                    username: Some(username.to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub(crate) fn unauthenticated_review(error: &str) -> TokenReview {
        TokenReview {
            status: Some(TokenReviewStatus {
                authenticated: Some(false),
                error: Some(error.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub(crate) fn review_without_user() -> TokenReview {
        TokenReview {
            status: Some(TokenReviewStatus {
                authenticated: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub(crate) enum FakeResponse {
        Review(Box<TokenReview>),
        Error(http::StatusCode),
    }

    pub(crate) fn fake_kube_client(reviews: Vec<TokenReview>) -> (Client, Arc<AtomicUsize>) {
        fake_kube_client_with_responses(
            reviews
                .into_iter()
                .map(|r| FakeResponse::Review(Box::new(r)))
                .collect(),
        )
    }

    pub(crate) fn fake_kube_client_with_responses(
        responses: Vec<FakeResponse>,
    ) -> (Client, Arc<AtomicUsize>) {
        let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
        let calls = Arc::new(AtomicUsize::new(0));

        let service = tower::service_fn({
            let responses = responses.clone();
            let calls = calls.clone();
            move |_request: http::Request<Body>| {
                let responses = responses.clone();
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    let response_item = responses
                        .lock()
                        .expect("fake response queue lock")
                        .pop_front()
                        .expect("fake response");
                    let response = match response_item {
                        FakeResponse::Review(review) => {
                            let body = serde_json::to_vec(&*review).expect("serialize TokenReview");
                            http::Response::builder()
                                .status(http::StatusCode::OK)
                                .header(CONTENT_TYPE, "application/json")
                                .body(Body::from(body))
                                .expect("fake TokenReview response body")
                        }
                        FakeResponse::Error(status) => http::Response::builder()
                            .status(status)
                            .body(Body::empty())
                            .expect("fake error response body"),
                    };
                    Ok::<_, BoxError>(response)
                }
            }
        });

        (Client::new(service, "default"), calls)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    use test_util::{
        allowed_ref, bearer_headers, fake_kube_client, sa_token, security_config, token_review_for,
        unauthenticated_review,
    };

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

    #[test]
    fn unavailable_maps_to_grpc_code() {
        let status = Denial::Unavailable.into_status();
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[tokio::test]
    async fn verify_accepts_allowlisted_identity_and_uses_positive_cache() {
        let (client, calls) =
            fake_kube_client(vec![token_review_for("system:serviceaccount:vllm:worker")]);
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = AuthState::new(client, &config);
        let headers = bearer_headers(&sa_token("vllm", "worker", "good"));

        let first = state.verify(&headers).await.expect("first token review");
        let second = state.verify(&headers).await.expect("cached token review");

        assert_eq!(first.namespace, "vllm");
        assert_eq!(first.service_account, "worker");
        assert_eq!(second.namespace, "vllm");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn verify_rejects_identity_outside_allowlist() {
        let (client, _calls) =
            fake_kube_client(vec![token_review_for("system:serviceaccount:other:worker")]);
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = AuthState::new(client, &config);
        let headers = bearer_headers(&sa_token("vllm", "worker", "spoofed-sub"));

        let error = state
            .verify(&headers)
            .await
            .expect_err("non-allowlisted service account should be denied");

        assert!(
            matches!(error, Denial::PermissionDenied(message) if message.contains("other:worker"))
        );
    }

    #[tokio::test]
    async fn verify_negative_caches_definitive_token_rejections() {
        let (client, calls) = fake_kube_client(vec![unauthenticated_review("expired")]);
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = AuthState::new(client, &config);
        let headers = bearer_headers(&sa_token("vllm", "worker", "expired"));

        assert!(matches!(
            state.verify(&headers).await,
            Err(Denial::Unauthenticated)
        ));
        assert!(matches!(
            state.verify(&headers).await,
            Err(Denial::Unauthenticated)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn verify_returns_unavailable_for_api_errors_without_caching() {
        use test_util::FakeResponse;

        let (client, calls) = test_util::fake_kube_client_with_responses(vec![
            FakeResponse::Error(http::StatusCode::INTERNAL_SERVER_ERROR),
            FakeResponse::Error(http::StatusCode::INTERNAL_SERVER_ERROR),
        ]);
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = AuthState::new(client, &config);
        let headers = bearer_headers(&sa_token("vllm", "worker", "api-error"));

        assert!(matches!(
            state.verify(&headers).await,
            Err(Denial::Unavailable)
        ));
        assert!(matches!(
            state.verify(&headers).await,
            Err(Denial::Unavailable)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn verify_rejects_prefiltered_tokens_without_calling_the_apiserver() {
        let oversized = "a".repeat(token::MAX_TOKEN_BYTES + 1);
        let cases = [
            ("not a jwt at all", "opaque-token"),
            ("two segments", "header.payload"),
            ("empty signature", "header.payload."),
            ("undecodable payload", "aaaa.!!!!.cccc"),
            ("payload without sub", "aaaa.e30.cccc"),
            (
                "sub is not a service account",
                "aaaa.eyJzdWIiOiJhbGljZSJ9.cccc",
            ),
            ("over the length cap", oversized.as_str()),
        ];

        for (case, token) in cases {
            let (client, calls) = fake_kube_client(vec![]);
            let config = security_config(vec![allowed_ref("vllm", "worker")]);
            let state = AuthState::new(client, &config);

            assert!(
                matches!(
                    state.verify(&bearer_headers(token)).await,
                    Err(Denial::Unauthenticated)
                ),
                "{case} should be rejected"
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "{case} reached the apiserver"
            );
        }
    }

    #[tokio::test]
    async fn verify_rejects_unallowlisted_claim_without_calling_the_apiserver() {
        let (client, calls) = fake_kube_client(vec![]);
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = AuthState::new(client, &config);
        let headers = bearer_headers(&sa_token("other", "intruder", "nonce"));

        assert!(matches!(
            state.verify(&headers).await,
            Err(Denial::Unauthenticated)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
