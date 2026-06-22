// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bearer-token extraction and Kubernetes `TokenReview` verification.

use http::HeaderMap;
use http::header::AUTHORIZATION;
use k8s_openapi::api::authentication::v1::{TokenReview, TokenReviewSpec, UserInfo};
use kube::api::{Api, PostParams};
use kube::client::Client;
use secrecy::SecretString;

const EXTRA_POD_NAME: &str = "authentication.kubernetes.io/pod-name";
const EXTRA_POD_UID: &str = "authentication.kubernetes.io/pod-uid";

#[derive(Debug, Clone)]
pub struct CallerIdentity {
    pub namespace: String,
    pub service_account: String,
    pub pod_name: Option<String>,
    pub pod_uid: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum TokenError {
    #[error("kube api error during TokenReview: {0}")]
    Api(#[source] kube::Error),
    #[error("TokenReview returned no status")]
    NoStatus,
    #[error("token not authenticated: {0:?}")]
    NotAuthenticated(Option<String>),
    #[error("TokenReview returned no user info")]
    NoUser,
    #[error("caller is not a kubernetes service account")]
    NotServiceAccount,
}

impl TokenError {
    pub(crate) fn is_token_rejection(&self) -> bool {
        matches!(self, Self::NotAuthenticated(_) | Self::NotServiceAccount)
    }
}

pub(crate) fn extract_bearer(headers: &HeaderMap) -> Option<SecretString> {
    let value = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let raw = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))?;
    if raw.is_empty() {
        return None;
    }
    Some(SecretString::from(raw))
}

pub(crate) fn parse_sa_username(username: &str) -> Option<(String, String)> {
    let rest = username.strip_prefix("system:serviceaccount:")?;
    let (namespace, service_account) = rest.split_once(':')?;
    if namespace.is_empty() || service_account.is_empty() {
        return None;
    }
    Some((namespace.to_string(), service_account.to_string()))
}

pub(crate) fn caller_from_userinfo(user: &UserInfo) -> Option<CallerIdentity> {
    let username = user.username.as_deref()?;
    let (namespace, service_account) = parse_sa_username(username)?;
    let first = |key: &str| -> Option<String> {
        user.extra
            .as_ref()
            .and_then(|map| map.get(key))
            .and_then(|values| values.first().cloned())
    };
    Some(CallerIdentity {
        namespace,
        service_account,
        pod_name: first(EXTRA_POD_NAME),
        pod_uid: first(EXTRA_POD_UID),
    })
}

pub(crate) async fn review_token(
    client: &Client,
    token: &str,
    audiences: &[String],
) -> Result<CallerIdentity, TokenError> {
    let api: Api<TokenReview> = Api::all(client.clone());
    let review = TokenReview {
        spec: TokenReviewSpec {
            token: Some(token.to_string()),
            audiences: if audiences.is_empty() {
                None
            } else {
                Some(audiences.to_vec())
            },
        },
        ..Default::default()
    };
    let reviewed = api
        .create(&PostParams::default(), &review)
        .await
        .map_err(TokenError::Api)?;
    let status = reviewed.status.ok_or(TokenError::NoStatus)?;
    if !status.authenticated.unwrap_or(false) {
        return Err(TokenError::NotAuthenticated(status.error));
    }
    let user = status.user.ok_or(TokenError::NoUser)?;
    caller_from_userinfo(&user).ok_or(TokenError::NotServiceAccount)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::test_util::{
        fake_kube_client, review_without_user, token_review_for, unauthenticated_review,
    };
    use secrecy::ExposeSecret;
    use std::sync::atomic::Ordering;

    #[test]
    fn extracts_bearer_token_both_cases() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer abc.def".parse().unwrap());
        assert_eq!(extract_bearer(&headers).unwrap().expose_secret(), "abc.def");

        let mut lower = HeaderMap::new();
        lower.insert(AUTHORIZATION, "bearer xyz".parse().unwrap());
        assert_eq!(extract_bearer(&lower).unwrap().expose_secret(), "xyz");
    }

    #[test]
    fn rejects_missing_or_malformed_bearer() {
        assert!(extract_bearer(&HeaderMap::new()).is_none());

        let mut basic = HeaderMap::new();
        basic.insert(AUTHORIZATION, "Basic abc".parse().unwrap());
        assert!(extract_bearer(&basic).is_none());

        let mut empty = HeaderMap::new();
        empty.insert(AUTHORIZATION, "Bearer ".parse().unwrap());
        assert!(extract_bearer(&empty).is_none());
    }

    #[test]
    fn parses_valid_sa_username() {
        assert_eq!(
            parse_sa_username("system:serviceaccount:foo:bar"),
            Some(("foo".to_string(), "bar".to_string()))
        );
    }

    #[test]
    fn rejects_non_sa_usernames() {
        assert_eq!(parse_sa_username("system:node:node-1"), None);
        assert_eq!(parse_sa_username("alice"), None);
        assert_eq!(parse_sa_username("system:serviceaccount:foo:"), None);
        assert_eq!(parse_sa_username("system:serviceaccount::bar"), None);
        assert_eq!(parse_sa_username("system:serviceaccount:foo"), None);
    }

    #[test]
    fn caller_from_userinfo_extracts_pod_identity() {
        let mut extra = std::collections::BTreeMap::new();
        extra.insert(EXTRA_POD_NAME.to_string(), vec!["worker-0".to_string()]);
        extra.insert(EXTRA_POD_UID.to_string(), vec!["uid-123".to_string()]);
        let user = UserInfo {
            username: Some("system:serviceaccount:ns:sa".to_string()),
            extra: Some(extra),
            ..Default::default()
        };
        let caller = caller_from_userinfo(&user).unwrap();
        assert_eq!(caller.namespace, "ns");
        assert_eq!(caller.service_account, "sa");
        assert_eq!(caller.pod_name.as_deref(), Some("worker-0"));
        assert_eq!(caller.pod_uid.as_deref(), Some("uid-123"));
    }

    #[test]
    fn caller_from_userinfo_without_extra_has_no_pod_identity() {
        let user = UserInfo {
            username: Some("system:serviceaccount:ns:sa".to_string()),
            ..Default::default()
        };
        let caller = caller_from_userinfo(&user).unwrap();
        assert!(caller.pod_name.is_none());
        assert!(caller.pod_uid.is_none());
    }

    #[test]
    fn only_definitive_rejections_are_cacheable() {
        assert!(TokenError::NotAuthenticated(None).is_token_rejection());
        assert!(TokenError::NotServiceAccount.is_token_rejection());
        assert!(!TokenError::NoStatus.is_token_rejection());
        assert!(!TokenError::NoUser.is_token_rejection());
    }

    #[test]
    fn caller_from_userinfo_rejects_non_sa() {
        let user = UserInfo {
            username: Some("system:node:node-1".to_string()),
            ..Default::default()
        };
        assert!(caller_from_userinfo(&user).is_none());
    }

    #[tokio::test]
    async fn review_token_accepts_service_account_identity() {
        let (client, calls) =
            fake_kube_client(vec![token_review_for("system:serviceaccount:vllm:worker")]);
        let audiences = vec!["modelexpress".to_string()];

        let caller = review_token(&client, "token", &audiences)
            .await
            .expect("authenticated TokenReview");

        assert_eq!(caller.namespace, "vllm");
        assert_eq!(caller.service_account, "worker");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn review_token_rejects_unauthenticated_status() {
        let (client, _calls) = fake_kube_client(vec![unauthenticated_review("expired")]);

        let error = review_token(&client, "token", &[])
            .await
            .expect_err("unauthenticated TokenReview");

        assert!(matches!(
            error,
            TokenError::NotAuthenticated(Some(message)) if message == "expired"
        ));
    }

    #[tokio::test]
    async fn review_token_rejects_missing_status() {
        let (client, _calls) = fake_kube_client(vec![TokenReview::default()]);

        let error = review_token(&client, "token", &[])
            .await
            .expect_err("missing TokenReview status");

        assert!(matches!(error, TokenError::NoStatus));
    }

    #[tokio::test]
    async fn review_token_rejects_missing_user() {
        let (client, _calls) = fake_kube_client(vec![review_without_user()]);

        let error = review_token(&client, "token", &[])
            .await
            .expect_err("missing TokenReview user");

        assert!(matches!(error, TokenError::NoUser));
    }
}
