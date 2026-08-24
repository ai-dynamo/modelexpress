// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use google_cloud_auth::credentials::Builder;
use oci_client::secrets::RegistryAuth;
use std::env;

const OCI_BEARER_TOKEN_ENV_VAR: &str = "MODEL_EXPRESS_OCI_BEARER_TOKEN";
const OCI_USERNAME_ENV_VAR: &str = "MODEL_EXPRESS_OCI_USERNAME";
const OCI_PASSWORD_ENV_VAR: &str = "MODEL_EXPRESS_OCI_PASSWORD";
const OCI_TOKEN_ENV_VAR: &str = "MODEL_EXPRESS_OCI_TOKEN";
const GOOGLE_CLOUD_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const GOOGLE_ACCESS_TOKEN_USERNAME: &str = "oauth2accesstoken";

fn env_non_empty(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.is_empty())
}

fn from_env() -> Option<RegistryAuth> {
    if let Some(token) = env_non_empty(OCI_BEARER_TOKEN_ENV_VAR) {
        return Some(RegistryAuth::Bearer(token));
    }

    if let Some(username) = env_non_empty(OCI_USERNAME_ENV_VAR) {
        if let Some(password) = env_non_empty(OCI_PASSWORD_ENV_VAR) {
            return Some(RegistryAuth::Basic(username, password));
        }

        if let Some(token) = env_non_empty(OCI_TOKEN_ENV_VAR) {
            return Some(RegistryAuth::Basic(username, token));
        }
    }

    None
}

pub async fn resolve(registry: &str) -> Result<RegistryAuth> {
    if let Some(auth) = configured_auth(registry) {
        return Ok(auth);
    }

    let credentials = Builder::default()
        .with_scopes([GOOGLE_CLOUD_SCOPE])
        .build_access_token_credentials()
        .context("Failed to load Application Default Credentials for Google Artifact Registry")?;
    let access_token = credentials
        .access_token()
        .await
        .context("Failed to obtain a Google Artifact Registry access token")?;
    Ok(RegistryAuth::Basic(
        GOOGLE_ACCESS_TOKEN_USERNAME.to_string(),
        access_token.token,
    ))
}

fn configured_auth(registry: &str) -> Option<RegistryAuth> {
    from_env().or_else(|| (!is_gar_registry(registry)).then_some(RegistryAuth::Anonymous))
}

fn is_gar_registry(registry: &str) -> bool {
    registry
        .split_once(':')
        .map_or(registry, |(host, _)| host)
        .trim_end_matches('.')
        .ends_with(".pkg.dev")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, acquire_env_mutex};

    #[test]
    fn test_auth_precedence() {
        let env_lock = acquire_env_mutex();
        let _bearer = EnvVarGuard::set(&env_lock, OCI_BEARER_TOKEN_ENV_VAR, "bearer");
        let _username = EnvVarGuard::set(&env_lock, OCI_USERNAME_ENV_VAR, "user");
        let _password = EnvVarGuard::set(&env_lock, OCI_PASSWORD_ENV_VAR, "password");
        let _token = EnvVarGuard::set(&env_lock, OCI_TOKEN_ENV_VAR, "token");

        assert_eq!(from_env(), Some(RegistryAuth::Bearer("bearer".to_string())));
    }

    #[test]
    fn test_auth_uses_password_then_token_then_anonymous() {
        let env_lock = acquire_env_mutex();
        let _bearer = EnvVarGuard::remove(&env_lock, OCI_BEARER_TOKEN_ENV_VAR);
        let _username = EnvVarGuard::set(&env_lock, OCI_USERNAME_ENV_VAR, "user");
        let password = EnvVarGuard::set(&env_lock, OCI_PASSWORD_ENV_VAR, "password");
        let _token = EnvVarGuard::set(&env_lock, OCI_TOKEN_ENV_VAR, "token");

        assert_eq!(
            from_env(),
            Some(RegistryAuth::Basic(
                "user".to_string(),
                "password".to_string()
            ))
        );

        drop(password);
        let _password = EnvVarGuard::remove(&env_lock, OCI_PASSWORD_ENV_VAR);
        assert_eq!(
            from_env(),
            Some(RegistryAuth::Basic("user".to_string(), "token".to_string()))
        );

        let _username = EnvVarGuard::remove(&env_lock, OCI_USERNAME_ENV_VAR);
        assert_eq!(from_env(), None);
    }

    #[test]
    fn test_resolve_uses_static_auth_before_gar_adc() {
        let env_lock = acquire_env_mutex();
        let _bearer = EnvVarGuard::set(&env_lock, OCI_BEARER_TOKEN_ENV_VAR, "bearer");

        assert_eq!(
            configured_auth("us-docker.pkg.dev"),
            Some(RegistryAuth::Bearer("bearer".to_string()))
        );
    }

    #[test]
    fn test_resolve_uses_anonymous_auth_only_outside_gar() {
        let env_lock = acquire_env_mutex();
        let _bearer = EnvVarGuard::remove(&env_lock, OCI_BEARER_TOKEN_ENV_VAR);
        let _username = EnvVarGuard::remove(&env_lock, OCI_USERNAME_ENV_VAR);
        let _password = EnvVarGuard::remove(&env_lock, OCI_PASSWORD_ENV_VAR);
        let _token = EnvVarGuard::remove(&env_lock, OCI_TOKEN_ENV_VAR);

        assert_eq!(
            configured_auth("registry.example.com"),
            Some(RegistryAuth::Anonymous)
        );
        assert!(is_gar_registry("us-docker.pkg.dev"));
        assert!(!is_gar_registry("pkg.dev.example.com"));
    }
}
