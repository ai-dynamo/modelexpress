// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use oci_client::secrets::RegistryAuth;
use std::env;

const OCI_BEARER_TOKEN_ENV_VAR: &str = "MODEL_EXPRESS_OCI_BEARER_TOKEN";
const OCI_USERNAME_ENV_VAR: &str = "MODEL_EXPRESS_OCI_USERNAME";
const OCI_PASSWORD_ENV_VAR: &str = "MODEL_EXPRESS_OCI_PASSWORD";
const OCI_TOKEN_ENV_VAR: &str = "MODEL_EXPRESS_OCI_TOKEN";

fn env_non_empty(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.is_empty())
}

pub fn from_env() -> RegistryAuth {
    if let Some(token) = env_non_empty(OCI_BEARER_TOKEN_ENV_VAR) {
        return RegistryAuth::Bearer(token);
    }

    if let Some(username) = env_non_empty(OCI_USERNAME_ENV_VAR) {
        if let Some(password) = env_non_empty(OCI_PASSWORD_ENV_VAR) {
            return RegistryAuth::Basic(username, password);
        }

        if let Some(token) = env_non_empty(OCI_TOKEN_ENV_VAR) {
            return RegistryAuth::Basic(username, token);
        }
    }

    RegistryAuth::Anonymous
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

        assert_eq!(from_env(), RegistryAuth::Bearer("bearer".to_string()));
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
            RegistryAuth::Basic("user".to_string(), "password".to_string())
        );

        drop(password);
        let _password = EnvVarGuard::remove(&env_lock, OCI_PASSWORD_ENV_VAR);
        assert_eq!(
            from_env(),
            RegistryAuth::Basic("user".to_string(), "token".to_string())
        );

        let _username = EnvVarGuard::remove(&env_lock, OCI_USERNAME_ENV_VAR);
        assert_eq!(from_env(), RegistryAuth::Anonymous);
    }
}
