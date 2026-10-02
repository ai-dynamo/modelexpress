// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::envs::{
    DOCKER_CONFIG, HOME, MODEL_EXPRESS_OCI_BEARER_TOKEN, MODEL_EXPRESS_OCI_PASSWORD,
    MODEL_EXPRESS_OCI_TOKEN, MODEL_EXPRESS_OCI_USERNAME,
};
use anyhow::{Context, Result};
use docker_credential::{CredentialRetrievalError, DockerCredential};
use google_cloud_auth::credentials::Builder;
use oci_client::secrets::RegistryAuth;
use std::{
    env,
    fs::File,
    io::{BufReader, ErrorKind},
    path::PathBuf,
};

fn optional_env(key: &str) -> Result<Option<String>> {
    match env::var(key) {
        Ok(value) if value.is_empty() => anyhow::bail!("{key} must not be empty"),
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => anyhow::bail!("{key} must be UTF-8"),
    }
}

fn from_env() -> Result<Option<RegistryAuth>> {
    if let Some(token) = optional_env(MODEL_EXPRESS_OCI_BEARER_TOKEN)? {
        return Ok(Some(RegistryAuth::Bearer(token)));
    }
    let username = optional_env(MODEL_EXPRESS_OCI_USERNAME)?;
    let password = match optional_env(MODEL_EXPRESS_OCI_PASSWORD)? {
        Some(password) => Some(password),
        None => optional_env(MODEL_EXPRESS_OCI_TOKEN)?,
    };
    match (username, password) {
        (None, None) => Ok(None),
        (Some(username), Some(password)) => Ok(Some(RegistryAuth::Basic(username, password))),
        _ => anyhow::bail!("OCI credentials require a username and a password or token"),
    }
}

fn from_docker_config(registry: &str) -> Result<Option<RegistryAuth>> {
    let (directory, explicit) = match optional_env(DOCKER_CONFIG)? {
        Some(path) => (PathBuf::from(path), true),
        None => match env::var_os(HOME) {
            Some(home) => (PathBuf::from(home).join(".docker"), false),
            None => return Ok(None),
        },
    };
    let path = directory.join("config.json");
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if !explicit && error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("Failed to open Docker config {path:?}"));
        }
    };
    match docker_credential::get_credential_from_reader(BufReader::new(file), registry) {
        Ok(DockerCredential::UsernamePassword(user, password)) => {
            Ok(Some(RegistryAuth::Basic(user, password)))
        }
        Ok(DockerCredential::IdentityToken(_)) => anyhow::bail!(
            "Docker identity tokens are not supported by the OCI client; configure a username and password or an OCI bearer token"
        ),
        Err(CredentialRetrievalError::NoCredentialConfigured) => Ok(None),
        // A failed helper can put credentials in its output. Do not include that
        // output in logs or an error returned to the caller.
        Err(CredentialRetrievalError::HelperFailure { helper, .. }) => {
            anyhow::bail!("Docker credential helper {helper:?} failed")
        }
        Err(error) => Err(error).context("Failed to read Docker registry credentials"),
    }
}

pub async fn resolve(registry: &str) -> Result<RegistryAuth> {
    if let Some(auth) = from_env()? {
        return Ok(auth);
    }
    let server = registry.to_owned();
    if let Some(auth) = tokio::task::spawn_blocking(move || from_docker_config(&server))
        .await
        .context("Docker credential lookup task failed")??
    {
        return Ok(auth);
    }
    if !is_gar_registry(registry) {
        return Ok(RegistryAuth::Anonymous);
    }
    let credentials = Builder::default()
        .with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
        .build_access_token_credentials()
        .context("Failed to load Application Default Credentials for Google Artifact Registry")?;
    let access_token = credentials
        .access_token()
        .await
        .context("Failed to obtain a Google Artifact Registry access token")?;
    Ok(RegistryAuth::Basic(
        "oauth2accesstoken".to_string(),
        access_token.token,
    ))
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
    fn explicit_credentials_require_a_complete_pair() {
        let lock = acquire_env_mutex();
        let _bearer = EnvVarGuard::remove(&lock, MODEL_EXPRESS_OCI_BEARER_TOKEN);
        let _user = EnvVarGuard::set(&lock, MODEL_EXPRESS_OCI_USERNAME, "user");
        let _password = EnvVarGuard::remove(&lock, MODEL_EXPRESS_OCI_PASSWORD);
        let _token = EnvVarGuard::remove(&lock, MODEL_EXPRESS_OCI_TOKEN);
        assert_eq!(
            from_env().expect_err("partial credentials").to_string(),
            "OCI credentials require a username and a password or token"
        );
        let _password = EnvVarGuard::set(&lock, MODEL_EXPRESS_OCI_PASSWORD, "password");
        assert_eq!(
            from_env().expect("basic"),
            Some(RegistryAuth::Basic("user".into(), "password".into()))
        );
        let _bearer = EnvVarGuard::set(&lock, MODEL_EXPRESS_OCI_BEARER_TOKEN, "bearer");
        assert_eq!(
            from_env().expect("bearer"),
            Some(RegistryAuth::Bearer("bearer".into()))
        );
    }

    #[test]
    fn docker_config_credentials_and_errors() {
        let lock = acquire_env_mutex();
        let directory = tempfile::tempdir().expect("config directory");
        let _config = EnvVarGuard::set(
            &lock,
            DOCKER_CONFIG,
            directory.path().to_str().expect("path"),
        );
        let path = directory.path().join("config.json");
        std::fs::write(
            &path,
            br#"{"auths":{"registry.example.com":{"auth":"dXNlcjpwYXNzd29yZA=="}}}"#,
        )
        .expect("write config");
        assert_eq!(
            from_docker_config("registry.example.com").expect("credential"),
            Some(RegistryAuth::Basic("user".into(), "password".into()))
        );
        assert_eq!(
            from_docker_config("public.example.com").expect("no credential"),
            None
        );
        std::fs::write(&path, b"invalid").expect("malformed config");
        assert_eq!(
            from_docker_config("registry.example.com")
                .expect_err("invalid")
                .to_string(),
            "Failed to read Docker registry credentials"
        );
    }

    #[test]
    fn configured_helper_errors_do_not_fall_back() {
        let lock = acquire_env_mutex();
        let directory = tempfile::tempdir().expect("config directory");
        let _config = EnvVarGuard::set(
            &lock,
            DOCKER_CONFIG,
            directory.path().to_str().expect("path"),
        );
        std::fs::write(
            directory.path().join("config.json"),
            br#"{"credHelpers":{"registry.example.com":"modelexpress-nonexistent-test-helper"}}"#,
        )
        .expect("config");
        assert_eq!(
            from_docker_config("registry.example.com")
                .expect_err("helper error")
                .to_string(),
            "Failed to read Docker registry credentials"
        );
    }

    #[test]
    fn gar_host_matching() {
        assert!(is_gar_registry("us-docker.pkg.dev"));
        assert!(!is_gar_registry("pkg.dev.example.com"));
    }
}
