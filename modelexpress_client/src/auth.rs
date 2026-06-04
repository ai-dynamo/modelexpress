// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attaches a Kubernetes projected ServiceAccount token as `authorization: Bearer` on
//! every RPC, re-reading on TTL/mtime change. A no-op when no token file is present.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::{Request, Status};
use tracing::warn;

const DEFAULT_TOKEN_PATH: &str = "/var/run/secrets/tokens/modelexpress";
const ENV_TOKEN_PATH: &str = "MX_AUTH_TOKEN_PATH";
const ENV_TOKEN_TTL: &str = "MX_AUTH_TOKEN_TTL_SECONDS";
const DEFAULT_TTL: Duration = Duration::from_secs(60);

#[derive(Default)]
struct CachedToken {
    token: Option<String>,
    read_at: Option<Instant>,
    mtime: Option<SystemTime>,
    warned_missing: bool,
}

pub struct TokenProvider {
    path: PathBuf,
    ttl: Duration,
    cache: Mutex<CachedToken>,
}

impl TokenProvider {
    #[must_use]
    pub fn from_env() -> Self {
        let path = std::env::var(ENV_TOKEN_PATH)
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_TOKEN_PATH));
        let ttl = std::env::var(ENV_TOKEN_TTL)
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .map_or(DEFAULT_TTL, Duration::from_secs);
        Self {
            path,
            ttl,
            cache: Mutex::new(CachedToken::default()),
        }
    }

    fn token(&self) -> Option<String> {
        let now = Instant::now();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if let Some(read_at) = cache.read_at
            && now.duration_since(read_at) < self.ttl
        {
            return cache.token.clone();
        }

        match std::fs::metadata(&self.path).and_then(|meta| meta.modified()) {
            Ok(mtime) => {
                if cache.token.is_some() && cache.mtime == Some(mtime) {
                    cache.read_at = Some(now);
                    return cache.token.clone();
                }
                match std::fs::read_to_string(&self.path) {
                    Ok(contents) => {
                        let token = contents.trim();
                        cache.token = (!token.is_empty()).then(|| token.to_string());
                        cache.read_at = Some(now);
                        cache.mtime = Some(mtime);
                        cache.warned_missing = false;
                        cache.token.clone()
                    }
                    Err(e) => {
                        Self::warn_once(&mut cache, &format!("failed to read token: {e}"));
                        None
                    }
                }
            }
            Err(_) => {
                Self::warn_once(
                    &mut cache,
                    "token file not found; RPCs sent without a bearer token \
                     (server must have auth off)",
                );
                cache.token = None;
                cache.read_at = Some(now);
                None
            }
        }
    }

    fn warn_once(cache: &mut CachedToken, message: &str) {
        if !cache.warned_missing {
            warn!("modelexpress auth: {message}");
            cache.warned_missing = true;
        }
    }
}

#[derive(Clone)]
pub struct AuthInterceptor {
    provider: Arc<TokenProvider>,
}

impl AuthInterceptor {
    #[must_use]
    pub fn new(provider: Arc<TokenProvider>) -> Self {
        Self { provider }
    }
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        if let Some(token) = self.provider.token() {
            let value: MetadataValue<_> = format!("Bearer {token}")
                .parse()
                .map_err(|_| Status::internal("auth token is not a valid header value"))?;
            request.metadata_mut().insert("authorization", value);
        }
        Ok(request)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_token_from_file() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "  my-token").expect("write");
        let provider = TokenProvider {
            path: file.path().to_path_buf(),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        };
        assert_eq!(provider.token().as_deref(), Some("my-token"));
    }

    #[test]
    fn missing_file_is_none() {
        let provider = TokenProvider {
            path: PathBuf::from("/no/such/token/file"),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        };
        assert!(provider.token().is_none());
    }

    #[test]
    fn empty_file_is_none() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let provider = TokenProvider {
            path: file.path().to_path_buf(),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        };
        assert!(provider.token().is_none());
    }
}
