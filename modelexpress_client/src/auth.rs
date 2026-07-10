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
                cache.mtime = None;
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
    use std::fs;
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

    #[test]
    fn cached_token_is_returned_until_ttl_expires() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "old-token").expect("write old token");
        let provider = TokenProvider {
            path: file.path().to_path_buf(),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        };

        assert_eq!(provider.token().as_deref(), Some("old-token"));
        fs::write(file.path(), "new-token\n").expect("write new token");

        assert_eq!(provider.token().as_deref(), Some("old-token"));
    }

    #[test]
    fn cached_token_reuses_mtime_when_ttl_expires_without_file_change() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "same-token").expect("write token");
        let provider = TokenProvider {
            path: file.path().to_path_buf(),
            ttl: Duration::ZERO,
            cache: Mutex::new(CachedToken::default()),
        };

        assert_eq!(provider.token().as_deref(), Some("same-token"));
        let first_read_at = provider
            .cache
            .lock()
            .expect("cache lock")
            .read_at
            .expect("read timestamp");

        assert_eq!(provider.token().as_deref(), Some("same-token"));
        let second_read_at = provider
            .cache
            .lock()
            .expect("cache lock")
            .read_at
            .expect("read timestamp");
        assert!(second_read_at >= first_read_at);
    }

    #[test]
    fn token_is_reread_after_ttl_when_mtime_changes() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "old-token").expect("write old token");
        let provider = TokenProvider {
            path: file.path().to_path_buf(),
            ttl: Duration::ZERO,
            cache: Mutex::new(CachedToken::default()),
        };

        assert_eq!(provider.token().as_deref(), Some("old-token"));
        fs::write(file.path(), "new-token\n").expect("write new token");
        provider.cache.lock().expect("cache lock").mtime = Some(SystemTime::UNIX_EPOCH);

        assert_eq!(provider.token().as_deref(), Some("new-token"));
    }

    #[test]
    fn interceptor_attaches_authorization_metadata() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "rpc-token").expect("write token");
        let provider = Arc::new(TokenProvider {
            path: file.path().to_path_buf(),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        });
        let mut interceptor = AuthInterceptor::new(provider);

        let request = interceptor
            .call(Request::new(()))
            .expect("interceptor request");

        assert_eq!(
            request
                .metadata()
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer rpc-token")
        );
    }

    #[test]
    fn interceptor_rejects_invalid_header_value() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "bad\nvalue").expect("write token");
        let provider = Arc::new(TokenProvider {
            path: file.path().to_path_buf(),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        });
        let mut interceptor = AuthInterceptor::new(provider);

        let status = interceptor
            .call(Request::new(()))
            .expect_err("invalid metadata value");

        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn missing_file_is_rechecked_immediately() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("late-mounted-token");
        let provider = TokenProvider {
            path: path.clone(),
            ttl: DEFAULT_TTL,
            cache: Mutex::new(CachedToken::default()),
        };

        assert!(provider.token().is_none());

        fs::write(&path, "appeared-token\n").expect("write token");

        assert_eq!(provider.token().as_deref(), Some("appeared-token"));
    }
}
