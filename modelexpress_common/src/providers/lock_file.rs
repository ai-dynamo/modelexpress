// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::digest::{DigestError, sha256_hex};
use crate::models::ModelProvider;
use anyhow::Result;
use fd_lock::{RwLock as FileRwLock, RwLockWriteGuard as FileWriteGuard};
use futures::Future;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

const FILE_LOCK_POLL_INTERVAL: Duration = Duration::from_secs(1);
const MODEL_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Errors returned by bounded or non-blocking advisory file lock operations.
#[derive(Debug, thiserror::Error)]
pub enum LockFileError {
    /// Lock remained contended until the bounded wait expired.
    #[error("timed out after {timeout:?} waiting for lock file '{}'", path.display())]
    Busy { path: PathBuf, timeout: Duration },
    /// Filesystem or OS error while opening or locking the file (including
    /// `ENOLCK`, `EOPNOTSUPP`, and `ENOSYS`). Callers must fail closed.
    #[error("lock file I/O error for '{}': {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// Failed to compute SHA-256 while deriving a model lock filename.
    #[error("failed to hash model lock key: {0}")]
    Digest(#[from] DigestError),
    /// Blocking lock helper task panicked or was cancelled.
    #[error("blocking lock task failed for '{}': {reason}", path.display())]
    TaskJoin { path: PathBuf, reason: String },
}

/// Owned shared (`LOCK_SH`) advisory file lock guard.
///
/// Dropping this guard closes the underlying open file description, releasing
/// the `flock` lock synchronously.
#[must_use = "if dropped immediately, the shared file lock is released"]
#[derive(Debug)]
pub struct SharedLockGuard {
    _lock_file: FileRwLock<fs::File>,
}

/// Owned exclusive (`LOCK_EX`) advisory file lock guard.
///
/// Dropping this guard closes the underlying open file description, releasing
/// the `flock` lock synchronously.
#[must_use = "if dropped immediately, the exclusive file lock is released"]
#[derive(Debug)]
pub struct ExclusiveLockGuard {
    _lock_file: FileRwLock<fs::File>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LockMode {
    Shared,
    Exclusive,
}

enum TryAcquireStep {
    Acquired(FileRwLock<fs::File>),
    Busy(FileRwLock<fs::File>),
}

pub struct LockFile;

impl LockFile {
    /// Derive the per-model advisory lock path for `(provider, canonical_model_name)`.
    ///
    /// Uses `<cache_dir>/.mx/locks/<provider>-<sha256>.lock` so lock files never
    /// collide with provider-internal lock subdirectories (such as NGC's
    /// `.mx/locks/ngc/...`) and remain fixed-length regardless of model name.
    /// Lock files are never deleted so inodes remain stable across callers.
    pub fn model_lock_path(
        cache_dir: &Path,
        provider: ModelProvider,
        canonical_model_name: &str,
    ) -> Result<PathBuf, LockFileError> {
        let provider_slug = match provider {
            ModelProvider::HuggingFace => "huggingface",
            ModelProvider::Ngc => "ngc",
            ModelProvider::S3 => "s3",
            ModelProvider::Gcs => "gcs",
        };
        let digest = sha256_hex(canonical_model_name.as_bytes())?;
        Ok(cache_dir
            .join(".mx/locks")
            .join(format!("{provider_slug}-{digest}.lock")))
    }

    fn try_acquire<'a>(
        lock: &'a mut FileRwLock<fs::File>,
        lock_path: &Path,
    ) -> Result<Option<FileWriteGuard<'a, fs::File>>> {
        match lock.try_write() {
            Ok(guard) => Ok(Some(guard)),
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(None)
            }
            Err(err) => Err(anyhow::anyhow!(
                "Failed to acquire lock file '{}': {}",
                lock_path.display(),
                err
            )),
        }
    }

    fn open_io(path: &Path) -> Result<FileRwLock<fs::File>, LockFileError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| LockFileError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|source| LockFileError::Io {
                path: path.to_path_buf(),
                source,
            })?;

        Ok(FileRwLock::new(file))
    }

    fn try_lock_owned(
        mut lock_file: FileRwLock<fs::File>,
        lock_path: &Path,
        mode: LockMode,
    ) -> Result<TryAcquireStep, LockFileError> {
        loop {
            // Suppress `LOCK_UN` on guard drop so lock ownership transfers to the
            // caller's owned guard; dropping `FileRwLock<fs::File>` closes the open
            // file description and releases `flock` synchronously.
            let attempt = match mode {
                LockMode::Shared => lock_file.try_read().map(std::mem::forget),
                LockMode::Exclusive => lock_file.try_write().map(std::mem::forget),
            };
            match attempt {
                Ok(()) => return Ok(TryAcquireStep::Acquired(lock_file)),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    return Ok(TryAcquireStep::Busy(lock_file));
                }
                Err(source) => {
                    return Err(LockFileError::Io {
                        path: lock_path.to_path_buf(),
                        source,
                    });
                }
            }
        }
    }

    async fn acquire_timeout(
        lock_path: &Path,
        mode: LockMode,
        timeout: Duration,
    ) -> Result<FileRwLock<fs::File>, LockFileError> {
        let deadline = tokio::time::Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(tokio::time::Instant::now);
        let open_path = lock_path.to_path_buf();
        let mut lock_file = tokio::task::spawn_blocking(move || Self::open_io(&open_path))
            .await
            .map_err(|err| LockFileError::TaskJoin {
                path: lock_path.to_path_buf(),
                reason: err.to_string(),
            })??;

        loop {
            let path = lock_path.to_path_buf();
            let step =
                tokio::task::spawn_blocking(move || Self::try_lock_owned(lock_file, &path, mode))
                    .await
                    .map_err(|err| LockFileError::TaskJoin {
                        path: lock_path.to_path_buf(),
                        reason: err.to_string(),
                    })??;

            match step {
                TryAcquireStep::Acquired(file) => return Ok(file),
                TryAcquireStep::Busy(returned_lock) => {
                    lock_file = returned_lock;
                    let now = tokio::time::Instant::now();
                    if now >= deadline {
                        return Err(LockFileError::Busy {
                            path: lock_path.to_path_buf(),
                            timeout,
                        });
                    }
                    let sleep_for =
                        MODEL_LOCK_POLL_INTERVAL.min(deadline.saturating_duration_since(now));
                    tokio::time::sleep(sleep_for).await;
                }
            }
        }
    }

    /// Acquire a shared (`LOCK_SH`) advisory lock, polling non-blockingly up to `timeout`.
    ///
    /// Fails closed on any filesystem/OS error (`ENOLCK`, `EOPNOTSUPP`, `ENOSYS`, I/O).
    pub async fn acquire_shared_timeout(
        lock_path: &Path,
        timeout: Duration,
    ) -> Result<SharedLockGuard, LockFileError> {
        let _lock_file = Self::acquire_timeout(lock_path, LockMode::Shared, timeout).await?;
        Ok(SharedLockGuard { _lock_file })
    }

    /// Acquire an exclusive (`LOCK_EX`) advisory lock, polling non-blockingly up to `timeout`.
    ///
    /// Fails closed on any filesystem/OS error (`ENOLCK`, `EOPNOTSUPP`, `ENOSYS`, I/O).
    pub async fn acquire_exclusive_timeout(
        lock_path: &Path,
        timeout: Duration,
    ) -> Result<ExclusiveLockGuard, LockFileError> {
        let _lock_file = Self::acquire_timeout(lock_path, LockMode::Exclusive, timeout).await?;
        Ok(ExclusiveLockGuard { _lock_file })
    }

    /// Attempt to acquire an exclusive (`LOCK_EX`) advisory lock without waiting.
    ///
    /// Returns `Ok(Some(guard))` if acquired, `Ok(None)` if currently held, or
    /// `Err(LockFileError::Io)` (fail-closed) on any filesystem/OS error.
    pub async fn try_acquire_exclusive(
        lock_path: &Path,
    ) -> Result<Option<ExclusiveLockGuard>, LockFileError> {
        let path = lock_path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let lock_file = Self::open_io(&path)?;
            match Self::try_lock_owned(lock_file, &path, LockMode::Exclusive)? {
                TryAcquireStep::Acquired(_lock_file) => Ok(Some(ExclusiveLockGuard { _lock_file })),
                TryAcquireStep::Busy(_) => Ok(None),
            }
        })
        .await
        .map_err(|err| LockFileError::TaskJoin {
            path: lock_path.to_path_buf(),
            reason: err.to_string(),
        })?
    }

    pub async fn with_exclusive<T, F, Fut>(lock_path: &Path, op: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut lock_file = Self::open_io(lock_path)?;
        let _lock = loop {
            if let Some(lock) = Self::try_acquire(&mut lock_file, lock_path)? {
                break lock;
            }
            tokio::time::sleep(FILE_LOCK_POLL_INTERVAL).await;
        };

        op().await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::{Mutex, oneshot};

    #[tokio::test]
    async fn test_with_exclusive_creates_parent_directories_and_lock_file() {
        let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let lock_path = temp_dir.path().join("nested/locks/model.lock");

        let result = LockFile::with_exclusive(&lock_path, || async { Ok(7usize) })
            .await
            .expect("Expected lock acquisition");

        assert_eq!(result, 7);
        assert!(lock_path.is_file());
    }

    #[tokio::test]
    async fn test_with_exclusive_serializes_concurrent_access() {
        let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let lock_path = Arc::new(temp_dir.path().join("model.lock"));
        let events = Arc::new(Mutex::new(Vec::new()));
        let (first_started_tx, first_started_rx) = oneshot::channel();
        let (release_first_tx, release_first_rx) = oneshot::channel();

        let first_lock_path = Arc::clone(&lock_path);
        let first_events = Arc::clone(&events);
        let first = tokio::spawn(async move {
            LockFile::with_exclusive(&first_lock_path, || async move {
                first_events.lock().await.push("first-start");
                first_started_tx
                    .send(())
                    .expect("Expected first-start notification");
                release_first_rx
                    .await
                    .expect("Expected release notification");
                first_events.lock().await.push("first-end");
                Ok::<(), anyhow::Error>(())
            })
            .await
        });

        first_started_rx
            .await
            .expect("Expected first lock to be acquired");

        let second_lock_path = Arc::clone(&lock_path);
        let second_events = Arc::clone(&events);
        let second = tokio::spawn(async move {
            LockFile::with_exclusive(&second_lock_path, || async move {
                second_events.lock().await.push("second");
                Ok::<(), anyhow::Error>(())
            })
            .await
        });

        tokio::time::sleep(FILE_LOCK_POLL_INTERVAL + FILE_LOCK_POLL_INTERVAL).await;
        assert_eq!(
            *events.lock().await,
            vec!["first-start"],
            "second task should wait while the first task holds the lock"
        );

        release_first_tx
            .send(())
            .expect("Expected first lock release signal");
        first
            .await
            .expect("Expected first task to join")
            .expect("Expected first lock task to succeed");
        second
            .await
            .expect("Expected second task to join")
            .expect("Expected second lock task to succeed");

        assert_eq!(
            *events.lock().await,
            vec!["first-start", "first-end", "second"]
        );
    }

    #[test]
    fn test_model_lock_path_is_deterministic_and_provider_scoped() {
        let cache_dir = Path::new("/var/cache/mx");
        let hf = LockFile::model_lock_path(cache_dir, ModelProvider::HuggingFace, "org/model")
            .expect("hf lock path");
        let hf2 = LockFile::model_lock_path(cache_dir, ModelProvider::HuggingFace, "org/model")
            .expect("hf lock path repeat");
        let ngc = LockFile::model_lock_path(cache_dir, ModelProvider::Ngc, "org/model")
            .expect("ngc lock path");
        let s3 = LockFile::model_lock_path(cache_dir, ModelProvider::S3, "s3://bucket/org/model")
            .expect("s3 lock path");
        let gcs = LockFile::model_lock_path(cache_dir, ModelProvider::Gcs, "gs://bucket/org/model")
            .expect("gcs lock path");

        assert_eq!(hf, hf2);
        assert_ne!(hf, ngc);
        assert_ne!(s3, gcs);
        for (path, prefix) in [
            (&hf, "huggingface-"),
            (&ngc, "ngc-"),
            (&s3, "s3-"),
            (&gcs, "gcs-"),
        ] {
            assert_eq!(path.parent(), Some(Path::new("/var/cache/mx/.mx/locks")));
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .expect("utf-8 lock filename");
            assert!(name.starts_with(prefix), "unexpected lock name: {name}");
            assert!(name.ends_with(".lock"), "unexpected lock name: {name}");
        }
    }

    #[tokio::test]
    async fn test_shared_locks_allow_concurrency_and_release_synchronously_on_drop() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let lock_path = LockFile::model_lock_path(
            temp_dir.path(),
            ModelProvider::HuggingFace,
            "org/shared-model",
        )
        .expect("lock path");

        let shared_a = LockFile::acquire_shared_timeout(&lock_path, Duration::from_secs(1))
            .await
            .expect("first shared lock");
        let shared_b = LockFile::acquire_shared_timeout(&lock_path, Duration::from_secs(1))
            .await
            .expect("second concurrent shared lock");

        // Exclusive try-lock must observe Busy while any shared lock is held.
        assert!(
            LockFile::try_acquire_exclusive(&lock_path)
                .await
                .expect("try_acquire_exclusive")
                .is_none()
        );

        drop(shared_a);
        assert!(
            LockFile::try_acquire_exclusive(&lock_path)
                .await
                .expect("try_acquire_exclusive after dropping one shared")
                .is_none()
        );

        // Dropping the last shared lock releases flock synchronously (no sleep needed).
        drop(shared_b);
        let exclusive = LockFile::try_acquire_exclusive(&lock_path)
            .await
            .expect("try_acquire_exclusive after all shared dropped")
            .expect("exclusive lock must succeed immediately after shared drop");

        // Shared bounded wait must time out with Busy while exclusive is held.
        let busy_err = LockFile::acquire_shared_timeout(&lock_path, Duration::from_millis(80))
            .await
            .expect_err("shared wait must time out while exclusive held");
        assert!(matches!(busy_err, LockFileError::Busy { .. }));

        drop(exclusive);

        // Exclusive release is also synchronous.
        let _shared_c = LockFile::acquire_shared_timeout(&lock_path, Duration::ZERO)
            .await
            .expect("shared lock must succeed immediately after exclusive drop");
    }

    #[tokio::test]
    async fn test_bounded_exclusive_lock_waits_for_shared_release_or_times_out() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let lock_path = LockFile::model_lock_path(temp_dir.path(), ModelProvider::S3, "s3://b/m")
            .expect("lock path");

        let shared = LockFile::acquire_shared_timeout(&lock_path, Duration::from_secs(1))
            .await
            .expect("shared lock");

        let timeout_err =
            LockFile::acquire_exclusive_timeout(&lock_path, Duration::from_millis(80))
                .await
                .expect_err("exclusive wait must time out while shared held");
        assert!(matches!(timeout_err, LockFileError::Busy { .. }));

        let path_for_task = lock_path.clone();
        let exclusive_task = tokio::spawn(async move {
            LockFile::acquire_exclusive_timeout(&path_for_task, Duration::from_secs(2)).await
        });

        tokio::time::sleep(Duration::from_millis(70)).await;
        assert!(!exclusive_task.is_finished());
        drop(shared);

        let _exclusive = exclusive_task
            .await
            .expect("join")
            .expect("exclusive lock should acquire once shared drops");
    }

    #[tokio::test]
    async fn test_lock_acquisition_fails_closed_on_io_error() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        // Place a regular file at `.mx` so creating `.mx/locks` fails with EEXIST/ENOTDIR.
        fs::write(temp_dir.path().join(".mx"), b"not-a-directory").expect("write blocker file");
        let lock_path =
            LockFile::model_lock_path(temp_dir.path(), ModelProvider::S3, "s3://b/fail-closed")
                .expect("lock path");

        assert!(matches!(
            LockFile::acquire_shared_timeout(&lock_path, Duration::from_secs(1)).await,
            Err(LockFileError::Io { .. })
        ));
        assert!(matches!(
            LockFile::acquire_exclusive_timeout(&lock_path, Duration::from_secs(1)).await,
            Err(LockFileError::Io { .. })
        ));
        assert!(matches!(
            LockFile::try_acquire_exclusive(&lock_path).await,
            Err(LockFileError::Io { .. })
        ));
    }
}
