// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use fd_lock::{RwLock as FileRwLock, RwLockWriteGuard as FileWriteGuard};
use futures::Future;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

const FILE_LOCK_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub struct LockFile;

impl LockFile {
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

    fn open(path: &Path) -> Result<FileRwLock<fs::File>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create lock directory '{}'", parent.display())
            })?;
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("Failed to create lock file '{}'", path.display()))?;

        Ok(FileRwLock::new(file))
    }

    pub async fn with_exclusive<T, F, Fut>(lock_path: &Path, op: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut lock_file = Self::open(lock_path)?;
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
}
