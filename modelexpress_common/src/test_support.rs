// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::sync::{LazyLock, Mutex, MutexGuard};
use tracing::warn;

static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Serialize access to process environment mutations across tests.
pub fn acquire_env_mutex() -> MutexGuard<'static, ()> {
    match ENV_MUTEX.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!("ENV_MUTEX was poisoned; recovering lock");
            poisoned.into_inner()
        }
    }
}

/// Restore a process environment variable to its previous value when dropped.
#[must_use]
pub struct EnvVarGuard<'a> {
    key: String,
    previous: Option<String>,
    _env_lock: &'a MutexGuard<'static, ()>,
}

impl<'a> EnvVarGuard<'a> {
    /// Set an environment variable while holding `env_lock` and restore it on drop.
    pub fn set(env_lock: &'a MutexGuard<'static, ()>, key: &str, value: &str) -> Self {
        let previous = env::var(key).ok();
        unsafe {
            env::set_var(key, value);
        }
        Self {
            key: key.to_string(),
            previous,
            _env_lock: env_lock,
        }
    }

    /// Remove an environment variable while holding `env_lock` and restore it on drop.
    pub fn remove(env_lock: &'a MutexGuard<'static, ()>, key: &str) -> Self {
        let previous = env::var(key).ok();
        unsafe {
            env::remove_var(key);
        }
        Self {
            key: key.to_string(),
            previous,
            _env_lock: env_lock,
        }
    }
}

impl Drop for EnvVarGuard<'_> {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(value) => env::set_var(&self.key, value),
                None => env::remove_var(&self.key),
            }
        }
    }
}
