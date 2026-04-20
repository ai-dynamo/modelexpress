// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multicast DNS service discovery.
//!
//! Thin wrapper around the [`mdns-sd`](https://crates.io/crates/mdns-sd) crate
//! that provides a single peer-oriented abstraction: register our own service
//! once, continuously browse for peers of the same service type, and emit a
//! callback for each resolved peer.
//!
//! Default service type: `_mx-peer._tcp.local.` (note the trailing dot -
//! mdns-sd requires it).
//!
//! # Interop with the Python half
//!
//! The Python companion module encodes host and port inside TXT records
//! (keys `host` and `port`), a libp2p-historical convention. This Rust
//! implementation uses standard RFC 6763 SRV+A records for host and port,
//! and TXT purely for user metadata. Cross-language discovery between
//! Python and Rust peers therefore requires the Python side to be
//! updated to emit and parse SRV+A records. Within a single language,
//! discovery works out of the box.

use std::collections::HashMap;
use std::sync::Arc;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

/// Default service type for MX peers. The trailing dot is required by
/// [`mdns_sd`].
pub const DEFAULT_SERVICE_TYPE: &str = "_mx-peer._tcp.local.";

/// Callback invoked for each resolved peer.
///
/// Arguments: `(instance_name, port, addresses, txt_properties)`. The
/// `addresses` set contains every IP the peer advertises; callers pick
/// one (or iterate) according to their routing preferences.
pub type OnResolved =
    Arc<dyn Fn(String, u16, Vec<String>, HashMap<String, String>) + Send + Sync + 'static>;

/// A live mDNS presence: advertises one service and browses for peers.
///
/// Drop the value to stop the daemon (best-effort shutdown).
pub struct MdnsDiscovery {
    daemon: ServiceDaemon,
    _browser: JoinHandle<()>,
}

/// Configuration for [`MdnsDiscovery::start`].
///
/// All fields are required; there's no builder yet because the surface is
/// small enough to pass directly.
pub struct Config {
    /// Service type (must end with `.local.`). Defaults to
    /// [`DEFAULT_SERVICE_TYPE`] when `None`.
    pub service_type: Option<String>,
    /// Instance name (the label prefix on the service-type FQDN).
    /// Defaults to a random alphanumeric string when `None`.
    pub instance_name: Option<String>,
    /// Hostname to register in the SRV record (typically `"<instance>.local."`).
    pub hostname: String,
    /// IP addresses this peer advertises. Comma-separated string or a
    /// single IP literal: see mdns-sd's `AsIpAddrs` for supported forms.
    pub ip: String,
    /// Port this peer listens on.
    pub port: u16,
    /// TXT properties for metadata. The mDNS wire spec limits each
    /// `key=value` to 255 bytes.
    pub txt: HashMap<String, String>,
    /// Invoked for each resolved peer (excluding ourselves).
    pub on_resolved: OnResolved,
}

impl MdnsDiscovery {
    /// Start the daemon, register our service, and begin browsing for peers.
    ///
    /// The returned value must be kept alive for discovery to continue;
    /// dropping it shuts the daemon down.
    ///
    /// # Errors
    ///
    /// Returns [`mdns_sd::Error`] if the daemon cannot be created, the
    /// service info is invalid, registration fails, or browsing cannot
    /// start.
    pub fn start(config: Config) -> Result<Self, mdns_sd::Error> {
        let service_type = config
            .service_type
            .unwrap_or_else(|| DEFAULT_SERVICE_TYPE.to_owned());
        let instance_name = config.instance_name.unwrap_or_else(random_instance_name);

        let daemon = ServiceDaemon::new()?;

        let service_info = ServiceInfo::new(
            &service_type,
            &instance_name,
            &config.hostname,
            config.ip.as_str(),
            config.port,
            config.txt,
        )?;
        daemon.register(service_info)?;

        let receiver = daemon.browse(&service_type)?;
        let on_resolved = config.on_resolved;
        let self_instance = instance_name;
        let browser = tokio::spawn(async move {
            while let Ok(event) = receiver.recv_async().await {
                match event {
                    ServiceEvent::ServiceResolved(resolved) => {
                        let fullname = resolved.get_fullname();
                        if is_own_instance(fullname, &self_instance) {
                            continue;
                        }
                        let addresses: Vec<String> = resolved
                            .get_addresses()
                            .iter()
                            .map(|scoped| scoped.to_string())
                            .collect();
                        let port = resolved.get_port();
                        let txt: HashMap<String, String> = resolved
                            .get_properties()
                            .iter()
                            .map(|p| (p.key().to_owned(), p.val_str().to_owned()))
                            .collect();
                        let instance_label = instance_label_from_fullname(fullname, &service_type)
                            .unwrap_or_else(|| fullname.to_owned());
                        on_resolved(instance_label, port, addresses, txt);
                    }
                    ServiceEvent::ServiceRemoved(ty, fullname) => {
                        debug!("service removed: {fullname} (type {ty})");
                    }
                    other => {
                        debug!("mdns event: {other:?}");
                    }
                }
            }
            debug!("mdns browser channel closed");
        });

        Ok(Self {
            daemon,
            _browser: browser,
        })
    }
}

impl Drop for MdnsDiscovery {
    fn drop(&mut self) {
        if let Err(e) = self.daemon.shutdown() {
            warn!("mdns daemon shutdown failed: {e}");
        }
    }
}

/// Returns true when the `fullname` belongs to our own registered instance.
///
/// Fullnames are formatted `{instance}.{service_type}`. A naive
/// `starts_with` check on a bare instance label is enough because
/// mdns-sd ensures the separator between the label and the service type
/// is always a `.`.
fn is_own_instance(fullname: &str, instance: &str) -> bool {
    let prefix = format!("{instance}.");
    fullname.starts_with(&prefix)
}

/// Strip the service-type suffix to recover the bare instance label.
///
/// Returns `None` when the fullname does not end with the service type.
fn instance_label_from_fullname(fullname: &str, service_type: &str) -> Option<String> {
    let suffix = format!(".{service_type}");
    fullname.strip_suffix(suffix.as_str()).map(str::to_owned)
}

/// Generate a 32-character lowercase alphanumeric instance name.
///
/// Not cryptographically random: we only need uniqueness on a LAN, and the
/// SystemTime + process-id mix is good enough for that. No external `rand`
/// dep.
fn random_instance_name() -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    const LEN: usize = 32;
    const ALPHABET_LEN: u64 = 36;

    let mut state: u64 = seed_state();
    let mut out = String::with_capacity(LEN);
    for _ in 0..LEN {
        state = splitmix64(state);
        let idx = (state.checked_rem(ALPHABET_LEN).unwrap_or(0)) as usize;
        let byte = *ALPHABET.get(idx).unwrap_or(&b'0');
        out.push(byte as char);
    }
    out
}

/// Seed the PRNG from wall-clock nanos XOR a decorrelated process id.
fn seed_state() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = u64::from(std::process::id());
    nanos ^ pid.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// SplitMix64: one step of the SplitMix64 PRNG.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_service_type_constant() {
        assert_eq!(DEFAULT_SERVICE_TYPE, "_mx-peer._tcp.local.");
    }

    #[test]
    fn random_instance_name_shape() {
        let name = random_instance_name();
        assert_eq!(name.len(), 32);
        assert!(
            name.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
            "expected lowercase alphanumeric, got {name:?}",
        );
    }

    #[test]
    fn random_instance_name_varies() {
        // Two draws from a 36^32 space. Collision probability is vanishingly
        // small; if this ever trips, the RNG seed is broken.
        let a = random_instance_name();
        std::thread::sleep(std::time::Duration::from_nanos(1));
        let b = random_instance_name();
        assert_ne!(a, b, "two draws produced identical values");
    }

    #[test]
    fn is_own_instance_matches_prefix() {
        assert!(is_own_instance(
            "myinstance._mx-peer._tcp.local.",
            "myinstance"
        ));
    }

    #[test]
    fn is_own_instance_rejects_different() {
        assert!(!is_own_instance(
            "otherinstance._mx-peer._tcp.local.",
            "myinstance"
        ));
    }

    #[test]
    fn is_own_instance_rejects_partial_prefix() {
        // "myinstance" is a prefix of "myinstance2" but not with the dot.
        assert!(!is_own_instance(
            "myinstance2._mx-peer._tcp.local.",
            "myinstance",
        ));
    }

    #[test]
    fn instance_label_extracts_from_fullname() {
        let label =
            instance_label_from_fullname("myinstance._mx-peer._tcp.local.", "_mx-peer._tcp.local.");
        assert_eq!(label, Some("myinstance".to_owned()));
    }

    #[test]
    fn instance_label_returns_none_on_wrong_suffix() {
        let label =
            instance_label_from_fullname("myinstance._other._tcp.local.", "_mx-peer._tcp.local.");
        assert_eq!(label, None);
    }

    #[test]
    fn splitmix64_changes_state() {
        assert_ne!(splitmix64(0), 0);
        assert_ne!(splitmix64(1), splitmix64(2));
    }

    // No end-to-end test of MdnsDiscovery::start - it binds to 224.0.0.251
    // which may not be available in sandboxed CI environments. Wire-format
    // correctness is covered by mdns-sd's own test suite.
}
