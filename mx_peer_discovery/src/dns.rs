// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DNS and SLURM-hostlist substrate probes.
//!
//! Resolves hostnames to IP addresses via tokio's async DNS. Two entry
//! points:
//!
//! - [`resolve_hostname`]: single hostname (e.g. a Kubernetes headless
//!   Service name) -> list of IPs.
//! - [`resolve_hostlist`]: SLURM compact hostlist notation -> list of IPs
//!   (expanded through [`crate::slurm::expand_hostlist`], each name
//!   resolved, results merged and deduplicated).
//!
//! Both return ordered, deduplicated IPs. Callers are responsible for
//! dialing, connection setup, and filtering their own IPs via
//! [`filter_own_ips`].
//!
//! Behavior matches the Python half for cross-language parity.

use std::collections::HashSet;
use std::net::SocketAddr;

use tokio::net::lookup_host;
use tracing::{debug, warn};

use crate::slurm::expand_hostlist;

/// Resolve a hostname to its TCP-reachable IPv4/IPv6 addresses.
///
/// Returns an ordered, deduplicated list of IP strings. Empty on DNS
/// failure (logged at WARN and swallowed) or if the host resolves to
/// zero addresses.
pub async fn resolve_hostname(hostname: &str, port: u16) -> Vec<String> {
    let addr_str = format!("{hostname}:{port}");
    match lookup_host(addr_str.as_str()).await {
        Ok(iter) => dedup_ips(iter),
        Err(e) => {
            warn!("DNS resolution failed for {hostname}: {e}");
            Vec::new()
        }
    }
}

/// Expand a SLURM hostlist and resolve each name to IPs.
///
/// Unresolvable hostnames are logged at DEBUG and skipped. Empty if the
/// hostlist expands to no names or no hostnames resolve.
pub async fn resolve_hostlist(hostlist: &str, port: u16) -> Vec<String> {
    let hostnames = expand_hostlist(hostlist);
    if hostnames.is_empty() {
        warn!("hostlist expansion produced no hosts: {hostlist:?}");
        return Vec::new();
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut result: Vec<String> = Vec::new();
    for host in hostnames {
        let addr_str = format!("{host}:{port}");
        match lookup_host(addr_str.as_str()).await {
            Ok(iter) => {
                for addr in iter {
                    let ip = addr.ip().to_string();
                    if seen.insert(ip.clone()) {
                        result.push(ip);
                    }
                }
            }
            Err(e) => {
                debug!("hostlist: failed to resolve {host}: {e}");
            }
        }
    }
    result
}

/// Remove IPs matching our own from the list, preserving order.
///
/// The substrate layer does not introspect interfaces: the caller decides
/// what "own" means (typically the bind address plus any observed
/// external IPs).
#[must_use]
pub fn filter_own_ips(ips: Vec<String>, own: &HashSet<String>) -> Vec<String> {
    ips.into_iter().filter(|ip| !own.contains(ip)).collect()
}

/// Deduplicate IPs from a sequence of SocketAddr, preserving first-seen order.
fn dedup_ips<I: IntoIterator<Item = SocketAddr>>(addrs: I) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut result: Vec<String> = Vec::new();
    for addr in addrs {
        let ip = addr.ip().to_string();
        if seen.insert(ip.clone()) {
            result.push(ip);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    fn sa_v4(a: u8, b: u8, c: u8, d: u8, port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(a, b, c, d)), port)
    }

    fn sa_v6(segs: [u16; 8], port: u16) -> SocketAddr {
        let [a, b, c, d, e, f, g, h] = segs;
        SocketAddr::new(IpAddr::V6(Ipv6Addr::new(a, b, c, d, e, f, g, h)), port)
    }

    // -- dedup_ips: the testable slice of resolve_hostname/resolve_hostlist --

    #[test]
    fn dedup_unique() {
        let addrs = vec![
            sa_v4(10, 0, 0, 1, 4001),
            sa_v4(10, 0, 0, 2, 4001),
            sa_v4(10, 0, 0, 3, 4001),
        ];
        assert_eq!(
            dedup_ips(addrs),
            vec![
                "10.0.0.1".to_owned(),
                "10.0.0.2".to_owned(),
                "10.0.0.3".to_owned(),
            ],
        );
    }

    #[test]
    fn dedup_removes_duplicates_preserving_order() {
        let addrs = vec![
            sa_v4(10, 0, 0, 1, 4001),
            sa_v4(10, 0, 0, 2, 4001),
            sa_v4(10, 0, 0, 1, 4001),
            sa_v4(10, 0, 0, 3, 4001),
        ];
        assert_eq!(
            dedup_ips(addrs),
            vec![
                "10.0.0.1".to_owned(),
                "10.0.0.2".to_owned(),
                "10.0.0.3".to_owned(),
            ],
        );
    }

    #[test]
    fn dedup_empty() {
        let addrs: Vec<SocketAddr> = Vec::new();
        let empty: Vec<String> = Vec::new();
        assert_eq!(dedup_ips(addrs), empty);
    }

    #[test]
    fn dedup_mixed_v4_v6() {
        let addrs = vec![
            sa_v4(10, 0, 0, 1, 4001),
            sa_v6([0xfe80, 0, 0, 0, 0, 0, 0, 1], 4001),
            sa_v4(10, 0, 0, 1, 4001),
        ];
        assert_eq!(
            dedup_ips(addrs),
            vec!["10.0.0.1".to_owned(), "fe80::1".to_owned()],
        );
    }

    #[test]
    fn dedup_different_ports_same_ip_dedupes_to_one() {
        let addrs = vec![sa_v4(10, 0, 0, 1, 4001), sa_v4(10, 0, 0, 1, 4002)];
        assert_eq!(dedup_ips(addrs), vec!["10.0.0.1".to_owned()]);
    }

    // -- filter_own_ips --

    #[test]
    fn filter_removes_matching() {
        let ips = vec![
            "10.0.0.1".to_owned(),
            "10.0.0.2".to_owned(),
            "10.0.0.3".to_owned(),
        ];
        let own: HashSet<String> = ["10.0.0.2".to_owned()].into_iter().collect();
        assert_eq!(
            filter_own_ips(ips, &own),
            vec!["10.0.0.1".to_owned(), "10.0.0.3".to_owned()],
        );
    }

    #[test]
    fn filter_preserves_order() {
        let ips = vec![
            "10.0.0.3".to_owned(),
            "10.0.0.1".to_owned(),
            "10.0.0.2".to_owned(),
        ];
        let own: HashSet<String> = ["10.0.0.1".to_owned()].into_iter().collect();
        assert_eq!(
            filter_own_ips(ips, &own),
            vec!["10.0.0.3".to_owned(), "10.0.0.2".to_owned()],
        );
    }

    #[test]
    fn filter_empty_own_set() {
        let ips = vec!["10.0.0.1".to_owned(), "10.0.0.2".to_owned()];
        let own: HashSet<String> = HashSet::new();
        assert_eq!(filter_own_ips(ips.clone(), &own), ips);
    }

    #[test]
    fn filter_empty_list() {
        let ips: Vec<String> = Vec::new();
        let own: HashSet<String> = ["10.0.0.1".to_owned()].into_iter().collect();
        let empty: Vec<String> = Vec::new();
        assert_eq!(filter_own_ips(ips, &own), empty);
    }

    #[test]
    fn filter_all_self() {
        let ips = vec!["10.0.0.1".to_owned(), "10.0.0.2".to_owned()];
        let own: HashSet<String> = ips.iter().cloned().collect();
        let empty: Vec<String> = Vec::new();
        assert_eq!(filter_own_ips(ips, &own), empty);
    }

    // -- resolve_hostname: live loopback probe --
    //
    // tokio::net::lookup_host cannot be trait-mocked without a wrapper;
    // rather than adding that abstraction for v1, we rely on loopback
    // resolution (guaranteed by any POSIX resolver) as a live integration
    // test, and on dedup_ips tests for the dedup/order logic. Python-side
    // unit tests exercise the error-handling path via mock.patch.

    #[tokio::test]
    async fn resolve_hostname_localhost_returns_loopback() {
        let ips = resolve_hostname("localhost", 80).await;
        assert!(!ips.is_empty(), "localhost must resolve to at least one IP",);
        // At least one result must be loopback (127.0.0.1 or ::1).
        let has_loopback = ips.iter().any(|ip| ip == "127.0.0.1" || ip == "::1");
        assert!(has_loopback, "expected loopback, got {ips:?}");
    }

    #[tokio::test]
    async fn resolve_hostname_unresolvable_returns_empty() {
        // .invalid is reserved by RFC 6761 to never resolve.
        let ips = resolve_hostname("nonexistent.invalid", 80).await;
        assert!(ips.is_empty(), "expected empty, got {ips:?}");
    }

    #[tokio::test]
    async fn resolve_hostlist_empty_input_returns_empty() {
        let ips = resolve_hostlist("", 4001).await;
        assert!(ips.is_empty());
    }

    #[tokio::test]
    async fn resolve_hostlist_localhost_single() {
        // Single-entry hostlist with a name that resolves.
        let ips = resolve_hostlist("localhost", 80).await;
        assert!(!ips.is_empty());
    }
}
