// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static peer-list substrate.
//!
//! For deployments without any discovery substrate (no Kubernetes API, no
//! SLURM, no mDNS reachability), peers can be listed explicitly via
//! configuration.
//!
//! Input format: comma-separated `host:port` entries. IPv6 addresses must
//! be bracketed: `[::1]:4001`. Whitespace around entries is ignored.
//! Malformed entries are logged at WARN and skipped (the rest still
//! parse), so a single typo doesn't blank the whole list.
//!
//! Wire-compatible with the companion Python module (`static`, renamed
//! here because `static` is a Rust keyword).

use std::env;
use tracing::warn;

/// Default environment variable name for
/// [`endpoints_from_env`]-style callers.
pub const DEFAULT_ENV_VAR: &str = "MX_PEER_ENDPOINTS";

/// Parse a comma-separated endpoint string into `(host, port)` tuples.
///
/// Malformed entries are logged at WARN and skipped; the rest still parse.
#[must_use]
pub fn parse_endpoints(value: &str) -> Vec<(String, u16)> {
    if value.trim().is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for entry in value.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        if let Some(parsed) = parse_one_endpoint(entry) {
            result.push(parsed);
        }
    }
    result
}

/// Read an environment variable and parse it as an endpoint list.
///
/// Returns an empty vector if the variable is unset or empty.
#[must_use]
pub fn endpoints_from_env(env_var: &str) -> Vec<(String, u16)> {
    parse_endpoints(&env::var(env_var).unwrap_or_default())
}

/// Parse a single `host:port` entry. IPv6 must be bracketed.
///
/// Returns `None` on malformed input (with a WARN log line).
fn parse_one_endpoint(entry: &str) -> Option<(String, u16)> {
    let (host, port_str) = if let Some(after_open) = entry.strip_prefix('[') {
        let Some(close_rel) = after_open.find(']') else {
            warn!("malformed IPv6 endpoint (missing ']'): {entry:?}");
            return None;
        };
        let host = after_open.get(..close_rel).unwrap_or("");
        let after_close = after_open.get(close_rel.saturating_add(1)..).unwrap_or("");
        let Some(port_str) = after_close.strip_prefix(':') else {
            warn!("missing ':port' after bracketed host: {entry:?}");
            return None;
        };
        (host, port_str)
    } else {
        let Some((host, port_str)) = entry.rsplit_once(':') else {
            warn!("endpoint missing ':port': {entry:?}");
            return None;
        };
        (host, port_str)
    };

    if host.is_empty() {
        warn!("endpoint missing host: {entry:?}");
        return None;
    }

    let port: u16 = match port_str.parse() {
        Ok(p) => p,
        Err(_) => {
            warn!("invalid or out-of-range port in endpoint: {entry:?}");
            return None;
        }
    };
    if port == 0 {
        warn!("port out of range in endpoint: {entry:?}");
        return None;
    }

    Some((host.to_owned(), port))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_endpoints: basic --

    #[test]
    fn parse_single_endpoint() {
        assert_eq!(
            parse_endpoints("10.0.0.1:4001"),
            vec![("10.0.0.1".to_owned(), 4001)]
        );
    }

    #[test]
    fn parse_multiple_endpoints() {
        assert_eq!(
            parse_endpoints("10.0.0.1:4001,10.0.0.2:4002,10.0.0.3:4003"),
            vec![
                ("10.0.0.1".to_owned(), 4001),
                ("10.0.0.2".to_owned(), 4002),
                ("10.0.0.3".to_owned(), 4003),
            ],
        );
    }

    #[test]
    fn parse_hostname_not_ip() {
        assert_eq!(
            parse_endpoints("worker-01.cluster.local:4001"),
            vec![("worker-01.cluster.local".to_owned(), 4001)]
        );
    }

    #[test]
    fn parse_preserves_order() {
        assert_eq!(
            parse_endpoints("c:1,a:2,b:3"),
            vec![
                ("c".to_owned(), 1),
                ("a".to_owned(), 2),
                ("b".to_owned(), 3),
            ],
        );
    }

    // -- parse_endpoints: whitespace --

    #[test]
    fn parse_whitespace_around_entries() {
        assert_eq!(
            parse_endpoints("  10.0.0.1:4001  ,  10.0.0.2:4002  "),
            vec![("10.0.0.1".to_owned(), 4001), ("10.0.0.2".to_owned(), 4002),],
        );
    }

    #[test]
    fn parse_empty_string() {
        let empty: Vec<(String, u16)> = Vec::new();
        assert_eq!(parse_endpoints(""), empty);
    }

    #[test]
    fn parse_whitespace_only() {
        let empty: Vec<(String, u16)> = Vec::new();
        assert_eq!(parse_endpoints("   "), empty);
    }

    #[test]
    fn parse_trailing_comma() {
        assert_eq!(
            parse_endpoints("10.0.0.1:4001,"),
            vec![("10.0.0.1".to_owned(), 4001)]
        );
    }

    #[test]
    fn parse_double_comma() {
        assert_eq!(
            parse_endpoints("10.0.0.1:4001,,10.0.0.2:4002"),
            vec![("10.0.0.1".to_owned(), 4001), ("10.0.0.2".to_owned(), 4002),],
        );
    }

    // -- parse_endpoints: IPv6 --

    #[test]
    fn parse_ipv6_bracketed() {
        assert_eq!(
            parse_endpoints("[::1]:4001"),
            vec![("::1".to_owned(), 4001)]
        );
    }

    #[test]
    fn parse_ipv6_full_address() {
        assert_eq!(
            parse_endpoints("[fe80::1234:5678]:4001"),
            vec![("fe80::1234:5678".to_owned(), 4001)]
        );
    }

    #[test]
    fn parse_ipv6_mixed_with_ipv4() {
        assert_eq!(
            parse_endpoints("10.0.0.1:4001,[::1]:4002"),
            vec![("10.0.0.1".to_owned(), 4001), ("::1".to_owned(), 4002),],
        );
    }

    // -- parse_endpoints: malformed entries skipped --

    #[test]
    fn parse_missing_port_skipped() {
        assert_eq!(
            parse_endpoints("10.0.0.1,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_non_numeric_port_skipped() {
        assert_eq!(
            parse_endpoints("10.0.0.1:abc,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_port_zero_skipped() {
        assert_eq!(
            parse_endpoints("10.0.0.1:0,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_port_too_high_skipped() {
        // 65536 exceeds u16::MAX, parse fails
        assert_eq!(
            parse_endpoints("10.0.0.1:65536,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_negative_port_skipped() {
        assert_eq!(
            parse_endpoints("10.0.0.1:-1,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_ipv6_missing_close_bracket() {
        assert_eq!(
            parse_endpoints("[::1:4001,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_ipv6_missing_port() {
        assert_eq!(
            parse_endpoints("[::1],10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_missing_host() {
        assert_eq!(
            parse_endpoints(":4001,10.0.0.2:4002"),
            vec![("10.0.0.2".to_owned(), 4002)],
        );
    }

    #[test]
    fn parse_all_malformed_returns_empty() {
        let empty: Vec<(String, u16)> = Vec::new();
        assert_eq!(parse_endpoints("nope,also-nope,:4001"), empty);
    }

    #[test]
    fn parse_port_boundary_values() {
        assert_eq!(
            parse_endpoints("10.0.0.1:1,10.0.0.2:65535"),
            vec![("10.0.0.1".to_owned(), 1), ("10.0.0.2".to_owned(), 65535),],
        );
    }

    // -- endpoints_from_env --
    //
    // Tests use unique env var names per case since std::env is global and
    // tests may run in parallel. SAFETY: std::env::set_var/remove_var are
    // unsafe in Rust 2024 edition.

    #[test]
    fn env_reads_custom_var() {
        unsafe {
            env::set_var("MX_PEER_DISCOVERY_TEST_VAR_1", "host1:1234");
        }
        let result = endpoints_from_env("MX_PEER_DISCOVERY_TEST_VAR_1");
        unsafe {
            env::remove_var("MX_PEER_DISCOVERY_TEST_VAR_1");
        }
        assert_eq!(result, vec![("host1".to_owned(), 1234)]);
    }

    #[test]
    fn env_unset_returns_empty() {
        let empty: Vec<(String, u16)> = Vec::new();
        // Use a name extremely unlikely to be set in the environment.
        assert_eq!(
            endpoints_from_env("MX_PEER_DISCOVERY_TEST_VAR_DEFINITELY_UNSET"),
            empty
        );
    }

    #[test]
    fn env_empty_returns_empty() {
        unsafe {
            env::set_var("MX_PEER_DISCOVERY_TEST_VAR_EMPTY", "");
        }
        let result = endpoints_from_env("MX_PEER_DISCOVERY_TEST_VAR_EMPTY");
        unsafe {
            env::remove_var("MX_PEER_DISCOVERY_TEST_VAR_EMPTY");
        }
        let empty: Vec<(String, u16)> = Vec::new();
        assert_eq!(result, empty);
    }

    #[test]
    fn env_reads_multiple_endpoints() {
        unsafe {
            env::set_var(
                "MX_PEER_DISCOVERY_TEST_VAR_MULTI",
                "10.0.0.1:4001,10.0.0.2:4002",
            );
        }
        let result = endpoints_from_env("MX_PEER_DISCOVERY_TEST_VAR_MULTI");
        unsafe {
            env::remove_var("MX_PEER_DISCOVERY_TEST_VAR_MULTI");
        }
        assert_eq!(
            result,
            vec![("10.0.0.1".to_owned(), 4001), ("10.0.0.2".to_owned(), 4002),],
        );
    }

    #[test]
    fn default_env_var_name() {
        assert_eq!(DEFAULT_ENV_VAR, "MX_PEER_ENDPOINTS");
    }
}
