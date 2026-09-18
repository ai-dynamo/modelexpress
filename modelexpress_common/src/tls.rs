// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TLS settings.

use serde::{Deserialize, Serialize};

/// Minimum TLS protocol version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TlsVersion {
    Tls10,
    Tls11,
    Tls12,
    Tls13,
}

impl TlsVersion {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tls10 => "TLS1.0",
            Self::Tls11 => "TLS1.1",
            Self::Tls12 => "TLS1.2",
            Self::Tls13 => "TLS1.3",
        }
    }
}

impl Serialize for TlsVersion {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for TlsVersion {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for TlsVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TlsVersion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_ascii_lowercase();
        let digits = normalized
            .strip_prefix("versiontls")
            .or_else(|| normalized.strip_prefix("tlsv"))
            .or_else(|| normalized.strip_prefix("tls"))
            .unwrap_or(&normalized)
            .replace(['.', '_'], "");
        match digits.as_str() {
            "10" => Ok(Self::Tls10),
            "11" => Ok(Self::Tls11),
            "12" => Ok(Self::Tls12),
            "13" => Ok(Self::Tls13),
            _ => Err(format!(
                "unknown TLS version '{s}' (expected TLS1.0, TLS1.1, TLS1.2 or TLS1.3)"
            )),
        }
    }
}

/// Split a mixed cipher list into what OpenSSL configures through
/// `set_cipher_list` (TLS 1.2 and below) and `set_ciphersuites` (TLS 1.3).
#[must_use]
pub fn split_cipher_suites(names: &[String]) -> (Vec<String>, Vec<String>) {
    names
        .iter()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .partition(|name| !name.starts_with("TLS_"))
}

#[cfg(test)]
mod tests {
    use crate::tls::{TlsVersion, split_cipher_suites};

    #[test]
    fn parses_flag_and_version_prefixed_spellings() {
        for (input, expected) in [
            ("TLS1.2", TlsVersion::Tls12),
            ("tls1.3", TlsVersion::Tls13),
            ("VersionTLS12", TlsVersion::Tls12),
            ("VersionTLS13", TlsVersion::Tls13),
            ("VersionTLS10", TlsVersion::Tls10),
            ("VersionTLS11", TlsVersion::Tls11),
            ("TLSv1.2", TlsVersion::Tls12),
            ("1.3", TlsVersion::Tls13),
            (" TLS1.2 ", TlsVersion::Tls12),
        ] {
            assert_eq!(input.parse::<TlsVersion>(), Ok(expected), "{input}");
        }
    }

    #[test]
    fn rejects_unknown_versions() {
        for input in ["TLS1.4", "SSL3", "", "VersionTLS", "1.2.3"] {
            assert!(input.parse::<TlsVersion>().is_err(), "{input}");
        }
    }

    #[test]
    fn display_round_trips_through_from_str() {
        for version in [
            TlsVersion::Tls10,
            TlsVersion::Tls11,
            TlsVersion::Tls12,
            TlsVersion::Tls13,
        ] {
            assert_eq!(version.to_string().parse::<TlsVersion>(), Ok(version));
        }
    }

    #[test]
    fn versions_order_by_strength() {
        assert!(TlsVersion::Tls10 < TlsVersion::Tls11);
        assert!(TlsVersion::Tls11 < TlsVersion::Tls12);
        assert!(TlsVersion::Tls12 < TlsVersion::Tls13);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn serde_uses_the_flag_spelling() {
        let json = serde_json::to_string(&TlsVersion::Tls13).expect("serialize");
        assert_eq!(json, "\"TLS1.3\"");
        let parsed: TlsVersion = serde_json::from_str("\"TLS1.2\"").expect("deserialize");
        assert_eq!(parsed, TlsVersion::Tls12);
    }

    #[test]
    fn splits_intermediate_profile_cipher_list() {
        let profile: Vec<String> = [
            "TLS_AES_128_GCM_SHA256",
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "ECDHE-ECDSA-AES128-GCM-SHA256",
            "ECDHE-RSA-AES128-GCM-SHA256",
            "",
            " ECDHE-ECDSA-CHACHA20-POLY1305 ",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let (tls12, tls13) = split_cipher_suites(&profile);
        assert_eq!(
            tls12,
            [
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
                "ECDHE-ECDSA-CHACHA20-POLY1305",
            ]
        );
        assert_eq!(
            tls13,
            [
                "TLS_AES_128_GCM_SHA256",
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
            ]
        );
    }

    #[test]
    fn split_of_empty_list_is_empty() {
        let (tls12, tls13) = split_cipher_suites(&[]);
        assert!(tls12.is_empty());
        assert!(tls13.is_empty());
    }
}
