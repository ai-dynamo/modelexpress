// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, Read};

/// Read JSON payload from various sources (inline, file, or stdin)
pub fn read_payload(
    payload: Option<String>,
    payload_file: Option<String>,
) -> Result<Option<HashMap<String, Value>>, Box<dyn std::error::Error>> {
    let json_str = match (payload, payload_file) {
        (Some(p), None) => {
            if p == "-" {
                // Read from stdin
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                buffer
            } else {
                p
            }
        }
        (None, Some(f)) => std::fs::read_to_string(f)?,
        (Some(_), Some(_)) => {
            return Err("Cannot specify both --payload and --payload-file".into());
        }
        (None, None) => return Ok(None),
    };

    let payload: HashMap<String, Value> = serde_json::from_str(&json_str)?;
    Ok(Some(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_read_payload_json() {
        let payload = r#"{"key": "value", "number": 42}"#;
        let result = read_payload(Some(payload.to_string()), None)
            .expect("Should be able to parse valid JSON payload");
        assert!(result.is_some());
        let map = result.expect("Result should contain a payload map");
        assert_eq!(
            map.get("key").expect("Map should contain 'key' field"),
            "value"
        );
        assert_eq!(
            map.get("number")
                .expect("Map should contain 'number' field"),
            42
        );
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_read_payload_empty() {
        let result = read_payload(None, None).expect("Should succeed when no payload is provided");
        assert!(result.is_none());
    }

    #[test]
    fn test_read_payload_conflict() {
        let result = read_payload(Some("{}".to_string()), Some("file.json".to_string()));
        assert!(result.is_err());
    }
}
