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
    fn test_read_payload_json() {
        let payload = r#"{"key": "value", "number": 42}"#;
        let result = read_payload(Some(payload.to_string()), None).unwrap();
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map.get("key").unwrap(), "value");
        assert_eq!(map.get("number").unwrap(), 42);
    }

    #[test]
    fn test_read_payload_empty() {
        let result = read_payload(None, None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_read_payload_conflict() {
        let result = read_payload(Some("{}".to_string()), Some("file.json".to_string()));
        assert!(result.is_err());
    }
}
