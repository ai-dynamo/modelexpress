// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SLURM compact hostlist expansion.
//!
//! Parses SLURM compact hostlist notation into individual hostnames. This is
//! the same syntax `scontrol show hostnames` emits and `SLURM_JOB_NODELIST`
//! carries.
//!
//! Supported syntax:
//! - Simple list: `host1,host2,host3`
//! - Ranges: `node[01-04]` -> `node01, node02, node03, node04`
//! - Bracket lists: `gpu[1,3,5]` -> `gpu1, gpu3, gpu5`
//! - Mixed: `gpu[1-3,5]` -> `gpu1, gpu2, gpu3, gpu5`
//! - Multiple bracket groups: `rack[1-2]-node[01-04]` -> cartesian product
//! - Zero-padding preserved from range endpoints

/// Expand a SLURM compact hostlist into individual hostnames.
///
/// Returns an empty vector for empty input.
#[must_use]
pub fn expand_hostlist(hostlist: &str) -> Vec<String> {
    if hostlist.trim().is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for entry in split_top_level(hostlist) {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let segments = parse_segments(entry);
        for combo in cartesian_product(&segments) {
            result.push(combo);
        }
    }
    result
}

/// Split a hostlist string on commas that are NOT inside brackets.
fn split_top_level(hostlist: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0usize;
    for (i, ch) in hostlist.char_indices() {
        match ch {
            '[' => depth = depth.saturating_add(1),
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(hostlist.get(start..i).unwrap_or(""));
                start = i.saturating_add(1);
            }
            _ => {}
        }
    }
    parts.push(hostlist.get(start..).unwrap_or(""));
    parts
}

/// Parse a single hostlist entry into segments for cartesian product.
///
/// Each segment is a list of strings. Literal text becomes a single-element
/// list; bracket expressions expand to multiple elements.
fn parse_segments(entry: &str) -> Vec<Vec<String>> {
    let mut segments: Vec<Vec<String>> = Vec::new();
    let mut remaining = entry;
    loop {
        match remaining.find('[') {
            None => {
                if !remaining.is_empty() {
                    segments.push(vec![remaining.to_owned()]);
                }
                break;
            }
            Some(bracket_start) => {
                if bracket_start > 0 {
                    let literal = remaining.get(..bracket_start).unwrap_or("");
                    segments.push(vec![literal.to_owned()]);
                }
                let after_open = remaining
                    .get(bracket_start.saturating_add(1)..)
                    .unwrap_or("");
                match after_open.find(']') {
                    None => {
                        // Malformed: treat rest as literal (matches Python behavior)
                        let rest = remaining.get(bracket_start..).unwrap_or("");
                        segments.push(vec![rest.to_owned()]);
                        break;
                    }
                    Some(close_rel) => {
                        let content = after_open.get(..close_rel).unwrap_or("");
                        segments.push(expand_bracket(content));
                        remaining = after_open.get(close_rel.saturating_add(1)..).unwrap_or("");
                    }
                }
            }
        }
    }
    segments
}

/// Expand a bracket expression like `1-3,5,8-9` into individual strings.
fn expand_bracket(content: &str) -> Vec<String> {
    let mut result = Vec::new();
    for part in content.split(',') {
        let part = part.trim();
        if let Some(range) = parse_numeric_range(part) {
            result.extend(range);
        } else {
            result.push(part.to_owned());
        }
    }
    result
}

/// Parse a `"N-M"` string where both sides are non-negative integers with
/// no sign prefix. Returns the expanded list with zero-padding preserved
/// from the wider endpoint, or `None` if the input is not a valid range.
fn parse_numeric_range(s: &str) -> Option<Vec<String>> {
    let (start_str, end_str) = s.split_once('-')?;
    if start_str.is_empty() || end_str.is_empty() {
        return None;
    }
    if !start_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if !end_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let start_val: u64 = start_str.parse().ok()?;
    let end_val: u64 = end_str.parse().ok()?;
    if end_val < start_val {
        return None;
    }
    let width = start_str.len().max(end_str.len());
    Some(
        (start_val..=end_val)
            .map(|v| format!("{v:0width$}"))
            .collect(),
    )
}

/// Cartesian product: given a list of segments, produce all concatenations.
fn cartesian_product(segments: &[Vec<String>]) -> Vec<String> {
    let mut result: Vec<String> = vec![String::new()];
    for segment in segments {
        let mut next: Vec<String> = Vec::new();
        for prefix in &result {
            for s in segment {
                let mut combined = String::new();
                combined.push_str(prefix);
                combined.push_str(s);
                next.push(combined);
            }
        }
        result = next;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_list() {
        assert_eq!(expand_hostlist("host1,host2"), vec!["host1", "host2"]);
    }

    #[test]
    fn simple_list_three() {
        assert_eq!(
            expand_hostlist("a,b,c"),
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
        );
    }

    #[test]
    fn range() {
        assert_eq!(
            expand_hostlist("node[01-04]"),
            vec!["node01", "node02", "node03", "node04"],
        );
    }

    #[test]
    fn range_no_padding() {
        assert_eq!(
            expand_hostlist("node[1-4]"),
            vec!["node1", "node2", "node3", "node4"],
        );
    }

    #[test]
    fn bracket_list() {
        assert_eq!(expand_hostlist("gpu[1,3,5]"), vec!["gpu1", "gpu3", "gpu5"],);
    }

    #[test]
    fn mixed_range_list() {
        assert_eq!(
            expand_hostlist("gpu[1-3,5]"),
            vec!["gpu1", "gpu2", "gpu3", "gpu5"],
        );
    }

    #[test]
    fn multiple_brackets() {
        assert_eq!(
            expand_hostlist("rack[1-2]-node[01-02]"),
            vec![
                "rack1-node01",
                "rack1-node02",
                "rack2-node01",
                "rack2-node02",
            ],
        );
    }

    #[test]
    fn no_brackets() {
        assert_eq!(expand_hostlist("plain-host"), vec!["plain-host"]);
    }

    #[test]
    fn empty() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(expand_hostlist(""), empty);
    }

    #[test]
    fn whitespace_only() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(expand_hostlist("   "), empty);
    }

    #[test]
    fn comma_with_brackets() {
        assert_eq!(expand_hostlist("a,b[1-2],c"), vec!["a", "b1", "b2", "c"],);
    }

    #[test]
    fn zero_padding_preserved() {
        assert_eq!(expand_hostlist("n[08-10]"), vec!["n08", "n09", "n10"],);
    }

    #[test]
    fn zero_padding_uses_wider_endpoint() {
        // start "1", end "03" -> width 2
        assert_eq!(expand_hostlist("n[1-03]"), vec!["n01", "n02", "n03"],);
    }

    #[test]
    fn whitespace_around_entries() {
        assert_eq!(expand_hostlist(" a , b "), vec!["a", "b"],);
    }

    #[test]
    fn mixed_entries() {
        assert_eq!(
            expand_hostlist("host1,node[1-2],gpu[01,03]"),
            vec!["host1", "node1", "node2", "gpu01", "gpu03"],
        );
    }

    #[test]
    fn unterminated_bracket_is_literal() {
        // Missing closing bracket: rest treated as literal.
        assert_eq!(expand_hostlist("node[1-3"), vec!["node[1-3"]);
    }

    #[test]
    fn single_element_bracket() {
        assert_eq!(expand_hostlist("node[5]"), vec!["node5"]);
    }

    #[test]
    fn non_numeric_bracket_is_literal() {
        assert_eq!(expand_hostlist("node[abc]"), vec!["nodeabc"]);
    }
}
