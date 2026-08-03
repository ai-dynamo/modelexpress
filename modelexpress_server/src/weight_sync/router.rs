// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-arithmetic region router: maps resolved element-run regions to
//! NIXL RDMA descriptors using the TrainerTable shard layout.
//!
//! This is the Rust mirror of
//! `modelexpress/weight_transfer/planner/router.py`.
//! Both implementations must produce identical output for the same inputs.

use modelexpress_common::grpc::weight_sync::{
    M2nDescriptorProto, RdmaDescriptorProto, ResolvedRegionProto,
};
use serde::Deserialize;

/// Error produced while routing regions against a TrainerTable.
#[derive(Debug)]
pub enum RouteError {
    /// A flat element-run list did not contain an even number of entries.
    OddElemRuns(usize),
    /// Address or byte-count arithmetic overflowed or went negative.
    AddressOverflow,
    /// A region named a tensor that is absent from the TrainerTable.
    UnknownTensor(String),
    /// An element of the source region is owned by no shard.
    UncoveredElement { row: i64, col: i64 },
    /// A 2-D tile shard is padded or strided, so its row stride is not its width.
    PaddedTile {
        row_start: i64,
        row_end: i64,
        col_start: i64,
        col_end: i64,
        row_bytes: i64,
        expected: i64,
    },
    /// Source and destination element counts disagree.
    ElemCountMismatch { src: i64, dst: i64 },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OddElemRuns(len) => write!(
                f,
                "element-run list must have an even number of entries, got {len}"
            ),
            Self::AddressOverflow => write!(f, "address arithmetic overflowed"),
            Self::UnknownTensor(name) => write!(
                f,
                "no trainer tensor named {name:?} in the trainer table; routing it away \
                 silently would leave the parameter stale while the transfer still \
                 reported success"
            ),
            Self::UncoveredElement { row, col } => write!(
                f,
                "element at (row={row}, col={col}) is not covered by any shard"
            ),
            Self::PaddedTile {
                row_start,
                row_end,
                col_start,
                col_end,
                row_bytes,
                expected,
            } => write!(
                f,
                "shard rows {row_start}:{row_end} cols {col_start}:{col_end} has \
                 row_bytes={row_bytes}, expected {expected}; 2-D tile routing requires \
                 densely packed tiles"
            ),
            Self::ElemCountMismatch { src, dst } => write!(
                f,
                "source and destination element counts differ ({src} != {dst}); zipping \
                 them would emit a partial descriptor list that transfers as success"
            ),
        }
    }
}

impl std::error::Error for RouteError {}

/// Trainer shard descriptor, mirroring `protocol.types.TrainerShard`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainerShard {
    pub agent_index: u32,
    pub row_start: i64,
    pub row_end: i64,
    pub device_addr: u64,
    pub row_bytes: i64,
    pub device_id: i32,
    /// First column owned by this shard; 0 for row-only sharding.
    #[serde(default)]
    pub col_start: i64,
    /// One past the last column owned; `-1` means "to the full tensor width".
    #[serde(default = "default_col_end")]
    pub col_end: i64,
}

const fn default_col_end() -> i64 {
    -1
}

/// Trainer tensor descriptor, mirroring `protocol.types.TrainerTensor`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainerTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub shards: Vec<TrainerShard>,
}

/// Trainer table, mirroring `protocol.types.TrainerTable`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainerTableJson {
    pub step: i64,
    pub agents: Vec<String>, // base64-encoded NIXL metadata blobs
    pub tensors: Vec<TrainerTensor>,
}

/// Route a list of resolved regions against a TrainerTable, returning RDMA
/// descriptors ready for NIXL execution.
///
/// Algorithm: for each region, iterate its source element runs, split at
/// shard boundaries, compute GPU byte addresses, zip with destination runs.
pub fn route_regions(
    regions: &[ResolvedRegionProto],
    table: &TrainerTableJson,
) -> Result<Vec<RdmaDescriptorProto>, RouteError> {
    let mut descriptors = Vec::new();

    for region in regions {
        let tensor = table
            .tensors
            .iter()
            .find(|t| t.name == region.tensor_name)
            .ok_or_else(|| RouteError::UnknownTensor(region.tensor_name.clone()))?;

        let full_width: i64 = tensor
            .shape
            .get(1..)
            .unwrap_or(&[])
            .iter()
            .try_fold(1i64, |acc, d| acc.checked_mul(*d))
            .ok_or(RouteError::AddressOverflow)?;

        // Sort by row_start, then col_start so 2-D tiles are visited in order.
        let mut shards = tensor.shards.clone();
        shards.sort_by_key(|s| (s.row_start, s.col_start));

        let row_only = is_row_only(&shards, full_width);
        if !row_only {
            check_dense_tiles(&shards, full_width, i64::from(region.element_size))?;
        }

        let src_runs = unpack_runs(&region.src_elem_runs)?;
        let dst_runs = unpack_runs(&region.dst_elem_runs)?;

        // Split all src runs across shard boundaries -> (shard, shard_rel_offset, count)
        let mut src_triples: Vec<(TrainerShard, i64, i64)> = Vec::new();
        for &(off, count) in &src_runs {
            src_triples.extend(split_run(off, count, full_width, &shards, row_only)?);
        }

        let new_descs = zip_src_dst(
            &src_triples,
            &dst_runs,
            region.dst_addr,
            i64::from(region.element_size),
        )?;
        descriptors.extend(new_descs);
    }

    Ok(descriptors)
}

/// Route resolved regions for all workers in one pass, returning per-worker
/// M2N descriptor slices tagged with both src and dst agent indices.
///
/// Each worker's regions are routed independently against the shared
/// TrainerTable.  The resulting descriptors are annotated with
/// `dst_agent_index = worker_rank` so the trainer side can identify
/// which worker each descriptor targets.
pub fn route_all_workers(
    workers: &[(i32, &[ResolvedRegionProto])],
    table: &TrainerTableJson,
) -> Result<Vec<(i32, Vec<M2nDescriptorProto>)>, RouteError> {
    workers
        .iter()
        .map(|(rank, regions)| {
            let rdma_descs = route_regions(regions, table)?;
            let m2n_descs = rdma_descs
                .into_iter()
                .map(|d| M2nDescriptorProto {
                    src_agent_index: d.agent_index,
                    dst_agent_index: *rank as u32,
                    src_addr: d.src_addr,
                    dst_addr: d.dst_addr,
                    nbytes: d.nbytes,
                })
                .collect();
            Ok((*rank, m2n_descs))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn unpack_runs(flat: &[i64]) -> Result<Vec<(i64, i64)>, RouteError> {
    let pairs = flat.chunks_exact(2);
    if !pairs.remainder().is_empty() {
        return Err(RouteError::OddElemRuns(flat.len()));
    }
    Ok(pairs.map(|c| (c[0], c[1])).collect())
}

/// Convert an element count to a byte count, rejecting overflow and negatives.
fn elems_to_bytes(elems: i64, element_size: i64) -> Result<u64, RouteError> {
    elems
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(RouteError::AddressOverflow)
}

/// Offset a base device address by `elems * element_size` bytes.
fn offset_addr(base: u64, elems: i64, element_size: i64) -> Result<u64, RouteError> {
    base.checked_add(elems_to_bytes(elems, element_size)?)
        .ok_or(RouteError::AddressOverflow)
}

/// Checked `i64` arithmetic; any overflow becomes [`RouteError::AddressOverflow`].
fn ck(value: Option<i64>) -> Result<i64, RouteError> {
    value.ok_or(RouteError::AddressOverflow)
}

/// Resolve a shard's `col_end`, mapping the `-1` sentinel to the full width.
const fn resolved_col_end(shard: &TrainerShard, full_width: i64) -> i64 {
    if shard.col_end == -1 {
        full_width
    } else {
        shard.col_end
    }
}

/// True when every shard spans the full column range, i.e. row-only sharding.
fn is_row_only(shards: &[TrainerShard], full_width: i64) -> bool {
    shards
        .iter()
        .all(|s| s.col_start == 0 && resolved_col_end(s, full_width) == full_width)
}

/// Reject 2-D tile shards whose row stride is not the dense tile width.
///
/// The local offset math in [`shard_for_elem_2d`] addresses a tile as
/// `shard_width` contiguous elements per row.  A shard whose `row_bytes`
/// disagrees is padded or a strided view, and every row after the first would
/// resolve to the wrong address.
fn check_dense_tiles(
    shards: &[TrainerShard],
    full_width: i64,
    element_size: i64,
) -> Result<(), RouteError> {
    for shard in shards {
        let col_end = resolved_col_end(shard, full_width);
        let width = ck(col_end.checked_sub(shard.col_start))?;
        let expected = ck(width.checked_mul(element_size))?;
        if shard.row_bytes != expected {
            return Err(RouteError::PaddedTile {
                row_start: shard.row_start,
                row_end: shard.row_end,
                col_start: shard.col_start,
                col_end,
                row_bytes: shard.row_bytes,
                expected,
            });
        }
    }
    Ok(())
}

/// Split `pos` into a `(row, col)` pair against the full tensor width.
fn row_col(pos: i64, full_width: i64) -> Result<(i64, i64), RouteError> {
    if full_width > 0 {
        Ok((
            ck(pos.checked_div(full_width))?,
            ck(pos.checked_rem(full_width))?,
        ))
    } else {
        Ok((0, 0))
    }
}

/// Return the shard owning `(row, col)` and the element offset within it.
///
/// For 2-D tile shards the local offset is
/// `(row - row_start) * shard_width + (col - col_start)`, which keeps
/// `device_addr + local_off * element_size` correct for any element in a tile.
fn shard_for_elem_2d(
    row: i64,
    col: i64,
    shards: &[TrainerShard],
    full_width: i64,
) -> Result<(TrainerShard, i64), RouteError> {
    for shard in shards {
        let col_end = resolved_col_end(shard, full_width);
        if shard.row_start <= row && row < shard.row_end && shard.col_start <= col && col < col_end
        {
            let width = ck(col_end.checked_sub(shard.col_start))?;
            let local_off = ck(
                ck(ck(row.checked_sub(shard.row_start))?.checked_mul(width))?
                    .checked_add(ck(col.checked_sub(shard.col_start))?),
            )?;
            return Ok((shard.clone(), local_off));
        }
    }
    Err(RouteError::UncoveredElement { row, col })
}

/// Fast path for row-only sharding: split runs at row-shard boundaries.
fn split_run_row_only(
    run_offset: i64,
    run_count: i64,
    elems_per_row: i64,
    shards: &[TrainerShard],
) -> Result<Vec<(TrainerShard, i64, i64)>, RouteError> {
    let mut result = Vec::new();
    let mut pos = run_offset;
    let mut remaining = run_count;

    while remaining > 0 {
        let (row, col) = row_col(pos, elems_per_row)?;

        let shard = shards
            .iter()
            .find(|s| s.row_start <= row && row < s.row_end)
            .ok_or(RouteError::UncoveredElement { row, col })?
            .clone();

        let shard_rel = ck(
            ck(ck(row.checked_sub(shard.row_start))?.checked_mul(elems_per_row))?.checked_add(col),
        )?;
        let elems_until_shard_end = ck(ck(
            ck(shard.row_end.checked_sub(row))?.checked_mul(elems_per_row)
        )?
        .checked_sub(col))?;
        if elems_until_shard_end <= 0 {
            return Err(RouteError::UncoveredElement { row, col });
        }
        let count = remaining.min(elems_until_shard_end);

        result.push((shard, shard_rel, count));
        pos = ck(pos.checked_add(count))?;
        remaining = ck(remaining.checked_sub(count))?;
    }

    Ok(result)
}

/// Split an element run at row AND column shard boundaries (2-D tiling).
///
/// Advances one column segment at a time, stopping at the earlier of the
/// column boundary, the original-tensor row end, or the end of the run.
/// Stopping at the row end is required because crossing it jumps to
/// `row + 1, col = 0`, which may land in a different column shard.
fn split_run_2d(
    run_offset: i64,
    run_count: i64,
    full_width: i64,
    shards: &[TrainerShard],
) -> Result<Vec<(TrainerShard, i64, i64)>, RouteError> {
    let mut result = Vec::new();
    let mut pos = run_offset;
    let mut remaining = run_count;

    while remaining > 0 {
        let (row, col) = row_col(pos, full_width)?;
        let (shard, local_off) = shard_for_elem_2d(row, col, shards, full_width)?;
        let col_end = resolved_col_end(&shard, full_width);

        let elems_to_col_boundary = ck(col_end.checked_sub(col))?;
        let elems_to_row_end = ck(full_width.checked_sub(col))?;
        let segment = remaining.min(elems_to_col_boundary).min(elems_to_row_end);
        if segment <= 0 {
            return Err(RouteError::UncoveredElement { row, col });
        }

        result.push((shard, local_off, segment));
        pos = ck(pos.checked_add(segment))?;
        remaining = ck(remaining.checked_sub(segment))?;
    }

    Ok(result)
}

fn split_run(
    run_offset: i64,
    run_count: i64,
    full_width: i64,
    shards: &[TrainerShard],
    row_only: bool,
) -> Result<Vec<(TrainerShard, i64, i64)>, RouteError> {
    if row_only {
        split_run_row_only(run_offset, run_count, full_width, shards)
    } else {
        split_run_2d(run_offset, run_count, full_width, shards)
    }
}

fn zip_src_dst(
    src_triples: &[(TrainerShard, i64, i64)],
    dst_runs: &[(i64, i64)],
    dst_base_addr: u64,
    element_size: i64,
) -> Result<Vec<RdmaDescriptorProto>, RouteError> {
    // A short side would otherwise emit a partial descriptor list that the
    // caller executes and reports as a successful transfer.
    let src_total = src_triples
        .iter()
        .try_fold(0i64, |acc, &(_, _, count)| acc.checked_add(count))
        .ok_or(RouteError::AddressOverflow)?;
    let dst_total = dst_runs
        .iter()
        .try_fold(0i64, |acc, &(_, count)| acc.checked_add(count))
        .ok_or(RouteError::AddressOverflow)?;
    if src_total != dst_total {
        return Err(RouteError::ElemCountMismatch {
            src: src_total,
            dst: dst_total,
        });
    }

    let mut descriptors = Vec::new();

    let mut src_iter = src_triples.iter().peekable();
    let mut dst_iter = dst_runs.iter().peekable();

    let mut src_rem: i64 = 0;
    let mut src_rel: i64 = 0;
    let mut cur_shard: Option<&TrainerShard> = None;
    let mut dst_off: i64 = 0;
    let mut dst_rem: i64 = 0;

    // Prime src
    if let Some((shard, rel, count)) = src_iter.next() {
        cur_shard = Some(shard);
        src_rel = *rel;
        src_rem = *count;
    }
    // Prime dst
    if let Some(&(off, count)) = dst_iter.next() {
        dst_off = off;
        dst_rem = count;
    }

    while let Some(shard) = cur_shard {
        if dst_rem == 0 {
            break;
        }

        let count = src_rem.min(dst_rem);
        let src_addr = offset_addr(shard.device_addr, src_rel, element_size)?;
        let dst_addr = offset_addr(dst_base_addr, dst_off, element_size)?;
        let nbytes = elems_to_bytes(count, element_size)?;

        descriptors.push(RdmaDescriptorProto {
            agent_index: shard.agent_index,
            src_addr,
            dst_addr,
            nbytes,
        });

        src_rel = ck(src_rel.checked_add(count))?;
        src_rem = ck(src_rem.checked_sub(count))?;
        dst_off = ck(dst_off.checked_add(count))?;
        dst_rem = ck(dst_rem.checked_sub(count))?;

        if src_rem == 0 {
            match src_iter.next() {
                Some((s, rel, cnt)) => {
                    cur_shard = Some(s);
                    src_rel = *rel;
                    src_rem = *cnt;
                }
                None => break,
            }
        }
        if dst_rem == 0 {
            match dst_iter.next() {
                Some(&(off, cnt)) => {
                    dst_off = off;
                    dst_rem = cnt;
                }
                None => break,
            }
        }
    }

    Ok(descriptors)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn shard(
        agent_index: u32,
        row_start: i64,
        row_end: i64,
        col_start: i64,
        col_end: i64,
        device_addr: u64,
        row_bytes: i64,
    ) -> TrainerShard {
        TrainerShard {
            agent_index,
            row_start,
            row_end,
            device_addr,
            row_bytes,
            device_id: 0,
            col_start,
            col_end,
        }
    }

    fn region(
        tensor_name: &str,
        src_elem_runs: Vec<i64>,
        dst_elem_runs: Vec<i64>,
    ) -> ResolvedRegionProto {
        ResolvedRegionProto {
            tensor_name: tensor_name.to_owned(),
            src_elem_runs,
            dst_addr: 0x1000,
            dst_elem_runs,
            element_size: 4,
            dst_device_id: 0,
        }
    }

    /// 4x8 tensor split into two row-only shards of 2 rows each.
    fn row_only_table() -> TrainerTableJson {
        TrainerTableJson {
            step: 0,
            agents: vec![],
            tensors: vec![TrainerTensor {
                name: "w".to_owned(),
                dtype: "float32".to_owned(),
                shape: vec![4, 8],
                shards: vec![
                    shard(0, 0, 2, 0, -1, 0x10000, 32),
                    shard(1, 2, 4, 0, -1, 0x20000, 32),
                ],
            }],
        }
    }

    /// 4x8 tensor split into four 2x4 tiles: rows {0..2, 2..4} x cols {0..4, 4..8}.
    fn tiled_table() -> TrainerTableJson {
        TrainerTableJson {
            step: 0,
            agents: vec![],
            tensors: vec![TrainerTensor {
                name: "w".to_owned(),
                dtype: "float32".to_owned(),
                shape: vec![4, 8],
                shards: vec![
                    shard(0, 0, 2, 0, 4, 0x10000, 16),
                    shard(1, 0, 2, 4, 8, 0x20000, 16),
                    shard(2, 2, 4, 0, 4, 0x30000, 16),
                    shard(3, 2, 4, 4, 8, 0x40000, 16),
                ],
            }],
        }
    }

    #[test]
    fn row_only_routing_splits_at_the_shard_boundary() {
        let table = row_only_table();
        // Whole tensor: 32 elements starting at 0.
        let regions = vec![region("w", vec![0, 32], vec![0, 32])];
        let descs = route_regions(&regions, &table).expect("routing failed");

        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].agent_index, 0);
        assert_eq!(descs[0].src_addr, 0x10000);
        assert_eq!(descs[0].nbytes, 64);
        assert_eq!(descs[1].agent_index, 1);
        assert_eq!(descs[1].src_addr, 0x20000);
        assert_eq!(descs[1].nbytes, 64);
    }

    #[test]
    fn column_sharded_routing_selects_the_owning_tile() {
        let table = tiled_table();
        let regions = vec![region("w", vec![0, 32], vec![0, 32])];
        let descs = route_regions(&regions, &table).expect("routing failed");

        // Each original row crosses a column boundary, so each of the 4 rows
        // yields 2 segments of 4 elements: 8 descriptors, agents 0,1,0,1,2,3,2,3.
        assert_eq!(descs.len(), 8);
        let agents: Vec<u32> = descs.iter().map(|d| d.agent_index).collect();
        assert_eq!(agents, vec![0, 1, 0, 1, 2, 3, 2, 3]);
        assert!(descs.iter().all(|d| d.nbytes == 16));

        // Row 1, cols 0..4 is the second row of tile 0: local offset 4.
        assert_eq!(descs[2].src_addr, 0x10000 + 4 * 4);
        // Row 1, cols 4..8 is the second row of tile 1: local offset 4.
        assert_eq!(descs[3].src_addr, 0x20000 + 4 * 4);
        // Row 2, cols 0..4 is the first row of tile 2: local offset 0.
        assert_eq!(descs[4].src_addr, 0x30000);
    }

    #[test]
    fn column_sharded_routing_would_be_wrong_under_row_only_matching() {
        // Regression guard for the row-only mirror: matching by row alone
        // picks tile 0 for every column, so the col 4..8 half would address
        // agent 0 instead of agent 1.
        let table = tiled_table();
        let regions = vec![region("w", vec![4, 4], vec![0, 4])];
        let descs = route_regions(&regions, &table).expect("routing failed");

        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].agent_index, 1);
        assert_eq!(descs[0].src_addr, 0x20000);
    }

    #[test]
    fn odd_element_run_list_is_rejected() {
        let table = row_only_table();
        let regions = vec![region("w", vec![0, 32, 4], vec![0, 32])];
        let err = route_regions(&regions, &table).expect_err("expected an error");
        assert!(matches!(err, RouteError::OddElemRuns(3)));
    }

    #[test]
    fn unknown_tensor_is_rejected_instead_of_skipped() {
        let table = row_only_table();
        let regions = vec![region("absent", vec![0, 8], vec![0, 8])];
        let err = route_regions(&regions, &table).expect_err("expected an error");
        assert!(matches!(err, RouteError::UnknownTensor(name) if name == "absent"));
    }

    #[test]
    fn padded_tile_is_rejected() {
        let mut table = tiled_table();
        // Tile width 4 * 4 bytes = 16; declare a padded stride instead.
        table.tensors[0].shards[0].row_bytes = 24;
        let regions = vec![region("w", vec![0, 32], vec![0, 32])];
        let err = route_regions(&regions, &table).expect_err("expected an error");
        assert!(matches!(err, RouteError::PaddedTile { row_bytes: 24, .. }));
    }

    #[test]
    fn mismatched_src_and_dst_element_counts_are_rejected() {
        let table = row_only_table();
        let regions = vec![region("w", vec![0, 16], vec![0, 8])];
        let err = route_regions(&regions, &table).expect_err("expected an error");
        assert!(matches!(
            err,
            RouteError::ElemCountMismatch { src: 16, dst: 8 }
        ));
    }

    #[test]
    fn element_outside_every_shard_is_rejected() {
        let table = row_only_table();
        // Row 4 is past row_end of the last shard.
        let regions = vec![region("w", vec![32, 8], vec![0, 8])];
        let err = route_regions(&regions, &table).expect_err("expected an error");
        assert!(matches!(err, RouteError::UncoveredElement { row: 4, .. }));
    }

    #[test]
    fn overflowing_address_math_is_rejected_not_panicking() {
        let mut table = row_only_table();
        table.tensors[0].shards[0].device_addr = u64::MAX - 8;
        // Start mid-shard so the descriptor offsets past the base address.
        let regions = vec![region("w", vec![4, 12], vec![0, 12])];
        let err = route_regions(&regions, &table).expect_err("expected an error");
        assert!(matches!(err, RouteError::AddressOverflow));
    }
}
