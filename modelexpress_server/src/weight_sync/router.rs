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

/// Trainer shard descriptor, mirroring `protocol.types.TrainerShard`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainerShard {
    pub agent_index: u32,
    pub row_start: i64,
    pub row_end: i64,
    pub device_addr: u64,
    pub row_bytes: i64,
    pub device_id: i32,
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
) -> Vec<RdmaDescriptorProto> {
    let mut descriptors = Vec::new();

    for region in regions {
        let tensor = match table.tensors.iter().find(|t| t.name == region.tensor_name) {
            Some(t) => t,
            None => continue,
        };

        let elems_per_row: i64 = if tensor.shape.len() > 1 {
            tensor.shape[1..].iter().product()
        } else {
            1
        };

        let mut shards = tensor.shards.clone();
        shards.sort_by_key(|s| s.row_start);

        let src_runs = unpack_runs(&region.src_elem_runs);
        let dst_runs = unpack_runs(&region.dst_elem_runs);

        // Split all src runs across shard boundaries -> (shard, shard_rel_offset, count)
        let src_triples: Vec<(TrainerShard, i64, i64)> = src_runs
            .iter()
            .flat_map(|&(off, count)| split_run(off, count, elems_per_row, &shards))
            .collect();

        let new_descs = zip_src_dst(
            &src_triples,
            &dst_runs,
            region.dst_addr,
            region.element_size as i64,
        );
        descriptors.extend(new_descs);
    }

    descriptors
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
) -> Vec<(i32, Vec<M2nDescriptorProto>)> {
    workers
        .iter()
        .map(|(rank, regions)| {
            let rdma_descs = route_regions(regions, table);
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
            (*rank, m2n_descs)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn unpack_runs(flat: &[i64]) -> Vec<(i64, i64)> {
    flat.chunks(2).map(|c| (c[0], c[1])).collect()
}

#[allow(clippy::arithmetic_side_effects)]
fn split_run(
    run_offset: i64,
    run_count: i64,
    elems_per_row: i64,
    shards: &[TrainerShard],
) -> Vec<(TrainerShard, i64, i64)> {
    let mut result = Vec::new();
    let mut pos = run_offset;
    let mut remaining = run_count;

    while remaining > 0 {
        let row = if elems_per_row > 0 {
            pos / elems_per_row
        } else {
            0
        };
        let col = if elems_per_row > 0 {
            pos % elems_per_row
        } else {
            0
        };

        let shard = match shards
            .iter()
            .find(|s| s.row_start <= row && row < s.row_end)
        {
            Some(s) => s.clone(),
            None => break, // element not covered by any shard
        };

        let elems_until_shard_end = (shard.row_end - row) * elems_per_row - col;
        let count = remaining.min(elems_until_shard_end);
        let shard_rel = (row - shard.row_start) * elems_per_row + col;

        result.push((shard, shard_rel, count));
        pos += count;
        remaining -= count;
    }

    result
}

#[allow(clippy::arithmetic_side_effects)]
fn zip_src_dst(
    src_triples: &[(TrainerShard, i64, i64)],
    dst_runs: &[(i64, i64)],
    dst_base_addr: u64,
    element_size: i64,
) -> Vec<RdmaDescriptorProto> {
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
        let src_addr = shard.device_addr + (src_rel * element_size) as u64;
        let dst_addr = dst_base_addr + (dst_off * element_size) as u64;

        descriptors.push(RdmaDescriptorProto {
            agent_index: shard.agent_index,
            src_addr,
            dst_addr,
            nbytes: (count * element_size) as u64,
        });

        src_rel += count;
        src_rem -= count;
        dst_off += count;
        dst_rem -= count;

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

    descriptors
}
