// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation of worker-local, address-independent tensor bindings.

#![allow(clippy::result_large_err)] // Match tonic service validation helpers.

use modelexpress_common::grpc::refit::{
    GetWeightVersionShardManifestRequest, TrainerTensorsMetadata, WeightVersionShard,
    refit_worker_service_client::RefitWorkerServiceClient,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tonic::Status;

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct BoxCoverage {
    shard_offset: Vec<u64>,
    shape: Vec<u64>,
}

#[derive(Deserialize, Serialize)]
struct TensorCoverage {
    name: String,
    dtype: String,
    elsize: u64,
    full_shape: Vec<u64>,
    shards: Vec<BoxCoverage>,
}

#[derive(Deserialize, Serialize)]
struct BoundManifest {
    tensors: Vec<TensorCoverage>,
}

fn invalid(message: &str) -> Status {
    Status::invalid_argument(message)
}

fn volume(shape: &[u64]) -> Result<u64, Status> {
    shape.iter().try_fold(1_u64, |volume, &extent| {
        if extent == 0 {
            return Err(invalid("tensor coverage dimensions must be positive"));
        }
        volume
            .checked_mul(extent)
            .ok_or_else(|| invalid("tensor coverage volume overflows uint64"))
    })
}

pub(super) async fn validate_publication(
    shard: &WeightVersionShard,
    metadata: &TrainerTensorsMetadata,
) -> Result<(), Status> {
    if shard.logical_shard_id != metadata.logical_shard_id
        || shard.manifest_endpoint != metadata.metadata_endpoint
    {
        return Err(Status::failed_precondition(
            "publication does not match worker binding",
        ));
    }
    let endpoint =
        tonic::transport::Endpoint::from_shared(format!("http://{}", shard.manifest_endpoint))
            .map_err(|_| invalid("invalid manifest endpoint"))?
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30));
    let channel = endpoint
        .connect()
        .await
        .map_err(|_| Status::unavailable("could not connect to publication manifest endpoint"))?;
    let response = RefitWorkerServiceClient::new(channel)
        .max_decoding_message_size(100 * 1024 * 1024)
        .get_weight_version_shard_manifest(GetWeightVersionShardManifestRequest {
            version_id: shard.version_id.clone(),
            logical_shard_id: shard.logical_shard_id.clone(),
        })
        .await?
        .into_inner();
    if response.manifest_digest != shard.manifest_digest
        || format!("{:x}", Sha256::digest(&response.manifest)) != shard.manifest_digest
    {
        return Err(invalid("publication manifest digest does not match"));
    }
    let mut manifest: BoundManifest = serde_json::from_slice(&response.manifest)
        .map_err(|_| invalid("invalid publication tensor manifest"))?;
    manifest.tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let mut total_bytes = 0_u64;
    for tensor in &mut manifest.tensors {
        tensor.shards.sort();
        for coverage in &tensor.shards {
            let bytes = volume(&coverage.shape)?
                .checked_mul(tensor.elsize)
                .ok_or_else(|| invalid("publication byte count overflows uint64"))?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("publication byte count overflows uint64"))?;
        }
    }
    if shard.tensor_count != manifest.tensors.len() as u64 || shard.total_bytes != total_bytes {
        return Err(invalid(
            "publication tensor or byte count does not match manifest",
        ));
    }
    // Value serialization uses canonical sorted object keys, matching bind-time encoding.
    let value = serde_json::to_value(&manifest).map_err(|_| invalid("invalid bound coverage"))?;
    let bytes = serde_json::to_vec(&value).map_err(|_| invalid("invalid bound coverage"))?;
    if format!("{:x}", Sha256::digest(bytes)) != metadata.logical_shard_id {
        return Err(invalid(
            "publication coverage differs from bound logical shard",
        ));
    }
    Ok(())
}
