// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic-source `ModelService` implementation used by the bench server.
//!
//! Reuses the production `ModelService` proto trait so the wire format,
//! `FileChunk` shape, mpsc-backed streaming pattern, and tonic plumbing all
//! match production exactly. Disk I/O is replaced with a pre-allocated
//! source buffer that is sliced into per-chunk `Vec<u8>` payloads, mirroring
//! the production `buffer[..bytes_read].to_vec()` allocation.

use crate::fill_pattern;
use crate::model_name::BenchSpec;
use modelexpress_common::grpc::model::{
    FileChunk, ModelDownloadRequest, ModelFileInfo, ModelFileList, ModelFilesRequest,
    ModelStatus as GrpcModelStatus, ModelStatusUpdate, model_service_server::ModelService,
};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{debug, info};

const BENCH_BUFFER_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Synthetic-source service. Holds a single pre-filled source buffer that is
/// reused across every RPC and every chunk.
#[derive(Debug, Clone)]
pub struct BenchModelService {
    source: Arc<Vec<u8>>,
    mpsc_cap: usize,
}

impl BenchModelService {
    /// Build a new service with a source buffer at least `source_buf_size`
    /// bytes wide. The buffer is filled deterministically once at startup so
    /// no compression layer can inflate the apparent throughput.
    pub fn new(source_buf_size: usize, mpsc_cap: usize) -> Self {
        let mut buf = vec![0u8; source_buf_size.max(1)];
        fill_pattern(&mut buf, BENCH_BUFFER_SEED);
        info!(
            "BenchModelService ready: source_buf={} bytes, mpsc_cap={}",
            buf.len(),
            mpsc_cap
        );
        Self {
            source: Arc::new(buf),
            mpsc_cap: mpsc_cap.max(1),
        }
    }
}

#[tonic::async_trait]
impl ModelService for BenchModelService {
    type EnsureModelDownloadedStream = ReceiverStream<Result<ModelStatusUpdate, Status>>;
    type StreamModelFilesStream = ReceiverStream<Result<FileChunk, Status>>;

    async fn ensure_model_downloaded(
        &self,
        request: Request<ModelDownloadRequest>,
    ) -> Result<Response<Self::EnsureModelDownloadedStream>, Status> {
        // Synthetic source has nothing to fetch; emit a single DOWNLOADED
        // status so a client wired through request_model() can be reused.
        let inner = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let _ = tx
            .send(Ok(ModelStatusUpdate {
                model_name: inner.model_name,
                status: GrpcModelStatus::Downloaded as i32,
                message: Some("bench: synthetic source, no download".to_string()),
                provider: inner.provider,
            }))
            .await;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn list_model_files(
        &self,
        request: Request<ModelFilesRequest>,
    ) -> Result<Response<ModelFileList>, Status> {
        let inner = request.into_inner();
        let spec = BenchSpec::parse(&inner.model_name)
            .map_err(|e| Status::invalid_argument(format!("invalid bench model_name: {e:?}")))?;
        let files: Vec<ModelFileInfo> = (0..spec.file_count)
            .map(|idx| ModelFileInfo {
                relative_path: format!("bench-{idx:04}.bin"),
                size: spec.bytes_per_file,
            })
            .collect();
        Ok(Response::new(ModelFileList {
            model_name: inner.model_name,
            files,
            total_size: spec.total_bytes(),
        }))
    }

    async fn stream_model_files(
        &self,
        request: Request<ModelFilesRequest>,
    ) -> Result<Response<Self::StreamModelFilesStream>, Status> {
        let inner = request.into_inner();
        let spec = BenchSpec::parse(&inner.model_name)
            .map_err(|e| Status::invalid_argument(format!("invalid bench model_name: {e:?}")))?;
        let chunk_size = if inner.chunk_size == 0 {
            modelexpress_common::constants::DEFAULT_TRANSFER_CHUNK_SIZE
        } else {
            inner.chunk_size as usize
        };
        if chunk_size > self.source.len() {
            return Err(Status::invalid_argument(format!(
                "requested chunk_size {} exceeds source buffer {}",
                chunk_size,
                self.source.len(),
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(self.mpsc_cap);
        let source = Arc::clone(&self.source);

        debug!(
            "stream_model_files: bytes_per_file={}, file_count={}, chunk_size={}",
            spec.bytes_per_file, spec.file_count, chunk_size
        );

        tokio::spawn(async move {
            let total_files = spec.file_count;
            for file_idx in 0..total_files {
                let relative_path = format!("bench-{file_idx:04}.bin");
                let total_size = spec.bytes_per_file;
                let is_last_file = file_idx.saturating_add(1) == total_files;
                let mut offset: u64 = 0;
                if total_size == 0 {
                    let chunk = FileChunk {
                        relative_path: relative_path.clone(),
                        data: Vec::new(),
                        offset: 0,
                        total_size: 0,
                        is_last_chunk: true,
                        is_last_file,
                        commit_hash: None,
                        blake3: None,
                    };
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    continue;
                }
                while offset < total_size {
                    let remaining = total_size.saturating_sub(offset);
                    let take = remaining.min(chunk_size as u64) as usize;
                    let is_last_chunk = (remaining as usize) == take;
                    // Same memcpy shape as production: slice the reused
                    // source into a fresh Vec<u8> per chunk.
                    let data = source[..take].to_vec();
                    let chunk = FileChunk {
                        relative_path: relative_path.clone(),
                        data,
                        offset,
                        total_size,
                        is_last_chunk,
                        is_last_file: is_last_file && is_last_chunk,
                        commit_hash: None,
                        blake3: None,
                    };
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    offset = offset.saturating_add(take as u64);
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
