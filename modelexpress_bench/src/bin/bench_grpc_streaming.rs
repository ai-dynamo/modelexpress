// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `bench_grpc_streaming`: measures the ModelExpress gRPC `StreamModelFiles`
//! path end-to-end with synthetic byte sources and in-memory sinks (no disk
//! on either end). Run two pods, one in `serve` mode and one in `client`
//! mode, and the client emits a single JSON-line result on stdout suitable
//! for concatenation into a sweep file.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use modelexpress_bench::bench_service::BenchModelService;
use modelexpress_bench::model_name::BenchSpec;
use modelexpress_bench::validation::StrictValidator;
use modelexpress_bench::{LatencySummary, RunResult};
use modelexpress_common::grpc::model::ModelFilesRequest;
use modelexpress_common::grpc::model::model_service_client::ModelServiceClient;
use modelexpress_common::grpc::model::model_service_server::ModelServiceServer;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Endpoint, Server};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "ModelExpress gRPC streaming throughput benchmark"
)]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Run the synthetic-source gRPC server. No disk I/O.
    Serve(ServeArgs),
    /// Run the in-memory-sink gRPC client. No disk I/O. Emits one JSON object on stdout.
    Client(ClientArgs),
}

#[derive(Parser, Debug)]
struct ServeArgs {
    /// Address to bind, eg. 0.0.0.0:8001
    #[arg(long, default_value = "0.0.0.0:8001")]
    addr: SocketAddr,

    /// mpsc channel capacity for the per-RPC streaming task. Production uses 16.
    #[arg(long, default_value_t = 16)]
    mpsc_cap: usize,

    /// Source-buffer size in bytes. Must be >= the largest chunk size any client will request.
    /// Defaults to 16 MiB which covers any reasonable chunk-size sweep.
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    source_buf_size: usize,
}

#[derive(Parser, Debug)]
struct ClientArgs {
    /// Server URL, eg. http://10.0.0.1:8001
    #[arg(long)]
    server_addr: String,

    /// Total bytes to transfer in the measured run, parseable as plain digits or with K/M/G suffixes.
    #[arg(long, value_parser = parse_size, default_value = "1G")]
    total_bytes: u64,

    /// Number of synthetic files to split the payload across. Splitting exposes the
    /// per-file open/close overhead without doing real file I/O.
    #[arg(long, default_value_t = 1)]
    file_count: u64,

    /// Chunk size in bytes for streaming.
    #[arg(long, value_parser = parse_size, default_value = "1M")]
    chunk_size: u64,

    /// Optional warmup payload run before the timed run, parseable like total_bytes.
    /// Discarded from results. Use to absorb TCP slow-start and tonic startup cost.
    #[arg(long, value_parser = parse_size, default_value = "0")]
    warmup_bytes: u64,

    /// Run the production client validation invariants on every chunk
    /// (offset, total_size, sequential-write, terminator). Adds CPU per chunk.
    #[arg(long)]
    strict: bool,

    /// mpsc cap to record in the result. Server-side cap is set by the serve command;
    /// this is informational on the client side.
    #[arg(long, default_value_t = 16)]
    mpsc_cap: u32,

    /// Free-form label written into the result JSON, eg. "same-node-1m".
    #[arg(long, default_value = "")]
    label: String,
}

fn parse_size(s: &str) -> Result<u64, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("empty size".to_string());
    }
    let (num_part, mult): (&str, u64) = if let Some(rest) = trimmed
        .strip_suffix('G')
        .or_else(|| trimmed.strip_suffix('g'))
    {
        (rest, 1024 * 1024 * 1024)
    } else if let Some(rest) = trimmed
        .strip_suffix('M')
        .or_else(|| trimmed.strip_suffix('m'))
    {
        (rest, 1024 * 1024)
    } else if let Some(rest) = trimmed
        .strip_suffix('K')
        .or_else(|| trimmed.strip_suffix('k'))
    {
        (rest, 1024)
    } else {
        (trimmed, 1)
    };
    let value: u64 = num_part
        .parse()
        .map_err(|e| format!("invalid size {trimmed:?}: {e}"))?;
    value
        .checked_mul(mult)
        .ok_or_else(|| format!("size {trimmed:?} overflows u64"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.mode {
        Mode::Serve(args) => run_server(args).await,
        Mode::Client(args) => run_client(args).await,
    }
}

async fn run_server(args: ServeArgs) -> Result<()> {
    info!(
        "bench server starting on {} (mpsc_cap={})",
        args.addr, args.mpsc_cap
    );
    let svc = BenchModelService::new(args.source_buf_size, args.mpsc_cap);
    let listener = TcpListener::bind(args.addr)
        .await
        .with_context(|| format!("binding {}", args.addr))?;
    let local = listener
        .local_addr()
        .with_context(|| "reading local addr")?;
    info!("bench server listening on {}", local);
    Server::builder()
        .add_service(ModelServiceServer::new(svc))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await
        .context("tonic server failed")?;
    Ok(())
}

async fn run_client(args: ClientArgs) -> Result<()> {
    if args.file_count == 0 {
        anyhow::bail!("file_count must be >= 1");
    }
    if args.chunk_size == 0 {
        anyhow::bail!("chunk_size must be >= 1");
    }
    if args.total_bytes == 0 {
        anyhow::bail!("total_bytes must be >= 1");
    }
    let chunk_u32: u32 = u32::try_from(args.chunk_size)
        .context("chunk_size larger than u32::MAX (proto field is u32)")?;
    let bytes_per_file = args
        .total_bytes
        .checked_div(args.file_count)
        .ok_or_else(|| anyhow::anyhow!("file_count must be > 0"))?;
    if bytes_per_file == 0 {
        anyhow::bail!("total_bytes / file_count must be >= 1");
    }
    let spec = BenchSpec::new(bytes_per_file, args.file_count);

    info!(
        "bench client connecting to {} (chunk_size={}, total={}, files={}, strict={}, warmup={})",
        args.server_addr,
        args.chunk_size,
        args.total_bytes,
        args.file_count,
        args.strict,
        args.warmup_bytes
    );
    let endpoint = Endpoint::from_shared(args.server_addr.clone())
        .with_context(|| format!("parsing endpoint {:?}", args.server_addr))?;
    let channel: Channel = endpoint
        .connect()
        .await
        .context("connecting to bench server")?;
    let mut client = ModelServiceClient::new(channel);

    if args.warmup_bytes > 0 {
        let warmup_spec = BenchSpec::new(args.warmup_bytes, 1);
        info!("warmup: streaming {} bytes", args.warmup_bytes);
        let _ = drain_stream(&mut client, &warmup_spec, chunk_u32, false).await?;
    }

    let drain = drain_stream(&mut client, &spec, chunk_u32, args.strict).await?;
    let result = build_result(&args, &spec, drain);
    let line = serde_json::to_string(&result).context("serialising result")?;
    println!("{line}");
    Ok(())
}

#[derive(Debug)]
struct DrainOutcome {
    elapsed: Duration,
    ttfb: Duration,
    chunks_received: u64,
    bytes_received: u64,
    chunk_latencies: Vec<Duration>,
}

async fn drain_stream(
    client: &mut ModelServiceClient<Channel>,
    spec: &BenchSpec,
    chunk_size: u32,
    strict: bool,
) -> Result<DrainOutcome> {
    let request = ModelFilesRequest {
        model_name: spec.encode(),
        provider: 0,
        chunk_size,
    };
    let mut validator = if strict {
        Some(StrictValidator::new())
    } else {
        None
    };

    let start = Instant::now();
    let mut stream = client
        .stream_model_files(tonic::Request::new(request))
        .await
        .context("stream_model_files RPC")?
        .into_inner();
    let mut last = Instant::now();
    let mut ttfb: Option<Duration> = None;
    let mut chunks_received: u64 = 0;
    let mut bytes_received: u64 = 0;
    let mut chunk_latencies: Vec<Duration> = Vec::new();

    while let Some(chunk) = stream
        .message()
        .await
        .context("receiving FileChunk from stream")?
    {
        let now = Instant::now();
        if ttfb.is_none() {
            ttfb = Some(now.duration_since(start));
        } else {
            chunk_latencies.push(now.duration_since(last));
        }
        last = now;
        chunks_received = chunks_received.saturating_add(1);
        bytes_received = bytes_received.saturating_add(chunk.data.len() as u64);
        if let Some(v) = validator.as_mut() {
            v.observe(&chunk).context("strict validation failed")?;
        }
    }
    let elapsed = start.elapsed();

    if let Some(v) = validator {
        let files = v.finish().context("strict validation final check")?;
        if files != spec.file_count {
            warn!(
                "strict validator saw {} complete files, expected {}",
                files, spec.file_count
            );
        }
    }

    let expected = spec.total_bytes();
    if bytes_received != expected {
        warn!(
            "received {} bytes, expected {} ({} short)",
            bytes_received,
            expected,
            expected.saturating_sub(bytes_received)
        );
    }

    Ok(DrainOutcome {
        elapsed,
        ttfb: ttfb.unwrap_or_default(),
        chunks_received,
        bytes_received,
        chunk_latencies,
    })
}

#[allow(clippy::cast_precision_loss, clippy::arithmetic_side_effects)]
fn build_result(args: &ClientArgs, spec: &BenchSpec, mut drain: DrainOutcome) -> RunResult {
    let elapsed_ns = u64::try_from(drain.elapsed.as_nanos()).unwrap_or(u64::MAX);
    let ttfb_ns = u64::try_from(drain.ttfb.as_nanos()).unwrap_or(u64::MAX);
    let elapsed_secs = drain.elapsed.as_secs_f64();
    let bytes_per_sec = if elapsed_secs > 0.0 {
        drain.bytes_received as f64 / elapsed_secs
    } else {
        0.0
    };
    let gibibits_per_sec = bytes_per_sec * 8.0 / (1024.0 * 1024.0 * 1024.0);
    let chunks_per_sec = if elapsed_secs > 0.0 {
        drain.chunks_received as f64 / elapsed_secs
    } else {
        0.0
    };
    let summary = LatencySummary::from_samples(&mut drain.chunk_latencies);
    RunResult {
        label: args.label.clone(),
        server_addr: args.server_addr.clone(),
        total_bytes: spec.total_bytes(),
        chunk_size: u32::try_from(args.chunk_size).unwrap_or(u32::MAX),
        strict_validation: args.strict,
        mpsc_cap: args.mpsc_cap,
        warmup_bytes: args.warmup_bytes,
        elapsed_ns,
        bytes_per_sec,
        gibibits_per_sec,
        chunks_received: drain.chunks_received,
        chunks_per_sec,
        ttfb_ns,
        chunk_recv_latency: summary,
    }
}
