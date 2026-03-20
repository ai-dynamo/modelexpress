// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DHT (Kademlia) backend for P2P model metadata storage.
//!
//! Uses rust-libp2p's Kademlia DHT for decentralized metadata storage.
//! Workers discover each other without any centralized server. Wire-compatible
//! with the Python `mx_libp2p` implementation.
//!
//! ## Key Schema
//!
//! - `/mx/{source_id}/attrs` - source attributes JSON (model_name, etc.)
//! - `/mx/{source_id}/{worker_id}/{rank}` - per-worker record (JSON)
//! - `/mx/{source_id}/{worker_id}/workers` - directory of ranks for this worker_id
//! - `/mx/{source_id}/instances` - directory of worker_ids for this source
//! - `/mx/_sources` - global source_id list
//!
//! All records use JSON encoding and are well under the 64KB DHT record limit.

use super::{
    BackendMetadataRecord, MetadataBackend, MetadataResult, ModelMetadataRecord,
    SourceInstanceInfo, TensorRecord, WorkerRecord,
};
use async_trait::async_trait;
use libp2p::futures::StreamExt;
use libp2p::kad::store::{MemoryStore, RecordStore};
use libp2p::{
    Multiaddr, StreamProtocol, SwarmBuilder, identify, kad,
    kad::{Mode, Record, RecordKey},
    noise, tcp, yamux,
};
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::net::ToSocketAddrs;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// DHT key helpers
// ---------------------------------------------------------------------------

fn worker_key(source_id: &str, worker_id: &str, rank: u32) -> Vec<u8> {
    format!("/mx/{source_id}/{worker_id}/{rank}").into_bytes()
}

fn worker_directory_key(source_id: &str, worker_id: &str) -> Vec<u8> {
    format!("/mx/{source_id}/{worker_id}/workers").into_bytes()
}

fn instances_key(source_id: &str) -> Vec<u8> {
    format!("/mx/{source_id}/instances").into_bytes()
}

fn attrs_key(source_id: &str) -> Vec<u8> {
    format!("/mx/{source_id}/attrs").into_bytes()
}

fn sources_key() -> Vec<u8> {
    b"/mx/_sources".to_vec()
}

// ---------------------------------------------------------------------------
// JSON types (shared with the Python DHT client)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorRecordJson {
    name: String,
    addr: u64,
    size: u64,
    device_id: u32,
    dtype: String,
}

impl From<TensorRecord> for TensorRecordJson {
    fn from(r: TensorRecord) -> Self {
        Self {
            name: r.name,
            addr: r.addr,
            size: r.size,
            device_id: r.device_id,
            dtype: r.dtype,
        }
    }
}

impl From<TensorRecordJson> for TensorRecord {
    fn from(j: TensorRecordJson) -> Self {
        Self {
            name: j.name,
            addr: j.addr,
            size: j.size,
            device_id: j.device_id,
            dtype: j.dtype,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRecordJson {
    worker_rank: u32,
    #[serde(default)]
    backend_type: Option<String>,
    #[serde(default)]
    metadata_endpoint: Option<String>,
    #[serde(default)]
    agent_name: Option<String>,
    #[serde(default)]
    transfer_engine_session_id: Option<String>,
    tensors: Vec<TensorRecordJson>,
    #[serde(default)]
    status: i32,
    #[serde(default)]
    updated_at: i64,
}

impl From<WorkerRecord> for WorkerRecordJson {
    fn from(r: WorkerRecord) -> Self {
        let backend_type = r.backend_metadata.backend_type_str().to_string();
        let transfer_engine_sid = match r.backend_metadata {
            BackendMetadataRecord::TransferEngine(sid) => Some(sid),
            _ => None,
        };
        Self {
            worker_rank: r.worker_rank,
            backend_type: Some(backend_type),
            metadata_endpoint: if r.metadata_endpoint.is_empty() {
                None
            } else {
                Some(r.metadata_endpoint)
            },
            agent_name: if r.agent_name.is_empty() {
                None
            } else {
                Some(r.agent_name)
            },
            transfer_engine_session_id: transfer_engine_sid,
            tensors: r.tensors.into_iter().map(TensorRecordJson::from).collect(),
            status: r.status,
            updated_at: r.updated_at,
        }
    }
}

impl From<WorkerRecordJson> for WorkerRecord {
    fn from(j: WorkerRecordJson) -> Self {
        Self {
            worker_rank: j.worker_rank,
            backend_metadata: BackendMetadataRecord::from_flat(
                j.transfer_engine_session_id,
                j.backend_type.as_deref(),
            ),
            metadata_endpoint: j.metadata_endpoint.unwrap_or_default(),
            agent_name: j.agent_name.unwrap_or_default(),
            tensors: j.tensors.into_iter().map(TensorRecord::from).collect(),
            status: j.status,
            updated_at: j.updated_at,
        }
    }
}

/// Source-level attributes stored once per source_id.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SourceAttributesJson {
    pub model_name: String,
    #[serde(default)]
    pub mx_version: String,
    #[serde(default)]
    pub mx_source_type: i32,
    #[serde(default)]
    pub backend_framework: i32,
    #[serde(default)]
    pub tensor_parallel_size: u32,
    #[serde(default)]
    pub pipeline_parallel_size: u32,
    #[serde(default)]
    pub expert_parallel_size: u32,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub quantization: String,
}

impl From<&SourceIdentity> for SourceAttributesJson {
    fn from(id: &SourceIdentity) -> Self {
        Self {
            model_name: id.model_name.clone(),
            mx_version: id.mx_version.clone(),
            mx_source_type: id.mx_source_type,
            backend_framework: id.backend_framework,
            tensor_parallel_size: id.tensor_parallel_size,
            pipeline_parallel_size: id.pipeline_parallel_size,
            expert_parallel_size: id.expert_parallel_size,
            dtype: id.dtype.clone(),
            quantization: id.quantization.clone(),
        }
    }
}

/// Directory record listing all known worker ranks for a source_id/worker_id pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerDirectoryJson {
    ranks: Vec<u32>,
    updated_at: i64,
}

/// Directory record listing all known worker_ids for a source_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstancesDirectoryJson {
    worker_ids: Vec<String>,
    updated_at: i64,
}

/// Global source_id list.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SourcesListJson {
    source_ids: Vec<String>,
    updated_at: i64,
}

// ---------------------------------------------------------------------------
// Command channel types
// ---------------------------------------------------------------------------

enum Command {
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
        resp: oneshot::Sender<MetadataResult<()>>,
    },
    Get {
        key: Vec<u8>,
        resp: oneshot::Sender<MetadataResult<Option<Vec<u8>>>>,
    },
    Remove {
        key: Vec<u8>,
        resp: oneshot::Sender<MetadataResult<()>>,
    },
    Dial {
        addr: Multiaddr,
        resp: oneshot::Sender<MetadataResult<()>>,
    },
    Bootstrap {
        resp: oneshot::Sender<MetadataResult<()>>,
    },
}

// ---------------------------------------------------------------------------
// Swarm behaviour
// ---------------------------------------------------------------------------

#[derive(libp2p::swarm::NetworkBehaviour)]
struct Behaviour {
    kademlia: kad::Behaviour<MemoryStore>,
    identify: identify::Behaviour,
}

// ---------------------------------------------------------------------------
// Swarm event loop
// ---------------------------------------------------------------------------

/// Pending GET query state: holds the oneshot sender until a result arrives.
/// `take()` the sender on first `FoundRecord` to respond immediately.
type PendingGet = Option<oneshot::Sender<MetadataResult<Option<Vec<u8>>>>>;

async fn run_swarm(mut swarm: libp2p::Swarm<Behaviour>, mut cmd_rx: mpsc::Receiver<Command>) {
    let mut pending_gets: HashMap<kad::QueryId, PendingGet> = HashMap::new();

    loop {
        tokio::select! {
            event = swarm.select_next_some() => {
                handle_swarm_event(event, &mut swarm, &mut pending_gets);
            }
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(command) => handle_command(&mut swarm, command, &mut pending_gets),
                    None => {
                        info!("DHT command channel closed, shutting down swarm");
                        break;
                    }
                }
            }
        }
    }
}

fn handle_swarm_event(
    event: libp2p::swarm::SwarmEvent<BehaviourEvent>,
    swarm: &mut libp2p::Swarm<Behaviour>,
    pending_gets: &mut HashMap<kad::QueryId, PendingGet>,
) {
    match event {
        libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Kademlia(kad_event)) => {
            handle_kad_event(kad_event, pending_gets);
        }
        libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Identify(
            identify::Event::Received { peer_id, info, .. },
        )) => {
            debug!(
                "Identified peer {peer_id}: {} protocols",
                info.protocols.len()
            );
            for addr in &info.listen_addrs {
                swarm
                    .behaviour_mut()
                    .kademlia
                    .add_address(&peer_id, addr.clone());
            }
        }
        libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
            info!("DHT listening on {address}");
        }
        libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. } => {
            debug!("Connected to peer {peer_id}");
        }
        _ => {}
    }
}

fn handle_kad_event(event: kad::Event, pending_gets: &mut HashMap<kad::QueryId, PendingGet>) {
    #[allow(clippy::single_match)]
    match event {
        kad::Event::OutboundQueryProgressed {
            id, result, step, ..
        } => match result {
            kad::QueryResult::GetRecord(Ok(kad::GetRecordOk::FoundRecord(peer_record))) => {
                if let Some(sender_opt) = pending_gets.get_mut(&id)
                    && let Some(sender) = sender_opt.take()
                {
                    let _ = sender.send(Ok(Some(peer_record.record.value)));
                }
                if step.last {
                    pending_gets.remove(&id);
                }
            }
            kad::QueryResult::GetRecord(Ok(kad::GetRecordOk::FinishedWithNoAdditionalRecord {
                ..
            })) => {
                if let Some(Some(sender)) = pending_gets.remove(&id) {
                    let _ = sender.send(Ok(None));
                }
            }
            kad::QueryResult::GetRecord(Err(_)) => {
                // All GetRecord errors (NotFound, QuorumFailed, Timeout)
                // are treated as "not found" rather than hard errors.
                if let Some(Some(sender)) = pending_gets.remove(&id) {
                    let _ = sender.send(Ok(None));
                }
            }
            kad::QueryResult::Bootstrap(Ok(_)) => {
                debug!("Bootstrap step completed");
            }
            kad::QueryResult::Bootstrap(Err(e)) => {
                warn!("Bootstrap error: {e:?}");
            }
            _ => {}
        },
        _ => {}
    }
}

fn handle_command(
    swarm: &mut libp2p::Swarm<Behaviour>,
    command: Command,
    pending_gets: &mut HashMap<kad::QueryId, PendingGet>,
) {
    match command {
        Command::Put { key, value, resp } => {
            let record = Record {
                key: RecordKey::new(&key),
                value,
                publisher: None,
                expires: None,
            };
            // put_record stores locally immediately, then starts replication.
            // We respond as soon as the local store accepts the record.
            match swarm
                .behaviour_mut()
                .kademlia
                .put_record(record, kad::Quorum::One)
            {
                Ok(_) => {
                    let _ = resp.send(Ok(()));
                }
                Err(e) => {
                    let _ = resp.send(Err(Box::new(std::io::Error::other(format!(
                        "put_record rejected: {e:?}"
                    )))));
                }
            }
        }
        Command::Get { key, resp } => {
            let query_id = swarm
                .behaviour_mut()
                .kademlia
                .get_record(RecordKey::new(&key));
            pending_gets.insert(query_id, Some(resp));
        }
        Command::Remove { key, resp } => {
            let record_key = RecordKey::new(&key);
            swarm
                .behaviour_mut()
                .kademlia
                .store_mut()
                .remove(&record_key);
            let _ = resp.send(Ok(()));
        }
        Command::Dial { addr, resp } => match swarm.dial(addr) {
            Ok(()) => {
                let _ = resp.send(Ok(()));
            }
            Err(e) => {
                let _ = resp.send(Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    format!("dial failed: {e}"),
                ))));
            }
        },
        Command::Bootstrap { resp } => match swarm.behaviour_mut().kademlia.bootstrap() {
            Ok(_) => {
                let _ = resp.send(Ok(()));
            }
            Err(e) => {
                let _ = resp.send(Err(Box::new(std::io::Error::other(format!(
                    "bootstrap failed: {e:?}"
                )))));
            }
        },
    }
}

// ---------------------------------------------------------------------------
// DhtBackend
// ---------------------------------------------------------------------------

/// Configuration for the DHT backend.
#[derive(Debug, Clone)]
pub struct DhtConfig {
    pub listen_addr: String,
    pub bootstrap_peers: Vec<String>,
    pub bootstrap_dns: Option<String>,
    pub bootstrap_dns_port: u16,
    pub record_ttl_secs: u64,
}

impl Default for DhtConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:4001".to_string(),
            bootstrap_peers: Vec::new(),
            bootstrap_dns: None,
            bootstrap_dns_port: 4001,
            record_ttl_secs: 300,
        }
    }
}

impl DhtConfig {
    /// Build from environment variables.
    pub fn from_env() -> Self {
        let listen_addr = std::env::var("MX_DHT_LISTEN").unwrap_or_else(|_| "0.0.0.0:4001".into());
        let bootstrap_peers = std::env::var("MX_DHT_BOOTSTRAP_PEERS")
            .map(|s| s.split(',').map(|p| p.trim().to_string()).collect())
            .unwrap_or_default();
        let bootstrap_dns = std::env::var("MX_DHT_BOOTSTRAP_DNS").ok();
        let bootstrap_dns_port: u16 = std::env::var("MX_DHT_BOOTSTRAP_DNS_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4001);
        let record_ttl_secs: u64 = std::env::var("MX_DHT_RECORD_TTL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        Self {
            listen_addr,
            bootstrap_peers,
            bootstrap_dns,
            bootstrap_dns_port,
            record_ttl_secs,
        }
    }
}

/// Kademlia DHT metadata backend.
///
/// Runs a libp2p swarm in a background tokio task. All trait methods
/// communicate with the swarm via an mpsc command channel.
pub struct DhtBackend {
    cmd_tx: mpsc::Sender<Command>,
    config: DhtConfig,
    #[allow(dead_code)]
    task_handle: tokio::task::JoinHandle<()>,
}

impl DhtBackend {
    /// Create a new DHT backend and start the libp2p swarm.
    ///
    /// The swarm begins listening immediately. Call `connect()` to bootstrap
    /// into an existing network.
    pub async fn new(config: DhtConfig) -> MetadataResult<Self> {
        let listen_multiaddr: Multiaddr =
            format!("/ip4/{}", config.listen_addr.replace(':', "/tcp/"))
                .parse()
                .map_err(|e| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("invalid listen address '{}': {e}", config.listen_addr),
                    ))
                })?;

        let record_ttl = Duration::from_secs(config.record_ttl_secs);

        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(format!(
                    "transport build failed: {e}"
                )))
            })?
            .with_behaviour(|key| {
                let peer_id = key.public().to_peer_id();
                let mut kad_config = kad::Config::new(StreamProtocol::new("/ipfs/kad/1.0.0"));
                kad_config.set_record_ttl(Some(record_ttl));
                let store = MemoryStore::new(peer_id);
                let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);
                kademlia.set_mode(Some(Mode::Server));

                let identify_config =
                    identify::Config::new("/modelexpress/dht/0.1.0".to_string(), key.public());
                let identify = identify::Behaviour::new(identify_config);

                Behaviour { kademlia, identify }
            })
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::other(format!(
                    "behaviour build failed: {e}"
                )))
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        swarm.listen_on(listen_multiaddr).map_err(|e| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                format!("listen failed: {e}"),
            ))
        })?;

        let (cmd_tx, cmd_rx) = mpsc::channel(256);
        let task_handle = tokio::spawn(run_swarm(swarm, cmd_rx));

        info!(
            "DHT backend created (listen={}, ttl={}s)",
            config.listen_addr, config.record_ttl_secs
        );

        Ok(Self {
            cmd_tx,
            config,
            task_handle,
        })
    }

    // -- Low-level DHT operations (used by MetadataBackend trait methods) ------

    async fn dht_put(&self, key: Vec<u8>, value: Vec<u8>) -> MetadataResult<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Put {
                key,
                value,
                resp: resp_tx,
            })
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT swarm task gone",
                ))
            })?;
        resp_rx
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT response channel dropped",
                ))
            })?
    }

    async fn dht_get(&self, key: Vec<u8>) -> MetadataResult<Option<Vec<u8>>> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Get { key, resp: resp_tx })
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT swarm task gone",
                ))
            })?;
        resp_rx
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT response channel dropped",
                ))
            })?
    }

    async fn dht_remove(&self, key: Vec<u8>) -> MetadataResult<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Remove { key, resp: resp_tx })
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT swarm task gone",
                ))
            })?;
        resp_rx
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT response channel dropped",
                ))
            })?
    }

    async fn dial_peer(&self, addr: Multiaddr) -> MetadataResult<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Dial {
                addr,
                resp: resp_tx,
            })
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT swarm task gone",
                ))
            })?;
        resp_rx
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT response channel dropped",
                ))
            })?
    }

    async fn trigger_bootstrap(&self) -> MetadataResult<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Bootstrap { resp: resp_tx })
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT swarm task gone",
                ))
            })?;
        resp_rx
            .await
            .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "DHT response channel dropped",
                ))
            })?
    }

    // -- Helper: get + deserialize JSON from DHT ------------------------------

    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        key: Vec<u8>,
    ) -> MetadataResult<Option<T>> {
        match self.dht_get(key).await? {
            Some(bytes) => {
                let value: T = serde_json::from_slice(&bytes)?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    // -- Helper: serialize JSON + put into DHT --------------------------------

    async fn put_json<T: Serialize>(&self, key: Vec<u8>, value: &T) -> MetadataResult<()> {
        let bytes = serde_json::to_vec(value)?;
        self.dht_put(key, bytes).await
    }
}

// ---------------------------------------------------------------------------
// MetadataBackend trait implementation
// ---------------------------------------------------------------------------

fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

#[async_trait]
impl MetadataBackend for DhtBackend {
    async fn connect(&self) -> MetadataResult<()> {
        let mut dialed_any = false;

        // Dial explicit bootstrap peers
        for peer_str in &self.config.bootstrap_peers {
            let addr: Multiaddr = peer_str.parse().map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid bootstrap peer '{peer_str}': {e}"),
                ))
            })?;
            match self.dial_peer(addr).await {
                Ok(()) => {
                    dialed_any = true;
                    debug!("Dialed bootstrap peer: {peer_str}");
                }
                Err(e) => {
                    warn!("Failed to dial bootstrap peer {peer_str}: {e}");
                }
            }
        }

        // DNS-based bootstrap
        if let Some(dns_host) = &self.config.bootstrap_dns {
            let port = self.config.bootstrap_dns_port;
            let resolve_target = format!("{dns_host}:{port}");
            match resolve_target.to_socket_addrs() {
                Ok(addrs) => {
                    for addr in addrs {
                        let ma_str = if addr.is_ipv6() {
                            format!("/ip6/{}/tcp/{port}", addr.ip())
                        } else {
                            format!("/ip4/{}/tcp/{port}", addr.ip())
                        };
                        if let Ok(ma) = ma_str.parse::<Multiaddr>() {
                            match self.dial_peer(ma).await {
                                Ok(()) => {
                                    dialed_any = true;
                                    debug!("Dialed DNS bootstrap peer: {addr}");
                                }
                                Err(e) => {
                                    debug!("Failed to dial DNS peer {addr}: {e}");
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("DNS bootstrap resolution failed for '{resolve_target}': {e}");
                }
            }
        }

        // Trigger Kademlia bootstrap if we connected to any peers
        if dialed_any {
            // Small delay to let connections establish and Identify exchange complete
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Err(e) = self.trigger_bootstrap().await {
                warn!("Kademlia bootstrap failed (non-fatal): {e}");
            }
        } else if !self.config.bootstrap_peers.is_empty() || self.config.bootstrap_dns.is_some() {
            warn!("No bootstrap peers reachable; this node will serve records locally only");
        }

        info!("DHT backend connected");
        Ok(())
    }

    async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let now = now_millis();
        let source_id = crate::source_identity::compute_mx_source_id(identity);

        // Write each worker's record
        let mut new_ranks: BTreeSet<u32> = BTreeSet::new();
        for worker_meta in workers {
            let rank = worker_meta.worker_rank;
            new_ranks.insert(rank);
            let record: WorkerRecordJson = WorkerRecord::from(worker_meta).into();
            self.put_json(worker_key(&source_id, worker_id, rank), &record)
                .await?;
        }

        // Merge into the per-worker_id rank directory (read-modify-write, acceptable for DHT)
        let mut all_ranks: BTreeSet<u32> = BTreeSet::new();
        if let Some(existing) = self
            .get_json::<WorkerDirectoryJson>(worker_directory_key(&source_id, worker_id))
            .await?
        {
            all_ranks.extend(existing.ranks);
        }
        all_ranks.extend(new_ranks);
        self.put_json(
            worker_directory_key(&source_id, worker_id),
            &WorkerDirectoryJson {
                ranks: all_ranks.into_iter().collect(),
                updated_at: now,
            },
        )
        .await?;

        // Merge into the per-source instances directory
        let mut worker_ids: BTreeSet<String> = BTreeSet::new();
        if let Some(existing) = self
            .get_json::<InstancesDirectoryJson>(instances_key(&source_id))
            .await?
        {
            worker_ids.extend(existing.worker_ids);
        }
        worker_ids.insert(worker_id.to_string());
        self.put_json(
            instances_key(&source_id),
            &InstancesDirectoryJson {
                worker_ids: worker_ids.into_iter().collect(),
                updated_at: now,
            },
        )
        .await?;

        // Store source attributes (model_name, etc.)
        let attrs = SourceAttributesJson::from(identity);
        self.put_json(attrs_key(&source_id), &attrs).await?;

        // Add to global sources list
        let mut source_ids: BTreeSet<String> = BTreeSet::new();
        if let Some(existing) = self.get_json::<SourcesListJson>(sources_key()).await? {
            source_ids.extend(existing.source_ids);
        }
        source_ids.insert(source_id.clone());
        self.put_json(
            sources_key(),
            &SourcesListJson {
                source_ids: source_ids.into_iter().collect(),
                updated_at: now,
            },
        )
        .await?;

        debug!(
            "Published metadata for '{}' (source_id={source_id}, worker_id={worker_id})",
            identity.model_name
        );
        Ok(())
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        // Read the rank directory for this source_id/worker_id
        let directory = match self
            .get_json::<WorkerDirectoryJson>(worker_directory_key(source_id, worker_id))
            .await?
        {
            Some(d) => d,
            None => return Ok(None),
        };

        // Fetch model_name from source attributes
        let model_name = self
            .get_json::<SourceAttributesJson>(attrs_key(source_id))
            .await?
            .map(|a| a.model_name)
            .unwrap_or_default();

        // Fetch each worker's record
        let mut workers = Vec::with_capacity(directory.ranks.len());
        for rank in &directory.ranks {
            if let Some(w) = self
                .get_json::<WorkerRecordJson>(worker_key(source_id, worker_id, *rank))
                .await?
            {
                workers.push(WorkerRecord::from(w));
            }
        }

        if workers.is_empty() {
            return Ok(None);
        }

        Ok(Some(ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name,
            workers,
            published_at: directory.updated_at,
        }))
    }

    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        // Read the instances directory to find all worker_ids
        if let Some(instances) = self
            .get_json::<InstancesDirectoryJson>(instances_key(source_id))
            .await?
        {
            for wid in &instances.worker_ids {
                // Read rank directory for each worker_id
                if let Some(dir) = self
                    .get_json::<WorkerDirectoryJson>(worker_directory_key(source_id, wid))
                    .await?
                {
                    for rank in &dir.ranks {
                        self.dht_remove(worker_key(source_id, wid, *rank)).await?;
                    }
                }
                self.dht_remove(worker_directory_key(source_id, wid))
                    .await?;
            }
        }

        self.dht_remove(instances_key(source_id)).await?;
        self.dht_remove(attrs_key(source_id)).await?;

        // Remove from global sources list
        if let Some(existing) = self.get_json::<SourcesListJson>(sources_key()).await? {
            let source_ids: Vec<String> = existing
                .source_ids
                .into_iter()
                .filter(|s| s != source_id)
                .collect();
            self.put_json(
                sources_key(),
                &SourcesListJson {
                    source_ids,
                    updated_at: now_millis(),
                },
            )
            .await?;
        }

        debug!("Removed metadata for source_id={source_id}");
        Ok(())
    }

    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>> {
        // Collect source_ids to query
        let source_ids: Vec<String> = if let Some(sid) = source_id {
            vec![sid]
        } else {
            match self.get_json::<SourcesListJson>(sources_key()).await? {
                Some(list) => list.source_ids,
                None => Vec::new(),
            }
        };

        let mut result = Vec::new();

        for sid in &source_ids {
            let model_name = self
                .get_json::<SourceAttributesJson>(attrs_key(sid))
                .await?
                .map(|a| a.model_name)
                .unwrap_or_default();

            let instances = match self
                .get_json::<InstancesDirectoryJson>(instances_key(sid))
                .await?
            {
                Some(i) => i,
                None => continue,
            };

            for wid in &instances.worker_ids {
                let dir = match self
                    .get_json::<WorkerDirectoryJson>(worker_directory_key(sid, wid))
                    .await?
                {
                    Some(d) => d,
                    None => continue,
                };

                for rank in &dir.ranks {
                    let record = match self
                        .get_json::<WorkerRecordJson>(worker_key(sid, wid, *rank))
                        .await?
                    {
                        Some(r) => r,
                        None => continue,
                    };

                    if status_filter.is_some_and(|required| record.status != required as i32) {
                        continue;
                    }

                    result.push(SourceInstanceInfo {
                        source_id: sid.clone(),
                        worker_id: wid.clone(),
                        model_name: model_name.clone(),
                        worker_rank: *rank,
                    });
                }
            }
        }

        Ok(result)
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        let source_ids = match self.get_json::<SourcesListJson>(sources_key()).await? {
            Some(list) => list.source_ids,
            None => return Ok(Vec::new()),
        };

        let mut sources = Vec::new();
        for sid in source_ids {
            let model_name = self
                .get_json::<SourceAttributesJson>(attrs_key(&sid))
                .await?
                .map(|a| a.model_name)
                .unwrap_or_default();
            sources.push((sid, model_name));
        }
        Ok(sources)
    }

    async fn update_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()> {
        let key = worker_key(source_id, worker_id, worker_rank);
        let mut record: WorkerRecordJson = self.get_json(key.clone()).await?.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "update_status: rank {worker_rank} not found in source '{source_id}' worker '{worker_id}'"
                ),
            ))
        })?;

        record.status = status as i32;
        record.updated_at = updated_at;
        self.put_json(key, &record).await?;

        debug!(
            "Updated status for source '{source_id}' worker '{worker_id}' rank {worker_rank} -> {}",
            status as i32
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::p2p::SourceIdentity;

    fn test_identity(model_name: &str) -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: 0,
            model_name: model_name.to_string(),
            backend_framework: 1,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 1,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
        }
    }

    #[test]
    fn test_key_format() {
        assert_eq!(
            worker_key("abc123", "wid-0", 0),
            b"/mx/abc123/wid-0/0".to_vec()
        );
        assert_eq!(
            worker_directory_key("abc123", "wid-0"),
            b"/mx/abc123/wid-0/workers".to_vec()
        );
        assert_eq!(instances_key("abc123"), b"/mx/abc123/instances".to_vec());
        assert_eq!(attrs_key("abc123"), b"/mx/abc123/attrs".to_vec());
        assert_eq!(sources_key(), b"/mx/_sources".to_vec());
    }

    #[test]
    fn test_worker_record_json_roundtrip() {
        let record = WorkerRecord {
            worker_rank: 0,
            backend_metadata: BackendMetadataRecord::Nixl,
            metadata_endpoint: "10.0.0.1:5555".to_string(),
            agent_name: "mx-auto-worker0-abc123".to_string(),
            tensors: vec![TensorRecord {
                name: "layer.0.weight".to_string(),
                addr: 0x7fff_0000_0000,
                size: 1024,
                device_id: 0,
                dtype: "torch.float16".to_string(),
            }],
            status: 2, // READY
            updated_at: 1234567890,
        };

        let json: WorkerRecordJson = record.into();
        let bytes = serde_json::to_vec(&json).expect("serialize");
        let deserialized: WorkerRecordJson = serde_json::from_slice(&bytes).expect("deserialize");
        let back: WorkerRecord = deserialized.into();

        assert_eq!(back.worker_rank, 0);
        assert_eq!(back.metadata_endpoint, "10.0.0.1:5555");
        assert_eq!(back.agent_name, "mx-auto-worker0-abc123");
        assert_eq!(back.tensors.len(), 1);
        assert_eq!(back.tensors[0].addr, 0x7fff_0000_0000);
        assert_eq!(back.status, 2);
    }

    #[test]
    fn test_worker_directory_json_roundtrip() {
        let dir = WorkerDirectoryJson {
            ranks: vec![0, 1, 2, 3],
            updated_at: 9999,
        };
        let bytes = serde_json::to_vec(&dir).expect("serialize");
        let back: WorkerDirectoryJson = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(back.ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_sources_list_json_roundtrip() {
        let list = SourcesListJson {
            source_ids: vec!["aaaa1111".to_string(), "bbbb2222".to_string()],
            updated_at: 9999,
        };
        let bytes = serde_json::to_vec(&list).expect("serialize");
        let back: SourcesListJson = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(back.source_ids, vec!["aaaa1111", "bbbb2222"]);
    }

    #[test]
    fn test_source_attributes_from_identity() {
        let id = test_identity("deepseek-ai/DeepSeek-V3");
        let attr = SourceAttributesJson::from(&id);
        assert_eq!(attr.model_name, "deepseek-ai/DeepSeek-V3");
        assert_eq!(attr.mx_version, "0.3.0");
        assert_eq!(attr.tensor_parallel_size, 1);
    }

    #[tokio::test]
    async fn test_publish_get_remove() {
        let node = DhtBackend::new(DhtConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .expect("node");

        let identity = test_identity("test-model");
        let source_id = crate::source_identity::compute_mx_source_id(&identity);
        let worker_id = "worker-uuid-001";

        let worker = WorkerMetadata {
            worker_rank: 0,
            metadata_endpoint: "10.0.0.1:5555".to_string(),
            agent_name: "test-agent".to_string(),
            tensors: vec![],
            status: 2,
            updated_at: 1000,
            transfer_engine_session_id: String::new(),
        };

        node.publish_metadata(&identity, worker_id, vec![worker])
            .await
            .expect("publish");

        // get_metadata
        let result = node.get_metadata(&source_id, worker_id).await.expect("get");
        let record = result.expect("should find metadata");
        assert_eq!(record.source_id, source_id);
        assert_eq!(record.worker_id, worker_id);
        assert_eq!(record.model_name, "test-model");
        assert_eq!(record.workers.len(), 1);
        assert_eq!(record.workers[0].worker_rank, 0);
        assert_eq!(record.workers[0].metadata_endpoint, "10.0.0.1:5555");

        // list_sources
        let sources = node.list_sources().await.expect("list_sources");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, source_id);
        assert_eq!(sources[0].1, "test-model");

        // list_workers (no filter)
        let workers = node.list_workers(None, None).await.expect("list_workers");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].source_id, source_id);
        assert_eq!(workers[0].worker_id, worker_id);
        assert_eq!(workers[0].worker_rank, 0);

        // list_workers with status filter (should match)
        let ready_status = SourceStatus::try_from(2).expect("valid status");
        let filtered = node
            .list_workers(None, Some(ready_status))
            .await
            .expect("list_workers filtered");
        assert_eq!(filtered.len(), 1);

        // list_workers with status filter (should not match)
        let loading_status = SourceStatus::try_from(1).expect("valid status");
        let filtered_none = node
            .list_workers(None, Some(loading_status))
            .await
            .expect("list_workers filtered none");
        assert!(filtered_none.is_empty());

        // update_status
        let ready = SourceStatus::try_from(3).expect("valid");
        node.update_status(&source_id, worker_id, 0, ready, 2000)
            .await
            .expect("update_status");

        let updated = node
            .get_metadata(&source_id, worker_id)
            .await
            .expect("get after update")
            .expect("should find");
        assert_eq!(updated.workers[0].status, 3);

        // remove_metadata
        node.remove_metadata(&source_id).await.expect("remove");

        let after_remove = node
            .get_metadata(&source_id, worker_id)
            .await
            .expect("get after remove");
        assert!(after_remove.is_none());

        let sources_after = node.list_sources().await.expect("list after remove");
        assert!(sources_after.is_empty());
    }

    #[tokio::test]
    async fn test_incremental_publish_same_worker_id() {
        let node = DhtBackend::new(DhtConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .expect("node");

        let identity = test_identity("model");
        let source_id = crate::source_identity::compute_mx_source_id(&identity);
        let worker_id = "instance-001";

        // Publish rank 0
        let w0 = WorkerMetadata {
            worker_rank: 0,
            metadata_endpoint: "10.0.0.1:5555".to_string(),
            agent_name: "agent-0".to_string(),
            tensors: vec![],
            status: 1,
            updated_at: 1000,
            transfer_engine_session_id: String::new(),
        };
        node.publish_metadata(&identity, worker_id, vec![w0])
            .await
            .expect("pub w0");

        // Publish rank 1 (incremental - same worker_id)
        let w1 = WorkerMetadata {
            worker_rank: 1,
            metadata_endpoint: "10.0.0.2:5556".to_string(),
            agent_name: "agent-1".to_string(),
            tensors: vec![],
            status: 1,
            updated_at: 1000,
            transfer_engine_session_id: String::new(),
        };
        node.publish_metadata(&identity, worker_id, vec![w1])
            .await
            .expect("pub w1");

        // Should now have both workers
        let result = node
            .get_metadata(&source_id, worker_id)
            .await
            .expect("get")
            .expect("found");
        assert_eq!(result.workers.len(), 2);
        assert_eq!(result.workers[0].worker_rank, 0);
        assert_eq!(result.workers[1].worker_rank, 1);
    }

    #[tokio::test]
    async fn test_multiple_worker_ids_same_source() {
        let node = DhtBackend::new(DhtConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .expect("node");

        let identity = test_identity("shared-model");
        let source_id = crate::source_identity::compute_mx_source_id(&identity);
        let wid_a = "instance-aaa";
        let wid_b = "instance-bbb";

        let make_worker = |rank: u32| WorkerMetadata {
            worker_rank: rank,
            metadata_endpoint: format!("10.0.0.{}:5555", rank + 1),
            agent_name: format!("agent-{rank}"),
            tensors: vec![],
            status: 2,
            updated_at: 1000,
            transfer_engine_session_id: String::new(),
        };

        node.publish_metadata(&identity, wid_a, vec![make_worker(0)])
            .await
            .expect("pub wid_a");
        node.publish_metadata(&identity, wid_b, vec![make_worker(1)])
            .await
            .expect("pub wid_b");

        // list_workers should return two entries (one per worker_id/rank)
        let workers = node
            .list_workers(Some(source_id.clone()), None)
            .await
            .expect("list_workers");
        assert_eq!(workers.len(), 2);

        // list_sources should return the source once
        let sources = node.list_sources().await.expect("list_sources");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, source_id);
    }
}
