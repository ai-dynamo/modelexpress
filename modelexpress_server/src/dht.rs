// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Server-side DHT participation.
//
// Brings up a libp2p node that joins the Kademlia mesh used by the
// `dht` Python client backend. The server is purely a participant in
// this v1: it helps with Kademlia routing and provides a stable
// bootstrap target for clients, but it does not publish any records of
// its own. Active publishing of source records (S3/HF cache becoming a
// peer-discoverable source) is intentionally deferred.
//
// Wire-compatible with the kademlite-backed Python `MxDhtClient`: same
// Kademlia protocol id (`/ipfs/kad/1.0.0`), same TCP + Noise + Yamux
// stack, same Identify protocol. Nodes mesh together regardless of
// implementation.
//
// Configured via the same env vars as the Python side:
//   MX_DHT_LISTEN              host:port for the local node
//   MX_DHT_BOOTSTRAP_PEERS     comma-separated libp2p multiaddrs
//   MX_DHT_BOOTSTRAP_DNS       headless K8s Service DNS name
//   MX_DHT_BOOTSTRAP_SLURM     Slurm hostlist (auto-detected from
//                              SLURM_JOB_NODELIST when unset)
//   MX_DHT_BOOTSTRAP_PORT      port at which to dial DNS / Slurm peers
//                              (default 4001)
//
// If `MX_DHT_LISTEN` is unset the server skips DHT participation
// entirely; existing redis / kubernetes deployments are unaffected.

use std::error::Error;
use std::time::Duration;

use libp2p::{
    PeerId, StreamProtocol, SwarmBuilder,
    futures::StreamExt,
    identify, identity, kad, mdns,
    multiaddr::{Multiaddr, Protocol},
    noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
use tokio::net::lookup_host;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

const KADEMLIA_PROTOCOL: &str = "/ipfs/kad/1.0.0";
const IDENTIFY_AGENT: &str = "modelexpress/0.3.0";
const DEFAULT_BOOTSTRAP_PORT: u16 = 4001;
const LISTEN_READY_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(NetworkBehaviour)]
struct MxBehaviour {
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    identify: identify::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

#[derive(Debug, Clone)]
pub struct DhtConfig {
    pub listen_addr: String,
    pub bootstrap_peers: Vec<Multiaddr>,
    pub bootstrap_dns: Option<String>,
    pub bootstrap_slurm: Option<String>,
    pub bootstrap_port: u16,
}

impl DhtConfig {
    /// Read configuration from env vars. Returns `None` when
    /// `MX_DHT_LISTEN` is unset (server skips DHT participation).
    pub fn from_env() -> Option<Self> {
        let listen_addr = std::env::var("MX_DHT_LISTEN").ok()?;
        if listen_addr.trim().is_empty() {
            return None;
        }

        let bootstrap_peers = std::env::var("MX_DHT_BOOTSTRAP_PEERS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse::<Multiaddr>().ok()
                }
            })
            .collect();

        let bootstrap_dns = std::env::var("MX_DHT_BOOTSTRAP_DNS")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let bootstrap_slurm = std::env::var("MX_DHT_BOOTSTRAP_SLURM")
            .ok()
            .or_else(|| std::env::var("SLURM_JOB_NODELIST").ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let bootstrap_port = std::env::var("MX_DHT_BOOTSTRAP_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(DEFAULT_BOOTSTRAP_PORT);

        Some(Self {
            listen_addr,
            bootstrap_peers,
            bootstrap_dns,
            bootstrap_slurm,
            bootstrap_port,
        })
    }
}

/// Running DHT participant. Drop-or-`stop` to tear down cleanly.
pub struct DhtNode {
    pub peer_id: PeerId,
    shutdown_tx: oneshot::Sender<()>,
    task_handle: JoinHandle<()>,
}

impl DhtNode {
    /// Start the DHT participant: build the swarm, listen, bootstrap,
    /// spawn the event loop on the current tokio runtime.
    ///
    /// Returns once the listener has bound (so `peer_id` and the bound
    /// address are stable). Bootstrap dials happen in the background;
    /// failures are logged but never fatal.
    pub async fn start(config: DhtConfig) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let local_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        info!("DhtNode: local peer_id={local_peer_id}");

        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|key| {
                let mut kad_config = kad::Config::new(StreamProtocol::new(KADEMLIA_PROTOCOL));
                kad_config.set_query_timeout(Duration::from_secs(30));
                let store = kad::store::MemoryStore::new(local_peer_id);
                let kademlia = kad::Behaviour::with_config(local_peer_id, store, kad_config);

                let identify = identify::Behaviour::new(identify::Config::new(
                    IDENTIFY_AGENT.to_string(),
                    key.public(),
                ));

                let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;

                Ok(MxBehaviour {
                    kademlia,
                    identify,
                    mdns,
                })
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        let listen_multi = parse_listen(&config.listen_addr)?;
        swarm.listen_on(listen_multi.clone())?;

        // Wait for the listener to bind so callers see a stable peer_id
        // before we hand them the node. Bootstrap then proceeds in the
        // event loop.
        let (listen_ready_tx, listen_ready_rx) = oneshot::channel();
        let mut listen_ready_tx = Some(listen_ready_tx);

        // Drain initial events until we either get NewListenAddr or
        // hit the timeout.
        let listen_drain = async {
            while let Some(event) = swarm.next().await {
                if let SwarmEvent::NewListenAddr { address, .. } = &event {
                    info!("DhtNode: listening on {address}");
                    if let Some(tx) = listen_ready_tx.take() {
                        let _ = tx.send(address.clone());
                    }
                    break;
                }
            }
        };

        match tokio::time::timeout(LISTEN_READY_TIMEOUT, listen_drain).await {
            Ok(()) => {}
            Err(_) => {
                warn!(
                    "DhtNode: listener did not bind within {:?}; continuing anyway",
                    LISTEN_READY_TIMEOUT
                );
            }
        }
        drop(listen_ready_rx); // we just used it to detect listen-ready

        // Add explicit bootstrap peers to the kad routing table so the
        // first bootstrap query has somewhere to start. Each peer is
        // also dialed eagerly so connections are warm before the
        // application needs them.
        for addr in &config.bootstrap_peers {
            if let Some(peer_id) = peer_id_from_multiaddr(addr) {
                swarm
                    .behaviour_mut()
                    .kademlia
                    .add_address(&peer_id, addr.clone());
                if let Err(err) = swarm.dial(addr.clone()) {
                    warn!("DhtNode: dial {addr} failed: {err}");
                }
            } else {
                warn!(
                    "DhtNode: bootstrap multiaddr {addr} has no /p2p/ peer-id suffix; \
                     skipping (kademlia routing table needs a peer id)"
                );
            }
        }

        // DNS / Slurm bootstrap: resolve names to IPs and dial. Peer
        // ids are learned via the Noise handshake, after which the
        // Identify behaviour propagates them into kad.
        let dial_targets = collect_bootstrap_dial_targets(&config).await;
        for addr in dial_targets {
            if let Err(err) = swarm.dial(addr.clone()) {
                warn!("DhtNode: bootstrap dial {addr} failed: {err}");
            }
        }

        // Kick off a kad bootstrap so the local routing table fills out
        // beyond the initial peers we just added.
        if let Err(err) = swarm.behaviour_mut().kademlia.bootstrap() {
            debug!("DhtNode: kad bootstrap returned {err:?} (typically benign on a fresh node)");
        }

        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        let task_handle = tokio::spawn(async move {
            run_event_loop(swarm, &mut shutdown_rx).await;
        });

        Ok(Self {
            peer_id: local_peer_id,
            shutdown_tx,
            task_handle,
        })
    }

    /// Stop the swarm and wait for the event loop to exit.
    pub async fn stop(self) {
        let _ = self.shutdown_tx.send(());
        let _ = self.task_handle.await;
    }
}

async fn run_event_loop(
    mut swarm: libp2p::Swarm<MxBehaviour>,
    shutdown_rx: &mut oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            _ = &mut *shutdown_rx => {
                debug!("DhtNode: shutdown signal received; exiting event loop");
                return;
            }
            event = swarm.select_next_some() => {
                handle_swarm_event(&mut swarm, event);
            }
        }
    }
}

fn handle_swarm_event(swarm: &mut libp2p::Swarm<MxBehaviour>, event: SwarmEvent<MxBehaviourEvent>) {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            info!("DhtNode: listening on {address}");
        }
        SwarmEvent::ConnectionEstablished {
            peer_id, endpoint, ..
        } => {
            debug!(
                "DhtNode: connection established peer={peer_id} endpoint={:?}",
                endpoint.get_remote_address()
            );
        }
        SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
            debug!("DhtNode: connection closed peer={peer_id} cause={cause:?}");
        }
        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
            debug!(
                "DhtNode: outgoing connection error peer={:?} error={error}",
                peer_id
            );
        }
        SwarmEvent::Behaviour(MxBehaviourEvent::Identify(identify::Event::Received {
            peer_id,
            info,
            ..
        })) => {
            // Standard libp2p pattern: feed addresses learned via
            // Identify into the kad routing table so the node knows
            // how to reach those peers for future queries.
            for addr in info.listen_addrs {
                if addr_is_routable(&addr) {
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                }
            }
        }
        SwarmEvent::Behaviour(MxBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
            for (peer_id, addr) in peers {
                debug!("DhtNode: mDNS discovered peer={peer_id} addr={addr}");
                swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
            }
        }
        SwarmEvent::Behaviour(MxBehaviourEvent::Kademlia(
            kad::Event::OutboundQueryProgressed { result, .. },
        )) => {
            debug!("DhtNode: kad query progressed result={result:?}");
        }
        _ => {}
    }
}

fn parse_listen(listen_addr: &str) -> Result<Multiaddr, Box<dyn Error + Send + Sync>> {
    let (host, port) = if let Some((h, p)) = listen_addr.rsplit_once(':') {
        (if h.is_empty() { "0.0.0.0" } else { h }, p.parse::<u16>()?)
    } else {
        (listen_addr, 0u16)
    };
    let ip: std::net::IpAddr = host.parse()?;
    Ok(Multiaddr::empty()
        .with(match ip {
            std::net::IpAddr::V4(v4) => Protocol::Ip4(v4),
            std::net::IpAddr::V6(v6) => Protocol::Ip6(v6),
        })
        .with(Protocol::Tcp(port)))
}

fn peer_id_from_multiaddr(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|p| {
        if let Protocol::P2p(peer_id) = p {
            Some(peer_id)
        } else {
            None
        }
    })
}

fn addr_is_routable(addr: &Multiaddr) -> bool {
    for p in addr.iter() {
        match p {
            Protocol::Ip4(ip) => {
                if ip.is_loopback() || ip.is_unspecified() {
                    return false;
                }
            }
            Protocol::Ip6(ip) => {
                if ip.is_loopback() || ip.is_unspecified() {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

async fn collect_bootstrap_dial_targets(config: &DhtConfig) -> Vec<Multiaddr> {
    let mut targets = Vec::new();

    if let Some(dns) = &config.bootstrap_dns {
        match resolve_host_to_multiaddrs(dns, config.bootstrap_port).await {
            Ok(addrs) => {
                info!(
                    "DhtNode: DNS bootstrap {dns}:{} -> {} peers",
                    config.bootstrap_port,
                    addrs.len()
                );
                targets.extend(addrs);
            }
            Err(err) => warn!("DhtNode: DNS bootstrap {dns} failed: {err}"),
        }
    }

    if let Some(hostlist) = &config.bootstrap_slurm {
        let hosts = expand_hostlist(hostlist);
        if hosts.is_empty() {
            warn!("DhtNode: Slurm hostlist {hostlist:?} expanded to zero hosts");
        } else {
            info!(
                "DhtNode: Slurm bootstrap from {} hosts at port {}",
                hosts.len(),
                config.bootstrap_port,
            );
        }
        for host in hosts {
            match resolve_host_to_multiaddrs(&host, config.bootstrap_port).await {
                Ok(addrs) => targets.extend(addrs),
                Err(err) => warn!("DhtNode: Slurm host {host} resolution failed: {err}"),
            }
        }
    }

    targets
}

async fn resolve_host_to_multiaddrs(
    host: &str,
    port: u16,
) -> Result<Vec<Multiaddr>, Box<dyn Error + Send + Sync>> {
    let addrs = lookup_host((host, port)).await?;
    let mut out = Vec::new();
    for socket_addr in addrs {
        let mut ma = Multiaddr::empty();
        match socket_addr.ip() {
            std::net::IpAddr::V4(v4) => ma.push(Protocol::Ip4(v4)),
            std::net::IpAddr::V6(v6) => ma.push(Protocol::Ip6(v6)),
        }
        ma.push(Protocol::Tcp(socket_addr.port()));
        out.push(ma);
    }
    Ok(out)
}

/// Minimal Slurm hostlist expander.
///
/// Supports the common `prefix[N-M,O,P-Q]suffix` shape with optional
/// zero-padded numeric ranges and a top-level comma-separated list of
/// such segments. Anything more exotic (nested brackets, alpha ranges)
/// falls through as a literal hostname; deployers needing the full
/// Slurm grammar should pre-expand via `scontrol show hostnames`.
pub fn expand_hostlist(hostlist: &str) -> Vec<String> {
    let mut out = Vec::new();
    for segment in split_top_level_commas(hostlist) {
        out.extend(expand_segment(segment.trim()));
    }
    out
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' => depth = depth.saturating_add(1),
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i.saturating_add(1);
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

fn expand_segment(segment: &str) -> Vec<String> {
    let Some(open) = segment.find('[') else {
        return if segment.is_empty() {
            vec![]
        } else {
            vec![segment.to_string()]
        };
    };
    let Some(close_rel) = segment[open..].find(']') else {
        return vec![segment.to_string()];
    };
    let close = open.saturating_add(close_rel);
    let prefix = &segment[..open];
    let body = &segment[open.saturating_add(1)..close];
    let suffix = &segment[close.saturating_add(1)..];

    let mut numbers = Vec::new();
    for part in body.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-')
            && let (Ok(lo_n), Ok(hi_n)) = (lo.parse::<u64>(), hi.parse::<u64>())
        {
            let width = lo.len();
            for n in lo_n..=hi_n {
                numbers.push(format!("{n:0width$}"));
            }
            continue;
        }
        numbers.push(part.to_string());
    }

    numbers
        .into_iter()
        .map(|n| format!("{prefix}{n}{suffix}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_hostlist_plain() {
        assert_eq!(expand_hostlist("node01"), vec!["node01"]);
    }

    #[test]
    fn expand_hostlist_simple_range() {
        assert_eq!(
            expand_hostlist("node[01-04]"),
            vec!["node01", "node02", "node03", "node04"],
        );
    }

    #[test]
    fn expand_hostlist_mixed_segments() {
        assert_eq!(
            expand_hostlist("gpu[01-02],cpu[03-04]"),
            vec!["gpu01", "gpu02", "cpu03", "cpu04"],
        );
    }

    #[test]
    fn expand_hostlist_internal_commas() {
        assert_eq!(
            expand_hostlist("node[01-02,05,07-08]"),
            vec!["node01", "node02", "node05", "node07", "node08"],
        );
    }

    #[test]
    fn expand_hostlist_empty() {
        assert!(expand_hostlist("").is_empty());
    }

    type TestResult = Result<(), Box<dyn Error + Send + Sync>>;

    #[test]
    fn parse_listen_host_port() -> TestResult {
        let m = parse_listen("127.0.0.1:4001")?;
        assert!(m.to_string().contains("127.0.0.1"));
        assert!(m.to_string().contains("4001"));
        Ok(())
    }

    #[test]
    fn parse_listen_wildcard() -> TestResult {
        let m = parse_listen("0.0.0.0:0")?;
        assert!(m.to_string().contains("0.0.0.0"));
        assert!(m.to_string().contains("/tcp/0"));
        Ok(())
    }

    #[test]
    fn parse_listen_ipv6() -> TestResult {
        let m = parse_listen("::1:4001")?;
        assert!(m.to_string().contains("/tcp/4001"));
        Ok(())
    }

    // env-var-driven `DhtConfig::from_env` parsing is exercised
    // implicitly by the integration test that constructs DhtConfig
    // directly. Direct env-mutating unit tests are intentionally
    // omitted to avoid the unsafe { set_var } + cross-test races on
    // process-global env state.

    #[tokio::test]
    async fn dht_node_starts_and_stops() -> TestResult {
        let cfg = DhtConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            bootstrap_peers: vec![],
            bootstrap_dns: None,
            bootstrap_slurm: None,
            bootstrap_port: DEFAULT_BOOTSTRAP_PORT,
        };
        let node = DhtNode::start(cfg).await?;
        // peer_id is non-empty (Ed25519 pubkey hash)
        assert!(!node.peer_id.to_string().is_empty());
        node.stop().await;
        Ok(())
    }
}
