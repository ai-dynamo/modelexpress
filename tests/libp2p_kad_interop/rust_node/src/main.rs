// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust side of the libp2p Kademlia DHT interop test.
//!
//! Modes:
//!   --mode put  : Start a node, put a record, print the multiaddr, wait.
//!   --mode get  : Connect to a peer, get the record, verify, exit.

use std::time::Duration;

use clap::Parser;
use libp2p::futures::StreamExt;
use libp2p::{
    Multiaddr, StreamProtocol, SwarmBuilder,
    identify, kad,
    kad::{store::MemoryStore, Mode, Record},
    noise, tcp, yamux,
};
use tokio::time::timeout;
use tracing::{info, warn};

#[derive(Parser)]
struct Args {
    /// "put" or "get"
    #[arg(long)]
    mode: String,

    /// Key to store/retrieve
    #[arg(long, default_value = "/mx/model:test-model:worker:0")]
    key: String,

    /// Value to store (put mode only)
    #[arg(long, default_value = r#"{"rank":0,"tensors":[{"name":"layer.0.weight","size":1024}]}"#)]
    value: String,

    /// Peer multiaddr to connect to (get mode only)
    #[arg(long)]
    peer: Option<String>,

    /// How long to wait for operations (seconds)
    #[arg(long, default_value = "30")]
    timeout_secs: u64,
}

#[derive(libp2p::swarm::NetworkBehaviour)]
struct Behaviour {
    kademlia: kad::Behaviour<MemoryStore>,
    identify: identify::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let mut swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let peer_id = key.public().to_peer_id();
            let mut kad_config = kad::Config::new(StreamProtocol::new("/ipfs/kad/1.0.0"));
            kad_config.set_record_ttl(Some(Duration::from_secs(300)));
            let store = MemoryStore::new(peer_id);
            let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);
            kademlia.set_mode(Some(Mode::Server));

            let identify = identify::Behaviour::new(identify::Config::new(
                "/kad-interop/0.1.0".to_string(),
                key.public(),
            ));

            Behaviour {
                kademlia,
                identify,
            }
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Listen on all interfaces, random port
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    // Wait for the listener to be ready and print the address
    let local_addr = loop {
        if let libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } =
            swarm.select_next_some().await
        {
            break address;
        }
    };

    let local_peer_id = *swarm.local_peer_id();
    let full_addr = format!("{local_addr}/p2p/{local_peer_id}");
    info!("Listening on: {full_addr}");

    // Print the address on stdout for the test script to capture
    println!("LISTEN_ADDR={full_addr}");

    match args.mode.as_str() {
        "put" => run_put(&mut swarm, &args).await?,
        "get" => run_get(&mut swarm, &args).await?,
        other => {
            return Err(format!("Unknown mode: {other}").into());
        }
    }

    Ok(())
}

async fn run_put(
    swarm: &mut libp2p::Swarm<Behaviour>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let key = kad::RecordKey::new(&args.key);
    let record = Record {
        key,
        value: args.value.as_bytes().to_vec(),
        publisher: None,
        expires: None,
    };

    // In server mode with just one node, the record goes into local store.
    // When the Python peer connects and queries, it will find it.
    swarm
        .behaviour_mut()
        .kademlia
        .put_record(record, kad::Quorum::One)?;

    info!("Record stored locally, waiting for peers to query...");

    // Process events until timeout - handle incoming queries
    let deadline = Duration::from_secs(args.timeout_secs);
    let _ = timeout(deadline, async {
        loop {
            match swarm.select_next_some().await {
                libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Kademlia(event)) => {
                    info!("Kad event: {event:?}");
                }
                libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Identify(event)) => {
                    info!("Identify event: {event:?}");
                }
                libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    info!("Peer connected: {peer_id}");
                }
                other => {
                    tracing::debug!("Swarm event: {other:?}");
                }
            }
        }
    })
    .await;

    info!("Put node shutting down.");
    Ok(())
}

async fn run_get(
    swarm: &mut libp2p::Swarm<Behaviour>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let peer_addr: Multiaddr = args
        .peer
        .as_ref()
        .ok_or("--peer is required in get mode")?
        .parse()?;

    info!("Dialing peer: {peer_addr}");
    swarm.dial(peer_addr)?;

    // Wait for connection, then look up the record
    let deadline = Duration::from_secs(args.timeout_secs);
    let key = kad::RecordKey::new(&args.key);

    let result = timeout(deadline, async {
        // First wait for connection + identify exchange
        let mut connected = false;
        loop {
            match swarm.select_next_some().await {
                libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    info!("Connected to: {peer_id}");
                    connected = true;
                }
                libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. },
                )) => {
                    info!("Identified peer {peer_id}: {:?}", info.protocols);
                    // Add their listened addresses to kademlia
                    for addr in &info.listen_addrs {
                        swarm
                            .behaviour_mut()
                            .kademlia
                            .add_address(&peer_id, addr.clone());
                    }
                    if connected {
                        break;
                    }
                }
                _ => {}
            }
        }

        // Small delay to let routing tables settle
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Now query for the record
        let query_id = swarm.behaviour_mut().kademlia.get_record(key.clone());
        info!("Started get_record query: {query_id:?}");

        loop {
            match swarm.select_next_some().await {
                libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Kademlia(
                    kad::Event::OutboundQueryProgressed {
                        result: kad::QueryResult::GetRecord(result),
                        ..
                    },
                )) => match result {
                    Ok(kad::GetRecordOk::FoundRecord(kad::PeerRecord { record, .. })) => {
                        let value = String::from_utf8_lossy(&record.value);
                        info!("Got record: key={:?}, value={value}", record.key);
                        println!("RECORD_VALUE={value}");
                        return Ok(value.to_string());
                    }
                    Ok(kad::GetRecordOk::FinishedWithNoAdditionalRecord { .. }) => {
                        info!("Query finished, no more records.");
                    }
                    Err(e) => {
                        warn!("GetRecord error: {e:?}");
                        return Err(format!("GetRecord failed: {e:?}"));
                    }
                },
                libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Kademlia(event)) => {
                    info!("Kad event: {event:?}");
                }
                _ => {}
            }
        }
    })
    .await;

    match result {
        Ok(Ok(value)) => {
            info!("Test passed! Retrieved value: {value}");
            println!("RESULT=OK");
        }
        Ok(Err(e)) => {
            eprintln!("Test failed: {e}");
            println!("RESULT=FAIL");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("Test timed out after {}s", args.timeout_secs);
            println!("RESULT=TIMEOUT");
            std::process::exit(1);
        }
    }

    Ok(())
}
