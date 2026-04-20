// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interop test peer.
//!
//! Advertises one MX peer on the default service type and prints each
//! discovered peer to stdout as a tab-delimited record, for consumption
//! by the Python-side interop test harness.
//!
//! Arguments:
//! - `--instance-name <label>`: the instance label to advertise
//! - `--port <u16>`: the port to advertise
//! - `--ip <addr>`: the IP to advertise (typically 127.0.0.1 for loopback tests)
//! - `--hostname <name>`: the SRV-target hostname (e.g. `rust.local.`)
//! - `--duration-secs <N>`: how long to run before exiting (default 15)
//!
//! Output format (per discovered peer, tab-delimited):
//!     DISCOVERED\t<instance>\t<port>\t<addr1,addr2,...>\t<key1=val1,key2=val2,...>
//!
//! The binary exits cleanly after `duration_secs` or on the usual signals.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::env;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use mx_peer_discovery::mdns::{Config, MdnsDiscovery};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (instance_name, port, ip, hostname, duration_secs) = parse_args()?;

    let mut txt = HashMap::new();
    txt.insert("lang".to_owned(), "rust".to_owned());

    let config = Config {
        service_type: None,
        instance_name: Some(instance_name),
        hostname,
        ip,
        port,
        txt,
        on_resolved: Arc::new(|instance, peer_port, addrs, peer_txt| {
            let addrs_joined = addrs.join(",");
            let txt_joined: Vec<String> =
                peer_txt.iter().map(|(k, v)| format!("{k}={v}")).collect();
            let txt_joined = txt_joined.join(",");
            println!("DISCOVERED\t{instance}\t{peer_port}\t{addrs_joined}\t{txt_joined}");
            // Flush so the reader sees events as they happen, not buffered.
            let _ = std::io::stdout().flush();
        }),
    };

    let _discovery = MdnsDiscovery::start(config)?;
    tokio::time::sleep(Duration::from_secs(duration_secs)).await;
    Ok(())
}

fn parse_args() -> Result<(String, u16, String, String, u64), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut instance_name: Option<String> = None;
    let mut port: u16 = 0;
    let mut ip = String::new();
    let mut hostname = String::new();
    let mut duration_secs: u64 = 15;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--instance-name" => instance_name = args.next(),
            "--port" => {
                port = args
                    .next()
                    .ok_or("--port requires a value")?
                    .parse()
                    .map_err(|_| "--port must be a u16")?;
            }
            "--ip" => ip = args.next().ok_or("--ip requires a value")?,
            "--hostname" => hostname = args.next().ok_or("--hostname requires a value")?,
            "--duration-secs" => {
                duration_secs = args
                    .next()
                    .ok_or("--duration-secs requires a value")?
                    .parse()
                    .map_err(|_| "--duration-secs must be a u64")?;
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let instance_name = instance_name.ok_or("--instance-name required")?;
    if port == 0 {
        return Err("--port required".into());
    }
    if ip.is_empty() {
        return Err("--ip required".into());
    }
    if hostname.is_empty() {
        return Err("--hostname required".into());
    }

    Ok((instance_name, port, ip, hostname, duration_secs))
}
