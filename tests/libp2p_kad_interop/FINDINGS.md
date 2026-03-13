# libp2p Kademlia DHT Interop Test - Findings

## Summary

We validated that `rust-libp2p` (v0.56) and `py-libp2p` (v0.6.0) can participate in the same
Kademlia DHT network and exchange key-value records bidirectionally. This confirms that a
DHT-based metadata backend for ModelExpress is viable using native implementations in both
languages, with no sidecar or bridge process required.

## Test Results

| Direction | Result | Latency (localhost) |
|---|---|---|
| Rust put -> Python get | PASS | ~2s (including connection setup) |
| Python put -> Rust get | PASS | ~1s (including connection setup) |

The JSON metadata payload round-trips perfectly in both directions with no data loss or
encoding issues.

## Protocol Stack

Both sides must agree on the following protocol stack for interop to work:

| Layer | Protocol | Rust crate | Python package |
|---|---|---|---|
| Transport | TCP | `libp2p-tcp` | built-in |
| Security | Noise (`/noise`) | `libp2p-noise` | `noiseprotocol` |
| Mux | Yamux | `libp2p-yamux` | built-in yamux |
| Discovery | Identify (`/ipfs/id/1.0.0`) | `libp2p-identify` | built-in |
| DHT | Kademlia (`/ipfs/kad/1.0.0`) | `libp2p-kad` | `libp2p.kad_dht` |

## Critical Gotchas

### 1. py-libp2p is trio-native, NOT asyncio

py-libp2p uses [trio](https://trio.readthedocs.io/) as its async runtime, not asyncio.
Attempting to use `asyncio.run()` will fail with:

```
AttributeError: 'RunContext' object has no attribute 'runner'
```

All code must use `trio.run()` and trio-based concurrency primitives (`trio.open_nursery()`,
`trio.sleep()`, `trio.fail_after()`, etc.). The `host.run()` context manager internally uses
`background_trio_service()` which requires a trio event loop.

**Impact for ModelExpress:** If the vLLM loader needs to participate in the DHT directly,
it must either run trio (which conflicts with vLLM's asyncio usage) or use a separate
thread/process with a trio event loop. Alternatively, a thin local bridge (e.g., a trio-based
DHT daemon exposing a simple HTTP or Unix socket API) could decouple the async runtimes.

### 2. Ed25519 keys are REQUIRED for Rust interop

py-libp2p's v0.5.0 release specifically fixed interop with rust-libp2p by switching to
Ed25519 keys. Using secp256k1 keys causes multistream-select handshake failures during
the security upgrade phase:

```
MultiselectClient handshake: read failed: fail to read from multiselect communicator
libp2p.exceptions.MultiError: failed to upgrade mux for peer ...
```

**What to use:**
- Python: `from libp2p.crypto.ed25519 import create_new_key_pair`
- Rust: `SwarmBuilder::with_new_identity()` (generates Ed25519 by default)

Do NOT use `from libp2p.crypto.secp256k1 import create_new_key_pair` for interop scenarios.

### 3. py-libp2p requires namespaced keys with registered validators

Keys used with `put_value()` / `get_value()` in py-libp2p MUST be namespaced in the format
`/namespace/key`. The namespace must have a registered validator, or the DHT will reject
the record.

```python
from libp2p.records.validator import Validator

class MxValidator(Validator):
    def validate(self, key: str, value: bytes) -> None:
        if not value:
            raise ValueError("Value cannot be empty")

    def select(self, key: str, values: list[bytes]) -> int:
        return 0

dht.register_validator("mx", MxValidator())
await dht.put_value("/mx/model:test-model:worker:0", value)
```

The Rust side uses raw `RecordKey` bytes and doesn't enforce namespacing, so the key format
only matters for the Python side. Both sides must use the exact same key bytes.

### 4. host.get_addrs() includes peer ID already

`host.get_addrs()` returns multiaddrs that already contain the `/p2p/{peer_id}` component.
Appending the peer ID again produces a malformed address with a double `/p2p/` suffix that
Rust's multiaddr parser rejects:

```
# WRONG - double /p2p/ suffix
/ip4/127.0.0.1/tcp/44339/p2p/12D3KooW.../p2p/12D3KooW...

# CORRECT
/ip4/127.0.0.1/tcp/44339/p2p/12D3KooW...
```

Use `str(host.get_addrs()[0])` directly, not `f"{addrs[0]}/p2p/{peer_id}"`.

### 5. Single-node Quorum::One fails (but data is still stored locally)

When a Rust node calls `put_record()` with `Quorum::One` and there are no other peers,
the operation reports a quorum failure:

```
PutRecord(Err(QuorumFailed { key: ..., success: [], quorum: 1 }))
```

This is expected - quorum replication requires at least one remote peer. However, the record
IS stored in the local `MemoryStore` before replication is attempted. Connecting peers can
still retrieve it via `get_record()`.

For a production DHT with multiple nodes, this is a non-issue. For testing with a single
node, either ignore the quorum error or use direct local store insertion.

### 6. DHT service must be started with background_trio_service()

The KadDHT object in py-libp2p needs to run as a background service to process incoming
requests (FindNode, GetValue, PutValue). Without `background_trio_service(dht)`, the node
will accept connections but won't respond to DHT queries.

```python
from libp2p.tools.async_service import background_trio_service

dht = KadDHT(host, DHTMode.SERVER)
async with background_trio_service(dht):
    # DHT is now serving requests
    await dht.put_value(key, value)
```

### 7. Building fastecdsa on systems without libgmp-dev

py-libp2p depends on `fastecdsa`, which requires GMP headers to build from source. On
systems without `libgmp-dev` (and no sudo access), you can extract the dev package
without installing it:

```bash
apt-get download libgmp-dev
dpkg-deb -x libgmp-dev*.deb /tmp/gmp-local/
# Symlink the shared lib (static lib is not PIC, can't be used)
ln -sf /usr/lib/x86_64-linux-gnu/libgmp.so.10 /tmp/gmp-local/usr/lib/x86_64-linux-gnu/libgmp.so
# Build with the extracted headers
C_INCLUDE_PATH=/tmp/gmp-local/usr/include LIBRARY_PATH=/tmp/gmp-local/usr/lib/x86_64-linux-gnu \
    pip install libp2p
```

Note: on x86_64 Linux with Python 3.12+, pip may find a pre-built manylinux wheel for
fastecdsa, making this unnecessary. The build-from-source path is a fallback.

### 8. Rust Cargo not in PATH via tunnel/CI

When running via remote execution (VS Code tunnel, CI, etc.), `~/.cargo/bin` may not be
in PATH. Always use full paths or explicitly set PATH:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Architecture Implications for ModelExpress

### What works today

- Rust and Python nodes can join the same DHT and exchange metadata records
- Per-worker keys (`/mx/model:{name}:worker:{rank}`) avoid merge conflicts entirely
- Lightweight metadata (tensor names, addresses, sizes, dtypes, endpoints) fits well within
  DHT value size limits
- No central infrastructure required - nodes ARE the metadata store

### What needs more investigation

1. **trio vs asyncio in vLLM context.** vLLM uses asyncio. py-libp2p uses trio. These don't
   mix trivially. Options:
   - Dedicated trio thread running the DHT, bridging to asyncio via queues
   - Local HTTP/Unix socket daemon (thin trio process, asyncio client)
   - anyio compatibility layer (untested with py-libp2p)

2. **Bootstrap mechanism.** Nodes need at least one known peer to join the DHT.
   - K8s: headless Service provides DNS-based peer discovery
   - Bare metal: static seed node list or mDNS (`py-libp2p` has zeroconf support)
   - Hybrid: bootstrap node as a lightweight always-on process

3. **NIXL blob exchange.** This test only validated small JSON metadata. The actual NIXL
   agent blobs should NOT go through the DHT - only their indexes/endpoints. The NIXL
   handshake happens directly peer-to-peer after the DHT provides addressing information.

4. **Record expiration and refresh.** Kademlia records expire (Rust default: configurable via
   `set_record_ttl()`). Nodes need to periodically republish their metadata to stay
   discoverable. This maps naturally to a health-check pattern.

5. **Multi-node DHT behavior.** This test validated 2-node interop (1 publisher, 1 consumer).
   Production scenarios with 8-64 nodes need testing for routing table convergence, record
   replication, and partition recovery.

## Files

```
tests/libp2p_kad_interop/
  FINDINGS.md              # This document
  run_interop_test.sh      # Orchestrates both test directions
  python_node.py           # py-libp2p 0.6.0 Kademlia node
  requirements.txt         # pip install libp2p multiaddr
  rust_node/
    Cargo.toml             # Standalone crate (excluded from workspace)
    src/main.rs            # rust-libp2p 0.56 Kademlia node
```
