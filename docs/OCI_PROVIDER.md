<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OCI Provider

ModelExpress can download file-oriented OCI model artifacts. The provider supports raw file blobs and simple archive layers. It uses the Rust `oci-client` crate for registry reference parsing, authentication, manifest fetches, and blob streaming.

Each layer supplies model files. The OCI config remains registry metadata; ModelExpress does not turn it into a model manifest. Container filesystem features such as whiteouts, links, and special files are unsupported.

## References

Use `--provider oci` with a registry-qualified reference that includes a tag or digest:

```bash
modelexpress-cli model download registry.example.com/team/model:v1 --provider oci
modelexpress-cli model download oci://registry.example.com/team/model:v1 --provider oci
modelexpress-cli model download registry.example.com/team/model@sha256:<digest> --provider oci
```

The optional `oci://` prefix is stripped before parsing and cache key generation.

## Artifact Format

Raw file layers must include `org.opencontainers.image.title` or `org.cncf.model.filepath`. ModelExpress uses that annotation as the output path relative to the model directory.

Archive layers are supported when their media type is `tar` or `tar+zstd`, including `application/vnd.oci.image.layer.v1.tar+zstd` and model-specific media types ending in `.tar`. Tar member paths are materialized relative to the model directory. Layer titles are labels only; include any desired directory prefixes in the tar member names.

The provider rejects empty paths, absolute paths, `.` and `..` components, backslashes, non-UTF-8 path data, duplicate output paths, symlinks, hardlinks, and special archive entries. README files, dotfiles, and images are skipped. When `ignore_weights=true`, raw weight-file layers are skipped before download and archive-like layers are skipped as whole blobs.

GBuild full-compile artifacts use `application/vnd.groq.gbuild.full-compile.v1` and require a digest reference. They contain four layers:

| Layer | Contents |
| --- | --- |
| `manifest.json` | Runtime Manifest V2 JSON |
| `manifest.v2.capnp.bin` | The same manifest in Cap'n Proto |
| `compile-metadata.tar.zst` | Preset snapshot and optional LPUSim result |
| `payload.tar.zst` | Runtime files for all compile partitions |

ModelExpress checks the layer names, media types, and blob digests. It extracts the complete GBuild tree, including files the generic provider filters. GBuild validates the published manifest and its file set; runtime readers validate model semantics. ModelExpress does not carry a second Manifest V2 parser. These artifacts require a full download (`ignore_weights=false`).

With `--strategy direct --format json`, the successful response includes the materialized `path`.

Example artifact layout:

```bash
oras push registry.example.com/team/model:v1 \
  config.json:application/json \
  tokenizer.json:application/json \
  model.safetensors:application/octet-stream
```

Example archive artifact layout:

```text
layer media type: application/vnd.oci.image.layer.v1.tar+zstd
tar members:
  tokenizer/tokenizer.json
  part-0/program.0.gas
  part-1/program.8.gas
```

This materializes those same tar member paths under the cache entry.

## Authentication

Authentication uses this precedence:

1. `MODEL_EXPRESS_OCI_BEARER_TOKEN`
2. `MODEL_EXPRESS_OCI_USERNAME` plus `MODEL_EXPRESS_OCI_PASSWORD`
3. `MODEL_EXPRESS_OCI_USERNAME` plus `MODEL_EXPRESS_OCI_TOKEN`
4. Docker config credentials and configured helpers (`DOCKER_CONFIG/config.json`, or `~/.docker/config.json`)
5. Application Default Credentials for registries under `*.pkg.dev`
6. Anonymous access for other registries

Partial or empty explicit credentials, malformed Docker config, and failed credential helpers stop the download. Docker identity tokens are unsupported by the OCI client and produce an explicit error. For GAR, `GOOGLE_APPLICATION_CREDENTIALS` can point to a workload-identity configuration.

## Cache Layout

OCI artifacts are cached under the ModelExpress cache root:

```text
<cache-root>/oci/<registry>/<repo...>/tags/<tag>/<mode>/files
<cache-root>/oci/<registry>/<repo...>/digests/<algorithm>-<hex>/<mode>/files
```

Repository slashes are encoded as `%2F`. The mode is `full` or `metadata`, so a metadata-only download cannot satisfy a full request. Each published entry has a `complete` marker beside its `files` directory.

## Publish Behavior

Downloads materialize into a staging directory:

```text
<cache-root>/oci/.tmp/<uuid>/files
```

Raw blobs stream directly into files. Archive blobs stream to a temporary file, extract on a blocking worker, and are removed before publication. The worker retains the staging directory until extraction ends, including when the request is cancelled.

Direct downloads and server streams use the same completion rule: write all requested files, add the completion marker, then rename the staged entry into the final path. Concurrent downloads reuse the first completed entry. Interrupted transfers remain outside the reusable cache. Clear an incomplete or corrupt final entry before retrying.
