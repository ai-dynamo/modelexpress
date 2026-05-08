<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OCI Provider

ModelExpress can download file-oriented OCI model artifacts. The provider supports raw file blobs and simple archive layers. It uses the Rust `oci-client` crate for registry reference parsing, authentication, manifest fetches, and blob streaming.

OCI support is a materializer, not a container image unpacker. It does not apply whiteouts, root filesystem merges, symlinks, hardlinks, or special files.

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
4. Anonymous access

## Cache Layout

OCI artifacts are cached under the ModelExpress cache root:

```text
<cache-root>/oci/<registry>/<repo...>/tags/<tag>/files
<cache-root>/oci/<registry>/<repo...>/digests/<algorithm>-<hex>/files
```

The provider follows NGC-like cache reuse semantics: `ignore_weights` affects which files are materialized during the download, but it is not part of the cache identity. An existing non-empty `files` directory for the same OCI reference is reused.

## Publish Behavior

Downloads materialize into a staging directory:

```text
<cache-root>/oci/.tmp/<uuid>/files
```

Raw blobs stream directly into files. Archive blobs stream to a temporary blob file under the staging entry, extract into `files`, and are removed before publish.

After all selected blobs are written, the staging entry is atomically renamed into the final cache path. If the final cache entry already exists and has a non-empty `files` directory, ModelExpress removes the staging entry and reuses the existing cache. If the final cache entry exists but is incomplete or corrupt, publish fails with a cache-corruption error and removes the staging entry; clear the corrupt cache entry before retrying.
