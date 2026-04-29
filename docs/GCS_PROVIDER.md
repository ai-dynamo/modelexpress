<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GCS Provider

This document describes the Rust `GcsProvider`, which lets ModelExpress download and cache model files from Google Cloud Storage with the same provider abstraction used by Hugging Face and NGC.

This is separate from the Python ModelStreamer path. ModelStreamer can stream `gs://` URIs directly into a vLLM load path when `MX_MODEL_URI` is set; `GcsProvider` is the ModelExpress model provider used by the CLI, server, cache listing, cache clearing, and gRPC file streaming paths.

## Entry Points

The provider is selected with `ModelProvider::Gcs`, exposed in the CLI as `--provider gcs`.

```bash
modelexpress-cli model download gs://my-bucket/models/qwen/rev-1 --provider gcs
modelexpress-cli model clear --provider gcs gs://my-bucket/models/qwen/rev-1
```

In `server-only` mode, the ModelExpress server performs the GCS download and therefore needs GCS credentials. In `direct` mode, the client performs the download and needs credentials. In `smart-fallback` mode, the client tries the server path first and then falls back to direct download.

## Model Names

GCS model names must be full URLs:

```text
gs://<bucket>/<object-prefix>
```

The provider canonicalizes names by removing a trailing slash from the object prefix. It rejects empty names, missing buckets, missing object prefixes, empty path segments, and `.` or `..` path segments. The bucket component must be a single path segment.

Examples:

```text
gs://model-bucket/deepseek-ai/DeepSeek-V3/rev-1
gs://model-bucket/org/model/snapshots/2026-04-29
```

## Authentication

The provider uses Google Application Default Credentials through the Google Cloud Storage Rust client. Common options are:

- `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON key file
- local ADC from `gcloud auth application-default login`
- GKE Workload Identity or another platform-provided ADC source

The identity needs permission to list and read objects under the model prefix, for example `storage.objects.list` and `storage.objects.get`.

Credentials are local to the process doing the download. They are not sent through ModelExpress gRPC.

## Cache Layout

GCS models are stored under the configured ModelExpress cache root:

```text
<cache-root>/gcs/<bucket>/<object-prefix>/
```

For `gs://model-bucket/deepseek-ai/DeepSeek-V3/rev-1`, the cache path is:

```text
<cache-root>/gcs/model-bucket/deepseek-ai/DeepSeek-V3/rev-1/
```

Provider-private metadata lives under `.mx/` inside the model directory:

```text
<model-dir>/
  config.json
  tokenizer.json
  weights/model.safetensors
  .mx/
    manifest.json
    manifest.lock
    locks/<relative-file>.lock
    parts/<relative-file>.part
```

The `.mx/locks` and `.mx/parts` paths mirror the remote relative object paths so nested model files can be locked and resumed independently.

## Manifest

Before downloading files, the provider lists objects under the requested prefix and writes a cache manifest:

```json
{
  "version": 1,
  "model": "gs://model-bucket/deepseek-ai/DeepSeek-V3/rev-1",
  "files": [
    {
      "path": "config.json",
      "size": 128,
      "crc32c": "00000000",
      "generation": 123456789
    }
  ]
}
```

The manifest is written to `.mx/manifest.json` under an exclusive `.mx/manifest.lock`. It records each downloadable file's relative path, expected size, CRC32C checksum, and GCS generation when available.

An existing manifest is reused if it is parseable, has the expected version/model, has at least one file, has safe relative paths, and has valid CRC32C fields. The provider does not re-list GCS while a valid manifest exists. To force a fresh listing after changing remote objects, clear the cached model or remove its manifest.

## Download Flow

The high-level download flow is:

1. Parse and canonicalize the `gs://` model name.
2. Resolve the cache path under `<cache-root>/gcs/<bucket>/<object-prefix>`.
3. If the manifest is valid and all requested files already exist, return the cached path.
4. Check that this model prefix does not overlap an existing cached ancestor or descendant model.
5. Build or load `.mx/manifest.json` by listing GCS objects under the prefix.
6. Convert manifest entries into download tasks, applying `ignore_weights` when requested.
7. Download files in parallel, with one lock and temp file per destination file.
8. Verify each completed temp file and atomically promote it into the model directory.
9. Recheck cache completeness before reporting success.

Object listing uses pages of up to 1000 objects. Downloads run with up to 8 concurrent workers.

## File Filtering

The manifest includes downloadable objects below the requested prefix, excluding:

- directory marker objects
- dotfiles
- `README.md`
- common image files such as `png`, `jpg`, `jpeg`, `gif`, `webp`, `svg`, `ico`, `bmp`, `tiff`, and `tif`

Unsafe relative paths and paths under the reserved `.mx` metadata directory are rejected instead of cached.

When `ignore_weights` is true, weight files are skipped for that request. Weight detection matches extensions such as `.bin`, `.safetensors`, `.h5`, `.msgpack`, `.ckpt.index`, `.iop`, and `.gas`.

## Concurrency and Resume

The provider uses file locks from the `fd-lock` crate:

- `.mx/manifest.lock` serializes manifest creation.
- `.mx/locks/<relative-file>.lock` serializes each file download.

If another process is already downloading the same file, later processes wait and then reuse the completed file if it exists.

Partial downloads are written to `.mx/parts/<relative-file>.part`. If a partial file is smaller than the expected size and the GCS generation is known, the provider resumes with a range read from the existing byte offset and pins the read to that generation. If the generation is not known, partial files are discarded instead of resumed. Oversized partial files are also discarded.

## Integrity Checks

The provider records remote size and CRC32C in the manifest. Newly downloaded temp files are checked before promotion:

- final byte length must match the manifest size
- local CRC32C must match the manifest CRC32C

Only verified temp files are moved into the model directory. If verification fails, the temp file is removed and the download fails.

Cache hits are cheaper: a valid manifest plus the presence of requested files is enough to satisfy the request. Existing cached files are not re-hashed on every access.

## Cache Listing and Clearing

The shared cache layer delegates GCS operations to `GcsProviderCache`.

Listing walks `<cache-root>/gcs/` by bucket and reports directories that contain a valid manifest and all files required by that manifest. Clearing requires the same full `gs://` model name and removes the corresponding cache directory when it is safe to do so.

The provider refuses to create or delete ambiguous overlapping cache entries. For example, a cached `gs://bucket/a/b` model blocks treating `gs://bucket/a` or `gs://bucket/a/b/c` as a separate cached model until the overlap is removed.
