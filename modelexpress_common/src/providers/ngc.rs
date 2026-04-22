// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Utils,
    cache::{ModelInfo, ProviderCache, directory_size},
    models::{ModelProvider, WeightFormat},
    providers::ModelProviderTrait,
};
use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, HeaderValue};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

const NGC_API_ENDPOINT_ENV_VAR: &str = "NGC_API_ENDPOINT";
const NGC_AUTH_ENDPOINT_ENV_VAR: &str = "NGC_AUTH_ENDPOINT";
const DEFAULT_NGC_API_BASE: &str = "https://api.ngc.nvidia.com";
const DEFAULT_NGC_AUTHN_BASE: &str = "https://authn.nvidia.com";

fn ngc_api_base() -> String {
    std::env::var(NGC_API_ENDPOINT_ENV_VAR).unwrap_or_else(|_| DEFAULT_NGC_API_BASE.to_string())
}

fn ngc_authn_base() -> String {
    std::env::var(NGC_AUTH_ENDPOINT_ENV_VAR).unwrap_or_else(|_| DEFAULT_NGC_AUTHN_BASE.to_string())
}
const NGC_API_KEY_ENV_VAR: &str = "NGC_API_KEY";
const NGC_CLI_API_KEY_ENV_VAR: &str = "NGC_CLI_API_KEY";
const NGC_CLI_CONFIG_PATH: &str = ".ngc/config";
const MODEL_EXPRESS_CACHE_ENV_VAR: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";
const DEFAULT_NGC_CACHE_SUBDIR: &str = ".cache";
const PAGE_SIZE: u32 = 500;

#[derive(Debug, Clone)]
struct NgcArtifactId {
    org: String,
    team: Option<String>,
    artifact_type: String,
    name: String,
    version: String,
}

/// Parse an NGC artifact name into its components.
///
/// Supported formats:
/// - `org/name/version` — org-level artifact, artifact type defaults to `"models"`
/// - `org/team/name/version` — team artifact, artifact type defaults to `"models"`
/// - `org/team/type/name/version` — explicit artifact type (e.g. `"resources"`)
///
/// Also accepts the `ngc://` URI scheme as a prefix; it is stripped before parsing.
/// URIs of the form `ngc://catalog/org/name:version` are mapped to the team-models format:
/// `catalog` becomes the NGC org, `org` becomes the team, artifact type is `"models"`,
/// and the `name:version` segment is split on `:`.
fn parse_model_name(model_name: &str) -> Result<NgcArtifactId> {
    let model_name = model_name.trim();
    let model_name = model_name.strip_prefix("ngc://").unwrap_or(model_name);
    let parts: Vec<&str> = model_name.split('/').collect();
    if parts.iter().any(|s| *s == "." || *s == "..") {
        anyhow::bail!("NGC model name segments must not be '.' or '..'; got '{model_name}'");
    }
    match parts.len() {
        3 => {
            // If the last segment contains a colon it is a `ngc://` URI of the form
            // `catalog/org/name:version` (e.g. `nim/nvidia/nemotron-3-super:v1`).
            // Map it to the team-models format so the correct NGC API path is used.
            if let Some((name, version)) = parts[2].split_once(':') {
                Ok(NgcArtifactId {
                    org: parts[0].to_string(),
                    team: Some(parts[1].to_string()),
                    artifact_type: "models".to_string(),
                    name: name.to_string(),
                    version: version.to_string(),
                })
            } else {
                Ok(NgcArtifactId {
                    org: parts[0].to_string(),
                    team: None,
                    artifact_type: "models".to_string(),
                    name: parts[1].to_string(),
                    version: parts[2].to_string(),
                })
            }
        }
        4 => Ok(NgcArtifactId {
            org: parts[0].to_string(),
            team: Some(parts[1].to_string()),
            artifact_type: "models".to_string(),
            name: parts[2].to_string(),
            version: parts[3].to_string(),
        }),
        5 => Ok(NgcArtifactId {
            org: parts[0].to_string(),
            team: Some(parts[1].to_string()),
            artifact_type: parts[2].to_string(),
            name: parts[3].to_string(),
            version: parts[4].to_string(),
        }),
        _ => anyhow::bail!(
            "NGC model name must be 'org/name/version', 'org/team/name/version', or \
             'org/team/type/name/version'; got '{model_name}'"
        ),
    }
}

/// Resolve the cache directory for NGC models.
///
/// Priority:
/// 1. Provided `cache_dir` argument
/// 2. `MODEL_EXPRESS_CACHE_DIRECTORY` env var
/// 3. `~/.cache/ngc` (home directory fallback)
fn get_cache_dir(cache_dir: Option<PathBuf>) -> PathBuf {
    if let Some(dir) = cache_dir {
        return dir;
    }
    if let Ok(path) = env::var(MODEL_EXPRESS_CACHE_ENV_VAR) {
        return PathBuf::from(path);
    }
    let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(DEFAULT_NGC_CACHE_SUBDIR)
}

/// Build the local directory path for an NGC artifact.
///
/// Layout: `<cache_root>/ngc/<org>[/<team>]/<artifact_type>/<name>/<version>/`
///
/// Using a path-based layout keeps the cache human-readable and avoids
/// character-escaping issues in directory names.
fn model_dir(cache_root: &Path, id: &NgcArtifactId) -> PathBuf {
    let mut path = cache_root.join("ngc").join(&id.org);
    if let Some(team) = &id.team {
        path = path.join(team);
    }
    path.join(&id.artifact_type)
        .join(&id.name)
        .join(&id.version)
}

/// Resolve the NGC API key from environment or the NGC CLI config file.
///
/// Priority:
/// 1. `NGC_API_KEY` env var
/// 2. `NGC_CLI_API_KEY` env var
/// 3. `~/.ngc/config` (or `$NGC_CLI_HOME/.ngc/config`) — INI or JSON
fn get_ngc_api_key() -> Result<String> {
    for var in [NGC_API_KEY_ENV_VAR, NGC_CLI_API_KEY_ENV_VAR] {
        if let Ok(v) = env::var(var) {
            let trimmed = v.trim().to_string();
            if !trimmed.is_empty() {
                return Ok(trimmed);
            }
        }
    }

    let config_path = env::var("NGC_CLI_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = Utils::get_home_dir().unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home)
        })
        .join(NGC_CLI_CONFIG_PATH);

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read NGC config {config_path:?}"))?;
        if let Some(key) = extract_api_key_from_config(&content) {
            return Ok(key);
        }
    }

    anyhow::bail!(
        "NGC API key not set. Set {NGC_API_KEY_ENV_VAR} or {NGC_CLI_API_KEY_ENV_VAR}, \
         or run 'ngc config set' to configure the NGC CLI."
    )
}

fn extract_api_key_from_config(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') {
        let obj: serde_json::Value = serde_json::from_str(trimmed).ok()?;
        for key in ["apikey", "api_key", "APIKey", "apiKey"] {
            if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        None
    } else {
        // INI format: key = value (sections and comments are ignored)
        trimmed.lines().find_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                return None;
            }
            let (k, v) = line.split_once('=')?;
            let k = k.trim();
            let v = v.trim().trim_matches('"').trim_matches('\'');
            if (k.eq_ignore_ascii_case("apikey") || k.eq_ignore_ascii_case("api_key"))
                && !v.is_empty()
            {
                Some(v.to_string())
            } else {
                None
            }
        })
    }
}

// ── NGC API response types ────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RequestStatus {
    status_code: Option<String>,
    status_description: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaginationInfo {
    total_pages: Option<i32>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelVersion {
    storage_version: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelFileV2 {
    path: String,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactFilesResponse {
    request_status: Option<RequestStatus>,
    pagination_info: Option<PaginationInfo>,
    // V1 storage: presigned URLs returned directly
    urls: Option<Vec<String>>,
    filepath: Option<Vec<String>>,
    // V2 storage: file metadata only; downloads use /files/{path} with Bearer auth
    model_files: Option<Vec<ModelFileV2>>,
    model_version: Option<ModelVersion>,
}

impl ArtifactFilesResponse {
    fn check_request_status(&self) -> Result<()> {
        if let Some(code) = self
            .request_status
            .as_ref()
            .and_then(|rs| rs.status_code.as_deref())
            .filter(|code| *code != "SUCCESS")
        {
            let desc = self
                .request_status
                .as_ref()
                .and_then(|rs| rs.status_description.as_deref())
                .unwrap_or("unknown error");
            anyhow::bail!("NGC API error ({code}): {desc}");
        }
        Ok(())
    }

    fn is_v2_storage(&self) -> bool {
        self.model_version
            .as_ref()
            .and_then(|mv| mv.storage_version.as_deref())
            == Some("V2")
    }

    /// Extract (download_urls, rel_paths).
    ///
    /// V1: presigned URLs come directly from the API response.
    /// V2: download URLs are constructed as `{files_base_url}/{path}`; the NGC API
    ///     redirects to a presigned S3/CDN URL. reqwest strips Authorization on
    ///     cross-origin redirect, so callers must pass Bearer auth for the initial hop.
    fn into_files(self, files_base_url: &str) -> Result<(Vec<String>, Vec<String>)> {
        if self.is_v2_storage() {
            let files = self.model_files.unwrap_or_default();
            let paths: Vec<String> = files.into_iter().map(|f| f.path).collect();
            let urls: Vec<String> = paths
                .iter()
                .map(|p| format!("{files_base_url}/{p}"))
                .collect();
            return Ok((urls, paths));
        }
        let urls = self.urls.unwrap_or_default();
        let paths = self.filepath.unwrap_or_default();
        if urls.len() != paths.len() {
            anyhow::bail!(
                "NGC API returned mismatched url/filepath counts ({} vs {})",
                urls.len(),
                paths.len()
            );
        }
        Ok((urls, paths))
    }
}

// ── Auth helpers ──────────────────────────────────────────────────────────────

/// Exchange an NGC API key for a short-lived Bearer token via authn.nvidia.com.
///
/// Personal keys (`nvapi-` prefix) are used directly as Bearer tokens without
/// exchange. Legacy and service keys are exchanged via authn.nvidia.com with an
/// org-qualified scope (`group/ngc:{org}` or `group/ngc:{org}/{team}`).
async fn fetch_token(
    client: &reqwest::Client,
    api_key: &str,
    id: &NgcArtifactId,
) -> Result<String> {
    if api_key.starts_with("nvapi-") {
        return Ok(api_key.to_string());
    }
    let scope = match &id.team {
        Some(team) => format!("group/ngc:{}/{}", id.org, team),
        None => format!("group/ngc:{}", id.org),
    };
    let url = format!("{}/token", ngc_authn_base());
    let response = client
        .get(&url)
        .query(&[("service", "ngc"), ("scope", scope.as_str())])
        .basic_auth("$oauthtoken", Some(api_key.trim()))
        .send()
        .await
        .context("NGC token request failed")?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("Failed to read NGC token response")?;

    if !status.is_success() {
        anyhow::bail!("NGC token endpoint returned {status}: {body}");
    }

    #[derive(serde::Deserialize)]
    struct TokenResponse {
        token: String,
    }
    let parsed: TokenResponse = serde_json::from_str(&body)
        .with_context(|| format!("Failed to parse NGC token response: {body}"))?;
    Ok(parsed.token)
}

fn bearer_header(token: &str) -> Result<HeaderValue> {
    HeaderValue::try_from(format!("Bearer {}", token.trim()))
        .context("NGC token contains characters that are invalid in an HTTP header")
}

// ── NGC API calls ─────────────────────────────────────────────────────────────

/// Fetch all presigned download URLs for an org-level artifact (no team), paging as needed.
///
/// Org-level responses contain presigned S3/CDN URLs; the Authorization header
/// must NOT be forwarded when fetching those URLs.
async fn fetch_org_artifact_files(
    client: &reqwest::Client,
    auth: &HeaderValue,
    id: &NgcArtifactId,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut all_urls: Vec<String> = Vec::new();
    let mut all_paths: Vec<String> = Vec::new();
    let mut page = 1i32;

    loop {
        let api_base = ngc_api_base();
        let url = format!(
            "{api_base}/v2/org/{}/{}/{}/versions/{}/files?page-size={PAGE_SIZE}&page-number={page}",
            id.org, id.artifact_type, id.name, id.version
        );
        let files_base = format!(
            "{api_base}/v2/org/{}/{}/{}/versions/{}/files",
            id.org, id.artifact_type, id.name, id.version
        );
        debug!("NGC org files: GET {url}");

        let response = client
            .get(&url)
            .header(AUTHORIZATION, auth.clone())
            .send()
            .await
            .context("NGC org files request failed")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read NGC org files response")?;

        if !status.is_success() {
            anyhow::bail!("NGC API returned {status}: {body}");
        }

        let parsed: ArtifactFilesResponse = serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse NGC org files response: {body}"))?;
        parsed.check_request_status()?;

        let total_pages = parsed
            .pagination_info
            .as_ref()
            .and_then(|p| p.total_pages)
            .unwrap_or(1);

        let (urls, paths) = parsed.into_files(&files_base)?;
        all_urls.extend(urls);
        all_paths.extend(paths);

        if page >= total_pages {
            break;
        }
        page = page.saturating_add(1);
    }

    Ok((all_urls, all_paths))
}

/// Fetch all download URLs for a team artifact, paging as needed.
///
/// V1 storage: returns presigned URLs directly; Bearer auth must NOT be forwarded to those URLs.
/// V2 storage: returns file paths; download URLs are `/files/{path}` with Bearer auth.
///   reqwest strips Authorization on cross-origin redirect (to S3/CDN), so it is safe to
///   always pass Bearer auth for team artifacts regardless of storage version.
///
/// Some NGC orgs (e.g. the `nim` public catalog) use UAM which gates the bulk `/files`
/// listing endpoint behind org membership the token may not carry, returning 400
/// "Org contex missing". In that case we fall back to fetching `checksums.blake3` —
/// a per-artifact manifest that is accessible without org membership — and deriving
/// the file list from it. Individual `/files/{path}` downloads are not UAM-gated.
async fn fetch_team_artifact_files(
    client: &reqwest::Client,
    auth: &HeaderValue,
    id: &NgcArtifactId,
    team: &str,
) -> Result<(Vec<String>, Vec<String>)> {
    let files_base = format!(
        "{}/v2/org/{}/team/{}/{}/{}/versions/{}/files",
        ngc_api_base(),
        id.org,
        team,
        id.artifact_type,
        id.name,
        id.version
    );
    let mut all_urls: Vec<String> = Vec::new();
    let mut all_paths: Vec<String> = Vec::new();
    let mut page = 1i32;
    let mut uam_blocked = false;

    loop {
        let url = format!("{files_base}?page-size={PAGE_SIZE}&page-number={page}");
        debug!("NGC team files: GET {url}");

        let response = client
            .get(&url)
            .header(AUTHORIZATION, auth.clone())
            .send()
            .await
            .context("NGC team files request failed")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read NGC team files response")?;

        if status == reqwest::StatusCode::BAD_REQUEST || status == reqwest::StatusCode::UNAUTHORIZED
        {
            warn!(
                "NGC bulk file listing returned {status} for {}/{}/{}/{} — \
                 falling back to checksums.blake3 manifest (response: {body})",
                id.org, team, id.artifact_type, id.name
            );
            uam_blocked = true;
            break;
        }

        if !status.is_success() {
            anyhow::bail!("NGC API returned {status}: {body}");
        }

        let parsed: ArtifactFilesResponse = serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse NGC team files response: {body}"))?;
        parsed.check_request_status()?;

        let total_pages = parsed
            .pagination_info
            .as_ref()
            .and_then(|p| p.total_pages)
            .unwrap_or(1);

        let (urls, paths) = parsed.into_files(&files_base)?;
        all_urls.extend(urls);
        all_paths.extend(paths);

        if page >= total_pages {
            break;
        }
        page = page.saturating_add(1);
    }

    if uam_blocked {
        let (urls, paths) = fetch_files_via_checksum_manifest(client, auth, &files_base).await?;
        all_urls.extend(urls);
        all_paths.extend(paths);
    }

    if all_urls.is_empty() {
        anyhow::bail!("NGC team artifact has no files or is not accessible with the provided key");
    }
    Ok((all_urls, all_paths))
}

/// Fall-back file enumeration for NGC orgs whose bulk `/files` listing endpoint is
/// gated behind UAM org membership.
///
/// `checksums.blake3` is a per-artifact manifest (format: `{hash}  {path}\n`) that
/// is served without org membership checks. We download it, parse the paths, and
/// return `(per_file_urls, paths)` where each URL is `{files_base}/{path}`.
async fn fetch_files_via_checksum_manifest(
    client: &reqwest::Client,
    auth: &HeaderValue,
    files_base: &str,
) -> Result<(Vec<String>, Vec<String>)> {
    let manifest_url = format!("{files_base}/checksums.blake3");
    debug!("NGC checksum manifest: GET {manifest_url}");

    let response = client
        .get(&manifest_url)
        .header(AUTHORIZATION, auth.clone())
        .send()
        .await
        .context("NGC checksum manifest request failed")?;

    let status = response.status();
    let body = response
        .text()
        .await
        .context("Failed to read NGC checksum manifest")?;

    if !status.is_success() {
        anyhow::bail!(
            "NGC checksum manifest unavailable ({status}): {body}. \
             Cannot enumerate files for this artifact."
        );
    }

    // Format: "{blake3_hash}  {relative/path/to/file}\n"
    let paths: Vec<String> = body
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            // Two spaces separate hash from path.
            line.split_once("  ")
                .map(|(_, path)| path.trim().to_string())
        })
        .collect();

    if paths.is_empty() {
        anyhow::bail!("NGC checksum manifest is empty or could not be parsed");
    }

    let urls: Vec<String> = paths.iter().map(|p| format!("{files_base}/{p}")).collect();

    debug!("Enumerated {} files from checksums.blake3", paths.len());
    Ok((urls, paths))
}

/// NGC Bearer tokens expire after ~15 minutes. For large model downloads that
/// take much longer, we must refresh the token periodically. This interval is
/// conservative to avoid hitting expiry mid-transfer.
const TOKEN_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10 * 60);

fn download_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(10))
        .unwrap_or(8)
}

/// Thread-safe Bearer token that auto-refreshes when expired.
struct SharedToken {
    auth: tokio::sync::RwLock<(HeaderValue, std::time::Instant)>,
    api_key: String,
    artifact_id: NgcArtifactId,
}

impl SharedToken {
    fn new(auth: HeaderValue, api_key: String, artifact_id: NgcArtifactId) -> Self {
        Self {
            auth: tokio::sync::RwLock::new((auth, std::time::Instant::now())),
            api_key,
            artifact_id,
        }
    }

    async fn get(&self, client: &reqwest::Client) -> Result<HeaderValue> {
        {
            let guard = self.auth.read().await;
            if guard.1.elapsed() < TOKEN_REFRESH_INTERVAL {
                return Ok(guard.0.clone());
            }
        }
        let mut guard = self.auth.write().await;
        if guard.1.elapsed() < TOKEN_REFRESH_INTERVAL {
            return Ok(guard.0.clone());
        }
        debug!(
            "Refreshing NGC Bearer token (elapsed {:?})",
            guard.1.elapsed()
        );
        let new_token = fetch_token(client, &self.api_key, &self.artifact_id).await?;
        let header = bearer_header(&new_token)?;
        *guard = (header.clone(), std::time::Instant::now());
        Ok(header)
    }
}

/// Download a single file: stream response body to a temp file, then atomically rename.
async fn download_one_file(
    client: &reqwest::Client,
    url: &str,
    rel_path: &str,
    dest: &Path,
    auth: Option<&HeaderValue>,
) -> Result<()> {
    use futures::StreamExt as _;
    use tokio::io::AsyncWriteExt as _;

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Failed to create directory {parent:?}"))?;
    }

    debug!("Downloading NGC file: {rel_path}");
    let mut request = client.get(url);
    if let Some(header) = auth {
        request = request.header(AUTHORIZATION, header.clone());
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("Failed to request {rel_path}"))?;

    let file_status = response.status();
    if !file_status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|e| format!("(failed to read body: {e})"));
        anyhow::bail!("Failed to download {rel_path}: HTTP {file_status} {body}");
    }

    let temp_path = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .with_context(|| format!("Failed to create {temp_path:?}"))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.with_context(|| format!("Error reading response body for {rel_path}"))?;
        file.write_all(&bytes)
            .await
            .with_context(|| format!("Failed to write to {temp_path:?}"))?;
    }

    file.flush()
        .await
        .with_context(|| format!("Failed to flush {temp_path:?}"))?;
    drop(file);

    tokio::fs::rename(&temp_path, &dest)
        .await
        .with_context(|| format!("Failed to rename {temp_path:?} to {dest:?}"))?;

    debug!("Downloaded {}", dest.display());
    Ok(())
}

/// Stream-download a set of (url, relative_path) pairs into `dest_dir` with
/// up to `DOWNLOAD_CONCURRENCY` files in flight.
///
/// Files are written incrementally — no full file is held in memory — making
/// this safe for 100 GB+ model weight files.
///
/// When `token_refresh` is provided, the Bearer token is refreshed automatically
/// via a thread-safe `SharedToken` using double-checked locking so concurrent
/// tasks don't stampede the auth endpoint.
async fn download_files(
    client: &reqwest::Client,
    auth: Option<&HeaderValue>,
    urls: Vec<String>,
    rel_paths: Vec<String>,
    dest_dir: &Path,
    ignore_weights: bool,
    token_refresh: Option<(&str, &NgcArtifactId)>,
) -> Result<usize> {
    use futures::StreamExt as _;

    let shared_token = token_refresh.map(|(api_key, id)| {
        std::sync::Arc::new(SharedToken::new(
            auth.cloned()
                .unwrap_or_else(|| HeaderValue::from_static("")),
            api_key.to_string(),
            id.clone(),
        ))
    });

    let mut cached = 0usize;
    let mut to_download: Vec<(String, String, PathBuf)> = Vec::new();

    for (url, rel_path) in urls.into_iter().zip(rel_paths) {
        let filename = Path::new(&rel_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&rel_path);

        if NgcProvider::is_ignored(filename) || NgcProvider::is_image(Path::new(&rel_path)) {
            debug!("Skipping ignored file: {rel_path}");
            continue;
        }
        if ignore_weights && NgcProvider::is_weight_file(filename) {
            debug!("Skipping weight file (ignore_weights=true): {rel_path}");
            continue;
        }

        let dest = dest_dir.join(&rel_path);
        if dest.exists() {
            debug!("Skipping already-cached file: {rel_path}");
            cached = cached.saturating_add(1);
            continue;
        }

        to_download.push((url, rel_path, dest));
    }

    let concurrency = download_concurrency();
    info!(
        "NGC download: {} files to download, {} already cached ({} concurrent)",
        to_download.len(),
        cached,
        concurrency
    );

    let results: Vec<Result<()>> = futures::stream::iter(to_download)
        .map(|(url, rel_path, dest)| {
            let client = client.clone();
            let token = shared_token.clone();
            let static_auth = auth.cloned();
            async move {
                let auth_header = if let Some(ref tok) = token {
                    Some(tok.get(&client).await?)
                } else {
                    static_auth
                };
                download_one_file(&client, &url, &rel_path, &dest, auth_header.as_ref()).await
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    for result in &results {
        if let Err(e) = result {
            anyhow::bail!("{e}");
        }
    }

    Ok(cached.saturating_add(results.len()))
}

// ── Cache management ──────────────────────────────────────────────────────────

pub(crate) struct NgcProviderCache;

/// Recursively walk `dir`, collecting leaf directories (those containing at
/// least one file) as NGC model entries. The model name is reconstructed from
/// the path relative to `ngc_root` using the inverse of `model_dir`:
///
/// - Depth 4 (`org/artifact_type/name/version`): emits `org/name/version` (3-segment,
///   no team). The `artifact_type` component is dropped because the 3-segment parse
///   path always defaults to `artifact_type = "models"`.
/// - Depth 5 (`org/team/artifact_type/name/version`): emits all 5 parts. The 5-segment
///   parse path preserves the explicit `artifact_type`.
///
/// This ensures that names returned by `list_models` round-trip correctly through
/// `parse_model_name` → `model_dir` back to the original directory.
fn collect_ngc_models(ngc_root: &Path, dir: &Path, models: &mut Vec<ModelInfo>) -> Result<()> {
    let mut has_files = false;
    let mut subdirs: Vec<PathBuf> = Vec::new();

    for entry in fs::read_dir(dir).with_context(|| format!("Failed to read directory {dir:?}"))? {
        let entry = entry.with_context(|| format!("Failed to read entry in {dir:?}"))?;
        let path = entry.path();
        if path.is_file() {
            has_files = true;
        } else if path.is_dir() {
            subdirs.push(path);
        }
    }

    if has_files {
        let rel = dir
            .strip_prefix(ngc_root)
            .with_context(|| format!("Failed to strip prefix {ngc_root:?} from {dir:?}"))?;
        let parts: Vec<&str> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        // Depth 4 = org/artifact_type/name/version (no team): emit org/name/version
        // to match the 3-segment input format (artifact_type is always "models" here).
        // All other depths fall through to a plain join, which handles depth 5 correctly.
        let name = if parts.len() == 4 {
            format!("{}/{}/{}", parts[0], parts[2], parts[3])
        } else {
            parts.join("/")
        };
        models.push(ModelInfo {
            provider: ModelProvider::Ngc,
            name,
            size: directory_size(dir)?,
            path: dir.to_path_buf(),
        });
    } else {
        for subdir in subdirs {
            collect_ngc_models(ngc_root, &subdir, models)?;
        }
    }

    Ok(())
}

impl ProviderCache for NgcProviderCache {
    fn clear_model(&self, cache_root: &Path, model_name: &str) -> Result<()> {
        let id = parse_model_name(model_name)?;
        let path = model_dir(cache_root, &id);
        if path.exists() {
            fs::remove_dir_all(&path).with_context(|| format!("Failed to remove {path:?}"))?;
            info!("Cleared NGC model: {model_name}");
        } else {
            warn!("NGC model '{model_name}' not found in cache");
        }
        Ok(())
    }

    fn resolve_model_path(
        &self,
        cache_root: &Path,
        model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        let id = parse_model_name(model_name)?;
        let path = model_dir(cache_root, &id);
        if path.exists() {
            Ok(path)
        } else {
            anyhow::bail!("NGC model '{model_name}' not found in cache (expected {path:?})");
        }
    }

    fn list_models(&self, cache_root: &Path) -> Result<Vec<ModelInfo>> {
        let ngc_root = cache_root.join("ngc");
        if !ngc_root.exists() {
            return Ok(Vec::new());
        }
        let mut models = Vec::new();
        collect_ngc_models(&ngc_root, &ngc_root, &mut models)?;
        Ok(models)
    }
}

// ── Provider implementation ───────────────────────────────────────────────────

pub struct NgcProvider;

#[async_trait::async_trait]
impl ModelProviderTrait for NgcProvider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
        _weight_format: WeightFormat,
    ) -> Result<PathBuf> {
        let cache_root = get_cache_dir(cache_dir);
        let id = parse_model_name(model_name)?;
        let dest = model_dir(&cache_root, &id);

        tokio::fs::create_dir_all(&dest)
            .await
            .with_context(|| format!("Failed to create model directory {dest:?}"))?;

        let api_key = get_ngc_api_key()?;
        let client = reqwest::Client::builder()
            .build()
            .context("Failed to build HTTP client")?;
        let token = fetch_token(&client, &api_key, &id).await?;
        let auth = bearer_header(&token)?;

        info!(
            "Downloading NGC artifact: org={} team={:?} type={} name={} version={}",
            id.org, id.team, id.artifact_type, id.name, id.version
        );

        let (urls, paths, auth_for_download, token_refresh) = if let Some(team) = &id.team {
            let (u, p) = fetch_team_artifact_files(&client, &auth, &id, team).await?;
            (u, p, Some(auth.clone()), Some((api_key.as_str(), &id)))
        } else {
            let (u, p) = fetch_org_artifact_files(&client, &auth, &id).await?;
            // Org artifact URLs are presigned; do not forward the Authorization header
            // and token refresh is unnecessary (presigned URLs carry their own auth).
            (u, p, None, None)
        };

        if urls.is_empty() {
            anyhow::bail!("NGC artifact '{model_name}' has no downloadable files");
        }

        let downloaded = download_files(
            &client,
            auth_for_download.as_ref(),
            urls,
            paths,
            &dest,
            ignore_weights,
            token_refresh,
        )
        .await?;

        if downloaded == 0 {
            anyhow::bail!("No files downloaded for '{model_name}' (all filtered by ignore rules)");
        }

        info!("Downloaded {downloaded} files for NGC model {model_name}");
        Ok(dest)
    }

    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        let id = match parse_model_name(model_name) {
            Ok(id) => id,
            Err(e) => {
                warn!("Invalid NGC model name '{model_name}': {e}");
                return Ok(());
            }
        };
        let path = model_dir(&cache_dir, &id);
        if path.exists() {
            tokio::fs::remove_dir_all(&path)
                .await
                .with_context(|| format!("Failed to remove {path:?}"))?;
            info!("Deleted NGC model cache: {}", path.display());
        } else {
            info!(
                "NGC model '{model_name}' not found in cache at {}",
                path.display()
            );
        }
        Ok(())
    }

    /// Return the cached model path if it exists and is non-empty, otherwise error.
    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let id = parse_model_name(model_name)?;
        let path = model_dir(&cache_dir, &id);

        if !path.exists() {
            anyhow::bail!("NGC model '{model_name}' not found in cache (expected {path:?})");
        }

        // An empty directory means a previous download was interrupted; treat it as a cache miss
        // so the caller re-downloads rather than silently using an incomplete model.
        let is_empty = match tokio::fs::read_dir(&path).await {
            Ok(mut rd) => rd.next_entry().await.ok().flatten().is_none(),
            Err(_) => false,
        };
        if is_empty {
            anyhow::bail!("NGC model '{model_name}' cache directory is empty (expected {path:?})");
        }

        info!(
            "NGC model '{model_name}' found in cache at {}",
            path.display()
        );
        Ok(path)
    }

    fn provider_name(&self) -> &'static str {
        "NGC"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_three_segments_no_team() {
        let id = parse_model_name("nvidia/llama_v2_70b/1").expect("parse failed");
        assert_eq!(id.org, "nvidia");
        assert!(id.team.is_none());
        assert_eq!(id.artifact_type, "models");
        assert_eq!(id.name, "llama_v2_70b");
        assert_eq!(id.version, "1");
    }

    #[test]
    fn test_parse_four_segments_with_team() {
        let id =
            parse_model_name("nim/meta/llama-3.1-8b-instruct/hf-8c22764").expect("parse failed");
        assert_eq!(id.org, "nim");
        assert_eq!(id.team.as_deref(), Some("meta"));
        assert_eq!(id.artifact_type, "models");
        assert_eq!(id.name, "llama-3.1-8b-instruct");
        assert_eq!(id.version, "hf-8c22764");
    }

    #[test]
    fn test_parse_five_segments_explicit_type() {
        let id = parse_model_name("nim/meta/models/llama-3.1-8b-instruct/1").expect("parse failed");
        assert_eq!(id.org, "nim");
        assert_eq!(id.team.as_deref(), Some("meta"));
        assert_eq!(id.artifact_type, "models");
        assert_eq!(id.name, "llama-3.1-8b-instruct");
        assert_eq!(id.version, "1");
    }

    #[test]
    fn test_parse_five_segments_resource_type() {
        let id = parse_model_name("nvidia/team/resources/my-resource/v1").expect("parse failed");
        assert_eq!(id.artifact_type, "resources");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_model_name("").is_err());
        assert!(parse_model_name("org").is_err());
        assert!(parse_model_name("org/name").is_err());
        assert!(parse_model_name("a/b/c/d/e/f").is_err());
    }

    #[test]
    fn test_parse_rejects_path_traversal() {
        assert!(parse_model_name("../etc/passwd/1").is_err());
        assert!(parse_model_name("nvidia/../../../etc/1").is_err());
        assert!(parse_model_name("nvidia/./llama/1").is_err());
        assert!(parse_model_name("ngc://nim/../model:v1").is_err());
    }

    #[test]
    fn test_model_dir_no_team() {
        let id = NgcArtifactId {
            org: "nvidia".to_string(),
            team: None,
            artifact_type: "models".to_string(),
            name: "llama-3".to_string(),
            version: "1".to_string(),
        };
        let dir = model_dir(Path::new("/cache"), &id);
        assert_eq!(dir, Path::new("/cache/ngc/nvidia/models/llama-3/1"));
    }

    #[test]
    fn test_model_dir_with_team() {
        let id = NgcArtifactId {
            org: "nim".to_string(),
            team: Some("meta".to_string()),
            artifact_type: "models".to_string(),
            name: "llama-3.1-8b-instruct".to_string(),
            version: "hf-8c22764".to_string(),
        };
        let dir = model_dir(Path::new("/cache"), &id);
        assert_eq!(
            dir,
            Path::new("/cache/ngc/nim/meta/models/llama-3.1-8b-instruct/hf-8c22764")
        );
    }

    #[test]
    fn test_extract_api_key_ini() {
        let content = "[DEFAULT]\napikey = mykey123\n";
        assert_eq!(
            extract_api_key_from_config(content),
            Some("mykey123".to_string())
        );
    }

    #[test]
    fn test_extract_api_key_ini_quoted() {
        let content = "api_key = \"mykey456\"\n";
        assert_eq!(
            extract_api_key_from_config(content),
            Some("mykey456".to_string())
        );
    }

    #[test]
    fn test_extract_api_key_json() {
        let content = r#"{"apikey": "jsonkey789"}"#;
        assert_eq!(
            extract_api_key_from_config(content),
            Some("jsonkey789".to_string())
        );
    }

    #[test]
    fn test_extract_api_key_json_alternate_field() {
        let content = r#"{"api_key": "alt_key"}"#;
        assert_eq!(
            extract_api_key_from_config(content),
            Some("alt_key".to_string())
        );
    }

    #[test]
    fn test_extract_api_key_missing() {
        assert!(extract_api_key_from_config("").is_none());
        assert!(extract_api_key_from_config("[DEFAULT]\n# no key\n").is_none());
        assert!(extract_api_key_from_config("{}").is_none());
    }

    #[test]
    fn test_ngc_provider_name() {
        assert_eq!(NgcProvider.provider_name(), "NGC");
    }

    #[tokio::test]
    async fn test_get_model_path_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = NgcProvider
            .get_model_path("nvidia/llama_v2_70b/1", dir.path().to_path_buf())
            .await;
        assert!(result.is_err());
        assert!(
            result
                .expect_err("expected error")
                .to_string()
                .contains("not found in cache")
        );
    }

    #[tokio::test]
    async fn test_get_model_path_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("ngc/nvidia/models/llama_v2_70b/1");
        tokio::fs::create_dir_all(&model_path)
            .await
            .expect("create dirs");
        // Directory must be non-empty for get_model_path to succeed
        tokio::fs::write(model_path.join("config.json"), b"{}")
            .await
            .expect("write sentinel");

        let result = NgcProvider
            .get_model_path("nvidia/llama_v2_70b/1", dir.path().to_path_buf())
            .await;
        assert!(result.is_ok());
        assert_eq!(result.expect("path"), model_path);
    }

    #[tokio::test]
    async fn test_delete_model_not_in_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = NgcProvider
            .delete_model("nvidia/llama_v2_70b/1", dir.path().to_path_buf())
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delete_model_removes_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("ngc/nvidia/models/llama_v2_70b/1");
        tokio::fs::create_dir_all(&model_path)
            .await
            .expect("create dirs");
        assert!(model_path.exists());

        NgcProvider
            .delete_model("nvidia/llama_v2_70b/1", dir.path().to_path_buf())
            .await
            .expect("delete");
        assert!(!model_path.exists());
    }

    #[test]
    fn test_ngc_provider_trait_object() {
        let _provider: Box<dyn ModelProviderTrait> = Box::new(NgcProvider);
    }

    #[test]
    fn test_parse_ngc_uri_scheme_three_segments() {
        // ngc:// prefix with colon-separated version: org/team/name:version
        let id = parse_model_name("ngc://nim/nvidia/nemotron-3-super-120b-a12b:rl-030326-fp8")
            .expect("parse failed");
        assert_eq!(id.org, "nim");
        assert_eq!(id.team.as_deref(), Some("nvidia"));
        assert_eq!(id.artifact_type, "models");
        assert_eq!(id.name, "nemotron-3-super-120b-a12b");
        assert_eq!(id.version, "rl-030326-fp8");
    }

    #[test]
    fn test_parse_ngc_uri_scheme_four_segments() {
        let id = parse_model_name("ngc://nim/meta/llama-3.1-8b-instruct/hf-8c22764")
            .expect("parse failed");
        assert_eq!(id.org, "nim");
        assert_eq!(id.team.as_deref(), Some("meta"));
        assert_eq!(id.artifact_type, "models");
        assert_eq!(id.name, "llama-3.1-8b-instruct");
        assert_eq!(id.version, "hf-8c22764");
    }

    #[test]
    fn test_is_v2_storage_true() {
        let resp = ArtifactFilesResponse {
            request_status: None,
            pagination_info: None,
            urls: None,
            filepath: None,
            model_files: Some(vec![ModelFileV2 {
                path: "config.json".to_string(),
            }]),
            model_version: Some(ModelVersion {
                storage_version: Some("V2".to_string()),
            }),
        };
        assert!(resp.is_v2_storage());
    }

    #[test]
    fn test_is_v2_storage_false_when_absent() {
        let resp = ArtifactFilesResponse {
            request_status: None,
            pagination_info: None,
            urls: Some(vec!["https://example.com/file".to_string()]),
            filepath: Some(vec!["file.bin".to_string()]),
            model_files: None,
            model_version: None,
        };
        assert!(!resp.is_v2_storage());
    }

    #[test]
    fn test_into_files_v2() {
        let resp = ArtifactFilesResponse {
            request_status: None,
            pagination_info: None,
            urls: None,
            filepath: None,
            model_files: Some(vec![
                ModelFileV2 {
                    path: "config.json".to_string(),
                },
                ModelFileV2 {
                    path: "tokenizer.json".to_string(),
                },
            ]),
            model_version: Some(ModelVersion {
                storage_version: Some("V2".to_string()),
            }),
        };
        let (urls, paths) = resp.into_files("https://api.ngc.nvidia.com/v2/org/nim/team/nvidia/models/mymodel/versions/v1/files").expect("into_files");
        assert_eq!(paths, vec!["config.json", "tokenizer.json"]);
        assert!(urls[0].ends_with("/files/config.json"));
        assert!(urls[1].ends_with("/files/tokenizer.json"));
    }

    #[test]
    fn test_into_files_v1() {
        let resp = ArtifactFilesResponse {
            request_status: None,
            pagination_info: None,
            urls: Some(vec![
                "https://s3.example.com/presigned/config.json?token=abc".to_string(),
            ]),
            filepath: Some(vec!["config.json".to_string()]),
            model_files: None,
            model_version: None,
        };
        let (urls, paths) = resp.into_files("unused_base").expect("into_files");
        assert_eq!(paths, vec!["config.json"]);
        assert!(urls[0].contains("presigned"));
    }

    #[test]
    fn test_checksum_manifest_parsing() {
        // Simulate the output of fetch_files_via_checksum_manifest's body parsing logic
        let manifest = "\
            abc123def456  config.json\n\
            789xyz000aaa  tokenizer.json\n\
            \n\
            deadbeef1234  model.safetensors\n\
        ";
        let paths: Vec<String> = manifest
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }
                line.split_once("  ")
                    .map(|(_, path)| path.trim().to_string())
            })
            .collect();
        assert_eq!(
            paths,
            vec!["config.json", "tokenizer.json", "model.safetensors"]
        );
    }

    #[tokio::test]
    async fn test_get_model_path_returns_cached_when_non_empty() {
        // get_model_path is the cache-check entry point (mirrors HuggingFace pattern).
        // download_model always re-downloads; callers should check get_model_path first.
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("ngc/nvidia/models/llama_v2_70b/1");
        tokio::fs::create_dir_all(&model_path)
            .await
            .expect("create dirs");
        tokio::fs::write(model_path.join("config.json"), b"{}")
            .await
            .expect("write");

        let result = NgcProvider
            .get_model_path("nvidia/llama_v2_70b/1", dir.path().to_path_buf())
            .await;
        assert!(result.is_ok());
        assert_eq!(result.expect("path"), model_path);
    }

    #[tokio::test]
    async fn test_get_model_path_fails_for_empty_dir() {
        // An empty directory is treated as a cache miss so interrupted downloads are retried.
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("ngc/nvidia/models/llama_v2_70b/1");
        tokio::fs::create_dir_all(&model_path)
            .await
            .expect("create dirs");

        let result = NgcProvider
            .get_model_path("nvidia/llama_v2_70b/1", dir.path().to_path_buf())
            .await;
        assert!(result.is_err());
        assert!(result.expect_err("error").to_string().contains("empty"));
    }

    /// Org-level model names (3-segment input) must round-trip through
    /// list_models → parse_model_name → model_dir back to the original path.
    #[test]
    fn test_list_models_org_level_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_root = dir.path();

        // Simulate a cached org-level model: nvidia/llama_v2_70b/1
        // model_dir produces: <cache>/ngc/nvidia/models/llama_v2_70b/1/
        let model_path = cache_root.join("ngc/nvidia/models/llama_v2_70b/1");
        std::fs::create_dir_all(&model_path).expect("create dirs");
        std::fs::write(model_path.join("config.json"), b"{}").expect("write");

        let cache = NgcProviderCache;
        let models = cache.list_models(cache_root).expect("list_models");
        assert_eq!(models.len(), 1);

        let name = &models[0].name;
        // Must be 3-segment so it round-trips correctly
        assert_eq!(name, "nvidia/llama_v2_70b/1");

        // Round-trip: parse the name and rebuild the path
        let id = parse_model_name(name).expect("parse");
        let rebuilt = model_dir(cache_root, &id);
        assert_eq!(rebuilt, model_path);
    }

    /// Team-level model names (5-segment form) must round-trip correctly.
    #[test]
    fn test_list_models_team_level_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_root = dir.path();

        // Simulate a cached team-level model: nvidia/nemo/llama/1
        // model_dir produces: <cache>/ngc/nvidia/nemo/models/llama/1/
        let model_path = cache_root.join("ngc/nvidia/nemo/models/llama/1");
        std::fs::create_dir_all(&model_path).expect("create dirs");
        std::fs::write(model_path.join("config.json"), b"{}").expect("write");

        let cache = NgcProviderCache;
        let models = cache.list_models(cache_root).expect("list_models");
        assert_eq!(models.len(), 1);

        let name = &models[0].name;
        // Must be 5-segment: org/team/artifact_type/name/version
        assert_eq!(name, "nvidia/nemo/models/llama/1");

        // Round-trip
        let id = parse_model_name(name).expect("parse");
        let rebuilt = model_dir(cache_root, &id);
        assert_eq!(rebuilt, model_path);
    }

    // ── Additional coverage tests ─────────────────────────────────────────

    #[test]
    fn test_bearer_header_valid() {
        let header = bearer_header("test-token-123").expect("bearer_header");
        assert_eq!(header.to_str().expect("to_str"), "Bearer test-token-123");
    }

    #[test]
    fn test_download_concurrency_returns_reasonable_value() {
        let c = download_concurrency();
        assert!((1..=10).contains(&c));
    }

    #[test]
    fn test_get_cache_dir_explicit() {
        let dir = get_cache_dir(Some(PathBuf::from("/my/cache")));
        assert_eq!(dir, PathBuf::from("/my/cache"));
    }

    #[test]
    fn test_check_request_status_success() {
        let resp = ArtifactFilesResponse {
            request_status: Some(RequestStatus {
                status_code: Some("SUCCESS".to_string()),
                status_description: Some("OK".to_string()),
            }),
            pagination_info: None,
            urls: None,
            filepath: None,
            model_files: None,
            model_version: None,
        };
        assert!(resp.check_request_status().is_ok());
    }

    #[test]
    fn test_check_request_status_error() {
        let resp = ArtifactFilesResponse {
            request_status: Some(RequestStatus {
                status_code: Some("INVALID_REQUEST".to_string()),
                status_description: Some("Bad stuff happened".to_string()),
            }),
            pagination_info: None,
            urls: None,
            filepath: None,
            model_files: None,
            model_version: None,
        };
        let err = resp.check_request_status().expect_err("should fail");
        assert!(err.to_string().contains("INVALID_REQUEST"));
        assert!(err.to_string().contains("Bad stuff happened"));
    }

    #[test]
    fn test_check_request_status_none() {
        let resp = ArtifactFilesResponse {
            request_status: None,
            pagination_info: None,
            urls: None,
            filepath: None,
            model_files: None,
            model_version: None,
        };
        assert!(resp.check_request_status().is_ok());
    }

    #[test]
    fn test_into_files_v1_mismatched_lengths() {
        let resp = ArtifactFilesResponse {
            request_status: None,
            pagination_info: None,
            urls: Some(vec!["a".to_string(), "b".to_string()]),
            filepath: Some(vec!["x".to_string()]),
            model_files: None,
            model_version: None,
        };
        assert!(resp.into_files("unused").is_err());
    }

    #[test]
    fn test_ngc_provider_cache_clear_nonexistent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NgcProviderCache;
        let result = cache.clear_model(dir.path(), "nvidia/llama/1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_ngc_provider_cache_clear_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("ngc/nvidia/models/llama/1");
        std::fs::create_dir_all(&model_path).expect("create dirs");
        std::fs::write(model_path.join("config.json"), b"{}").expect("write");

        let cache = NgcProviderCache;
        cache
            .clear_model(dir.path(), "nvidia/llama/1")
            .expect("clear");
        assert!(!model_path.exists());
    }

    #[test]
    fn test_ngc_provider_cache_resolve_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_path = dir.path().join("ngc/nvidia/models/llama/1");
        std::fs::create_dir_all(&model_path).expect("create dirs");

        let cache = NgcProviderCache;
        let resolved = cache
            .resolve_model_path(dir.path(), "nvidia/llama/1", None)
            .expect("resolve");
        assert_eq!(resolved, model_path);
    }

    #[test]
    fn test_ngc_provider_cache_resolve_nonexistent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NgcProviderCache;
        let result = cache.resolve_model_path(dir.path(), "nvidia/llama/1", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ngc_provider_cache_list_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NgcProviderCache;
        let models = cache.list_models(dir.path()).expect("list");
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_token_nvapi_skips_exchange() {
        let client = reqwest::Client::new();
        let id = NgcArtifactId {
            org: "nvidia".to_string(),
            team: None,
            artifact_type: "models".to_string(),
            name: "test".to_string(),
            version: "1".to_string(),
        };
        let token = fetch_token(&client, "nvapi-test-key-12345", &id)
            .await
            .expect("fetch_token");
        assert_eq!(token, "nvapi-test-key-12345");
    }

    // ── WireMock-based integration tests ──────────────────────────────────

    use crate::test_support::{EnvVarGuard, acquire_env_mutex};
    use std::sync::MutexGuard;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct MockNgcServer<'a> {
        _server: MockServer,
        pub cache_path: PathBuf,
        _api_endpoint_guard: EnvVarGuard<'a>,
        _auth_endpoint_guard: EnvVarGuard<'a>,
        _api_key_guard: EnvVarGuard<'a>,
        _cache_guard: EnvVarGuard<'a>,
    }

    impl<'a> MockNgcServer<'a> {
        async fn new(env_lock: &'a MutexGuard<'static, ()>) -> Self {
            let temp_dir = tempfile::TempDir::new().expect("tempdir");
            let server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path_regex(r"^/token"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"token": "mock-jwt-token"})),
                )
                .mount(&server)
                .await;

            Mock::given(method("GET"))
                .and(path_regex(
                    r"/v2/org/.*/team/.*/models/.*/versions/.*/files$",
                ))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "requestStatus": {"statusCode": "SUCCESS"},
                    "paginationInfo": {"totalPages": 1},
                    "modelVersion": {"storageVersion": "V2"},
                    "modelFiles": [
                        {"path": "config.json"},
                        {"path": "tokenizer.json"}
                    ]
                })))
                .mount(&server)
                .await;

            Mock::given(method("GET"))
                .and(path_regex(
                    r"/v2/org/.*/team/.*/models/.*/versions/.*/files/.+",
                ))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0u8; 64]))
                .mount(&server)
                .await;

            let api_endpoint_guard =
                EnvVarGuard::set(env_lock, NGC_API_ENDPOINT_ENV_VAR, &server.uri());
            let auth_endpoint_guard =
                EnvVarGuard::set(env_lock, NGC_AUTH_ENDPOINT_ENV_VAR, &server.uri());
            let api_key_guard = EnvVarGuard::set(env_lock, NGC_API_KEY_ENV_VAR, "mock-legacy-key");
            let cache_guard = EnvVarGuard::set(
                env_lock,
                MODEL_EXPRESS_CACHE_ENV_VAR,
                temp_dir.path().to_str().expect("path"),
            );

            Self {
                _server: server,
                cache_path: temp_dir.path().to_path_buf(),
                _api_endpoint_guard: api_endpoint_guard,
                _auth_endpoint_guard: auth_endpoint_guard,
                _api_key_guard: api_key_guard,
                _cache_guard: cache_guard,
            }
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_model_v2_with_mock() {
        let env_lock = acquire_env_mutex();
        let mock = MockNgcServer::new(&env_lock).await;

        let result = NgcProvider
            .download_model(
                "nim/nvidia/test-model/v1",
                Some(mock.cache_path.clone()),
                false,
                WeightFormat::default(),
            )
            .await;

        assert!(result.is_ok(), "download failed: {result:?}");
        let model_path = result.expect("path");
        assert!(model_path.join("config.json").exists());
        assert!(model_path.join("tokenizer.json").exists());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_model_skips_ignored_files() {
        let env_lock = acquire_env_mutex();
        let temp_dir = tempfile::TempDir::new().expect("tempdir");
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path_regex(r"^/token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"token": "mock-jwt"})),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(
                r"/v2/org/.*/team/.*/models/.*/versions/.*/files$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "requestStatus": {"statusCode": "SUCCESS"},
                "paginationInfo": {"totalPages": 1},
                "modelVersion": {"storageVersion": "V2"},
                "modelFiles": [
                    {"path": "config.json"},
                    {"path": "README.md"},
                    {"path": ".gitignore"},
                    {"path": "photo.png"}
                ]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(
                r"/v2/org/.*/team/.*/models/.*/versions/.*/files/config\.json",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"{}".to_vec()))
            .mount(&server)
            .await;

        let _api = EnvVarGuard::set(&env_lock, NGC_API_ENDPOINT_ENV_VAR, &server.uri());
        let _auth = EnvVarGuard::set(&env_lock, NGC_AUTH_ENDPOINT_ENV_VAR, &server.uri());
        let _key = EnvVarGuard::set(&env_lock, NGC_API_KEY_ENV_VAR, "mock-key");

        let result = NgcProvider
            .download_model(
                "nim/nvidia/test-model/v1",
                Some(temp_dir.path().to_path_buf()),
                false,
                WeightFormat::default(),
            )
            .await;

        assert!(result.is_ok(), "download failed: {result:?}");
        let model_path = result.expect("path");
        assert!(model_path.join("config.json").exists());
        assert!(!model_path.join("README.md").exists());
        assert!(!model_path.join(".gitignore").exists());
        assert!(!model_path.join("photo.png").exists());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_model_uam_fallback() {
        let env_lock = acquire_env_mutex();
        let temp_dir = tempfile::TempDir::new().expect("tempdir");
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path_regex(r"^/token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"token": "mock-jwt"})),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(
                r"/v2/org/.*/team/.*/models/.*/versions/.*/files$",
            ))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "requestStatus": {
                    "statusCode": "INVALID_REQUEST",
                    "statusDescription": "Org contex missing"
                }
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(r"checksums\.blake3$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("abc123  config.json\ndef456  tokenizer.json\n"),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(
                r"/v2/org/.*/team/.*/models/.*/versions/.*/files/(config|tokenizer)\.json$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0u8; 32]))
            .mount(&server)
            .await;

        let _api = EnvVarGuard::set(&env_lock, NGC_API_ENDPOINT_ENV_VAR, &server.uri());
        let _auth = EnvVarGuard::set(&env_lock, NGC_AUTH_ENDPOINT_ENV_VAR, &server.uri());
        let _key = EnvVarGuard::set(&env_lock, NGC_API_KEY_ENV_VAR, "mock-key");

        let result = NgcProvider
            .download_model(
                "nim/nvidia/test-model/v1",
                Some(temp_dir.path().to_path_buf()),
                false,
                WeightFormat::default(),
            )
            .await;

        assert!(result.is_ok(), "UAM fallback download failed: {result:?}");
        let model_path = result.expect("path");
        assert!(model_path.join("config.json").exists());
        assert!(model_path.join("tokenizer.json").exists());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_download_model_with_nvapi_key() {
        let env_lock = acquire_env_mutex();
        let temp_dir = tempfile::TempDir::new().expect("tempdir");
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path_regex(
                r"/v2/org/.*/team/.*/models/.*/versions/.*/files$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "requestStatus": {"statusCode": "SUCCESS"},
                "paginationInfo": {"totalPages": 1},
                "modelVersion": {"storageVersion": "V2"},
                "modelFiles": [{"path": "config.json"}]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path_regex(
                r"/v2/org/.*/team/.*/models/.*/versions/.*/files/.+",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"{}".to_vec()))
            .mount(&server)
            .await;

        let _api = EnvVarGuard::set(&env_lock, NGC_API_ENDPOINT_ENV_VAR, &server.uri());
        let _auth = EnvVarGuard::set(&env_lock, NGC_AUTH_ENDPOINT_ENV_VAR, &server.uri());
        let _key = EnvVarGuard::set(&env_lock, NGC_API_KEY_ENV_VAR, "nvapi-test-key-xyz");

        let result = NgcProvider
            .download_model(
                "nim/nvidia/test-model/v1",
                Some(temp_dir.path().to_path_buf()),
                false,
                WeightFormat::default(),
            )
            .await;

        assert!(result.is_ok(), "nvapi key download failed: {result:?}");
    }
}
