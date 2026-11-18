// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A `kube::Client` served by a closure, for tests that need no cluster.

use std::sync::{Arc, Mutex, PoisonError};

use chrono::Utc;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;
use kube::Client;
use kube::client::Body;
use serde_json::{Value, json};
use tower::BoxError;

use crate::registry::k8s_types::{
    CACHE_FILES_FINALIZER, ModelCacheEntry, ModelCacheEntrySpec, ModelCacheEntryStatus,
};

pub const NAMESPACE: &str = "test-ns";

/// One request the fake API server received.
#[derive(Debug, Clone)]
pub struct Call {
    pub method: http::Method,
    pub path: String,
    pub body: Value,
}

impl Call {
    pub fn is(&self, method: http::Method, path_suffix: &str) -> bool {
        self.method == method && self.path.ends_with(path_suffix)
    }
}

#[derive(Clone, Default)]
pub struct Calls(Arc<Mutex<Vec<Call>>>);

impl Calls {
    pub fn all(&self) -> Vec<Call> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub fn matching(&self, method: http::Method, path_suffix: &str) -> Vec<Call> {
        self.all()
            .into_iter()
            .filter(|call| call.is(method.clone(), path_suffix))
            .collect()
    }
}

/// `respond` answers each request with a status code and a JSON body.
pub fn client(
    respond: impl Fn(&Call) -> (http::StatusCode, Value) + Send + Sync + 'static,
) -> (Client, Calls) {
    let calls = Calls::default();
    let respond = Arc::new(respond);
    let service = tower::service_fn({
        let calls = calls.clone();
        move |request: http::Request<Body>| {
            let calls = calls.clone();
            let respond = respond.clone();
            async move {
                let (parts, body) = request.into_parts();
                let bytes = body.collect_bytes().await?;
                let call = Call {
                    method: parts.method,
                    path: parts.uri.path().to_string(),
                    body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
                };
                let (status, payload) = respond(&call);
                calls
                    .0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(call);
                let response = http::Response::builder()
                    .status(status)
                    .header(http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&payload)?))?;
                Ok::<_, BoxError>(response)
            }
        }
    });
    (Client::new(service, NAMESPACE), calls)
}

/// The body the API server sends with an error status.
pub fn failure(status: http::StatusCode) -> (http::StatusCode, Value) {
    let body = json!({
        "kind": "Status",
        "apiVersion": "v1",
        "status": "Failure",
        "message": status.to_string(),
        "reason": status.canonical_reason().unwrap_or("Unknown"),
        "code": status.as_u16(),
    });
    (status, body)
}

pub fn ok(payload: &impl serde::Serialize) -> (http::StatusCode, Value) {
    (
        http::StatusCode::OK,
        serde_json::to_value(payload).unwrap_or(Value::Null),
    )
}

pub fn list(items: &[ModelCacheEntry]) -> (http::StatusCode, Value) {
    let body = json!({
        "apiVersion": "modelexpress.nvidia.com/v1alpha1",
        "kind": "ModelCacheEntryList",
        "metadata": { "resourceVersion": "1" },
        "items": items,
    });
    (http::StatusCode::OK, body)
}

/// A CR as the API server returns it: named, versioned, and with the finalizer.
pub fn entry(
    name: &str,
    model_name: &str,
    provider: &str,
    status: Option<ModelCacheEntryStatus>,
) -> ModelCacheEntry {
    let mut cr = ModelCacheEntry::new(
        name,
        ModelCacheEntrySpec {
            model_name: model_name.to_string(),
            provider: provider.to_string(),
        },
    );
    cr.metadata.namespace = Some(NAMESPACE.to_string());
    cr.metadata.uid = Some(format!("uid-{name}"));
    cr.metadata.resource_version = Some("7".to_string());
    cr.metadata.finalizers = Some(vec![CACHE_FILES_FINALIZER.to_string()]);
    cr.status = status;
    cr
}

pub fn status(
    phase: &str,
    claim_id: Option<&str>,
    lease_expires_at: Option<&str>,
) -> ModelCacheEntryStatus {
    ModelCacheEntryStatus {
        phase: phase.to_string(),
        claim_id: claim_id.map(String::from),
        lease_expires_at: lease_expires_at.map(String::from),
        ..ModelCacheEntryStatus::default()
    }
}

/// Mark a CR the way a delete request does while a finalizer is present.
pub fn marked(mut cr: ModelCacheEntry) -> ModelCacheEntry {
    cr.metadata.deletion_timestamp = Some(Time(Utc::now()));
    cr
}
