// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tower middleware that authenticates a wrapped gRPC service via [`AuthState::verify`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use http::{Request, Response};
use tonic::body::Body;
use tonic::server::NamedService;
use tower::{Layer, Service};
use tracing::{debug, warn};

use crate::auth::AuthState;

#[derive(Clone)]
pub struct AuthLayer {
    state: Arc<AuthState>,
}

impl AuthLayer {
    #[must_use]
    pub fn new(state: Arc<AuthState>) -> Self {
        Self { state }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthService {
            inner,
            state: self.state.clone(),
        }
    }
}

#[derive(Clone)]
pub struct AuthService<S> {
    inner: S,
    state: Arc<AuthState>,
}

impl<S: NamedService> NamedService for AuthService<S> {
    const NAME: &'static str = S::NAME;
}

impl<S> Service<Request<Body>> for AuthService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Response<Body>, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let state = self.state.clone();

        Box::pin(async move {
            match state.verify(req.headers()).await {
                Ok(caller) => {
                    debug!(
                        namespace = %caller.namespace,
                        service_account = %caller.service_account,
                        pod = caller.pod_name.as_deref().unwrap_or("?"),
                        path = %req.uri().path(),
                        "auth ok"
                    );
                    let mut req = req;
                    req.extensions_mut().insert(caller);
                    inner.call(req).await
                }
                Err(denial) => {
                    warn!(reason = %denial, path = %req.uri().path(), "auth denied");
                    Ok(denial.into_status().into_http::<Body>())
                }
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::future::{Ready, ready};
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

    use http::header::AUTHORIZATION;
    use http::{HeaderValue, StatusCode};

    use crate::auth::CallerIdentity;
    use crate::auth::test_util::{
        allowed_ref, fake_kube_client, security_config, token_review_for,
    };

    struct Dummy;
    impl NamedService for Dummy {
        const NAME: &'static str = "model_express.p2p.P2pService";
    }

    #[test]
    fn forwards_wrapped_service_name() {
        assert_eq!(
            <AuthService<Dummy> as NamedService>::NAME,
            <Dummy as NamedService>::NAME
        );
    }

    #[derive(Clone, Default)]
    struct RecordingService {
        caller: Arc<Mutex<Option<CallerIdentity>>>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Service<Request<Body>> for RecordingService {
        type Response = Response<Body>;
        type Error = Infallible;
        type Future = Ready<Result<Response<Body>, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request<Body>) -> Self::Future {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.caller.lock().expect("recording service caller lock") =
                req.extensions().get::<CallerIdentity>().cloned();
            ready(Ok(Response::new(Body::empty())))
        }
    }

    #[tokio::test]
    async fn authenticated_request_reaches_inner_service_with_identity() {
        let (client, kube_calls) =
            fake_kube_client(vec![token_review_for("system:serviceaccount:vllm:worker")]);
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = Arc::new(AuthState::new(client, &config));
        let inner = RecordingService::default();
        let caller = inner.caller.clone();
        let calls = inner.calls.clone();
        let mut service = AuthLayer::new(state).layer(inner);
        let request = Request::builder()
            .uri("/model_express.ModelService/ListModelFiles")
            .header(AUTHORIZATION, HeaderValue::from_static("Bearer good-token"))
            .body(Body::empty())
            .expect("request");

        let response = service.call(request).await.expect("inner service response");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let caller = caller
            .lock()
            .expect("recording service caller lock")
            .clone()
            .expect("caller identity extension");
        assert_eq!(caller.namespace, "vllm");
        assert_eq!(caller.service_account, "worker");
        assert_eq!(kube_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn denied_request_returns_grpc_error_without_calling_inner() {
        let (client, kube_calls) = fake_kube_client(Vec::new());
        let config = security_config(vec![allowed_ref("vllm", "worker")]);
        let state = Arc::new(AuthState::new(client, &config));
        let inner = RecordingService::default();
        let calls = inner.calls.clone();
        let mut service = AuthLayer::new(state).layer(inner);
        let request = Request::builder()
            .uri("/model_express.ModelService/ListModelFiles")
            .body(Body::empty())
            .expect("request");

        let response = service.call(request).await.expect("denial response");

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(kube_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            response
                .headers()
                .get("grpc-status")
                .and_then(|value| value.to_str().ok()),
            Some("16")
        );
    }
}
