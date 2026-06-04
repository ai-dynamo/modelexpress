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
mod tests {
    use super::*;

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
}
