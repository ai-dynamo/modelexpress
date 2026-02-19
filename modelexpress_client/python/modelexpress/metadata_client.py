# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metadata client abstraction for ModelExpress P2P transfers.

Supports multiple backends:
- Redis: Original implementation, stores JSON in Redis keys
- Kubernetes: Uses CRDs and ConfigMaps for native K8s integration
- grpc / memory: Uses ModelExpress server gRPC (PublishReady/GetReady)

The backend is selected via the MX_METADATA_BACKEND environment variable:
- "grpc", "memory" (default): Use gRPC to ModelExpress server (for in-memory server)
- "redis": Use Redis backend
- "kubernetes", "k8s", "crd": Use Kubernetes CRD backend
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger("modelexpress.metadata_client")


@dataclass
class WorkerReadyInfo:
    """Information about worker readiness."""
    session_id: str | None
    nixl_ready: bool
    metadata_hash: str | None = None
    timestamp: float | None = None


class MetadataBackend(ABC):
    """Abstract base class for metadata backends."""

    def publish_ready(
        self,
        model_name: str,
        worker_id: int,
        session_id: str | None = None,
        metadata_hash: str = "",
    ) -> bool:
        """Convenience: publish ready with auto-generated session_id if needed."""
        if session_id is None:
            session_id = str(__import__("uuid").uuid4())
        return self.publish_ready_signal(
            model_name=model_name,
            worker_id=worker_id,
            session_id=session_id,
            metadata_hash=metadata_hash or None,
        )
    
    @abstractmethod
    def publish_ready_signal(
        self,
        model_name: str,
        worker_id: int,
        session_id: str,
        metadata_hash: str | None = None,
    ) -> bool:
        """Publish worker ready signal."""
        pass
    
    @abstractmethod
    def get_ready_signal(
        self,
        model_name: str,
        worker_id: int,
    ) -> WorkerReadyInfo | None:
        """Get worker ready signal."""
        pass
    
    @abstractmethod
    def wait_for_ready(
        self,
        model_name: str,
        worker_id: int,
        timeout_seconds: int = 7200,
        poll_interval: int = 10,
    ) -> tuple[bool, str | None, str | None]:
        """
        Wait for source ready signal.
        
        Returns:
            (success, session_id, metadata_hash)
        """
        pass


class RedisBackend(MetadataBackend):
    """Redis-based metadata backend."""
    
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        """Get or create Redis client."""
        if self._client is not None:
            return self._client
        
        try:
            import redis
        except ImportError:
            logger.warning("Redis client not available")
            return None
        
        redis_host = os.environ.get("MX_REDIS_HOST", "modelexpress-server")
        redis_port = int(os.environ.get("MX_REDIS_PORT", "6379"))
        
        try:
            self._client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
            )
            self._client.ping()
            return self._client
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {redis_host}:{redis_port}: {e}")
            return None
    
    def publish_ready_signal(
        self,
        model_name: str,
        worker_id: int,
        session_id: str,
        metadata_hash: str | None = None,
    ) -> bool:
        client = self._get_client()
        if client is None:
            return False
        
        key = f"mx:nixl_ready:{model_name}:worker:{worker_id}"
        
        ready_info = {
            "session_id": session_id,
            "timestamp": time.time(),
            "metadata_hash": metadata_hash,
            "nixl_ready": True,
        }
        
        try:
            # Set with 4 hour TTL
            client.setex(key, 14400, json.dumps(ready_info))
            logger.info(f"[Worker {worker_id}] Published ready flag to Redis")
            return True
        except Exception as e:
            logger.warning(f"[Worker {worker_id}] Failed to publish ready flag: {e}")
            return False
    
    def get_ready_signal(
        self,
        model_name: str,
        worker_id: int,
    ) -> WorkerReadyInfo | None:
        client = self._get_client()
        if client is None:
            return None
        
        key = f"mx:nixl_ready:{model_name}:worker:{worker_id}"
        
        try:
            data = client.get(key)
            if data:
                info = json.loads(data)
                return WorkerReadyInfo(
                    session_id=info.get("session_id"),
                    nixl_ready=info.get("nixl_ready", False),
                    metadata_hash=info.get("metadata_hash"),
                    timestamp=info.get("timestamp"),
                )
        except Exception as e:
            logger.warning(f"[Worker {worker_id}] Error getting ready signal: {e}")
        
        return None
    
    def wait_for_ready(
        self,
        model_name: str,
        worker_id: int,
        timeout_seconds: int = 7200,
        poll_interval: int = 10,
    ) -> tuple[bool, str | None, str | None]:
        client = self._get_client()
        if client is None:
            logger.error(f"[Worker {worker_id}] Redis not available — cannot verify source readiness")
            return False, None, None
        
        key = f"mx:nixl_ready:{model_name}:worker:{worker_id}"
        start_time = time.time()
        
        logger.info(f"[Worker {worker_id}] Waiting for NIXL ready flag at {key}...")
        
        while time.time() - start_time < timeout_seconds:
            try:
                data = client.get(key)
                if data:
                    ready_info = json.loads(data)
                    if ready_info.get("nixl_ready"):
                        session_id = ready_info.get("session_id")
                        metadata_hash = ready_info.get("metadata_hash")
                        logger.info(
                            f"[Worker {worker_id}] Source ready! "
                            f"session={session_id[:8] if session_id else 'N/A'}..."
                        )
                        return True, session_id, metadata_hash
            except Exception as e:
                logger.warning(f"[Worker {worker_id}] Error checking ready flag: {e}")
            
            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:
                logger.info(
                    f"[Worker {worker_id}] Still waiting for source ready "
                    f"({elapsed}s/{timeout_seconds}s)..."
                )
        
        logger.error(
            f"[Worker {worker_id}] Timeout waiting for source ready "
            f"after {timeout_seconds}s"
        )
        return False, None, None


class GrpcBackend(MetadataBackend):
    """gRPC-based backend: uses ModelExpress server (PublishReady/GetReady).
    
    Use with MX_METADATA_BACKEND=memory or grpc when the server uses in-memory backend.
    """
    
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        if self._client is not None:
            return self._client
        from .client import MxClient
        url = os.environ.get("MX_SERVER_ADDRESS", os.environ.get("MODEL_EXPRESS_URL", "modelexpress-server:8001"))
        self._client = MxClient(url)
        return self._client
    
    def publish_ready_signal(
        self,
        model_name: str,
        worker_id: int,
        session_id: str,
        metadata_hash: str | None = None,
    ) -> bool:
        client = self._get_client()
        return client.publish_ready(
            model_name=model_name,
            worker_id=worker_id,
            session_id=session_id,
            metadata_hash=metadata_hash or "",
        )
    
    def get_ready_signal(
        self,
        model_name: str,
        worker_id: int,
    ) -> WorkerReadyInfo | None:
        try:
            response = self._get_client().get_ready(model_name, worker_id)
            if response.found and response.ready:
                return WorkerReadyInfo(
                    session_id=response.session_id,
                    nixl_ready=True,
                    metadata_hash=response.metadata_hash or None,
                    timestamp=None,
                )
        except Exception as e:
            logger.debug("GetReady failed: %s", e)
        return None
    
    def wait_for_ready(
        self,
        model_name: str,
        worker_id: int,
        timeout_seconds: int = 7200,
        poll_interval: int = 10,
    ) -> tuple[bool, str | None, str | None]:
        return self._get_client().wait_for_ready(
            model_name=model_name,
            worker_id=worker_id,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        )


class KubernetesBackend(MetadataBackend):
    """Kubernetes CRD-based metadata backend."""
    
    def __init__(self):
        self._api = None
        self._namespace = os.environ.get(
            "MX_METADATA_NAMESPACE",
            os.environ.get("POD_NAMESPACE", "default")
        )
    
    def _get_api(self):
        """Get or create Kubernetes API client."""
        if self._api is not None:
            return self._api
        
        try:
            from kubernetes import client, config
            
            # Try in-cluster config first, then fall back to kubeconfig
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            self._api = client.CustomObjectsApi()
            return self._api
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")
            return None
    
    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """Convert model name to valid K8s resource name.

        Must match the Rust implementation in k8s_types.rs::sanitize_model_name.
        """
        name = model_name.lower().replace("/", "-").replace("_", "-")
        name = re.sub(r"[^a-z0-9\-.]", "", name)
        return name.strip("-")
    
    def publish_ready_signal(
        self,
        model_name: str,
        worker_id: int,
        session_id: str,
        metadata_hash: str | None = None,
    ) -> bool:
        api = self._get_api()
        if api is None:
            return False
        
        cr_name = self._sanitize_model_name(model_name)
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                cr = api.get_namespaced_custom_object(
                    group="modelexpress.nvidia.com",
                    version="v1alpha1",
                    namespace=self._namespace,
                    plural="modelmetadatas",
                    name=cr_name,
                )
                
                resource_version = cr.get("metadata", {}).get("resourceVersion")
                workers = cr.get("status", {}).get("workers", [])
                
                found = False
                for worker in workers:
                    if worker.get("workerRank") == worker_id:
                        worker["ready"] = True
                        worker["stabilityVerified"] = True
                        worker["sessionId"] = session_id
                        found = True
                        break
                
                if not found:
                    workers.append({
                        "workerRank": worker_id,
                        "ready": True,
                        "stabilityVerified": True,
                        "sessionId": session_id,
                    })
                
                patch = {
                    "metadata": {"resourceVersion": resource_version},
                    "status": {
                        "workers": workers,
                        "phase": "Ready" if all(w.get("stabilityVerified") for w in workers) else "Initializing",
                    }
                }
                
                api.patch_namespaced_custom_object_status(
                    group="modelexpress.nvidia.com",
                    version="v1alpha1",
                    namespace=self._namespace,
                    plural="modelmetadatas",
                    name=cr_name,
                    body=patch,
                )
                
                logger.info(f"[Worker {worker_id}] Published ready flag to K8s CRD")
                return True
                
            except Exception as e:
                if "409" in str(e) and attempt < max_retries - 1:
                    logger.debug(f"[Worker {worker_id}] Conflict on CRD update, retrying ({attempt + 1}/{max_retries})")
                    time.sleep(0.1 * (attempt + 1))
                    continue
                logger.warning(f"[Worker {worker_id}] Failed to publish ready flag to K8s: {e}")
                return False
        
        return False
    
    def get_ready_signal(
        self,
        model_name: str,
        worker_id: int,
    ) -> WorkerReadyInfo | None:
        api = self._get_api()
        if api is None:
            return None
        
        cr_name = self._sanitize_model_name(model_name)
        
        try:
            cr = api.get_namespaced_custom_object(
                group="modelexpress.nvidia.com",
                version="v1alpha1",
                namespace=self._namespace,
                plural="modelmetadatas",
                name=cr_name,
            )
            
            workers = cr.get("status", {}).get("workers", [])
            for worker in workers:
                if worker.get("workerRank") == worker_id:
                    return WorkerReadyInfo(
                        session_id=worker.get("sessionId"),
                        nixl_ready=worker.get("ready", False),
                    )
            
        except Exception as e:
            logger.debug(f"[Worker {worker_id}] Error getting ready signal from K8s: {e}")
        
        return None
    
    def wait_for_ready(
        self,
        model_name: str,
        worker_id: int,
        timeout_seconds: int = 7200,
        poll_interval: int = 10,
    ) -> tuple[bool, str | None, str | None]:
        api = self._get_api()
        if api is None:
            logger.error(
                f"[Worker {worker_id}] K8s client not available — cannot verify source readiness"
            )
            return False, None, None
        
        cr_name = self._sanitize_model_name(model_name)
        start_time = time.time()
        
        logger.info(f"[Worker {worker_id}] Waiting for ModelMetadata CR '{cr_name}' ready...")
        
        # Try watch-based approach first, fall back to polling
        try:
            from kubernetes import watch
            
            w = watch.Watch()
            
            for event in w.stream(
                api.list_namespaced_custom_object,
                group="modelexpress.nvidia.com",
                version="v1alpha1",
                namespace=self._namespace,
                plural="modelmetadatas",
                field_selector=f"metadata.name={cr_name}",
                timeout_seconds=timeout_seconds,
            ):
                if time.time() - start_time > timeout_seconds:
                    break
                
                cr = event.get("object", {})
                workers = cr.get("status", {}).get("workers", [])
                
                for worker in workers:
                    if worker.get("workerRank") == worker_id:
                        if worker.get("ready"):
                            session_id = worker.get("sessionId")
                            logger.info(
                                f"[Worker {worker_id}] Source ready (K8s watch)! "
                                f"session={session_id[:8] if session_id else 'N/A'}..."
                            )
                            w.stop()
                            return True, session_id, None
            
        except Exception as e:
            logger.debug(f"[Worker {worker_id}] K8s watch failed: {e}, falling back to polling")
        
        # Fallback to polling
        while time.time() - start_time < timeout_seconds:
            info = self.get_ready_signal(model_name, worker_id)
            if info and info.nixl_ready:
                logger.info(
                    f"[Worker {worker_id}] Source ready (K8s poll)! "
                    f"session={info.session_id[:8] if info.session_id else 'N/A'}..."
                )
                return True, info.session_id, info.metadata_hash
            
            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:
                logger.info(
                    f"[Worker {worker_id}] Still waiting for source ready "
                    f"({elapsed}s/{timeout_seconds}s)..."
                )
        
        logger.error(
            f"[Worker {worker_id}] Timeout waiting for source ready "
            f"after {timeout_seconds}s"
        )
        return False, None, None


def get_metadata_backend() -> MetadataBackend:
    """
    Get the appropriate metadata backend based on configuration.
    
    Returns:
        MetadataBackend instance (Redis, Kubernetes, or Grpc)
    """
    backend_type = os.environ.get("MX_METADATA_BACKEND", "grpc").lower()
    
    if backend_type in ("kubernetes", "k8s", "crd"):
        logger.info("Using Kubernetes CRD metadata backend")
        return KubernetesBackend()
    elif backend_type == "redis":
        logger.info("Using Redis metadata backend")
        return RedisBackend()
    else:
        logger.info("Using gRPC metadata backend (ModelExpress server)")
        return GrpcBackend()


# Singleton instance
_backend: MetadataBackend | None = None


def get_backend() -> MetadataBackend:
    """Get the singleton metadata backend instance."""
    global _backend
    if _backend is None:
        _backend = get_metadata_backend()
    return _backend
