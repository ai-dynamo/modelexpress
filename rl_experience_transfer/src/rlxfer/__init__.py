# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-independent reinforcement-learning experience transfer."""

from .adapters import AdapterRegistry, ExperienceAdapter, create_adapter
from .api import Delivery, ExperienceConsumer, ExperienceProducer
from .compatibility import CompatibilityReport, CompatibilityRequirements, check_compatibility
from .contracts import ConsumerContract, SchemaMigration, SchemaMigrationRegistry
from .errors import MigrationError
from .model import (
    SCHEMA_VERSION,
    Episode,
    ExperienceBatch,
    ExperienceMetadata,
    PolicyVersion,
    SampleIdentity,
    TensorPayload,
    Trajectory,
    TransferDescriptor,
    Transition,
)
from .serialization import (
    AuthenticatedExperienceSerializer,
    BufferManager,
    DefaultBufferManager,
    ExperienceSerializer,
    JsonExperienceSerializer,
    SerializationLimits,
    SerializedExperience,
)
from .state import (
    DeadLetter,
    DeliveryStateStore,
    InMemoryDeliveryState,
    SqliteDeliveryState,
)
from .tracing import TraceContext, trace_context_from, with_trace_context
from .transport import (
    DeliveryReceipt,
    ExperienceTransport,
    HealthStatus,
    ReceiptResult,
    ReceiptState,
    TransferPlan,
    TransportCapabilities,
    TransportConfig,
    TransportFactory,
    TransportRegistry,
    create_transport,
)
from .transports.fallback import FallbackTransport

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AdapterRegistry",
    "AuthenticatedExperienceSerializer",
    "BufferManager",
    "CompatibilityReport",
    "CompatibilityRequirements",
    "ConsumerContract",
    "DeadLetter",
    "DefaultBufferManager",
    "Delivery",
    "DeliveryReceipt",
    "DeliveryStateStore",
    "Episode",
    "ExperienceAdapter",
    "ExperienceBatch",
    "ExperienceConsumer",
    "ExperienceMetadata",
    "ExperienceProducer",
    "ExperienceSerializer",
    "ExperienceTransport",
    "FallbackTransport",
    "HealthStatus",
    "InMemoryDeliveryState",
    "JsonExperienceSerializer",
    "MigrationError",
    "PolicyVersion",
    "ReceiptResult",
    "ReceiptState",
    "SampleIdentity",
    "SchemaMigration",
    "SchemaMigrationRegistry",
    "SerializationLimits",
    "SerializedExperience",
    "SqliteDeliveryState",
    "TensorPayload",
    "TraceContext",
    "Trajectory",
    "TransferDescriptor",
    "TransferPlan",
    "Transition",
    "TransportCapabilities",
    "TransportConfig",
    "TransportFactory",
    "TransportRegistry",
    "check_compatibility",
    "create_adapter",
    "create_transport",
    "trace_context_from",
    "with_trace_context",
]
