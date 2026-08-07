# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumer preflight contracts and explicit schema migrations."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .compatibility import CompatibilityRequirements, ensure_compatible
from .errors import CompatibilityError, MigrationError
from .model import SCHEMA_VERSION, ExperienceBatch

SchemaMigration = Callable[[ExperienceBatch], ExperienceBatch]


class SchemaMigrationRegistry:
    """Instance-scoped graph of explicit, deterministic schema migrations."""

    def __init__(
        self,
        migrations: Mapping[tuple[str, str], SchemaMigration] | None = None,
    ) -> None:
        self._migrations: dict[tuple[str, str], SchemaMigration] = {}
        for (source, target), migration in (migrations or {}).items():
            self.register(source, target, migration)

    def register(self, source: str, target: str, migration: SchemaMigration) -> None:
        """Register one directed migration without replacing an existing edge."""

        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or not source
            or not target
            or source == target
        ):
            raise ValueError("migration versions must be distinct non-empty strings")
        if not callable(migration):
            raise TypeError("migration must be callable")
        edge = (source, target)
        if edge in self._migrations:
            raise ValueError(f"migration {source!r} -> {target!r} is already registered")
        self._migrations[edge] = migration

    def migrate(
        self,
        batch: ExperienceBatch,
        target: str = SCHEMA_VERSION,
    ) -> ExperienceBatch:
        """Apply the shortest registered migration path and validate the result."""

        if not isinstance(batch, ExperienceBatch) or not isinstance(target, str) or not target:
            raise TypeError("migrate requires an ExperienceBatch and non-empty target version")
        source = batch.metadata.schema_version
        result = batch
        original_id = batch.experience_id
        for expected_source, expected_target in self._path(source, target):
            try:
                migrated = self._migrations[(expected_source, expected_target)](result)
            except Exception as error:
                raise MigrationError(
                    f"schema migration {expected_source!r} -> {expected_target!r} failed"
                ) from error
            if not isinstance(migrated, ExperienceBatch):
                raise MigrationError("schema migration must return an ExperienceBatch")
            if migrated.metadata.schema_version != expected_target:
                raise MigrationError(
                    f"schema migration {expected_source!r} -> {expected_target!r} returned "
                    f"version {migrated.metadata.schema_version!r}"
                )
            if migrated.experience_id != original_id:
                raise MigrationError("schema migration changed experience_id")
            result = migrated
        if target == SCHEMA_VERSION:
            result.validate()
        return result

    def _path(self, source: str, target: str) -> tuple[tuple[str, str], ...]:
        if source == target:
            return ()
        queue: deque[tuple[str, tuple[tuple[str, str], ...]]] = deque([(source, ())])
        visited = {source}
        while queue:
            version, path = queue.popleft()
            for edge in sorted(edge for edge in self._migrations if edge[0] == version):
                next_version = edge[1]
                next_path = (*path, edge)
                if next_version == target:
                    return next_path
                if next_version not in visited:
                    visited.add(next_version)
                    queue.append((next_version, next_path))
        raise MigrationError(f"no schema migration path from {source!r} to {target!r}")


@dataclass(frozen=True, slots=True)
class ConsumerContract:
    """Schema, field, and semantic requirements checked before publication."""

    requirements: CompatibilityRequirements
    supported_schema_versions: frozenset[str] = frozenset({SCHEMA_VERSION})
    required_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.requirements, CompatibilityRequirements):
            raise TypeError("requirements must be CompatibilityRequirements")
        if not self.supported_schema_versions or any(
            not isinstance(version, str) or not version
            for version in self.supported_schema_versions
        ):
            raise ValueError("supported_schema_versions must contain non-empty versions")
        if SCHEMA_VERSION not in self.supported_schema_versions:
            raise ValueError(f"supported_schema_versions must include {SCHEMA_VERSION!r}")
        if any(
            not isinstance(path, str) or not path or ".." in path for path in self.required_fields
        ):
            raise ValueError("required_fields must contain valid dotted paths")

    def negotiate(
        self,
        batch: ExperienceBatch,
        migrations: SchemaMigrationRegistry | None = None,
    ) -> ExperienceBatch:
        """Migrate when configured, then reject unsupported or incomplete experience."""

        result = batch
        if result.metadata.schema_version != SCHEMA_VERSION:
            if migrations is None:
                raise CompatibilityError(
                    f"consumer does not support schema {result.metadata.schema_version!r}; "
                    f"supported versions: {sorted(self.supported_schema_versions)!r}"
                )
            result = migrations.migrate(result)
        ensure_compatible(result, self.requirements)
        missing = sorted(path for path in self.required_fields if not _field_present(result, path))
        if missing:
            raise CompatibilityError(f"consumer-required fields are missing: {missing!r}")
        return result


def _field_present(root: object, path: str) -> bool:
    return _present(root, tuple(path.split(".")))


def _present(value: object, parts: tuple[str, ...]) -> bool:
    if not parts:
        return value is not None
    if isinstance(value, Mapping):
        for width in range(len(parts), 0, -1):
            key = ".".join(parts[:width])
            if key in value:
                return _present(value[key], parts[width:])
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value) and all(_present(item, parts) for item in value)
    return hasattr(value, parts[0]) and _present(getattr(value, parts[0]), parts[1:])
