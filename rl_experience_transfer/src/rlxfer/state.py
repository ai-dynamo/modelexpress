# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumer delivery state with in-memory and durable SQLite implementations."""

from __future__ import annotations

import math
import os
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import time
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DeadLetter:
    """Content-free record of a permanently rejected delivery."""

    experience_id: str
    idempotency_key: str
    reason: str
    attempt: int
    recorded_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.experience_id, self.idempotency_key, self.reason)
        ):
            raise ValueError("dead-letter identifiers and reason must be non-empty")
        if len(self.reason) > 2048:
            raise ValueError("dead-letter reason cannot exceed 2048 characters")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("dead-letter attempt must be a positive integer")
        if (
            isinstance(self.recorded_at, bool)
            or not isinstance(self.recorded_at, (int, float))
            or not math.isfinite(self.recorded_at)
            or self.recorded_at < 0
        ):
            raise ValueError("dead-letter timestamp must be a non-negative finite number")


class DeliveryStateStore(Protocol):
    """Consumer-owned idempotency and dead-letter persistence contract."""

    def was_consumed(self, idempotency_key: str) -> bool: ...

    def mark_consumed(self, idempotency_key: str) -> None: ...

    def record_dead_letter(self, letter: DeadLetter) -> None: ...

    def dead_letters(self, limit: int = 100) -> tuple[DeadLetter, ...]: ...


class InMemoryDeliveryState:
    """Bounded process-local delivery state used by default."""

    def __init__(self, max_entries: int = 4096) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._consumed: set[str] = set()
        self._order: deque[str] = deque()
        self._dead_letters: deque[DeadLetter] = deque(maxlen=max_entries)
        self._lock = Lock()

    def was_consumed(self, idempotency_key: str) -> bool:
        with self._lock:
            return idempotency_key in self._consumed

    def mark_consumed(self, idempotency_key: str) -> None:
        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        with self._lock:
            if idempotency_key in self._consumed:
                return
            self._consumed.add(idempotency_key)
            self._order.append(idempotency_key)
            while len(self._order) > self._max_entries:
                self._consumed.remove(self._order.popleft())

    def record_dead_letter(self, letter: DeadLetter) -> None:
        with self._lock:
            self._dead_letters.append(letter)

    def dead_letters(self, limit: int = 100) -> tuple[DeadLetter, ...]:
        """Return up to ``limit`` newest retained dead letters, newest first."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            return tuple(reversed(self._dead_letters))[:limit]


class SqliteDeliveryState:
    """Multiprocess-safe durable idempotency and dead-letter state."""

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5.0) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive")
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._timeout = timeout
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS consumed "
                "(idempotency_key TEXT PRIMARY KEY, consumed_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS dead_letters "
                "(id INTEGER PRIMARY KEY, experience_id TEXT NOT NULL, "
                "idempotency_key TEXT NOT NULL, reason TEXT NOT NULL, "
                "attempt INTEGER NOT NULL, recorded_at REAL NOT NULL)"
            )
        os.chmod(self._path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=self._timeout)
        connection.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
        return connection

    def was_consumed(self, idempotency_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM consumed WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return row is not None

    def mark_consumed(self, idempotency_key: str) -> None:
        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO consumed VALUES (?, ?)",
                (idempotency_key, time()),
            )

    def record_dead_letter(self, letter: DeadLetter) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO dead_letters "
                "(experience_id, idempotency_key, reason, attempt, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    letter.experience_id,
                    letter.idempotency_key,
                    letter.reason,
                    letter.attempt,
                    letter.recorded_at,
                ),
            )

    def dead_letters(self, limit: int = 100) -> tuple[DeadLetter, ...]:
        """Return up to ``limit`` newest dead letters, newest first."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT experience_id, idempotency_key, reason, attempt, recorded_at "
                "FROM dead_letters ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(DeadLetter(*row) for row in rows)
