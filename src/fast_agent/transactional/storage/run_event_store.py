from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Self

from fast_agent.transactional.models import RunId
from fast_agent.transactional.run_events import (
    RunEvent,
    RunEventKind,
    RunFailed,
    RunProjection,
    RunRecovered,
    RunRecoveryStarted,
    RunStarted,
    replay_run,
)
from fast_agent.transactional.storage.event_store import SQLiteEventStore

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


@dataclass(frozen=True, slots=True)
class StoredRunEvent:
    sequence: int
    event: RunEvent


class SQLiteRunEventStore:
    """Append and replay run-level facts from the shared transaction database."""

    def __init__(self, path: Path) -> None:
        with SQLiteEventStore(path):
            pass
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row

    def append(self, event: RunEvent) -> StoredRunEvent:
        current = self.events_for_run(event.run_id)
        replay_run([*(item.event for item in current), event])
        payload = _payload(event)
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO run_events (run_id, event_kind, occurred_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.kind.value,
                    event.occurred_at.isoformat(),
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not assign a run-event sequence")
        return StoredRunEvent(cursor.lastrowid, event)

    def events_for_run(self, run_id: RunId) -> list[StoredRunEvent]:
        rows = self._connection.execute(
            """
            SELECT sequence, run_id, event_kind, occurred_at, payload_json
            FROM run_events WHERE run_id = ? ORDER BY sequence
            """,
            (run_id,),
        ).fetchall()
        return [_stored_event(row) for row in rows]

    def replay(self, run_id: RunId) -> RunProjection:
        return replay_run([item.event for item in self.events_for_run(run_id)])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _payload(event: RunEvent) -> dict[str, str]:
    if isinstance(event, RunStarted):
        return {"profile": event.profile}
    if isinstance(event, RunRecoveryStarted):
        return {
            "failure_signature": event.failure_signature,
            "checkpoint_id": event.checkpoint_id,
        }
    if isinstance(event, RunRecovered):
        return {"workspace_version": event.workspace_version}
    return {"reason": event.reason}


def _stored_event(row: sqlite3.Row) -> StoredRunEvent:
    kind = RunEventKind(str(row["event_kind"]))
    occurred_at = datetime.fromisoformat(str(row["occurred_at"]))
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise ValueError("Run-event payload must be an object")
    fields = {"run_id": RunId(str(row["run_id"])), "occurred_at": occurred_at}
    if kind is RunEventKind.STARTED:
        event: RunEvent = RunStarted(**fields, profile=_required_string(payload, "profile"))
    elif kind is RunEventKind.RECOVERY_STARTED:
        event = RunRecoveryStarted(
            **fields,
            failure_signature=_required_string(payload, "failure_signature"),
            checkpoint_id=_required_string(payload, "checkpoint_id"),
        )
    elif kind is RunEventKind.RECOVERED:
        event = RunRecovered(
            **fields,
            workspace_version=_required_string(payload, "workspace_version"),
        )
    else:
        event = RunFailed(**fields, reason=_required_string(payload, "reason"))
    return StoredRunEvent(int(row["sequence"]), event)


def _required_string(payload: dict[object, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Run-event payload field '{key}' must be a string")
    return value
