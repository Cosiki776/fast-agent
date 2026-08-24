from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Self

from fast_agent.transactional.models import RunId
from fast_agent.transactional.run_events import (
    PromotionApplied,
    PromotionRejected,
    RunEvent,
    RunEventKind,
    RunFailed,
    RunProjection,
    RunRecovered,
    RunRecoveryStarted,
    RunStarted,
    RunVerificationFailed,
    RunVerificationStarted,
    RunVerified,
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


def _payload(event: RunEvent) -> dict[str, object]:
    if isinstance(event, RunStarted):
        return {"profile": event.profile}
    if isinstance(event, RunRecoveryStarted):
        return {
            "failure_signature": event.failure_signature,
            "checkpoint_id": event.checkpoint_id,
        }
    if isinstance(event, RunRecovered):
        return {"workspace_version": event.workspace_version}
    if isinstance(event, RunVerificationStarted):
        return {"command": event.command}
    if isinstance(event, RunVerificationFailed):
        return {
            "exit_code": event.exit_code,
            "timed_out": event.timed_out,
            "stdout_artifact_id": event.stdout_artifact_id,
            "stderr_artifact_id": event.stderr_artifact_id,
        }
    if isinstance(event, RunVerified):
        return {
            "workspace_version": event.workspace_version,
            "stdout_artifact_id": event.stdout_artifact_id,
            "stderr_artifact_id": event.stderr_artifact_id,
        }
    if isinstance(event, PromotionApplied):
        return {
            "workspace_version": event.workspace_version,
            "patch_artifact_id": event.patch_artifact_id,
        }
    if isinstance(event, PromotionRejected):
        return {
            "reason": event.reason,
            "patch_artifact_id": event.patch_artifact_id,
        }
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
    elif kind is RunEventKind.VERIFICATION_STARTED:
        event = RunVerificationStarted(
            **fields,
            command=_required_string(payload, "command"),
        )
    elif kind is RunEventKind.VERIFICATION_FAILED:
        event = RunVerificationFailed(
            **fields,
            exit_code=_required_int(payload, "exit_code"),
            timed_out=_required_bool(payload, "timed_out"),
            stdout_artifact_id=_required_string(payload, "stdout_artifact_id"),
            stderr_artifact_id=_required_string(payload, "stderr_artifact_id"),
        )
    elif kind is RunEventKind.VERIFIED:
        event = RunVerified(
            **fields,
            workspace_version=_required_string(payload, "workspace_version"),
            stdout_artifact_id=_required_string(payload, "stdout_artifact_id"),
            stderr_artifact_id=_required_string(payload, "stderr_artifact_id"),
        )
    elif kind is RunEventKind.PROMOTION_APPLIED:
        event = PromotionApplied(
            **fields,
            workspace_version=_required_string(payload, "workspace_version"),
            patch_artifact_id=_required_string(payload, "patch_artifact_id"),
        )
    elif kind is RunEventKind.PROMOTION_REJECTED:
        event = PromotionRejected(
            **fields,
            reason=_required_string(payload, "reason"),
            patch_artifact_id=_required_string(payload, "patch_artifact_id"),
        )
    else:
        event = RunFailed(**fields, reason=_required_string(payload, "reason"))
    return StoredRunEvent(int(row["sequence"]), event)


def _required_string(payload: dict[object, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Run-event payload field '{key}' must be a string")
    return value


def _required_int(payload: dict[object, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Run-event payload field '{key}' must be an integer")
    return value


def _required_bool(payload: dict[object, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Run-event payload field '{key}' must be a boolean")
    return value
