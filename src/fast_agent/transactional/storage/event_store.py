from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Self, TypedDict

from fast_agent.transactional.events import (
    ToolAuthorized,
    ToolCheckpointed,
    ToolCheckpointFailed,
    ToolCommitted,
    ToolDenied,
    ToolEvent,
    ToolEventKind,
    ToolExecutionFailed,
    ToolExecutionStarted,
    ToolFailed,
    ToolProposed,
    ToolResultStored,
    ToolRollbackStarted,
    ToolRolledBack,
    ToolValidated,
    ToolValidationFailed,
)
from fast_agent.transactional.models import (
    JsonValue,
    RunId,
    ToolCallId,
    ToolEffect,
    TransactionId,
)
from fast_agent.transactional.state_machine import (
    TransactionProjection,
    apply_event,
    replay_transaction,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

SCHEMA_VERSION: Final = 2

_CREATE_EVENTS_TABLE: Final = """
CREATE TABLE IF NOT EXISTS tool_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
)
"""

_CREATE_RUN_SEQUENCE_INDEX: Final = """
CREATE INDEX IF NOT EXISTS idx_tool_events_run_sequence
ON tool_events (run_id, sequence)
"""

_CREATE_TRANSACTION_SEQUENCE_INDEX: Final = """
CREATE INDEX IF NOT EXISTS idx_tool_events_transaction_sequence
ON tool_events (transaction_id, sequence)
"""

_CREATE_TERMINAL_EVENT_INDEX: Final = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_events_one_terminal
ON tool_events (transaction_id)
WHERE event_kind IN ('tool.committed', 'tool.failed')
"""

_CREATE_RUN_EVENTS_TABLE: Final = """
CREATE TABLE IF NOT EXISTS run_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
)
"""

_CREATE_RUN_EVENT_INDEX: Final = """
CREATE INDEX IF NOT EXISTS idx_run_events_run_sequence
ON run_events (run_id, sequence)
"""

_SELECT_FOR_RUN: Final = """
SELECT sequence, run_id, transaction_id, tool_call_id, event_kind, occurred_at, payload_json
FROM tool_events
WHERE run_id = ?
ORDER BY sequence
"""

_SELECT_FOR_TRANSACTION: Final = """
SELECT sequence, run_id, transaction_id, tool_call_id, event_kind, occurred_at, payload_json
FROM tool_events
WHERE transaction_id = ?
ORDER BY sequence
"""

_INSERT_EVENT: Final = """
INSERT INTO tool_events (
    run_id,
    transaction_id,
    tool_call_id,
    event_kind,
    occurred_at,
    payload_json
)
VALUES (?, ?, ?, ?, ?, ?)
"""


class EventStoreError(RuntimeError):
    """Base error raised by the transactional event store."""


class UnsupportedSchemaVersionError(EventStoreError):
    """Raised when a database uses an unsupported event-store schema."""


class DuplicateTerminalEventError(EventStoreError):
    """Raised when concurrent writers try to complete a transaction twice."""


class EventDecodingError(EventStoreError):
    """Raised when a persisted event cannot be decoded."""


@dataclass(frozen=True, slots=True)
class StoredToolEvent:
    """A tool event paired with its database-assigned fact sequence."""

    sequence: int
    event: ToolEvent


@dataclass(frozen=True, slots=True)
class _EventIdentity:
    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId
    occurred_at: datetime


class _EventIdentityKwargs(TypedDict):
    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId
    occurred_at: datetime


class SQLiteEventStore:
    """Append-only SQLite store for transactional tool facts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        try:
            self._initialize_schema()
        except Exception:
            self._connection.close()
            raise

    @property
    def schema_version(self) -> int:
        row = self._connection.execute("PRAGMA user_version").fetchone()
        if row is None:
            raise EventStoreError("SQLite did not return an event-store schema version")
        return _require_int(row[0], "schema version")

    def append(self, event: ToolEvent) -> StoredToolEvent:
        """Validate and append one event in a short write transaction."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            current = self._projection_for_append(event.transaction_id)
            apply_event(current, event)
            cursor = self._connection.execute(
                _INSERT_EVENT,
                (
                    event.run_id,
                    event.transaction_id,
                    event.tool_call_id,
                    event.kind.value,
                    _encode_timestamp(event.occurred_at),
                    _encode_payload(event),
                ),
            )
            sequence = cursor.lastrowid
            if sequence is None:
                raise EventStoreError("SQLite did not assign an event sequence")
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            if event.kind in {ToolEventKind.COMMITTED, ToolEventKind.FAILED}:
                raise DuplicateTerminalEventError(
                    f"Transaction '{event.transaction_id}' already has a terminal event"
                ) from exc
            raise
        except Exception:
            self._connection.rollback()
            raise

        return StoredToolEvent(sequence=sequence, event=event)

    def events_for_run(self, run_id: RunId) -> list[StoredToolEvent]:
        """Return a run's facts in database sequence order."""
        rows = self._connection.execute(_SELECT_FOR_RUN, (run_id,)).fetchall()
        return [_stored_event_from_row(row) for row in rows]

    def events_for_transaction(
        self,
        transaction_id: TransactionId,
    ) -> list[StoredToolEvent]:
        """Return one transaction's facts in database sequence order."""
        rows = self._connection.execute(
            _SELECT_FOR_TRANSACTION,
            (transaction_id,),
        ).fetchall()
        return [_stored_event_from_row(row) for row in rows]

    def replay(self, transaction_id: TransactionId) -> TransactionProjection:
        """Rebuild one transaction from its persisted fact sequence."""
        stored_events = self.events_for_transaction(transaction_id)
        return replay_transaction(item.event for item in stored_events)

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

    def _initialize_schema(self) -> None:
        version = self.schema_version
        if version not in {0, 1, SCHEMA_VERSION}:
            raise UnsupportedSchemaVersionError(
                f"Event-store schema version {version} is not supported; expected {SCHEMA_VERSION}"
            )
        if version == SCHEMA_VERSION:
            return

        with self._connection:
            if version == 0:
                self._connection.execute(_CREATE_EVENTS_TABLE)
                self._connection.execute(_CREATE_RUN_SEQUENCE_INDEX)
                self._connection.execute(_CREATE_TRANSACTION_SEQUENCE_INDEX)
                self._connection.execute(_CREATE_TERMINAL_EVENT_INDEX)
            self._connection.execute(_CREATE_RUN_EVENTS_TABLE)
            self._connection.execute(_CREATE_RUN_EVENT_INDEX)
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _projection_for_append(
        self,
        transaction_id: TransactionId,
    ) -> TransactionProjection | None:
        stored_events = self.events_for_transaction(transaction_id)
        if not stored_events:
            return None
        return replay_transaction(item.event for item in stored_events)


def _encode_payload(event: ToolEvent) -> str:
    payload = _payload_for_event(event)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _encode_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EventStoreError("Event timestamp must include a UTC offset")
    return value.isoformat()


def _payload_for_event(event: ToolEvent) -> dict[str, JsonValue]:
    if isinstance(event, ToolProposed):
        return {
            "tool_name": event.tool_name,
            "arguments": event.arguments,
            "effect": event.effect.value,
        }
    if isinstance(event, ToolCheckpointed):
        return {"checkpoint_id": event.checkpoint_id}
    if isinstance(event, ToolCheckpointFailed | ToolExecutionFailed):
        return {
            "error_type": event.error_type,
            "message": event.message,
        }
    if isinstance(event, ToolResultStored):
        return {
            "artifact_id": event.artifact_id,
            "is_error": event.is_error,
        }
    if isinstance(event, ToolValidationFailed | ToolDenied | ToolRollbackStarted | ToolFailed):
        return {"reason": event.reason}
    if isinstance(event, ToolRolledBack):
        return {"checkpoint_id": event.checkpoint_id}
    if isinstance(
        event,
        ToolValidated | ToolAuthorized | ToolExecutionStarted | ToolCommitted,
    ):
        return {}
    raise TypeError(f"Unsupported transactional event type: {type(event).__name__}")


def _stored_event_from_row(row: sqlite3.Row) -> StoredToolEvent:
    sequence = _require_int(row["sequence"], "sequence")
    identity = _identity_from_row(row)
    kind_text = _require_str(row["event_kind"], "event kind")
    try:
        kind = ToolEventKind(kind_text)
    except ValueError as exc:
        raise EventDecodingError(f"Unknown tool event kind '{kind_text}'") from exc

    payload_text = _require_str(row["payload_json"], "event payload")
    try:
        raw_payload: object = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise EventDecodingError("Event payload is not valid JSON") from exc
    payload = _require_json_object(raw_payload, "event payload")
    event = _event_from_record(kind, identity, payload)
    return StoredToolEvent(sequence=sequence, event=event)


def _identity_from_row(row: sqlite3.Row) -> _EventIdentity:
    occurred_at_text = _require_str(row["occurred_at"], "event timestamp")
    try:
        occurred_at = datetime.fromisoformat(occurred_at_text)
    except ValueError as exc:
        raise EventDecodingError(
            f"Event timestamp '{occurred_at_text}' is not valid ISO 8601"
        ) from exc
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise EventDecodingError("Event timestamp must include a UTC offset")

    return _EventIdentity(
        run_id=RunId(_require_str(row["run_id"], "run ID")),
        transaction_id=TransactionId(_require_str(row["transaction_id"], "transaction ID")),
        tool_call_id=ToolCallId(_require_str(row["tool_call_id"], "tool call ID")),
        occurred_at=occurred_at,
    )


def _event_from_record(
    kind: ToolEventKind,
    identity: _EventIdentity,
    payload: dict[str, JsonValue],
) -> ToolEvent:
    if kind is ToolEventKind.PROPOSED:
        return ToolProposed(
            run_id=identity.run_id,
            transaction_id=identity.transaction_id,
            tool_call_id=identity.tool_call_id,
            occurred_at=identity.occurred_at,
            tool_name=_payload_str(payload, "tool_name"),
            arguments=_payload_object(payload, "arguments"),
            effect=_payload_effect(payload),
        )
    if kind is ToolEventKind.VALIDATED:
        return ToolValidated(**_identity_kwargs(identity))
    if kind is ToolEventKind.AUTHORIZED:
        return ToolAuthorized(**_identity_kwargs(identity))
    if kind is ToolEventKind.CHECKPOINTED:
        return ToolCheckpointed(
            **_identity_kwargs(identity),
            checkpoint_id=_payload_str(payload, "checkpoint_id"),
        )
    if kind is ToolEventKind.CHECKPOINT_FAILED:
        return ToolCheckpointFailed(
            **_identity_kwargs(identity),
            error_type=_payload_str(payload, "error_type"),
            message=_payload_str(payload, "message"),
        )
    if kind is ToolEventKind.EXECUTION_STARTED:
        return ToolExecutionStarted(**_identity_kwargs(identity))
    if kind is ToolEventKind.RESULT_STORED:
        return ToolResultStored(
            **_identity_kwargs(identity),
            artifact_id=_payload_str(payload, "artifact_id"),
            is_error=_payload_bool(payload, "is_error"),
        )
    if kind is ToolEventKind.COMMITTED:
        return ToolCommitted(**_identity_kwargs(identity))
    if kind is ToolEventKind.VALIDATION_FAILED:
        return ToolValidationFailed(
            **_identity_kwargs(identity),
            reason=_payload_str(payload, "reason"),
        )
    if kind is ToolEventKind.DENIED:
        return ToolDenied(
            **_identity_kwargs(identity),
            reason=_payload_str(payload, "reason"),
        )
    if kind is ToolEventKind.EXECUTION_FAILED:
        return ToolExecutionFailed(
            **_identity_kwargs(identity),
            error_type=_payload_str(payload, "error_type"),
            message=_payload_str(payload, "message"),
        )
    if kind is ToolEventKind.ROLLBACK_STARTED:
        return ToolRollbackStarted(
            **_identity_kwargs(identity),
            reason=_payload_str(payload, "reason"),
        )
    if kind is ToolEventKind.ROLLED_BACK:
        return ToolRolledBack(
            **_identity_kwargs(identity),
            checkpoint_id=_payload_str(payload, "checkpoint_id"),
        )
    return ToolFailed(
        **_identity_kwargs(identity),
        reason=_payload_str(payload, "reason"),
    )


def _identity_kwargs(identity: _EventIdentity) -> _EventIdentityKwargs:
    return {
        "run_id": identity.run_id,
        "transaction_id": identity.transaction_id,
        "tool_call_id": identity.tool_call_id,
        "occurred_at": identity.occurred_at,
    }


def _payload_str(payload: dict[str, JsonValue], key: str) -> str:
    return _require_str(payload.get(key), f"event payload field '{key}'")


def _payload_bool(payload: dict[str, JsonValue], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise EventDecodingError(f"Event payload field '{key}' must be a boolean")
    return value


def _payload_object(payload: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    return _require_json_object(payload.get(key), f"event payload field '{key}'")


def _payload_effect(payload: dict[str, JsonValue]) -> ToolEffect:
    value = _payload_str(payload, "effect")
    try:
        return ToolEffect(value)
    except ValueError as exc:
        raise EventDecodingError(f"Unknown tool effect '{value}'") from exc


def _require_json_object(value: object, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise EventDecodingError(f"{label.capitalize()} must be a JSON object")
    normalized = _require_json_value(value, label)
    if not isinstance(normalized, dict):
        raise EventDecodingError(f"{label.capitalize()} must be a JSON object")
    return normalized


def _require_json_value(value: object, label: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_require_json_value(item, label) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EventDecodingError(f"{label.capitalize()} contains a non-string key")
            result[key] = _require_json_value(item, label)
        return result
    raise EventDecodingError(f"{label.capitalize()} contains a non-JSON value")


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EventDecodingError(f"{label.capitalize()} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EventDecodingError(f"{label.capitalize()} must be an integer")
    return value
