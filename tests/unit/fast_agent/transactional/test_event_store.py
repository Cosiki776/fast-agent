from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict

import pytest

from fast_agent.transactional.events import (
    ToolAuthorized,
    ToolCheckpointed,
    ToolCommitted,
    ToolDenied,
    ToolEvent,
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
    RunId,
    ToolCallId,
    ToolEffect,
    TransactionId,
    TransactionState,
)
from fast_agent.transactional.state_machine import (
    InvalidTransitionError,
    TransactionIdentityError,
)
from fast_agent.transactional.storage.event_store import (
    SCHEMA_VERSION,
    EventStoreError,
    SQLiteEventStore,
    UnsupportedSchemaVersionError,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class EventIdentity:
    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId


class EventIdentityKwargs(TypedDict):
    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId


RUN_ID = RunId("run-1")


def identity(name: str) -> EventIdentity:
    return EventIdentity(
        run_id=RUN_ID,
        transaction_id=TransactionId(f"transaction-{name}"),
        tool_call_id=ToolCallId(f"tool-call-{name}"),
    )


def identity_kwargs(value: EventIdentity) -> EventIdentityKwargs:
    return {
        "run_id": value.run_id,
        "transaction_id": value.transaction_id,
        "tool_call_id": value.tool_call_id,
    }


def proposed(value: EventIdentity) -> ToolProposed:
    return ToolProposed(
        **identity_kwargs(value),
        tool_name="read_text_file",
        arguments={"path": "README.md", "line_end": 20},
        effect=ToolEffect.READ,
    )


def full_success_path(value: EventIdentity) -> list[ToolEvent]:
    fields = identity_kwargs(value)
    return [
        proposed(value),
        ToolValidated(**fields),
        ToolAuthorized(**fields),
        ToolCheckpointed(**fields, checkpoint_id=f"checkpoint-{value.transaction_id}"),
        ToolExecutionStarted(**fields),
        ToolResultStored(**fields, artifact_id="artifact-success", is_error=False),
        ToolCommitted(**fields),
    ]


def validation_failure_path(value: EventIdentity) -> list[ToolEvent]:
    fields = identity_kwargs(value)
    return [
        proposed(value),
        ToolValidationFailed(**fields, reason="arguments do not match the schema"),
        ToolFailed(**fields, reason="validation failed"),
    ]


def denial_path(value: EventIdentity) -> list[ToolEvent]:
    fields = identity_kwargs(value)
    return [
        proposed(value),
        ToolValidated(**fields),
        ToolDenied(**fields, reason="path is outside the workspace"),
        ToolFailed(**fields, reason="policy denied"),
    ]


def rollback_failure_path(value: EventIdentity) -> list[ToolEvent]:
    fields = identity_kwargs(value)
    return [
        proposed(value),
        ToolExecutionStarted(**fields),
        ToolExecutionFailed(
            **fields,
            error_type="timeout",
            message="tool execution timed out",
        ),
        ToolRollbackStarted(**fields, reason="restore the last checkpoint"),
        ToolRolledBack(**fields, checkpoint_id="checkpoint-before-execution"),
        ToolFailed(**fields, reason="execution failed and workspace was restored"),
    ]


def test_append_query_and_reopen_preserve_fact_sequence(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "events.sqlite3"
    successful = identity("success")
    failed = identity("validation-failed")
    success_events = full_success_path(successful)
    failure_events = validation_failure_path(failed)

    with SQLiteEventStore(database_path) as store:
        stored = [
            *(store.append(event) for event in success_events),
            *(store.append(event) for event in failure_events),
        ]

        assert store.schema_version == SCHEMA_VERSION
        assert [item.sequence for item in stored] == list(range(1, len(stored) + 1))
        assert [item.event for item in store.events_for_run(RUN_ID)] == [
            *success_events,
            *failure_events,
        ]

    with SQLiteEventStore(database_path) as reopened:
        assert [item.event for item in reopened.events_for_transaction(
            successful.transaction_id
        )] == success_events
        assert [item.event for item in reopened.events_for_transaction(
            failed.transaction_id
        )] == failure_events


def test_all_event_payload_shapes_round_trip(tmp_path: Path) -> None:
    paths = [
        full_success_path(identity("success")),
        validation_failure_path(identity("validation-failed")),
        denial_path(identity("denied")),
        rollback_failure_path(identity("rolled-back")),
    ]

    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        for events in paths:
            for event in events:
                store.append(event)

        for events in paths:
            transaction_id = events[0].transaction_id
            assert [
                item.event for item in store.events_for_transaction(transaction_id)
            ] == events


def test_replay_uses_persisted_event_order(tmp_path: Path) -> None:
    successful = identity("success")
    failed = identity("failed")

    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        for event in full_success_path(successful):
            store.append(event)
        for event in rollback_failure_path(failed):
            store.append(event)

        success_projection = store.replay(successful.transaction_id)
        failure_projection = store.replay(failed.transaction_id)

    assert success_projection.state is TransactionState.COMMITTED
    assert success_projection.event_count == 7
    assert failure_projection.state is TransactionState.FAILED
    assert failure_projection.event_count == 6


def test_invalid_transition_is_not_persisted(tmp_path: Path) -> None:
    event_identity = identity("invalid")
    fields = identity_kwargs(event_identity)

    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        store.append(proposed(event_identity))

        with pytest.raises(InvalidTransitionError):
            store.append(ToolCommitted(**fields))

        stored = store.events_for_transaction(event_identity.transaction_id)

    assert [item.event.kind for item in stored] == [ToolProposed.kind]


def test_mixed_transaction_identity_is_not_persisted(tmp_path: Path) -> None:
    original = identity("shared")
    mismatched = EventIdentity(
        run_id=RunId("other-run"),
        transaction_id=original.transaction_id,
        tool_call_id=original.tool_call_id,
    )

    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        store.append(proposed(original))

        with pytest.raises(TransactionIdentityError, match="does not match"):
            store.append(
                ToolExecutionStarted(**identity_kwargs(mismatched)),
            )

        stored = store.events_for_transaction(original.transaction_id)

    assert [item.event.kind for item in stored] == [ToolProposed.kind]


def test_terminal_transaction_rejects_duplicate_completion(tmp_path: Path) -> None:
    event_identity = identity("complete")
    events = full_success_path(event_identity)

    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        for event in events:
            store.append(event)

        with pytest.raises(InvalidTransitionError):
            store.append(
                ToolFailed(
                    **identity_kwargs(event_identity),
                    reason="late duplicate completion",
                )
            )

        stored = store.events_for_transaction(event_identity.transaction_id)

    assert [item.event for item in stored] == events


def test_append_rejects_naive_timestamp_without_persisting(tmp_path: Path) -> None:
    event_identity = identity("naive-timestamp")
    event = proposed(event_identity)
    event = replace(event, occurred_at=event.occurred_at.replace(tzinfo=None))

    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        with pytest.raises(EventStoreError, match="timestamp must include a UTC offset"):
            store.append(event)

        assert store.events_for_transaction(event_identity.transaction_id) == []


def test_open_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA user_version = 99")
    connection.close()

    with pytest.raises(
        UnsupportedSchemaVersionError,
        match="schema version 99 is not supported",
    ):
        SQLiteEventStore(database_path)
