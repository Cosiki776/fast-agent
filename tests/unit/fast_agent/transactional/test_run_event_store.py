from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, TypedDict

from fast_agent.transactional.events import (
    ToolCommitted,
    ToolExecutionStarted,
    ToolProposed,
    ToolResultStored,
)
from fast_agent.transactional.models import RunId, ToolCallId, ToolEffect, TransactionId
from fast_agent.transactional.run_events import (
    PromotionApplied,
    RunEventKind,
    RunRecovered,
    RunRecoveryStarted,
    RunStarted,
    RunState,
    RunVerificationFailed,
    RunVerificationStarted,
    RunVerified,
)
from fast_agent.transactional.storage.event_store import SCHEMA_VERSION, SQLiteEventStore
from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore

if TYPE_CHECKING:
    from pathlib import Path

RUN_ID = RunId("run-1")


class _ToolIdentity(TypedDict):
    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId


def test_run_events_round_trip_and_project_recovery_state(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    events = [
        RunStarted(run_id=RUN_ID, profile="full"),
        RunRecoveryStarted(
            run_id=RUN_ID,
            failure_signature="sha256:failure",
            checkpoint_id="checkpoint-1",
        ),
        RunRecovered(run_id=RUN_ID, workspace_version="version-1"),
        RunVerificationStarted(run_id=RUN_ID, command="pytest"),
        RunVerificationFailed(
            run_id=RUN_ID,
            exit_code=1,
            timed_out=False,
            stdout_artifact_id="stdout-1",
            stderr_artifact_id="stderr-1",
        ),
        RunVerificationStarted(run_id=RUN_ID, command="pytest"),
        RunVerified(
            run_id=RUN_ID,
            workspace_version="version-2",
            stdout_artifact_id="stdout-2",
            stderr_artifact_id="stderr-2",
        ),
        PromotionApplied(
            run_id=RUN_ID,
            workspace_version="version-2",
            patch_artifact_id="patch-1",
        ),
    ]

    with SQLiteRunEventStore(path) as store:
        for event in events:
            store.append(event)

    with SQLiteRunEventStore(path) as reopened:
        stored = reopened.events_for_run(RUN_ID)
        projection = reopened.replay(RUN_ID)

    assert [item.sequence for item in stored] == list(range(1, 9))
    assert [item.event for item in stored] == events
    assert [item.event.kind for item in stored] == [
        RunEventKind.STARTED,
        RunEventKind.RECOVERY_STARTED,
        RunEventKind.RECOVERED,
        RunEventKind.VERIFICATION_STARTED,
        RunEventKind.VERIFICATION_FAILED,
        RunEventKind.VERIFICATION_STARTED,
        RunEventKind.VERIFIED,
        RunEventKind.PROMOTION_APPLIED,
    ]
    assert projection.state is RunState.PROMOTED


def test_schema_v1_migrates_without_losing_tool_events(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    transaction_id = TransactionId("transaction-1")
    identity: _ToolIdentity = {
        "run_id": RUN_ID,
        "transaction_id": transaction_id,
        "tool_call_id": ToolCallId("call-1"),
    }
    with SQLiteEventStore(path) as store:
        store.append(
            ToolProposed(
                **identity,
                tool_name="read_text_file",
                arguments={"path": "README.md"},
                effect=ToolEffect.READ,
            )
        )
        store.append(ToolExecutionStarted(**identity))
        store.append(ToolResultStored(**identity, artifact_id="artifact-1", is_error=False))
        store.append(ToolCommitted(**identity))

    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE run_events")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    with SQLiteEventStore(path) as migrated:
        assert migrated.schema_version == SCHEMA_VERSION == 2
        assert [item.event.kind for item in migrated.events_for_transaction(transaction_id)] == [
            ToolProposed.kind,
            ToolExecutionStarted.kind,
            ToolResultStored.kind,
            ToolCommitted.kind,
        ]
    with SQLiteRunEventStore(path) as run_events:
        run_events.append(RunStarted(run_id=RUN_ID, profile="full"))
        assert run_events.replay(RUN_ID).state is RunState.ACTIVE
