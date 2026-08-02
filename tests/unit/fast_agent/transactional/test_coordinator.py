from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.coordinator import (
    TransactionCoordinator,
    classify_tool_effect,
    serialize_tool_result,
)
from fast_agent.transactional.events import (
    ToolEventKind,
    ToolProposed,
    ToolResultStored,
)
from fast_agent.transactional.execution import (
    ToolExecutionOutcome,
    ToolExecutionRequest,
)
from fast_agent.transactional.models import (
    RunId,
    ToolCallId,
    ToolEffect,
    TransactionId,
    TransactionState,
)
from fast_agent.transactional.storage.artifact_store import ArtifactId, FileArtifactStore
from fast_agent.transactional.storage.event_store import SQLiteEventStore

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.transactional.coordinator import ToolDenialResolver
    from fast_agent.transactional.events import ToolEvent


RUN_ID = RunId("run-1")
TOOL_CALL_ID = ToolCallId("call-1")
TRANSACTION_ID = TransactionId("transaction-1")


def _request(tool_name: str = "write_text_file") -> ToolExecutionRequest:
    return ToolExecutionRequest(
        run_id=RUN_ID,
        tool_call_id=TOOL_CALL_ID,
        tool_name=tool_name,
        arguments={"path": "notes.txt", "content": "hello"},
    )


def _result(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={"text": text},
        isError=is_error,
    )


def _coordinator(
    tmp_path: Path,
    *,
    denial_reason: ToolDenialResolver | None = None,
) -> tuple[TransactionCoordinator, SQLiteEventStore, FileArtifactStore]:
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    coordinator = TransactionCoordinator(
        event_store,
        artifact_store,
        denial_reason=denial_reason,
        transaction_id_factory=lambda: TRANSACTION_ID,
    )
    return coordinator, event_store, artifact_store


def _events(event_store: SQLiteEventStore) -> list[ToolEvent]:
    return [item.event for item in event_store.events_for_transaction(TRANSACTION_ID)]


@pytest.mark.asyncio
async def test_success_persists_intent_result_and_commit_in_order(tmp_path: Path) -> None:
    coordinator, event_store, artifact_store = _coordinator(tmp_path)
    expected_result = _result("written")
    executions = 0

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        # Intent and the execution boundary must be durable before the side effect starts.
        assert [event.kind for event in _events(event_store)] == [
            ToolEventKind.PROPOSED,
            ToolEventKind.EXECUTION_STARTED,
        ]
        return ToolExecutionOutcome(result=expected_result)

    outcome = await coordinator.coordinate(_request(), call_next)
    events = _events(event_store)

    assert executions == 1
    assert outcome.result is expected_result
    assert [event.kind for event in events] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.EXECUTION_STARTED,
        ToolEventKind.RESULT_STORED,
        ToolEventKind.COMMITTED,
    ]
    proposed = events[0]
    assert isinstance(proposed, ToolProposed)
    assert proposed.effect is ToolEffect.WORKSPACE_WRITE
    stored = events[2]
    assert isinstance(stored, ToolResultStored)
    assert artifact_store.read(ArtifactId(stored.artifact_id)) == serialize_tool_result(
        expected_result
    )
    assert event_store.replay(TRANSACTION_ID).state is TransactionState.COMMITTED
    event_store.close()


@pytest.mark.asyncio
async def test_error_result_is_stored_before_transaction_fails(tmp_path: Path) -> None:
    coordinator, event_store, artifact_store = _coordinator(tmp_path)
    expected_result = _result("missing file", is_error=True)

    async def call_next() -> ToolExecutionOutcome:
        return ToolExecutionOutcome(result=expected_result)

    outcome = await coordinator.coordinate(_request("read_text_file"), call_next)
    events = _events(event_store)

    assert outcome.result is expected_result
    assert [event.kind for event in events] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.EXECUTION_STARTED,
        ToolEventKind.RESULT_STORED,
        ToolEventKind.FAILED,
    ]
    stored = events[2]
    assert isinstance(stored, ToolResultStored)
    assert stored.is_error is True
    assert artifact_store.read(ArtifactId(stored.artifact_id)) == serialize_tool_result(
        expected_result
    )
    assert event_store.replay(TRANSACTION_ID).state is TransactionState.FAILED
    event_store.close()


@pytest.mark.asyncio
async def test_denial_returns_synthetic_result_without_execution(tmp_path: Path) -> None:
    coordinator, event_store, _ = _coordinator(
        tmp_path,
        denial_reason=lambda request: "write access disabled",
    )
    executed = False

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executed
        executed = True
        return ToolExecutionOutcome(result=_result("unexpected"))

    outcome = await coordinator.coordinate(_request(), call_next)

    assert executed is False
    assert outcome.result.isError is True
    assert outcome.result.structuredContent == {
        "status": "denied",
        "reason": "write access disabled",
    }
    assert [event.kind for event in _events(event_store)] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.DENIED,
        ToolEventKind.FAILED,
    ]
    event_store.close()


@pytest.mark.asyncio
async def test_exception_becomes_explicit_failed_tool_result(tmp_path: Path) -> None:
    coordinator, event_store, _ = _coordinator(tmp_path)

    async def call_next() -> ToolExecutionOutcome:
        raise OSError("disk unavailable")

    outcome = await coordinator.coordinate(_request(), call_next)

    assert outcome.result.isError is True
    assert outcome.result.structuredContent == {
        "status": "failed",
        "error_type": "OSError",
        "message": "disk unavailable",
    }
    assert [event.kind for event in _events(event_store)] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.EXECUTION_STARTED,
        ToolEventKind.EXECUTION_FAILED,
        ToolEventKind.FAILED,
    ]
    event_store.close()


def test_effect_classifier_uses_explicit_first_phase_rules() -> None:
    assert classify_tool_effect("read_text_file") is ToolEffect.READ
    assert classify_tool_effect("write_text_file") is ToolEffect.WORKSPACE_WRITE
    assert classify_tool_effect("apply_patch") is ToolEffect.WORKSPACE_WRITE
    assert classify_tool_effect("execute") is ToolEffect.WORKSPACE_WRITE
    assert classify_tool_effect("remote_tool") is ToolEffect.EXTERNAL_UNKNOWN


def test_raw_result_serialization_preserves_standard_mcp_fields() -> None:
    payload = json.loads(serialize_tool_result(_result("完成")))

    assert payload == {
        "content": [{"type": "text", "text": "完成"}],
        "structuredContent": {"text": "完成"},
        "isError": False,
    }
