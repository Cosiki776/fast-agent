from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.budget import RunBudgetLimits, RunBudgetTracker
from fast_agent.transactional.context.reducers import (
    CodingToolResultReducer,
    ReducerLimits,
    ToolResultReducer,
)
from fast_agent.transactional.coordinator import (
    TransactionCoordinator,
    classify_tool_effect,
    serialize_tool_result,
)
from fast_agent.transactional.events import (
    ToolCheckpointed,
    ToolCheckpointFailed,
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
from fast_agent.transactional.recovery.controller import RecoveryController
from fast_agent.transactional.run_events import RunEventKind, RunStarted
from fast_agent.transactional.storage.artifact_store import ArtifactId, FileArtifactStore
from fast_agent.transactional.storage.event_store import SQLiteEventStore
from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fast_agent.transactional.coordinator import ToolDenialResolver
    from fast_agent.transactional.events import ToolEvent
    from fast_agent.transactional.recovery.classifier import ToolEffectClassifier


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
    result_reducer: ToolResultReducer | None = None,
    run_budget: RunBudgetTracker | None = None,
    checkpoint_creator: Callable[[ToolExecutionRequest], str] | None = None,
    effect_classifier: ToolEffectClassifier | None = None,
    recovery_controller: RecoveryController | None = None,
    checkpoint_restorer: Callable[[str], str] | None = None,
    run_event_store: SQLiteRunEventStore | None = None,
    transaction_id_factory: Callable[[], TransactionId] = lambda: TRANSACTION_ID,
) -> tuple[TransactionCoordinator, SQLiteEventStore, FileArtifactStore]:
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    coordinator = TransactionCoordinator(
        event_store,
        artifact_store,
        denial_reason=denial_reason,
        checkpoint_creator=checkpoint_creator,
        checkpoint_restorer=checkpoint_restorer,
        result_reducer=result_reducer,
        run_budget=run_budget,
        effect_classifier=effect_classifier,
        recovery_controller=recovery_controller,
        run_event_store=run_event_store,
        transaction_id_factory=transaction_id_factory,
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
    assert classify_tool_effect("process") is ToolEffect.WORKSPACE_WRITE
    assert classify_tool_effect("remote_tool") is ToolEffect.EXTERNAL_UNKNOWN


@pytest.mark.asyncio
async def test_write_executes_only_after_checkpoint_event_is_persisted(tmp_path: Path) -> None:
    coordinator, event_store, _ = _coordinator(
        tmp_path,
        checkpoint_creator=lambda request: "checkpoint-1",
    )
    executions = 0

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        assert [event.kind for event in _events(event_store)] == [
            ToolEventKind.PROPOSED,
            ToolEventKind.VALIDATED,
            ToolEventKind.AUTHORIZED,
            ToolEventKind.CHECKPOINTED,
            ToolEventKind.EXECUTION_STARTED,
        ]
        return ToolExecutionOutcome(result=_result("written"))

    await coordinator.coordinate(_request(), call_next)

    assert executions == 1
    checkpointed = _events(event_store)[3]
    assert isinstance(checkpointed, ToolCheckpointed)
    assert checkpointed.checkpoint_id == "checkpoint-1"
    event_store.close()


@pytest.mark.asyncio
async def test_checkpoint_creation_failure_is_fail_closed(tmp_path: Path) -> None:
    def fail_checkpoint(request: ToolExecutionRequest) -> str:
        raise OSError("snapshot disk unavailable")

    coordinator, event_store, _ = _coordinator(
        tmp_path,
        checkpoint_creator=fail_checkpoint,
    )
    executions = 0

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        return ToolExecutionOutcome(result=_result("unexpected"))

    outcome = await coordinator.coordinate(_request(), call_next)

    assert executions == 0
    assert outcome.result.structuredContent == {
        "status": "checkpoint_failed",
        "error_type": "OSError",
        "message": "snapshot disk unavailable",
    }
    assert [event.kind for event in _events(event_store)] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.VALIDATED,
        ToolEventKind.AUTHORIZED,
        ToolEventKind.CHECKPOINT_FAILED,
        ToolEventKind.FAILED,
    ]
    failed = _events(event_store)[3]
    assert isinstance(failed, ToolCheckpointFailed)
    event_store.close()


@pytest.mark.asyncio
async def test_checkpoint_event_persistence_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, event_store, _ = _coordinator(
        tmp_path,
        checkpoint_creator=lambda request: "checkpoint-1",
    )
    real_append = event_store.append

    def fail_checkpoint_event(event: ToolEvent) -> object:
        if isinstance(event, ToolCheckpointed):
            raise OSError("event store unavailable")
        return real_append(event)

    monkeypatch.setattr(event_store, "append", fail_checkpoint_event)
    executions = 0

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        return ToolExecutionOutcome(result=_result("unexpected"))

    outcome = await coordinator.coordinate(_request(), call_next)

    assert executions == 0
    assert outcome.result.structuredContent == {
        "status": "checkpoint_failed",
        "error_type": "OSError",
        "message": "event store unavailable",
    }
    assert [event.kind for event in _events(event_store)] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.VALIDATED,
        ToolEventKind.AUTHORIZED,
        ToolEventKind.CHECKPOINT_FAILED,
        ToolEventKind.FAILED,
    ]
    event_store.close()


@pytest.mark.asyncio
async def test_unknown_effect_is_denied_without_execution(tmp_path: Path) -> None:
    coordinator, event_store, _ = _coordinator(tmp_path)
    executions = 0

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        return ToolExecutionOutcome(result=_result("unexpected"))

    outcome = await coordinator.coordinate(_request("unknown_local_tool"), call_next)

    assert executions == 0
    assert outcome.result.structuredContent == {
        "status": "denied",
        "reason": "tool effect is unknown: unknown_local_tool",
    }
    event_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raises", "invalid"])
async def test_effect_classifier_failure_is_denied_fail_closed(
    tmp_path: Path,
    mode: str,
) -> None:
    def raises(request: ToolExecutionRequest) -> ToolEffect:
        del request
        raise RuntimeError("classifier unavailable")

    def invalid(request: ToolExecutionRequest) -> str:
        del request
        return "read"

    classifier = raises if mode == "raises" else cast("ToolEffectClassifier", invalid)
    coordinator, event_store, _ = _coordinator(tmp_path, effect_classifier=classifier)
    executions = 0

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        return ToolExecutionOutcome(result=_result("unexpected"))

    outcome = await coordinator.coordinate(_request("remote_tool"), call_next)

    assert executions == 0
    assert outcome.result.structuredContent == {
        "status": "denied",
        "reason": "tool effect is unknown: remote_tool",
    }
    proposed = _events(event_store)[0]
    assert isinstance(proposed, ToolProposed)
    assert proposed.effect is ToolEffect.EXTERNAL_UNKNOWN
    event_store.close()


@pytest.mark.asyncio
async def test_repeated_write_failure_rolls_back_and_returns_handoff(tmp_path: Path) -> None:
    transaction_ids = iter((TransactionId("transaction-1"), TransactionId("transaction-2")))
    restored: list[str] = []
    run_events = SQLiteRunEventStore(tmp_path / "events.sqlite3")
    run_events.append(RunStarted(run_id=RUN_ID, profile="full"))
    coordinator, event_store, _ = _coordinator(
        tmp_path,
        checkpoint_creator=lambda request: f"checkpoint-{request.tool_call_id}",
        checkpoint_restorer=lambda checkpoint_id: restored.append(checkpoint_id) or "version-1",
        recovery_controller=RecoveryController(),
        run_event_store=run_events,
        transaction_id_factory=lambda: next(transaction_ids),
    )

    async def call_next() -> ToolExecutionOutcome:
        return ToolExecutionOutcome(result=_result("same assertion failed", is_error=True))

    await coordinator.coordinate(_request(), call_next)
    second_request = ToolExecutionRequest(
        run_id=RUN_ID,
        tool_call_id=ToolCallId("call-2"),
        tool_name="write_text_file",
        arguments={"path": "notes.txt", "content": "second attempt"},
    )
    outcome = await coordinator.coordinate(second_request, call_next)

    assert restored == ["checkpoint-call-1"]
    assert outcome.result.structuredContent is not None
    assert outcome.result.structuredContent["status"] == "recovery_handoff"
    assert [
        item.event.kind
        for item in event_store.events_for_transaction(TransactionId("transaction-2"))
    ] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.VALIDATED,
        ToolEventKind.AUTHORIZED,
        ToolEventKind.CHECKPOINTED,
        ToolEventKind.EXECUTION_STARTED,
        ToolEventKind.RESULT_STORED,
        ToolEventKind.ROLLBACK_STARTED,
        ToolEventKind.ROLLED_BACK,
        ToolEventKind.FAILED,
    ]
    assert [item.event.kind for item in run_events.events_for_run(RUN_ID)] == [
        RunEventKind.STARTED,
        RunEventKind.RECOVERY_STARTED,
        RunEventKind.RECOVERED,
    ]
    run_events.close()
    event_store.close()


@pytest.mark.asyncio
async def test_restore_mismatch_stops_with_workspace_divergence(tmp_path: Path) -> None:
    transaction_ids = iter((TransactionId("transaction-1"), TransactionId("transaction-2")))

    def fail_restore(checkpoint_id: str) -> str:
        del checkpoint_id
        raise RuntimeError("hash mismatch")

    coordinator, event_store, _ = _coordinator(
        tmp_path,
        checkpoint_creator=lambda request: f"checkpoint-{request.tool_call_id}",
        checkpoint_restorer=fail_restore,
        recovery_controller=RecoveryController(),
        transaction_id_factory=lambda: next(transaction_ids),
    )

    async def call_next() -> ToolExecutionOutcome:
        return ToolExecutionOutcome(result=_result("same assertion failed", is_error=True))

    await coordinator.coordinate(_request(), call_next)
    outcome = await coordinator.coordinate(
        ToolExecutionRequest(
            run_id=RUN_ID,
            tool_call_id=ToolCallId("call-2"),
            tool_name="write_text_file",
            arguments={},
        ),
        call_next,
    )

    assert outcome.result.structuredContent == {
        "status": "workspace_divergence",
        "error_type": "RuntimeError",
        "message": "hash mismatch",
    }
    assert [
        item.event.kind
        for item in event_store.events_for_transaction(TransactionId("transaction-2"))
    ][-2:] == [ToolEventKind.ROLLBACK_STARTED, ToolEventKind.FAILED]
    event_store.close()


def test_raw_result_serialization_preserves_standard_mcp_fields() -> None:
    payload = json.loads(serialize_tool_result(_result("完成")))

    assert payload == {
        "content": [{"type": "text", "text": "完成"}],
        "structuredContent": {"text": "完成"},
        "isError": False,
    }


@pytest.mark.asyncio
async def test_reducer_returns_bounded_result_after_raw_artifact_is_stored(tmp_path: Path) -> None:
    reducer = CodingToolResultReducer(ReducerLimits(max_result_bytes=512, max_items=4))
    coordinator, event_store, artifact_store = _coordinator(tmp_path, result_reducer=reducer)
    raw_text = "\n".join(f"log line {index}" for index in range(2_000))
    expected_result = _result(raw_text)

    async def call_next() -> ToolExecutionOutcome:
        return ToolExecutionOutcome(result=expected_result)

    outcome = await coordinator.coordinate(
        ToolExecutionRequest(
            run_id=RUN_ID,
            tool_call_id=TOOL_CALL_ID,
            tool_name="execute",
            arguments={"command": "python noisy_job.py"},
        ),
        call_next,
    )
    events = _events(event_store)
    stored = events[2]
    assert isinstance(stored, ToolResultStored)

    assert artifact_store.read(ArtifactId(stored.artifact_id)) == serialize_tool_result(
        expected_result
    )
    content = outcome.result.content[0]
    assert isinstance(content, TextContent)
    assert len(content.text.encode("utf-8")) <= 512
    assert raw_text not in content.text
    assert f"full_output_artifact: {stored.artifact_id}" in content.text
    event_store.close()


@pytest.mark.asyncio
async def test_reducer_failure_returns_bounded_fallback_with_artifact_reference(
    tmp_path: Path,
) -> None:
    def failing_reducer(
        request: ToolExecutionRequest,
        result: CallToolResult,
        artifact_id: ArtifactId,
        /,
    ) -> CallToolResult:
        raise RuntimeError("reducer bug")

    coordinator, event_store, _ = _coordinator(tmp_path, result_reducer=failing_reducer)

    async def call_next() -> ToolExecutionOutcome:
        return ToolExecutionOutcome(result=_result("x" * 20_000))

    outcome = await coordinator.coordinate(_request("execute"), call_next)
    stored = _events(event_store)[2]
    assert isinstance(stored, ToolResultStored)
    content = outcome.result.content[0]
    assert isinstance(content, TextContent)
    assert len(content.text.encode("utf-8")) <= 8 * 1024
    assert f"full_output_artifact: {stored.artifact_id}" in content.text
    event_store.close()


@pytest.mark.asyncio
async def test_exhausted_tool_budget_returns_explicit_result_without_execution(
    tmp_path: Path,
) -> None:
    budget = RunBudgetTracker(RunBudgetLimits(max_tool_calls=0))
    coordinator, event_store, _ = _coordinator(tmp_path, run_budget=budget)
    executed = False

    async def call_next() -> ToolExecutionOutcome:
        nonlocal executed
        executed = True
        return ToolExecutionOutcome(result=_result("unexpected"))

    outcome = await coordinator.coordinate(_request(), call_next)

    assert executed is False
    assert outcome.result.isError is True
    assert outcome.result.structuredContent == {
        "status": "budget_exhausted",
        "dimensions": ["tool_calls"],
    }
    assert [event.kind for event in _events(event_store)] == [
        ToolEventKind.PROPOSED,
        ToolEventKind.DENIED,
        ToolEventKind.FAILED,
    ]
    event_store.close()


@pytest.mark.asyncio
async def test_coordinator_records_serialized_artifact_bytes_in_run_budget(tmp_path: Path) -> None:
    budget = RunBudgetTracker(RunBudgetLimits(max_artifact_output_bytes=10_000))
    coordinator, event_store, _ = _coordinator(tmp_path, run_budget=budget)
    expected_result = _result("stored output")

    async def call_next() -> ToolExecutionOutcome:
        return ToolExecutionOutcome(result=expected_result)

    await coordinator.coordinate(_request(), call_next)

    assert budget.snapshot.tool_calls == 1
    assert budget.snapshot.artifact_output_bytes == len(serialize_tool_result(expected_result))
    event_store.close()
