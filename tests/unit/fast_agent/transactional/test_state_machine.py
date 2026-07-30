from __future__ import annotations

from dataclasses import dataclass

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
    EmptyTransactionError,
    InvalidTransitionError,
    TransactionIdentityError,
    apply_event,
    replay_transaction,
)


@dataclass(frozen=True, slots=True)
class EventIdentity:
    run_id: RunId = RunId("run-1")
    transaction_id: TransactionId = TransactionId("transaction-1")
    tool_call_id: ToolCallId = ToolCallId("tool-call-1")


IDENTITY = EventIdentity()


def proposed(identity: EventIdentity = IDENTITY) -> ToolProposed:
    return ToolProposed(
        run_id=identity.run_id,
        transaction_id=identity.transaction_id,
        tool_call_id=identity.tool_call_id,
        tool_name="read_text_file",
        arguments={"path": "README.md"},
        effect=ToolEffect.READ,
    )


def test_replay_minimal_vertical_slice_commits() -> None:
    events: list[ToolEvent] = [
        proposed(),
        ToolExecutionStarted(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
        ToolResultStored(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
            artifact_id="artifact-1",
            is_error=False,
        ),
        ToolCommitted(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
    ]

    projection = replay_transaction(events)

    assert projection.run_id == IDENTITY.run_id
    assert projection.transaction_id == IDENTITY.transaction_id
    assert projection.tool_call_id == IDENTITY.tool_call_id
    assert projection.state is TransactionState.COMMITTED
    assert projection.event_count == 4
    assert projection.last_event_kind is ToolCommitted.kind
    assert projection.last_occurred_at == events[-1].occurred_at


def test_replay_full_governed_path_commits() -> None:
    events: list[ToolEvent] = [
        proposed(),
        ToolValidated(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
        ToolAuthorized(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
        ToolCheckpointed(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
            checkpoint_id="checkpoint-1",
        ),
        ToolExecutionStarted(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
        ToolResultStored(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
            artifact_id="artifact-1",
            is_error=False,
        ),
        ToolCommitted(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
    ]

    projection = replay_transaction(events)

    assert projection.state is TransactionState.COMMITTED
    assert projection.event_count == 7


@pytest.mark.parametrize(
    "events",
    [
        [
            proposed(),
            ToolValidationFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="arguments do not match the tool schema",
            ),
            ToolFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="validation failed",
            ),
        ],
        [
            proposed(),
            ToolValidated(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
            ),
            ToolDenied(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="path is outside the workspace",
            ),
            ToolFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="policy denied",
            ),
        ],
        [
            proposed(),
            ToolExecutionStarted(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
            ),
            ToolExecutionFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                error_type="timeout",
                message="tool execution timed out",
            ),
            ToolRollbackStarted(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="restore the last checkpoint",
            ),
            ToolRolledBack(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                checkpoint_id="checkpoint-1",
            ),
            ToolFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="execution failed and workspace was restored",
            ),
        ],
    ],
    ids=["validation-failed", "denied", "rolled-back"],
)
def test_replay_failure_paths_finish_failed(events: list[ToolEvent]) -> None:
    projection = replay_transaction(events)

    assert projection.state is TransactionState.FAILED
    assert projection.event_count == len(events)


def test_first_event_must_be_proposed() -> None:
    execution_started = ToolExecutionStarted(
        run_id=IDENTITY.run_id,
        transaction_id=IDENTITY.transaction_id,
        tool_call_id=IDENTITY.tool_call_id,
    )

    with pytest.raises(
        InvalidTransitionError,
        match="cannot transition transaction from '<none>' to 'executing'",
    ):
        replay_transaction([execution_started])


def test_commit_requires_stored_result() -> None:
    committed = ToolCommitted(
        run_id=IDENTITY.run_id,
        transaction_id=IDENTITY.transaction_id,
        tool_call_id=IDENTITY.tool_call_id,
    )

    with pytest.raises(
        InvalidTransitionError,
        match="cannot transition transaction from 'proposed' to 'committed'",
    ):
        replay_transaction([proposed(), committed])


@pytest.mark.parametrize(
    "terminal_event",
    [
        ToolCommitted(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
        ),
        ToolFailed(
            run_id=IDENTITY.run_id,
            transaction_id=IDENTITY.transaction_id,
            tool_call_id=IDENTITY.tool_call_id,
            reason="execution failed",
        ),
    ],
    ids=["committed", "failed"],
)
def test_terminal_state_rejects_more_events(terminal_event: ToolEvent) -> None:
    if isinstance(terminal_event, ToolCommitted):
        events: list[ToolEvent] = [
            proposed(),
            ToolExecutionStarted(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
            ),
            ToolResultStored(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                artifact_id="artifact-1",
                is_error=False,
            ),
            terminal_event,
        ]
    else:
        events = [
            proposed(),
            ToolValidationFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="invalid arguments",
            ),
            terminal_event,
        ]

    projection = replay_transaction(events)

    with pytest.raises(InvalidTransitionError):
        apply_event(
            projection,
            ToolFailed(
                run_id=IDENTITY.run_id,
                transaction_id=IDENTITY.transaction_id,
                tool_call_id=IDENTITY.tool_call_id,
                reason="late duplicate failure",
            ),
        )


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        (
            EventIdentity(run_id=RunId("other-run")),
            "Event run ID 'other-run' does not match",
        ),
        (
            EventIdentity(transaction_id=TransactionId("other-transaction")),
            "Event transaction ID 'other-transaction' does not match",
        ),
        (
            EventIdentity(tool_call_id=ToolCallId("other-tool-call")),
            "Event tool call ID 'other-tool-call' does not match",
        ),
    ],
    ids=["run-id", "transaction-id", "tool-call-id"],
)
def test_replay_rejects_mixed_transaction_identity(
    identity: EventIdentity,
    message: str,
) -> None:
    execution_started = ToolExecutionStarted(
        run_id=identity.run_id,
        transaction_id=identity.transaction_id,
        tool_call_id=identity.tool_call_id,
    )

    with pytest.raises(TransactionIdentityError, match=message):
        replay_transaction([proposed(), execution_started])


def test_replay_rejects_empty_event_stream() -> None:
    with pytest.raises(
        EmptyTransactionError,
        match="Cannot replay a transaction without events",
    ):
        replay_transaction([])
