from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from fast_agent.transactional.events import (
    ToolCommitted,
    ToolEvent,
    ToolEventKind,
    ToolExecutionStarted,
    ToolFailed,
    ToolProposed,
    ToolValidated,
)
from fast_agent.transactional.models import (
    RunId,
    ToolCallId,
    ToolEffect,
    TransactionId,
    new_run_id,
    new_transaction_id,
)

RUN_ID = RunId("run-1")
TRANSACTION_ID = TransactionId("transaction-1")
TOOL_CALL_ID = ToolCallId("tool-call-1")


def proposed_event() -> ToolProposed:
    return ToolProposed(
        run_id=RUN_ID,
        transaction_id=TRANSACTION_ID,
        tool_call_id=TOOL_CALL_ID,
        tool_name="read_text_file",
        arguments={"path": "README.md"},
        effect=ToolEffect.READ,
    )


def test_generated_domain_ids_are_unique_string_values() -> None:
    first_run_id = new_run_id()
    second_run_id = new_run_id()
    first_transaction_id = new_transaction_id()
    second_transaction_id = new_transaction_id()

    assert isinstance(first_run_id, str)
    assert isinstance(first_transaction_id, str)
    assert first_run_id
    assert first_transaction_id
    assert first_run_id != second_run_id
    assert first_transaction_id != second_transaction_id


def test_tool_effect_values_are_stable_for_persistence() -> None:
    assert ToolEffect.READ.value == "read"
    assert ToolEffect.WORKSPACE_WRITE.value == "workspace_write"
    assert ToolEffect.EXTERNAL_UNKNOWN.value == "external_unknown"


def test_proposed_event_preserves_identity_and_payload() -> None:
    event = proposed_event()

    assert event.run_id == RUN_ID
    assert event.transaction_id == TRANSACTION_ID
    assert event.tool_call_id == TOOL_CALL_ID
    assert event.tool_name == "read_text_file"
    assert event.arguments == {"path": "README.md"}
    assert event.effect is ToolEffect.READ
    assert event.kind is ToolEventKind.PROPOSED
    assert event.occurred_at.utcoffset() == timedelta(0)


def test_tool_events_are_frozen_after_creation() -> None:
    event = proposed_event()

    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'tool_name'"):
        setattr(event, "tool_name", "bash")


@pytest.mark.parametrize(
    ("event", "expected_kind"),
    [
        (
            ToolValidated(
                run_id=RUN_ID,
                transaction_id=TRANSACTION_ID,
                tool_call_id=TOOL_CALL_ID,
            ),
            ToolEventKind.VALIDATED,
        ),
        (
            ToolExecutionStarted(
                run_id=RUN_ID,
                transaction_id=TRANSACTION_ID,
                tool_call_id=TOOL_CALL_ID,
            ),
            ToolEventKind.EXECUTION_STARTED,
        ),
        (
            ToolCommitted(
                run_id=RUN_ID,
                transaction_id=TRANSACTION_ID,
                tool_call_id=TOOL_CALL_ID,
            ),
            ToolEventKind.COMMITTED,
        ),
        (
            ToolFailed(
                run_id=RUN_ID,
                transaction_id=TRANSACTION_ID,
                tool_call_id=TOOL_CALL_ID,
                reason="execution failed",
            ),
            ToolEventKind.FAILED,
        ),
    ],
    ids=["validated", "execution-started", "committed", "failed"],
)
def test_representative_event_kinds_are_stable(
    event: ToolEvent,
    expected_kind: ToolEventKind,
) -> None:
    assert event.kind is expected_kind
