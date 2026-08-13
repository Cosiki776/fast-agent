from __future__ import annotations

from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.budget import RunBudgetLimits, RunBudgetTracker
from fast_agent.transactional.execution import ToolExecutionRequest
from fast_agent.transactional.models import RunId, ToolCallId, ToolEffect
from fast_agent.transactional.recovery.controller import (
    RecoveryAction,
    RecoveryController,
    RecoveryHandoff,
    recovery_handoff_result,
)


def _request(tool_name: str = "write_text_file") -> ToolExecutionRequest:
    return ToolExecutionRequest(
        run_id=RunId("run-1"),
        tool_call_id=ToolCallId("call-1"),
        tool_name=tool_name,
        arguments={},
    )


def _failed_result(message: str = "same assertion failed") -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structured_content={"status": "failed", "message": message},
        is_error=True,
    )


def test_repeated_write_failure_rolls_back_to_first_stable_checkpoint() -> None:
    controller = RecoveryController()

    first = controller.decide(
        _request(), ToolEffect.WORKSPACE_WRITE, _failed_result(), "checkpoint-stable"
    )
    second = controller.decide(
        _request(), ToolEffect.WORKSPACE_WRITE, _failed_result(), "checkpoint-later"
    )

    assert first.action is RecoveryAction.CONTINUE
    assert second.action is RecoveryAction.ROLLBACK_REPLAN
    assert second.checkpoint_id == "checkpoint-stable"
    assert second.failure_count == 2


def test_read_retry_and_recovery_budget_are_bounded() -> None:
    budget = RunBudgetTracker(RunBudgetLimits(max_recovery_attempts=1))
    controller = RecoveryController(run_budget=budget, repeated_failure_threshold=3)

    first = controller.decide(_request("read_text_file"), ToolEffect.READ, _failed_result(), None)
    second = controller.decide(_request("read_text_file"), ToolEffect.READ, _failed_result(), None)

    assert first.action is RecoveryAction.RETRY_READ
    assert second.action is RecoveryAction.ABORT
    assert budget.snapshot.recovery_attempts == 1


def test_policy_and_divergence_failures_abort_without_recovery() -> None:
    controller = RecoveryController()
    denied = CallToolResult(
        content=[TextContent(type="text", text="denied")],
        structured_content={"status": "denied", "message": "denied"},
        is_error=True,
    )
    diverged = CallToolResult(
        content=[TextContent(type="text", text="restore mismatch")],
        structured_content={
            "status": "workspace_divergence",
            "message": "restore mismatch",
        },
        is_error=True,
    )

    assert (
        controller.decide(_request(), ToolEffect.WORKSPACE_WRITE, denied, "checkpoint").action
        is RecoveryAction.ABORT
    )
    assert (
        controller.decide(_request(), ToolEffect.WORKSPACE_WRITE, diverged, "checkpoint").action
        is RecoveryAction.ABORT
    )


def test_handoff_result_is_bounded_and_machine_readable() -> None:
    handoff = RecoveryHandoff(
        workspace_version="cp-1",
        rolled_back=True,
        reverted_files=("src/order.py",),
        failed_hypothesis="wrong comparison" * 1000,
        evidence=("same assertion failed twice",),
        forbidden_repeat=("repeat the same patch",),
        required_next_step="re-localize callers",
    )

    result = recovery_handoff_result(_failed_result("x" * 10_000), handoff, max_result_bytes=512)

    content = result.content[0]
    assert isinstance(content, TextContent)
    assert len(content.text.encode()) <= 512
    assert result.structured_content is not None
    assert result.structured_content["status"] == "recovery_handoff"
