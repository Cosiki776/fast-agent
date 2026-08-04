from __future__ import annotations

import pytest

from fast_agent.transactional.budget import (
    BudgetDimension,
    RunBudgetLimits,
    RunBudgetTracker,
)


def test_tool_call_budget_refuses_the_next_action_at_hard_limit() -> None:
    tracker = RunBudgetTracker(RunBudgetLimits(max_tool_calls=2))

    assert tracker.start_tool_call().allowed is True
    second = tracker.start_tool_call()
    exhausted = tracker.start_tool_call()

    assert second.allowed is True
    assert second.warnings == (BudgetDimension.TOOL_CALLS,)
    assert exhausted.allowed is False
    assert exhausted.exhausted == (BudgetDimension.TOOL_CALLS,)
    assert tracker.snapshot.tool_calls == 2


def test_artifact_output_budget_blocks_the_following_tool_call() -> None:
    tracker = RunBudgetTracker(RunBudgetLimits(max_artifact_output_bytes=100))

    assert tracker.start_tool_call().allowed is True
    tracker.record_artifact_output(100)
    decision = tracker.start_tool_call()

    assert decision.allowed is False
    assert decision.exhausted == (BudgetDimension.ARTIFACT_OUTPUT_BYTES,)
    assert tracker.snapshot.artifact_output_bytes == 100


def test_llm_and_token_budgets_share_run_state() -> None:
    tracker = RunBudgetTracker(RunBudgetLimits(max_llm_calls=3, max_tokens=10))

    assert tracker.start_llm_call().allowed is True
    tracker.record_token_usage(10)
    decision = tracker.start_llm_call()

    assert decision.allowed is False
    assert decision.exhausted == (BudgetDimension.TOKENS,)
    assert tracker.snapshot.llm_calls == 1
    assert tracker.snapshot.total_tokens == 10


def test_missing_provider_usage_is_unknown_and_not_estimated() -> None:
    tracker = RunBudgetTracker(RunBudgetLimits(max_llm_calls=2, max_tokens=1))

    assert tracker.start_llm_call().allowed is True
    tracker.record_token_usage(None)
    second = tracker.start_llm_call()

    assert second.allowed is True
    assert tracker.snapshot.total_tokens is None
    assert tracker.start_llm_call().exhausted == (BudgetDimension.LLM_CALLS,)


def test_wall_time_uses_injected_monotonic_clock() -> None:
    now = 10.0

    def clock() -> float:
        return now

    tracker = RunBudgetTracker(RunBudgetLimits(max_wall_time_seconds=5), clock=clock)
    now = 15.0

    decision = tracker.check_wall_time()

    assert decision.allowed is False
    assert decision.exhausted == (BudgetDimension.WALL_TIME,)
    assert tracker.snapshot.elapsed_seconds == 5.0


def test_recovery_budget_has_a_separate_controlled_action() -> None:
    tracker = RunBudgetTracker(RunBudgetLimits(max_recovery_attempts=1))

    assert tracker.start_recovery().allowed is True
    decision = tracker.start_recovery()

    assert decision.allowed is False
    assert decision.exhausted == (BudgetDimension.RECOVERY_ATTEMPTS,)
    assert tracker.snapshot.recovery_attempts == 1


def test_budget_limits_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RunBudgetLimits(max_tool_calls=-1)
    with pytest.raises(ValueError, match="non-negative"):
        RunBudgetLimits(max_wall_time_seconds=-0.1)
