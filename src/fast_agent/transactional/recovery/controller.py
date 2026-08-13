from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.models import ToolEffect
from fast_agent.transactional.recovery.classifier import (
    FailureClassification,
    FailureClassifier,
    FailureKind,
)

if TYPE_CHECKING:
    from fast_agent.transactional.budget import RunBudgetTracker
    from fast_agent.transactional.execution import ToolExecutionRequest


class RecoveryAction(StrEnum):
    CONTINUE = "continue"
    RETRY_READ = "retry_read"
    ROLLBACK_REPLAN = "rollback_replan"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    failure: FailureClassification
    failure_count: int
    checkpoint_id: str | None = None
    budget_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryHandoff:
    workspace_version: str
    rolled_back: bool
    reverted_files: tuple[str, ...]
    failed_hypothesis: str
    evidence: tuple[str, ...]
    forbidden_repeat: tuple[str, ...]
    required_next_step: str


class RecoveryController:
    """Choose bounded recovery actions without re-executing tool effects."""

    def __init__(
        self,
        *,
        run_budget: RunBudgetTracker | None = None,
        failure_classifier: FailureClassifier | None = None,
        repeated_failure_threshold: int = 2,
    ) -> None:
        if repeated_failure_threshold < 2:
            raise ValueError("repeated_failure_threshold must be at least two")
        self._run_budget = run_budget
        self._failure_classifier = failure_classifier or FailureClassifier()
        self._repeated_failure_threshold = repeated_failure_threshold
        self._failure_counts: dict[str, int] = {}
        self._first_checkpoints: dict[str, str] = {}

    def decide(
        self,
        request: ToolExecutionRequest,
        effect: ToolEffect,
        result: CallToolResult,
        checkpoint_id: str | None,
    ) -> RecoveryDecision:
        failure = self._failure_classifier.classify(request.tool_name, result)
        count = self._failure_counts.get(failure.signature, 0) + 1
        self._failure_counts[failure.signature] = count

        if checkpoint_id is not None:
            self._first_checkpoints.setdefault(failure.signature, checkpoint_id)

        if failure.kind in {
            FailureKind.BUDGET_EXHAUSTED,
            FailureKind.POLICY_DENIED,
            FailureKind.VALIDATION,
            FailureKind.WORKSPACE_DIVERGENCE,
        }:
            return RecoveryDecision(RecoveryAction.ABORT, failure, count)

        if effect is ToolEffect.READ:
            if count >= self._repeated_failure_threshold:
                return RecoveryDecision(RecoveryAction.ABORT, failure, count)
            if not self._start_recovery():
                return RecoveryDecision(
                    RecoveryAction.ABORT,
                    failure,
                    count,
                    budget_exhausted=True,
                )
            return RecoveryDecision(RecoveryAction.RETRY_READ, failure, count)

        stable_checkpoint = self._first_checkpoints.get(failure.signature)
        if count < self._repeated_failure_threshold:
            return RecoveryDecision(RecoveryAction.CONTINUE, failure, count)
        if stable_checkpoint is None:
            return RecoveryDecision(RecoveryAction.ABORT, failure, count)
        if not self._start_recovery():
            return RecoveryDecision(
                RecoveryAction.ABORT,
                failure,
                count,
                budget_exhausted=True,
            )
        return RecoveryDecision(
            RecoveryAction.ROLLBACK_REPLAN,
            failure,
            count,
            checkpoint_id=stable_checkpoint,
        )

    def _start_recovery(self) -> bool:
        return self._run_budget is None or self._run_budget.start_recovery().allowed


def recovery_handoff_result(
    result: CallToolResult,
    handoff: RecoveryHandoff,
    *,
    max_result_bytes: int = 4096,
) -> CallToolResult:
    """Encode recovery facts into the bounded Tool Result seen by the next LLM call."""

    payload = json.dumps(asdict(handoff), ensure_ascii=False, sort_keys=True)
    evidence = _result_text(result)
    text = _truncate_utf8(
        f"Previous bounded tool result:\n{evidence}\nRecovery Handoff:\n{payload}",
        max_result_bytes,
    )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content={"status": "recovery_handoff", "handoff": asdict(handoff)},
        is_error=True,
    )


def _result_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _truncate_utf8(text: str, max_bytes: int) -> str:
    if max_bytes < 256:
        raise ValueError("max_result_bytes must be at least 256")
    payload = text.encode()
    if len(payload) <= max_bytes:
        return text
    marker = "\n[recovery handoff truncated]"
    available = max_bytes - len(marker.encode())
    return payload[:available].decode(errors="ignore") + marker
