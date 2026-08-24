from __future__ import annotations

from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.execution import ToolExecutionRequest
from fast_agent.transactional.models import RunId, ToolCallId, ToolEffect
from fast_agent.transactional.recovery.classifier import (
    FailureClassifier,
    FailureKind,
    LocalCodingEffectClassifier,
)


def _request(tool_name: str, *, server_name: str | None = None) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        run_id=RunId("run-1"),
        tool_call_id=ToolCallId("call-1"),
        tool_name=tool_name,
        arguments={},
        server_name=server_name,
    )


def _error(status: str, message: str, *, error_type: str = "") -> CallToolResult:
    structured = {"status": status, "message": message}
    if error_type:
        structured["error_type"] = error_type
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structured_content=structured,
        is_error=True,
    )


def test_local_effect_classifier_only_trusts_explicit_rules() -> None:
    classifier = LocalCodingEffectClassifier()

    assert classifier(_request("read_text_file")) is ToolEffect.READ
    assert classifier(_request("write_text_file")) is ToolEffect.WORKSPACE_WRITE
    assert classifier(_request("bash")) is ToolEffect.WORKSPACE_WRITE
    assert classifier(_request("exec")) is ToolEffect.WORKSPACE_WRITE
    assert classifier(_request("remote_tool")) is ToolEffect.EXTERNAL_UNKNOWN


def test_external_server_tool_cannot_inherit_a_local_tool_effect() -> None:
    classifier = LocalCodingEffectClassifier()

    assert (
        classifier(_request("read_text_file", server_name="remote-files"))
        is ToolEffect.EXTERNAL_UNKNOWN
    )
    assert (
        classifier(_request("execute", server_name="remote-shell")) is ToolEffect.EXTERNAL_UNKNOWN
    )


def test_failure_classifier_covers_runtime_failure_categories() -> None:
    classifier = FailureClassifier()

    cases = (
        (_error("validation_failed", "bad arguments"), FailureKind.VALIDATION),
        (_error("denied", "not allowed"), FailureKind.POLICY_DENIED),
        (_error("timeout", "command timed out"), FailureKind.TIMEOUT),
        (_error("budget_exhausted", "tool limit"), FailureKind.BUDGET_EXHAUSTED),
        (
            _error("failed", "restore mismatch", error_type="WorkspaceDivergenceError"),
            FailureKind.WORKSPACE_DIVERGENCE,
        ),
        (_error("failed", "pytest failed"), FailureKind.TASK_FAILURE),
    )
    for result, expected in cases:
        assert classifier.classify("bash", result).kind is expected


def test_failure_signature_is_stable_for_equivalent_whitespace() -> None:
    classifier = FailureClassifier()

    first = classifier.classify("bash", _error("failed", "same assertion failed"))
    second = classifier.classify("bash", _error("failed", " same   assertion failed "))

    assert first.signature == second.signature
