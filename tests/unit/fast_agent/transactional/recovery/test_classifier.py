from __future__ import annotations

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.tools.shell_process import process_result
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
    assert classifier(_request("edit_file")) is ToolEffect.WORKSPACE_WRITE
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


def _shell_failure(process_id: str, output: str = "", exit_code: int = 1) -> CallToolResult:
    return process_result(
        f"{output}process_id: {process_id}\nprocess exit code was {exit_code}",
        is_error=True,
        metadata={
            "process_id": process_id,
            "lifecycle": "session",
            "process_status": "failed",
            "exit_code": exit_code,
        },
    )


@pytest.mark.parametrize("output", ["", "AssertionError: expected 10, got 11\n"])
def test_shell_failure_signature_ignores_only_generated_process_id(output: str) -> None:
    classifier = FailureClassifier()
    first_result = _shell_failure("process-1", output)
    original = first_result.model_dump_json()
    first = classifier.classify("execute", first_result)
    second = classifier.classify("execute", _shell_failure("process-2", output))

    assert first.signature == second.signature
    assert first.kind is FailureKind.TASK_FAILURE
    assert "process-1" in first.summary  # Diagnostic evidence retains the real ID.
    assert first_result.model_dump_json() == original


@pytest.mark.parametrize(
    ("output", "exit_code"),
    [("AssertionError: expected 10, got 12\n", 1), ("AssertionError: expected 10, got 11\n", 2)],
)
def test_shell_failure_signature_preserves_actual_error_and_exit_code(
    output: str,
    exit_code: int,
) -> None:
    classifier = FailureClassifier()
    first = classifier.classify(
        "execute", _shell_failure("process-1", "AssertionError: expected 10, got 11\n")
    )
    second = classifier.classify("execute", _shell_failure("process-2", output, exit_code))
    assert first.signature != second.signature


def test_program_output_resembling_process_metadata_is_not_removed() -> None:
    classifier = FailureClassifier()
    first = classifier.classify("execute", _shell_failure("process-1", "process_id: process-1\n"))
    second = classifier.classify("execute", _shell_failure("process-2", "process_id: process-2\n"))
    assert first.signature != second.signature


@pytest.mark.parametrize(
    "case", ["missing_metadata", "mismatched_metadata", "structured_message", "other_tool"]
)
def test_process_id_normalization_requires_the_native_shell_envelope(case: str) -> None:
    classifier = FailureClassifier()
    results = [_shell_failure("process-1"), _shell_failure("process-2")]
    for result in results:
        if case == "missing_metadata":
            result.meta = None
        elif case == "mismatched_metadata":
            result.meta = _shell_failure("process-unrelated").meta
        elif case == "structured_message":
            block = result.content[0]
            assert isinstance(block, TextContent)
            result.structured_content = {"message": block.text}
    tool_name = "remote_tool" if case == "other_tool" else "execute"
    assert (
        classifier.classify(tool_name, results[0]).signature
        != classifier.classify(tool_name, results[1]).signature
    )
