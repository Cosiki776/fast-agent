from __future__ import annotations

from mcp.types import CallToolResult, TextContent

from fast_agent.mcp.tool_result_metadata import update_tool_result_display_metadata
from fast_agent.transactional.context.reducers import (
    CodingToolResultReducer,
    ReducerLimits,
)
from fast_agent.transactional.execution import ToolExecutionRequest
from fast_agent.transactional.models import RunId, ToolCallId
from fast_agent.transactional.storage.artifact_store import ArtifactId

ARTIFACT_ID = ArtifactId(f"sha256:{'a' * 64}")


def _request(command: str) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        run_id=RunId("run-1"),
        tool_call_id=ToolCallId("call-1"),
        tool_name="execute",
        arguments={"command": command},
    )


def _result(text: str, *, exit_code: int) -> CallToolResult:
    result = CallToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=exit_code != 0,
    )
    update_tool_result_display_metadata(result, {"exit_code": exit_code})
    return result


def _text(result: CallToolResult) -> str:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def test_pytest_reducer_extracts_failures_and_bounds_long_output() -> None:
    noise = "\n".join(f"captured log line {index}" for index in range(20_000))
    raw = "\n".join(
        [
            "tests/test_orders.py::test_cutoff FAILED",
            "tests/test_orders.py:42: in test_cutoff",
            "E   AssertionError: expected accepted order",
            "FAILED tests/test_orders.py::test_cutoff - AssertionError",
            noise,
            "1 failed in 1.25s",
        ]
    )
    reducer = CodingToolResultReducer(ReducerLimits(max_result_bytes=1024))

    reduced = reducer(_request("uv run pytest -q"), _result(raw, exit_code=1), ARTIFACT_ID)
    text = _text(reduced)

    assert len(text.encode("utf-8")) <= 1024
    assert "result_type: pytest" in text
    assert "exit_code: 1" in text
    assert "tests/test_orders.py::test_cutoff" in text
    assert "tests/test_orders.py:42" in text
    assert "AssertionError: expected accepted order" in text
    assert f"full_output_artifact: {ARTIFACT_ID}" in text
    assert "captured log line 19999" not in text
    assert reduced.is_error is True


def test_git_diff_reducer_reports_files_counts_and_bounded_hunks() -> None:
    raw = "\n".join(
        [
            "diff --git a/src/orders.py b/src/orders.py",
            "--- a/src/orders.py",
            "+++ b/src/orders.py",
            "@@ -1,2 +1,3 @@",
            " context",
            "-old default",
            "+new default",
            "+extra guard",
        ]
    )
    reducer = CodingToolResultReducer(ReducerLimits(max_result_bytes=1024))

    text = _text(reducer(_request("git diff --stat HEAD"), _result(raw, exit_code=0), ARTIFACT_ID))

    assert "result_type: git_diff" in text
    assert "changed_files:\n- src/orders.py" in text
    assert "added_lines: 2" in text
    assert "deleted_lines: 1" in text
    assert "@@ -1,2 +1,3 @@" in text
    assert f"full_output_artifact: {ARTIFACT_ID}" in text


def test_shell_fallback_preserves_exit_code_and_head_tail() -> None:
    raw = "\n".join(f"line-{index}" for index in range(100))
    reducer = CodingToolResultReducer(ReducerLimits(max_result_bytes=512, max_items=6))

    text = _text(reducer(_request("python job.py"), _result(raw, exit_code=7), ARTIFACT_ID))

    assert len(text.encode("utf-8")) <= 512
    assert "result_type: shell" in text
    assert "exit_code: 7" in text
    assert "line-0" in text
    assert "line-99" in text
    assert "truncated_bytes:" in text
    assert f"full_output_artifact: {ARTIFACT_ID}" in text


def test_single_item_fallback_does_not_expand_tail() -> None:
    reducer = CodingToolResultReducer(ReducerLimits(max_result_bytes=512, max_items=1))

    text = _text(
        reducer(
            _request("python job.py"),
            _result("first\nmiddle\nlast", exit_code=0),
            ARTIFACT_ID,
        )
    )

    assert "first\n..." in text
    assert "middle" not in text
    assert "last" not in text
