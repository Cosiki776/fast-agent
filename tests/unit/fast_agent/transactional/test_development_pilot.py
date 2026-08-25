from __future__ import annotations

from pathlib import Path

from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from benchmarks.transactional.run_development_pilot import (
    _controlled_command_metrics,
    _find_controlled_result,
    _task_prompt,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.transactional.benchmark import load_task_manifest

CONTROLLED_COMMAND = "python3 -m unittest discover -s tests"


def test_controlled_command_requires_an_exact_completed_shell_call() -> None:
    exact_history = _shell_history(CONTROLLED_COMMAND)

    matched = _find_controlled_result(exact_history, CONTROLLED_COMMAND)

    assert matched is not None
    call_id, result = matched
    assert call_id == "call-1"
    assert result.content == [TextContent(type="text", text="test output")]
    assert (
        _find_controlled_result(
            _shell_history(f"{CONTROLLED_COMMAND} | tail -20"),
            CONTROLLED_COMMAND,
        )
        is None
    )


def test_controlled_command_baseline_metrics_use_visible_result_size() -> None:
    metrics = _controlled_command_metrics(
        _shell_history(CONTROLLED_COMMAND),
        CONTROLLED_COMMAND,
        runtime=None,
    )

    assert metrics is not None
    assert metrics.observed is True
    assert metrics.tool_result_bytes_before_reducer == metrics.model_visible_result_bytes
    assert metrics.tool_result_bytes_before_reducer is not None
    assert metrics.tool_result_bytes_before_reducer > 0


def test_controlled_task_prompt_preserves_other_investigation_commands() -> None:
    task = load_task_manifest(Path("benchmarks/transactional/tasks/long-output-001.yaml"))

    prompt = _task_prompt(task)

    assert CONTROLLED_COMMAND in prompt
    assert "before editing" in prompt
    assert "without pipes, redirection, output filtering" in prompt
    assert "Other investigation commands remain unrestricted" in prompt


def _shell_history(command: str) -> list[PromptMessageExtended]:
    return [
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="execute",
                        arguments={"command": command},
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call-1": CallToolResult(
                    content=[TextContent(type="text", text="test output")],
                    is_error=False,
                )
            },
        ),
    ]
