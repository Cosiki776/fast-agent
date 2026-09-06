from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from benchmarks.transactional.run_development_pilot import (
    PilotResult,
    _controlled_command_metrics,
    _find_controlled_result,
    _pilot_tool_output,
    _task_prompt,
    _verify_workspace,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.transactional.benchmark import load_task_manifest, task_manifest_sha256
from fast_agent.transactional.settings import ToolOutputStrategy, TransactionalProfile

CONTROLLED_COMMAND = "python3 -m unittest discover -s tests"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (TransactionalProfile.BASELINE, "upstream"),
        (TransactionalProfile.REDUCER, "semantic"),
        (TransactionalProfile.FULL, "semantic"),
    ],
)
def test_pilot_default_packages_are_preserved(profile: TransactionalProfile, expected: str) -> None:
    assert _pilot_tool_output(profile, None).strategy.value == expected


def test_full_upstream_removes_semantic_version() -> None:
    settings = _pilot_tool_output(TransactionalProfile.FULL, "upstream")
    assert settings.strategy is ToolOutputStrategy.UPSTREAM
    assert settings.semantic_reducer_version is None


def test_baseline_cannot_claim_to_install_a_semantic_reducer() -> None:
    with pytest.raises(ValueError, match="does not install"):
        _pilot_tool_output(TransactionalProfile.BASELINE, "semantic")


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


@pytest.mark.parametrize("timed_out", [False, True])
def test_external_grader_failure_preserves_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timed_out: bool,
) -> None:
    task = load_task_manifest(Path("benchmarks/transactional/tasks/boundary-check-001.yaml"))
    result = PilotResult(
        task_id=task.id,
        manifest_sha256=task_manifest_sha256(task),
        profile=TransactionalProfile.BASELINE,
        tool_output_strategy=ToolOutputStrategy.UPSTREAM,
        semantic_reducer_version=None,
        model="scripted",
        base_commit=task.base_commit,
        implementation_commit="test",
        implementation_dirty=False,
        budget=task.budget,
        llm_calls=3,
        tool_calls=2,
    )

    def failed_grader(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        if timed_out:
            raise subprocess.TimeoutExpired(
                "verify", 1, output=b"started", stderr=b"failure details"
            )
        return subprocess.CompletedProcess(
            "verify", 1, stdout=b"started", stderr=b"failure details"
        )

    monkeypatch.setattr(subprocess, "run", failed_grader)
    with pytest.raises(RuntimeError, match="External verification"):
        _verify_workspace(task, result, tmp_path, tmp_path)
    assert result.verification_passed is False
    assert result.verification_timed_out is timed_out
    assert result.verification_exit_code == (None if timed_out else 1)
    assert (tmp_path / "verification.stdout").read_bytes() == b"started"
    assert (tmp_path / "verification.stderr").read_bytes() == b"failure details"
    assert result.verification_stdout_bytes == len(b"started")
    assert result.verification_stderr_bytes == len(b"failure details")
    assert result.verification_output_sha256
    assert result.llm_calls == 3 and result.tool_calls == 2
