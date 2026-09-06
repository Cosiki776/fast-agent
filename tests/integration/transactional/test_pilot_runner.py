from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING

import pytest
import yaml
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent

from benchmarks.transactional import run_development_pilot as pilot
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.transactional.settings import TransactionalProfile
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.display_suppression import suppress_interactive_display

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import Tool

    from fast_agent.llm.request_params import RequestParams


def _git_repository(root: Path) -> Path:
    root.mkdir()
    pilot._git(root, "init", "-q")
    (root / "tracked.txt").write_text("baseline\n")
    pilot._git(root, "add", "tracked.txt")
    pilot._git(
        root,
        "-c",
        "user.name=Pilot Test",
        "-c",
        "user.email=pilot@example.invalid",
        "commit",
        "-qm",
        "baseline",
    )
    return root


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "strategy"),
    [(profile, None) for profile in TransactionalProfile]
    + [pytest.param(TransactionalProfile.FULL, "upstream", id="full-upstream")],
)
@pytest.mark.parametrize(
    "scenario", ["llm_limit", "tool_limit", "provider_error", "wall_limit", "success"]
)
async def test_pilot_budget_and_failure_evidence_through_real_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: TransactionalProfile,
    strategy: str | None,
    scenario: str,
) -> None:
    root = _git_repository(tmp_path / "root")
    fixtures = root / ".txagent-fixtures"
    fixtures.mkdir()
    fixture = _git_repository(fixtures / "fixture")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(pilot, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(pilot, "FAST_AGENT_HOME", home)
    manifest = tmp_path / "task.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "id": "scripted",
                "repository": ".txagent-fixtures/fixture",
                "base_commit": pilot._git(fixture, "rev-parse", "HEAD"),
                "issue": "Perform the scripted repair",
                "verification": {"command": "grep -qx baseline tracked.txt", "timeout_seconds": 5},
                "budget": {
                    "max_llm_calls": 2,
                    "max_tool_calls": 1 if scenario == "tool_limit" else 3,
                    "max_wall_time_seconds": 2 if scenario == "wall_limit" else 15,
                },
            }
        )
    )
    calls = 0

    async def scripted(
        self: PassthroughLLM,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        nonlocal calls
        self.retry_count = 0
        del multipart_messages, request_params, tools, is_template
        calls += 1
        if scenario == "wall_limit":
            await asyncio.Event().wait()
        if calls == 2 and scenario == "provider_error":
            return PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="provider unavailable")],
                stop_reason=LlmStopReason.ERROR,
                channels={
                    FAST_AGENT_ERROR_CHANNEL: [TextContent(type="text", text="scripted timeout")]
                },
            )
        if calls == 2 and scenario == "success":
            return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)
        return Prompt.assistant(
            "read",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls={
                f"read-{calls}": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="execute",
                        arguments={"command": "cat tracked.txt"},
                    ),
                ),
            },
        )

    monkeypatch.setattr(PassthroughLLM, "_apply_prompt_provider_specific", scripted)
    output = tmp_path / "output"
    with suppress_interactive_display():
        result = await pilot.run(
            argparse.Namespace(
                manifest=str(manifest),
                profile=profile.value,
                tool_output_strategy=strategy,
                model="passthrough",
                output=str(output),
            )
        )
    saved = pilot.PilotResult.model_validate_json((output / "result.json").read_text())
    assert saved == result
    expected_strategy = strategy or (
        "upstream" if profile is TransactionalProfile.BASELINE else "semantic"
    )
    assert result.tool_output_strategy.value == expected_strategy
    if expected_strategy == "upstream":
        assert result.semantic_reducer_version is None
    assert result.llm_calls == calls == (1 if scenario == "wall_limit" else 2)
    assert result.tool_calls == (
        0 if scenario == "wall_limit" else 2 if scenario == "llm_limit" else 1
    )
    assert result.implementation_commit == pilot._git(root, "rev-parse", "HEAD")
    if scenario == "success":
        assert result.status == "completed"
        assert result.verification_passed is True
        assert result.promotion_result == (
            "applied" if profile is TransactionalProfile.FULL else None
        )
        assert not (output / "failure.json").exists()
    else:
        assert result.status == "failed"
        assert result.failure_reason
        assert (
            result.error_type
            == {
                "llm_limit": "PilotBudgetExceededError",
                "tool_limit": "PilotBudgetExceededError",
                "provider_error": "PilotProviderError",
                "wall_limit": "TimeoutError",
            }[scenario]
        )
        assert result.verification_passed is None
        assert result.verification_exit_code is None
        assert (output / "failure.json").is_file()
        assert not (output / "verification.stdout").exists()
    if scenario != "wall_limit":
        assert result.model_visible_tool_result_bytes > 0
        if profile is not TransactionalProfile.BASELINE:
            assert result.tool_events
            assert result.artifact_output_bytes > 0
            events = json.loads((output / "events.json").read_text())
            assert events["tool"]
    if profile is TransactionalProfile.FULL:
        assert "run.started" in result.run_events
        assert result.run_id is not None
        if scenario != "success":
            assert "run.failed" in result.run_events
