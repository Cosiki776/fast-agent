from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, Tool

from fast_agent.cli.runtime.harness_startup import run_cli_flow
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.config import get_settings, update_global_settings
from fast_agent.core.fastagent import FastAgent
from fast_agent.core.harness_app import AppOpenRequest
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.transactional.checkpoint.checkpoint import CheckpointManager
from fast_agent.transactional.run_events import RunEventKind
from fast_agent.transactional.settings import TransactionalProfile
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.display_suppression import suppress_interactive_display

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.agents.mcp_agent import McpAgent
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.harness import HarnessSession
    from fast_agent.llm.request_params import RequestParams
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.session.session_manager import SessionManager


class _TwoTimeoutsThenDoneLlm(PassthroughLLM):
    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command
        self._turn = 0
        self.final_request: list[PromptMessageExtended] | None = None

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del request_params, tools, is_template
        self._turn += 1
        if self._turn <= 2:
            return Prompt.assistant(
                "Run the bounded shell command",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    f"hang-{self._turn}": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="execute",
                            arguments={"command": self._command},
                        ),
                    )
                },
            )

        self.final_request = [message.model_copy(deep=True) for message in multipart_messages]
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class _OneCommandThenDoneLlm(PassthroughLLM):
    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command
        self._turn = 0
        self.final_request: list[PromptMessageExtended] | None = None

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del request_params, tools, is_template
        self._turn += 1
        if self._turn == 1:
            return Prompt.assistant(
                "Run the external command",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "external-1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="execute",
                            arguments={"command": self._command},
                        ),
                    )
                },
            )

        self.final_request = [message.model_copy(deep=True) for message in multipart_messages]
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


def _git_repository(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TxAgent Test",
            "-c",
            "user.email=txagent@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    return root


def _fast_agent(
    tmp_path: Path,
    *,
    profile: TransactionalProfile,
    workspace: Path,
    keep_worktree: bool = True,
) -> tuple[FastAgent, Path, Path]:
    runtime_root = tmp_path / "transactional-runtime"
    home = tmp_path / "home"
    home.mkdir()
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "default_model: passthrough",
                "session_history: false",
                "shell_execution:",
                "  tool_profile: native",
                "  write_text_file_mode: on",
                "  interactive_use_pty: false",
                "  show_bash: false",
                "transactional:",
                f"  profile: {profile.value}",
                f"  runtime_root: {runtime_root}",
                f"  keep_worktree: {str(keep_worktree).lower()}",
                "  shell_terminal_timeout_seconds: 0.15",
                "  max_wall_time_seconds: 30",
            ]
        ),
        encoding="utf-8",
    )
    fast = FastAgent(
        "transactional harness test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
        home=home,
        workspace=workspace,
    )

    @fast.agent(name="main", model="passthrough", shell=True, default=True)
    async def main() -> None:
        pass

    return fast, config_path, home


def _request(
    *,
    config_path: Path,
    home: Path,
    workspace: Path,
    message: str | None = None,
    prompt_file: str | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        name="transactional harness test",
        instruction=None,
        config_path=str(config_path),
        server_list=None,
        agent_cards=None,
        card_tools=None,
        model="passthrough",
        message=message,
        prompt_file=prompt_file,
        result_file=None,
        resume=None,
        startup_mcp_servers=None,
        mcp_startup_notices=(),
        agent_name="main",
        target_agent_name="main",
        skills_directory=None,
        home=home,
        no_home=False,
        shell_runtime=True,
        no_shell=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
        quiet=True,
        workspace=workspace,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "uses_prompt_file"),
    [(None, False), ("run once", False), (None, True)],
    ids=["repl", "message", "prompt-file"],
)
async def test_cli_modes_create_transactional_harness_runtime(
    tmp_path: Path,
    message: str | None,
    uses_prompt_file: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("run from file", encoding="utf-8")
    old_settings = get_settings()
    try:
        fast, config_path, home = _fast_agent(
            tmp_path,
            profile=TransactionalProfile.REDUCER,
            workspace=workspace,
        )
        request = _request(
            config_path=config_path,
            home=home,
            workspace=workspace,
            message=message,
            prompt_file=str(prompt_path) if uses_prompt_file else None,
        )
        observed: list[tuple[TransactionalProfile, bool]] = []

        async def flow(
            agent_app: AgentApp,
            request: AgentRunRequest,
            *,
            session_manager: SessionManager | None = None,
            harness_session: HarnessSession | None = None,
        ) -> None:
            del agent_app, request, session_manager
            assert harness_session is not None
            runtime = harness_session._record.transactional_runtime
            assert runtime is not None
            observed.append((runtime.profile, runtime.workspace is None))

        with suppress_interactive_display():
            await run_cli_flow(fast, request, flow=flow)

        assert observed == [(TransactionalProfile.REDUCER, True)]
    finally:
        update_global_settings(old_settings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_cli_denies_unknown_side_effect_before_shell_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _git_repository(tmp_path / "repository")
    old_settings = get_settings()

    async def deny(*args, **kwargs) -> str:
        del args, kwargs
        return "deny"

    monkeypatch.setattr("fast_agent.cli.runtime.tool_approval.get_selection_input", deny)
    try:
        fast, config_path, home = _fast_agent(
            tmp_path,
            profile=TransactionalProfile.FULL,
            workspace=repository,
        )
        llm = _OneCommandThenDoneLlm("curl https://example.test && touch should-not-exist.txt")
        request = _request(
            config_path=config_path,
            home=home,
            workspace=repository,
            message="request an external side effect",
        )
        observed = False

        async def flow(
            agent_app: AgentApp,
            request: AgentRunRequest,
            *,
            session_manager: SessionManager | None = None,
            harness_session: HarnessSession | None = None,
        ) -> None:
            nonlocal observed
            del request, session_manager
            assert harness_session is not None
            runtime = harness_session._record.transactional_runtime
            assert runtime is not None
            assert runtime.worktree is not None
            cast("McpAgent", agent_app["main"])._llm = llm

            assert await harness_session.send("request an external side effect") == "done"
            assert llm.final_request is not None
            denied = next(
                result
                for message in llm.final_request
                for result in (message.tool_results or {}).values()
            )
            assert denied.structured_content == {
                "status": "denied",
                "reason": "Tool approval denied",
            }
            assert not (runtime.worktree.worktree_path / "should-not-exist.txt").exists()
            observed = True

        with suppress_interactive_display():
            await run_cli_flow(fast, request, flow=flow)

        assert observed is True
    finally:
        update_global_settings(old_settings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_harness_sessions_own_isolated_transactional_resources(
    tmp_path: Path,
) -> None:
    repository = _git_repository(tmp_path / "repository")
    old_settings = get_settings()
    try:
        fast, _, _ = _fast_agent(
            tmp_path,
            profile=TransactionalProfile.FULL,
            workspace=repository,
        )
        with suppress_interactive_display():
            async with fast.harness() as harness:
                app = harness.app()
                async with (
                    app.open(AppOpenRequest(session_id="first", agent="main")) as first,
                    app.open(AppOpenRequest(session_id="second", agent="main")) as second,
                ):
                    first_runtime = first.env.harness_session._record.transactional_runtime
                    second_runtime = second.env.harness_session._record.transactional_runtime
                    assert first_runtime is not None
                    assert second_runtime is not None
                    assert first_runtime.run_id != second_runtime.run_id
                    assert first_runtime.workspace != second_runtime.workspace
                    assert first_runtime.event_store.path != second_runtime.event_store.path
                    assert first_runtime.artifact_store.root != second_runtime.artifact_store.root
                    assert first_runtime.budget is not second_runtime.budget
    finally:
        update_global_settings(old_settings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_harness_retains_unverified_worktree_when_cleanup_is_requested(
    tmp_path: Path,
) -> None:
    repository = _git_repository(tmp_path / "repository")
    old_settings = get_settings()
    worktree_path: Path | None = None
    try:
        fast, _, _ = _fast_agent(
            tmp_path,
            profile=TransactionalProfile.FULL,
            workspace=repository,
            keep_worktree=False,
        )
        with suppress_interactive_display():
            async with fast.harness() as harness:
                session = await harness.session("manual-review", agent_name="main")
                runtime = session.transactional_runtime
                assert runtime is not None
                assert runtime.workspace is not None
                worktree_path = runtime.workspace
                worktree_path.joinpath("result.txt").write_text("review me\n", encoding="utf-8")

                report = session.worktree_only_completion_report()
                assert report is not None
                assert report.added == ("result.txt",)
                assert not repository.joinpath("result.txt").exists()

        assert worktree_path is not None
        assert worktree_path.joinpath("result.txt").read_text(encoding="utf-8") == "review me\n"
    finally:
        update_global_settings(old_settings)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="process death assertion uses Unix signals")
async def test_full_harness_timeout_restores_checkpoint_and_hands_off_to_llm(
    tmp_path: Path,
) -> None:
    repository = _git_repository(tmp_path / "repository")
    old_settings = get_settings()
    try:
        fast, config_path, home = _fast_agent(
            tmp_path,
            profile=TransactionalProfile.FULL,
            workspace=repository,
        )
        script = "\n".join(
            [
                "from pathlib import Path",
                "import os, time",
                "Path('attempt.txt').write_text('dirty', encoding='utf-8')",
                "print(f'PID={os.getpid()}', flush=True)",
                "time.sleep(30)",
            ]
        )
        llm = _TwoTimeoutsThenDoneLlm(shlex.join([sys.executable, "-c", script]))
        request = _request(
            config_path=config_path,
            home=home,
            workspace=repository,
            message="exercise recovery",
        )
        observed = False

        async def flow(
            agent_app: AgentApp,
            request: AgentRunRequest,
            *,
            session_manager: SessionManager | None = None,
            harness_session: HarnessSession | None = None,
        ) -> None:
            nonlocal observed
            del request, session_manager
            assert harness_session is not None
            runtime = harness_session._record.transactional_runtime
            assert runtime is not None
            assert runtime.worktree is not None
            cast("McpAgent", agent_app["main"])._llm = llm

            assert await harness_session.send("exercise recovery") == "done"
            assert llm.final_request is not None
            tool_results = [
                result
                for message in llm.final_request
                for result in (message.tool_results or {}).values()
            ]
            handoff = next(
                (
                    result
                    for result in tool_results
                    if result.structured_content is not None
                    and result.structured_content.get("status") == "recovery_handoff"
                ),
                None,
            )
            assert handoff is not None, [result.model_dump() for result in tool_results]
            assert handoff.structured_content is not None
            handoff_data = handoff.structured_content["handoff"]
            assert isinstance(handoff_data, dict)
            assert handoff_data["rolled_back"] is True
            assert not (runtime.worktree.worktree_path / "attempt.txt").exists()

            checkpoints_root = runtime.artifact_store.root.parent / "checkpoints"
            checkpoint_manager = CheckpointManager(runtime.worktree, checkpoints_root)
            assert handoff_data["workspace_version"] == checkpoint_manager.current_version()
            assert runtime.run_event_store is not None
            assert [
                stored.event.kind
                for stored in runtime.run_event_store.events_for_run(runtime.run_id)
            ] == [
                RunEventKind.STARTED,
                RunEventKind.RECOVERY_STARTED,
                RunEventKind.RECOVERED,
            ]

            artifacts = b"\n".join(
                path.read_bytes()
                for path in (runtime.artifact_store.root / "objects").rglob("*")
                if path.is_file()
            )
            pids = {int(value) for value in re.findall(rb"PID=(\d+)", artifacts)}
            assert len(pids) == 2
            for pid in pids:
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
            observed = True

        with suppress_interactive_display():
            await run_cli_flow(fast, request, flow=flow)

        assert observed is True
    finally:
        update_global_settings(old_settings)
