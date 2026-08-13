from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.transactional.context.reducers import CodingToolResultReducer, ReducerLimits
from fast_agent.transactional.coordinator import TransactionCoordinator
from fast_agent.transactional.events import ToolResultStored
from fast_agent.transactional.models import RunId, TransactionId
from fast_agent.transactional.storage.artifact_store import ArtifactId, FileArtifactStore
from fast_agent.transactional.storage.event_store import SQLiteEventStore
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.display_suppression import suppress_interactive_display

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.llm.request_params import RequestParams
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


RUN_ID = RunId("agent-loop-run")
TRANSACTION_ID = TransactionId("agent-loop-transaction")
TOOL_CALL_ID = "agent-loop-shell-call"


class _LongShellThenDoneLlm(PassthroughLLM):
    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command
        self._turn = 0
        self.second_request: list[PromptMessageExtended] | None = None

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
                "Run the deterministic shell command",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    TOOL_CALL_ID: CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="bash",
                            arguments={"command": self._command},
                        ),
                    )
                },
            )

        self.second_request = [message.model_copy(deep=True) for message in multipart_messages]
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class _TransactionalToolLoopHarness(ToolAgent):
    def __init__(self, execution_agent: McpAgent) -> None:
        super().__init__(
            AgentConfig(name="transactional-loop-harness", use_history=False),
            tools=[],
        )
        self._execution_agent = execution_agent

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        del request_params
        return await self._execution_agent.run_tools(request)


def _execution_agent(
    tmp_path: Path,
) -> tuple[McpAgent, SQLiteEventStore, FileArtifactStore]:
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    coordinator = TransactionCoordinator(
        event_store,
        artifact_store,
        result_reducer=CodingToolResultReducer(ReducerLimits(max_result_bytes=512, max_items=6)),
        transaction_id_factory=lambda: TRANSACTION_ID,
    )
    settings = Settings(
        shell_execution=ShellSettings(
            write_text_file_mode="on",
            interactive_use_pty=False,
            show_bash=False,
        )
    )
    agent = McpAgent(
        config=AgentConfig(
            name="transactional-execution-agent",
            instruction="Run the local shell tool",
            servers=[],
            shell=True,
            cwd=tmp_path,
        ),
        connection_persistence=False,
        context=Context(config=settings),
        transactional_run_id=RUN_ID,
        tool_execution_interceptor=coordinator,
    )
    return agent, event_store, artifact_store


@pytest.mark.asyncio
async def test_bounded_shell_result_enters_next_llm_request(tmp_path: Path) -> None:
    raw_marker = "complete-agent-loop-evidence-" + "x" * 12_000
    script = "\n".join(
        [
            "print('FAILED tests/test_order.py::test_cutoff - AssertionError: cutoff rejected')",
            "print('E   AssertionError: cutoff rejected')",
            "print('tests/test_order.py:42: in test_cutoff')",
            f"print({raw_marker!r})",
            "raise SystemExit(1)",
        ]
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} pytest"
    execution_agent, event_store, artifact_store = _execution_agent(tmp_path)
    harness = _TransactionalToolLoopHarness(execution_agent)
    llm = _LongShellThenDoneLlm(command)
    harness._llm = llm
    try:
        with (
            suppress_interactive_display(),
            patch.object(
                execution_agent,
                "_available_tool_names_for_run_tools",
                new=AsyncMock(return_value=["bash"]),
            ),
            patch.object(
                harness.display,
                "show_assistant_message",
                new=AsyncMock(),
            ),
            patch.object(execution_agent.display, "show_tool_call"),
            patch.object(execution_agent.display, "show_tool_result"),
        ):
            response = await harness.generate_impl(
                [Prompt.user("Run the diagnostic shell command")],
                tools=[
                    Tool(
                        name="bash",
                        description="Run a shell command",
                        input_schema={
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    )
                ],
            )

        assert response.last_text() == "done"
        assert llm.second_request is not None
        tool_result_message = next(
            message for message in reversed(llm.second_request) if message.tool_results
        )
        assert tool_result_message.tool_results is not None
        visible_result = tool_result_message.tool_results[TOOL_CALL_ID]
        visible_content = visible_result.content[0]
        assert isinstance(visible_content, TextContent)
        assert len(visible_content.text.encode("utf-8")) <= 512
        assert raw_marker not in visible_content.text
        assert "exit_code: 1" in visible_content.text
        assert "tests/test_order.py::test_cutoff" in visible_content.text
        assert "AssertionError: cutoff rejected" in visible_content.text

        stored = next(
            item.event
            for item in event_store.events_for_transaction(TRANSACTION_ID)
            if isinstance(item.event, ToolResultStored)
        )
        assert f"full_output_artifact: {stored.artifact_id}" in visible_content.text
        raw_artifact = artifact_store.read(ArtifactId(stored.artifact_id))
        assert raw_marker.encode() in raw_artifact
    finally:
        await execution_agent._aggregator.close()
        event_store.close()
