from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
    Tool,
)

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.transactional.models import RunId
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.transactional.execution import (
        ToolCallNext,
        ToolExecutionInterceptor,
        ToolExecutionOutcome,
        ToolExecutionRequest,
    )


class RecordingShellRuntime:
    def __init__(self) -> None:
        self.tool = Tool(
            name="execute",
            description="Run a shell command",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
        self.tools = [self.tool]
        self.calls: list[str] = []
        self.active_calls = 0
        self.max_active_calls = 0

    def owns_tool(self, name: str) -> bool:
        return name == self.tool.name

    def metadata(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"variant": "shell", "command": arguments.get("command")}

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        assert self.owns_tool(name)
        return await self.execute(
            arguments,
            tool_use_id,
            show_tool_call_id=show_tool_call_id,
            defer_display_to_tool_result=defer_display_to_tool_result,
        )

    async def execute(
        self,
        arguments: dict[str, object] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        del show_tool_call_id, defer_display_to_tool_result
        command = str((arguments or {}).get("command", ""))
        self.calls.append(f"start:{tool_use_id}:{command}")
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.01)
        self.active_calls -= 1
        self.calls.append(f"end:{tool_use_id}:{command}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"ran {command}")],
            is_error=False,
        )


def _tool_request() -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="run both")],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="execute",
                    arguments={
                        "command": "first",
                        "metadata": {"labels": ["original"]},
                    },
                )
            ),
            "call-2": CallToolRequest(
                params=CallToolRequestParams(
                    name="execute",
                    arguments={"command": "second"},
                )
            ),
        },
    )


def _agent(
    *,
    interceptor: ToolExecutionInterceptor | None = None,
) -> tuple[McpAgent, RecordingShellRuntime]:
    agent = McpAgent(
        config=AgentConfig(
            name="test",
            instruction="Run tools",
            servers=[],
            shell=True,
        ),
        context=Context(),
        transactional_run_id=RunId("run-1") if interceptor is not None else None,
        tool_execution_interceptor=interceptor,
    )
    shell_runtime = RecordingShellRuntime()
    agent._shell_runtime = cast("Any", shell_runtime)
    agent._shell_runtime_enabled = True
    return agent, shell_runtime


@pytest.mark.asyncio
async def test_disabled_interceptor_preserves_parallel_baseline() -> None:
    agent, shell_runtime = _agent()

    result = await agent.run_tools(_tool_request())

    assert shell_runtime.max_active_calls == 2
    assert result.tool_results is not None
    assert list(result.tool_results) == ["call-1", "call-2"]
    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_interceptor_receives_each_id_and_forces_sequential_execution() -> None:
    observed: list[ToolExecutionRequest] = []

    async def interceptor(
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
    ) -> ToolExecutionOutcome:
        observed.append(request)
        if request.tool_call_id == "call-1":
            metadata = request.arguments["metadata"]
            assert isinstance(metadata, dict)
            labels = metadata["labels"]
            assert isinstance(labels, list)
            labels.append("intercepted")
        return await call_next()

    agent, shell_runtime = _agent(interceptor=interceptor)
    tool_request = _tool_request()

    result = await agent.run_tools(tool_request)

    assert [request.run_id for request in observed] == [RunId("run-1"), RunId("run-1")]
    assert [request.tool_call_id for request in observed] == ["call-1", "call-2"]
    assert [request.tool_name for request in observed] == ["execute", "execute"]
    assert [request.arguments for request in observed] == [
        {
            "command": "first",
            "metadata": {"labels": ["original", "intercepted"]},
        },
        {"command": "second"},
    ]
    assert tool_request.tool_calls
    assert tool_request.tool_calls["call-1"].params.arguments == {
        "command": "first",
        "metadata": {"labels": ["original"]},
    }
    assert shell_runtime.max_active_calls == 1
    assert shell_runtime.calls == [
        "start:call-1:first",
        "end:call-1:first",
        "start:call-2:second",
        "end:call-2:second",
    ]
    assert result.tool_results is not None
    assert list(result.tool_results) == ["call-1", "call-2"]
    await agent._aggregator.close()
