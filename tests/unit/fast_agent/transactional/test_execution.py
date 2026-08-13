from __future__ import annotations

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.execution import (
    ToolCallAlreadyExecutedError,
    ToolCallNext,
    ToolExecutionOutcome,
    ToolExecutionRequest,
    execute_with_interceptor,
)
from fast_agent.transactional.models import RunId, ToolCallId


def _text_result(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=is_error,
    )


def _request() -> ToolExecutionRequest:
    return ToolExecutionRequest(
        run_id=RunId("run-1"),
        tool_call_id=ToolCallId("call-1"),
        tool_name="write_text_file",
        arguments={"path": "notes.txt", "content": "hello"},
    )


@pytest.mark.asyncio
async def test_disabled_interceptor_executes_tool_once() -> None:
    calls = 0
    expected = _text_result("written")

    async def execute() -> CallToolResult:
        nonlocal calls
        calls += 1
        return expected

    outcome = await execute_with_interceptor(_request(), execute)

    assert calls == 1
    assert outcome.result is expected


@pytest.mark.asyncio
async def test_interceptor_observes_request_and_can_replace_result() -> None:
    observed: list[ToolExecutionRequest] = []

    async def execute() -> CallToolResult:
        return _text_result("raw")

    async def interceptor(
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
    ) -> ToolExecutionOutcome:
        observed.append(request)
        outcome = await call_next()
        assert outcome.result.content
        return ToolExecutionOutcome(result=_text_result("replaced"))

    outcome = await execute_with_interceptor(_request(), execute, interceptor)

    assert observed == [_request()]
    assert outcome.result.content
    content = outcome.result.content[0]
    assert isinstance(content, TextContent)
    assert content.text == "replaced"


@pytest.mark.asyncio
async def test_interceptor_can_return_synthetic_result_without_execution() -> None:
    executed = False

    async def execute() -> CallToolResult:
        nonlocal executed
        executed = True
        return _text_result("unexpected")

    async def deny(
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
    ) -> ToolExecutionOutcome:
        del request, call_next
        return ToolExecutionOutcome(result=_text_result("denied", is_error=True))

    outcome = await execute_with_interceptor(_request(), execute, deny)

    assert executed is False
    assert outcome.result.is_error is True


@pytest.mark.asyncio
async def test_call_next_rejects_second_execution() -> None:
    calls = 0

    async def execute() -> CallToolResult:
        nonlocal calls
        calls += 1
        return _text_result("written")

    async def call_twice(
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
    ) -> ToolExecutionOutcome:
        del request
        outcome = await call_next()
        with pytest.raises(
            ToolCallAlreadyExecutedError,
            match="call_next may only be called once",
        ):
            await call_next()
        return outcome

    outcome = await execute_with_interceptor(_request(), execute, call_twice)

    assert calls == 1
    assert outcome.result.is_error is False
