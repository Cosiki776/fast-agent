from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mcp.types import CallToolResult

if TYPE_CHECKING:
    from fast_agent.transactional.models import JsonValue, RunId, ToolCallId


@dataclass(frozen=True, slots=True)
class ToolExecutionRequest:
    """Stable identity and input for one intercepted tool call."""

    run_id: RunId
    tool_call_id: ToolCallId
    tool_name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ToolExecutionOutcome:
    """Result returned from an interceptor to the agent tool loop."""

    result: CallToolResult


type ToolCallNext = Callable[[], Awaitable[ToolExecutionOutcome]]
type ToolExecutor = Callable[[], Awaitable[CallToolResult]]


class ToolExecutionInterceptor(Protocol):
    """Optional middleware around one concrete tool execution."""

    async def __call__(
        self,
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
        /,
    ) -> ToolExecutionOutcome: ...


class ToolCallAlreadyExecutedError(RuntimeError):
    """Raised when an interceptor tries to execute the same tool call twice."""


class ToolExecutionUncertainError(RuntimeError):
    """Fail-stop signal for a started tool that may still produce side effects."""


@dataclass(slots=True)
class _SingleUseToolCall:
    execute: ToolExecutor
    _called: bool = False

    async def __call__(self) -> ToolExecutionOutcome:
        # The runtime owns this guard so an interceptor cannot accidentally
        # repeat a filesystem or shell side effect.
        if self._called:
            raise ToolCallAlreadyExecutedError("call_next may only be called once")
        self._called = True
        return ToolExecutionOutcome(result=await self.execute())


async def execute_with_interceptor(
    request: ToolExecutionRequest,
    execute: ToolExecutor,
    interceptor: ToolExecutionInterceptor | None = None,
) -> ToolExecutionOutcome:
    """Execute one tool directly or through a single-use interceptor boundary."""

    call_next = _SingleUseToolCall(execute)
    if interceptor is None:
        return await call_next()
    return await interceptor(request, call_next)
