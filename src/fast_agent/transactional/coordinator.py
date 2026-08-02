from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mcp.types import CallToolResult, TextContent

from fast_agent.tools.apply_patch_tool import APPLY_PATCH_TOOL_NAME
from fast_agent.tools.filesystem_tool_definitions import (
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
)
from fast_agent.transactional.events import (
    ToolCommitted,
    ToolDenied,
    ToolExecutionFailed,
    ToolExecutionStarted,
    ToolFailed,
    ToolProposed,
    ToolResultStored,
)
from fast_agent.transactional.execution import (
    ToolCallNext,
    ToolExecutionOutcome,
    ToolExecutionRequest,
)
from fast_agent.transactional.models import (
    ToolEffect,
    TransactionId,
    new_transaction_id,
)
from fast_agent.transactional.storage.artifact_store import ArtifactKind

if TYPE_CHECKING:
    from fast_agent.transactional.storage.artifact_store import FileArtifactStore
    from fast_agent.transactional.storage.event_store import SQLiteEventStore

type ToolDenialResolver = Callable[[ToolExecutionRequest], str | None]
type TransactionIdFactory = Callable[[], TransactionId]

_WORKSPACE_WRITE_TOOL_NAMES = frozenset(
    {
        WRITE_TEXT_FILE_TOOL_NAME,
        APPLY_PATCH_TOOL_NAME,
        "execute",
        "bash",
        "shell",
    }
)


class TransactionCoordinator:
    """Order one tool call across transaction persistence boundaries."""

    def __init__(
        self,
        event_store: SQLiteEventStore,
        artifact_store: FileArtifactStore,
        *,
        denial_reason: ToolDenialResolver | None = None,
        transaction_id_factory: TransactionIdFactory = new_transaction_id,
    ) -> None:
        self._event_store = event_store
        self._artifact_store = artifact_store
        self._denial_reason = denial_reason
        self._transaction_id_factory = transaction_id_factory

    async def coordinate(
        self,
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
        /,
    ) -> ToolExecutionOutcome:
        transaction_id = self._transaction_id_factory()
        self._event_store.append(
            ToolProposed(
                run_id=request.run_id,
                transaction_id=transaction_id,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                effect=classify_tool_effect(request.tool_name),
            )
        )

        denial_reason = self._denial_reason(request) if self._denial_reason is not None else None
        if denial_reason is not None:
            self._event_store.append(
                ToolDenied(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason=denial_reason,
                )
            )
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="policy denied",
                )
            )
            return ToolExecutionOutcome(result=_denied_result(denial_reason))

        self._event_store.append(
            ToolExecutionStarted(
                run_id=request.run_id,
                transaction_id=transaction_id,
                tool_call_id=request.tool_call_id,
            )
        )
        try:
            outcome = await call_next()
        except Exception as exc:
            error_type = type(exc).__name__
            message = str(exc)
            self._event_store.append(
                ToolExecutionFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    error_type=error_type,
                    message=message,
                )
            )
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="tool execution raised an exception",
                )
            )
            return ToolExecutionOutcome(
                result=_execution_failed_result(error_type=error_type, message=message)
            )

        result = outcome.result
        artifact = self._artifact_store.put(
            serialize_tool_result(result),
            media_type="application/json",
            kind=ArtifactKind.RAW_RESULT,
        )
        is_error = bool(result.isError)
        self._event_store.append(
            ToolResultStored(
                run_id=request.run_id,
                transaction_id=transaction_id,
                tool_call_id=request.tool_call_id,
                artifact_id=artifact.artifact_id,
                is_error=is_error,
            )
        )

        if is_error:
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="tool returned an error result",
                )
            )
        else:
            self._event_store.append(
                ToolCommitted(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                )
            )
        return outcome


def classify_tool_effect(tool_name: str) -> ToolEffect:
    """Classify the explicit first-phase local coding tool names."""

    if tool_name == READ_TEXT_FILE_TOOL_NAME:
        return ToolEffect.READ
    if tool_name in _WORKSPACE_WRITE_TOOL_NAMES:
        return ToolEffect.WORKSPACE_WRITE
    return ToolEffect.EXTERNAL_UNKNOWN


def serialize_tool_result(result: CallToolResult) -> bytes:
    """Serialize the complete standard MCP result stored as transaction evidence."""

    return result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")


def _denied_result(reason: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=f"Tool execution denied: {reason}")],
        structuredContent={"status": "denied", "reason": reason},
        isError=True,
    )


def _execution_failed_result(*, error_type: str, message: str) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Tool execution failed ({error_type}): {message}",
            )
        ],
        structuredContent={
            "status": "failed",
            "error_type": error_type,
            "message": message,
        },
        isError=True,
    )
