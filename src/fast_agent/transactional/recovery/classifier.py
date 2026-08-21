from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from mcp.types import CallToolResult, TextContent

from fast_agent.tools.apply_patch_tool import APPLY_PATCH_TOOL_NAME
from fast_agent.tools.filesystem_tool_definitions import (
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
)
from fast_agent.transactional.execution import ToolExecutionRequest
from fast_agent.transactional.models import RunId, ToolCallId, ToolEffect

_WORKSPACE_WRITE_TOOLS = frozenset(
    {
        WRITE_TEXT_FILE_TOOL_NAME,
        APPLY_PATCH_TOOL_NAME,
        "execute",
        "exec",
        "bash",
        "process",
        "shell",
    }
)


class FailureKind(StrEnum):
    """Deterministic failure categories used by recovery policy."""

    VALIDATION = "validation"
    POLICY_DENIED = "policy_denied"
    TIMEOUT = "timeout"
    TASK_FAILURE = "task_failure"
    WORKSPACE_DIVERGENCE = "workspace_divergence"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    kind: FailureKind
    signature: str
    summary: str


class ToolEffectClassifier(Protocol):
    """Classify one planned call before any real tool execution."""

    def __call__(self, request: ToolExecutionRequest, /) -> ToolEffect: ...


class LocalCodingEffectClassifier:
    """Explicit first-stage rules for supported local coding tools."""

    def __call__(self, request: ToolExecutionRequest, /) -> ToolEffect:
        if request.server_name is not None:
            return ToolEffect.EXTERNAL_UNKNOWN
        if request.tool_name == READ_TEXT_FILE_TOOL_NAME:
            return ToolEffect.READ
        if request.tool_name in _WORKSPACE_WRITE_TOOLS:
            return ToolEffect.WORKSPACE_WRITE
        return ToolEffect.EXTERNAL_UNKNOWN


class FailureClassifier:
    """Classify a failed MCP result and derive a stable repeat signature."""

    def classify(self, tool_name: str, result: CallToolResult) -> FailureClassification:
        status, error_type, message = _failure_fields(result)
        kind = _failure_kind(status, error_type, message)
        summary = message or status or f"{tool_name} returned an error"
        payload = json.dumps(
            {
                "error_type": error_type.casefold(),
                "kind": kind.value,
                "message": " ".join(summary.casefold().split()),
                "tool_name": tool_name,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return FailureClassification(
            kind=kind,
            signature=f"sha256:{hashlib.sha256(payload).hexdigest()}",
            summary=summary,
        )


def classify_local_tool_effect(tool_name: str) -> ToolEffect:
    """Compatibility helper for callers that only have a tool name."""

    request = ToolExecutionRequest(
        run_id=RunId("effect-classification"),
        tool_call_id=ToolCallId("effect-classification"),
        tool_name=tool_name,
        arguments={},
    )
    return LocalCodingEffectClassifier()(request)


def _failure_fields(result: CallToolResult) -> tuple[str, str, str]:
    structured = result.structured_content
    status = _string_field(structured, "status")
    error_type = _string_field(structured, "error_type")
    message = _string_field(structured, "message") or _result_text(result)
    return status, error_type, message


def _string_field(value: dict[str, object] | None, key: str) -> str:
    if value is None:
        return ""
    field = value.get(key)
    return field if isinstance(field, str) else ""


def _result_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _failure_kind(status: str, error_type: str, message: str) -> FailureKind:
    normalized = f"{status} {error_type} {message}".casefold()
    if status in {"validation_failed", "invalid_arguments"}:
        return FailureKind.VALIDATION
    if status == "denied":
        return FailureKind.POLICY_DENIED
    if status == "budget_exhausted":
        return FailureKind.BUDGET_EXHAUSTED
    if status == "workspace_divergence" or "workspacedivergence" in normalized:
        return FailureKind.WORKSPACE_DIVERGENCE
    if status == "timeout" or "timeout" in normalized or "timed out" in normalized:
        return FailureKind.TIMEOUT
    return FailureKind.TASK_FAILURE
