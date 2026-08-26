from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mcp.types import CallToolResult, TextContent

from fast_agent.mcp.tool_result_metadata import tool_result_display_metadata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fast_agent.transactional.execution import ToolExecutionRequest
    from fast_agent.transactional.storage.artifact_store import ArtifactId


_DEFAULT_MAX_RESULT_BYTES = 8 * 1024
_FAILED_TEST_PATTERN = re.compile(r"^FAILED\s+(?P<name>\S+?)(?:\s+-|$)")
_STACK_FRAME_PATTERN = re.compile(r"^(?P<path>[^\s:].*?\.py):(?P<line>\d+)(?::|\s+in\s+).*$")
_DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+) b/(.+)$")
_EXIT_CODE_PATTERN = re.compile(r"process exit code was (-?\d+)\s*$")


class ToolResultReducer(Protocol):
    """Reduce one persisted tool result before it enters model history."""

    def __call__(
        self,
        request: ToolExecutionRequest,
        result: CallToolResult,
        artifact_id: ArtifactId,
        /,
    ) -> CallToolResult: ...


@dataclass(frozen=True, slots=True)
class ReducerLimits:
    max_result_bytes: int = _DEFAULT_MAX_RESULT_BYTES
    max_items: int = 12
    max_hunks: int = 4
    max_hunk_lines: int = 24

    def __post_init__(self) -> None:
        if self.max_result_bytes < 256:
            raise ValueError("max_result_bytes must be at least 256")
        if self.max_items <= 0 or self.max_hunks <= 0 or self.max_hunk_lines <= 0:
            raise ValueError("reducer collection limits must be positive")


@dataclass(frozen=True, slots=True)
class CodingToolResultReducer:
    """Select deterministic pytest, diff, or shell fallback reduction."""

    limits: ReducerLimits = ReducerLimits()

    def __call__(
        self,
        request: ToolExecutionRequest,
        result: CallToolResult,
        artifact_id: ArtifactId,
        /,
    ) -> CallToolResult:
        text = _result_text(result)
        if len(text.encode("utf-8")) <= self.limits.max_result_bytes:
            return result
        command = request.arguments.get("command")
        command_text = command if isinstance(command, str) else ""
        command_tokens = _command_tokens(command_text)

        if _is_pytest_command(command_tokens):
            summary = _pytest_summary(text, result, self.limits)
        elif _is_git_diff_command(command_tokens):
            summary = _git_diff_summary(text, result, self.limits)
        else:
            summary = _shell_fallback_summary(text, result, self.limits)
        return _reduced_result(
            result,
            summary,
            artifact_id=artifact_id,
            max_result_bytes=self.limits.max_result_bytes,
        )


def bounded_fallback_result(
    result: CallToolResult,
    artifact_id: ArtifactId,
    *,
    max_result_bytes: int = _DEFAULT_MAX_RESULT_BYTES,
) -> CallToolResult:
    """Return a bounded result when a configured semantic reducer fails."""

    limits = ReducerLimits(max_result_bytes=max_result_bytes)
    summary = _shell_fallback_summary(_result_text(result), result, limits)
    return _reduced_result(
        result,
        summary,
        artifact_id=artifact_id,
        max_result_bytes=max_result_bytes,
    )


def _pytest_summary(text: str, result: CallToolResult, limits: ReducerLimits) -> str:
    lines = text.splitlines()
    failed_tests = _unique_matches(lines, _FAILED_TEST_PATTERN, "name", limits.max_items)
    exceptions = _unique_lines(
        (line.strip() for line in lines if line.startswith("E ") or line.startswith("E   ")),
        limits.max_items,
    )
    stack_frames = _unique_lines(
        (
            f"{match.group('path')}:{match.group('line')}"
            for line in lines
            if (match := _STACK_FRAME_PATTERN.match(line.strip())) is not None
        ),
        limits.max_items,
    )
    parts = ["result_type: pytest", f"exit_code: {_exit_code(result, text)}"]
    _append_section(parts, "failed_tests", failed_tests)
    _append_section(parts, "exceptions", exceptions)
    _append_section(parts, "stack_frames", stack_frames)
    if not failed_tests and not exceptions and not stack_frames:
        parts.extend(["output:", *_head_tail_lines(lines, limits.max_items)])
    return "\n".join(parts)


def _git_diff_summary(text: str, result: CallToolResult, limits: ReducerLimits) -> str:
    lines = text.splitlines()
    changed_files = _unique_lines(
        (match.group(2) for line in lines if (match := _DIFF_FILE_PATTERN.match(line)) is not None),
        limits.max_items,
    )
    additions = sum(line.startswith("+") and not line.startswith("+++") for line in lines)
    deletions = sum(line.startswith("-") and not line.startswith("---") for line in lines)
    hunks = _bounded_hunks(lines, limits)
    parts = [
        "result_type: git_diff",
        f"exit_code: {_exit_code(result, text)}",
        f"added_lines: {additions}",
        f"deleted_lines: {deletions}",
    ]
    _append_section(parts, "changed_files", changed_files)
    if hunks:
        parts.extend(["bounded_hunks:", *hunks])
    return "\n".join(parts)


def _shell_fallback_summary(
    text: str,
    result: CallToolResult,
    limits: ReducerLimits,
) -> str:
    lines = text.splitlines()
    retained = _head_tail_lines(lines, limits.max_items)
    retained_text = "\n".join(retained)
    truncated_bytes = max(
        len(text.encode("utf-8")) - len(retained_text.encode("utf-8")),
        0,
    )
    parts = [
        "result_type: shell",
        f"exit_code: {_exit_code(result, text)}",
        f"truncated_bytes: {truncated_bytes}",
        "output_head_tail:",
        *retained,
    ]
    return "\n".join(parts)


def _reduced_result(
    result: CallToolResult,
    summary: str,
    *,
    artifact_id: ArtifactId,
    max_result_bytes: int,
) -> CallToolResult:
    reference = f"output_artifact: {artifact_id}"
    text = _fit_with_reference(summary, reference, max_result_bytes)
    return result.model_copy(
        update={
            "content": [TextContent(type="text", text=text)],
            "structured_content": None,
        }
    )


def _fit_with_reference(summary: str, reference: str, max_bytes: int) -> str:
    separator = "\n"
    reserved = len((separator + reference).encode("utf-8"))
    available = max(max_bytes - reserved, 0)
    body = _truncate_utf8(summary, available)
    return f"{body}{separator if body else ''}{reference}"


def _truncate_utf8(text: str, max_bytes: int) -> str:
    payload = text.encode("utf-8")
    if len(payload) <= max_bytes:
        return text
    marker = "\n[summary truncated]"
    marker_bytes = marker.encode("utf-8")
    available = max(max_bytes - len(marker_bytes), 0)
    prefix = payload[:available].decode("utf-8", errors="ignore")
    return f"{prefix}{marker}" if available else ""


def _result_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _exit_code(result: CallToolResult, text: str) -> int | str:
    value = tool_result_display_metadata(result).get("exit_code")
    if type(value) is int:
        return value
    match = _EXIT_CODE_PATTERN.search(text)
    return int(match.group(1)) if match is not None else "unknown"


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _is_pytest_command(tokens: list[str]) -> bool:
    return any(token == "pytest" or token.endswith("/pytest") for token in tokens)


def _is_git_diff_command(tokens: list[str]) -> bool:
    return any(
        left == "git" and right == "diff" for left, right in zip(tokens, tokens[1:], strict=False)
    )


def _unique_matches(
    lines: list[str],
    pattern: re.Pattern[str],
    group: str,
    limit: int,
) -> list[str]:
    return _unique_lines(
        (
            match.group(group)
            for line in lines
            if (match := pattern.match(line.strip())) is not None
        ),
        limit,
    )


def _unique_lines(lines: Iterable[str], limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in lines:
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
        if len(values) == limit:
            break
    return values


def _append_section(parts: list[str], title: str, values: list[str]) -> None:
    if values:
        parts.append(f"{title}:")
        parts.extend(f"- {value}" for value in values)


def _head_tail_lines(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    head_count = (limit + 1) // 2
    tail_count = limit - head_count
    tail = lines[-tail_count:] if tail_count else []
    return [*lines[:head_count], "...", *tail]


def _bounded_hunks(lines: list[str], limits: ReducerLimits) -> list[str]:
    hunks: list[str] = []
    current_lines = 0
    current_hunks = 0
    for line in lines:
        if line.startswith("@@"):
            if current_hunks == limits.max_hunks:
                break
            current_hunks += 1
            current_lines = 0
        if current_hunks == 0 or current_lines == limits.max_hunk_lines:
            continue
        hunks.append(line)
        current_lines += 1
    return hunks
