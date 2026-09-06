from __future__ import annotations

import hashlib
import json
import re
import shlex
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fast_agent.constants import MAX_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.tools.apply_patch_tool import APPLY_PATCH_INPUT_FIELD, APPLY_PATCH_TOOL_NAME
from fast_agent.transactional.checkpoint._git import is_within
from fast_agent.transactional.checkpoint.snapshot import is_safe_snapshot_path
from fast_agent.transactional.models import ToolEffect
from fast_agent.utils.tool_names import is_shell_command_tool_name, matches_tool_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler
    from fast_agent.transactional.execution import ToolExecutionRequest


POLICY_VERSION = "coding-v1"

_PATH_ARGUMENTS = ("path", "cwd", "working_directory")
# This exact find predicate excludes Git entries; its pattern is not a path access.
# Keep the exception narrow: extra find expressions/actions still use the normal gate.
_FIND_GIT_EXCLUSION = re.compile(
    r"\A(find[ \t]+\.[ \t]+-type[ \t]+f[ \t]+(?:-not|!)[ \t]+-path[ \t]+)"
    r"(['\"])\./\.git/\*\2(?=[ \t]*(?:$|[;&|]))"
)
_PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Delete|Update) File: (.+)$", re.MULTILINE)
_PATCH_MOVE_PATH = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)
_SHELL_PRIVILEGE_ESCALATION = re.compile(
    r"(?:^|[;&|]\s*)(?:\S*/)?(?:sudo|doas|pkexec|su)(?:\s|$)",
    re.IGNORECASE,
)
_SHELL_HARD_DENIES = (
    re.compile(
        r"(?:^|[;&|]\s*)git\s+push\b[^;&|]*"
        r"(?:"
        r"\s(?:-f|--force(?:-with-lease|-if-includes)?|--mirror|--delete|--prune)(?:[=\s]|$)|"
        r"\s[+:]\S+"
        r")",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[;&|]\s*)git\s+update-ref(?:\s|$)", re.IGNORECASE),
    re.compile(
        r"(?:^|[;&|]\s*)git\s+remote\s+(?:add|remove|rename|set-url)(?:\s|$)", re.IGNORECASE
    ),
    re.compile(
        r"(?:^|[;&|]\s*)(?:terraform\s+destroy|kubectl\s+delete)(?:\s|$)",
        re.IGNORECASE,
    ),
)
_SHELL_APPROVAL_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:"
    r"git\s+push|"
    r"gh\s+pr\s+(?:create|close|merge|reopen)|"
    r"(?:npm|pnpm|yarn|cargo|docker)\s+(?:publish|push)|"
    r"twine\s+upload|terraform\s+apply|kubectl\s+apply|"
    r"curl|wget|ssh|scp|rsync|nc|ncat|sftp|ftp"
    r")(?:\s|$)",
    re.IGNORECASE,
)


class GovernanceDisposition(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    disposition: GovernanceDisposition
    reason: str

    @classmethod
    def allow(cls) -> GovernanceDecision:
        return cls(GovernanceDisposition.ALLOW, "coding policy allowed the tool")

    @classmethod
    def deny(cls, reason: str) -> GovernanceDecision:
        return cls(GovernanceDisposition.DENY, reason)

    @classmethod
    def require_approval(cls, reason: str) -> GovernanceDecision:
        return cls(GovernanceDisposition.REQUIRE_APPROVAL, reason)


class ToolGovernanceGate(Protocol):
    async def evaluate(
        self,
        request: ToolExecutionRequest,
        effect: ToolEffect,
        /,
    ) -> GovernanceDecision: ...


class CodingToolPolicy:
    """Best-effort application policy for one worktree, not an OS security sandbox."""

    version = POLICY_VERSION

    def __init__(
        self,
        workspace: Path,
        transactional_root: Path,
        *,
        max_shell_timeout_seconds: float,
        max_shell_output_bytes: int = MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    ) -> None:
        self._workspace = workspace.resolve()
        self._transactional_root = transactional_root.resolve()
        self._max_shell_timeout_seconds = max_shell_timeout_seconds
        self._max_shell_output_bytes = max_shell_output_bytes

    def evaluate(
        self,
        request: ToolExecutionRequest,
        effect: ToolEffect,
        /,
    ) -> GovernanceDecision:
        path_denial = self._path_denial(request)
        if path_denial is not None:
            return GovernanceDecision.deny(path_denial)

        if is_shell_command_tool_name(request.tool_name):
            shell_decision = self._shell_decision(request)
            if shell_decision is not None:
                return shell_decision

        if effect is ToolEffect.EXTERNAL_UNKNOWN:
            return GovernanceDecision.require_approval(
                f"tool effect requires approval: {request.tool_name}"
            )
        return GovernanceDecision.allow()

    def _path_denial(self, request: ToolExecutionRequest) -> str | None:
        paths: list[str] = []
        for key in _PATH_ARGUMENTS:
            if key not in request.arguments:
                continue
            value = request.arguments[key]
            if not isinstance(value, str) or not value.strip():
                return f"{key} must be a non-empty path"
            paths.append(value)

        if matches_tool_name(request.tool_name, APPLY_PATCH_TOOL_NAME):
            patch = request.arguments.get(APPLY_PATCH_INPUT_FIELD)
            if not isinstance(patch, str) or not patch.strip():
                return "apply_patch input must be non-empty"
            paths.extend(_PATCH_PATH.findall(patch))
            paths.extend(_PATCH_MOVE_PATH.findall(patch))

        for raw_path in paths:
            denial = self._validate_path(raw_path)
            if denial is not None:
                return denial
        return None

    def _validate_path(self, raw_path: str) -> str | None:
        candidate = Path(raw_path).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self._workspace / candidate).resolve()
        )
        if not is_within(resolved, self._workspace):
            return f"path escapes the Run worktree: {raw_path}"
        if is_within(resolved, self._transactional_root):
            return f"path targets transactional runtime data: {raw_path}"

        relative = resolved.relative_to(self._workspace).as_posix()
        parts = Path(relative).parts
        if ".git" in parts:
            return f"path targets protected Git metadata: {raw_path}"
        if not is_safe_snapshot_path(relative):
            return f"path targets credentials or secrets: {raw_path}"
        return None

    def _shell_decision(self, request: ToolExecutionRequest) -> GovernanceDecision | None:
        command = request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return GovernanceDecision.deny("shell command must be non-empty")

        timeout = request.arguments.get("timeout", request.arguments.get("hard_timeout_seconds"))
        if timeout is not None and (
            not isinstance(timeout, int | float)
            or isinstance(timeout, bool)
            or timeout <= 0
            or timeout > self._max_shell_timeout_seconds
        ):
            return GovernanceDecision.deny("shell timeout exceeds the coding policy limit")

        output_limit = request.arguments.get("output_byte_limit")
        if output_limit is not None and (
            not isinstance(output_limit, int)
            or isinstance(output_limit, bool)
            or output_limit <= 0
            or output_limit > self._max_shell_output_bytes
        ):
            return GovernanceDecision.deny("shell output limit exceeds the coding policy limit")

        normalized = " ".join(command.split())
        if _SHELL_PRIVILEGE_ESCALATION.search(normalized):
            return GovernanceDecision.deny("privilege escalation command is forbidden")
        if any(pattern.search(normalized) for pattern in _SHELL_HARD_DENIES):
            return GovernanceDecision.deny("destructive remote command is forbidden")
        if self._contains_protected_shell_path(command):
            return GovernanceDecision.deny("shell command targets a protected path")
        if _SHELL_APPROVAL_COMMAND.search(normalized):
            return GovernanceDecision.require_approval(
                "shell command may produce external network side effects"
            )
        return None

    def _contains_protected_shell_path(self, command: str) -> bool:
        command = _FIND_GIT_EXCLUSION.sub(r"\1'__excluded_git_entries__'", command)
        normalized = command.replace("\\", "/")
        if "../" in normalized or "/.git/" in normalized or " .git/" in normalized:
            return True
        if str(self._transactional_root) in command:
            return True
        if any(
            marker in normalized.casefold()
            for marker in ("/.env", "/.ssh/", "/.aws/", "/.gnupg/", "credentials.json")
        ):
            return True

        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError:
            return True

        command_start = True
        skip_payload = False
        for token in tokens:
            if token in {";", "&&", "||", "|", "&"}:
                command_start = True
                skip_payload = False
                continue
            if command_start:
                command_start = False
                continue
            if skip_payload:
                skip_payload = False
                continue
            if token in {"-c", "-Command"}:
                skip_payload = True
                continue
            if token.startswith("-") or "://" in token:
                continue
            if Path(token).is_absolute() and self._validate_path(token) is not None:
                return True
        return False


class CodingToolGovernanceGate:
    """Apply coding policy and request one context-bound approval when required."""

    def __init__(
        self,
        policy: CodingToolPolicy,
        *,
        current_workspace_version: Callable[[], str] | None = None,
        permission_handler: ToolPermissionHandler | None = None,
    ) -> None:
        self._policy = policy
        self._current_workspace_version = current_workspace_version
        self._permission_handler = permission_handler

    async def evaluate(
        self,
        request: ToolExecutionRequest,
        effect: ToolEffect,
        /,
    ) -> GovernanceDecision:
        decision = self._policy.evaluate(request, effect)
        if decision.disposition is not GovernanceDisposition.REQUIRE_APPROVAL:
            return decision
        if self._permission_handler is None or self._current_workspace_version is None:
            return GovernanceDecision.deny(decision.reason)

        try:
            binding = self._approval_binding(request)
            permission = await self._permission_handler.check_permission(
                tool_name=request.tool_name,
                server_name=request.server_name or "local",
                arguments=deepcopy(request.arguments),
                tool_use_id=str(request.tool_call_id),
            )
        except Exception as exc:
            return GovernanceDecision.deny(f"approval request failed: {exc}")

        if not permission.allowed:
            return GovernanceDecision.deny(
                permission.error_message or f"approval denied for tool: {request.tool_name}"
            )

        revalidated = self._policy.evaluate(request, effect)
        if revalidated.disposition is not GovernanceDisposition.REQUIRE_APPROVAL:
            return GovernanceDecision.deny("approval context changed during policy revalidation")
        if self._approval_binding(request) != binding:
            return GovernanceDecision.deny("approval context changed before tool execution")
        return GovernanceDecision.allow()

    def _approval_binding(self, request: ToolExecutionRequest) -> ApprovalBinding:
        current_version = self._current_workspace_version
        if current_version is None:
            raise RuntimeError("approval requires a workspace version provider")
        return ApprovalBinding(
            tool_name=request.tool_name,
            arguments_sha256=normalized_arguments_sha256(request),
            workspace_version=current_version(),
            policy_version=self._policy.version,
        )


@dataclass(frozen=True, slots=True)
class ApprovalBinding:
    tool_name: str
    arguments_sha256: str
    workspace_version: str
    policy_version: str


def normalized_arguments_sha256(request: ToolExecutionRequest) -> str:
    payload = json.dumps(
        request.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
