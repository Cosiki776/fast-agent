from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.mcp.tool_permission_handler import ToolPermissionResult
from fast_agent.transactional.execution import ToolExecutionRequest
from fast_agent.transactional.governance import (
    CodingToolGovernanceGate,
    CodingToolPolicy,
    GovernanceDecision,
    GovernanceDisposition,
)
from fast_agent.transactional.models import RunId, ToolCallId, ToolEffect

if TYPE_CHECKING:
    from pathlib import Path


def _request(tool_name: str, arguments: dict) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        run_id=RunId("run-1"),
        tool_call_id=ToolCallId("call-1"),
        tool_name=tool_name,
        arguments=arguments,
    )


def _policy(tmp_path: Path) -> tuple[CodingToolPolicy, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return (
        CodingToolPolicy(
            workspace,
            tmp_path / "runtime",
            max_shell_timeout_seconds=30,
        ),
        workspace,
    )


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", ".git/config", ".env", ".ssh/id_ed25519"],
)
def test_policy_hard_denies_protected_paths(tmp_path: Path, path: str) -> None:
    policy, _ = _policy(tmp_path)

    decision = policy.evaluate(
        _request("write_text_file", {"path": path, "content": "unsafe"}),
        ToolEffect.WORKSPACE_WRITE,
    )

    assert decision.disposition is GovernanceDisposition.DENY


def test_policy_allows_workspace_file_write(tmp_path: Path) -> None:
    policy, workspace = _policy(tmp_path)

    decision = policy.evaluate(
        _request(
            "write_text_file",
            {"path": str(workspace / "src" / "app.py"), "content": "safe"},
        ),
        ToolEffect.WORKSPACE_WRITE,
    )

    assert decision.disposition is GovernanceDisposition.ALLOW


def test_policy_denies_patch_escape(tmp_path: Path) -> None:
    policy, _ = _policy(tmp_path)
    patch = "*** Begin Patch\n*** Update File: ../outside.py\n@@\n-old\n+new\n*** End Patch"

    decision = policy.evaluate(
        _request("apply_patch", {"input": patch}),
        ToolEffect.WORKSPACE_WRITE,
    )

    assert decision.disposition is GovernanceDisposition.DENY


@pytest.mark.parametrize("negation", ["-not", "!"])
def test_policy_allows_find_excluding_git_metadata(tmp_path: Path, negation: str) -> None:
    policy, _ = _policy(tmp_path)
    command = (
        f'find . -type f {negation} -path "./.git/*" | head -50 '
        '&& echo "---" && ls -la order_service tests'
    )
    assert (
        policy.evaluate(
            _request("execute", {"command": command}), ToolEffect.WORKSPACE_WRITE
        ).disposition
        is GovernanceDisposition.ALLOW
    )


@pytest.mark.parametrize(
    "command",
    [
        'find . -type f -path "./.git/*"',
        'find .git -type f -not -path "./.git/*"',
        'find . -type f -not -path "./.git/*" -o -path "./.git/config"',
        'find . -type f -not -path "./.git/*" -exec cat .git/config \\;',
        'find . -type f -not -path "./.git/*" && cat .git/config',
        'find . -type f -not -path "./.git/*" > .git/config',
        'find . -type f -not -path "./.git/*" | cat /tmp/outside',
        'find . -type f -not -path "./.git/*"; cat ../outside',
    ],
)
def test_find_exclusion_does_not_bypass_protected_path_checks(tmp_path: Path, command: str) -> None:
    policy, _ = _policy(tmp_path)
    assert (
        policy.evaluate(
            _request("execute", {"command": command}), ToolEffect.WORKSPACE_WRITE
        ).disposition
        is GovernanceDisposition.DENY
    )


@pytest.mark.parametrize(
    "command",
    [
        "git push --force origin main",
        "git push --force-with-lease origin main",
        "git push --mirror origin",
        "git push --delete origin old-branch",
        "git push --prune origin",
        "git push origin +main:main",
        "git push origin :old-branch",
        "git update-ref refs/heads/main HEAD",
        "terraform destroy",
        "kubectl delete deployment api",
    ],
)
def test_policy_hard_denies_destructive_remote_commands(
    tmp_path: Path,
    command: str,
) -> None:
    policy, _ = _policy(tmp_path)

    decision = policy.evaluate(
        _request("execute", {"command": command}),
        ToolEffect.WORKSPACE_WRITE,
    )

    assert decision.disposition is GovernanceDisposition.DENY


@pytest.mark.parametrize(
    "command",
    ["sudo command", "/usr/bin/sudo command", "doas command", "pkexec command", "su -c command"],
)
def test_policy_hard_denies_explicit_privilege_escalation(
    tmp_path: Path,
    command: str,
) -> None:
    policy, _ = _policy(tmp_path)

    decision = policy.evaluate(
        _request("execute", {"command": command}),
        ToolEffect.WORKSPACE_WRITE,
    )

    assert decision == GovernanceDecision.deny("privilege escalation command is forbidden")


@pytest.mark.parametrize(
    "command",
    [
        "git push origin main",
        "gh pr create --title change",
        "gh pr merge 10",
        "npm publish",
        "docker push example/app:latest",
        "twine upload dist/package.whl",
        "terraform apply",
        "kubectl apply -f deployment.yaml",
        "curl https://example.test",
    ],
)
def test_policy_requires_approval_for_external_shell_command(
    tmp_path: Path,
    command: str,
) -> None:
    policy, _ = _policy(tmp_path)

    decision = policy.evaluate(
        _request("execute", {"command": command}),
        ToolEffect.WORKSPACE_WRITE,
    )

    assert decision.disposition is GovernanceDisposition.REQUIRE_APPROVAL


@pytest.mark.asyncio
async def test_gate_fails_closed_when_approval_is_unavailable(tmp_path: Path) -> None:
    policy, _ = _policy(tmp_path)
    gate = CodingToolGovernanceGate(policy)

    decision = await gate.evaluate(
        _request("remote_tool", {"target": "external"}),
        ToolEffect.EXTERNAL_UNKNOWN,
    )

    assert decision.disposition is GovernanceDisposition.DENY


class _PermissionHandler:
    def __init__(self, result: ToolPermissionResult) -> None:
        self.result = result
        self.calls = 0

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        del tool_name, server_name, arguments, tool_use_id
        self.calls += 1
        return self.result


@pytest.mark.asyncio
async def test_gate_allows_one_approved_unknown_tool(tmp_path: Path) -> None:
    policy, _ = _policy(tmp_path)
    handler = _PermissionHandler(ToolPermissionResult.allow())
    gate = CodingToolGovernanceGate(
        policy,
        current_workspace_version=lambda: "version-1",
        permission_handler=handler,
    )

    decision = await gate.evaluate(
        _request("remote_tool", {"target": "external"}),
        ToolEffect.EXTERNAL_UNKNOWN,
    )

    assert decision.disposition is GovernanceDisposition.ALLOW
    assert handler.calls == 1


@pytest.mark.asyncio
async def test_gate_denies_when_workspace_changes_during_approval(tmp_path: Path) -> None:
    policy, _ = _policy(tmp_path)
    versions = iter(("version-1", "version-2"))
    gate = CodingToolGovernanceGate(
        policy,
        current_workspace_version=lambda: next(versions),
        permission_handler=_PermissionHandler(ToolPermissionResult.allow()),
    )

    decision = await gate.evaluate(
        _request("remote_tool", {"target": "external"}),
        ToolEffect.EXTERNAL_UNKNOWN,
    )

    assert decision == GovernanceDecision.deny("approval context changed before tool execution")
