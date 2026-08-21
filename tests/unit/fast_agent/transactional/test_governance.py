from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.execution import ToolExecutionRequest
from fast_agent.transactional.governance import (
    CodingToolGovernanceGate,
    CodingToolPolicy,
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
