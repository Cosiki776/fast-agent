from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.assembly import TransactionalRuntimeAssembler
from fast_agent.transactional.checkpoint.worktree import WorktreeManager
from fast_agent.transactional.execution import ToolExecutionOutcome, ToolExecutionRequest
from fast_agent.transactional.models import RunId, ToolCallId
from fast_agent.transactional.run_events import RunEventKind
from fast_agent.transactional.settings import TransactionalProfile, TransactionalSettings

if TYPE_CHECKING:
    from pathlib import Path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "TxAgent Test")
    _git(root, "config", "user.email", "txagent@example.test")
    (root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "baseline")
    return root


def test_baseline_assembles_no_transactional_components(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    assembler = TransactionalRuntimeAssembler(TransactionalSettings(), runtime_root)

    assert assembler.assemble(run_id=RunId("baseline")) is None
    assert not runtime_root.exists()


def test_reducer_profile_omits_worktree_and_recovery(tmp_path: Path) -> None:
    runtime = TransactionalRuntimeAssembler(
        TransactionalSettings(profile=TransactionalProfile.REDUCER),
        tmp_path / "runtime",
    ).assemble(run_id=RunId("reducer"))

    assert runtime is not None
    try:
        assert runtime.profile is TransactionalProfile.REDUCER
        assert runtime.workspace is None
        assert runtime.controller is None
        assert runtime.run_event_store is None
    finally:
        runtime.close()


def test_full_profile_requires_run_worktree(tmp_path: Path) -> None:
    assembler = TransactionalRuntimeAssembler(
        TransactionalSettings(profile=TransactionalProfile.FULL),
        tmp_path / "runtime",
    )

    with pytest.raises(ValueError, match="requires a Run worktree"):
        assembler.assemble(run_id=RunId("full"))


@pytest.mark.asyncio
async def test_full_profile_composes_real_checkpoint_recovery(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    run_id = RunId("full")
    worktree_manager = WorktreeManager(repository, tmp_path / "worktrees")
    worktree = worktree_manager.create(run_id)
    runtime = TransactionalRuntimeAssembler(
        TransactionalSettings(profile=TransactionalProfile.FULL),
        tmp_path / "runtime",
    ).assemble(run_id=run_id, worktree=worktree)

    assert runtime is not None
    assert runtime.workspace is not None
    workspace = runtime.workspace
    try:
        transaction = 0

        async def fail_after_write() -> ToolExecutionOutcome:
            nonlocal transaction
            transaction += 1
            workspace.joinpath("tracked.txt").write_text(
                f"bad attempt {transaction}\n", encoding="utf-8"
            )
            return ToolExecutionOutcome(
                result=CallToolResult(
                    content=[TextContent(type="text", text="same assertion failed")],
                    structured_content={"status": "failed", "message": "same assertion failed"},
                    is_error=True,
                )
            )

        for call_id in ("call-1", "call-2"):
            outcome = await runtime.coordinator.coordinate(
                ToolExecutionRequest(
                    run_id=runtime.run_id,
                    tool_call_id=ToolCallId(call_id),
                    tool_name="write_text_file",
                    arguments={"path": "tracked.txt", "content": "bad"},
                ),
                fail_after_write,
            )

        assert workspace.joinpath("tracked.txt").read_text(encoding="utf-8") == "baseline\n"
        assert outcome.result.structured_content is not None
        assert outcome.result.structured_content["status"] == "recovery_handoff"
        assert runtime.run_event_store is not None
        assert [
            item.event.kind for item in runtime.run_event_store.events_for_run(runtime.run_id)
        ] == [
            RunEventKind.STARTED,
            RunEventKind.RECOVERY_STARTED,
            RunEventKind.RECOVERED,
        ]
    finally:
        runtime.close()
        worktree_manager.cleanup(worktree)


@pytest.mark.asyncio
async def test_full_profile_policy_denies_path_escape_before_execution(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    run_id = RunId("full-policy")
    worktree_manager = WorktreeManager(repository, tmp_path / "worktrees")
    worktree = worktree_manager.create(run_id)
    runtime = TransactionalRuntimeAssembler(
        TransactionalSettings(profile=TransactionalProfile.FULL),
        tmp_path / "runtime",
    ).assemble(run_id=run_id, worktree=worktree)

    assert runtime is not None
    executions = 0

    async def execute() -> ToolExecutionOutcome:
        nonlocal executions
        executions += 1
        raise AssertionError("policy denial must not execute the tool")

    try:
        outcome = await runtime.coordinator.coordinate(
            ToolExecutionRequest(
                run_id=run_id,
                tool_call_id=ToolCallId("escape"),
                tool_name="write_text_file",
                arguments={"path": "../outside.txt", "content": "unsafe"},
            ),
            execute,
        )

        assert executions == 0
        assert outcome.result.structured_content == {
            "status": "denied",
            "reason": "path escapes the Run worktree: ../outside.txt",
        }
    finally:
        runtime.close()
        worktree_manager.cleanup(worktree)
