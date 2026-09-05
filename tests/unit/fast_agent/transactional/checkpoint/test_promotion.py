from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.checkpoint._git import WorkspaceError
from fast_agent.transactional.checkpoint.promotion import (
    PromotionRejectedError,
    WorkspacePromoter,
)
from fast_agent.transactional.checkpoint.snapshot import WorkspaceSnapshotManager
from fast_agent.transactional.checkpoint.worktree import WorktreeManager
from fast_agent.transactional.models import RunId
from fast_agent.transactional.storage.artifact_store import FileArtifactStore

if TYPE_CHECKING:
    from pathlib import Path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _run(tmp_path: Path):
    source = tmp_path / "repository"
    source.mkdir()
    _git(source, "init", "--initial-branch=main")
    _git(source, "config", "user.name", "TxAgent Test")
    _git(source, "config", "user.email", "txagent@example.test")
    source.joinpath("modify.txt").write_text("before\n", encoding="utf-8")
    source.joinpath("delete.txt").write_text("delete\n", encoding="utf-8")
    _git(source, "add", "modify.txt", "delete.txt")
    _git(source, "commit", "-m", "baseline")
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")
    snapshot = snapshots.capture()
    worktrees = WorktreeManager(source, tmp_path / "worktrees")
    worktree = worktrees.create(RunId("run-1"), baseline=snapshot.base_commit)
    snapshots.materialize(snapshot, worktree)
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    return source, snapshots, snapshot, worktree, artifacts


def test_promotion_applies_exact_verified_agent_delta(tmp_path: Path) -> None:
    source, snapshots, snapshot, worktree, artifacts = _run(tmp_path)
    agent = worktree.worktree_path
    agent.joinpath("modify.txt").write_text("after\n", encoding="utf-8")
    agent.joinpath("delete.txt").unlink()
    agent.joinpath("nested").mkdir()
    added = agent / "nested" / "added.txt"
    added.write_text("added\n", encoding="utf-8")
    added.chmod(0o755)
    verified_version = str(snapshots.worktree_version(snapshot, worktree))

    result = WorkspacePromoter(
        snapshots,
        snapshot,
        worktree,
        artifacts,
    ).promote(verified_version)

    assert source.joinpath("modify.txt").read_text(encoding="utf-8") == "after\n"
    assert not source.joinpath("delete.txt").exists()
    assert source.joinpath("nested/added.txt").read_text(encoding="utf-8") == "added\n"
    assert source.joinpath("nested/added.txt").stat().st_mode & 0o111
    assert result.workspace_version == verified_version
    patch = json.loads(artifacts.read(result.patch_artifact_id))
    assert patch["delta"] == {
        "added": ["nested/added.txt"],
        "deleted": ["delete.txt"],
        "modified": ["modify.txt"],
    }


def test_promotion_rejects_changed_source_without_writing_agent_delta(tmp_path: Path) -> None:
    source, snapshots, snapshot, worktree, artifacts = _run(tmp_path)
    worktree.worktree_path.joinpath("modify.txt").write_text("agent\n", encoding="utf-8")
    source.joinpath("modify.txt").write_text("user\n", encoding="utf-8")
    verified_version = str(snapshots.worktree_version(snapshot, worktree))
    promoter = WorkspacePromoter(snapshots, snapshot, worktree, artifacts)

    with pytest.raises(PromotionRejectedError) as raised:
        promoter.promote(verified_version)

    assert source.joinpath("modify.txt").read_text(encoding="utf-8") == "user\n"
    assert worktree.worktree_path.joinpath("modify.txt").read_text(encoding="utf-8") == "agent\n"
    assert raised.value.worktree_path == worktree.worktree_path
    assert f"Agent result retained at: {worktree.worktree_path}" in str(raised.value)
    assert f"Patch artifact: {raised.value.patch_artifact_id}" in str(raised.value)
    assert "git -C" in str(raised.value)
    patch = json.loads(artifacts.read(raised.value.patch_artifact_id))
    assert patch["delta"]["modified"] == ["modify.txt"]


def test_promotion_restores_snapshot_if_post_apply_version_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshots, snapshot, worktree, artifacts = _run(tmp_path)
    worktree.worktree_path.joinpath("modify.txt").write_text("agent\n", encoding="utf-8")
    verified_version = str(snapshots.worktree_version(snapshot, worktree))
    monkeypatch.setattr(snapshots, "source_version", lambda snapshot: "unexpected-version")

    with pytest.raises(WorkspaceError, match="does not match"):
        WorkspacePromoter(snapshots, snapshot, worktree, artifacts).promote(verified_version)

    assert source.joinpath("modify.txt").read_text(encoding="utf-8") == "before\n"
