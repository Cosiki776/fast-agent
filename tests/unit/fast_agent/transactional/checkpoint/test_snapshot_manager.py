from __future__ import annotations

import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.checkpoint._git import WorkspaceError
from fast_agent.transactional.checkpoint.checkpoint import CheckpointManager
from fast_agent.transactional.checkpoint.snapshot import (
    WorkspaceChangedError,
    WorkspaceSnapshotManager,
)
from fast_agent.transactional.checkpoint.worktree import WorktreeManager
from fast_agent.transactional.models import RunId

if TYPE_CHECKING:
    from pathlib import Path


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "TxAgent Test")
    _git(root, "config", "user.email", "txagent@example.test")
    (root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    (root / "deleted.txt").write_text("delete later\n", encoding="utf-8")
    _git(root, "add", "tracked.txt", "deleted.txt")
    _git(root, "commit", "-m", "baseline")
    return root


def _dirty_source(tmp_path: Path) -> Path:
    root = _repository(tmp_path)
    (root / "tracked.txt").write_text("user change\n", encoding="utf-8")
    (root / "deleted.txt").unlink()
    (root / "staged.txt").write_text("staged addition\n", encoding="utf-8")
    _git(root, "add", "staged.txt", "deleted.txt")
    (root / "untracked.txt").write_text("local addition\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (root / "private.pem").write_text("private key\n", encoding="utf-8")
    return root


def test_snapshot_materializes_dirty_workspace_without_touching_source(tmp_path: Path) -> None:
    source = _dirty_source(tmp_path)
    source_status = _git(source, "status", "--short")
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")
    snapshot = snapshots.capture()
    worktree = WorktreeManager(source, tmp_path / "worktrees").create(
        RunId("run-1"),
        baseline=snapshot.base_commit,
    )

    snapshots.materialize(snapshot, worktree)
    root = worktree.worktree_path

    assert snapshot.selected_branch == "main"
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "user change\n"
    assert not (root / "deleted.txt").exists()
    assert (root / "staged.txt").read_text(encoding="utf-8") == "staged addition\n"
    assert (root / "untracked.txt").read_text(encoding="utf-8") == "local addition\n"
    assert not (root / ".env").exists()
    assert not (root / "private.pem").exists()
    assert set(snapshot.excluded_paths) == {".env", "private.pem"}
    assert _git(source, "status", "--short") == source_status


def test_initial_snapshot_can_be_checkpointed_and_restored(tmp_path: Path) -> None:
    source = _dirty_source(tmp_path)
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")
    snapshot = snapshots.capture()
    worktree = WorktreeManager(source, tmp_path / "worktrees").create(
        RunId("run-1"),
        baseline=snapshot.base_commit,
    )
    snapshots.materialize(snapshot, worktree)
    checkpoints = CheckpointManager(worktree, tmp_path / "checkpoints")
    checkpoint = checkpoints.create()

    (worktree.worktree_path / "tracked.txt").write_text("agent mistake\n", encoding="utf-8")
    checkpoints.restore(checkpoint)

    assert (worktree.worktree_path / "tracked.txt").read_text(encoding="utf-8") == "user change\n"
    assert checkpoints.current_version() == checkpoint.workspace_version


def test_agent_delta_excludes_changes_present_before_the_run(tmp_path: Path) -> None:
    source = _dirty_source(tmp_path)
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")
    snapshot = snapshots.capture()
    worktree = WorktreeManager(source, tmp_path / "worktrees").create(
        RunId("run-1"),
        baseline=snapshot.base_commit,
    )
    snapshots.materialize(snapshot, worktree)

    empty_delta = snapshots.agent_delta(snapshot, worktree)
    (worktree.worktree_path / "tracked.txt").write_text("agent change\n", encoding="utf-8")
    (worktree.worktree_path / "agent.txt").write_text("new result\n", encoding="utf-8")
    delta = snapshots.agent_delta(snapshot, worktree)

    assert empty_delta.added == empty_delta.modified == empty_delta.deleted == ()
    assert delta.added == ("agent.txt",)
    assert delta.modified == ("tracked.txt",)
    assert delta.deleted == ()
    assert "staged.txt" not in delta.added
    assert "untracked.txt" not in delta.added


def test_source_change_is_reported_without_blocking_agent_delta(tmp_path: Path) -> None:
    source = _dirty_source(tmp_path)
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")
    snapshot = snapshots.capture()
    worktree = WorktreeManager(source, tmp_path / "worktrees").create(
        RunId("run-1"),
        baseline=snapshot.base_commit,
    )
    snapshots.materialize(snapshot, worktree)
    (source / "tracked.txt").write_text("concurrent user change\n", encoding="utf-8")

    delta = snapshots.agent_delta(snapshot, worktree)

    assert delta.added == delta.modified == delta.deleted == ()
    with pytest.raises(WorkspaceChangedError, match="changed after snapshot"):
        snapshots.assert_source_unchanged(snapshot)


def test_snapshot_rejects_symlink_escape(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (source / "escape.txt").symlink_to(outside)
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")

    with pytest.raises(WorkspaceError, match="Symlink escapes"):
        snapshots.capture()


@pytest.mark.parametrize("escaped_path", ["../outside.txt", "/tmp/outside.txt"])
def test_materialize_rejects_path_escape(tmp_path: Path, escaped_path: str) -> None:
    source = _repository(tmp_path)
    snapshots = WorkspaceSnapshotManager(source, tmp_path / "snapshots")
    snapshot = snapshots.capture()
    worktree = WorktreeManager(source, tmp_path / "worktrees").create(
        RunId("run-1"),
        baseline=snapshot.base_commit,
    )
    escaped = replace(
        snapshot,
        files=(replace(snapshot.files[0], path=escaped_path), *snapshot.files[1:]),
    )

    with pytest.raises(WorkspaceError, match="escapes its root"):
        snapshots.materialize(escaped, worktree)
