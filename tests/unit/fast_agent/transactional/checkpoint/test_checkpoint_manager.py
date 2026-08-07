from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.checkpoint._git import (
    UnsupportedWorkspaceError,
    WorkspaceError,
)
from fast_agent.transactional.checkpoint.checkpoint import (
    CheckpointManager,
    WorkspaceDivergenceError,
    WorkspaceFileKind,
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
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "baseline")
    return root


def _worktree(tmp_path: Path) -> tuple[Path, CheckpointManager]:
    repository = _repository(tmp_path)
    worktree = WorktreeManager(repository, tmp_path / "worktrees").create(RunId("run-1"))
    return worktree.worktree_path, CheckpointManager(worktree, tmp_path / "checkpoints")


def test_checkpoint_restores_tracked_untracked_delete_and_rename(tmp_path: Path) -> None:
    root, manager = _worktree(tmp_path)
    (root / "tracked.txt").write_text("checkpoint version\n", encoding="utf-8")
    (root / "untracked.txt").write_text("keep me\n", encoding="utf-8")
    (root / "rename-source.txt").write_text("rename me\n", encoding="utf-8")
    _git(root, "add", "rename-source.txt")
    _git(root, "commit", "-m", "add rename source")
    checkpoint = manager.create()

    (root / "tracked.txt").unlink()
    (root / "untracked.txt").write_text("changed\n", encoding="utf-8")
    (root / "rename-source.txt").rename(root / "rename-target.txt")
    (root / "new.txt").write_text("remove me\n", encoding="utf-8")
    _git(root, "add", "new.txt")
    assert manager.current_version() != checkpoint.workspace_version

    restored = manager.restore(checkpoint)

    assert restored == checkpoint.workspace_version
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "checkpoint version\n"
    assert (root / "untracked.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (root / "rename-source.txt").read_text(encoding="utf-8") == "rename me\n"
    assert not (root / "rename-target.txt").exists()
    assert not (root / "new.txt").exists()


def test_checkpoint_records_manifest_and_stable_workspace_version(tmp_path: Path) -> None:
    root, manager = _worktree(tmp_path)
    (root / "tracked.txt").unlink()
    (root / "tracked-added.txt").write_text("tracked addition\n", encoding="utf-8")
    (root / "untracked.txt").write_text("untracked addition\n", encoding="utf-8")
    _git(root, "add", "tracked-added.txt")

    before = manager.current_version()
    checkpoint = manager.create()
    entries = {entry.path: entry for entry in checkpoint.files}
    manifest_path = tmp_path / "checkpoints" / checkpoint.checkpoint_id / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest: object = json.loads(manifest_text)

    assert checkpoint.workspace_version == before == manager.current_version()
    assert entries["tracked.txt"].kind is WorkspaceFileKind.TRACKED
    assert entries["tracked.txt"].exists is False
    assert entries["tracked-added.txt"].kind is WorkspaceFileKind.TRACKED
    assert entries["untracked.txt"].kind is WorkspaceFileKind.UNTRACKED
    assert isinstance(manifest, dict)
    assert f'"workspace_version":"{checkpoint.workspace_version}"' in manifest_text


def test_restore_preserves_ignored_and_outside_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    worktree = WorktreeManager(repository, tmp_path / "worktrees").create(RunId("run-1"))
    root = worktree.worktree_path
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-m", "ignore runtime file")
    ignored = root / "ignored.txt"
    outside = tmp_path / "outside.txt"
    ignored.write_text("before\n", encoding="utf-8")
    outside.write_text("before\n", encoding="utf-8")
    manager = CheckpointManager(worktree, tmp_path / "checkpoints")
    checkpoint = manager.create()

    ignored.write_text("after\n", encoding="utf-8")
    outside.write_text("after\n", encoding="utf-8")
    (root / "tracked.txt").write_text("bad change\n", encoding="utf-8")
    manager.restore(checkpoint)

    assert ignored.read_text(encoding="utf-8") == "after\n"
    assert outside.read_text(encoding="utf-8") == "after\n"
    assert manager.current_version() == checkpoint.workspace_version


def test_restore_detects_corrupted_checkpoint_content(tmp_path: Path) -> None:
    _, manager = _worktree(tmp_path)
    checkpoint = manager.create()
    snapshot = tmp_path / "checkpoints" / checkpoint.checkpoint_id / "files" / "tracked.txt"
    snapshot.write_text("corrupted\n", encoding="utf-8")

    with pytest.raises(WorkspaceDivergenceError, match="does not match"):
        manager.restore(checkpoint)


def test_checkpoint_rejects_nested_git_repository(tmp_path: Path) -> None:
    root, manager = _worktree(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    _git(nested, "init")

    with pytest.raises(UnsupportedWorkspaceError, match="Nested Git repositories"):
        manager.create()


def test_checkpoint_root_cannot_pollute_run_worktree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    worktree = WorktreeManager(repository, tmp_path / "worktrees").create(RunId("run-1"))

    with pytest.raises(WorkspaceError, match="outside source and Run worktrees"):
        CheckpointManager(worktree, worktree.worktree_path / ".checkpoints")
