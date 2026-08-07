from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.checkpoint._git import WorkspaceError
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


def test_worktree_is_detached_isolated_and_explicitly_cleaned(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "tracked.txt").write_text("user change\n", encoding="utf-8")
    (repository / "local.txt").write_text("local only\n", encoding="utf-8")
    source_status = _git(repository, "status", "--short")
    manager = WorktreeManager(repository, tmp_path / "worktrees")

    worktree = manager.create(RunId("run-1"))
    (worktree.worktree_path / "tracked.txt").write_text("run change\n", encoding="utf-8")

    assert worktree.repository_root == repository
    assert worktree.baseline_commit == _git(repository, "rev-parse", "HEAD")
    assert _git(worktree.worktree_path, "branch", "--show-current") == ""
    assert (repository / "tracked.txt").read_text(encoding="utf-8") == "user change\n"
    assert _git(repository, "status", "--short") == source_status
    assert worktree.worktree_path.exists()

    manager.cleanup(worktree)
    manager.cleanup(worktree)
    assert not worktree.worktree_path.exists()
    assert (repository / "tracked.txt").read_text(encoding="utf-8") == "user change\n"
    assert _git(repository, "status", "--short") == source_status


def test_runs_use_distinct_worktrees(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")

    first = manager.create(RunId("run-1"))
    second = manager.create(RunId("run-2"))

    assert first.worktree_path != second.worktree_path
    assert first.baseline_commit == second.baseline_commit


def test_cleanup_rejects_a_path_outside_the_managed_root(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = WorktreeManager(repository, tmp_path / "worktrees")
    worktree = manager.create(RunId("run-1"))
    escaped = type(worktree)(
        repository_root=worktree.repository_root,
        baseline_commit=worktree.baseline_commit,
        worktree_path=tmp_path / "outside",
        run_id=worktree.run_id,
    )

    with pytest.raises(WorkspaceError, match="escapes the managed root"):
        manager.cleanup(escaped)


def test_worktree_root_cannot_pollute_source_repository(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(WorkspaceError, match="outside the source repository"):
        WorktreeManager(repository, repository / ".txagent-worktrees")
