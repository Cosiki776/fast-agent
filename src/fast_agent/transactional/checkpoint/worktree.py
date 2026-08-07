from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.transactional.checkpoint._git import (
    WorkspaceError,
    git,
    is_within,
    repository_root,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.transactional.models import RunId


@dataclass(frozen=True, slots=True)
class WorktreeMetadata:
    repository_root: Path
    baseline_commit: str
    worktree_path: Path
    run_id: RunId


class WorktreeManager:
    """Create isolated detached worktrees and remove them explicitly."""

    def __init__(self, repository: Path, worktrees_root: Path) -> None:
        self.repository_root = repository_root(repository)
        self.worktrees_root = worktrees_root.resolve()
        if is_within(self.worktrees_root, self.repository_root):
            raise WorkspaceError("Worktree root must be outside the source repository")
        self.worktrees_root.mkdir(parents=True, exist_ok=True)

    def create(self, run_id: RunId, *, baseline: str = "HEAD") -> WorktreeMetadata:
        baseline_commit = git(self.repository_root, "rev-parse", f"{baseline}^{{commit}}")
        path = self.worktrees_root / _run_directory_name(run_id)
        if path.exists():
            raise WorkspaceError(f"Run worktree already exists: {path}")

        git(self.repository_root, "worktree", "add", "--detach", str(path), baseline_commit)
        return WorktreeMetadata(
            repository_root=self.repository_root,
            baseline_commit=baseline_commit,
            worktree_path=path,
            run_id=run_id,
        )

    def cleanup(self, worktree: WorktreeMetadata) -> None:
        """Explicitly remove a managed worktree; repeated cleanup is harmless."""
        path = self._managed_path(worktree)
        if path.exists():
            git(self.repository_root, "worktree", "remove", "--force", str(path))
        else:
            git(self.repository_root, "worktree", "prune")

    def _managed_path(self, worktree: WorktreeMetadata) -> Path:
        if worktree.repository_root.resolve() != self.repository_root:
            raise WorkspaceError("Worktree belongs to a different repository")
        path = worktree.worktree_path.resolve()
        try:
            path.relative_to(self.worktrees_root)
        except ValueError as exc:
            raise WorkspaceError("Worktree path escapes the managed root") from exc
        if path != self.worktrees_root / _run_directory_name(worktree.run_id):
            raise WorkspaceError("Worktree path does not match its Run ID")
        return path


def _run_directory_name(run_id: RunId) -> str:
    digest = hashlib.sha256(str(run_id).encode()).hexdigest()[:20]
    return f"run-{digest}"
