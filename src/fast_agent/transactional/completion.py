from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.transactional.checkpoint.snapshot import AgentDelta
    from fast_agent.transactional.models import RunId

_MAX_LISTED_PATHS = 20


@dataclass(frozen=True, slots=True)
class WorktreeOnlyCompletionReport:
    """Deterministic handoff for a Full Run without completion verification."""

    run_id: RunId
    worktree_path: Path
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]

    @classmethod
    def from_delta(
        cls,
        run_id: RunId,
        worktree_path: Path,
        delta: AgentDelta,
    ) -> "WorktreeOnlyCompletionReport":
        return cls(
            run_id=run_id,
            worktree_path=worktree_path,
            added=delta.added,
            modified=delta.modified,
            deleted=delta.deleted,
        )

    @property
    def changed_file_count(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def render_text(self) -> str:
        lines = [
            "TxAgent result retained for manual review",
            f"Run: {self.run_id}",
            "Verification: not configured",
            "Promotion: not applied",
            (
                "Changes: "
                f"{len(self.added)} added, {len(self.modified)} modified, "
                f"{len(self.deleted)} deleted"
            ),
        ]
        _append_paths(lines, "Added", self.added)
        _append_paths(lines, "Modified", self.modified)
        _append_paths(lines, "Deleted", self.deleted)
        lines.extend(worktree_review_lines(self.worktree_path))
        lines.append("Configure transactional.verification.command to enable verified promotion.")
        return "\n".join(lines)


def worktree_review_lines(worktree_path: Path) -> tuple[str, ...]:
    """Return the shared manual-review locator and commands for a retained Worktree."""
    quoted_worktree = shlex.quote(str(worktree_path))
    return (
        f"Agent result retained at: {worktree_path}",
        "Review commands:",
        f"  git -C {quoted_worktree} status --short",
        f"  git -C {quoted_worktree} diff --no-ext-diff",
    )


def _append_paths(lines: list[str], label: str, paths: tuple[str, ...]) -> None:
    if not paths:
        return
    lines.append(f"{label}:")
    lines.extend(f"  - {path}" for path in paths[:_MAX_LISTED_PATHS])
    omitted = len(paths) - _MAX_LISTED_PATHS
    if omitted > 0:
        lines.append(f"  - ... and {omitted} more")
