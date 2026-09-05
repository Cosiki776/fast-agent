from __future__ import annotations

import base64
import json
import os
import shlex
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.transactional.checkpoint._git import WorkspaceError
from fast_agent.transactional.checkpoint.snapshot import WorkspaceChangedError
from fast_agent.transactional.storage.artifact_store import ArtifactId, ArtifactKind

if TYPE_CHECKING:
    from fast_agent.transactional.checkpoint.snapshot import (
        AgentDelta,
        WorkspaceSnapshot,
        WorkspaceSnapshotManager,
    )
    from fast_agent.transactional.checkpoint.worktree import WorktreeMetadata
    from fast_agent.transactional.storage.artifact_store import FileArtifactStore


@dataclass(frozen=True, slots=True)
class PromotionResult:
    workspace_version: str
    patch_artifact_id: ArtifactId


class PromotionRejectedError(WorkspaceChangedError):
    def __init__(
        self,
        reason: str,
        patch_artifact_id: ArtifactId,
        worktree_path: Path,
    ) -> None:
        self.reason = reason
        self.patch_artifact_id = patch_artifact_id
        self.worktree_path = worktree_path
        quoted_worktree = shlex.quote(str(worktree_path))
        super().__init__(
            "\n".join(
                (
                    f"Promotion rejected: {reason}",
                    f"Agent result retained at: {worktree_path}",
                    f"Patch artifact: {patch_artifact_id}",
                    "Review commands:",
                    f"  git -C {quoted_worktree} status --short",
                    f"  git -C {quoted_worktree} diff --no-ext-diff",
                )
            )
        )


class WorkspacePromoter:
    """Apply a verified Agent Delta only while the source fingerprint is unchanged."""

    def __init__(
        self,
        snapshots: WorkspaceSnapshotManager,
        snapshot: WorkspaceSnapshot,
        worktree: WorktreeMetadata,
        artifacts: FileArtifactStore,
    ) -> None:
        self._snapshots = snapshots
        self._snapshot = snapshot
        self._worktree = worktree
        self._artifacts = artifacts

    def promote(self, verified_version: str) -> PromotionResult:
        current_agent_version = str(
            self._snapshots.worktree_version(self._snapshot, self._worktree)
        )
        if current_agent_version != verified_version:
            raise WorkspaceError("Agent workspace changed after completion verification")

        delta = self._snapshots.agent_delta(self._snapshot, self._worktree)
        patch = self._artifacts.put(
            _encode_patch(self._worktree.worktree_path, delta, verified_version),
            media_type="application/json",
            kind=ArtifactKind.WORKSPACE_PATCH,
        )
        try:
            self._snapshots.assert_source_unchanged(self._snapshot)
        except WorkspaceChangedError as exc:
            raise PromotionRejectedError(
                str(exc),
                patch.artifact_id,
                self._worktree.worktree_path,
            ) from exc

        try:
            self._apply(delta)
            promoted_version = str(self._snapshots.source_version(self._snapshot))
            if promoted_version != verified_version:
                raise WorkspaceError(
                    "Promoted workspace does not match the verified Agent workspace"
                )
        except Exception:
            self._restore(delta)
            raise
        return PromotionResult(promoted_version, patch.artifact_id)

    def _apply(self, delta: AgentDelta) -> None:
        source = self._snapshot.source_root
        agent = self._worktree.worktree_path
        for relative_path in sorted(delta.deleted, key=_path_depth, reverse=True):
            path = _workspace_path(source, relative_path)
            _remove_entry(path)
            _remove_empty_parents(path.parent, source)
        for relative_path in sorted((*delta.added, *delta.modified), key=_path_depth):
            _replace_entry(
                _workspace_path(agent, relative_path),
                _workspace_path(source, relative_path),
            )

    def _restore(self, delta: AgentDelta) -> None:
        source = self._snapshot.source_root
        changed = (*delta.added, *delta.modified, *delta.deleted)
        for relative_path in sorted(changed, key=_path_depth, reverse=True):
            path = _workspace_path(source, relative_path)
            _remove_entry(path)
            _remove_empty_parents(path.parent, source)
        initial = {entry.path: entry for entry in self._snapshot.files}
        for relative_path in sorted(changed, key=_path_depth):
            entry = initial.get(relative_path)
            if entry is not None and entry.exists:
                _replace_entry(
                    self._snapshots.stored_entry(self._snapshot, relative_path),
                    _workspace_path(source, relative_path),
                )


def _encode_patch(root: Path, delta: AgentDelta, verified_version: str) -> bytes:
    entries: list[dict[str, object]] = []
    for relative_path in (*delta.added, *delta.modified):
        path = _workspace_path(root, relative_path)
        if path.is_symlink():
            entries.append(
                {
                    "path": relative_path,
                    "symlink_target": os.readlink(path),
                }
            )
        else:
            entries.append(
                {
                    "path": relative_path,
                    "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "executable": bool(path.stat().st_mode & 0o111),
                }
            )
    payload = {
        "delta": asdict(delta),
        "entries": entries,
        "verified_workspace_version": verified_version,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _workspace_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise WorkspaceError(f"Workspace path escapes its root: {relative_path}")
    destination = root / candidate
    try:
        destination.parent.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise WorkspaceError(f"Workspace path escapes its root: {relative_path}") from exc
    return destination


def _replace_entry(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        if source.is_symlink():
            temporary = destination.parent / f".{destination.name}.promotion"
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(os.readlink(source))
        else:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        raise WorkspaceError(f"Workspace entry is not a file or symlink: {path}")
    path.unlink(missing_ok=True)


def _remove_empty_parents(directory: Path, root: Path) -> None:
    while directory != root and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


def _path_depth(path: str) -> int:
    return len(Path(path).parts)
