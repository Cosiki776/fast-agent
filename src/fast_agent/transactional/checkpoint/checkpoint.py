from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NewType
from uuid import uuid4

from fast_agent.transactional.checkpoint._git import (
    UnsupportedWorkspaceError,
    WorkspaceError,
    git,
    git_path,
    git_paths,
    is_within,
    repository_root,
)

if TYPE_CHECKING:
    from fast_agent.transactional.checkpoint.worktree import WorktreeMetadata
    from fast_agent.transactional.models import RunId

CheckpointId = NewType("CheckpointId", str)
WorkspaceVersion = NewType("WorkspaceVersion", str)


class WorkspaceDivergenceError(WorkspaceError):
    """Raised when restore does not reproduce the checkpointed version."""


class WorkspaceFileKind(StrEnum):
    TRACKED = "tracked"
    UNTRACKED = "untracked"


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    path: str
    kind: WorkspaceFileKind
    exists: bool
    executable: bool
    content_sha256: str | None
    symlink_target: str | None


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    checkpoint_id: CheckpointId
    run_id: RunId
    baseline_commit: str
    workspace_version: WorkspaceVersion
    files: tuple[WorkspaceFile, ...]


class CheckpointManager:
    """Snapshot and restore non-ignored state inside one Run worktree."""

    def __init__(self, worktree: WorktreeMetadata, checkpoints_root: Path) -> None:
        self.worktree = worktree
        self.checkpoints_root = checkpoints_root.resolve()
        if is_within(self.checkpoints_root, worktree.repository_root) or is_within(
            self.checkpoints_root, worktree.worktree_path
        ):
            raise WorkspaceError("Checkpoint root must be outside source and Run worktrees")
        self.checkpoints_root.mkdir(parents=True, exist_ok=True)
        if repository_root(worktree.worktree_path) != worktree.worktree_path.resolve():
            raise WorkspaceError("Checkpoint worktree is not a repository root")

    def create(self) -> CheckpointMetadata:
        files = self._workspace_files()
        workspace_version = _workspace_version(files)
        checkpoint_id = CheckpointId(uuid4().hex)
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        snapshot_dir = checkpoint_dir / "files"
        snapshot_dir.mkdir(parents=True)

        for entry in files:
            if entry.exists:
                _copy_workspace_entry(
                    _workspace_path(self.worktree.worktree_path, entry.path),
                    snapshot_dir / entry.path,
                )
        shutil.copy2(git_path(self.worktree.worktree_path, "index"), checkpoint_dir / "index")

        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            run_id=self.worktree.run_id,
            baseline_commit=self.worktree.baseline_commit,
            workspace_version=workspace_version,
            files=files,
        )
        (checkpoint_dir / "manifest.json").write_text(
            json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return metadata

    def restore(self, checkpoint: CheckpointMetadata) -> WorkspaceVersion:
        self._validate_checkpoint(checkpoint)
        current_paths = {entry.path for entry in self._workspace_files()}
        checkpoint_paths = {entry.path for entry in checkpoint.files}
        for relative_path in sorted(current_paths | checkpoint_paths, reverse=True):
            path = _workspace_path(self.worktree.worktree_path, relative_path)
            if path.is_dir() and not path.is_symlink():
                raise UnsupportedWorkspaceError(
                    f"Workspace entry is not a file or symlink: {relative_path}"
                )
            path.unlink(missing_ok=True)

        checkpoint_dir = self._checkpoint_dir(checkpoint.checkpoint_id)
        shutil.copy2(checkpoint_dir / "index", git_path(self.worktree.worktree_path, "index"))
        for entry in checkpoint.files:
            if entry.exists:
                _copy_workspace_entry(
                    checkpoint_dir / "files" / entry.path,
                    _workspace_path(self.worktree.worktree_path, entry.path),
                )
        _remove_empty_directories(self.worktree.worktree_path)

        restored = _workspace_version(self._workspace_files())
        if restored != checkpoint.workspace_version:
            raise WorkspaceDivergenceError(
                f"Restored workspace version {restored} does not match {checkpoint.workspace_version}"
            )
        return restored

    def current_version(self) -> WorkspaceVersion:
        return _workspace_version(self._workspace_files())

    def _workspace_files(self) -> tuple[WorkspaceFile, ...]:
        _validate_supported_repository(self.worktree.worktree_path)
        tracked = set(git_paths(self.worktree.worktree_path, "ls-files", "--cached", "-z"))
        untracked = set(
            git_paths(
                self.worktree.worktree_path,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
        )
        return tuple(
            _workspace_file(
                self.worktree.worktree_path,
                path,
                WorkspaceFileKind.TRACKED if path in tracked else WorkspaceFileKind.UNTRACKED,
            )
            for path in sorted(tracked | untracked)
        )

    def _checkpoint_dir(self, checkpoint_id: CheckpointId) -> Path:
        if not checkpoint_id or any(
            character not in "0123456789abcdef" for character in checkpoint_id
        ):
            raise WorkspaceError("Checkpoint ID must be lowercase hexadecimal")
        path = (self.checkpoints_root / str(checkpoint_id)).resolve()
        try:
            path.relative_to(self.checkpoints_root)
        except ValueError as exc:
            raise WorkspaceError("Checkpoint path escapes the checkpoint root") from exc
        return path

    def _validate_checkpoint(self, checkpoint: CheckpointMetadata) -> None:
        if checkpoint.run_id != self.worktree.run_id:
            raise WorkspaceError("Checkpoint belongs to a different Run")
        if checkpoint.baseline_commit != self.worktree.baseline_commit:
            raise WorkspaceError("Checkpoint belongs to a different baseline")
        checkpoint_dir = self._checkpoint_dir(checkpoint.checkpoint_id)
        if not (checkpoint_dir / "manifest.json").is_file():
            raise WorkspaceError(f"Checkpoint does not exist: {checkpoint.checkpoint_id}")


def _validate_supported_repository(root: Path) -> None:
    modes = git(root, "ls-files", "--stage").splitlines()
    if any(line.startswith("160000 ") for line in modes):
        raise UnsupportedWorkspaceError("Git submodules are not supported")
    nested = next(
        (candidate for candidate in root.rglob(".git") if candidate != root / ".git"),
        None,
    )
    if nested is not None:
        raise UnsupportedWorkspaceError(f"Nested Git repositories are not supported: {nested.parent}")


def _workspace_file(root: Path, relative_path: str, kind: WorkspaceFileKind) -> WorkspaceFile:
    path = _workspace_path(root, relative_path)
    if not path.exists() and not path.is_symlink():
        return WorkspaceFile(relative_path, kind, False, False, None, None)
    if path.is_symlink():
        target = os.readlink(path)
        digest = hashlib.sha256(target.encode()).hexdigest()
        return WorkspaceFile(relative_path, kind, True, False, digest, target)
    if not path.is_file():
        raise UnsupportedWorkspaceError(f"Workspace entry is not a file: {relative_path}")
    content = path.read_bytes()
    return WorkspaceFile(
        relative_path,
        kind,
        True,
        bool(path.stat().st_mode & 0o111),
        hashlib.sha256(content).hexdigest(),
        None,
    )


def _workspace_version(files: tuple[WorkspaceFile, ...]) -> WorkspaceVersion:
    payload = json.dumps(
        [asdict(entry) for entry in files],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return WorkspaceVersion(f"sha256:{hashlib.sha256(payload).hexdigest()}")


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


def _copy_workspace_entry(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    else:
        shutil.copy2(source, destination)


def _remove_empty_directories(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if directory != root and not any(directory.iterdir()):
            directory.rmdir()
