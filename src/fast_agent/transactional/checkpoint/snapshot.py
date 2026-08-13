from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NewType
from uuid import uuid4

from fast_agent.transactional.checkpoint._git import (
    UnsupportedWorkspaceError,
    WorkspaceError,
    git,
    git_bytes,
    git_path,
    git_paths,
    is_within,
    repository_root,
)

if TYPE_CHECKING:
    from fast_agent.transactional.checkpoint.worktree import WorktreeMetadata

SnapshotId = NewType("SnapshotId", str)
WorkspaceFingerprint = NewType("WorkspaceFingerprint", str)
SnapshotVersion = NewType("SnapshotVersion", str)
type SnapshotPathFilter = Callable[[str], bool]


class WorkspaceChangedError(WorkspaceError):
    """Raised when the source workspace changed after capture."""


class SnapshotFileKind(StrEnum):
    TRACKED = "tracked"
    UNTRACKED = "untracked"


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    path: str
    kind: SnapshotFileKind
    exists: bool
    executable: bool
    content_sha256: str | None
    symlink_target: str | None


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    snapshot_id: SnapshotId
    source_root: Path
    selected_branch: str
    base_commit: str
    source_fingerprint: WorkspaceFingerprint
    workspace_version: SnapshotVersion
    files: tuple[SnapshotFile, ...]
    excluded_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentDelta:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CapturedWorkspace:
    branch: str
    base_commit: str
    index_sha256: str
    files: tuple[SnapshotFile, ...]
    excluded_paths: tuple[str, ...]
    content: dict[str, bytes | str]


class WorkspaceSnapshotManager:
    """Seed Run worktrees from a stable copy of the user's current workspace."""

    def __init__(
        self,
        source_workspace: Path,
        snapshots_root: Path,
        *,
        path_filter: SnapshotPathFilter | None = None,
    ) -> None:
        self.source_root = repository_root(source_workspace)
        self.snapshots_root = snapshots_root.resolve()
        if is_within(self.snapshots_root, self.source_root):
            raise WorkspaceError("Snapshot root must be outside the source workspace")
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        self._path_filter = path_filter or is_safe_snapshot_path

    def capture(self) -> WorkspaceSnapshot:
        captured = self._capture()
        fingerprint = _fingerprint(captured)
        if _fingerprint(self._capture(include_content=False)) != fingerprint:
            raise WorkspaceChangedError("Source workspace changed while capturing its snapshot")

        snapshot_id = SnapshotId(uuid4().hex)
        snapshot_dir = self._snapshot_dir(snapshot_id)
        files_dir = snapshot_dir / "files"
        files_dir.mkdir(parents=True)
        for entry in captured.files:
            if entry.exists:
                _write_snapshot_entry(files_dir / entry.path, captured.content[entry.path], entry)

        snapshot = WorkspaceSnapshot(
            snapshot_id=snapshot_id,
            source_root=self.source_root,
            selected_branch=captured.branch,
            base_commit=captured.base_commit,
            source_fingerprint=fingerprint,
            workspace_version=_content_version(captured.files),
            files=captured.files,
            excluded_paths=captured.excluded_paths,
        )
        manifest = asdict(snapshot)
        manifest["source_root"] = str(snapshot.source_root)
        (snapshot_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return snapshot

    def materialize(self, snapshot: WorkspaceSnapshot, worktree: WorktreeMetadata) -> None:
        self._validate_snapshot(snapshot)
        self._validate_worktree(snapshot, worktree)
        root = worktree.worktree_path.resolve()
        if repository_root(root) != root:
            raise WorkspaceError("Snapshot destination is not a repository root")

        current_paths = _managed_paths(root, snapshot.base_commit)
        snapshot_paths = {entry.path for entry in snapshot.files}
        for relative_path in sorted(current_paths | snapshot_paths | set(snapshot.excluded_paths)):
            path = _workspace_path(root, relative_path)
            if path.is_dir() and not path.is_symlink():
                raise UnsupportedWorkspaceError(
                    f"Workspace entry is not a file or symlink: {relative_path}"
                )
            path.unlink(missing_ok=True)

        files_dir = self._snapshot_dir(snapshot.snapshot_id) / "files"
        for entry in snapshot.files:
            if entry.exists:
                _copy_snapshot_entry(files_dir / entry.path, _workspace_path(root, entry.path))
        _remove_empty_directories(root)

        materialized = self._files_at(root, snapshot.base_commit)
        if _content_version(materialized) != snapshot.workspace_version:
            raise WorkspaceChangedError("Materialized Run worktree does not match its snapshot")

    def assert_source_unchanged(self, snapshot: WorkspaceSnapshot) -> None:
        self._validate_snapshot(snapshot)
        if _fingerprint(self._capture(include_content=False)) != snapshot.source_fingerprint:
            raise WorkspaceChangedError("Source workspace changed after snapshot capture")

    def agent_delta(self, snapshot: WorkspaceSnapshot, worktree: WorktreeMetadata) -> AgentDelta:
        self._validate_worktree(snapshot, worktree)
        current = {
            entry.path: entry
            for entry in self._files_at(worktree.worktree_path, snapshot.base_commit)
        }
        initial = {entry.path: entry for entry in snapshot.files}
        added: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        for path in sorted(current.keys() | initial.keys()):
            before = initial.get(path)
            after = current.get(path)
            before_exists = before is not None and before.exists
            after_exists = after is not None and after.exists
            if not before_exists and after_exists:
                added.append(path)
            elif before_exists and not after_exists:
                deleted.append(path)
            elif (
                before is not None
                and after is not None
                and _file_identity(before) != _file_identity(after)
            ):
                modified.append(path)
        return AgentDelta(tuple(added), tuple(modified), tuple(deleted))

    def _capture(self, *, include_content: bool = True) -> _CapturedWorkspace:
        _validate_supported_workspace(self.source_root)
        branch = git(self.source_root, "symbolic-ref", "--short", "HEAD")
        base_commit = git(self.source_root, "rev-parse", "HEAD^{commit}")
        index_sha256 = hashlib.sha256(
            git_bytes(self.source_root, "ls-files", "--stage", "-z")
        ).hexdigest()
        files, excluded, content = _capture_files(
            self.source_root,
            base_commit,
            self._path_filter,
            include_content=include_content,
        )
        return _CapturedWorkspace(branch, base_commit, index_sha256, files, excluded, content)

    def _files_at(self, root: Path, base_commit: str) -> tuple[SnapshotFile, ...]:
        files, _, _ = _capture_files(
            root,
            base_commit,
            self._path_filter,
            include_content=False,
        )
        return files

    def _snapshot_dir(self, snapshot_id: SnapshotId) -> Path:
        if not snapshot_id or any(character not in "0123456789abcdef" for character in snapshot_id):
            raise WorkspaceError("Snapshot ID must be lowercase hexadecimal")
        return self.snapshots_root / str(snapshot_id)

    def _validate_snapshot(self, snapshot: WorkspaceSnapshot) -> None:
        if snapshot.source_root.resolve() != self.source_root:
            raise WorkspaceError("WorkspaceSnapshot belongs to a different source workspace")
        if not (self._snapshot_dir(snapshot.snapshot_id) / "manifest.json").is_file():
            raise WorkspaceError(f"WorkspaceSnapshot does not exist: {snapshot.snapshot_id}")

    def _validate_worktree(
        self,
        snapshot: WorkspaceSnapshot,
        worktree: WorktreeMetadata,
    ) -> None:
        if worktree.repository_root.resolve() != self.source_root:
            raise WorkspaceError("Run worktree belongs to a different source repository")
        if worktree.baseline_commit != snapshot.base_commit:
            raise WorkspaceError("Run worktree baseline does not match the WorkspaceSnapshot")


def is_safe_snapshot_path(path: str) -> bool:
    """Exclude common credential paths until the PR9 Policy is available."""
    parts = tuple(part.lower() for part in Path(path).parts)
    name = parts[-1] if parts else ""
    if name == ".env" or name.startswith(".env."):
        return False
    if name in {"credentials", "credentials.json", "secrets.json", "service-account.json"}:
        return False
    if name in {"id_rsa", "id_ed25519"} or Path(name).suffix in {".key", ".p12", ".pem", ".pfx"}:
        return False
    return not any(part in {".aws", ".gnupg", ".ssh", "credentials", "secrets"} for part in parts)


def _capture_files(
    root: Path,
    base_commit: str,
    path_filter: SnapshotPathFilter,
    *,
    include_content: bool,
) -> tuple[tuple[SnapshotFile, ...], tuple[str, ...], dict[str, bytes | str]]:
    base_paths = set(git_paths(root, "ls-tree", "-r", "--name-only", "-z", base_commit))
    tracked = set(git_paths(root, "ls-files", "--cached", "-z"))
    untracked = set(git_paths(root, "ls-files", "--others", "--exclude-standard", "-z"))
    kinds = {
        path: SnapshotFileKind.TRACKED
        if path in base_paths | tracked
        else SnapshotFileKind.UNTRACKED
        for path in base_paths | tracked | untracked
    }
    excluded = tuple(sorted(path for path in kinds if not path_filter(path)))
    files: list[SnapshotFile] = []
    content: dict[str, bytes | str] = {}
    for relative_path in sorted(path for path in kinds if path_filter(path)):
        entry, value = _capture_file(root, relative_path, kinds[relative_path])
        files.append(entry)
        if include_content and value is not None:
            content[relative_path] = value
    return tuple(files), excluded, content


def _capture_file(
    root: Path,
    relative_path: str,
    kind: SnapshotFileKind,
) -> tuple[SnapshotFile, bytes | str | None]:
    path = _workspace_path(root, relative_path)
    if not path.exists() and not path.is_symlink():
        return SnapshotFile(relative_path, kind, False, False, None, None), None
    if path.is_symlink():
        _validate_symlink(root, path)
        target = os.readlink(path)
        digest = hashlib.sha256(target.encode()).hexdigest()
        return SnapshotFile(relative_path, kind, True, False, digest, target), target
    if not path.is_file():
        raise UnsupportedWorkspaceError(f"Workspace entry is not a file: {relative_path}")
    value = path.read_bytes()
    return (
        SnapshotFile(
            relative_path,
            kind,
            True,
            bool(path.stat().st_mode & 0o111),
            hashlib.sha256(value).hexdigest(),
            None,
        ),
        value,
    )


def _managed_paths(root: Path, base_commit: str) -> set[str]:
    return (
        set(git_paths(root, "ls-tree", "-r", "--name-only", "-z", base_commit))
        | set(git_paths(root, "ls-files", "--cached", "-z"))
        | set(git_paths(root, "ls-files", "--others", "--exclude-standard", "-z"))
    )


def _fingerprint(captured: _CapturedWorkspace) -> WorkspaceFingerprint:
    payload = json.dumps(
        {
            "base_commit": captured.base_commit,
            "branch": captured.branch,
            "excluded_paths": captured.excluded_paths,
            "files": [asdict(entry) for entry in captured.files],
            "index_sha256": captured.index_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return WorkspaceFingerprint(f"sha256:{hashlib.sha256(payload).hexdigest()}")


def _content_version(files: tuple[SnapshotFile, ...]) -> SnapshotVersion:
    payload = json.dumps(
        [
            {
                "content_sha256": entry.content_sha256,
                "executable": entry.executable,
                "exists": entry.exists,
                "path": entry.path,
                "symlink_target": entry.symlink_target,
            }
            for entry in files
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SnapshotVersion(f"sha256:{hashlib.sha256(payload).hexdigest()}")


def _file_identity(entry: SnapshotFile) -> tuple[bool, bool, str | None, str | None]:
    return entry.exists, entry.executable, entry.content_sha256, entry.symlink_target


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


def _validate_symlink(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise WorkspaceError(f"Symlink escapes the workspace: {path.relative_to(root)}") from exc


def _write_snapshot_entry(path: Path, content: bytes | str, entry: SnapshotFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if entry.symlink_target is not None:
        path.symlink_to(content)
    else:
        if not isinstance(content, bytes):
            raise WorkspaceError(f"Snapshot content type is invalid: {entry.path}")
        path.write_bytes(content)
        if entry.executable:
            path.chmod(path.stat().st_mode | 0o111)


def _copy_snapshot_entry(source: Path, destination: Path) -> None:
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


def _validate_supported_workspace(root: Path) -> None:
    modes = git(root, "ls-files", "--stage").splitlines()
    if any(line.startswith("160000 ") for line in modes):
        raise UnsupportedWorkspaceError("Git submodules are not supported")
    if git_path(root, "info/sparse-checkout").exists():
        raise UnsupportedWorkspaceError("Sparse checkouts are not supported")
    nested = next(
        (candidate for candidate in root.rglob(".git") if candidate != root / ".git"),
        None,
    )
    if nested is not None:
        raise UnsupportedWorkspaceError(
            f"Nested Git repositories are not supported: {nested.parent}"
        )
