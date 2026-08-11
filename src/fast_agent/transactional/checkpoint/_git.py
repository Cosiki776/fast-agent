from __future__ import annotations

import subprocess
from pathlib import Path


class WorkspaceError(RuntimeError):
    """Base error for transactional workspace operations."""


class UnsupportedWorkspaceError(WorkspaceError):
    """Raised when a repository uses an unsupported workspace shape."""


def repository_root(path: Path) -> Path:
    resolved = path.resolve()
    root = Path(git(resolved, "rev-parse", "--show-toplevel")).resolve()
    superproject = git(resolved, "rev-parse", "--show-superproject-working-tree")
    if superproject:
        raise UnsupportedWorkspaceError("Git submodules are not supported")
    return root


def git_path(root: Path, name: str) -> Path:
    return Path(git(root, "rev-parse", "--git-path", name)).resolve()


def git_paths(root: Path, *args: str) -> tuple[str, ...]:
    output = git_bytes(root, *args).decode(errors="surrogateescape")
    return tuple(path for path in output.split("\0") if path)


def git(root: Path, *args: str) -> str:
    return git_bytes(root, *args).decode(errors="replace").strip()


def git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise WorkspaceError(detail or f"Git command failed: {' '.join(args)}")
    return result.stdout


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
