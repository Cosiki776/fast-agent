from __future__ import annotations


def is_within_workspace(root: str, candidate: str) -> bool:
    """Return whether a candidate path is contained by the workspace root."""
    return candidate.startswith(root)
