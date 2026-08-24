from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.transactional.storage.artifact_store import FileArtifactStore
from fast_agent.transactional.verification import CompletionVerifier, VerificationSpec

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_verifier_persists_complete_output_and_bounds_failure_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    verifier = CompletionVerifier(
        LocalShellExecutor(logger=logging.getLogger(__name__), working_directory=workspace),
        artifacts,
    )

    result = await verifier.verify(
        VerificationSpec(
            command="printf '0123456789'; printf 'failure' >&2; exit 7",
            timeout_seconds=5,
            max_evidence_bytes=80,
        ),
        workspace,
    )

    assert result.passed is False
    assert result.exit_code == 7
    assert result.timed_out is False
    assert artifacts.read(result.stdout_artifact_id) == b"0123456789"
    assert artifacts.read(result.stderr_artifact_id) == b"failure"
    assert len(result.evidence.encode()) <= 80


@pytest.mark.asyncio
async def test_verifier_passes_only_for_zero_exit_without_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verifier = CompletionVerifier(
        LocalShellExecutor(logger=logging.getLogger(__name__), working_directory=workspace),
        FileArtifactStore(tmp_path / "artifacts"),
    )

    result = await verifier.verify(
        VerificationSpec(command='test "$PWD" = "$(pwd)"', timeout_seconds=5),
        workspace,
    )

    assert result.passed is True
    assert result.exit_code == 0
