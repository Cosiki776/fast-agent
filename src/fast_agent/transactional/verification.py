from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.transactional.storage.artifact_store import ArtifactId, ArtifactKind

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from fast_agent.tools.execution_environment import ShellEnvironment
    from fast_agent.transactional.storage.artifact_store import FileArtifactStore


class VerificationSpec(BaseModel):
    command: str = Field(min_length=1)
    timeout_seconds: int = Field(gt=0)
    max_evidence_bytes: int = Field(default=16 * 1024, ge=256)

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    exit_code: int
    timed_out: bool
    stdout_artifact_id: ArtifactId
    stderr_artifact_id: ArtifactId
    evidence: str


class CompletionVerificationError(RuntimeError):
    """Raised with bounded verifier evidence when a completion claim is false."""

    def __init__(self, result: VerificationResult) -> None:
        self.result = result
        super().__init__(result.evidence)


class CompletionVerifier:
    """Run one configured completion command in the isolated Run workspace."""

    def __init__(
        self,
        environment: ShellEnvironment,
        artifact_store: FileArtifactStore,
    ) -> None:
        self._environment = environment
        self._artifact_store = artifact_store

    async def verify(self, spec: VerificationSpec, workspace: Path) -> VerificationResult:
        execution = await self._environment.execute(
            ShellExecutionRequest(
                command=spec.command,
                cwd=str(workspace),
                timeout=spec.timeout_seconds,
            )
        )
        result = execution.result
        stdout = result.stdout.encode()
        stderr = result.stderr.encode()
        stdout_artifact = self._artifact_store.put(
            stdout,
            media_type="text/plain; charset=utf-8",
            kind=ArtifactKind.STDOUT,
        )
        stderr_artifact = self._artifact_store.put(
            stderr,
            media_type="text/plain; charset=utf-8",
            kind=ArtifactKind.STDERR,
        )
        passed = not execution.timed_out and result.exit_code == 0
        evidence = _bounded_evidence(
            result.stdout,
            result.stderr,
            exit_code=result.exit_code,
            timed_out=execution.timed_out,
            max_bytes=spec.max_evidence_bytes,
            stdout_artifact_id=stdout_artifact.artifact_id,
            stderr_artifact_id=stderr_artifact.artifact_id,
        )
        return VerificationResult(
            passed=passed,
            exit_code=result.exit_code,
            timed_out=execution.timed_out,
            stdout_artifact_id=stdout_artifact.artifact_id,
            stderr_artifact_id=stderr_artifact.artifact_id,
            evidence=evidence,
        )


def _bounded_evidence(
    stdout: str,
    stderr: str,
    *,
    exit_code: int,
    timed_out: bool,
    max_bytes: int,
    stdout_artifact_id: ArtifactId,
    stderr_artifact_id: ArtifactId,
) -> str:
    lines = [line for line in f"{stdout}\n{stderr}".splitlines() if line]
    important = _unique_lines(
        line.strip()
        for line in lines
        if re.search(r"(?:FAILED|ERROR|FAIL:|ERROR:|AssertionError|Traceback)", line)
    )
    head = lines[:6]
    tail = lines[-12:] if len(lines) > 6 else []
    parts = [
        "Completion verification failed.",
        f"exit_code: {exit_code}",
        f"timed_out: {str(timed_out).lower()}",
    ]
    if important:
        parts.extend(("key_errors:", *(f"- {line}" for line in important[:12])))
    if head:
        parts.extend(("output_head:", *head))
    if tail:
        parts.extend(("output_tail:", *tail))
    parts.extend(
        (
            f"full_stdout_artifact: {stdout_artifact_id}",
            f"full_stderr_artifact: {stderr_artifact_id}",
        )
    )
    return _fit_utf8("\n".join(parts), max_bytes)


def _unique_lines(lines: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for line in lines:
        if line and line not in unique:
            unique.append(line)
    return unique


def _fit_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    suffix = b"\n[output truncated]"
    encoded = encoded[: max_bytes - len(suffix)]
    while True:
        try:
            return encoded.decode() + suffix.decode()
        except UnicodeDecodeError:
            encoded = encoded[:-1]
