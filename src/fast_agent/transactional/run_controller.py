from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from fast_agent.transactional.run_events import (
    RunFailed,
    RunState,
    RunVerificationFailed,
    RunVerificationStarted,
    RunVerified,
)
from fast_agent.transactional.verification import CompletionVerificationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from fast_agent.transactional.budget import RunBudgetTracker
    from fast_agent.transactional.models import RunId
    from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore
    from fast_agent.transactional.verification import CompletionVerifier, VerificationSpec

ResultT = TypeVar("ResultT")


class TransactionalRunTerminatedError(RuntimeError):
    """Raised when a caller tries to continue a failed transactional run."""


class TransactionalCodingRun:
    """Guard one transactional run's agent turns with shared state and budget."""

    def __init__(
        self,
        run_id: RunId,
        run_events: SQLiteRunEventStore,
        budget: RunBudgetTracker,
        *,
        verifier: CompletionVerifier | None = None,
        verification_spec: VerificationSpec | None = None,
        workspace: Path | None = None,
        workspace_version: Callable[[], str] | None = None,
    ) -> None:
        self.run_id = run_id
        self._run_events = run_events
        self._budget = budget
        self._verifier = verifier
        self._verification_spec = verification_spec
        self._workspace = workspace
        self._workspace_version = workspace_version

    @property
    def state(self) -> RunState:
        return self._run_events.replay(self.run_id).state

    async def call_agent_once(self, call: Callable[[], Awaitable[ResultT]]) -> ResultT:
        if self.state is RunState.FAILED:
            raise TransactionalRunTerminatedError(f"Transactional run '{self.run_id}' has failed")
        wall_time = self._budget.check_wall_time()
        if not wall_time.allowed:
            self.fail("budget exhausted: wall_time")
            raise TransactionalRunTerminatedError("Transactional run wall-time budget is exhausted")
        try:
            result = await call()
            await self._verify_completion()
            return result
        except CompletionVerificationError:
            raise
        except Exception as exc:
            self.fail(f"agent turn failed: {type(exc).__name__}: {exc}")
            raise

    async def _verify_completion(self) -> None:
        if self._verifier is None or self._verification_spec is None:
            return
        if self._workspace is None or self._workspace_version is None:
            raise RuntimeError("Completion verifier requires a workspace and version provider")
        self._run_events.append(
            RunVerificationStarted(
                run_id=self.run_id,
                command=self._verification_spec.command,
            )
        )
        result = await self._verifier.verify(self._verification_spec, self._workspace)
        if not result.passed:
            self._run_events.append(
                RunVerificationFailed(
                    run_id=self.run_id,
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                    stdout_artifact_id=result.stdout_artifact_id,
                    stderr_artifact_id=result.stderr_artifact_id,
                )
            )
            raise CompletionVerificationError(result)
        self._run_events.append(
            RunVerified(
                run_id=self.run_id,
                workspace_version=self._workspace_version(),
                stdout_artifact_id=result.stdout_artifact_id,
                stderr_artifact_id=result.stderr_artifact_id,
            )
        )

    def fail(self, reason: str) -> None:
        if self.state is not RunState.FAILED:
            self._run_events.append(RunFailed(run_id=self.run_id, reason=reason))
