from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from fast_agent.transactional.checkpoint.promotion import PromotionRejectedError
from fast_agent.transactional.run_events import (
    PromotionApplied,
    PromotionRejected,
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
    from fast_agent.transactional.checkpoint.promotion import PromotionResult
    from fast_agent.transactional.models import RunId
    from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore
    from fast_agent.transactional.verification import CompletionVerifier, VerificationSpec

ResultT = TypeVar("ResultT")


class TransactionalRunTerminatedError(RuntimeError):
    """Raised when a caller tries to continue a failed transactional run."""


class VerificationWorkspaceChangedError(RuntimeError):
    """Raised when a successful verification command mutates the candidate workspace."""


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
        promote: Callable[[str], PromotionResult] | None = None,
        create_verification_checkpoint: Callable[[], str] | None = None,
        restore_verification_checkpoint: Callable[[str], str] | None = None,
    ) -> None:
        if (create_verification_checkpoint is None) != (restore_verification_checkpoint is None):
            raise ValueError("Verification checkpoint callbacks must be configured together")
        self.run_id = run_id
        self._run_events = run_events
        self._budget = budget
        self._verifier = verifier
        self._verification_spec = verification_spec
        self._workspace = workspace
        self._workspace_version = workspace_version
        self._promote = promote
        self._create_verification_checkpoint = create_verification_checkpoint
        self._restore_verification_checkpoint = restore_verification_checkpoint

    @property
    def state(self) -> RunState:
        return self._run_events.replay(self.run_id).state

    async def call_agent_once(self, call: Callable[[], Awaitable[ResultT]]) -> ResultT:
        if self.state in {
            RunState.FAILED,
            RunState.VERIFIED,
            RunState.PROMOTED,
            RunState.PROMOTION_REJECTED,
        }:
            raise TransactionalRunTerminatedError(
                f"Transactional run '{self.run_id}' is terminal ({self.state.value})"
            )
        try:
            return await self._call_until_verified(call, None)
        except (CompletionVerificationError, PromotionRejectedError):
            raise
        except Exception as exc:
            self.fail(f"agent turn failed: {type(exc).__name__}: {exc}")
            raise

    async def call_agent_until_verified(
        self,
        call: Callable[[], Awaitable[ResultT]],
        retry: Callable[[str], Awaitable[ResultT]],
    ) -> ResultT:
        if self.state in {
            RunState.FAILED,
            RunState.VERIFIED,
            RunState.PROMOTED,
            RunState.PROMOTION_REJECTED,
        }:
            raise TransactionalRunTerminatedError(
                f"Transactional run '{self.run_id}' is terminal ({self.state.value})"
            )
        try:
            return await self._call_until_verified(call, retry)
        except PromotionRejectedError:
            raise
        except Exception as exc:
            self.fail(f"agent turn failed: {type(exc).__name__}: {exc}")
            raise

    async def _call_until_verified(
        self,
        call: Callable[[], Awaitable[ResultT]],
        retry: Callable[[str], Awaitable[ResultT]] | None,
    ) -> ResultT:
        next_call = call
        while True:
            wall_time = self._budget.check_wall_time()
            if not wall_time.allowed:
                self.fail("budget exhausted: wall_time")
                raise TransactionalRunTerminatedError(
                    "Transactional run wall-time budget is exhausted"
                )
            result = await next_call()
            try:
                await self._verify_completion()
            except CompletionVerificationError as exc:
                if retry is None:
                    raise
                evidence = exc.result.evidence

                async def retry_call() -> ResultT:
                    return await retry(evidence)

                next_call = retry_call
                continue
            return result

    async def _verify_completion(self) -> None:
        if self._verifier is None or self._verification_spec is None:
            return
        if self._workspace is None or self._workspace_version is None:
            raise RuntimeError("Completion verifier requires a workspace and version provider")
        candidate_version = self._workspace_version()
        verification_checkpoint = (
            self._create_verification_checkpoint()
            if self._create_verification_checkpoint is not None
            else None
        )
        self._run_events.append(
            RunVerificationStarted(
                run_id=self.run_id,
                command=self._verification_spec.command,
            )
        )
        result = await self._verifier.verify(self._verification_spec, self._workspace)
        verified_version = self._workspace_version()
        workspace_changed = verified_version != candidate_version
        workspace_restored = False
        if workspace_changed and verification_checkpoint is not None:
            if self._restore_verification_checkpoint is None:
                raise RuntimeError("Verification checkpoint restore is unavailable")
            self._restore_verification_checkpoint(verification_checkpoint)
            restored_version = self._workspace_version()
            if restored_version != candidate_version:
                raise RuntimeError("Could not restore the pre-verification Agent workspace")
            workspace_restored = True
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
        if workspace_changed:
            if not workspace_restored:
                self._run_events.append(
                    RunVerificationFailed(
                        run_id=self.run_id,
                        exit_code=result.exit_code,
                        timed_out=result.timed_out,
                        stdout_artifact_id=result.stdout_artifact_id,
                        stderr_artifact_id=result.stderr_artifact_id,
                    )
                )
                raise VerificationWorkspaceChangedError(
                    "Completion verification modified non-ignored workspace files; "
                    "refused promotion because no verification checkpoint was available"
                )
            verified_version = candidate_version
        self._run_events.append(
            RunVerified(
                run_id=self.run_id,
                workspace_version=verified_version,
                stdout_artifact_id=result.stdout_artifact_id,
                stderr_artifact_id=result.stderr_artifact_id,
            )
        )
        if self._promote is not None:
            try:
                promotion = self._promote(verified_version)
            except PromotionRejectedError as exc:
                self._run_events.append(
                    PromotionRejected(
                        run_id=self.run_id,
                        reason=exc.reason,
                        patch_artifact_id=exc.patch_artifact_id,
                    )
                )
                raise
            self._run_events.append(
                PromotionApplied(
                    run_id=self.run_id,
                    workspace_version=promotion.workspace_version,
                    patch_artifact_id=promotion.patch_artifact_id,
                )
            )

    def fail(self, reason: str) -> None:
        if self.state is not RunState.FAILED:
            self._run_events.append(RunFailed(run_id=self.run_id, reason=reason))
