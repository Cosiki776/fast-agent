from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.transactional.budget import RunBudgetLimits, RunBudgetTracker
from fast_agent.transactional.models import RunId
from fast_agent.transactional.run_controller import (
    TransactionalCodingRun,
    TransactionalRunTerminatedError,
)
from fast_agent.transactional.run_events import RunStarted, RunState
from fast_agent.transactional.storage.artifact_store import FileArtifactStore
from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore
from fast_agent.transactional.verification import (
    CompletionVerificationError,
    CompletionVerifier,
    VerificationSpec,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_controller_reuses_run_and_stops_after_wall_time_budget(tmp_path: Path) -> None:
    now = 0.0

    def clock() -> float:
        return now

    run_id = RunId("run-1")
    budget = RunBudgetTracker(RunBudgetLimits(max_wall_time_seconds=5), clock=clock)
    with SQLiteRunEventStore(tmp_path / "events.sqlite3") as events:
        events.append(RunStarted(run_id=run_id, profile="full"))
        controller = TransactionalCodingRun(run_id, events, budget)

        async def call() -> str:
            return "done"

        assert await controller.call_agent_once(call) == "done"
        now = 5.0
        with pytest.raises(TransactionalRunTerminatedError, match="wall-time"):
            await controller.call_agent_once(call)

        assert controller.state is RunState.FAILED


@pytest.mark.asyncio
async def test_agent_exception_marks_run_failed(tmp_path: Path) -> None:
    run_id = RunId("run-1")
    budget = RunBudgetTracker(RunBudgetLimits())
    with SQLiteRunEventStore(tmp_path / "events.sqlite3") as events:
        events.append(RunStarted(run_id=run_id, profile="full"))
        controller = TransactionalCodingRun(run_id, events, budget)

        async def fail() -> str:
            raise RuntimeError("provider unavailable")

        with pytest.raises(RuntimeError, match="provider unavailable"):
            await controller.call_agent_once(fail)

        assert controller.state is RunState.FAILED


@pytest.mark.asyncio
async def test_controller_verifies_completion_and_allows_retry_after_failure(
    tmp_path: Path,
) -> None:
    run_id = RunId("run-1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    budget = RunBudgetTracker(RunBudgetLimits())
    verifier = CompletionVerifier(
        LocalShellExecutor(logger=logging.getLogger(__name__), working_directory=workspace),
        FileArtifactStore(tmp_path / "artifacts"),
    )
    spec = VerificationSpec(command="test -f complete", timeout_seconds=5)
    with SQLiteRunEventStore(tmp_path / "events.sqlite3") as events:
        events.append(RunStarted(run_id=run_id, profile="full"))
        controller = TransactionalCodingRun(
            run_id,
            events,
            budget,
            verifier=verifier,
            verification_spec=spec,
            workspace=workspace,
            workspace_version=lambda: "version-1",
        )

        async def call() -> str:
            return "done"

        with pytest.raises(CompletionVerificationError, match="exit_code: 1"):
            await controller.call_agent_once(call)
        assert controller.state is RunState.VERIFICATION_FAILED

        workspace.joinpath("complete").touch()
        assert await controller.call_agent_once(call) == "done"
        assert controller.state is RunState.VERIFIED


@pytest.mark.asyncio
async def test_controller_returns_verifier_evidence_to_agent_before_retry(tmp_path: Path) -> None:
    run_id = RunId("run-feedback")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    budget = RunBudgetTracker(RunBudgetLimits())
    verifier = CompletionVerifier(
        LocalShellExecutor(logger=logging.getLogger(__name__), working_directory=workspace),
        FileArtifactStore(tmp_path / "artifacts"),
    )
    feedback: list[str] = []
    with SQLiteRunEventStore(tmp_path / "events.sqlite3") as events:
        events.append(RunStarted(run_id=run_id, profile="full"))
        controller = TransactionalCodingRun(
            run_id,
            events,
            budget,
            verifier=verifier,
            verification_spec=VerificationSpec(
                command="test -f complete",
                timeout_seconds=5,
            ),
            workspace=workspace,
            workspace_version=lambda: "version-1",
        )

        async def initial_call() -> str:
            return "incorrect completion"

        async def retry(evidence: str) -> str:
            feedback.append(evidence)
            workspace.joinpath("complete").touch()
            return "corrected completion"

        result = await controller.call_agent_until_verified(initial_call, retry)

        assert result == "corrected completion"
        assert len(feedback) == 1
        assert "Completion verification failed" in feedback[0]
        assert controller.state is RunState.VERIFIED
