from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.budget import RunBudgetLimits, RunBudgetTracker
from fast_agent.transactional.models import RunId
from fast_agent.transactional.run_controller import (
    TransactionalCodingRun,
    TransactionalRunTerminatedError,
)
from fast_agent.transactional.run_events import RunStarted, RunState
from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore

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
