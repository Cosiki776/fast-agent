from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from fast_agent.transactional.run_events import RunFailed, RunState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.transactional.budget import RunBudgetTracker
    from fast_agent.transactional.models import RunId
    from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore

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
    ) -> None:
        self.run_id = run_id
        self._run_events = run_events
        self._budget = budget

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
            return await call()
        except Exception as exc:
            self.fail(f"agent turn failed: {type(exc).__name__}: {exc}")
            raise

    def fail(self, reason: str) -> None:
        if self.state is not RunState.FAILED:
            self._run_events.append(RunFailed(run_id=self.run_id, reason=reason))
