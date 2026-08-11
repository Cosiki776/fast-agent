from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from fast_agent.transactional.models import RunId


class RunEventKind(StrEnum):
    STARTED = "run.started"
    RECOVERY_STARTED = "run.recovery_started"
    RECOVERED = "run.recovered"
    FAILED = "run.failed"


class RunState(StrEnum):
    ACTIVE = "active"
    RECOVERING = "recovering"
    FAILED = "failed"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunEventBase:
    run_id: RunId
    occurred_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.STARTED
    profile: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecoveryStarted(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.RECOVERY_STARTED
    failure_signature: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecovered(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.RECOVERED
    workspace_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.FAILED
    reason: str


type RunEvent = RunStarted | RunRecoveryStarted | RunRecovered | RunFailed


@dataclass(frozen=True, slots=True)
class RunProjection:
    run_id: RunId
    state: RunState
    event_count: int


def replay_run(events: list[RunEvent]) -> RunProjection:
    if not events or not isinstance(events[0], RunStarted):
        raise ValueError("Run event stream must start with run.started")
    state = RunState.ACTIVE
    run_id = events[0].run_id
    for event in events[1:]:
        if event.run_id != run_id:
            raise ValueError("Run event stream contains mixed run IDs")
        if state is RunState.FAILED:
            raise ValueError("A failed run cannot accept more events")
        if isinstance(event, RunRecoveryStarted):
            if state is not RunState.ACTIVE:
                raise ValueError("Recovery can only start from an active run")
            state = RunState.RECOVERING
        elif isinstance(event, RunRecovered):
            if state is not RunState.RECOVERING:
                raise ValueError("run.recovered requires an active recovery")
            state = RunState.ACTIVE
        elif isinstance(event, RunFailed):
            state = RunState.FAILED
        else:
            raise ValueError("run.started may only occur once")
    return RunProjection(run_id=run_id, state=state, event_count=len(events))
