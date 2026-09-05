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
    VERIFICATION_STARTED = "run.verification_started"
    VERIFICATION_FAILED = "run.verification_failed"
    VERIFIED = "run.verified"
    AGENT_COMPLETED = "run.agent_completed"
    PROMOTION_APPLIED = "promotion.applied"
    PROMOTION_REJECTED = "promotion.rejected"
    FAILED = "run.failed"


class RunState(StrEnum):
    ACTIVE = "active"
    RECOVERING = "recovering"
    VERIFYING = "verifying"
    VERIFICATION_FAILED = "verification_failed"
    VERIFIED = "verified"
    AGENT_COMPLETED = "agent_completed"
    PROMOTED = "promoted"
    PROMOTION_REJECTED = "promotion_rejected"
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
class RunVerificationStarted(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.VERIFICATION_STARTED
    command: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunVerificationFailed(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.VERIFICATION_FAILED
    exit_code: int
    timed_out: bool
    stdout_artifact_id: str
    stderr_artifact_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunVerified(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.VERIFIED
    workspace_version: str
    stdout_artifact_id: str
    stderr_artifact_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunAgentCompleted(RunEventBase):
    """Agent finished without an independent verification command."""

    kind: ClassVar[RunEventKind] = RunEventKind.AGENT_COMPLETED
    workspace_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionApplied(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.PROMOTION_APPLIED
    workspace_version: str
    patch_artifact_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionRejected(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.PROMOTION_REJECTED
    reason: str
    patch_artifact_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(RunEventBase):
    kind: ClassVar[RunEventKind] = RunEventKind.FAILED
    reason: str


type RunEvent = (
    RunStarted
    | RunRecoveryStarted
    | RunRecovered
    | RunVerificationStarted
    | RunVerificationFailed
    | RunVerified
    | RunAgentCompleted
    | PromotionApplied
    | PromotionRejected
    | RunFailed
)


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
        if state in {RunState.FAILED, RunState.PROMOTED, RunState.PROMOTION_REJECTED}:
            raise ValueError(f"A {state.value} run cannot accept more events")
        if isinstance(event, RunRecoveryStarted):
            if state is not RunState.ACTIVE:
                raise ValueError("Recovery can only start from an active run")
            state = RunState.RECOVERING
        elif isinstance(event, RunRecovered):
            if state is not RunState.RECOVERING:
                raise ValueError("run.recovered requires an active recovery")
            state = RunState.ACTIVE
        elif isinstance(event, RunVerificationStarted):
            if state not in {RunState.ACTIVE, RunState.VERIFICATION_FAILED}:
                raise ValueError("Verification can only start from an active run")
            state = RunState.VERIFYING
        elif isinstance(event, RunVerificationFailed):
            if state is not RunState.VERIFYING:
                raise ValueError("run.verification_failed requires active verification")
            state = RunState.VERIFICATION_FAILED
        elif isinstance(event, RunVerified):
            if state is not RunState.VERIFYING:
                raise ValueError("run.verified requires active verification")
            state = RunState.VERIFIED
        elif isinstance(event, RunAgentCompleted):
            if state is not RunState.ACTIVE:
                raise ValueError("run.agent_completed requires an active run")
            state = RunState.AGENT_COMPLETED
        elif isinstance(event, PromotionApplied):
            if state not in {RunState.VERIFIED, RunState.AGENT_COMPLETED}:
                raise ValueError("promotion.applied requires a completed run")
            state = RunState.PROMOTED
        elif isinstance(event, PromotionRejected):
            if state not in {RunState.VERIFIED, RunState.AGENT_COMPLETED}:
                raise ValueError("promotion.rejected requires a completed run")
            state = RunState.PROMOTION_REJECTED
        elif isinstance(event, RunFailed):
            state = RunState.FAILED
        else:
            raise ValueError("run.started may only occur once")
    return RunProjection(run_id=run_id, state=state, event_count=len(events))
