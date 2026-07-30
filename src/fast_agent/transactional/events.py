from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from fast_agent.transactional.models import (
        JsonValue,
        RunId,
        ToolCallId,
        ToolEffect,
        TransactionId,
    )


class ToolEventKind(StrEnum):
    """Stable event names persisted by the transactional event store."""

    PROPOSED = "tool.proposed"
    VALIDATED = "tool.validated"
    AUTHORIZED = "tool.authorized"
    CHECKPOINTED = "tool.checkpointed"
    EXECUTION_STARTED = "tool.execution_started"
    RESULT_STORED = "tool.result_stored"
    COMMITTED = "tool.committed"
    VALIDATION_FAILED = "tool.validation_failed"
    DENIED = "tool.denied"
    EXECUTION_FAILED = "tool.execution_failed"
    ROLLBACK_STARTED = "tool.rollback_started"
    ROLLED_BACK = "tool.rolled_back"
    FAILED = "tool.failed"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolEventBase:
    """Identity and ordering metadata shared by every tool event."""

    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId
    occurred_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolProposed(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.PROPOSED

    tool_name: str
    arguments: dict[str, JsonValue]
    effect: ToolEffect


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolValidated(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.VALIDATED


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolAuthorized(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.AUTHORIZED


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCheckpointed(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.CHECKPOINTED

    checkpoint_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionStarted(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.EXECUTION_STARTED


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultStored(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.RESULT_STORED

    artifact_id: str
    is_error: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCommitted(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.COMMITTED


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolValidationFailed(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.VALIDATION_FAILED

    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDenied(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.DENIED

    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionFailed(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.EXECUTION_FAILED

    error_type: str
    message: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRollbackStarted(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.ROLLBACK_STARTED

    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRolledBack(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.ROLLED_BACK

    checkpoint_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolFailed(ToolEventBase):
    kind: ClassVar[ToolEventKind] = ToolEventKind.FAILED

    reason: str


type ToolEvent = (
    ToolProposed
    | ToolValidated
    | ToolAuthorized
    | ToolCheckpointed
    | ToolExecutionStarted
    | ToolResultStored
    | ToolCommitted
    | ToolValidationFailed
    | ToolDenied
    | ToolExecutionFailed
    | ToolRollbackStarted
    | ToolRolledBack
    | ToolFailed
)
