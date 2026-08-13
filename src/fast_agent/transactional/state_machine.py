from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from fast_agent.transactional.events import ToolEventKind
from fast_agent.transactional.models import TransactionState

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from fast_agent.transactional.events import ToolEvent
    from fast_agent.transactional.models import RunId, ToolCallId, TransactionId

_STATE_BY_EVENT_KIND: Final[dict[ToolEventKind, TransactionState]] = {
    ToolEventKind.PROPOSED: TransactionState.PROPOSED,
    ToolEventKind.VALIDATED: TransactionState.VALIDATED,
    ToolEventKind.AUTHORIZED: TransactionState.AUTHORIZED,
    ToolEventKind.CHECKPOINTED: TransactionState.CHECKPOINTED,
    ToolEventKind.CHECKPOINT_FAILED: TransactionState.CHECKPOINT_FAILED,
    ToolEventKind.EXECUTION_STARTED: TransactionState.EXECUTING,
    ToolEventKind.RESULT_STORED: TransactionState.RESULT_STORED,
    ToolEventKind.COMMITTED: TransactionState.COMMITTED,
    ToolEventKind.VALIDATION_FAILED: TransactionState.VALIDATION_FAILED,
    ToolEventKind.DENIED: TransactionState.DENIED,
    ToolEventKind.EXECUTION_FAILED: TransactionState.EXECUTION_FAILED,
    ToolEventKind.ROLLBACK_STARTED: TransactionState.ROLLING_BACK,
    ToolEventKind.ROLLED_BACK: TransactionState.ROLLED_BACK,
    ToolEventKind.FAILED: TransactionState.FAILED,
}

# The first vertical slice executes directly after proposal. Validation,
# authorization, and checkpointing remain valid optional stages until their
# dedicated runtime components are connected.
_ALLOWED_TRANSITIONS: Final[dict[TransactionState | None, frozenset[TransactionState]]] = {
    None: frozenset({TransactionState.PROPOSED}),
    TransactionState.PROPOSED: frozenset(
        {
            TransactionState.VALIDATED,
            TransactionState.EXECUTING,
            TransactionState.VALIDATION_FAILED,
            TransactionState.DENIED,
        }
    ),
    TransactionState.VALIDATED: frozenset(
        {
            TransactionState.AUTHORIZED,
            TransactionState.DENIED,
        }
    ),
    TransactionState.AUTHORIZED: frozenset(
        {
            TransactionState.CHECKPOINTED,
            TransactionState.CHECKPOINT_FAILED,
            TransactionState.EXECUTING,
        }
    ),
    TransactionState.CHECKPOINTED: frozenset({TransactionState.EXECUTING}),
    TransactionState.CHECKPOINT_FAILED: frozenset({TransactionState.FAILED}),
    TransactionState.EXECUTING: frozenset(
        {
            TransactionState.RESULT_STORED,
            TransactionState.EXECUTION_FAILED,
        }
    ),
    TransactionState.RESULT_STORED: frozenset(
        {
            TransactionState.COMMITTED,
            TransactionState.ROLLING_BACK,
            TransactionState.FAILED,
        }
    ),
    TransactionState.VALIDATION_FAILED: frozenset({TransactionState.FAILED}),
    TransactionState.DENIED: frozenset({TransactionState.FAILED}),
    TransactionState.EXECUTION_FAILED: frozenset(
        {
            TransactionState.ROLLING_BACK,
            TransactionState.FAILED,
        }
    ),
    TransactionState.ROLLING_BACK: frozenset(
        {
            TransactionState.ROLLED_BACK,
            TransactionState.FAILED,
        }
    ),
    TransactionState.ROLLED_BACK: frozenset({TransactionState.FAILED}),
    TransactionState.COMMITTED: frozenset(),
    TransactionState.FAILED: frozenset(),
}


class TransactionStateError(ValueError):
    """Base error for an invalid transaction event stream."""


class InvalidTransitionError(TransactionStateError):
    """Raised when an event cannot follow the current projected state."""

    def __init__(
        self,
        current_state: TransactionState | None,
        next_state: TransactionState,
        event_kind: ToolEventKind,
    ) -> None:
        current_label = current_state.value if current_state is not None else "<none>"
        super().__init__(
            f"Event '{event_kind.value}' cannot transition transaction "
            f"from '{current_label}' to '{next_state.value}'"
        )
        self.current_state = current_state
        self.next_state = next_state
        self.event_kind = event_kind


class TransactionIdentityError(TransactionStateError):
    """Raised when replay mixes events from different tool transactions."""


class EmptyTransactionError(TransactionStateError):
    """Raised when a transaction projection is requested without events."""


@dataclass(frozen=True, slots=True)
class TransactionProjection:
    """Current transaction state derived only from its immutable events."""

    run_id: RunId
    transaction_id: TransactionId
    tool_call_id: ToolCallId
    state: TransactionState
    event_count: int
    last_event_kind: ToolEventKind
    last_occurred_at: datetime


def state_for_event(event: ToolEvent) -> TransactionState:
    """Return the lifecycle state established by an event."""
    return _STATE_BY_EVENT_KIND[event.kind]


def validate_transition(
    current_state: TransactionState | None,
    event: ToolEvent,
) -> TransactionState:
    """Validate an event and return the state it establishes."""
    next_state = state_for_event(event)
    if next_state not in _ALLOWED_TRANSITIONS[current_state]:
        raise InvalidTransitionError(current_state, next_state, event.kind)
    return next_state


def apply_event(
    projection: TransactionProjection | None,
    event: ToolEvent,
) -> TransactionProjection:
    """Apply one event to a projection after validating identity and order."""
    if projection is not None:
        # Report a mixed event stream before interpreting it as a lifecycle error.
        _validate_event_identity(projection, event)

    current_state = projection.state if projection is not None else None
    next_state = validate_transition(current_state, event)

    if projection is None:
        return TransactionProjection(
            run_id=event.run_id,
            transaction_id=event.transaction_id,
            tool_call_id=event.tool_call_id,
            state=next_state,
            event_count=1,
            last_event_kind=event.kind,
            last_occurred_at=event.occurred_at,
        )

    return replace(
        projection,
        state=next_state,
        event_count=projection.event_count + 1,
        last_event_kind=event.kind,
        last_occurred_at=event.occurred_at,
    )


def replay_transaction(events: Iterable[ToolEvent]) -> TransactionProjection:
    """Rebuild a transaction projection from events in store sequence order."""
    projection: TransactionProjection | None = None
    for event in events:
        projection = apply_event(projection, event)

    if projection is None:
        raise EmptyTransactionError("Cannot replay a transaction without events")
    return projection


def _validate_event_identity(
    projection: TransactionProjection,
    event: ToolEvent,
) -> None:
    if event.run_id != projection.run_id:
        raise TransactionIdentityError(
            f"Event run ID '{event.run_id}' does not match transaction run ID '{projection.run_id}'"
        )
    if event.transaction_id != projection.transaction_id:
        raise TransactionIdentityError(
            f"Event transaction ID '{event.transaction_id}' does not match "
            f"'{projection.transaction_id}'"
        )
    if event.tool_call_id != projection.tool_call_id:
        raise TransactionIdentityError(
            f"Event tool call ID '{event.tool_call_id}' does not match '{projection.tool_call_id}'"
        )
