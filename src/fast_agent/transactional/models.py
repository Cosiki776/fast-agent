from __future__ import annotations

from enum import StrEnum
from typing import NewType
from uuid import uuid4

# Keep persistence values string-compatible while preventing accidental ID mixing
# at typed boundaries.
RunId = NewType("RunId", str)
TransactionId = NewType("TransactionId", str)
ToolCallId = NewType("ToolCallId", str)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def new_run_id() -> RunId:
    """Create an opaque identifier for one transactional coding run."""
    return RunId(uuid4().hex)


def new_transaction_id() -> TransactionId:
    """Create an opaque identifier for one governed tool call."""
    return TransactionId(uuid4().hex)


class ToolEffect(StrEnum):
    """Side-effect class used by transactional coding policy."""

    READ = "read"
    WORKSPACE_WRITE = "workspace_write"
    EXTERNAL_UNKNOWN = "external_unknown"


class TransactionState(StrEnum):
    """Projected lifecycle state for one transactional tool call."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    CHECKPOINTED = "checkpointed"
    CHECKPOINT_FAILED = "checkpoint_failed"
    EXECUTING = "executing"
    RESULT_STORED = "result_stored"
    COMMITTED = "committed"
    VALIDATION_FAILED = "validation_failed"
    DENIED = "denied"
    EXECUTION_FAILED = "execution_failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
