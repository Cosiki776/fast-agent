from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class BudgetDimension(StrEnum):
    TOOL_CALLS = "tool_calls"
    LLM_CALLS = "llm_calls"
    WALL_TIME = "wall_time"
    ARTIFACT_OUTPUT_BYTES = "artifact_output_bytes"
    TOKENS = "tokens"
    RECOVERY_ATTEMPTS = "recovery_attempts"


@dataclass(frozen=True, slots=True)
class RunBudgetLimits:
    max_tool_calls: int | None = None
    max_llm_calls: int | None = None
    max_wall_time_seconds: float | None = None
    max_artifact_output_bytes: int | None = None
    max_tokens: int | None = None
    max_recovery_attempts: int | None = None
    soft_limit_ratio: float = 0.8

    def __post_init__(self) -> None:
        values = (
            self.max_tool_calls,
            self.max_llm_calls,
            self.max_wall_time_seconds,
            self.max_artifact_output_bytes,
            self.max_tokens,
            self.max_recovery_attempts,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("budget limits must be non-negative")
        if not 0 < self.soft_limit_ratio < 1:
            raise ValueError("soft_limit_ratio must be between zero and one")


@dataclass(frozen=True, slots=True)
class RunBudgetSnapshot:
    tool_calls: int
    llm_calls: int
    elapsed_seconds: float
    artifact_output_bytes: int
    total_tokens: int | None
    recovery_attempts: int


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    exhausted: tuple[BudgetDimension, ...] = ()
    warnings: tuple[BudgetDimension, ...] = ()


class RunBudgetTracker:
    """Track one run's shared resource usage and gate its next controlled action."""

    def __init__(
        self,
        limits: RunBudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._started_at = clock()
        self._tool_calls = 0
        self._llm_calls = 0
        self._artifact_output_bytes = 0
        self._total_tokens = 0
        self._token_usage_complete = True
        self._recovery_attempts = 0

    @property
    def limits(self) -> RunBudgetLimits:
        return self._limits

    @property
    def snapshot(self) -> RunBudgetSnapshot:
        return RunBudgetSnapshot(
            tool_calls=self._tool_calls,
            llm_calls=self._llm_calls,
            elapsed_seconds=self._clock() - self._started_at,
            artifact_output_bytes=self._artifact_output_bytes,
            total_tokens=self._total_tokens if self._token_usage_complete else None,
            recovery_attempts=self._recovery_attempts,
        )

    def start_tool_call(self) -> BudgetDecision:
        dimensions = (
            BudgetDimension.TOOL_CALLS,
            BudgetDimension.ARTIFACT_OUTPUT_BYTES,
            BudgetDimension.WALL_TIME,
        )
        decision = self._before_action(dimensions)
        if not decision.allowed:
            return decision
        self._tool_calls += 1
        return BudgetDecision(allowed=True, warnings=self._warnings(dimensions))

    def record_artifact_output(self, byte_count: int) -> tuple[BudgetDimension, ...]:
        if byte_count < 0:
            raise ValueError("artifact output byte count must be non-negative")
        self._artifact_output_bytes += byte_count
        return self._warnings((BudgetDimension.ARTIFACT_OUTPUT_BYTES,))

    def start_llm_call(self) -> BudgetDecision:
        dimensions = (
            BudgetDimension.LLM_CALLS,
            BudgetDimension.TOKENS,
            BudgetDimension.WALL_TIME,
        )
        decision = self._before_action(dimensions)
        if not decision.allowed:
            return decision
        self._llm_calls += 1
        return BudgetDecision(allowed=True, warnings=self._warnings(dimensions))

    def record_token_usage(self, total_tokens: int | None) -> tuple[BudgetDimension, ...]:
        if total_tokens is None:
            self._token_usage_complete = False
            return ()
        if total_tokens < 0:
            raise ValueError("token usage must be non-negative")
        self._total_tokens += total_tokens
        return self._warnings((BudgetDimension.TOKENS,))

    def start_recovery(self) -> BudgetDecision:
        dimensions = (BudgetDimension.RECOVERY_ATTEMPTS,)
        decision = self._before_action(dimensions)
        if not decision.allowed:
            return decision
        self._recovery_attempts += 1
        return BudgetDecision(allowed=True, warnings=self._warnings(dimensions))

    def check_wall_time(self) -> BudgetDecision:
        return self._before_action((BudgetDimension.WALL_TIME,))

    def remaining_wall_time_seconds(self) -> float | None:
        limit = self._limits.max_wall_time_seconds
        if limit is None:
            return None
        return max(0.0, limit - self.snapshot.elapsed_seconds)

    def _before_action(self, dimensions: Iterable[BudgetDimension]) -> BudgetDecision:
        checked = tuple(dimensions)
        exhausted = tuple(dimension for dimension in checked if self._hard_limit_reached(dimension))
        if exhausted:
            return BudgetDecision(allowed=False, exhausted=exhausted)
        return BudgetDecision(allowed=True, warnings=self._warnings(checked))

    def _warnings(
        self,
        dimensions: Iterable[BudgetDimension],
    ) -> tuple[BudgetDimension, ...]:
        return tuple(dimension for dimension in dimensions if self._soft_limit_reached(dimension))

    def _hard_limit_reached(self, dimension: BudgetDimension) -> bool:
        limit = self._limit(dimension)
        usage = self._usage(dimension)
        return limit is not None and usage is not None and usage >= limit

    def _soft_limit_reached(self, dimension: BudgetDimension) -> bool:
        limit = self._limit(dimension)
        usage = self._usage(dimension)
        return (
            limit is not None
            and limit > 0
            and usage is not None
            and usage >= limit * self._limits.soft_limit_ratio
            and usage <= limit
        )

    def _limit(self, dimension: BudgetDimension) -> int | float | None:
        match dimension:
            case BudgetDimension.TOOL_CALLS:
                return self._limits.max_tool_calls
            case BudgetDimension.LLM_CALLS:
                return self._limits.max_llm_calls
            case BudgetDimension.WALL_TIME:
                return self._limits.max_wall_time_seconds
            case BudgetDimension.ARTIFACT_OUTPUT_BYTES:
                return self._limits.max_artifact_output_bytes
            case BudgetDimension.TOKENS:
                return self._limits.max_tokens
            case BudgetDimension.RECOVERY_ATTEMPTS:
                return self._limits.max_recovery_attempts

    def _usage(self, dimension: BudgetDimension) -> int | float | None:
        snapshot = self.snapshot
        match dimension:
            case BudgetDimension.TOOL_CALLS:
                return snapshot.tool_calls
            case BudgetDimension.LLM_CALLS:
                return snapshot.llm_calls
            case BudgetDimension.WALL_TIME:
                return snapshot.elapsed_seconds
            case BudgetDimension.ARTIFACT_OUTPUT_BYTES:
                return snapshot.artifact_output_bytes
            case BudgetDimension.TOKENS:
                return snapshot.total_tokens
            case BudgetDimension.RECOVERY_ATTEMPTS:
                return snapshot.recovery_attempts
