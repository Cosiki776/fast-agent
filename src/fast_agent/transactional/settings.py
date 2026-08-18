from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from fast_agent.transactional.budget import RunBudgetLimits

NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class TransactionalProfile(StrEnum):
    BASELINE = "baseline"
    REDUCER = "reducer"
    FULL = "full"


class TransactionalSettings(BaseModel):
    """Configuration for the constrained transactional coding profiles."""

    model_config = ConfigDict(extra="forbid")

    profile: TransactionalProfile = TransactionalProfile.BASELINE
    mode: Literal["coding"] = "coding"
    runtime_root: str | None = None
    keep_worktree: bool = True
    max_tool_calls: NonNegativeInt | None = 80
    max_llm_calls: NonNegativeInt | None = 20
    max_wall_time_seconds: NonNegativeFloat | None = 1800
    max_artifact_output_bytes: NonNegativeInt | None = 32 * 1024 * 1024
    max_tokens: NonNegativeInt | None = None
    max_recovery_attempts: NonNegativeInt | None = 3
    shell_terminal_timeout_seconds: NonNegativeFloat = 300

    def budget_limits(self) -> RunBudgetLimits:
        return RunBudgetLimits(
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
            max_wall_time_seconds=self.max_wall_time_seconds,
            max_artifact_output_bytes=self.max_artifact_output_bytes,
            max_tokens=self.max_tokens,
            max_recovery_attempts=self.max_recovery_attempts,
        )
