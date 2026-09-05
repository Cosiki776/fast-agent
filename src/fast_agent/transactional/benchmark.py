from __future__ import annotations

import hashlib
import json
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fast_agent.transactional.verification import VerificationSpec  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


type NonNegativeInt = Annotated[int, Field(ge=0)]


class BenchmarkBudgetSpec(BaseModel):
    max_llm_calls: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class BenchmarkTaskManifest(BaseModel):
    id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    controlled_command: str | None = Field(default=None, min_length=1)
    verification: VerificationSpec
    budget: BenchmarkBudgetSpec

    model_config = ConfigDict(frozen=True, extra="forbid")


class BenchmarkProfile(StrEnum):
    BASELINE = "baseline"
    REDUCER = "reducer"


class BenchmarkObservation(BaseModel):
    """Metrics supplied by the PR6 development harness for one execution."""

    verification_command_passed: bool
    llm_calls: NonNegativeInt
    tool_calls: NonNegativeInt
    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    full_output_bytes: NonNegativeInt
    model_visible_output_bytes: NonNegativeInt
    reducer_latency_seconds: Annotated[float, Field(ge=0)] | None

    model_config = ConfigDict(frozen=True, extra="forbid")


class BenchmarkRunMetrics(BenchmarkObservation):
    profile: BenchmarkProfile
    task_manifest_sha256: str
    wall_time_seconds: Annotated[float, Field(ge=0)]


class BenchmarkABResult(BaseModel):
    task_id: str
    baseline: BenchmarkRunMetrics
    reducer: BenchmarkRunMetrics

    model_config = ConfigDict(frozen=True, extra="forbid")


type BenchmarkExecutor = Callable[
    [BenchmarkTaskManifest, BenchmarkProfile],
    Awaitable[BenchmarkObservation],
]


class TransactionalBenchmarkRunner:
    """Run the PR6 development-only baseline/reducer smoke comparison."""

    def __init__(
        self,
        executor: BenchmarkExecutor,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._executor = executor
        self._clock = clock

    async def run_ab(self, task: BenchmarkTaskManifest) -> BenchmarkABResult:
        manifest_hash = task_manifest_sha256(task)
        baseline = await self._run(task, BenchmarkProfile.BASELINE, manifest_hash)
        reducer = await self._run(task, BenchmarkProfile.REDUCER, manifest_hash)
        return BenchmarkABResult(task_id=task.id, baseline=baseline, reducer=reducer)

    async def _run(
        self,
        task: BenchmarkTaskManifest,
        profile: BenchmarkProfile,
        manifest_hash: str,
    ) -> BenchmarkRunMetrics:
        started_at = self._clock()
        observation = await self._executor(task, profile)
        elapsed = self._clock() - started_at
        return BenchmarkRunMetrics(
            **observation.model_dump(),
            profile=profile,
            task_manifest_sha256=manifest_hash,
            wall_time_seconds=elapsed,
        )


def load_task_manifest(path: Path) -> BenchmarkTaskManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark task manifest must be a mapping: {path}")
    return BenchmarkTaskManifest.model_validate(payload)


def load_task_manifests(directory: Path) -> list[BenchmarkTaskManifest]:
    return [load_task_manifest(path) for path in sorted(directory.glob("*.yaml"))]


def task_manifest_sha256(task: BenchmarkTaskManifest) -> str:
    payload = json.dumps(
        task.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
