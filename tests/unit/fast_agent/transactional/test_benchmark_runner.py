from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fast_agent.transactional.benchmark import (
    BenchmarkObservation,
    BenchmarkProfile,
    BenchmarkTaskManifest,
    TransactionalBenchmarkRunner,
    load_task_manifest,
    load_task_manifests,
)

TASKS = Path("benchmarks/transactional/tasks")


def test_development_task_manifests_are_frozen_and_complete() -> None:
    tasks = load_task_manifests(TASKS)

    assert [task.id for task in tasks] == [
        "boundary-check-001",
        "default-value-001",
        "missing-await-001",
    ]
    assert {task.budget.max_llm_calls for task in tasks} == {20}
    assert {task.budget.max_tool_calls for task in tasks} == {80}
    assert all(task.verification.command == "uv run pytest -q" for task in tasks)
    with pytest.raises(ValidationError, match="frozen"):
        tasks[0].issue = "changed"  # ty: ignore[invalid-assignment]  # frozen mutation is under test


def test_manifest_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text(
        """
id: invalid
repository: fixtures/repository
base_commit: deadbee
issue: Invalid task
verification:
  command: uv run pytest -q
  timeout_seconds: 120
budget:
  max_llm_calls: 2
  max_tool_calls: 4
unexpected: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_task_manifest(manifest)


@pytest.mark.asyncio
async def test_ab_runner_uses_same_manifest_and_records_required_metrics() -> None:
    task = load_task_manifest(TASKS / "boundary-check-001.yaml")
    calls: list[tuple[BenchmarkTaskManifest, BenchmarkProfile]] = []
    now = 10.0

    def clock() -> float:
        return now

    async def execute(
        received_task: BenchmarkTaskManifest,
        profile: BenchmarkProfile,
    ) -> BenchmarkObservation:
        nonlocal now
        calls.append((received_task, profile))
        now += 2.0 if profile is BenchmarkProfile.BASELINE else 1.0
        return BenchmarkObservation(
            verification_command_passed=True,
            llm_calls=3,
            tool_calls=5,
            input_tokens=None,
            output_tokens=None,
            full_output_bytes=20_000,
            model_visible_output_bytes=(20_000 if profile is BenchmarkProfile.BASELINE else 800),
            reducer_latency_seconds=(None if profile is BenchmarkProfile.BASELINE else 0.002),
        )

    result = await TransactionalBenchmarkRunner(execute, clock=clock).run_ab(task)

    assert calls == [
        (task, BenchmarkProfile.BASELINE),
        (task, BenchmarkProfile.REDUCER),
    ]
    assert calls[0][0] is calls[1][0]
    assert result.baseline.task_manifest_sha256 == result.reducer.task_manifest_sha256
    assert result.baseline.wall_time_seconds == 2.0
    assert result.reducer.wall_time_seconds == 1.0
    assert result.baseline.model_visible_output_bytes == 20_000
    assert result.reducer.model_visible_output_bytes == 800
    assert result.baseline.reducer_latency_seconds is None
    assert result.reducer.reducer_latency_seconds == 0.002
    assert result.baseline.input_tokens is None
    assert result.reducer.output_tokens is None

    payload = json.loads(result.model_dump_json())
    assert payload["baseline"]["verification_command_passed"] is True
    assert "run.verified" not in result.model_dump_json()
