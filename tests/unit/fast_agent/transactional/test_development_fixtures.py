from __future__ import annotations

import subprocess
from pathlib import Path

from benchmarks.transactional.prepare_development_fixtures import prepare_fixtures
from fast_agent.transactional.benchmark import load_task_manifests

TASKS_ROOT = Path("benchmarks/transactional/tasks")


def test_development_fixtures_have_reproducible_failing_baselines(tmp_path: Path) -> None:
    generated_root = tmp_path / "fixtures"
    prepared = prepare_fixtures(generated_root)
    tasks = load_task_manifests(TASKS_ROOT)

    assert prepared == {task.id: task.base_commit for task in tasks}
    for task in tasks:
        workspace = generated_root / Path(task.repository).name
        verification = subprocess.run(
            task.verification.command,
            cwd=workspace,
            shell=True,
            check=False,
            capture_output=True,
            timeout=task.verification.timeout_seconds,
        )
        assert verification.returncode != 0, task.id
