from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from fast_agent.transactional.benchmark import load_task_manifests

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = Path(__file__).with_name("tasks")
SOURCES_ROOT = Path(__file__).with_name("fixture_sources")
GENERATED_ROOT = REPOSITORY_ROOT / ".txagent-fixtures"
_FIXED_GIT_ENV = {
    "GIT_AUTHOR_NAME": "TxAgent Fixture",
    "GIT_AUTHOR_EMAIL": "txagent-fixture@example.invalid",
    "GIT_AUTHOR_DATE": "2026-08-24T00:00:00+00:00",
    "GIT_COMMITTER_NAME": "TxAgent Fixture",
    "GIT_COMMITTER_EMAIL": "txagent-fixture@example.invalid",
    "GIT_COMMITTER_DATE": "2026-08-24T00:00:00+00:00",
}


def prepare_fixtures(generated_root: Path = GENERATED_ROOT) -> dict[str, str]:
    generated_root = generated_root.resolve()
    generated_root.mkdir(exist_ok=True)
    prepared: dict[str, str] = {}
    for task in load_task_manifests(TASKS_ROOT):
        manifest_destination = (REPOSITORY_ROOT / task.repository).resolve()
        if manifest_destination.parent != GENERATED_ROOT.resolve():
            raise ValueError(f"Fixture manifest must target {GENERATED_ROOT}")
        destination = generated_root / manifest_destination.name
        source = SOURCES_ROOT / destination.name
        if not source.is_dir():
            raise FileNotFoundError(f"Fixture source does not exist: {source}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        _git(destination, "init", "--initial-branch=main", "-q")
        _git(destination, "add", ".")
        _git(destination, "commit", "-q", "-m", "test: add transactional development fixture")
        base_commit = _git(destination, "rev-parse", "HEAD")
        if base_commit != task.base_commit:
            raise RuntimeError(f"{task.id} generated {base_commit}, expected {task.base_commit}")
        prepared[task.id] = base_commit
        print(f"prepared {task.id}: {base_commit}")
    return prepared


def main() -> None:
    prepare_fixtures()


def _git(root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.update(_FIXED_GIT_ENV)
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


if __name__ == "__main__":
    main()
