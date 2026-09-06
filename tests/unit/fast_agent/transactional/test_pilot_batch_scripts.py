from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "benchmarks" / "transactional"
STUB_UV = """
import json
import os
import sys
from pathlib import Path

if sys.argv[1:3] == ["run", "python"]:
    os.execv(sys.executable, [sys.executable, *sys.argv[3:]])
assert sys.argv[1:3] == ["run", "benchmarks/transactional/run_development_pilot.py"]
args = sys.argv[3:]
options = dict(zip(args[::2], args[1::2], strict=True))
calls = Path(os.environ["PILOT_STUB_CALLS"])
first = not calls.exists()
with calls.open("a") as stream:
    stream.write(json.dumps(options) + "\\n")
failure = os.environ["PILOT_STUB_FAILURE"] if first else "none"
if failure == "missing_result":
    raise SystemExit(1)
output = Path(options["--output"])
output.mkdir()
result = {
    "status": "completed" if failure == "none" else "failed",
    "verification_passed": failure == "none",
    "tool_output_strategy": options["--tool-output-strategy"],
    "error_type": "PilotProviderError" if failure == "provider" else "RuntimeError",
}
if failure == "cancelled":
    result["status"] = "cancelled"
(output / "result.json").write_text(json.dumps(result))
print("scripted run")
raise SystemExit(0 if failure == "none" else 1)
"""


@pytest.mark.parametrize("supplementary", [False, True], ids=["nine-runs", "three-runs"])
@pytest.mark.parametrize("failure", ["none", "task", "provider", "missing_result", "cancelled"])
def test_batch_entry_points_and_failure_handling(
    tmp_path: Path,
    supplementary: bool,
    failure: str,
) -> None:
    root = tmp_path / "project"
    scripts = root / "benchmarks" / "transactional"
    tasks = scripts / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "placeholder.yaml").write_text("fixture for checksum recording\n")
    for name in ("run_development_pilot.sh", "run_full_upstream_pilot.sh"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Pilot Test",
            "-c",
            "user.email=pilot@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "uv"
    stub.write_text(f"#!{sys.executable}\n{STUB_UV}")
    stub.chmod(0o700)
    calls_path = tmp_path / "calls.jsonl"
    entry = "run_full_upstream_pilot.sh" if supplementary else "run_development_pilot.sh"
    completed = subprocess.run(
        ["bash", str(scripts / entry), "scripted-model"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{stub_dir}:{os.environ.get('PATH', '')}",
            "PILOT_STUB_CALLS": str(calls_path),
            "PILOT_STUB_FAILURE": failure,
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    stops = failure in {"provider", "missing_result", "cancelled"}
    assert (completed.returncode != 0) is stops, completed.stdout + completed.stderr
    assert len(calls) == (1 if stops else 3 if supplementary else 9)
    assert all(call["--model"] == "scripted-model" for call in calls)
    packages = {(call["--profile"], call["--tool-output-strategy"]) for call in calls}
    if supplementary:
        assert packages == {("full", "upstream")}
    elif not stops:
        assert packages == {("baseline", "upstream"), ("reducer", "semantic"), ("full", "semantic")}
    if not stops:
        assert len({call["--manifest"] for call in calls}) == 3
    outputs = [Path(call["--output"]) for call in calls]
    assert len(set(outputs)) == len(calls)
    assert all(path.is_relative_to(root / ".txagent-runs") for path in outputs)
    batch = outputs[0].parent
    assert len((batch / "exit_codes.tsv").read_text().splitlines()) == len(calls)
    assert (batch / "implementation_commit.txt").read_text().strip()
    assert (batch / "manifest_checksums.txt").is_file()
    assert all(path.with_suffix(".log").is_file() for path in outputs)
