from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict

from fast_agent import FastAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.transactional.benchmark import load_task_manifest, task_manifest_sha256
from fast_agent.transactional.events import ToolResultStored
from fast_agent.transactional.settings import TransactionalProfile
from fast_agent.transactional.storage.artifact_store import ArtifactId
from fast_agent.types import RequestParams
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import CallToolResult

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.transactional.assembly import TransactionalRuntime
    from fast_agent.transactional.benchmark import BenchmarkTaskManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FAST_AGENT_HOME = REPOSITORY_ROOT / ".fast-agent"


@dataclass(frozen=True, slots=True)
class ControlledCommandMetrics:
    observed: bool
    tool_result_bytes_before_reducer: int | None = None
    model_visible_result_bytes: int | None = None


class PilotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    manifest_sha256: str
    profile: TransactionalProfile
    model: str
    base_commit: str
    run_id: str | None
    verification_passed: bool
    verification_exit_code: int
    verification_stdout_bytes: int
    verification_stderr_bytes: int
    verification_output_sha256: str
    llm_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    artifact_output_bytes: int
    model_visible_tool_result_bytes: int
    controlled_command: str | None
    controlled_command_observed: bool | None
    controlled_tool_result_bytes_before_reducer: int | None
    controlled_model_visible_result_bytes: int | None
    recovery_attempts: int
    wall_time_seconds: float
    run_events: tuple[str, ...]
    promotion_result: str | None
    response: str


async def run(args: argparse.Namespace) -> PilotResult:
    manifest_path = Path(args.manifest).resolve()
    task = load_task_manifest(manifest_path)
    profile = TransactionalProfile(args.profile)
    source = (REPOSITORY_ROOT / task.repository).resolve()
    _validate_fixture(source, task.base_commit)

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    output.mkdir(parents=True)
    workspace = output / "workspace"
    _git(REPOSITORY_ROOT, "clone", "--quiet", str(source), str(workspace))
    _git(workspace, "checkout", "--quiet", "-B", "txagent-pilot-base", task.base_commit)

    config_path = FAST_AGENT_HOME / f"pr11-pilot-{os.getpid()}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_model": args.model,
                "session_history": False,
                "shell_execution": {
                    "tool_profile": "native",
                    "write_text_file_mode": "on",
                    "interactive_use_pty": False,
                    "show_bash": False,
                },
                "transactional": {
                    "profile": profile.value,
                    "runtime_root": str(output / "runtime"),
                    "keep_worktree": True,
                    "max_llm_calls": task.budget.max_llm_calls,
                    "max_tool_calls": task.budget.max_tool_calls,
                    "max_wall_time_seconds": 600,
                    "max_recovery_attempts": 2,
                    "verification": (
                        task.verification.model_dump(mode="json")
                        if profile is TransactionalProfile.FULL
                        else None
                    ),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    fast = FastAgent(
        "TxAgent PR11 development Pilot",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
        home=FAST_AGENT_HOME,
        workspace=workspace,
    )

    @fast.agent(
        name="pilot",
        model=args.model,
        instruction=(
            "Work as a coding agent inside the provided repository. Inspect the code and tests, "
            "make the smallest correct change, run relevant tests, and do not merely describe a fix."
        ),
        shell=True,
        default=True,
    )
    async def pilot_agent() -> None:
        pass

    started_at = time.perf_counter()
    try:
        async with fast.harness() as harness:
            session = await harness.session(f"pr11-{task.id}-{profile.value}", agent_name="pilot")
            response = await session.send(
                _task_prompt(task),
                request_params=RequestParams(parallel_tool_calls=False),
            )
            agent = session.agent_app["pilot"]
            if not isinstance(agent, McpAgent):
                raise TypeError("Development Pilot requires an MCP coding agent")
            usage = agent.usage_accumulator
            usage_summary = usage.summary if usage is not None else None
            runtime = session.transactional_runtime
            if runtime is None:
                run_id = None
                run_events: tuple[str, ...] = ()
                artifact_output_bytes = 0
                recovery_attempts = 0
            else:
                run_id = str(runtime.run_id)
                snapshot = runtime.budget.snapshot
                artifact_output_bytes = snapshot.artifact_output_bytes
                recovery_attempts = snapshot.recovery_attempts
                run_events = (
                    tuple(
                        item.event.kind.value
                        for item in runtime.run_event_store.events_for_run(runtime.run_id)
                    )
                    if runtime.run_event_store is not None
                    else ()
                )
            model_visible_bytes = _model_visible_tool_result_bytes(agent)
            controlled = _controlled_command_metrics(
                agent.message_history,
                task.controlled_command,
                runtime,
            )
    finally:
        config_path.unlink(missing_ok=True)

    verification = subprocess.run(
        task.verification.command,
        cwd=workspace,
        shell=True,
        check=False,
        capture_output=True,
        timeout=task.verification.timeout_seconds,
    )
    stdout = verification.stdout
    stderr = verification.stderr
    output.joinpath("verification.stdout").write_bytes(stdout)
    output.joinpath("verification.stderr").write_bytes(stderr)
    verification_payload = stdout + b"\0" + stderr
    result = PilotResult(
        task_id=task.id,
        manifest_sha256=task_manifest_sha256(task),
        profile=profile,
        model=args.model,
        base_commit=task.base_commit,
        run_id=run_id,
        verification_passed=verification.returncode == 0,
        verification_exit_code=verification.returncode,
        verification_stdout_bytes=len(stdout),
        verification_stderr_bytes=len(stderr),
        verification_output_sha256=hashlib.sha256(verification_payload).hexdigest(),
        llm_calls=usage_summary.provider_attempts if usage_summary is not None else 0,
        tool_calls=usage_summary.tool_calls if usage_summary is not None else 0,
        input_tokens=(usage_summary.prompt.total if usage_summary is not None else None),
        output_tokens=(usage_summary.completion.total if usage_summary is not None else None),
        artifact_output_bytes=artifact_output_bytes,
        model_visible_tool_result_bytes=model_visible_bytes,
        controlled_command=task.controlled_command,
        controlled_command_observed=(controlled.observed if controlled is not None else None),
        controlled_tool_result_bytes_before_reducer=(
            controlled.tool_result_bytes_before_reducer if controlled is not None else None
        ),
        controlled_model_visible_result_bytes=(
            controlled.model_visible_result_bytes if controlled is not None else None
        ),
        recovery_attempts=recovery_attempts,
        wall_time_seconds=time.perf_counter() - started_at,
        run_events=run_events,
        promotion_result=_promotion_result(run_events),
        response=_sanitize(response, workspace),
    )
    output.joinpath("result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return result


def _task_prompt(task: BenchmarkTaskManifest) -> str:
    if task.controlled_command is None:
        return task.issue
    return (
        f"{task.issue}\n\n"
        "Controlled-output requirement: before editing the implementation, run this exact "
        f"command once:\n\n    {task.controlled_command}\n\n"
        "Run it exactly as written, without pipes, redirection, output filtering, or additional "
        "arguments. Other investigation commands remain unrestricted."
    )


def _model_visible_tool_result_bytes(agent: McpAgent) -> int:
    total = 0
    for message in agent.message_history:
        if message.tool_results is None:
            continue
        total += sum(
            len(result.model_dump_json(by_alias=True, exclude_none=True).encode())
            for result in message.tool_results.values()
        )
    return total


def _controlled_command_metrics(
    history: Sequence[PromptMessageExtended],
    command: str | None,
    runtime: TransactionalRuntime | None,
) -> ControlledCommandMetrics | None:
    if command is None:
        return None
    matched = _find_controlled_result(history, command)
    if matched is None:
        return ControlledCommandMetrics(observed=False)
    call_id, result = matched
    visible_bytes = _serialized_result_bytes(result)
    before_reducer_bytes = _tool_result_bytes_before_reducer(runtime, call_id, visible_bytes)
    return ControlledCommandMetrics(
        observed=True,
        tool_result_bytes_before_reducer=before_reducer_bytes,
        model_visible_result_bytes=visible_bytes,
    )


def _find_controlled_result(
    history: Sequence[PromptMessageExtended],
    command: str,
) -> tuple[str, CallToolResult] | None:
    results = {
        call_id: result
        for message in history
        for call_id, result in (message.tool_results or {}).items()
    }
    for message in history:
        for call_id, request in (message.tool_calls or {}).items():
            arguments = request.params.arguments
            if (
                is_shell_command_tool_name(request.params.name)
                and isinstance(arguments, dict)
                and arguments.get("command") == command
                and call_id in results
            ):
                return call_id, results[call_id]
    return None


def _tool_result_bytes_before_reducer(
    runtime: TransactionalRuntime | None,
    call_id: str,
    visible_bytes: int,
) -> int:
    if runtime is None:
        return visible_bytes
    for item in runtime.event_store.events_for_run(runtime.run_id):
        event = item.event
        if isinstance(event, ToolResultStored) and str(event.tool_call_id) == call_id:
            return runtime.artifact_store.metadata(ArtifactId(event.artifact_id)).byte_size
    raise RuntimeError(f"Controlled command artifact was not found: {call_id}")


def _serialized_result_bytes(result: CallToolResult) -> int:
    return len(result.model_dump_json(by_alias=True, exclude_none=True).encode())


def _promotion_result(events: tuple[str, ...]) -> str | None:
    if "promotion.applied" in events:
        return "applied"
    if "promotion.rejected" in events:
        return "rejected"
    return None


def _sanitize(text: str, workspace: Path) -> str:
    return text.replace(str(workspace), "<workspace>").replace(
        str(REPOSITORY_ROOT),
        "<fast-agent>",
    )


def _validate_fixture(source: Path, base_commit: str) -> None:
    if source.parent != (REPOSITORY_ROOT / ".txagent-fixtures").resolve():
        raise ValueError("Pilot fixture must be inside .txagent-fixtures")
    if _git(source, "rev-parse", "HEAD") != base_commit:
        raise ValueError(f"Fixture HEAD does not match manifest base_commit: {source}")
    if _git(source, "status", "--porcelain"):
        raise ValueError(f"Fixture repository is dirty: {source}")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=[item.value for item in TransactionalProfile],
    )
    parser.add_argument("--model", default="aliyun.qwen3.8-max")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        output = Path(args.output).resolve()
        if output.is_dir() and not isinstance(exc, FileExistsError):
            output.joinpath("failure.json").write_text(
                json.dumps(
                    {"error_type": type(exc).__name__, "message": str(exc)},
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(result.model_dump_json(indent=2))
    if result.controlled_command_observed is False:
        print("Controlled command was not observed exactly as specified", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
