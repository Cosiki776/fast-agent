from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict

from fast_agent import FastAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.transactional.benchmark import (
    BenchmarkBudgetSpec,
    load_task_manifest,
    task_manifest_sha256,
)
from fast_agent.transactional.budget import RunBudgetLimits, RunBudgetTracker
from fast_agent.transactional.events import ToolResultStored
from fast_agent.transactional.run_events import RunState
from fast_agent.transactional.settings import (
    SemanticReducerVersion,
    ToolOutputStrategy,
    TransactionalProfile,
)
from fast_agent.transactional.storage.artifact_store import ArtifactId
from fast_agent.types import RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import CallToolResult

    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.core.harness import HarnessSession
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.transactional.assembly import TransactionalRuntime
    from fast_agent.transactional.benchmark import BenchmarkTaskManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FAST_AGENT_HOME = REPOSITORY_ROOT / ".fast-agent"


class PilotBudgetExceededError(RuntimeError):
    pass


class PilotProviderError(RuntimeError):
    pass


class PilotIncompleteResponseError(RuntimeError):
    pass


class PilotObserver:
    """Apply the same LLM/tool admission limits to every Profile via public hooks.

    Tool calls count admitted planned calls, including calls denied by Full policy.
    A batch that exceeds the remaining allowance is rejected in its entirety,
    before execution. Provider retries are inside one logical LLM call.
    """

    def __init__(self, spec: BenchmarkBudgetSpec) -> None:
        self.spec = spec
        self.budget = RunBudgetTracker(
            RunBudgetLimits(
                max_llm_calls=spec.max_llm_calls,
                max_tool_calls=spec.max_tool_calls,
                max_wall_time_seconds=spec.max_wall_time_seconds,
            )
        )
        # Keep observations independently of history, which can omit an aborted turn.
        self.messages: list[PromptMessageExtended] = []

    def install(self, agent: McpAgent) -> None:
        existing = agent.tool_runner_hooks or ToolRunnerHooks()

        async def before_llm(runner: ToolRunner, messages: list[PromptMessageExtended]) -> None:
            decision = self.budget.start_llm_call()
            if not decision.allowed:
                dimensions = ", ".join(item.value for item in decision.exhausted)
                raise PilotBudgetExceededError(f"Pilot budget exhausted: {dimensions}")
            if existing.before_llm_call is not None:
                await existing.before_llm_call(runner, messages)

        async def after_llm(runner: ToolRunner, message: PromptMessageExtended) -> None:
            self.messages.append(message.model_copy(deep=True))
            if existing.after_llm_call is not None:
                await existing.after_llm_call(runner, message)
            if message.channels and FAST_AGENT_ERROR_CHANNEL in message.channels:
                details = "\n".join(
                    block.text
                    for block in message.channels[FAST_AGENT_ERROR_CHANNEL]
                    if isinstance(block, TextContent)
                )
                raise PilotProviderError(details or message.all_text())
            self._check_wall_time()
            calls = len(message.tool_calls or {})
            remaining = self.spec.max_tool_calls - self.budget.snapshot.tool_calls
            if calls > remaining:
                raise PilotBudgetExceededError("Pilot budget exhausted: tool_calls")
            # Reject at after_llm: before_tool hook errors are converted into tool results.
            if message.stop_reason not in {
                LlmStopReason.TOOL_USE,
                LlmStopReason.END_TURN,
                LlmStopReason.STOP_SEQUENCE,
            }:
                raise PilotIncompleteResponseError(
                    f"Agent did not complete normally: {message.stop_reason}"
                )

        async def before_tools(runner: ToolRunner, message: PromptMessageExtended) -> None:
            if existing.before_tool_call is not None:
                await existing.before_tool_call(runner, message)
            for _ in message.tool_calls or {}:
                decision = self.budget.start_tool_call()
                if not decision.allowed:
                    raise PilotBudgetExceededError(
                        "Pilot budget exhausted: tool_calls or wall_time"
                    )

        async def after_tools(runner: ToolRunner, message: PromptMessageExtended) -> None:
            self.messages.append(message.model_copy(deep=True))
            if existing.after_tool_call is not None:
                await existing.after_tool_call(runner, message)

        agent.tool_runner_hooks = replace(
            existing,
            before_llm_call=before_llm,
            after_llm_call=after_llm,
            before_tool_call=before_tools,
            after_tool_call=after_tools,
        )

    def _check_wall_time(self) -> None:
        if not self.budget.check_wall_time().allowed:
            raise PilotBudgetExceededError("Pilot budget exhausted: wall_time")


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
    tool_output_strategy: ToolOutputStrategy
    semantic_reducer_version: SemanticReducerVersion | None
    model: str
    base_commit: str
    schema_version: int = 2
    implementation_commit: str
    implementation_dirty: bool
    budget: BenchmarkBudgetSpec
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    error_type: str | None = None
    failure_reason: str | None = None
    run_id: str | None = None
    verification_passed: bool | None = None
    verification_exit_code: int | None = None
    verification_timed_out: bool = False
    verification_stdout_bytes: int = 0
    verification_stderr_bytes: int = 0
    verification_output_sha256: str | None = None
    llm_calls: int = 0
    tool_calls: int = 0
    provider_attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    artifact_output_bytes: int = 0
    model_visible_tool_result_bytes: int = 0
    controlled_command: str | None = None
    controlled_command_observed: bool | None = None
    controlled_tool_result_bytes_before_reducer: int | None = None
    controlled_model_visible_result_bytes: int | None = None
    recovery_attempts: int = 0
    wall_time_seconds: float = 0
    run_events: tuple[str, ...] = ()
    tool_events: tuple[str, ...] = ()
    promotion_result: str | None = None
    response: str = ""


async def run(args: argparse.Namespace) -> PilotResult:
    manifest_path = Path(args.manifest).resolve()
    task = load_task_manifest(manifest_path)
    profile = TransactionalProfile(args.profile)
    tool_output_strategy = (
        ToolOutputStrategy.UPSTREAM
        if profile is TransactionalProfile.BASELINE
        else ToolOutputStrategy.SEMANTIC
    )
    semantic_reducer_version = (
        SemanticReducerVersion.V1 if tool_output_strategy is ToolOutputStrategy.SEMANTIC else None
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    output.mkdir(parents=True)
    workspace = output / "workspace"
    result = PilotResult(
        task_id=task.id,
        manifest_sha256=task_manifest_sha256(task),
        profile=profile,
        tool_output_strategy=tool_output_strategy,
        semantic_reducer_version=semantic_reducer_version,
        model=args.model,
        base_commit=task.base_commit,
        implementation_commit=_git(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        implementation_dirty=bool(_git(REPOSITORY_ROOT, "status", "--porcelain")),
        budget=task.budget,
        controlled_command=task.controlled_command,
    )
    started_at = time.perf_counter()
    config_path = FAST_AGENT_HOME / f"pr11-pilot-{os.getpid()}.yaml"
    try:
        source = (REPOSITORY_ROOT / task.repository).resolve()
        _validate_fixture(source, task.base_commit)
        _git(REPOSITORY_ROOT, "clone", "--quiet", str(source), str(workspace))
        _git(workspace, "checkout", "--quiet", "-B", "txagent-pilot-base", task.base_commit)
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
                        "tool_output": {
                            "strategy": tool_output_strategy.value,
                            **(
                                {"semantic_reducer_version": semantic_reducer_version.value}
                                if semantic_reducer_version is not None
                                else {}
                            ),
                        },
                        "runtime_root": str(output / "runtime"),
                        "keep_worktree": True,
                        "max_llm_calls": None,
                        "max_tool_calls": None,
                        "max_wall_time_seconds": None,
                        "max_artifact_output_bytes": None,
                        "max_tokens": None,
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

        async with fast.harness() as harness:
            session = await harness.session(f"pr11-{task.id}-{profile.value}", agent_name="pilot")
            await _execute_session(session, task, result, output, workspace, agent_name="pilot")
        _verify_workspace(task, result, output, workspace)
        if result.controlled_command_observed is False:
            raise RuntimeError("Controlled command was not observed exactly as specified")
        result.status = "completed"
    except asyncio.CancelledError as exc:
        _record_failure(result, exc, workspace)
        raise
    except Exception as exc:
        _record_failure(result, exc, workspace)
    finally:
        config_path.unlink(missing_ok=True)
        result.wall_time_seconds = time.perf_counter() - started_at
        _save_result(result, output)
    return result


async def _execute_session(
    session: HarnessSession,
    task: BenchmarkTaskManifest,
    result: PilotResult,
    output: Path,
    workspace: Path,
    *,
    agent_name: str,
) -> None:
    agent = session.agent_app[agent_name]
    if not isinstance(agent, McpAgent):
        raise TypeError("Development Pilot requires an MCP coding agent")
    observer = PilotObserver(task.budget)
    observer.install(agent)
    try:
        async with asyncio.timeout(task.budget.max_wall_time_seconds):
            result.response = _sanitize(
                await session.send(
                    _task_prompt(task),
                    request_params=RequestParams(
                        parallel_tool_calls=False,
                        max_iterations=task.budget.max_llm_calls + 1,
                    ),
                ),
                workspace,
            )
    except (Exception, asyncio.CancelledError) as exc:
        _record_failure(result, exc, workspace)
        runtime = session.transactional_runtime
        if runtime is not None and runtime.controller is not None:
            if runtime.controller.state not in {
                RunState.FAILED,
                RunState.PROMOTED,
                RunState.PROMOTION_REJECTED,
            }:
                runtime.controller.fail(
                    f"Pilot stopped: {result.error_type}: {result.failure_reason}"
                )
        raise
    finally:
        # Observe before Harness closes stores, even when the turn never entered history.
        result.llm_calls = observer.budget.snapshot.llm_calls
        result.tool_calls = observer.budget.snapshot.tool_calls
        usage = agent.usage_accumulator
        if usage is not None:
            result.provider_attempts = usage.summary.provider_attempts
            if usage.summary.provider_attempts > 0:
                result.input_tokens = usage.summary.prompt.total
                result.output_tokens = usage.summary.completion.total
        result.model_visible_tool_result_bytes = _model_visible_tool_result_bytes(observer.messages)
        runtime = session.transactional_runtime
        if runtime is not None:
            result.run_id = str(runtime.run_id)
            result.artifact_output_bytes = runtime.budget.snapshot.artifact_output_bytes
            result.recovery_attempts = runtime.budget.snapshot.recovery_attempts
            run_records = (
                runtime.run_event_store.events_for_run(runtime.run_id)
                if runtime.run_event_store is not None
                else []
            )
            tool_records = runtime.event_store.events_for_run(runtime.run_id)
            result.run_events = tuple(item.event.kind.value for item in run_records)
            result.tool_events = tuple(item.event.kind.value for item in tool_records)
            result.promotion_result = _promotion_result(result.run_events)
            output.joinpath("events.json").write_text(
                _sanitize(
                    json.dumps(
                        {
                            "run": [
                                {
                                    "sequence": item.sequence,
                                    "kind": item.event.kind.value,
                                    "payload": asdict(item.event),
                                }
                                for item in run_records
                            ],
                            "tool": [
                                {
                                    "sequence": item.sequence,
                                    "kind": item.event.kind.value,
                                    "payload": asdict(item.event),
                                }
                                for item in tool_records
                            ],
                        },
                        default=str,
                        indent=2,
                    ),
                    workspace,
                ),
                encoding="utf-8",
            )
        controlled = _controlled_command_metrics(
            observer.messages, task.controlled_command, runtime
        )
        if controlled is not None:
            result.controlled_command_observed = controlled.observed
            result.controlled_tool_result_bytes_before_reducer = (
                controlled.tool_result_bytes_before_reducer
            )
            result.controlled_model_visible_result_bytes = controlled.model_visible_result_bytes
        _save_result(result, output)


def _record_failure(result: PilotResult, exc: BaseException, workspace: Path) -> None:
    result.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
    result.error_type = type(exc).__name__
    result.failure_reason = _sanitize(str(exc) or type(exc).__name__, workspace)


def _save_result(result: PilotResult, output: Path) -> None:
    output.joinpath("result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if result.status in {"failed", "cancelled"}:
        output.joinpath("failure.json").write_text(
            json.dumps(
                {"error_type": result.error_type, "message": result.failure_reason}, indent=2
            ),
            encoding="utf-8",
        )


def _verify_workspace(
    task: BenchmarkTaskManifest,
    result: PilotResult,
    output: Path,
    workspace: Path,
) -> None:
    try:
        verification = subprocess.run(
            task.verification.command,
            cwd=workspace,
            shell=True,
            check=False,
            capture_output=True,
            timeout=task.verification.timeout_seconds,
        )
        stdout, stderr = verification.stdout, verification.stderr
        result.verification_exit_code = verification.returncode
        result.verification_passed = verification.returncode == 0
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
        result.verification_timed_out = True
        result.verification_passed = False
    output.joinpath("verification.stdout").write_bytes(stdout)
    output.joinpath("verification.stderr").write_bytes(stderr)
    result.verification_stdout_bytes = len(stdout)
    result.verification_stderr_bytes = len(stderr)
    result.verification_output_sha256 = hashlib.sha256(stdout + b"\0" + stderr).hexdigest()
    if not result.verification_passed:
        raise RuntimeError(
            "External verification timed out"
            if result.verification_timed_out
            else f"External verification failed: exit_code={result.verification_exit_code}"
        )


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


def _model_visible_tool_result_bytes(history: Sequence[PromptMessageExtended]) -> int:
    total = 0
    for message in history:
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
) -> int | None:
    if runtime is None:
        return visible_bytes
    for item in runtime.event_store.events_for_run(runtime.run_id):
        event = item.event
        if isinstance(event, ToolResultStored) and str(event.tool_call_id) == call_id:
            return runtime.artifact_store.metadata(ArtifactId(event.artifact_id)).byte_size
    # A policy denial/aborted execution may have a visible result but no stored raw output.
    return None


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
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
