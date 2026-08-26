from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.transactional.benchmark import (
    BenchmarkObservation,
    BenchmarkProfile,
    BenchmarkTaskManifest,
    TransactionalBenchmarkRunner,
    load_task_manifest,
)
from fast_agent.transactional.context.reducers import (
    CodingToolResultReducer,
    ReducerLimits,
    ToolResultReducer,
)
from fast_agent.transactional.coordinator import TransactionCoordinator
from fast_agent.transactional.events import ToolResultStored
from fast_agent.transactional.models import RunId, TransactionId
from fast_agent.transactional.storage.artifact_store import ArtifactId, FileArtifactStore
from fast_agent.transactional.storage.event_store import SQLiteEventStore
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.transactional.execution import ToolExecutionRequest


RUN_ID = RunId("benchmark-smoke-run")
TRANSACTION_ID = TransactionId("benchmark-smoke-transaction")
TOOL_CALL_ID = "benchmark-shell-call"


class _TimedReducer:
    def __init__(self, reducer: ToolResultReducer) -> None:
        self._reducer = reducer
        self.latency_seconds: float | None = None

    def __call__(
        self,
        request: ToolExecutionRequest,
        result: CallToolResult,
        artifact_id: ArtifactId,
        /,
    ) -> CallToolResult:
        started_at = time.perf_counter()
        reduced = self._reducer(request, result, artifact_id)
        self.latency_seconds = time.perf_counter() - started_at
        return reduced


def _agent(
    directory: Path,
    *,
    result_reducer: ToolResultReducer | None,
) -> tuple[McpAgent, SQLiteEventStore, FileArtifactStore]:
    event_store = SQLiteEventStore(directory / "events.sqlite3")
    artifact_store = FileArtifactStore(directory / "artifacts")
    coordinator = TransactionCoordinator(
        event_store,
        artifact_store,
        result_reducer=result_reducer,
        transaction_id_factory=lambda: TRANSACTION_ID,
    )
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="on"))
    agent = McpAgent(
        config=AgentConfig(
            name=f"benchmark-{directory.name}",
            instruction="Run the benchmark tool",
            servers=[],
            shell=True,
            cwd=directory,
        ),
        context=Context(config=settings),
        transactional_run_id=RUN_ID,
        tool_execution_interceptor=coordinator,
    )
    return agent, event_store, artifact_store


def _request(command: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Run the benchmark command")],
        tool_calls={
            TOOL_CALL_ID: CallToolRequest(
                params=CallToolRequestParams(name="bash", arguments={"command": command})
            )
        },
    )


@pytest.mark.asyncio
async def test_runner_collects_reducer_metrics_from_real_shell_result(tmp_path: Path) -> None:
    raw_marker = "complete-benchmark-evidence-" + "x" * 12_000
    script = "\n".join(
        [
            "print('FAILED tests/test_order.py::test_cutoff - AssertionError: cutoff rejected')",
            "print('E   AssertionError: cutoff rejected')",
            "print('tests/test_order.py:42: in test_cutoff')",
            f"print({raw_marker!r})",
            "raise SystemExit(1)",
        ]
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} pytest"
    task = load_task_manifest(Path("benchmarks/transactional/tasks/boundary-check-001.yaml"))
    model_visible_text: dict[BenchmarkProfile, str] = {}
    artifact_payloads: dict[BenchmarkProfile, bytes] = {}

    async def execute(
        received_task: BenchmarkTaskManifest,
        profile: BenchmarkProfile,
    ) -> BenchmarkObservation:
        assert received_task is task
        profile_path = tmp_path / profile.value
        profile_path.mkdir()
        timed_reducer: _TimedReducer | None = None
        result_reducer: ToolResultReducer | None = None
        if profile is BenchmarkProfile.REDUCER:
            timed_reducer = _TimedReducer(
                CodingToolResultReducer(ReducerLimits(max_result_bytes=512, max_items=6))
            )
            result_reducer = timed_reducer

        agent, event_store, artifact_store = _agent(
            profile_path,
            result_reducer=result_reducer,
        )
        try:
            message = await agent.run_tools(_request(command))
            assert message.tool_results is not None
            visible_result = message.tool_results[TOOL_CALL_ID]
            visible_content = visible_result.content[0]
            assert isinstance(visible_content, TextContent)
            model_visible_text[profile] = visible_content.text

            stored = next(
                item.event
                for item in event_store.events_for_transaction(TRANSACTION_ID)
                if isinstance(item.event, ToolResultStored)
            )
            raw_artifact = artifact_store.read(ArtifactId(stored.artifact_id))
            artifact_payloads[profile] = raw_artifact
            return BenchmarkObservation(
                verification_command_passed=True,
                llm_calls=0,
                tool_calls=1,
                input_tokens=None,
                output_tokens=None,
                full_output_bytes=len(raw_artifact),
                model_visible_output_bytes=len(
                    visible_result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
                ),
                reducer_latency_seconds=(
                    timed_reducer.latency_seconds if timed_reducer is not None else None
                ),
            )
        finally:
            await agent._aggregator.close()
            event_store.close()

    result = await TransactionalBenchmarkRunner(execute).run_ab(task)

    baseline_text = model_visible_text[BenchmarkProfile.BASELINE]
    reducer_text = model_visible_text[BenchmarkProfile.REDUCER]
    assert raw_marker in baseline_text
    assert raw_marker not in reducer_text
    assert "exit_code: 1" in reducer_text
    assert "tests/test_order.py::test_cutoff" in reducer_text
    assert "AssertionError: cutoff rejected" in reducer_text
    assert "output_artifact:" in reducer_text
    assert raw_marker.encode() in artifact_payloads[BenchmarkProfile.REDUCER]
    assert result.reducer.model_visible_output_bytes < result.baseline.model_visible_output_bytes
    assert result.baseline.reducer_latency_seconds is None
    assert result.reducer.reducer_latency_seconds is not None
