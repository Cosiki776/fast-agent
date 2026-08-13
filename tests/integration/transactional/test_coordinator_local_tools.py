from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.transactional.coordinator import TransactionCoordinator
from fast_agent.transactional.events import ToolEventKind, ToolResultStored
from fast_agent.transactional.models import RunId, TransactionId
from fast_agent.transactional.storage.artifact_store import ArtifactId, FileArtifactStore
from fast_agent.transactional.storage.event_store import SQLiteEventStore
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.transactional.coordinator import ToolDenialResolver


RUN_ID = RunId("integration-run")
TRANSACTION_ID = TransactionId("integration-transaction")


def _agent(
    tmp_path: Path,
    *,
    denial_reason: ToolDenialResolver | None = None,
) -> tuple[McpAgent, SQLiteEventStore, FileArtifactStore]:
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    coordinator = TransactionCoordinator(
        event_store,
        artifact_store,
        denial_reason=denial_reason,
        transaction_id_factory=lambda: TRANSACTION_ID,
    )
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="on"))
    agent = McpAgent(
        config=AgentConfig(
            name="transactional-test",
            instruction="Run local tools",
            servers=[],
            shell=True,
            cwd=tmp_path,
        ),
        context=Context(config=settings),
        transactional_run_id=RUN_ID,
        tool_execution_interceptor=coordinator,
    )
    return agent, event_store, artifact_store


def _request(
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Use the local tool")],
        tool_calls={
            tool_call_id: CallToolRequest(
                params=CallToolRequestParams(name=tool_name, arguments=arguments)
            )
        },
    )


def _event_kinds(event_store: SQLiteEventStore) -> list[ToolEventKind]:
    return [item.event.kind for item in event_store.events_for_transaction(TRANSACTION_ID)]


@pytest.mark.asyncio
async def test_real_write_tool_commits_after_raw_result_is_stored(tmp_path: Path) -> None:
    agent, event_store, artifact_store = _agent(tmp_path)
    output_path = tmp_path / "generated" / "notes.txt"
    try:
        message = await agent.run_tools(
            _request(
                "write-call",
                "write_text_file",
                {"path": str(output_path), "content": "transactional write"},
            )
        )

        assert output_path.read_text(encoding="utf-8") == "transactional write"
        assert message.tool_results is not None
        tool_result = message.tool_results["write-call"]
        assert tool_result.is_error is False
        assert _event_kinds(event_store) == [
            ToolEventKind.PROPOSED,
            ToolEventKind.EXECUTION_STARTED,
            ToolEventKind.RESULT_STORED,
            ToolEventKind.COMMITTED,
        ]
        events = [item.event for item in event_store.events_for_transaction(TRANSACTION_ID)]
        assert {event.tool_call_id for event in events} == {"write-call"}
        stored = events[2]
        assert isinstance(stored, ToolResultStored)
        raw_result = json.loads(artifact_store.read(ArtifactId(stored.artifact_id)).decode("utf-8"))
        assert raw_result["isError"] is False
        assert raw_result["content"] == [
            {
                "type": "text",
                "text": (
                    f"Successfully wrote {len('transactional write')} characters to {output_path}"
                ),
            }
        ]
    finally:
        await agent._aggregator.close()
        event_store.close()


@pytest.mark.asyncio
async def test_denied_write_returns_tool_result_without_touching_workspace(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "denied.txt"
    agent, event_store, _ = _agent(
        tmp_path,
        denial_reason=lambda request: "workspace writes are disabled",
    )
    try:
        message = await agent.run_tools(
            _request(
                "denied-call",
                "write_text_file",
                {"path": str(output_path), "content": "must not be written"},
            )
        )

        assert output_path.exists() is False
        assert message.tool_results is not None
        tool_result = message.tool_results["denied-call"]
        assert tool_result.is_error is True
        assert tool_result.structured_content == {
            "status": "denied",
            "reason": "workspace writes are disabled",
        }
        assert _event_kinds(event_store) == [
            ToolEventKind.PROPOSED,
            ToolEventKind.DENIED,
            ToolEventKind.FAILED,
        ]
    finally:
        await agent._aggregator.close()
        event_store.close()


@pytest.mark.asyncio
async def test_failed_read_returns_stored_error_result(tmp_path: Path) -> None:
    agent, event_store, artifact_store = _agent(tmp_path)
    missing_path = tmp_path / "missing.txt"
    try:
        message = await agent.run_tools(
            _request("read-call", "read_text_file", {"path": str(missing_path)})
        )

        assert message.tool_results is not None
        tool_result = message.tool_results["read-call"]
        assert tool_result.is_error is True
        assert _event_kinds(event_store) == [
            ToolEventKind.PROPOSED,
            ToolEventKind.EXECUTION_STARTED,
            ToolEventKind.RESULT_STORED,
            ToolEventKind.FAILED,
        ]
        events = [item.event for item in event_store.events_for_transaction(TRANSACTION_ID)]
        stored = events[2]
        assert isinstance(stored, ToolResultStored)
        raw_result = json.loads(artifact_store.read(ArtifactId(stored.artifact_id)).decode("utf-8"))
        assert raw_result["isError"] is True
        assert "missing.txt" in raw_result["content"][0]["text"]
    finally:
        await agent._aggregator.close()
        event_store.close()
