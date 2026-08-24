from __future__ import annotations

import pytest

from fast_agent.cli.runtime.tool_approval import CliToolApprovalHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(("selection", "allowed"), [("a", True), ("deny", False), (None, False)])
async def test_cli_tool_approval_is_allow_once_or_deny(
    monkeypatch: pytest.MonkeyPatch,
    selection: str | None,
    allowed: bool,
) -> None:
    async def choose(*args, **kwargs) -> str | None:
        del args, kwargs
        return selection

    monkeypatch.setattr("fast_agent.cli.runtime.tool_approval.get_selection_input", choose)

    result = await CliToolApprovalHandler().check_permission(
        "remote_tool",
        "example-server",
        {"target": "external"},
        "call-1",
    )

    assert result.allowed is allowed
    assert result.remember is False
    assert result.is_cancelled is (selection is None)
