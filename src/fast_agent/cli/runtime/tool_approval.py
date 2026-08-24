from __future__ import annotations

import json
from typing import Any

from rich.text import Text

from fast_agent.mcp.tool_permission_handler import ToolPermissionResult
from fast_agent.ui.console import rich_print
from fast_agent.ui.prompt.input import get_selection_input
from fast_agent.utils.text import strip_casefold

_ARGUMENT_PREVIEW_LIMIT = 800


class CliToolApprovalHandler:
    """Request one non-persistent tool approval from the terminal user."""

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        del tool_use_id
        preview = json.dumps(
            arguments or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(preview) > _ARGUMENT_PREVIEW_LIMIT:
            preview = f"{preview[: _ARGUMENT_PREVIEW_LIMIT - 1]}…"

        rich_print(Text("Transactional tool approval required", style="bold yellow"))
        rich_print(Text(f"  Tool: {tool_name}"))
        rich_print(Text(f"  Server: {server_name}"))
        rich_print(Text(f"  Arguments: {preview}"))
        rich_print(
            Text(
                "  Warning: external side effects cannot be rolled back and are not exactly-once.",
                style="yellow",
            )
        )
        selection = await get_selection_input(
            "Allow this tool once? [a/D] ",
            options=["a", "allow", "d", "deny"],
            default="d",
            allow_cancel=True,
        )
        allowed = strip_casefold(selection or "") in {"a", "allow"}
        if allowed:
            return ToolPermissionResult.allow()
        return ToolPermissionResult(
            allowed=False,
            is_cancelled=selection is None,
            error_message=(
                "Tool approval cancelled" if selection is None else "Tool approval denied"
            ),
        )
