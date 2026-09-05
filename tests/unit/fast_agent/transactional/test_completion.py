from __future__ import annotations

from pathlib import Path

from fast_agent.transactional.checkpoint.snapshot import AgentDelta
from fast_agent.transactional.completion import WorktreeOnlyCompletionReport
from fast_agent.transactional.models import RunId


def test_worktree_only_completion_report_describes_manual_review_handoff() -> None:
    report = WorktreeOnlyCompletionReport.from_delta(
        RunId("run-123"),
        Path("/tmp/agent worktree"),
        AgentDelta(
            added=("new.py",),
            modified=("src/existing.py",),
            deleted=("old.txt",),
        ),
    )

    rendered = report.render_text()

    assert report.changed_file_count == 3
    assert "Verification: not configured" in rendered
    assert "Promotion: not applied" in rendered
    assert "Agent result retained at: /tmp/agent worktree" in rendered
    assert "Changes: 1 added, 1 modified, 1 deleted" in rendered
    assert "  - new.py" in rendered
    assert "  - src/existing.py" in rendered
    assert "  - old.txt" in rendered
    assert "git -C '/tmp/agent worktree' status --short" in rendered
    assert "transactional.verification.command" in rendered
