from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fast_agent.transactional.budget import RunBudgetTracker
from fast_agent.transactional.checkpoint.checkpoint import CheckpointManager
from fast_agent.transactional.context.reducers import CodingToolResultReducer, ToolResultReducer
from fast_agent.transactional.coordinator import TransactionCoordinator
from fast_agent.transactional.governance import CodingToolGovernanceGate, CodingToolPolicy
from fast_agent.transactional.models import RunId, new_run_id
from fast_agent.transactional.recovery.controller import RecoveryController
from fast_agent.transactional.run_controller import TransactionalCodingRun
from fast_agent.transactional.run_events import RunStarted, RunState
from fast_agent.transactional.settings import (
    SemanticReducerVersion,
    ToolOutputStrategy,
    TransactionalProfile,
    TransactionalSettings,
)
from fast_agent.transactional.storage.artifact_store import FileArtifactStore
from fast_agent.transactional.storage.event_store import SQLiteEventStore
from fast_agent.transactional.storage.run_event_store import SQLiteRunEventStore
from fast_agent.transactional.verification import CompletionVerifier

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler
    from fast_agent.transactional.checkpoint.checkpoint import CheckpointMetadata
    from fast_agent.transactional.checkpoint.snapshot import (
        WorkspaceSnapshot,
        WorkspaceSnapshotManager,
    )
    from fast_agent.transactional.checkpoint.worktree import WorktreeMetadata
    from fast_agent.transactional.execution import ToolExecutionRequest


@dataclass(slots=True)
class TransactionalRuntime:
    """Resources owned by one transactional Harness session."""

    run_id: RunId
    profile: TransactionalProfile
    event_store: SQLiteEventStore
    artifact_store: FileArtifactStore
    budget: RunBudgetTracker
    coordinator: TransactionCoordinator
    shell_terminal_timeout_seconds: float
    run_event_store: SQLiteRunEventStore | None = None
    controller: TransactionalCodingRun | None = None
    worktree: WorktreeMetadata | None = None
    _closed: bool = field(default=False, init=False)

    @property
    def workspace(self) -> Path | None:
        return self.worktree.worktree_path if self.worktree is not None else None

    @property
    def requires_manual_review(self) -> bool:
        """Whether this Run's Worktree must be retained for manual review."""
        return self.controller is not None and self.controller.state is RunState.PROMOTION_REJECTED

    def close(self) -> None:
        if self._closed:
            return
        self.event_store.close()
        if self.run_event_store is not None:
            self.run_event_store.close()
        self._closed = True


class TransactionalRuntimeAssembler:
    """Build exactly the run-scoped resources selected by one profile."""

    def __init__(
        self,
        settings: TransactionalSettings,
        runtime_root: Path,
    ) -> None:
        self._settings = settings
        self._runtime_root = runtime_root.resolve()

    def assemble(
        self,
        *,
        run_id: RunId | None = None,
        worktree: WorktreeMetadata | None = None,
        permission_handler: ToolPermissionHandler | None = None,
        snapshot_manager: WorkspaceSnapshotManager | None = None,
        snapshot: WorkspaceSnapshot | None = None,
    ) -> TransactionalRuntime | None:
        if self._settings.profile is TransactionalProfile.BASELINE:
            return None

        active_run_id = run_id or new_run_id()
        run_root = self._runtime_root / str(active_run_id)
        event_store = SQLiteEventStore(run_root / "events.sqlite3")
        artifact_store = FileArtifactStore(run_root / "artifacts")
        budget = RunBudgetTracker(self._settings.budget_limits())
        reducer = self._result_reducer()

        if self._settings.profile is TransactionalProfile.REDUCER:
            coordinator = TransactionCoordinator(
                event_store,
                artifact_store,
                result_reducer=reducer,
                run_budget=budget,
            )
            return TransactionalRuntime(
                run_id=active_run_id,
                profile=self._settings.profile,
                event_store=event_store,
                artifact_store=artifact_store,
                budget=budget,
                coordinator=coordinator,
                shell_terminal_timeout_seconds=self._settings.shell_terminal_timeout_seconds,
            )

        if worktree is None:
            event_store.close()
            raise ValueError("full transactional profile requires a Run worktree")
        return self._assemble_full(
            active_run_id,
            run_root,
            event_store,
            artifact_store,
            budget,
            reducer,
            worktree,
            permission_handler,
            snapshot_manager,
            snapshot,
        )

    def _assemble_full(
        self,
        run_id: RunId,
        run_root: Path,
        event_store: SQLiteEventStore,
        artifact_store: FileArtifactStore,
        budget: RunBudgetTracker,
        reducer: ToolResultReducer | None,
        worktree: WorktreeMetadata,
        permission_handler: ToolPermissionHandler | None,
        snapshot_manager: WorkspaceSnapshotManager | None,
        snapshot: WorkspaceSnapshot | None,
    ) -> TransactionalRuntime:
        if worktree.run_id != run_id:
            event_store.close()
            raise ValueError("Run worktree belongs to a different transactional run")
        checkpoint_manager = CheckpointManager(worktree, run_root / "checkpoints")
        governance_gate = CodingToolGovernanceGate(
            CodingToolPolicy(
                worktree.worktree_path,
                run_root,
                max_shell_timeout_seconds=self._settings.shell_terminal_timeout_seconds,
            ),
            current_workspace_version=lambda: str(checkpoint_manager.current_version()),
            permission_handler=permission_handler,
        )
        checkpoints: dict[str, CheckpointMetadata] = {}

        def save_checkpoint() -> str:
            checkpoint = checkpoint_manager.create()
            checkpoint_id = str(checkpoint.checkpoint_id)
            checkpoints[checkpoint_id] = checkpoint
            return checkpoint_id

        def create_checkpoint(request: ToolExecutionRequest) -> str:
            del request
            return save_checkpoint()

        def restore_checkpoint(checkpoint_id: str) -> str:
            return str(checkpoint_manager.restore(checkpoints[checkpoint_id]))

        run_events = SQLiteRunEventStore(event_store.path)
        run_events.append(RunStarted(run_id=run_id, profile=self._settings.profile.value))
        recovery = RecoveryController(run_budget=budget)
        coordinator = TransactionCoordinator(
            event_store,
            artifact_store,
            checkpoint_creator=create_checkpoint,
            checkpoint_restorer=restore_checkpoint,
            result_reducer=reducer,
            run_budget=budget,
            governance_gate=governance_gate,
            recovery_controller=recovery,
            run_event_store=run_events,
        )
        verifier = None
        promote = None
        if self._settings.verification is not None:
            from fast_agent.core.logging.logger import get_logger
            from fast_agent.tools.local_shell_executor import LocalShellExecutor

            verifier = CompletionVerifier(
                LocalShellExecutor(
                    logger=get_logger(__name__),
                    working_directory=worktree.worktree_path,
                ),
                artifact_store,
            )
            if snapshot_manager is None or snapshot is None:
                event_store.close()
                run_events.close()
                raise ValueError("workspace promotion requires the initial WorkspaceSnapshot")
        if snapshot_manager is not None and snapshot is not None:
            from fast_agent.transactional.checkpoint.promotion import WorkspacePromoter

            promoter = WorkspacePromoter(
                snapshot_manager,
                snapshot,
                worktree,
                artifact_store,
            )
            promote = promoter.promote
        controller = TransactionalCodingRun(
            run_id,
            run_events,
            budget,
            verifier=verifier,
            verification_spec=self._settings.verification,
            workspace=worktree.worktree_path,
            workspace_version=(
                (lambda: str(snapshot_manager.worktree_version(snapshot, worktree)))
                if snapshot_manager is not None and snapshot is not None
                else lambda: str(checkpoint_manager.current_version())
            ),
            promote=promote,
            create_verification_checkpoint=save_checkpoint,
            restore_verification_checkpoint=restore_checkpoint,
        )
        return TransactionalRuntime(
            run_id=run_id,
            profile=self._settings.profile,
            event_store=event_store,
            artifact_store=artifact_store,
            budget=budget,
            coordinator=coordinator,
            shell_terminal_timeout_seconds=self._settings.shell_terminal_timeout_seconds,
            run_event_store=run_events,
            controller=controller,
            worktree=worktree,
        )

    def _result_reducer(self) -> ToolResultReducer | None:
        tool_output = self._settings.tool_output
        if tool_output.strategy is ToolOutputStrategy.UPSTREAM:
            return None
        if tool_output.semantic_reducer_version is SemanticReducerVersion.V1:
            return CodingToolResultReducer()
        raise ValueError(
            f"Unsupported semantic reducer version: {tool_output.semantic_reducer_version}"
        )
