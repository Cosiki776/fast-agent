from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mcp.types import CallToolResult, TextContent

from fast_agent.transactional.context.reducers import (
    ToolResultReducer,
    bounded_fallback_result,
)
from fast_agent.transactional.events import (
    ToolAuthorized,
    ToolCheckpointed,
    ToolCheckpointFailed,
    ToolCommitted,
    ToolDenied,
    ToolExecutionFailed,
    ToolExecutionStarted,
    ToolFailed,
    ToolProposed,
    ToolResultStored,
    ToolValidated,
)
from fast_agent.transactional.execution import (
    ToolCallNext,
    ToolExecutionOutcome,
    ToolExecutionRequest,
)
from fast_agent.transactional.models import (
    ToolEffect,
    TransactionId,
    new_transaction_id,
)
from fast_agent.transactional.recovery.classifier import (
    LocalCodingEffectClassifier,
    ToolEffectClassifier,
    classify_local_tool_effect,
)
from fast_agent.transactional.storage.artifact_store import ArtifactKind

if TYPE_CHECKING:
    from fast_agent.transactional.budget import BudgetDimension, RunBudgetTracker
    from fast_agent.transactional.storage.artifact_store import FileArtifactStore
    from fast_agent.transactional.storage.event_store import SQLiteEventStore

type ToolDenialResolver = Callable[[ToolExecutionRequest], str | None]
type ToolCheckpointCreator = Callable[[ToolExecutionRequest], str]
type TransactionIdFactory = Callable[[], TransactionId]

class TransactionCoordinator:
    """Order one tool call across transaction persistence boundaries."""

    def __init__(
        self,
        event_store: SQLiteEventStore,
        artifact_store: FileArtifactStore,
        *,
        denial_reason: ToolDenialResolver | None = None,
        checkpoint_creator: ToolCheckpointCreator | None = None,
        result_reducer: ToolResultReducer | None = None,
        run_budget: RunBudgetTracker | None = None,
        effect_classifier: ToolEffectClassifier | None = None,
        transaction_id_factory: TransactionIdFactory = new_transaction_id,
    ) -> None:
        self._event_store = event_store
        self._artifact_store = artifact_store
        self._denial_reason = denial_reason
        self._checkpoint_creator = checkpoint_creator
        self._result_reducer = result_reducer
        self._run_budget = run_budget
        self._effect_classifier = effect_classifier or LocalCodingEffectClassifier()
        self._transaction_id_factory = transaction_id_factory

    async def coordinate(
        self,
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
        /,
    ) -> ToolExecutionOutcome:
        transaction_id = self._transaction_id_factory()
        effect = self._classify_effect(request)
        self._event_store.append(
            ToolProposed(
                run_id=request.run_id,
                transaction_id=transaction_id,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                arguments=request.arguments,
                effect=effect,
            )
        )

        if self._run_budget is not None:
            budget_decision = self._run_budget.start_tool_call()
            if not budget_decision.allowed:
                exhausted = ", ".join(item.value for item in budget_decision.exhausted)
                reason = f"budget exhausted: {exhausted}"
                self._event_store.append(
                    ToolDenied(
                        run_id=request.run_id,
                        transaction_id=transaction_id,
                        tool_call_id=request.tool_call_id,
                        reason=reason,
                    )
                )
                self._event_store.append(
                    ToolFailed(
                        run_id=request.run_id,
                        transaction_id=transaction_id,
                        tool_call_id=request.tool_call_id,
                        reason="budget exhausted",
                    )
                )
                return ToolExecutionOutcome(
                    result=_budget_exhausted_result(budget_decision.exhausted)
                )

        denial_reason = self._denial_reason(request) if self._denial_reason is not None else None
        if effect is ToolEffect.EXTERNAL_UNKNOWN:
            denial_reason = f"tool effect is unknown: {request.tool_name}"
        if denial_reason is not None:
            self._event_store.append(
                ToolDenied(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason=denial_reason,
                )
            )
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="policy denied",
                )
            )
            return ToolExecutionOutcome(result=_denied_result(denial_reason))

        if effect is ToolEffect.WORKSPACE_WRITE and self._checkpoint_creator is not None:
            self._event_store.append(
                ToolValidated(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                )
            )
            self._event_store.append(
                ToolAuthorized(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                )
            )
            try:
                checkpoint_id = self._checkpoint_creator(request)
                if not checkpoint_id:
                    raise ValueError("checkpoint creator returned an empty ID")
                self._event_store.append(
                    ToolCheckpointed(
                        run_id=request.run_id,
                        transaction_id=transaction_id,
                        tool_call_id=request.tool_call_id,
                        checkpoint_id=checkpoint_id,
                    )
                )
            except Exception as exc:
                self._record_checkpoint_failure(
                    request=request,
                    transaction_id=transaction_id,
                    error=exc,
                )
                return ToolExecutionOutcome(result=_checkpoint_failed_result(exc))

        self._event_store.append(
            ToolExecutionStarted(
                run_id=request.run_id,
                transaction_id=transaction_id,
                tool_call_id=request.tool_call_id,
            )
        )
        try:
            outcome = await call_next()
        except Exception as exc:
            error_type = type(exc).__name__
            message = str(exc)
            self._event_store.append(
                ToolExecutionFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    error_type=error_type,
                    message=message,
                )
            )
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="tool execution raised an exception",
                )
            )
            return ToolExecutionOutcome(
                result=_execution_failed_result(error_type=error_type, message=message)
            )

        result = outcome.result
        serialized_result = serialize_tool_result(result)
        artifact = self._artifact_store.put(
            serialized_result,
            media_type="application/json",
            kind=ArtifactKind.RAW_RESULT,
        )
        if self._run_budget is not None:
            self._run_budget.record_artifact_output(len(serialized_result))
        is_error = bool(result.isError)
        self._event_store.append(
            ToolResultStored(
                run_id=request.run_id,
                transaction_id=transaction_id,
                tool_call_id=request.tool_call_id,
                artifact_id=artifact.artifact_id,
                is_error=is_error,
            )
        )

        reduced_result = result
        if self._result_reducer is not None:
            try:
                reduced_result = self._result_reducer(request, result, artifact.artifact_id)
            except Exception:
                # The raw artifact is already durable, so reduction failure can safely
                # degrade to a bounded deterministic view without losing evidence.
                reduced_result = bounded_fallback_result(result, artifact.artifact_id)

        if is_error:
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="tool returned an error result",
                )
            )
        else:
            self._event_store.append(
                ToolCommitted(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                )
            )
        return ToolExecutionOutcome(result=reduced_result)

    async def __call__(
        self,
        request: ToolExecutionRequest,
        call_next: ToolCallNext,
        /,
    ) -> ToolExecutionOutcome:
        """Adapt the coordinator to the per-tool execution interceptor contract."""

        return await self.coordinate(request, call_next)

    def _classify_effect(self, request: ToolExecutionRequest) -> ToolEffect:
        try:
            effect = self._effect_classifier(request)
        except Exception:
            return ToolEffect.EXTERNAL_UNKNOWN
        return effect if isinstance(effect, ToolEffect) else ToolEffect.EXTERNAL_UNKNOWN

    def _record_checkpoint_failure(
        self,
        *,
        request: ToolExecutionRequest,
        transaction_id: TransactionId,
        error: Exception,
    ) -> None:
        try:
            self._event_store.append(
                ToolCheckpointFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            self._event_store.append(
                ToolFailed(
                    run_id=request.run_id,
                    transaction_id=transaction_id,
                    tool_call_id=request.tool_call_id,
                    reason="checkpoint failed",
                )
            )
        except Exception:
            # Checkpoint failure remains fail-closed even if its event cannot be persisted.
            return


def classify_tool_effect(tool_name: str) -> ToolEffect:
    """Classify the explicit first-phase local coding tool names."""

    return classify_local_tool_effect(tool_name)


def serialize_tool_result(result: CallToolResult) -> bytes:
    """Serialize the complete standard MCP result stored as transaction evidence."""

    return result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")


def _denied_result(reason: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=f"Tool execution denied: {reason}")],
        structuredContent={"status": "denied", "reason": reason},
        isError=True,
    )


def _execution_failed_result(*, error_type: str, message: str) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Tool execution failed ({error_type}): {message}",
            )
        ],
        structuredContent={
            "status": "failed",
            "error_type": error_type,
            "message": message,
        },
        isError=True,
    )


def _checkpoint_failed_result(error: Exception) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Tool execution stopped: checkpoint failed ({type(error).__name__}: {error})",
            )
        ],
        structuredContent={
            "status": "checkpoint_failed",
            "error_type": type(error).__name__,
            "message": str(error),
        },
        isError=True,
    )


def _budget_exhausted_result(
    dimensions: tuple[BudgetDimension, ...],
) -> CallToolResult:
    exhausted = [dimension.value for dimension in dimensions]
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Tool execution stopped: budget exhausted ({', '.join(exhausted)})",
            )
        ],
        structuredContent={"status": "budget_exhausted", "dimensions": exhausted},
        isError=True,
    )
