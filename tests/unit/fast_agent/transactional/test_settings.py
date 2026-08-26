from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_agent.config import Settings
from fast_agent.transactional.settings import (
    SemanticReducerVersion,
    ToolOutputStrategy,
    TransactionalProfile,
    TransactionalSettings,
)


def test_transactional_profile_defaults_to_baseline() -> None:
    transactional = Settings.model_validate({}).transactional

    assert transactional.profile is TransactionalProfile.BASELINE
    assert transactional.tool_output.strategy is ToolOutputStrategy.UPSTREAM
    assert transactional.tool_output.semantic_reducer_version is None


@pytest.mark.parametrize("profile", list(TransactionalProfile))
def test_transactional_profiles_load_from_settings(profile: TransactionalProfile) -> None:
    settings = Settings.model_validate({"transactional": {"profile": profile.value}})

    assert settings.transactional.profile is profile


def test_invalid_transactional_profile_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionalSettings.model_validate({"profile": "custom"})


def test_semantic_tool_output_requires_and_loads_version() -> None:
    settings = TransactionalSettings.model_validate(
        {
            "tool_output": {
                "strategy": "semantic",
                "semantic_reducer_version": "v1",
            }
        }
    )

    assert settings.tool_output.strategy is ToolOutputStrategy.SEMANTIC
    assert settings.tool_output.semantic_reducer_version is SemanticReducerVersion.V1

    with pytest.raises(ValidationError, match="requires semantic_reducer_version"):
        TransactionalSettings.model_validate({"tool_output": {"strategy": "semantic"}})


def test_upstream_tool_output_rejects_semantic_version() -> None:
    with pytest.raises(ValidationError, match="requires semantic tool output"):
        TransactionalSettings.model_validate(
            {"tool_output": {"strategy": "upstream", "semantic_reducer_version": "v1"}}
        )


def test_negative_transactional_budget_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionalSettings.model_validate({"max_tool_calls": -1})

    with pytest.raises(ValidationError):
        TransactionalSettings.model_validate({"shell_terminal_timeout_seconds": -1})


def test_transactional_profile_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSACTIONAL__PROFILE", "reducer")

    assert Settings(_env_file=None).transactional.profile is TransactionalProfile.REDUCER
