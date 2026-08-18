from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_agent.config import Settings
from fast_agent.transactional.settings import TransactionalProfile, TransactionalSettings


def test_transactional_profile_defaults_to_baseline() -> None:
    assert Settings.model_validate({}).transactional.profile is TransactionalProfile.BASELINE


@pytest.mark.parametrize("profile", list(TransactionalProfile))
def test_transactional_profiles_load_from_settings(profile: TransactionalProfile) -> None:
    settings = Settings.model_validate({"transactional": {"profile": profile.value}})

    assert settings.transactional.profile is profile


def test_invalid_transactional_profile_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionalSettings.model_validate({"profile": "custom"})


def test_negative_transactional_budget_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionalSettings.model_validate({"max_tool_calls": -1})


def test_transactional_profile_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSACTIONAL__PROFILE", "reducer")

    assert Settings(_env_file=None).transactional.profile is TransactionalProfile.REDUCER
