"""Provider validity is independent of account availability at passive mount."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import amplifier_module_provider_openai_chatgpt as module
import amplifier_module_provider_openai_chatgpt.provider as implementation
import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.testing import MockCoordinator
from amplifier_core.validation.provider import ProviderValidator
from amplifier_module_provider_openai_chatgpt import plan_auth
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider


def record(mode, state, *, renewable=False):
    if state == "missing":
        return None
    if mode == "chatgpt_plan":
        value = {
            "auth_mode": mode,
            "client_id": "fixture-client",
            "subject": "fixture-subject",
            "ext_agent_host_id": "fixture-host",
            "access_token": "fixture-access",
            "scopes": [plan_auth.PLAN_SCOPE],
            "expires_at": time.time() + (3600 if state == "valid" else -3600),
        }
    else:
        value = {
            "access_token": "fixture-access",
            "expires_at": (
                datetime.now(timezone.utc)
                + timedelta(hours=1 if state == "valid" else -1)
            ).isoformat(),
        }
    if renewable:
        value["refresh_token"] = "fixture-refresh"
    return value


def config_at(tmp_path, mode, state, *, renewable=False):
    path = tmp_path / "tokens.json"
    value = record(mode, state, renewable=renewable)
    if value:
        path.write_text(json.dumps(value))
    return {
        "auth_mode": mode,
        "token_file_path": str(path),
        "login_on_mount": False,
        "default_model": "fixture-model",
    }


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("No account/network calls in mount qualification")

    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("state", ["missing", "expired", "valid"])
async def test_core_validates_real_provider_without_mount_auth(
    tmp_path, monkeypatch, mode, state
):
    config = config_at(tmp_path, mode, state)
    reads = []

    def forbidden_read(*args, **kwargs):
        reads.append(True)
        raise AssertionError("Disabled mount must not read credentials")

    login = AsyncMock(side_effect=AssertionError("No automatic login"))
    refresh = AsyncMock(side_effect=AssertionError("No mount refresh"))
    with monkeypatch.context() as patch:
        patch.setattr(module, "load_tokens", forbidden_read)
        patch.setattr(module, "login", login)
        patch.setattr(plan_auth, "ensure_tokens", refresh)
        patch.setattr(plan_auth, "login", login)
        patch.setattr(plan_auth, "_read", forbidden_read)
        result = await ProviderValidator().validate(module.__name__, config=config)
        assert result.passed, result.summary()
        coordinator = MockCoordinator()
        cleanup = await module.mount(coordinator, config)
        provider = coordinator.mount_points["providers"]["openai-chatgpt"]
        assert isinstance(provider, ChatGPTProvider)
        assert provider.config is config
        assert callable(cleanup)
        assert not reads
        login.assert_not_awaited()
        refresh.assert_not_awaited()
    expected = (
        "authenticated"
        if state == "valid"
        else (
            "expired"
            if state == "expired" and mode == "chatgpt_codex"
            else "unauthenticated"
        )
    )
    assert provider.auth_status() == expected
    if state == "missing":
        with pytest.raises(llm_errors.AuthenticationError):
            await provider._ensure_valid_tokens()
    await cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
async def test_expired_tokens_remain_refreshable_on_request(
    tmp_path, monkeypatch, mode
):
    config = config_at(tmp_path, mode, "expired", renewable=True)
    coordinator = MockCoordinator()
    cleanup = await module.mount(coordinator, config)
    provider = coordinator.mount_points["providers"]["openai-chatgpt"]
    if mode == "chatgpt_plan":
        refresh = AsyncMock(
            return_value={
                "access_token": "fixture-renewed",
                "expires_in": 3600,
                "refresh_token": "fixture-rotated",
                "token_type": "Bearer",
                "scope": plan_auth.PLAN_SCOPE,
            }
        )
        monkeypatch.setattr(plan_auth, "_token_request", refresh)
    else:
        refreshed = record(mode, "valid")
        refreshed["access_token"] = "fixture-renewed"
        refresh = AsyncMock(return_value=refreshed)
        monkeypatch.setattr(implementation, "refresh_tokens", refresh)
    await provider._ensure_valid_tokens()
    refresh.assert_awaited_once()
    assert provider._tokens["access_token"] == "fixture-renewed"
    assert provider.auth_status() == "authenticated"
    await cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
async def test_expired_unrenewable_tokens_raise_typed_auth_error(tmp_path, mode):
    config = config_at(tmp_path, mode, "expired")
    coordinator = MockCoordinator()
    cleanup = await module.mount(coordinator, config)
    provider = coordinator.mount_points["providers"]["openai-chatgpt"]
    with pytest.raises(llm_errors.AuthenticationError):
        await provider._ensure_valid_tokens()
    await cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
async def test_disabled_login_does_not_hide_invalid_provider_config(tmp_path, mode):
    config = config_at(tmp_path, mode, "missing")
    config["auth_mode"] = "invalid-auth-mode"
    with pytest.raises(ValueError):
        await module.mount(MockCoordinator(), config)
