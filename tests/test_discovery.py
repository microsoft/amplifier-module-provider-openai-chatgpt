"""Strict discovery through the real fetch/mapping code, with no external I/O."""

import asyncio
import copy
import inspect
import json
import time
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from amplifier_core import llm_errors

from amplifier_module_provider_openai_chatgpt import models, plan_auth
from amplifier_module_provider_openai_chatgpt import provider as provider_module
from amplifier_module_provider_openai_chatgpt.provider import CODEX_MODE, ChatGPTProvider


_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    """A missing stub must fail rather than touch credentials or the network."""
    monkeypatch.setattr(
        httpx, "AsyncClient", Mock(side_effect=AssertionError("HTTP must be stubbed"))
    )
    monkeypatch.setattr(
        provider_module,
        "load_tokens",
        Mock(side_effect=AssertionError("Credential reads must be stubbed")),
    )
    monkeypatch.setattr(
        provider_module,
        "refresh_tokens",
        AsyncMock(side_effect=AssertionError("Credential refresh must be stubbed")),
    )
    monkeypatch.setattr(
        plan_auth,
        "ensure_tokens",
        AsyncMock(side_effect=AssertionError("Plan authentication must be stubbed")),
    )


def _entry(slug, **extra):
    return {
        "slug": slug,
        "display_name": f"Display {slug}",
        "visibility": "list",
        "supported_in_api": True,
        "context_window": 64_000,
        "max_output_tokens": 8_000,
        **extra,
    }


def _make_provider(mode, *, inject_tokens=True):
    provider = ChatGPTProvider(
        {
            "auth_mode": mode,
            "token_file_path": "/unused/discovery-credentials.json",
            "default_model": "configured-but-not-in-inventory",
            "model": "another-selection-only",
            "reasoning_effort": "high",
            "priority": 7,
            "models_client_version": "fixture-version",
        },
        tokens={
            "auth_mode": mode,
            "access_token": "fixture-access-token",
            "account_id": "fixture-account",
            "expires_at": "2999-01-01T00:00:00+00:00",
        } if inject_tokens else None,
    )
    provider._ensure_valid_tokens = AsyncMock()
    # Plausible legacy data exists even when strict discovery must fail.
    provider._models_cache = (
        time.monotonic(),
        models.to_model_infos(models.FALLBACK_MODELS),
    )
    provider._resolved_default_model = models.FALLBACK_MODELS[0]["slug"]
    return provider


@pytest.fixture(params=[CODEX_MODE, plan_auth.MODE])
def provider(request):
    return _make_provider(request.param)


def _install_transport(monkeypatch, handler):
    constructions = []

    def client(**kwargs):
        constructions.append(kwargs)
        return _ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return constructions


def _ids(result):
    assert set(result) == {"version", "source", "models"}
    assert result["version"] == 1
    assert result["source"] == "provider-api"
    return [model.id for model in result["models"]]


def test_optional_contract_has_no_required_arguments():
    assert inspect.iscoroutinefunction(ChatGPTProvider.discover_models)
    assert list(inspect.signature(ChatGPTProvider.discover_models).parameters) == ["self"]


@pytest.mark.asyncio
async def test_exact_ids_empty_and_repeated_calls_are_fresh(provider, monkeypatch):
    entries = [
        _entry("account-zeta", additional_speed_tiers=["fast"], priority=99),
        _entry("literal-fast", additional_speed_tiers=["fast"], priority=1),
        _entry("hidden", visibility="hide"),
    ]
    original = copy.deepcopy(entries)
    payloads = [
        {"models": entries},
        {"models": []},
        {"models": [_entry("new-account-model")]},
    ]
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=payloads[len(requests) - 1])

    constructions = _install_transport(monkeypatch, handle)
    cached = provider._models_cache
    resolved = provider._resolved_default_model
    first = await provider.discover_models()
    assert _ids(first) == ["account-zeta", "literal-fast"]
    assert first["models"][0].display_name == "Display account-zeta"
    assert first["models"][0].context_window == 64_000
    assert first["models"][0].max_output_tokens == 8_000
    assert _ids(await provider.discover_models()) == []
    assert _ids(await provider.discover_models()) == ["new-account-model"]
    assert len(requests) == len(constructions) == 3
    assert provider._ensure_valid_tokens.await_count == 3
    assert provider._models_cache is cached
    assert provider._resolved_default_model == resolved
    assert entries == original

    for request in requests:
        assert request.method == "GET"
        assert request.content == b""
        assert request.headers["authorization"] == "Bearer fixture-access-token"
        assert request.headers["accept"] == "application/json"
        if provider.auth_mode == CODEX_MODE:
            assert str(request.url.copy_with(query=None)) == models.MODELS_ENDPOINT
            assert dict(request.url.params) == {"client_version": "fixture-version"}
            assert request.headers["chatgpt-account-id"] == "fixture-account"
            assert request.headers["openai-beta"] == "responses=v1"
            assert request.headers["openai-originator"] == "codex"
        else:
            assert str(request.url) == plan_auth.RESOURCE + "/models"
            assert "chatgpt-account-id" not in request.headers
            assert "openai-beta" not in request.headers
            assert "openai-originator" not in request.headers
    for kwargs in constructions:
        assert kwargs["timeout"] == 30.0
        if provider.auth_mode == plan_auth.MODE:
            assert kwargs["follow_redirects"] is False


@pytest.mark.asyncio
async def test_discovery_never_reads_or_writes_legacy_caches(provider, monkeypatch):
    class GuardedProvider(ChatGPTProvider):
        def __getattribute__(self, name):
            if name in {"_models_cache", "_resolved_default_model", "_models_lock"}:
                raise AssertionError(f"Strict discovery read {name}")
            return super().__getattribute__(name)

        def __setattr__(self, name, value):
            if name in {"_models_cache", "_resolved_default_model", "_models_lock"}:
                raise AssertionError(f"Strict discovery wrote {name}")
            super().__setattr__(name, value)

    # Guard only after initialization; normal inference initialization is unchanged.
    provider.__class__ = GuardedProvider
    provider._get_catalog = AsyncMock(side_effect=AssertionError("legacy catalog"))
    provider._resolve_default_model = AsyncMock(side_effect=AssertionError("selection"))
    _install_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"models": []})
    )
    assert _ids(await provider.discover_models()) == []


@pytest.mark.asyncio
async def test_selection_fields_do_not_change_inventory_or_request(provider, monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": [_entry("live-account-model")]})

    _install_transport(monkeypatch, handle)
    before = await provider.discover_models()
    provider.config.update(
        model="different-selection",
        default_model="latest",
        reasoning_effort="low",
        priority=-20,
    )
    provider.default_model = "latest"
    provider.reasoning_effort = "low"
    provider.priority = -20
    after = await provider.discover_models()
    assert before == after
    assert requests[0].url == requests[1].url
    assert requests[0].headers == requests[1].headers


@pytest.mark.asyncio
async def test_auth_refresh_scope_is_used_on_each_call(provider, monkeypatch):
    provider = _make_provider(provider.auth_mode, inject_tokens=False)
    del provider._ensure_valid_tokens  # Exercise the actual helper and validator.
    tokens = [
        _valid_codex_tokens("first"),
        _valid_codex_tokens("second"),
    ]
    if provider.auth_mode == CODEX_MODE:
        storage = Mock(side_effect=tokens)
        monkeypatch.setattr(provider_module, "load_tokens", storage)
    else:
        storage = AsyncMock(side_effect=tokens)
        monkeypatch.setattr(plan_auth, "ensure_tokens", storage)
    cached = provider._models_cache
    resolved = provider._resolved_default_model
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    _install_transport(monkeypatch, handle)
    await provider.discover_models()
    await provider.discover_models()
    assert len(requests) == 2
    assert [r.headers["authorization"] for r in requests] == [
        "Bearer first-token",
        "Bearer second-token",
    ]
    if provider.auth_mode == CODEX_MODE:
        assert [r.headers["chatgpt-account-id"] for r in requests] == [
            "first-account",
            "second-account",
        ]
        assert storage.call_args_list == [
            call(path="/unused/discovery-credentials.json"),
            call(path="/unused/discovery-credentials.json"),
        ]
    else:
        assert storage.await_args_list == [
            call("/unused/discovery-credentials.json"),
            call("/unused/discovery-credentials.json"),
        ]
    assert provider._models_cache is cached
    assert provider._resolved_default_model == resolved
    provider_module.refresh_tokens.assert_not_awaited()


def _valid_codex_tokens(account):
    return {
        "access_token": f"{account}-token",
        "account_id": f"{account}-account",
        "refresh_token": f"{account}-refresh",
        "expires_at": "2999-01-01T00:00:00+00:00",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [None, {}, {"access_token": "invalid"}])
async def test_removed_or_invalid_file_cannot_reuse_previous_account(
    monkeypatch, replacement
):
    provider = _make_provider(CODEX_MODE, inject_tokens=False)
    del provider._ensure_valid_tokens
    storage = Mock(side_effect=[_valid_codex_tokens("first"), replacement])
    monkeypatch.setattr(provider_module, "load_tokens", storage)
    cached = provider._models_cache
    resolved = provider._resolved_default_model
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    _install_transport(monkeypatch, handle)
    assert _ids(await provider.discover_models()) == []
    with pytest.raises(llm_errors.AuthenticationError, match="configured credential file"):
        await provider.discover_models()
    assert len(requests) == 1
    assert storage.call_count == 2
    provider_module.refresh_tokens.assert_not_awaited()
    assert provider._models_cache is cached
    assert provider._resolved_default_model == resolved


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_succeeds", [False, True])
async def test_fresh_discovery_refreshes_only_current_file_account(
    monkeypatch, refresh_succeeds
):
    provider = _make_provider(CODEX_MODE, inject_tokens=False)
    del provider._ensure_valid_tokens
    expired_second = dict(
        _valid_codex_tokens("second"), expires_at="2000-01-01T00:00:00+00:00"
    )
    storage = Mock(side_effect=[_valid_codex_tokens("first"), expired_second])
    monkeypatch.setattr(provider_module, "load_tokens", storage)
    refresh = AsyncMock(
        return_value=_valid_codex_tokens("second") if refresh_succeeds else None
    )
    monkeypatch.setattr(provider_module, "refresh_tokens", refresh)
    cached = provider._models_cache
    resolved = provider._resolved_default_model
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    _install_transport(monkeypatch, handle)
    await provider.discover_models()
    if refresh_succeeds:
        await provider.discover_models()
        assert len(requests) == 2
        assert requests[1].headers["authorization"] == "Bearer second-token"
        assert requests[1].headers["chatgpt-account-id"] == "second-account"
    else:
        with pytest.raises(llm_errors.AuthenticationError):
            await provider.discover_models()
        assert len(requests) == 1
    refresh.assert_awaited_once_with(
        "second-refresh", path="/unused/discovery-credentials.json"
    )
    assert storage.call_count == 2
    assert provider._models_cache is cached
    assert provider._resolved_default_model == resolved


@pytest.mark.asyncio
@pytest.mark.parametrize("disk_tokens", [None, _valid_codex_tokens("different")])
async def test_explicit_codex_token_injection_keeps_precedence(monkeypatch, disk_tokens):
    provider = _make_provider(CODEX_MODE)
    del provider._ensure_valid_tokens
    storage = Mock(return_value=disk_tokens)
    monkeypatch.setattr(provider_module, "load_tokens", storage)
    cached = provider._models_cache
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    _install_transport(monkeypatch, handle)
    await provider.discover_models()
    await provider.discover_models()
    assert len(requests) == 2
    for request in requests:
        assert request.headers["authorization"] == "Bearer fixture-access-token"
        assert request.headers["chatgpt-account-id"] == "fixture-account"
    assert storage.call_count == 2
    assert provider._models_cache is cached
    provider_module.refresh_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_does_not_switch_auth_mode_for_replaced_file(monkeypatch):
    provider = _make_provider(CODEX_MODE, inject_tokens=False)
    del provider._ensure_valid_tokens
    monkeypatch.setattr(
        provider_module,
        "load_tokens",
        Mock(return_value={"auth_mode": plan_auth.MODE, **_valid_codex_tokens("plan")}),
    )
    cached = provider._models_cache
    with pytest.raises(llm_errors.AuthenticationError, match="auth_mode=chatgpt_plan"):
        await provider.discover_models()
    assert provider.auth_mode == CODEX_MODE
    assert provider._models_cache is cached
    httpx.AsyncClient.assert_not_called()
    plan_auth.ensure_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_token_helper_keeps_memory_precedence(monkeypatch):
    provider = _make_provider(CODEX_MODE, inject_tokens=False)
    del provider._ensure_valid_tokens
    first = _valid_codex_tokens("first")
    storage = Mock(return_value=first)
    monkeypatch.setattr(provider_module, "load_tokens", storage)
    await provider._ensure_valid_tokens()
    assert provider._tokens is first
    storage.return_value = _valid_codex_tokens("second")
    await provider._ensure_valid_tokens()
    assert provider._tokens is first
    assert storage.call_count == 3  # Initial mode + load, then legacy mode check.
    provider_module.refresh_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_configured_auth_path_is_preserved(provider, monkeypatch):
    # Exercise the real auth selector, but stub its storage/refresh boundary.
    del provider._ensure_valid_tokens
    if provider.auth_mode == plan_auth.MODE:
        ensure = AsyncMock(return_value=provider._tokens)
        monkeypatch.setattr(plan_auth, "ensure_tokens", ensure)
    else:
        load = Mock(return_value=provider._tokens)
        monkeypatch.setattr(provider_module, "load_tokens", load)
        monkeypatch.setattr(provider_module, "is_token_valid", lambda tokens: True)
    _install_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"models": []})
    )
    assert _ids(await provider.discover_models()) == []
    if provider.auth_mode == plan_auth.MODE:
        ensure.assert_awaited_once_with("/unused/discovery-credentials.json")
    else:
        load.assert_called_once_with(path="/unused/discovery-credentials.json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [
        (401, llm_errors.AuthenticationError),
        (403, llm_errors.AccessDeniedError),
        (404, llm_errors.NotFoundError),
        (429, llm_errors.RateLimitError),
        (503, llm_errors.ProviderUnavailableError),
    ],
)
async def test_http_failure_is_typed_not_empty_or_fallback(
    provider, monkeypatch, status, expected
):
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            status, json={"detail": "unavailable"}, headers={"retry-after": "5"}
        ),
    )
    cached = provider._models_cache
    with pytest.raises(expected) as error:
        await provider.discover_models()
    assert error.value.status_code == status
    assert error.value.provider == provider.name
    assert provider._models_cache is cached


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,expected",
    [
        (httpx.ConnectError, llm_errors.ProviderUnavailableError),
        (httpx.RemoteProtocolError, llm_errors.ProviderUnavailableError),
        (httpx.ReadTimeout, llm_errors.LLMTimeoutError),
    ],
)
async def test_transport_failure_is_typed_without_fallback(
    provider, monkeypatch, failure, expected
):
    def handle(request):
        raise failure("fixture transport failure", request=request)

    _install_transport(monkeypatch, handle)
    cached = provider._models_cache
    with pytest.raises(expected) as error:
        await provider.discover_models()
    assert error.value.retryable is True
    assert provider._models_cache is cached


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"models": None},
        {"models": {}},
        {"models": [None]},
        {"models": [{"visibility": "list", "supported_in_api": True}]},
        {"models": [_entry("")]},
        {"models": [_entry("  ")]},
        {"models": [_entry(123)]},
        {"models": [_entry("valid"), None]},
        {"models": [_entry("invalid-metadata", context_window={})]},
        {"models": [_entry("invalid-visibility", visibility=[])]},
    ],
)
async def test_malformed_catalog_is_not_success(provider, monkeypatch, payload):
    # Explicit content handles the JSON null case (httpx's json=None means no body).
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, text=json.dumps(payload))

    _install_transport(monkeypatch, handle)
    cached = provider._models_cache
    with pytest.raises(llm_errors.LLMError) as error:
        await provider.discover_models()
    assert error.value.retryable is False
    assert isinstance(error.value.__cause__, (ValueError, TypeError))
    assert len(calls) == 1
    assert provider._models_cache is cached


@pytest.mark.asyncio
@pytest.mark.parametrize("support", [None, "true", 1])
async def test_codex_malformed_support_flag_is_not_empty(monkeypatch, support):
    provider = _make_provider(CODEX_MODE)
    entry = _entry("malformed-support", supported_in_api=support)
    if support is None:
        del entry["supported_in_api"]
    _install_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"models": [entry]})
    )
    with pytest.raises(llm_errors.LLMError) as error:
        await provider.discover_models()
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_filtered_empty_catalog_remains_empty(provider, monkeypatch):
    entries = [_entry("hidden", visibility="hide")]
    if provider.auth_mode == CODEX_MODE:
        entries.append(_entry("unavailable-via-api", supported_in_api=False))
    else:
        entries.append({"slug": "not-listed"})
    _install_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"models": entries})
    )
    assert _ids(await provider.discover_models()) == []


@pytest.mark.asyncio
async def test_failure_does_not_prevent_fresh_discovery(provider, monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(200, json={"models": [_entry("recovered-model")]})

    _install_transport(monkeypatch, handle)
    cached = provider._models_cache
    with pytest.raises(llm_errors.ProviderUnavailableError):
        await provider.discover_models()
    assert _ids(await provider.discover_models()) == ["recovered-model"]
    assert len(calls) == 2
    assert provider._models_cache is cached


@pytest.mark.asyncio
async def test_invalid_json_is_typed(provider, monkeypatch):
    _install_transport(
        monkeypatch, lambda request: httpx.Response(200, text="not-json")
    )
    with pytest.raises(llm_errors.LLMError):
        await provider.discover_models()


@pytest.mark.asyncio
async def test_auth_failure_propagates_unchanged_before_http(provider):
    failure = llm_errors.AuthenticationError("Login required", provider=provider.name)
    provider._ensure_valid_tokens = AsyncMock(side_effect=failure)
    with pytest.raises(llm_errors.AuthenticationError) as error:
        await provider.discover_models()
    assert error.value is failure
    httpx.AsyncClient.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_is_not_relabelled(provider, monkeypatch):
    def handle(request):
        raise asyncio.CancelledError()

    _install_transport(monkeypatch, handle)
    with pytest.raises(asyncio.CancelledError):
        await provider.discover_models()


@pytest.mark.asyncio
async def test_legacy_list_cache_and_aliases_are_unchanged(provider, monkeypatch):
    cached = provider._models_cache
    assert await provider.list_models() is cached[1]
    httpx.AsyncClient.assert_not_called()
    provider._models_cache = None
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json={"models": [_entry("live", additional_speed_tiers=["fast"])]}
        )

    _install_transport(monkeypatch, handle)
    listed = await provider.list_models()
    assert [m.id for m in listed] == (
        ["live", "live-fast"] if provider.auth_mode == CODEX_MODE else ["live"]
    )
    assert await provider.list_models() is listed
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_legacy_empty_and_failure_contract_is_unchanged(provider, monkeypatch, empty):
    provider._models_cache = None
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200 if empty else 503,
            json={"models": []} if empty else {"detail": "unavailable"},
        ),
    )
    if provider.auth_mode == CODEX_MODE:
        assert await provider.list_models() == models.to_model_infos(models.FALLBACK_MODELS)
    else:
        with pytest.raises(ValueError if empty else llm_errors.ProviderUnavailableError):
            await provider.list_models()
    assert provider._models_cache is None


@pytest.mark.asyncio
async def test_legacy_codex_fetch_still_raises_value_error(monkeypatch):
    _install_transport(
        monkeypatch, lambda request: httpx.Response(401, text="Unauthorized request")
    )
    with pytest.raises(ValueError, match="401"):
        await models.fetch_models(access_token="fixture-token", account_id="fixture")