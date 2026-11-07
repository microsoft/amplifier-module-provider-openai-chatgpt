"""Strict discovery through the real fetch/mapping code, with no external I/O."""

import asyncio
import copy
import inspect
import json
import time
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from amplifier_core import ModelInfo, llm_errors
from amplifier_core.message_models import ChatRequest, Message

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
        provider_module,
        "refresh_tokens_strict",
        AsyncMock(side_effect=AssertionError("Strict refresh must be stubbed")),
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
    assert isinstance(result, list)
    return [model.id for model in result]


def test_canonical_contract_has_no_required_arguments():
    assert not hasattr(ChatGPTProvider, "discover_models")
    assert inspect.iscoroutinefunction(ChatGPTProvider.list_models)
    parameters = inspect.signature(ChatGPTProvider.list_models).parameters
    assert list(parameters) == ["self", "filters"]
    assert parameters["filters"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["filters"].default is None


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
    first = await provider.list_models(filters={})
    assert _ids(first) == ["account-zeta", "literal-fast"]
    assert first[0].display_name == "Display account-zeta"
    assert first[0].context_window == 64_000
    assert first[0].max_output_tokens == 8_000
    assert _ids(await provider.list_models(filters={})) == []
    assert _ids(await provider.list_models(filters={})) == ["new-account-model"]
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
    assert _ids(await provider.list_models(filters={})) == []


@pytest.mark.asyncio
async def test_selection_fields_do_not_change_inventory_or_request(provider, monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": [_entry("live-account-model")]})

    _install_transport(monkeypatch, handle)
    before = await provider.list_models(filters={})
    provider.config.update(
        model="different-selection",
        default_model="latest",
        reasoning_effort="low",
        priority=-20,
    )
    provider.default_model = "latest"
    provider.reasoning_effort = "low"
    provider.priority = -20
    after = await provider.list_models(filters={})
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
    await provider.list_models(filters={})
    await provider.list_models(filters={})
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
    assert _ids(await provider.list_models(filters={})) == []
    with pytest.raises(llm_errors.AuthenticationError, match="configured credential file"):
        await provider.list_models(filters={})
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
        return_value=_valid_codex_tokens("second"),
        side_effect=None if refresh_succeeds else llm_errors.AuthenticationError(
            "Synthetic revoked refresh", provider="openai-chatgpt", retryable=False
        ),
    )
    monkeypatch.setattr(provider_module, "refresh_tokens_strict", refresh)
    cached = provider._models_cache
    resolved = provider._resolved_default_model
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"models": []})

    _install_transport(monkeypatch, handle)
    await provider.list_models(filters={})
    if refresh_succeeds:
        await provider.list_models(filters={})
        assert len(requests) == 2
        assert requests[1].headers["authorization"] == "Bearer second-token"
        assert requests[1].headers["chatgpt-account-id"] == "second-account"
    else:
        with pytest.raises(llm_errors.AuthenticationError):
            await provider.list_models(filters={})
        assert len(requests) == 1
    refresh.assert_awaited_once_with(
        expired_second, path="/unused/discovery-credentials.json", persist=True
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
    await provider.list_models(filters={})
    await provider.list_models(filters={})
    assert len(requests) == 2
    for request in requests:
        assert request.headers["authorization"] == "Bearer fixture-access-token"
        assert request.headers["chatgpt-account-id"] == "fixture-account"
    storage.assert_not_called()
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
        await provider.list_models(filters={})
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
    assert _ids(await provider.list_models(filters={})) == []
    if provider.auth_mode == plan_auth.MODE:
        ensure.assert_awaited_once_with("/unused/discovery-credentials.json")
    else:
        load.assert_not_called()  # Explicit credentials do not consult the file.


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
        await provider.list_models(filters={})
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
        await provider.list_models(filters={})
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
        {"models": [_entry("gpt-6-partial")], "has_more": True},
        {"models": [], "has_more": "false"},
        {"models": [], "has_more": None},
        {"models": [_entry("gpt-6-partial")], "next_cursor": "another-page"},
        {"models": [_entry("gpt-6-partial")], "next_page": "another-page"},
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
        await provider.list_models(filters={})
    assert error.value.retryable is False
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__  # Malformed content can contain secrets.
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
        await provider.list_models(filters={})
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


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
    assert _ids(await provider.list_models(filters={})) == []


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
        await provider.list_models(filters={})
    assert _ids(await provider.list_models(filters={})) == ["recovered-model"]
    assert len(calls) == 2
    assert provider._models_cache is cached


@pytest.mark.asyncio
async def test_invalid_json_is_typed(provider, monkeypatch):
    _install_transport(
        monkeypatch, lambda request: httpx.Response(200, text="not-json")
    )
    with pytest.raises(llm_errors.LLMError):
        await provider.list_models(filters={})


@pytest.mark.asyncio
async def test_auth_failure_propagates_unchanged_before_http(provider):
    failure = llm_errors.AuthenticationError("Login required", provider=provider.name)
    provider._ensure_valid_tokens = AsyncMock(side_effect=failure)
    with pytest.raises(llm_errors.AuthenticationError) as error:
        await provider.list_models(filters={})
    assert error.value is failure
    httpx.AsyncClient.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_is_not_relabelled(provider, monkeypatch):
    def handle(request):
        raise asyncio.CancelledError()

    _install_transport(monkeypatch, handle)
    with pytest.raises(asyncio.CancelledError):
        await provider.list_models(filters={})


@pytest.mark.asyncio
async def test_inference_catalog_cache_and_aliases_are_unchanged(provider, monkeypatch):
    cached = provider._models_cache
    assert await provider._get_catalog() is cached[1]
    httpx.AsyncClient.assert_not_called()
    provider._models_cache = None
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json={"models": [_entry("live", additional_speed_tiers=["fast"])]}
        )

    _install_transport(monkeypatch, handle)
    listed = await provider._get_catalog()
    assert [m.id for m in listed] == (
        ["live", "live-fast"] if provider.auth_mode == CODEX_MODE else ["live"]
    )
    assert await provider._get_catalog() is listed
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_inference_empty_and_failure_contract_is_unchanged(provider, monkeypatch, empty):
    provider._models_cache = None
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200 if empty else 503,
            json={"models": []} if empty else {"detail": "unavailable"},
        ),
    )
    if provider.auth_mode == CODEX_MODE:
        assert await provider._get_catalog() == models.to_model_infos(models.FALLBACK_MODELS)
    else:
        with pytest.raises(ValueError if empty else llm_errors.ProviderUnavailableError):
            await provider._get_catalog()
    assert provider._models_cache is None


@pytest.mark.asyncio
async def test_legacy_codex_fetch_still_raises_value_error(monkeypatch):
    _install_transport(
        monkeypatch, lambda request: httpx.Response(401, text="Unauthorized request")
    )
    with pytest.raises(ValueError, match="401"):
        await models.fetch_models(access_token="fixture-token", account_id="fixture")


@pytest.mark.asyncio
async def test_default_gpt6_exact_account_ids_and_unknown_limits(provider, monkeypatch):
    returned = [
        "gpt-6-astra", "gpt-6.1-sol", "gpt-6-preview", "GpT-6-new-tier",
        "gpt-6-astra-2026-10-01", "gpt-6-literal-fast", "gpt-5.4", "unknown-future",
    ]
    entries = [_entry(slug, additional_speed_tiers=["fast"]) for slug in returned]
    del entries[3]["context_window"]
    del entries[3]["max_output_tokens"]
    _install_transport(
        monkeypatch, lambda _: httpx.Response(200, json={"models": entries})
    )
    rows = await provider.list_models()
    assert all(isinstance(row, ModelInfo) for row in rows)
    assert _ids(rows) == returned[:6]
    assert rows[3].context_window == rows[3].max_output_tokens == 0
    assert rows[3].capabilities == [] and rows[3].defaults == {}
    assert _ids(await provider.list_models(filters=None)) == returned[:6]
    assert _ids(await provider.list_models(filters={})) == returned
    # Nothing fabricated from additional_speed_tiers, even for a literal -fast ID.
    assert "gpt-6-astra-fast" not in _ids(rows)
    assert "gpt-6-literal-fast-fast" not in _ids(rows)


# Common conformance vectors from OpenAI PR141, commit 38b7b26.
@pytest.mark.asyncio
@pytest.mark.parametrize("selector,expected", [
    ({"include": ["gpt-5*", "o3-*"], "exclude": ["*-deep-research"]},
     {"gpt-5.6", "o3-mini"}),
    ({"include": []}, set()),
    ({"exclude": ["gpt-*"]}, {"o3-mini", "o3-deep-research"}),
    ({"include": ["GPT-6*"]}, {"gpt-6-solx"}),
    ({"include": ["gpt-6-????"]}, {"gpt-6-solx"}),
    ({"include": ["gpt-6-[st]*"], "exclude": ["GPT-6*"]}, set()),
])
async def test_custom_filters_replace_defaults(provider, monkeypatch, selector, expected):
    entries = [_entry(slug) for slug in [
        "gpt-5.6", "gpt-6-solx", "o3-mini", "o3-deep-research",
    ]]
    _install_transport(
        monkeypatch, lambda _: httpx.Response(200, json={"models": entries})
    )
    assert set(_ids(await provider.list_models(filters=selector))) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("selector,expected", [
    ({"hide_dated": True},
     {"gpt-6-sol", "gpt-6-only-2026-10-01", "gpt-6-preview"}),
    ({"hide_dated": True, "exclude": ["gpt-6-sol"]},
     {"gpt-6-sol-2026-10-01", "gpt-6-only-2026-10-01", "gpt-6-preview"}),
    ({"include": ["*-2026-10-01"], "hide_dated": True},
     {"gpt-6-sol-2026-10-01", "gpt-6-only-2026-10-01"}),
])
async def test_dated_alias_conformance(provider, monkeypatch, selector, expected):
    entries = [_entry(slug) for slug in [
        "gpt-6-sol", "gpt-6-sol-2026-10-01", "gpt-6-only-2026-10-01", "gpt-6-preview",
    ]]
    _install_transport(
        monkeypatch, lambda _: httpx.Response(200, json={"models": entries})
    )
    assert set(_ids(await provider.list_models(filters=selector))) == expected


@pytest.mark.asyncio
async def test_config_and_metadata_are_immutable_snapshots(monkeypatch, provider):
    config = {"include": ["gpt-5*"], "exclude": [], "hide_dated": True}
    p = ChatGPTProvider(
        {"auth_mode": provider.auth_mode, "model_filters": config},
        tokens=provider._tokens,
    )
    p._ensure_valid_tokens = AsyncMock()
    info = p.get_model_list_info()
    assert set(info) == {
        "version", "source", "filters_keyword", "filters_schema",
        "default_filters", "unfiltered", "override",
    }
    assert info["version"] == 1 and info["source"] == "provider-api"
    assert info["filters_keyword"] == "filters" and info["override"] == "replace"
    assert info["unfiltered"] == {}
    assert provider.get_model_list_info()["default_filters"] == {
        "include": ["gpt-6*"], "exclude": [], "hide_dated": False,
    }
    httpx.AsyncClient.assert_not_called()
    config["include"].append("*")
    p.config["model_filters"]["exclude"].append("*")
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    returned = ["gpt-5.4", "gpt-5.4-2026-10-01", "gpt-6-sol", "future"]
    _install_transport(monkeypatch, lambda _: httpx.Response(
        200, json={"models": [_entry(slug) for slug in returned]}
    ))
    assert _ids(await p.list_models()) == ["gpt-5.4"]
    assert _ids(await p.list_models(filters={})) == returned
    assert _ids(await p.list_models(filters={"include": []})) == []
    assert p.get_model_list_info()["default_filters"]["include"] == ["gpt-5*"]
    assert p.get_model_list_info()["filters_schema"]["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", [
    False, "all", [], {"include": "*"}, {"include": [None]}, {"include": [""]},
    {"include": [" "]}, {"exclude": [1]}, {"hide_dated": "false"},
    {"hide_dated": None}, {"typo": True},
])
async def test_invalid_filters_fail_before_auth_or_io(provider, filters):
    with pytest.raises((TypeError, ValueError)):
        ChatGPTProvider({"model_filters": filters})
    with pytest.raises((TypeError, ValueError)):
        await provider.list_models(filters=filters)
    provider._ensure_valid_tokens.assert_not_awaited()
    httpx.AsyncClient.assert_not_called()


def test_explicit_null_config_is_invalid():
    with pytest.raises(TypeError):
        ChatGPTProvider({"model_filters": None})


@pytest.mark.asyncio
async def test_override_snapshots_before_await_and_does_not_mutate_defaults(
    provider, monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(_):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"models": [
            _entry("gpt-5.4"), _entry("GpT-6-Preview"),
        ]})

    _install_transport(monkeypatch, handle)
    custom = {"include": ["gpt-5*"]}
    task = asyncio.create_task(provider.list_models(filters=custom))
    await entered.wait()
    custom["include"][:] = ["*"]
    second = asyncio.create_task(provider.list_models())
    release.set()
    assert _ids(await task) == ["gpt-5.4"]
    assert _ids(await second) == ["GpT-6-Preview"]
    assert _ids(await provider.list_models(filters={
        "include": ["gPt-6*"], "exclude": ["*PREVIEW"],
    })) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["context_window", "max_output_tokens"])
@pytest.mark.parametrize("value", [True, -1, "128000", {}, []])
async def test_malformed_limits_never_become_fabricated_metadata(
    provider, monkeypatch, key, value,
):
    _install_transport(monkeypatch, lambda _: httpx.Response(
        200, json={"models": [_entry("gpt-6-sol", **{key: value})]}
    ))
    with pytest.raises(llm_errors.LLMError) as caught:
        await provider.list_models()
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_explicit_older_inference_survives_listing_policy(provider, monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"models": [
                _entry("gpt-5.4"), _entry("gpt-6-sol"),
            ]})
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.4"
        assert payload["reasoning"]["effort"] == "high"
        assert "service_tier" not in payload
        return httpx.Response(200, text="data: " + json.dumps({
            "type": "response.completed",
            "response": {"id": "fixture-response", "model": "gpt-5.4",
                         "output": [], "usage": {"input_tokens": 1, "output_tokens": 0}},
        }) + "\n\n")

    _install_transport(monkeypatch, handle)
    assert _ids(await provider.list_models()) == ["gpt-6-sol"]
    assert _ids(await provider.list_models(filters={"include": ["gpt-5*"]})) == ["gpt-5.4"]
    await provider.complete(ChatRequest(
        model="gpt-5.4", messages=[Message(role="user", content="fixture")],
    ))
    assert [request.method for request in requests] == ["GET", "GET", "POST"]