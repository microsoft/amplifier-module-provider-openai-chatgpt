"""Public Responses wire and completion contract for ChatGPT plan mode."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai_chatgpt import mount, plan_auth
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider


def provider():
    return ChatGPTProvider(
        {
            "auth_mode": "chatgpt_plan",
            "token_file_path": "/unused/plan.json",
            "default_model": "account-model",
            "extra_request_params": {
                "store": True,
                "stream": False,
                "background": True,
            },
        },
        tokens={"auth_mode": "chatgpt_plan", "access_token": "opaque-plan-token"},
    )


def install_transport(monkeypatch, handle):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )


def stream_events(*events):
    return "\n".join("data: " + json.dumps(e) + "\n" for e in events)


@pytest.mark.asyncio
async def test_public_inference_wire_and_terminal_success(monkeypatch):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()

    def handle(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer opaque-plan-token"
        assert "chatgpt-account-id" not in request.headers
        assert "openai-originator" not in request.headers
        body = json.loads(request.content)
        assert body["store"] is False and body["stream"] is True
        assert "background" not in body
        assert body["model"] == "account-model-fast"  # Literal slug, never remapped.
        assert "service_tier" not in body
        return httpx.Response(
            200,
            text=stream_events(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-real",
                        "model": "account-model-fast",
                        "usage": {"input_tokens": 10, "output_tokens": 3},
                    },
                },
            ),
        )

    install_transport(monkeypatch, handle)
    result = await p.complete(
        ChatRequest(
            model="account-model-fast", messages=[Message(role="user", content="hello")]
        )
    )
    assert result.content[0].text == "hello"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "response.done", "response": {"usage": {}}}],
        [{"type": "response.output_text.delta", "delta": "partial"}],
    ],
)
async def test_interrupted_public_stream_is_never_success(monkeypatch, events):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()
    install_transport(
        monkeypatch, lambda request: httpx.Response(200, text=stream_events(*events))
    )
    with pytest.raises(llm_errors.LLMError, match="before response.completed"):
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))


@pytest.mark.asyncio
async def test_public_catalog_filters_preserves_order_and_does_not_synthesize_models(
    monkeypatch,
):
    def handle(request):
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert {k for k in request.headers if k.startswith("openai")} == set()
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "second-mini",
                        "display_name": "Fast choice",
                        "visibility": "list",
                        "additional_speed_tiers": ["fast"],
                    },
                    {"slug": "hidden", "visibility": "hide"},
                    {"slug": "not-listed"},
                    {"slug": "first", "visibility": "list"},
                ]
            },
        )

    install_transport(monkeypatch, handle)
    p = provider()
    p._ensure_valid_tokens = AsyncMock()
    models = await p.list_models()
    assert [m.id for m in models] == ["second-mini", "first"]
    assert models[0].display_name == "Fast choice"
    p.default_model = "latest"
    assert await p._resolve_default_model() == "second-mini"


@pytest.mark.asyncio
async def test_public_catalog_failure_has_no_legacy_fallback(monkeypatch):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            503,
            json={"detail": "temporarily unavailable"},
            headers={"x-request-id": "req-catalog"},
        ),
    )
    with pytest.raises(llm_errors.ProviderUnavailableError) as error:
        await p.list_models()
    assert error.value.request_id == "req-catalog"
    assert p._models_cache is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code,klass,retryable",
    [
        (
            429,
            "subscription_sharing_usage_limit_exceeded",
            llm_errors.RateLimitError,
            False,
        ),
        (
            503,
            "subscription_sharing_usage_unavailable",
            llm_errors.ProviderUnavailableError,
            True,
        ),
        (
            403,
            "subscription_sharing_user_not_eligible",
            llm_errors.AccessDeniedError,
            False,
        ),
        (
            401,
            "subscription_sharing_invalid_user",
            llm_errors.AuthenticationError,
            False,
        ),
        (
            400,
            "subscription_sharing_unsupported_capability",
            llm_errors.InvalidRequestError,
            False,
        ),
    ],
)
async def test_plan_errors_preserve_code_parameter_request_id_without_retry(
    monkeypatch, status, code, klass, retryable
):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={
                "error": {
                    "code": code,
                    "param": "tools",
                    "message": "Plan request blocked",
                }
            },
            headers={"x-request-id": "req-one"},
        )

    install_transport(monkeypatch, handle)
    with pytest.raises(klass) as error:
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert error.value.error_code == code
    assert error.value.error_param == "tools"
    assert error.value.request_id == "req-one"
    assert error.value.retryable is retryable
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_post_stream_limit_is_not_completion_or_retry(monkeypatch):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            text=stream_events(
                {"type": "response.output_text.delta", "delta": "partial"},
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "code": "subscription_sharing_usage_limit_exceeded",
                            "message": "App usage limit",
                        }
                    },
                },
            ),
        ),
    )
    with pytest.raises(llm_errors.RateLimitError) as error:
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert error.value.retryable is False
    assert "https://chatgpt.com/settings/usage" in str(error.value)
    assert error.value.error_code == "subscription_sharing_usage_limit_exceeded"


def test_mode_is_explicit_and_plan_metadata_has_no_codex_fallback():
    with pytest.raises(ValueError, match="authentication mode"):
        ChatGPTProvider({}, tokens={"auth_mode": "chatgpt_plan"})
    with pytest.raises(ValueError, match="authentication mode"):
        ChatGPTProvider({"auth_mode": "chatgpt_plan"}, tokens={"auth_mode": "oauth"})
    p = ChatGPTProvider({"auth_mode": "chatgpt_plan"})
    assert "auth:oauth_pkce" in p.get_info().capabilities
    assert "gpt-5.6" not in p.get_info().defaults["model"]
    assert "context_window" not in p.get_info().defaults


@pytest.mark.asyncio
async def test_background_mount_never_opens_consent(monkeypatch):
    monkeypatch.setattr(
        plan_auth,
        "ensure_tokens",
        AsyncMock(side_effect=plan_auth.PlanAuthError("Sign in required")),
    )
    login = AsyncMock()
    monkeypatch.setattr(plan_auth, "login", login)
    assert (
        await mount(MagicMock(), {"auth_mode": "chatgpt_plan", "login_on_mount": True})
        is None
    )
    assert not login.called
