"""Public Responses wire and completion contract for ChatGPT plan mode."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ToolResultBlock,
    ToolSpec,
)

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


@pytest.mark.parametrize("mode", [None, "chatgpt_codex", "legacy_codex"])
def test_codex_is_default_and_old_mode_is_compatibility_alias(mode):
    p = ChatGPTProvider({} if mode is None else {"auth_mode": mode})
    info = p.get_info()
    assert p.auth_mode == info.defaults["auth_mode"] == "chatgpt_codex"
    assert "auth:oauth_device_code" in info.capabilities
    fields = {field.id: field.model_dump() for field in info.config_fields}
    assert fields["auth_mode"]["choices"] == ["chatgpt_codex", "chatgpt_plan"]
    assert fields["auth_mode"]["default"] == "chatgpt_codex"
    assert fields["token_file_path"]["default"] is None
    assert fields["token_file_path"]["required"] is False
    assert fields["host_file_path"]["show_when"] == {"auth_mode": "chatgpt_plan"}


@pytest.mark.asyncio
async def test_plan_provider_login_passes_actual_application_name(
    monkeypatch, tmp_path
):
    login = AsyncMock(return_value={"access_token": "test"})
    monkeypatch.setattr(plan_auth, "login", login)
    p = ChatGPTProvider(
        {
            "auth_mode": "chatgpt_plan",
            "app_name": "Example Host",
            "token_file_path": str(tmp_path / "plan.json"),
        }
    )
    await p.login()
    assert login.call_args.kwargs["app_name"] == "Example Host"


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("path", ["", "   ", 17, False])
def test_explicit_invalid_path_never_selects_default_account(mode, path):
    with pytest.raises(ValueError, match="non-empty path"):
        ChatGPTProvider({"auth_mode": mode, "token_file_path": path})


@pytest.mark.asyncio
async def test_codex_login_never_overwrites_plan_profile(monkeypatch, tmp_path):
    import amplifier_module_provider_openai_chatgpt.provider as mod

    path = tmp_path / "profile.json"
    original = json.dumps({"auth_mode": "chatgpt_plan", "access_token": "kept"})
    path.write_text(original)
    login = AsyncMock()
    monkeypatch.setattr(mod, "oauth_login", login)
    p = ChatGPTProvider({"auth_mode": "chatgpt_codex", "token_file_path": str(path)})
    with pytest.raises(
        llm_errors.AuthenticationError, match="separate credential file"
    ):
        await p.login()
    login.assert_not_called()
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_invalid_mode_and_path_fail_before_mount_login(monkeypatch):
    import amplifier_module_provider_openai_chatgpt as mod

    login = AsyncMock()
    monkeypatch.setattr(mod, "login", login)
    for config in (
        {"auth_mode": "typo"},
        {"auth_mode": "chatgpt_plan", "token_file_path": ""},
    ):
        with pytest.raises(ValueError):
            await mount(MagicMock(), config)
    login.assert_not_called()


@pytest.mark.asyncio
async def test_plan_function_tool_round_trip_preserves_name_arguments_and_call_id(
    monkeypatch,
):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()
    sent = []
    tool = ToolSpec(
        name="read_file",
        description="Read a local file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        assert "tools" not in body  # No unsupported flat top-level functions.
        assert body["input"][0] == {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a local file",
                    "parameters": tool.parameters,
                }
            ],
        }
        if len(sent) == 1:
            assert body["tool_choice"] == {"type": "function", "name": "read_file"}
            item = {
                "type": "function_call",
                "call_id": "call-stable",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            }
        else:
            assert body["input"][-2:] == [
                {
                    "type": "function_call",
                    "call_id": "call-stable",
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-stable",
                    "output": "file contents",
                },
            ]
            item = {
                "type": "message",
                "content": [{"type": "output_text", "text": "Read complete"}],
            }
        return httpx.Response(
            200,
            text=stream_events(
                {"type": "response.output_item.done", "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp",
                        "usage": {"input_tokens": 8, "output_tokens": 3},
                    },
                },
            ),
        )

    install_transport(monkeypatch, handle)
    messages = [
        Message(role="system", content="Be helpful"),
        Message(role="user", content="Read README.md"),
    ]
    first = await p.complete(
        ChatRequest(
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "name": "read_file"},
        )
    )
    assert p.parse_tool_calls(first)[0].name == "read_file"
    assert p.parse_tool_calls(first)[0].arguments == {"path": "README.md"}
    second = await p.complete(
        ChatRequest(
            messages=[
                *messages,
                Message(role="assistant", content=first.content),
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_call_id="call-stable", output="file contents"
                        )
                    ],
                ),
            ],
            tools=[tool],
        )
    )
    assert second.content[0].text == "Read complete"
    assert len(sent) == 2


@pytest.mark.parametrize(
    "param,value",
    [
        ("temperature", 0),
        ("top_p", 0.8),
        ("max_output_tokens", 100),
        ("metadata", {}),
        ("background", False),
        ("previous_response_id", "resp-prior"),
        ("truncation", "auto"),
        ("conversation", "conv"),
    ],
)
def test_plan_unsupported_fields_fail_clearly_in_requests_and_extra_params(
    param, value
):
    p = provider()
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        p._build_payload(ChatRequest(messages=[], **{param: value}))
    assert error.value.error_code == "unsupported_plan_parameter"
    assert error.value.error_param == param
    assert error.value.retryable is False
    p.extra_request_params[param] = value
    with pytest.raises(llm_errors.InvalidRequestError):
        p._build_payload(ChatRequest(messages=[]))


@pytest.mark.parametrize(
    "kind",
    [
        "computer",
        "image_generation",
        "tool_search",
        "file_search",
        "code_interpreter",
        "mcp",
        "programmatic_tool_calling",
        "custom",
        "namespace",
    ],
)
def test_plan_unsupported_tools_never_silently_disappear(kind):
    p = provider()
    p.extra_request_params["tools"] = [{"type": kind, "name": "test"}]
    with pytest.raises(llm_errors.InvalidRequestError, match="additional_tools"):
        p._build_payload(ChatRequest(messages=[]))


def test_plan_later_system_messages_and_hosted_web_search():
    p = provider()
    p.extra_request_params["tools"] = [{"type": "web_search"}]
    payload = p._build_payload(
        ChatRequest(
            messages=[
                Message(role="system", content="Instructions"),
                Message(role="user", content="hi"),
                Message(role="system", content="Additional guidance"),
            ]
        )
    )
    assert payload["instructions"] == "Instructions"
    assert payload["input"][-1] == {
        "role": "developer",
        "content": [{"type": "input_text", "text": "Additional guidance"}],
    }
    assert payload["tools"] == [{"type": "web_search"}]


def test_codex_tools_and_request_parameter_behavior_are_unchanged():
    p = ChatGPTProvider(
        {
            "auth_mode": "chatgpt_codex",
            "extra_request_params": {"metadata": {"test": "kept"}},
        }
    )
    payload = p._build_payload(
        ChatRequest(
            messages=[Message(role="user", content="Hi")],
            temperature=0,
            tools=[ToolSpec(name="read_file", parameters={"type": "object"})],
        )
    )
    assert payload["tools"][0]["type"] == "function"
    assert payload["input"][0]["role"] == "user"
    assert payload["metadata"] == {"test": "kept"}
    assert "temperature" not in payload


def test_plan_image_input_is_not_silently_lost():
    from amplifier_core.message_models import ImageBlock

    p = provider()
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content=[
                    ImageBlock(
                        source={"type": "url", "url": "https://example.com/image.png"}
                    )
                ],
            )
        ]
    )
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        p._build_payload(request)
    assert error.value.error_param == "messages.content.image"


def test_plan_tool_lists_are_not_replaced_and_config_input_is_not_mutated():
    p = provider()
    p.extra_request_params["tools"] = [
        {"type": "function", "name": "configured", "parameters": {}}
    ]
    with pytest.raises(llm_errors.InvalidRequestError, match="supplied twice"):
        p._build_payload(
            ChatRequest(messages=[], tools=[ToolSpec(name="typed", parameters={})])
        )
    p.extra_request_params["input"] = [
        {"role": "system", "content": "Keep configuration unchanged"}
    ]
    payload = p._build_payload(ChatRequest(messages=[]))
    assert payload["input"][1]["role"] == "developer"
    assert p.extra_request_params["input"][0]["role"] == "system"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_plan_service_error_cannot_echo_bearer_token(monkeypatch, streaming):
    p = provider()
    p._ensure_valid_tokens = AsyncMock()

    def handle(request):
        if streaming:
            return httpx.Response(
                200,
                text=stream_events(
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {
                                "code": "subscription_sharing_invalid_user",
                                "message": "Rejected opaque-plan-token",
                            }
                        },
                    }
                ),
            )
        return httpx.Response(401, json={"detail": "Rejected opaque-plan-token"})

    install_transport(monkeypatch, handle)
    with pytest.raises(llm_errors.AuthenticationError) as error:
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert "opaque-plan-token" not in str(error.value)
    assert "opaque-plan-token" not in json.dumps(error.value.response_body)
