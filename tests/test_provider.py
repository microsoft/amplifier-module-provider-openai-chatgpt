"""Tests for provider.py — CHATGPT_MODELS catalog, get_info(), and list_models()."""

from __future__ import annotations

import asyncio
import json
import logging
import pytest
import httpx
from unittest.mock import ANY, AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# TestFallbackCatalog — FALLBACK_MODELS in models.py
# ---------------------------------------------------------------------------


class TestFallbackCatalog:
    """Verify the fallback model entries have the expected shape."""

    def test_fallback_models_nonempty(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        assert len(FALLBACK_MODELS) > 0

    def test_fallback_entries_have_slug(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        for entry in FALLBACK_MODELS:
            assert "slug" in entry, f"Missing 'slug' in {entry}"

    def test_fallback_first_entry_is_gpt_56_sol(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        assert len(FALLBACK_MODELS) > 0, "FALLBACK_MODELS must not be empty"
        assert FALLBACK_MODELS[0]["slug"] == "gpt-5.6-sol", (
            f"Expected gpt-5.6-sol as first fallback entry, got {FALLBACK_MODELS[0]['slug']!r}"
        )

    def test_fallback_does_not_contain_stale_models(self) -> None:
        """gpt-5.2 and gpt-5.3-codex are no longer served by the live catalog
        (confirmed via the raw models-endpoint payload -- see PR description)
        and must not linger in the fallback catalog."""
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        slugs = [m["slug"] for m in FALLBACK_MODELS]
        assert "gpt-5.2" not in slugs
        assert "gpt-5.3-codex" not in slugs

    def test_fallback_contains_current_generation(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        slugs = [m["slug"] for m in FALLBACK_MODELS]
        assert slugs == [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
        ]

    def test_fallback_gpt_56_sol_has_fast_tier(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        entry = next(m for m in FALLBACK_MODELS if m["slug"] == "gpt-5.6-sol")
        assert "fast" in entry.get("additional_speed_tiers", [])


# ---------------------------------------------------------------------------
# TestGetInfo — provider.get_info()
# ---------------------------------------------------------------------------


class TestGetInfo:
    def _make_provider(self) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        coordinator = MagicMock()
        tokens: dict | None = None
        return ChatGPTProvider(config, coordinator, tokens)

    def test_get_info_id(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert info.id == "openai-chatgpt"

    def test_get_info_display_name(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert info.display_name == "OpenAI ChatGPT"

    def test_get_info_capabilities_include_streaming(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert "streaming" in info.capabilities

    def test_get_info_capabilities_include_tools(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert "tools" in info.capabilities

    def test_get_info_model_default_shows_sentinel_with_fallback_noted(self) -> None:
        """Before resolution, defaults["model"] must be honest: it names the
        "latest" sentinel AND the fallback it would resolve to -- never a
        bare "latest" that looks like a real pinned model id, and never a
        network call (get_info() stays synchronous/side-effect-free)."""
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert "latest" in info.defaults["model"]
        assert "gpt-5.6-sol" in info.defaults["model"]

    def test_get_info_model_shows_resolved_id_once_resolved(self) -> None:
        """After resolution has happened (cached on the instance),
        defaults["model"] shows the concrete resolved id, not the sentinel."""
        provider = self._make_provider()
        provider._resolved_default_model = "gpt-5.6-terra"  # type: ignore[union-attr]
        info = provider.get_info()  # type: ignore[union-attr]
        assert info.defaults["model"] == "gpt-5.6-terra"

    def test_get_info_explicit_model_shown_verbatim(self) -> None:
        """An explicit non-sentinel default_model is shown as-is, with no
        sentinel/fallback framing."""
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        provider = ChatGPTProvider(config={"default_model": "gpt-5.4"})
        info = provider.get_info()  # type: ignore[union-attr]
        assert info.defaults["model"] == "gpt-5.4"

    def test_get_info_never_awaits_anything(self) -> None:
        """get_info() must stay synchronous and must not trigger a live
        catalog fetch -- app-cli's wizard calls it eagerly at mount time,
        potentially before login."""
        import inspect

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        assert not inspect.iscoroutinefunction(ChatGPTProvider.get_info)


# ---------------------------------------------------------------------------
# TestListModels — provider.list_models() shape validation
# ---------------------------------------------------------------------------


class TestListModels:
    """Verify that list_models() returns well-formed ModelInfo objects.

    Uses FALLBACK_MODELS as the mock return value of fetch_models so the
    tests are deterministic and require no network access.
    """

    def _make_provider(self) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        coordinator = MagicMock()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        return ChatGPTProvider(config, coordinator, tokens)

    def _sample_entries(self) -> list[dict]:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        return list(FALLBACK_MODELS)

    @pytest.mark.asyncio
    async def test_list_models_all_have_id(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.id, f"Model missing id: {m}"

    @pytest.mark.asyncio
    async def test_list_models_all_have_display_name(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.display_name, f"Model missing display_name: {m}"

    @pytest.mark.asyncio
    async def test_list_models_all_have_context_window(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.context_window > 0, f"Model missing context_window: {m}"

    @pytest.mark.asyncio
    async def test_list_models_all_have_max_output_tokens(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.max_output_tokens > 0, f"Model missing max_output_tokens: {m}"

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_objects(self) -> None:
        from amplifier_core import ModelInfo

        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert isinstance(m, ModelInfo), f"Expected ModelInfo, got {type(m)}"


# ---------------------------------------------------------------------------
# TestBuildPayload — provider._build_payload()
# ---------------------------------------------------------------------------

REJECTED_PARAMS = {
    "max_output_tokens",
    "temperature",
    "truncation",
    "parallel_tool_calls",
    "include",
}


class TestBuildPayload:
    def _make_provider(self, default_model: str = "gpt-4o") -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {"default_model": default_model}
        coordinator = MagicMock()
        tokens: dict | None = None
        return ChatGPTProvider(config, coordinator, tokens)

    def _make_request(self, **kwargs) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        messages = kwargs.pop("messages", [Message(role="user", content="hello")])
        return ChatRequest(messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_basic_structure_stream_true(self) -> None:
        """Payload must always have stream=True."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["stream"] is True

    def test_basic_structure_store_false(self) -> None:
        """Payload must always have store=False."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["store"] is False

    def test_basic_structure_uses_input_not_messages(self) -> None:
        """Payload uses 'input' array, not 'messages'."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hello")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "input" in payload
        assert "messages" not in payload

    def test_basic_structure_no_rejected_params(self) -> None:
        """Payload must not contain rejected parameters."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            max_output_tokens=1000,
            temperature=0.7,
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        for param in REJECTED_PARAMS:
            assert param not in payload, f"Rejected param '{param}' found in payload"

    def test_basic_structure_has_model(self) -> None:
        """Payload must include a model field."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "model" in payload
        assert payload["model"] == "gpt-4o"

    # ------------------------------------------------------------------
    # System message → instructions
    # ------------------------------------------------------------------

    def test_system_message_becomes_instructions(self) -> None:
        """System message content extracted as top-level 'instructions'."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload.get("instructions") == "You are a helpful assistant."

    def test_system_message_not_in_input(self) -> None:
        """System message must not appear in the input array."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="system", content="System prompt here."),
                Message(role="user", content="Hello"),
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        for item in payload["input"]:
            role = item.get("role")
            assert role not in ("system", "developer"), (
                f"System/developer message found in input array: {item}"
            )

    # ------------------------------------------------------------------
    # -fast suffix → priority service_tier
    # ------------------------------------------------------------------

    def test_fast_suffix_stripped_from_model(self) -> None:
        """Model name with -fast suffix has suffix stripped in payload."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-5.4-fast")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["model"] == "gpt-5.4"

    def test_fast_suffix_adds_priority_service_tier(self) -> None:
        """-fast suffix model adds service_tier='priority'."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-5.4-fast")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload.get("service_tier") == "priority"

    # ------------------------------------------------------------------
    # Non-fast model — no service_tier
    # ------------------------------------------------------------------

    def test_non_fast_model_no_service_tier(self) -> None:
        """Non -fast model must not include service_tier key."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "service_tier" not in payload

    # ------------------------------------------------------------------
    # Model override
    # ------------------------------------------------------------------

    def test_request_model_overrides_default(self) -> None:
        """Request model field overrides provider default_model."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            model="gpt-5.4",
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["model"] == "gpt-5.4"

    def test_request_model_fast_overrides_with_service_tier(self) -> None:
        """Request model with -fast suffix overrides default and adds service_tier."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            model="gpt-5.4-fast",
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["model"] == "gpt-5.4"
        assert payload.get("service_tier") == "priority"

    # ------------------------------------------------------------------
    # Tools conversion
    # ------------------------------------------------------------------

    def test_tools_converted_to_responses_api_format(self) -> None:
        """Tools converted to {type, name, description, parameters} format."""
        from amplifier_core.message_models import Message, ToolSpec

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="call a tool")],
            tools=[
                ToolSpec(
                    name="get_weather",
                    description="Gets the weather",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "tools" in payload
        tools = payload["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["description"] == "Gets the weather"
        assert tools[0]["parameters"] == {"type": "object", "properties": {}}

    def test_tools_adds_tool_choice_auto(self) -> None:
        """When tools present, tool_choice should be 'auto'."""
        from amplifier_core.message_models import Message, ToolSpec

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            tools=[ToolSpec(name="foo", parameters={"type": "object"})],
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload.get("tool_choice") == "auto"

    def test_no_tools_no_tool_choice(self) -> None:
        """Without tools, tool_choice must not appear in payload."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "tool_choice" not in payload

    # ------------------------------------------------------------------
    # Structured content (user message with TextBlock)
    # ------------------------------------------------------------------

    def test_user_message_string_content_converted(self) -> None:
        """String user content converted to input_text format."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hello")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        user_items = [i for i in payload["input"] if i.get("role") == "user"]
        assert len(user_items) == 1
        content = user_items[0]["content"]
        assert content == [{"type": "input_text", "text": "hello"}]

    def test_user_message_text_block_content_converted(self) -> None:
        """TextBlock user content converted to input_text format."""
        from amplifier_core.message_models import Message, TextBlock

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content=[TextBlock(text="world")])]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        user_items = [i for i in payload["input"] if i.get("role") == "user"]
        assert len(user_items) == 1
        content = user_items[0]["content"]
        assert {"type": "input_text", "text": "world"} in content

    # ------------------------------------------------------------------
    # Tool result conversion
    # ------------------------------------------------------------------

    def test_tool_result_converted_to_function_call_output(self) -> None:
        """ToolResultBlock → {type: function_call_output, call_id, output}."""
        from amplifier_core.message_models import Message, ToolResultBlock

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="user", content="run tool"),
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_call_id="call_123",
                            output="the result",
                        )
                    ],
                ),
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        tool_outputs = [
            i for i in payload["input"] if i.get("type") == "function_call_output"
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0]["call_id"] == "call_123"
        assert tool_outputs[0]["output"] == "the result"

    def test_assistant_tool_call_converted_to_function_call(self) -> None:
        """ToolCallBlock in assistant message → {type: function_call, call_id, name, arguments}."""
        from amplifier_core.message_models import Message, ToolCallBlock

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            id="call_abc",
                            name="search",
                            input={"query": "test"},
                        )
                    ],
                )
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        func_calls = [i for i in payload["input"] if i.get("type") == "function_call"]
        assert len(func_calls) == 1
        assert func_calls[0]["call_id"] == "call_abc"
        assert func_calls[0]["name"] == "search"
        # arguments must be a JSON string of the input dict
        args = func_calls[0]["arguments"]
        assert json.loads(args) == {"query": "test"}

    # ------------------------------------------------------------------
    # Reasoning effort
    # ------------------------------------------------------------------

    def test_reasoning_effort_adds_reasoning_block(self) -> None:
        """reasoning_effort in request → {reasoning: {effort, summary: 'detailed'}}."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="think")],
            reasoning_effort="high",
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "reasoning" in payload
        assert payload["reasoning"]["effort"] == "high"
        assert payload["reasoning"]["summary"] == "detailed"

    def test_no_reasoning_effort_no_reasoning_block(self) -> None:
        """Without reasoning_effort, no reasoning key in payload."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "reasoning" not in payload


# ---------------------------------------------------------------------------
# TestBuildHeaders — provider._build_headers()
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def _make_provider(
        self, tokens: dict | None = None, config: dict | None = None
    ) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        if config is None:
            config = {}
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def _valid_tokens(self) -> dict:
        return {"access_token": "test-access-tok", "account_id": "acct-123"}

    # ------------------------------------------------------------------
    # Happy path: all 6 required headers present with correct values
    # ------------------------------------------------------------------

    def test_authorization_header(self) -> None:
        """Authorization header must be 'Bearer {access_token}'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["Authorization"] == "Bearer test-access-tok"

    def test_chatgpt_account_id_header(self) -> None:
        """ChatGPT-Account-Id header must equal account_id from tokens."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["ChatGPT-Account-Id"] == "acct-123"

    def test_openai_beta_header(self) -> None:
        """OpenAI-Beta header must be 'responses=v1'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["OpenAI-Beta"] == "responses=v1"

    def test_openai_originator_header(self) -> None:
        """OpenAI-Originator header must be 'codex'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["OpenAI-Originator"] == "codex"

    def test_content_type_header(self) -> None:
        """Content-Type header must be 'application/json'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["Content-Type"] == "application/json"

    def test_accept_header(self) -> None:
        """accept header must be 'text/event-stream'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["accept"] == "text/event-stream"

    def test_all_six_fields_present(self) -> None:
        """Headers dict must have exactly the 6 required fields."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        expected_keys = {
            "Authorization",
            "ChatGPT-Account-Id",
            "OpenAI-Beta",
            "OpenAI-Originator",
            "Content-Type",
            "accept",
        }
        assert set(headers.keys()) == expected_keys

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_access_token_raises_auth_error(self) -> None:
        """Missing access_token raises AuthenticationError with 'No valid OAuth tokens'."""
        from amplifier_core import llm_errors as kernel_errors

        tokens = {"account_id": "acct-123"}  # no access_token
        provider = self._make_provider(tokens=tokens)
        with pytest.raises(
            kernel_errors.AuthenticationError, match="No valid OAuth tokens"
        ):
            provider._build_headers()  # type: ignore[union-attr]

    def test_none_tokens_raises_auth_error(self) -> None:
        """tokens=None raises AuthenticationError with 'No valid OAuth tokens'."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider(tokens=None)
        with pytest.raises(
            kernel_errors.AuthenticationError, match="No valid OAuth tokens"
        ):
            provider._build_headers()  # type: ignore[union-attr]

    def test_missing_account_id_raises_auth_error(self) -> None:
        """Missing account_id raises AuthenticationError mentioning 'account_id'."""
        from amplifier_core import llm_errors as kernel_errors

        tokens = {"access_token": "test-access-tok"}  # no account_id
        provider = self._make_provider(tokens=tokens)
        with pytest.raises(kernel_errors.AuthenticationError, match="account_id"):
            provider._build_headers()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestComplete — provider.complete()
# ---------------------------------------------------------------------------


def _make_sse_lines(
    text: str | None = None,
    tool_calls: list[dict] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[str]:
    """Build minimal SSE line list simulating a ChatGPT streaming response."""
    events = []

    # response.created
    events.append(
        json.dumps(
            {
                "type": "response.created",
                "response": {"id": "resp_test", "model": "gpt-4o"},
            }
        )
    )

    # Text content
    if text is not None:
        events.append(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    },
                }
            )
        )

    # Tool calls
    for tc in tool_calls or []:
        events.append(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
            )
        )

    # response.done with usage
    events.append(
        json.dumps(
            {
                "type": "response.done",
                "response": {
                    "id": "resp_test",
                    "model": "gpt-4o",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            }
        )
    )

    lines = [f"data: {e}" for e in events]
    lines.append("data: [DONE]")
    return lines


class _MockStreamResponse:
    """Async mock for an httpx streaming response."""

    def __init__(
        self,
        lines: list[str],
        status_code: int = 200,
        headers: dict | None = None,
        error_body: bytes = b"HTTP error body",
    ) -> None:
        self.status_code = status_code
        self._lines = lines
        self.headers = httpx.Headers(headers or {})
        self._error_body = error_body

    async def aiter_lines(self):  # type: ignore[return]
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return self._error_body


class _AsyncCM:
    """Simple async context manager wrapping a value."""

    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_sse_response(
    lines: list[str],
    status_code: int = 200,
    headers: dict | None = None,
    error_body: bytes = b"HTTP error body",
) -> "_AsyncCM":
    """Return an async-context-manager mock for httpx.AsyncClient.

    Usage::

        with patch("...httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            result = await provider.complete(request)
    """
    response = _MockStreamResponse(
        lines, status_code, headers=headers, error_body=error_body
    )
    mock_client = MagicMock()
    mock_client.stream.return_value = _AsyncCM(response)
    return _AsyncCM(mock_client)


class TestComplete:
    """Tests for ChatGPTProvider.complete()."""

    def _make_provider(self, raw: bool = False) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {"raw": raw, "default_model": "gpt-4o"}
        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        # Tokens that pass is_token_valid()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        return ChatGPTProvider(config, coordinator, tokens)

    def _make_request(self, **kwargs: object) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        messages = kwargs.pop("messages", [Message(role="user", content="hello")])
        return ChatRequest(messages=messages, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Simple text completion
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_simple_text_completion_content(self) -> None:
        """complete() returns ChatResponse with correct text content."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            text="Hello, world!", input_tokens=10, output_tokens=5
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        from amplifier_core.message_models import TextBlock

        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_simple_text_completion_usage(self) -> None:
        """complete() returns correct usage counts."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="Hello!", input_tokens=10, output_tokens=5)

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_simple_text_completion_finish_reason_stop(self) -> None:
        """Text-only response has finish_reason='stop'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="Done.")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.finish_reason == "stop"

    # ------------------------------------------------------------------
    # Tool call response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tool_call_response_name(self) -> None:
        """Tool call response populates tool name correctly."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        from amplifier_core.message_models import ToolCallBlock

        tool_blocks = [b for b in result.content if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_call_response_id(self) -> None:
        """Tool call response populates call ID correctly."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        from amplifier_core.message_models import ToolCallBlock

        tool_blocks = [b for b in result.content if isinstance(b, ToolCallBlock)]
        assert tool_blocks[0].id == "call_abc"

    @pytest.mark.asyncio
    async def test_tool_call_response_arguments_dict(self) -> None:
        """Tool call arguments are parsed to a dict."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"city": "London"}

    @pytest.mark.asyncio
    async def test_tool_call_response_finish_reason_tool_calls(self) -> None:
        """Response with tool calls has finish_reason='tool_calls'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_fallback_on_bad_json(self) -> None:
        """Invalid JSON arguments fall back to {'_raw': ...}."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {"id": "call_abc", "name": "foo", "arguments": "not-valid-json"}
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.tool_calls is not None
        assert "_raw" in result.tool_calls[0].arguments
        assert result.tool_calls[0].arguments["_raw"] == "not-valid-json"

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_emits_llm_request_event_provider(self) -> None:
        """complete() emits llm:request with provider='openai-chatgpt'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        assert len(request_calls) >= 1
        event_data = request_calls[0].args[1]
        assert event_data["provider"] == "openai-chatgpt"

    @pytest.mark.asyncio
    async def test_emits_llm_request_event_message_count(self) -> None:
        """complete() emits llm:request with correct message_count."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="user", content="hello"),
                Message(role="user", content="world"),
            ]
        )
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        event_data = request_calls[0].args[1]
        assert event_data["message_count"] == 2

    @pytest.mark.asyncio
    async def test_emits_llm_response_event_status_ok(self) -> None:
        """complete() emits llm:response with status='ok' on success."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        assert len(response_calls) >= 1
        event_data = response_calls[0].args[1]
        assert event_data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_emits_llm_response_event_duration_ms(self) -> None:
        """complete() emits llm:response with duration_ms present."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        event_data = response_calls[0].args[1]
        assert "duration_ms" in event_data
        assert isinstance(event_data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_emits_llm_response_event_provider(self) -> None:
        """complete() emits llm:response with provider='openai-chatgpt'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        event_data = response_calls[0].args[1]
        assert event_data["provider"] == "openai-chatgpt"

    # ------------------------------------------------------------------
    # Raw mode
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_raw_mode_request_event_includes_payload(self) -> None:
        """Raw mode: llm:request event includes raw payload."""
        provider = self._make_provider(raw=True)
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        event_data = request_calls[0].args[1]
        assert "raw" in event_data

    @pytest.mark.asyncio
    async def test_raw_mode_response_event_includes_raw_events(self) -> None:
        """Raw mode: llm:response event includes raw_events from SSE."""
        provider = self._make_provider(raw=True)
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        event_data = response_calls[0].args[1]
        assert "raw" in event_data

    @pytest.mark.asyncio
    async def test_non_raw_mode_request_no_payload(self) -> None:
        """Non-raw mode: llm:request event does NOT include payload."""
        provider = self._make_provider(raw=False)
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        event_data = request_calls[0].args[1]
        assert "payload" not in event_data


# ---------------------------------------------------------------------------
# TestCompleteErrors — error handling paths in complete()
# ---------------------------------------------------------------------------


class TestCompleteErrors:
    """Verify error handling in complete(): error events emitted before exceptions propagate."""

    def _make_provider_with_tokens(self) -> object:
        """Create ChatGPTProvider with valid OAuth tokens and a mock coordinator."""
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        return ChatGPTProvider({"default_model": "gpt-4o"}, coordinator, tokens)

    def _make_request(self) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        return ChatRequest(messages=[Message(role="user", content="hello")])

    @pytest.mark.asyncio
    async def test_sse_error_event_emits_error_status_then_raises(self) -> None:
        """SSE error event inside stream emits llm:response with status='error', then raises LLMError."""
        from amplifier_core import llm_errors as kernel_errors
        from amplifier_module_provider_openai_chatgpt._sse import SSEError

        provider = self._make_provider_with_tokens()
        request = self._make_request()

        # SSE lines containing an error event
        error_lines = [
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": "Something went wrong",
                        "code": "server_error",
                    },
                }
            ),
            "data: [DONE]",
        ]

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(error_lines)
            with pytest.raises(kernel_errors.LLMError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        # SSEError must never escape complete() unmapped
        assert not isinstance(exc_info.value, SSEError)

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        error_response_calls = [
            c
            for c in calls
            if c.args[0] == "llm:response" and c.args[1].get("status") == "error"
        ]
        assert len(error_response_calls) >= 1, (
            "Expected at least one llm:response event with status='error'"
        )

    @pytest.mark.asyncio
    async def test_missing_tokens_raises_auth_error(self) -> None:
        """Provider with no valid tokens raises AuthenticationError before making any HTTP request."""
        from amplifier_core import llm_errors as kernel_errors
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        provider = ChatGPTProvider(
            {"default_model": "gpt-4o"}, coordinator, tokens=None
        )
        request = self._make_request()

        # Patch the name bound in provider.py's namespace so _ensure_valid_tokens()
        # sees None for every disk-load attempt and raises AuthenticationError.
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value=None,
        ):
            with pytest.raises(kernel_errors.AuthenticationError):
                await provider.complete(request)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestMount — module-level mount() function
# ---------------------------------------------------------------------------


class TestMount:
    """Tests for the module-level mount() function in __init__.py."""

    @pytest.mark.asyncio
    async def test_mount_with_valid_token_file_succeeds(self) -> None:
        """mount() with valid tokens calls coordinator.mount once with 'providers'
        and name='openai-chatgpt', and returns a callable cleanup."""
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt import mount

        coordinator = MagicMock()
        # coordinator.mount is awaited in __init__.py, so it must be an AsyncMock
        coordinator.mount = AsyncMock()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        valid_tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }

        config = {"token_file_path": "/tmp/fake_tokens.json"}

        with patch(
            "amplifier_module_provider_openai_chatgpt.load_tokens",
            return_value=valid_tokens,
        ):
            with patch(
                "amplifier_module_provider_openai_chatgpt.is_token_valid",
                return_value=True,
            ):
                cleanup = await mount(coordinator, config)

        coordinator.mount.assert_called_once_with(
            "providers", ANY, name="openai-chatgpt"
        )
        assert callable(cleanup)

    @pytest.mark.asyncio
    async def test_mount_returns_none_no_tokens_login_disabled(self) -> None:
        """mount() returns None when no valid tokens and login_on_mount=False.
        coordinator.mount must not be called."""
        from amplifier_module_provider_openai_chatgpt import mount

        coordinator = MagicMock()
        config = {"token_file_path": "/tmp/fake_tokens.json", "login_on_mount": False}

        with patch(
            "amplifier_module_provider_openai_chatgpt.load_tokens",
            return_value=None,
        ):
            with patch(
                "amplifier_module_provider_openai_chatgpt.is_token_valid",
                return_value=False,
            ):
                result = await mount(coordinator, config)

        assert result is None
        coordinator.mount.assert_not_called()


# ---------------------------------------------------------------------------
# TestMountCallsLogin -- fail-before proof for the headline onboarding defect
# ---------------------------------------------------------------------------


class TestMountCallsLogin:
    """mount() with no valid tokens and login_on_mount=True (the default)
    must call oauth.login(). This is the fail-before proof for the headline
    onboarding defect: `amplifier provider add` constructs ChatGPTProvider
    directly and never calls mount() at all, so this login trigger --
    mount()'s ONLY entrypoint before this PR -- never fires from that path.
    """

    @pytest.mark.asyncio
    async def test_mount_calls_login_when_no_valid_tokens_and_login_on_mount_true(
        self,
    ) -> None:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        config = {"login_on_mount": True}

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        logged_in_tokens = {
            "access_token": "fresh_access",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        mock_login = AsyncMock(return_value=logged_in_tokens)

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.load_tokens",
                return_value=None,
            ),
            patch(
                "amplifier_module_provider_openai_chatgpt.is_token_valid",
                side_effect=[False, True],
            ),
            patch(
                "amplifier_module_provider_openai_chatgpt.login",
                mock_login,
            ),
        ):
            cleanup = await mount(coordinator, config)

        mock_login.assert_awaited_once()
        coordinator.mount.assert_called_once_with(
            "providers", ANY, name="openai-chatgpt"
        )
        assert callable(cleanup)


# ---------------------------------------------------------------------------
# TestListModelsDynamic — caching logic in _get_catalog() / list_models()
# ---------------------------------------------------------------------------


class TestListModelsDynamic:
    """Tests for the cached model fetching introduced by _get_catalog().

    Follows the same token-mocking approach as TestComplete: valid tokens
    are set on the provider so _ensure_valid_tokens() passes immediately,
    and fetch_models is patched to control catalog fetch behaviour.
    """

    def _make_provider(self, models_cache_ttl: float = 3600.0) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        config = {"models_cache_ttl": models_cache_ttl, "default_model": "gpt-4o"}
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def _sample_entries(self) -> list[dict]:
        """Single API-visible model entry (no fast tier) for deterministic tests."""
        return [
            {
                "slug": "gpt-4o",
                "display_name": "GPT-4o",
                "context_window": 128000,
                "supported_in_api": True,
                "visibility": "visible",
                "additional_speed_tiers": [],
                "supported_reasoning_levels": [],
                "default_reasoning_level": None,
            }
        ]

    # ------------------------------------------------------------------
    # First-call fetch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_fetches_on_first_call(self) -> None:
        """First call to list_models() fetches from API exactly once."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            result = await provider.list_models()  # type: ignore[union-attr]

        mock_fetch.assert_awaited_once()
        assert len(result) == 1
        assert result[0].id == "gpt-4o"

    # ------------------------------------------------------------------
    # Cache hit on second call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_uses_cache_on_second_call(self) -> None:
        """Second call within TTL returns cached result; fetch_models called once."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            await provider.list_models()  # type: ignore[union-attr]
            await provider.list_models()  # type: ignore[union-attr]

        mock_fetch.assert_awaited_once()

    # ------------------------------------------------------------------
    # Fallback on failure, cache stays empty
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_falls_back_on_failure(self) -> None:
        """When fetch_models raises, fallback models are returned and cache stays None."""
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        provider = self._make_provider()
        mock_fetch = AsyncMock(side_effect=ValueError("simulated API error"))

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            result = await provider.list_models()  # type: ignore[union-attr]

        # All fallback slugs must appear in the result (some produce -fast variants).
        fallback_slugs = {entry["slug"] for entry in FALLBACK_MODELS}
        result_ids = {m.id for m in result}
        for slug in fallback_slugs:
            assert slug in result_ids or f"{slug}-fast" in result_ids, (
                f"Expected fallback slug {slug!r} (or its -fast variant) in {result_ids}"
            )

        # Cache must NOT be populated on failure — next call should retry.
        assert provider._models_cache is None  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # TTL expiry causes re-fetch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_ttl_expiry_refetches(self) -> None:
        """With models_cache_ttl=0, each call triggers a fresh fetch."""
        provider = self._make_provider(models_cache_ttl=0.0)
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            await provider.list_models()  # type: ignore[union-attr]
            await provider.list_models()  # type: ignore[union-attr]

        assert mock_fetch.await_count == 2, (
            f"Expected 2 fetches with TTL=0, got {mock_fetch.await_count}"
        )

    # ------------------------------------------------------------------
    # Empty catalog triggers fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_falls_back_on_empty_catalog(self) -> None:
        """When fetch_models returns [], fallback models are returned and cache stays None."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=[])

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            result = await provider.list_models()  # type: ignore[union-attr]

        assert len(result) > 0, "Expected non-empty fallback models"
        # Cache must NOT be populated — next call should retry the live fetch.
        assert provider._models_cache is None  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Concurrent calls fetch exactly once
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_list_models_fetches_once(self) -> None:
        """5 concurrent list_models() calls result in exactly one fetch_models invocation."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            results = await asyncio.gather(
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
            )

        mock_fetch.assert_awaited_once()
        # All 5 callers should receive the same non-empty result.
        for result in results:
            assert len(result) > 0


# ---------------------------------------------------------------------------
# TestResolveDefaultModel — "latest" sentinel resolution (_resolve_default_model)
# ---------------------------------------------------------------------------


class TestResolveDefaultModel:
    """Tests for the "latest" default-model resolution design.

    Precedence: explicit config value > live-catalog resolution (first
    non-variant, flagship-first per the live-payload evidence) > static
    FALLBACK_MODELS[0]. Resolved lazily, cached for the provider instance's
    lifetime, one INFO log line. Unauthenticated resolution falls back
    silently -- unlike list_models()/_get_catalog(), which correctly
    propagate AuthenticationError (see TestListModelsAuthPropagation) since
    those ARE actual requests for the live catalog, not "what model name do
    I use" resolution.
    """

    def _make_provider(
        self,
        default_model: str = "latest",
        tokens: dict | None = None,
        token_file_path: str | None = None,
        coordinator: object | None = None,
    ) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {"default_model": default_model}
        if token_file_path is not None:
            config["token_file_path"] = token_file_path
        if coordinator is None:
            coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def _authenticated_tokens(self) -> dict:
        from datetime import datetime, timedelta, timezone

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        return {
            "access_token": "tok",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }

    def _sample_live_entries(self) -> list[dict]:
        """Flagship-first order with a -fast variant, mirroring the live
        catalog's own shape (see the raw-payload evidence in the PR body)."""
        return [
            {
                "slug": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "context_window": 272000,
                "supported_in_api": True,
                "visibility": "list",
                "additional_speed_tiers": ["fast"],
                "supported_reasoning_levels": [],
            },
            {
                "slug": "gpt-5.4-mini",
                "display_name": "GPT-5.4-Mini",
                "context_window": 272000,
                "supported_in_api": True,
                "visibility": "list",
                "additional_speed_tiers": [],
                "supported_reasoning_levels": [],
            },
        ]

    # ------------------------------------------------------------------
    # Precedence: explicit config value bypasses resolution entirely
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_explicit_model_untouched_by_resolution(self) -> None:
        """An explicit non-sentinel default_model is returned as-is --
        no catalog fetch, no network call at all."""
        provider = self._make_provider(
            default_model="gpt-5.4", tokens=self._authenticated_tokens()
        )
        mock_fetch = AsyncMock(side_effect=AssertionError("must not fetch"))

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            resolved = await provider._resolve_default_model()  # type: ignore[union-attr]

        assert resolved == "gpt-5.4"
        mock_fetch.assert_not_awaited()

    # ------------------------------------------------------------------
    # Sentinel + authenticated -> first non-variant catalog entry
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_sentinel_authenticated_resolves_first_non_variant(self) -> None:
        provider = self._make_provider(tokens=self._authenticated_tokens())
        mock_fetch = AsyncMock(return_value=self._sample_live_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            resolved = await provider._resolve_default_model()  # type: ignore[union-attr]

        assert resolved == "gpt-5.6-sol"
        mock_fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sentinel_authenticated_skips_variant_ids(self) -> None:
        """Defensive ordering case: even if a -mini/-fast id sorted first in
        the catalog, resolution skips it for the first NON-variant entry --
        matching the decision rule ('first catalog entry whose id has no
        variant suffix')."""
        provider = self._make_provider(tokens=self._authenticated_tokens())
        entries = [
            {
                "slug": "gpt-5.4-mini",
                "display_name": "GPT-5.4-Mini",
                "context_window": 272000,
                "supported_in_api": True,
                "visibility": "list",
                "additional_speed_tiers": [],
                "supported_reasoning_levels": [],
            },
            {
                "slug": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "context_window": 272000,
                "supported_in_api": True,
                "visibility": "list",
                "additional_speed_tiers": ["fast"],
                "supported_reasoning_levels": [],
            },
        ]
        mock_fetch = AsyncMock(return_value=entries)

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            resolved = await provider._resolve_default_model()  # type: ignore[union-attr]

        assert resolved == "gpt-5.6-sol"

    # ------------------------------------------------------------------
    # Unauthenticated -> FALLBACK_MODELS[0], no auth error raised
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unauthenticated_falls_back_silently(self, tmp_path) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        provider = self._make_provider(
            tokens=None, token_file_path=str(tmp_path / "tokens.json")
        )

        # No fetch_models mock needed -- _ensure_valid_tokens() fails before
        # any HTTP call is attempted (no tokens in memory, on disk, or via
        # refresh_token anywhere). Resolution must not raise.
        resolved = await provider._resolve_default_model()  # type: ignore[union-attr]

        assert resolved == FALLBACK_MODELS[0]["slug"] == "gpt-5.6-sol"

    # ------------------------------------------------------------------
    # Cache: catalog fetched (and resolved) once per provider instance
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_resolution_cached_for_instance_lifetime(self) -> None:
        provider = self._make_provider(tokens=self._authenticated_tokens())
        mock_fetch = AsyncMock(return_value=self._sample_live_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            first = await provider._resolve_default_model()  # type: ignore[union-attr]
            second = await provider._resolve_default_model()  # type: ignore[union-attr]

        assert first == second == "gpt-5.6-sol"
        mock_fetch.assert_awaited_once()

    # ------------------------------------------------------------------
    # Exactly one INFO log line names the resolution
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_resolution_logs_one_info_line(self, caplog) -> None:
        import logging

        provider = self._make_provider(tokens=self._authenticated_tokens())
        mock_fetch = AsyncMock(return_value=self._sample_live_entries())

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
                mock_fetch,
            ),
            caplog.at_level(logging.INFO),
        ):
            await provider._resolve_default_model()  # type: ignore[union-attr]
            await provider._resolve_default_model()  # type: ignore[union-attr]  # cached -- no 2nd log line

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        matching = [r for r in info_records if "gpt-5.6-sol" in r.getMessage()]
        assert len(matching) == 1, (
            f"Expected exactly one INFO line naming the resolution, got {info_records}"
        )

    # ------------------------------------------------------------------
    # complete() resolves "latest" and sends the resolved model
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_sends_resolved_model_not_literal_sentinel(self) -> None:
        """End-to-end: complete() must never send the literal string
        "latest" to the ChatGPT backend -- it resolves first."""
        from amplifier_core.message_models import ChatRequest, Message

        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        provider = self._make_provider(
            tokens=self._authenticated_tokens(), coordinator=coordinator
        )
        mock_fetch = AsyncMock(return_value=self._sample_live_entries())
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        sse_lines = _make_sse_lines(text="OK")

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
                mock_fetch,
            ),
            patch(
                "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
            ) as MockClient,
        ):
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

            # Inspect the actual JSON payload sent to the backend.
            _, call_kwargs = MockClient.return_value._value.stream.call_args
            sent_model = call_kwargs["json"]["model"]

        assert sent_model == "gpt-5.6-sol"
        assert provider._resolved_default_model == "gpt-5.6-sol"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestCompleteErrorMapping — HTTP status codes and exceptions → kernel errors
# ---------------------------------------------------------------------------


class TestCompleteErrorMapping:
    """Verify that every non-200 status code and transport exception
    escapes complete() as the expected kernel_errors subtype."""

    def _make_provider(self) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        return ChatGPTProvider({"default_model": "gpt-4o"}, coordinator, tokens)

    def _make_request(self) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        return ChatRequest(messages=[Message(role="user", content="hello")])

    # ------------------------------------------------------------------
    # HTTP 401 → AuthenticationError
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self) -> None:
        """HTTP 401 → AuthenticationError(retryable=False)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [], status_code=401, error_body=b'{"error": "unauthorized"}'
            )
            with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 401

    # ------------------------------------------------------------------
    # HTTP 429 → RateLimitError (with optional Retry-After)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit(self) -> None:
        """HTTP 429 → RateLimitError(retryable=True)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [], status_code=429, error_body=b'{"error": "rate limit exceeded"}'
            )
            with pytest.raises(kernel_errors.RateLimitError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_429_extracts_retry_after_header(self) -> None:
        """HTTP 429 with Retry-After header → RateLimitError.retry_after set."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [],
                status_code=429,
                headers={"retry-after": "30"},
                error_body=b'{"error": "rate limit exceeded"}',
            )
            with pytest.raises(kernel_errors.RateLimitError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retry_after == 30.0

    # ------------------------------------------------------------------
    # HTTP 500 → ProviderUnavailableError
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_500_raises_provider_unavailable(self) -> None:
        """HTTP 500 → ProviderUnavailableError(retryable=True)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [], status_code=500, error_body=b"Internal Server Error"
            )
            with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 500

    # ------------------------------------------------------------------
    # HTTP 403 — Cloudflare vs real 403
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cloudflare_403_detected(self) -> None:
        """HTTP 403 with text/html content-type → ProviderUnavailableError(retryable=True)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        cf_body = (
            b"<html><title>Just a moment...</title>"
            b"<p>Checking if the site connection is secure</p></html>"
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [],
                status_code=403,
                headers={"content-type": "text/html; charset=utf-8"},
                error_body=cf_body,
            )
            with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_real_403_not_cloudflare(self) -> None:
        """HTTP 403 with JSON response (no CF markers) → AccessDeniedError(retryable=False)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [],
                status_code=403,
                headers={"content-type": "application/json"},
                error_body=b'{"error": "access denied"}',
            )
            with pytest.raises(kernel_errors.AccessDeniedError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 403

    # ------------------------------------------------------------------
    # HTTP 400 sub-dispatch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_400_context_length_error(self) -> None:
        """HTTP 400 with 'context length' in body → ContextLengthError(retryable=False)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [],
                status_code=400,
                error_body=b'{"error": "This exceeds the context length limit"}',
            )
            with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_400_content_filter_error(self) -> None:
        """HTTP 400 with 'content filter' in body → ContentFilterError(retryable=False)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [],
                status_code=400,
                error_body=b'{"error": "Request blocked by content filter policy"}',
            )
            with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_400_generic_bad_request(self) -> None:
        """HTTP 400 without special keywords → InvalidRequestError(retryable=False)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(
                [],
                status_code=400,
                error_body=b'{"error": "invalid_request_error", "message": "Bad param"}',
            )
            with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 400

    # ------------------------------------------------------------------
    # httpx.ConnectError → ProviderUnavailableError
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_transport_error_mapped(self) -> None:
        """httpx.ConnectError during streaming → ProviderUnavailableError(retryable=True)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        mock_client = MagicMock()
        mock_client.stream.side_effect = httpx.ConnectError("Connection refused")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _AsyncCM(mock_client)
            with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is True

    # ------------------------------------------------------------------
    # SSEError inside stream → mapped kernel type
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_sse_error_mapped(self) -> None:
        """SSEError (generic server_error code) → LLMError(retryable=False)."""
        from amplifier_core import llm_errors as kernel_errors
        from amplifier_module_provider_openai_chatgpt._sse import SSEError

        provider = self._make_provider()
        request = self._make_request()

        error_lines = [
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": "internal server error",
                        "code": "server_error",
                    },
                }
            ),
            "data: [DONE]",
        ]

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(error_lines)
            with pytest.raises(kernel_errors.LLMError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        # SSEError must never escape complete() unmapped
        assert not isinstance(exc_info.value, SSEError)
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_sse_rate_limit_error_mapped(self) -> None:
        """SSEError with rate_limit code → RateLimitError(retryable=True)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        error_lines = [
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": "rate limit exceeded",
                        "code": "rate_limit_exceeded",
                    },
                }
            ),
            "data: [DONE]",
        ]

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(error_lines)
            with pytest.raises(kernel_errors.RateLimitError):
                await provider.complete(request)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # httpx.TimeoutException → LLMTimeoutError
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_timeout_mapped(self) -> None:
        """httpx.ReadTimeout during streaming → LLMTimeoutError(retryable=True)."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        class _TimeoutOnReadResponse:
            """Mock response that raises ReadTimeout when its line iterator is advanced."""

            status_code = 200
            headers = httpx.Headers({})

            async def aiter_lines(self):  # type: ignore[return]
                raise httpx.ReadTimeout("Request timed out")
                yield  # makes this an async generator  # noqa: unreachable

            async def aread(self) -> bytes:
                return b""

        mock_client = MagicMock()
        mock_client.stream.return_value = _AsyncCM(_TimeoutOnReadResponse())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _AsyncCM(mock_client)
            with pytest.raises(kernel_errors.LLMTimeoutError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# TestComplete401Retry — mid-session 401 retry logic
# ---------------------------------------------------------------------------


class TestComplete401Retry:
    """Verify the two-attempt 401 retry loop in complete().

    Design: first 401 triggers a token refresh + header rebuild + retry;
    a second 401 (or refresh failure) propagates AuthenticationError.
    Only 401s trigger this path — other error codes are not retried.
    """

    def _make_provider(self) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        config = {"default_model": "gpt-4o"}
        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def _make_request(self) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        return ChatRequest(messages=[Message(role="user", content="hello")])

    # ------------------------------------------------------------------
    # 1. First attempt 401, second attempt succeeds
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_401_retry_succeeds(self) -> None:
        """First attempt returns 401; second attempt succeeds.

        Verifies:
        - _ensure_valid_tokens called twice (start + retry).
        - headers rebuilt before second attempt.
        - response returned normally with correct content.
        - llm:request emitted exactly once (not re-emitted on retry).
        """
        from amplifier_core.message_models import TextBlock

        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            text="Hello after retry!", input_tokens=8, output_tokens=4
        )

        mock_401 = _make_sse_response(
            [], status_code=401, error_body=b'{"error": "unauthorized"}'
        )
        mock_200 = _make_sse_response(sse_lines, status_code=200)

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.side_effect = [mock_401, mock_200]
            with patch.object(
                provider,  # type: ignore[union-attr]
                "_ensure_valid_tokens",
                new_callable=AsyncMock,
            ) as mock_ensure:
                result = await provider.complete(request)  # type: ignore[union-attr]

        # Token refresh was triggered once (initial call + retry call = 2 total).
        assert mock_ensure.await_count == 2, (
            f"Expected _ensure_valid_tokens called twice, got {mock_ensure.await_count}"
        )
        # Both HTTP attempts were made.
        assert MockClient.call_count == 2

        # Response content is correct from the successful second attempt.
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Hello after retry!"

        # llm:request emitted exactly once — NOT re-emitted on retry.
        coordinator = provider._coordinator  # type: ignore[union-attr]
        request_events = [
            c
            for c in coordinator.hooks.emit.call_args_list
            if c.args[0] == "llm:request"
        ]
        assert len(request_events) == 1, (
            f"Expected exactly 1 llm:request event, got {len(request_events)}"
        )

    # ------------------------------------------------------------------
    # 2. Both attempts return 401
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_401_retry_fails_twice(self) -> None:
        """Both attempts return 401 → AuthenticationError raised after second attempt.

        Verifies:
        - Both HTTP calls were attempted.
        - AuthenticationError escapes with correct attributes.
        - llm:response error event is emitted.
        """
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        mock_401_a = _make_sse_response(
            [], status_code=401, error_body=b'{"error": "unauthorized"}'
        )
        mock_401_b = _make_sse_response(
            [], status_code=401, error_body=b'{"error": "unauthorized"}'
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.side_effect = [mock_401_a, mock_401_b]
            with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
                await provider.complete(request)  # type: ignore[union-attr]

        assert exc_info.value.status_code == 401
        assert exc_info.value.retryable is False
        # Both HTTP attempts were made (first + retry).
        assert MockClient.call_count == 2

        # llm:response error event must have been emitted.
        coordinator = provider._coordinator  # type: ignore[union-attr]
        error_events = [
            c
            for c in coordinator.hooks.emit.call_args_list
            if c.args[0] == "llm:response" and c.args[1].get("status") == "error"
        ]
        assert len(error_events) >= 1, (
            "Expected llm:response error event after double-401"
        )

    # ------------------------------------------------------------------
    # 3. First attempt 401, refresh itself fails
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_401_retry_refresh_fails(self) -> None:
        """First attempt 401; _ensure_valid_tokens() raises during retry.

        Verifies:
        - AuthenticationError from the failed refresh propagates immediately.
        - Only one HTTP call was made (refresh failed before second request).
        """
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        mock_401 = _make_sse_response(
            [], status_code=401, error_body=b'{"error": "unauthorized"}'
        )
        refresh_error = kernel_errors.AuthenticationError(
            "Token refresh failed — no valid refresh token",
            provider="openai-chatgpt",
            retryable=False,
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = mock_401
            # First call (start of complete()) succeeds; second call (retry) raises.
            with patch.object(
                provider,  # type: ignore[union-attr]
                "_ensure_valid_tokens",
                new=AsyncMock(side_effect=[None, refresh_error]),
            ):
                with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
                    await provider.complete(request)  # type: ignore[union-attr]

        assert isinstance(exc_info.value, kernel_errors.AuthenticationError)
        # Refresh failed before a second HTTP request was issued.
        assert MockClient.call_count == 1, (
            "Expected only 1 HTTP call — refresh failed before retry request"
        )

    # ------------------------------------------------------------------
    # 4. Non-401 errors are NOT retried
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_non_401_not_retried(self) -> None:
        """A 500 error is raised immediately — only one HTTP attempt is made."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request()

        mock_500 = _make_sse_response(
            [], status_code=500, error_body=b"Internal Server Error"
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = mock_500
            with pytest.raises(kernel_errors.ProviderUnavailableError):
                await provider.complete(request)  # type: ignore[union-attr]

        # Only one HTTP call — 500 must not trigger a retry.
        assert MockClient.call_count == 1, (
            f"Expected 1 HTTP call for 500 error, got {MockClient.call_count}"
        )


# ---------------------------------------------------------------------------
# TestGpt55ProValidator — _validate_gpt_5_5_pro_effort()
# ---------------------------------------------------------------------------


class TestGpt55ProValidator:
    """Unit tests for _validate_gpt_5_5_pro_effort() and its call site in _build_payload()."""

    def _validate(self, model_id: str, reasoning_param: object) -> None:
        """Call the module-level validator directly."""
        from amplifier_module_provider_openai_chatgpt.provider import (
            _validate_gpt_5_5_pro_effort,
        )

        _validate_gpt_5_5_pro_effort(model_id, reasoning_param)  # type: ignore[arg-type]

    def _make_provider(self) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        return ChatGPTProvider({"default_model": "gpt-5.5-pro"}, MagicMock(), None)

    def _make_request(self, **kwargs: object) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        messages = kwargs.pop("messages", [Message(role="user", content="hello")])
        return ChatRequest(messages=messages, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # String effort forms — rejected values
    # ------------------------------------------------------------------

    def test_rejects_low_effort(self) -> None:
        """String 'low' must raise InvalidRequestError for gpt-5.5-pro models."""
        from amplifier_core import llm_errors as kernel_errors

        with pytest.raises(kernel_errors.InvalidRequestError):
            self._validate("gpt-5.5-pro", "low")

    def test_rejects_none_effort_string(self) -> None:
        """String 'none' must raise InvalidRequestError for gpt-5.5-pro models."""
        from amplifier_core import llm_errors as kernel_errors

        with pytest.raises(kernel_errors.InvalidRequestError):
            self._validate("gpt-5.5-pro", "none")

    # ------------------------------------------------------------------
    # String effort forms — allowed values
    # ------------------------------------------------------------------

    def test_allows_medium_effort(self) -> None:
        """String 'medium' must not raise for gpt-5.5-pro models."""
        self._validate("gpt-5.5-pro", "medium")  # no error expected

    def test_allows_high_effort(self) -> None:
        """String 'high' must not raise for gpt-5.5-pro models."""
        self._validate("gpt-5.5-pro", "high")  # no error expected

    def test_allows_xhigh_effort(self) -> None:
        """String 'xhigh' must not raise for gpt-5.5-pro models."""
        self._validate("gpt-5.5-pro", "xhigh")  # no error expected

    def test_allows_unset(self) -> None:
        """None reasoning_param must not raise (effort not specified = caller's choice)."""
        self._validate("gpt-5.5-pro", None)  # no error expected

    # ------------------------------------------------------------------
    # Dict effort forms
    # ------------------------------------------------------------------

    def test_dict_form_low(self) -> None:
        """Dict {'effort': 'low'} must raise InvalidRequestError for gpt-5.5-pro models."""
        from amplifier_core import llm_errors as kernel_errors

        with pytest.raises(kernel_errors.InvalidRequestError):
            self._validate("gpt-5.5-pro", {"effort": "low"})

    def test_dict_form_medium(self) -> None:
        """Dict {'effort': 'medium'} must not raise for gpt-5.5-pro models."""
        self._validate("gpt-5.5-pro", {"effort": "medium"})  # no error expected

    # ------------------------------------------------------------------
    # Model prefix matching
    # ------------------------------------------------------------------

    def test_non_pro_model_skipped(self) -> None:
        """'gpt-5.5' (non-pro) with 'low' effort must not raise."""
        self._validate("gpt-5.5", "low")  # no error expected

    def test_dated_snapshot(self) -> None:
        """'gpt-5.5-pro-2026-04-23' still matches the gpt-5.5-pro prefix → must raise on 'low'."""
        from amplifier_core import llm_errors as kernel_errors

        with pytest.raises(kernel_errors.InvalidRequestError):
            self._validate("gpt-5.5-pro-2026-04-23", "low")

    # ------------------------------------------------------------------
    # Call-site integration: -fast suffix stripped BEFORE validation
    # ------------------------------------------------------------------

    def test_fast_suffix_stripped(self) -> None:
        """_build_payload() with model='gpt-5.5-pro-fast' and reasoning_effort='low'
        must raise InvalidRequestError, proving the validator runs AFTER -fast is stripped."""
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()
        request = self._make_request(model="gpt-5.5-pro-fast", reasoning_effort="low")

        with pytest.raises(kernel_errors.InvalidRequestError):
            provider._build_payload(request)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestAuthStatus -- ChatGPTProvider.auth_status() across all three states
# ---------------------------------------------------------------------------


class TestAuthStatus:
    """Verify auth_status() returns "authenticated" | "expired" |
    "unauthenticated" per its documented contract (provider.py)."""

    def _make_provider(self, tokens=None, token_file_path=None):
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        if token_file_path is not None:
            config["token_file_path"] = token_file_path
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def test_authenticated_when_in_memory_tokens_valid(self):
        from datetime import datetime, timedelta, timezone

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        provider = self._make_provider(
            tokens={"access_token": "tok", "expires_at": expires_at}
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value=None,
        ):
            assert provider.auth_status() == "authenticated"  # type: ignore[union-attr]

    def test_authenticated_when_disk_tokens_valid_and_memory_empty(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        provider = self._make_provider(
            tokens=None, token_file_path=str(tmp_path / "tokens.json")
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value={"access_token": "tok", "expires_at": expires_at},
        ):
            assert provider.auth_status() == "authenticated"  # type: ignore[union-attr]

    def test_expired_when_tokens_present_but_invalid(self):
        provider = self._make_provider(tokens={"access_token": "", "expires_at": None})

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value=None,
        ):
            assert provider.auth_status() == "expired"  # type: ignore[union-attr]

    def test_expired_when_only_disk_tokens_are_present_but_invalid(self, tmp_path):
        provider = self._make_provider(
            tokens=None, token_file_path=str(tmp_path / "tokens.json")
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value={"access_token": "", "expires_at": None},
        ):
            assert provider.auth_status() == "expired"  # type: ignore[union-attr]

    def test_unauthenticated_when_nothing_found_anywhere(self):
        provider = self._make_provider(tokens=None)

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value=None,
        ):
            assert provider.auth_status() == "unauthenticated"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestChatGPTProviderLogin -- instance login() wrapper
# ---------------------------------------------------------------------------


class TestChatGPTProviderLogin:
    """Verify ChatGPTProvider.login() adopts returned tokens and forwards
    print_fn through to oauth.login() (provider.py's thin wrapper)."""

    def _make_provider(self):
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config = {"token_file_path": "/tmp/chatgpt-provider-login-test-tokens.json"}
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, None)

    @pytest.mark.asyncio
    async def test_login_adopts_returned_tokens_and_forwards_print_fn(self):
        from datetime import datetime, timedelta, timezone

        provider = self._make_provider()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        new_tokens = {
            "access_token": "brand_new",
            "account_id": "acct-999",
            "expires_at": expires_at,
        }
        mock_oauth_login = AsyncMock(return_value=new_tokens)
        sentinel_print_fn = MagicMock()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.oauth_login",
            mock_oauth_login,
        ):
            result = await provider.login(print_fn=sentinel_print_fn)  # type: ignore[union-attr]

        assert result is True
        assert provider._tokens == new_tokens  # type: ignore[union-attr]
        mock_oauth_login.assert_awaited_once_with(
            token_file_path=provider._token_file_path,  # type: ignore[union-attr]
            print_fn=sentinel_print_fn,
        )


# ---------------------------------------------------------------------------
# TestListModelsAuthPropagation -- AuthenticationError propagates, no fallback
# ---------------------------------------------------------------------------


class TestListModelsAuthPropagation:
    """Verify list_models() lets AuthenticationError propagate instead of
    silently substituting the fallback catalog with a raw traceback --
    the headline onboarding defect this PR fixes (provider.py _get_catalog()).
    """

    def _make_provider(self):
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        coordinator = MagicMock()
        # No tokens at all -- _ensure_valid_tokens() will raise AuthenticationError.
        return ChatGPTProvider(config, coordinator, None)

    @pytest.mark.asyncio
    async def test_list_models_raises_authentication_error(self):
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
            return_value=None,
        ):
            with pytest.raises(kernel_errors.AuthenticationError):
                await provider.list_models()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_list_models_auth_failure_logs_no_traceback_anywhere(self, caplog):
        from amplifier_core import llm_errors as kernel_errors

        provider = self._make_provider()

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.provider.load_tokens",
                return_value=None,
            ),
            caplog.at_level(logging.DEBUG),
        ):
            with pytest.raises(kernel_errors.AuthenticationError):
                await provider.list_models()  # type: ignore[union-attr]

        for record in caplog.records:
            assert record.exc_info is None, (
                "AuthenticationError must propagate without any traceback "
                "being logged anywhere (no fallback path, no exc_info)"
            )


# ---------------------------------------------------------------------------
# TestListModelsFallbackLogging -- non-auth errors: one WARNING line, no
# exc_info; full traceback demoted to DEBUG
# ---------------------------------------------------------------------------


class TestListModelsFallbackLogging:
    """Verify the fallback path for non-auth errors logs a single WARNING
    line without exc_info, with the full traceback demoted to DEBUG
    (provider.py _get_catalog())."""

    def _make_provider(self):
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "tok",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        config: dict = {}
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    @pytest.mark.asyncio
    async def test_transport_error_falls_back_with_single_line_warning(self, caplog):
        provider = self._make_provider()
        mock_fetch = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
                mock_fetch,
            ),
            caplog.at_level(logging.DEBUG),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]

        assert len(models) > 0  # fallback catalog returned, not an empty list

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f"expected exactly one WARNING record, got {len(warning_records)}: "
            f"{[r.getMessage() for r in warning_records]}"
        )
        assert warning_records[0].exc_info is None, (
            "the WARNING record must not carry exc_info -- one line is enough"
        )

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(r.exc_info is not None for r in debug_records), (
            "the full traceback must still be available at DEBUG"
        )
