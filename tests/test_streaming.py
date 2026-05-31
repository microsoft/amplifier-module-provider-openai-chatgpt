"""Contract-conformance tests for llm:stream_* events.

These tests verify that ChatGPTProvider.complete() emits the five contract events
defined in docs/provider-streaming-contract.md.

All tests use a fake httpx response whose aiter_lines() yields canned SSE data:
lines with the Responses-API streaming event types.  The two-pass design (emit
during iteration + parse_sse_events after) means the SSE lines include BOTH
the incremental deltas AND the canonical response.output_item.done line that
parse_sse_events uses to assemble the final ChatResponse.

Note: this provider stores coordinator as self._coordinator (underscore).
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers: SSE line builders
# ---------------------------------------------------------------------------


def _j(obj: Any) -> str:
    return f"data: {json.dumps(obj)}"


def _build_text_stream_lines(
    deltas: list[str],
    output_index: int = 0,
    *,
    full_text: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[str]:
    """Build SSE lines for a text block with real streaming deltas.

    Includes both the streaming events (output_item.added/output_text.delta)
    AND the canonical output_item.done that parse_sse_events uses for assembly.
    """
    if full_text is None:
        full_text = "".join(deltas)

    lines = [
        _j({"type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4o"}}),
        _j({"type": "response.output_item.added",
            "output_index": output_index,
            "item": {"type": "message"}}),
    ]
    for delta in deltas:
        lines.append(_j({
            "type": "response.output_text.delta",
            "output_index": output_index,
            "delta": delta,
        }))
    lines += [
        _j({"type": "response.output_item.done",
            "output_index": output_index,
            "item": {"type": "message",
                     "content": [{"type": "output_text", "text": full_text}]}}),
        _j({"type": "response.done", "response": {
            "id": "resp_test", "model": "gpt-4o",
            "usage": {"input_tokens": input_tokens,
                      "output_tokens": output_tokens}}}),
        "data: [DONE]",
    ]
    return lines


def _build_thinking_stream_lines(
    deltas: list[str],
    output_index: int = 0,
    *,
    thinking_event: str = "response.reasoning_summary_text.delta",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[str]:
    """Build SSE lines for a reasoning/thinking block."""
    lines = [
        _j({"type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4o"}}),
        _j({"type": "response.output_item.added",
            "output_index": output_index,
            "item": {"type": "reasoning"}}),
    ]
    for delta in deltas:
        lines.append(_j({
            "type": thinking_event,
            "output_index": output_index,
            "delta": delta,
        }))
    lines += [
        _j({"type": "response.output_item.done",
            "output_index": output_index,
            "item": {"type": "reasoning", "summary": []}}),
        _j({"type": "response.done", "response": {
            "id": "resp_test", "model": "gpt-4o",
            "usage": {"input_tokens": input_tokens,
                      "output_tokens": output_tokens}}}),
        "data: [DONE]",
    ]
    return lines


def _build_tool_use_stream_lines(
    tool_name: str,
    tool_args_delta: str = '{"city": "London"}',
    output_index: int = 0,
    *,
    call_id: str = "call_abc",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[str]:
    """Build SSE lines for a function_call (tool_use) block."""
    return [
        _j({"type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4o"}}),
        _j({"type": "response.output_item.added",
            "output_index": output_index,
            "item": {"type": "function_call", "name": tool_name}}),
        # function_call_arguments.delta — should be silently consumed (no event emitted)
        _j({"type": "response.function_call_arguments.delta",
            "output_index": output_index, "delta": tool_args_delta}),
        _j({"type": "response.output_item.done",
            "output_index": output_index,
            "item": {"type": "function_call", "call_id": call_id,
                     "name": tool_name, "arguments": tool_args_delta}}),
        _j({"type": "response.done", "response": {
            "id": "resp_test", "model": "gpt-4o",
            "usage": {"input_tokens": input_tokens,
                      "output_tokens": output_tokens}}}),
        "data: [DONE]",
    ]


def _build_error_after_delta_lines(
    delta_before_error: str = "Hello",
    output_index: int = 0,
) -> list[str]:
    """Build SSE lines: one delta, then an error event."""
    return [
        _j({"type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4o"}}),
        _j({"type": "response.output_item.added",
            "output_index": output_index,
            "item": {"type": "message"}}),
        _j({"type": "response.output_text.delta",
            "output_index": output_index, "delta": delta_before_error}),
        _j({"type": "error",
            "error": {"message": "stream interrupted", "code": "server_error"}}),
        "data: [DONE]",
    ]


def _build_error_before_delta_lines() -> list[str]:
    """Build SSE lines: error event with NO prior deltas."""
    return [
        _j({"type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4o"}}),
        _j({"type": "error",
            "error": {"message": "bad request", "code": "invalid_request"}}),
        "data: [DONE]",
    ]


def _build_empty_delta_lines() -> list[str]:
    """SSE lines with an empty-string delta (should NOT emit block_delta)."""
    return [
        _j({"type": "response.created",
            "response": {"id": "resp_test", "model": "gpt-4o"}}),
        _j({"type": "response.output_item.added",
            "output_index": 0, "item": {"type": "message"}}),
        # empty delta — must NOT emit block_delta
        _j({"type": "response.output_text.delta",
            "output_index": 0, "delta": ""}),
        # non-empty delta — must emit block_delta
        _j({"type": "response.output_text.delta",
            "output_index": 0, "delta": "real text"}),
        _j({"type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message",
                     "content": [{"type": "output_text", "text": "real text"}]}}),
        _j({"type": "response.done", "response": {
            "id": "resp_test", "model": "gpt-4o",
            "usage": {"input_tokens": 5, "output_tokens": 2}}}),
        "data: [DONE]",
    ]


# ---------------------------------------------------------------------------
# Helpers: mock infrastructure
# ---------------------------------------------------------------------------


class _MockStreamResponse:
    """Async mock for an httpx streaming response."""

    def __init__(
        self,
        lines: list[str],
        status_code: int = 200,
        error_body: bytes = b"HTTP error body",
    ) -> None:
        self.status_code = status_code
        self._lines = lines
        self.headers = httpx.Headers({})
        self._error_body = error_body

    async def aiter_lines(self):  # type: ignore[return]
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return self._error_body


class _AsyncCM:
    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_sse_response(lines: list[str], status_code: int = 200) -> "_AsyncCM":
    response = _MockStreamResponse(lines, status_code)
    mock_client = MagicMock()
    mock_client.stream.return_value = _AsyncCM(response)
    return _AsyncCM(mock_client)


# ---------------------------------------------------------------------------
# Helpers: provider factory
# ---------------------------------------------------------------------------


def _make_provider(config: dict | None = None) -> Any:
    """Return a ChatGPTProvider with a valid coordinator and tokens."""
    from datetime import datetime, timedelta, timezone

    from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

    expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
    tokens = {
        "access_token": "test-tok",
        "account_id": "acct-123",
        "expires_at": expires_at,
    }
    coordinator = MagicMock()
    coordinator.hooks.emit = AsyncMock()

    cfg: dict = {"default_model": "gpt-4o"}
    if config:
        cfg.update(config)

    return ChatGPTProvider(cfg, coordinator, tokens)


def _emitted(provider: Any) -> list[tuple[str, dict]]:
    """Return a list of (event_name, payload) pairs in emit order."""
    return [
        (call.args[0], call.args[1])
        for call in provider._coordinator.hooks.emit.call_args_list
    ]


def _stream_events(provider: Any) -> list[tuple[str, dict]]:
    """Filter emitted events to llm:stream_* events only."""
    return [
        (name, payload)
        for name, payload in _emitted(provider)
        if name.startswith("llm:stream_")
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingContractConformance:
    """Contract-conformance tests for llm:stream_* events."""

    # ------------------------------------------------------------------
    # 1. block_start → deltas → block_end in order
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_text_block_events_in_order(self) -> None:
        """block_start → block_delta → block_end emitted in that order for a text block."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["Hello "], output_index=0, full_text="Hello ")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        names = [name for name, _ in events]

        assert "llm:stream_block_start" in names, "Expected llm:stream_block_start"
        assert "llm:stream_block_delta" in names, "Expected llm:stream_block_delta"
        assert "llm:stream_block_end" in names, "Expected llm:stream_block_end"

        start_i = names.index("llm:stream_block_start")
        delta_i = names.index("llm:stream_block_delta")
        end_i = names.index("llm:stream_block_end")
        assert start_i < delta_i < end_i, (
            f"Expected start({start_i}) < delta({delta_i}) < end({end_i})"
        )

    @pytest.mark.asyncio
    async def test_stream_block_start_payload(self) -> None:
        """llm:stream_block_start carries request_id, block_index=0, block_type='text'."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["hi"], output_index=0, full_text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        starts = [(n, p) for n, p in events if n == "llm:stream_block_start"]
        assert len(starts) == 1
        _, payload = starts[0]
        assert "request_id" in payload, "block_start must include request_id"
        assert payload["block_index"] == 0
        assert payload["block_type"] == "text"

    @pytest.mark.asyncio
    async def test_stream_block_end_payload(self) -> None:
        """llm:stream_block_end carries request_id, block_index, block_type."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["hi"], output_index=0, full_text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        ends = [(n, p) for n, p in events if n == "llm:stream_block_end"]
        assert len(ends) == 1
        _, payload = ends[0]
        assert "request_id" in payload, "block_end must include request_id"
        assert payload["block_index"] == 0
        assert payload["block_type"] == "text"

    # ------------------------------------------------------------------
    # 2. Per-block sequence counter (0-based)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_delta_sequence_is_per_block(self) -> None:
        """Three deltas for the same block → sequence values 0, 1, 2."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["A", "B", "C"], output_index=0, full_text="ABC")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        deltas = [(n, p) for n, p in events if n == "llm:stream_block_delta"]
        assert len(deltas) == 3, f"Expected 3 deltas, got {len(deltas)}"
        seqs = [p["sequence"] for _, p in deltas]
        assert seqs == [0, 1, 2], f"Expected sequences [0,1,2], got {seqs}"
        block_types = [p["block_type"] for _, p in deltas]
        assert block_types == ["text", "text", "text"], (
            f"Expected all deltas block_type='text', got {block_types}"
        )

    # ------------------------------------------------------------------
    # 3. Reasoning block → thinking_delta
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_thinking_delta_from_reasoning_summary(self) -> None:
        """reasoning_summary_text.delta → llm:stream_block_delta with block_type='thinking'."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_thinking_stream_lines(
            ["Thinking through this..."],
            thinking_event="response.reasoning_summary_text.delta",
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        # Must NOT emit the old separate thinking_delta event
        assert not any(n == "llm:stream_thinking_delta" for n, _ in events), (
            "llm:stream_thinking_delta must not be emitted (collapsed into block_delta)"
        )
        # Must emit llm:stream_block_delta with block_type='thinking'
        thinking_deltas = [
            (n, p) for n, p in events
            if n == "llm:stream_block_delta" and p.get("block_type") == "thinking"
        ]
        assert len(thinking_deltas) == 1, (
            f"Expected 1 llm:stream_block_delta(block_type='thinking'), got {len(thinking_deltas)}"
        )
        _, payload = thinking_deltas[0]
        assert payload["text"] == "Thinking through this..."
        assert payload["block_index"] == 0
        assert payload["sequence"] == 0

    @pytest.mark.asyncio
    async def test_stream_thinking_block_start_is_thinking_type(self) -> None:
        """reasoning output_item.added → llm:stream_block_start with block_type='thinking'."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_thinking_stream_lines(["..."])

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        starts = [(n, p) for n, p in events if n == "llm:stream_block_start"]
        assert len(starts) == 1
        _, payload = starts[0]
        assert payload["block_type"] == "thinking"

    # ------------------------------------------------------------------
    # 4. Tool-use block: start (with name) + end, NO arg deltas
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_tool_use_block_start_has_name(self) -> None:
        """function_call block → llm:stream_block_start with block_type='tool_use' and name."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_tool_use_stream_lines("get_weather")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        starts = [(n, p) for n, p in events if n == "llm:stream_block_start"]
        assert len(starts) == 1
        _, payload = starts[0]
        assert payload["block_type"] == "tool_use"
        assert payload.get("name") == "get_weather"

    @pytest.mark.asyncio
    async def test_stream_tool_use_no_arg_deltas(self) -> None:
        """function_call_arguments.delta is silently consumed — no llm:stream_block_delta."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_tool_use_stream_lines("get_weather")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        deltas = [n for n, _ in events if n == "llm:stream_block_delta"]
        assert deltas == [], (
            "function_call_arguments.delta must NOT emit llm:stream_block_delta"
        )

    @pytest.mark.asyncio
    async def test_stream_tool_use_block_end_emitted(self) -> None:
        """function_call block → llm:stream_block_end with block_type='tool_use'."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_tool_use_stream_lines("get_weather")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        ends = [(n, p) for n, p in events if n == "llm:stream_block_end"]
        assert len(ends) == 1
        _, payload = ends[0]
        assert payload["block_type"] == "tool_use"

    # ------------------------------------------------------------------
    # 5. Per-request override: metadata stream=False → no stream events
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_disabled_via_metadata_no_stream_events(self) -> None:
        """metadata={'stream': False} → no llm:stream_* events (still returns response)."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            metadata={"stream": False},
        )
        lines = _build_text_stream_lines(["Hello"], full_text="Hello")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            result = await provider.complete(request)

        # No stream events
        events = _stream_events(provider)
        assert events == [], f"Expected no stream events, got {events}"

        # But response is returned correctly
        assert result is not None
        assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_stream_disabled_via_metadata_still_emits_llm_events(self) -> None:
        """metadata={'stream': False} → llm:request and llm:response still emitted."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            metadata={"stream": False},
        )
        lines = _build_text_stream_lines(["Hello"], full_text="Hello")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        all_names = [n for n, _ in _emitted(provider)]
        assert "llm:request" in all_names
        assert "llm:response" in all_names

    @pytest.mark.asyncio
    async def test_stream_disabled_via_config(self) -> None:
        """config use_streaming=False → no llm:stream_* events."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider({"use_streaming": False})
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["Hello"], full_text="Hello")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        assert events == [], (
            f"Expected no stream events with use_streaming=False, got {events}"
        )

    # ------------------------------------------------------------------
    # 6. llm:stream_aborted: after partial emit / not without partial
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_aborted_after_partial_emit(self) -> None:
        """error SSE event after a delta → llm:stream_aborted emitted, then exception."""
        from amplifier_core import llm_errors as kernel_errors
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_error_after_delta_lines("Hello")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            with pytest.raises(kernel_errors.LLMError):
                await provider.complete(request)

        events = _stream_events(provider)
        aborted = [(n, p) for n, p in events if n == "llm:stream_aborted"]
        assert len(aborted) == 1, (
            f"Expected exactly 1 llm:stream_aborted, got {len(aborted)}"
        )
        _, payload = aborted[0]
        assert "request_id" in payload
        assert "error" in payload
        assert "type" in payload["error"]
        assert "msg" in payload["error"]

    @pytest.mark.asyncio
    async def test_stream_no_abort_when_no_delta_emitted(self) -> None:
        """error SSE with no prior deltas → NO llm:stream_aborted emitted."""
        from amplifier_core import llm_errors as kernel_errors
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_error_before_delta_lines()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            with pytest.raises(kernel_errors.LLMError):
                await provider.complete(request)

        events = _stream_events(provider)
        aborted = [n for n, _ in events if n == "llm:stream_aborted"]
        assert aborted == [], (
            f"Expected no llm:stream_aborted when no delta emitted, got {aborted}"
        )

    # ------------------------------------------------------------------
    # 7. Single request_id for all events in one call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_single_request_id_for_all_events(self) -> None:
        """All stream events in one complete() call share the same request_id (uuid4)."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["A", "B"], output_index=0, full_text="AB")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        assert len(events) >= 4, "Need block_start + 2 deltas + block_end"
        request_ids = {p["request_id"] for _, p in events}
        assert len(request_ids) == 1, (
            f"Expected one request_id across all stream events, got {request_ids}"
        )
        rid = next(iter(request_ids))
        try:
            parsed = uuid.UUID(rid, version=4)
        except ValueError:
            parsed = None
        assert parsed is not None, f"request_id {rid!r} is not a valid uuid4"

    # ------------------------------------------------------------------
    # 8. Empty delta is not emitted
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_empty_delta_not_emitted(self) -> None:
        """Empty-string delta is silently skipped — no llm:stream_block_delta for it."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_empty_delta_lines()

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        deltas = [(n, p) for n, p in events if n == "llm:stream_block_delta"]
        assert len(deltas) == 1, (
            f"Expected 1 delta (empty skipped), got {len(deltas)}: {deltas}"
        )
        assert deltas[0][1]["text"] == "real text"
        assert deltas[0][1]["block_type"] == "text", (
            "text delta must carry block_type='text'"
        )

    # ------------------------------------------------------------------
    # 9. Uses self._coordinator (underscore), not self.coordinator
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_uses_underscore_coordinator_attribute(self) -> None:
        """Provider emits via self._coordinator — no AttributeError, events arrive."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()

        # Structural assertion: _coordinator exists, coordinator (plain) does not
        assert hasattr(provider, "_coordinator"), (
            "Provider must expose coordinator as self._coordinator"
        )
        assert not hasattr(provider, "coordinator"), (
            "Provider must NOT have self.coordinator attribute (must use self._coordinator)"
        )

        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["hi"], full_text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            result = await provider.complete(request)

        assert result is not None
        events = _stream_events(provider)
        # With correct _coordinator usage: stream events are emitted
        assert len(events) >= 3, (
            f"Expected >= 3 stream events using _coordinator, got {len(events)}"
        )

    # ------------------------------------------------------------------
    # 10. reasoning_text.delta also maps to thinking_delta
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_reasoning_text_delta_maps_to_thinking(self) -> None:
        """response.reasoning_text.delta emits llm:stream_block_delta with block_type='thinking'."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_thinking_stream_lines(
            ["Deep thought..."],
            thinking_event="response.reasoning_text.delta",
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        # Must NOT emit the old separate thinking_delta event
        assert not any(n == "llm:stream_thinking_delta" for n, _ in events), (
            "llm:stream_thinking_delta must not be emitted (collapsed into block_delta)"
        )
        # Must emit llm:stream_block_delta with block_type='thinking'
        thinking_deltas = [
            (n, p) for n, p in events
            if n == "llm:stream_block_delta" and p.get("block_type") == "thinking"
        ]
        assert len(thinking_deltas) == 1, (
            f"Expected llm:stream_block_delta(block_type='thinking') for reasoning_text.delta, "
            f"got {thinking_deltas}"
        )

    # ------------------------------------------------------------------
    # 11. No coordinator → graceful no-op (no AttributeError)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_coordinator_no_stream_events(self) -> None:
        """Without a coordinator, streaming emits no events but returns response."""
        from datetime import datetime, timedelta, timezone

        from amplifier_core.message_models import ChatRequest, Message
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-tok",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        provider = ChatGPTProvider({"default_model": "gpt-4o"}, None, tokens)

        request = ChatRequest(messages=[Message(role="user", content="hi")])
        lines = _build_text_stream_lines(["hello"], full_text="hello")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            result = await provider.complete(request)

        assert result is not None

    # ------------------------------------------------------------------
    # 12. metadata stream=False with identity check (not ==, not truthiness)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_override_uses_identity_check_not_truthiness(self) -> None:
        """metadata['stream'] = 0 (falsy but not False) → stream events ARE emitted."""
        from amplifier_core.message_models import ChatRequest, Message

        provider = _make_provider()
        # 0 is falsy but is NOT `False` — the check must be `is False`, not bool()
        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            metadata={"stream": 0},
        )
        lines = _build_text_stream_lines(["Hello"], full_text="Hello")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            await provider.complete(request)

        events = _stream_events(provider)
        assert len(events) > 0, (
            "metadata['stream']=0 must NOT disable streaming "
            "(override requires `is False` identity check)"
        )
