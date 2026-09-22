"""Mechanism qualification only; mocked semantic recall is not live acceptance."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
)

from amplifier_module_provider_openai_chatgpt._compaction import (
    METADATA_KEY,
    checkpoint_message,
)
from amplifier_module_provider_openai_chatgpt._sse import SSEError, parse_sse_events
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

MODEL = "synthetic-model"
ITEM = {
    "type": "compaction",
    "encrypted_content": "synthetic\\nopaque==;☃",
    "id": None,
    "future": {"nullable": None, "array": [1, "two"]},
}
USAGE = {
    "input_tokens": 1000,
    "output_tokens": 80,
    "total_tokens": 1080,
    "input_tokens_details": {
        "cached_tokens": 600,
        "cache_write_tokens": 20,
        "future": None,
    },
    "output_tokens_details": {"reasoning_tokens": 50},
}


def provider(**config):
    return ChatGPTProvider(
        {
            "experimental_compaction": True,
            "default_model": MODEL,
            "token_file_path": "/nonexistent/synthetic-chatgpt-tokens",
            **config,
        },
        coordinator=MagicMock(hooks=MagicMock(emit=AsyncMock())),
        tokens={
            "access_token": "synthetic-token",
            "account_id": "synthetic-account",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    )


def events(
    *,
    item=ITEM,
    terminal="response.completed",
    status="completed",
    usage=USAGE,
    model=MODEL,
):
    return [
        {"type": "response.output_item.done", "item": deepcopy(item)},
        {
            "type": terminal,
            "response": {
                "id": "resp-synthetic",
                "model": model,
                "status": status,
                "usage": deepcopy(usage),
            },
        },
    ]


def lines(values):
    return ["data: " + json.dumps(e) for e in values]


def client_mock(values, *, error=None, status=200):
    response = MagicMock(status_code=status, headers=httpx.Headers())
    response.aread = AsyncMock(return_value=b'{"error":{"code":"not_supported"}}')

    async def iterator():
        for line in lines(values):
            yield line
        if error:
            raise error

    response.aiter_lines = iterator
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=response)
    stream.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.stream.return_value = stream
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def request():
    return ChatRequest(
        model=MODEL,
        messages=[
            Message(
                role="system", content="Keep corrections and unresolved constraints."
            ),
            Message(role="developer", content="Answer using only fixture facts."),
            Message(
                role="user",
                content="Codename ORCHID-LANTERN-624. Port 8799. Color amber. Offline operation remains required.",
            ),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call-fixture", name="lookup", input={"key": "port"}
                    )
                ],
            ),
            Message(
                role="tool",
                content=[
                    ToolResultBlock(tool_call_id="call-fixture", output="port=8799")
                ],
            ),
            Message(role="user", content="Correction: port is 8801 and color is blue."),
        ],
        tools=[
            ToolSpec(
                name="lookup",
                description="Synthetic definition; never executed",
                parameters={"type": "object"},
            )
        ],
    )


@pytest.mark.parametrize("terminal", ["response.done", "response.completed"])
def test_terminal_forms_preserve_checkpoint_and_usage_exactly_once(terminal):
    data = events(terminal=terminal)
    parsed = parse_sse_events(lines(data + data), require_compaction=True)
    assert parsed.compaction_items == [ITEM]
    assert parsed.raw_usage == USAGE
    assert parsed.input_tokens == 1000
    assert parsed.terminal_event == terminal


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "no-terminal",
        "bad-item",
        "no-status",
        "failed",
        "incomplete",
        "cancelled",
        "mixed",
        "malformed",
        "non-object",
    ],
)
def test_invalid_stream_never_yields_candidate(mutation):
    values = events()
    if mutation == "duplicate":
        values.insert(0, deepcopy(values[0]))
    elif mutation == "missing":
        values.pop(0)
    elif mutation == "no-terminal":
        values.pop()
    elif mutation == "bad-item":
        values[0]["item"]["encrypted_content"] = None
    elif mutation == "no-status":
        del values[-1]["response"]["status"]
    elif mutation in {"failed", "incomplete", "cancelled"}:
        values[-1]["type"] = "response." + mutation
    elif mutation == "mixed":
        values.insert(
            0,
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "must_not_execute",
                    "arguments": "{}",
                },
            },
        )
    elif mutation == "non-object":
        values.insert(0, ["not-an-event"])
    raw = lines(values)
    if mutation == "malformed":
        raw.insert(0, "data: {broken")
    with pytest.raises(SSEError):
        parse_sse_events(raw, require_compaction=True)


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled", "in_progress"])
def test_terminal_name_does_not_override_unsuccessful_status(status):
    with pytest.raises(SSEError):
        parse_sse_events(lines(events(status=status)), require_compaction=True)


@pytest.mark.parametrize("item", [ITEM, {k: v for k, v in ITEM.items() if k != "id"}])
def test_core_json_replay_is_lossless_and_nonmutating(item):
    p = provider()
    original = checkpoint_message(item, p._checkpoint_provenance(MODEL))
    restored = Message.model_validate_json(original.model_dump_json())
    req = request().model_copy(
        update={
            "messages": [
                request().messages[0],
                restored,
                Message(role="user", content="Continue"),
            ]
        }
    )
    wire = p._build_payload(req)
    assert wire["input"][0] == item
    wire["input"][0]["future"]["nullable"] = "changed in transport"
    assert restored == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "different-model"),
        ("endpoint", "https://different.invalid/responses"),
        ("provider", "openai"),
        ("account_sha256", "different-account"),
    ],
)
def test_incompatible_provenance_is_rejected(field, value):
    p = provider()
    provenance = p._checkpoint_provenance(MODEL)
    provenance[field] = value
    req = ChatRequest(model=MODEL, messages=[checkpoint_message(ITEM, provenance)])
    with pytest.raises(llm_errors.InvalidRequestError):
        p._build_payload(req)


@pytest.mark.parametrize("version", [0, 2, True, None, "1"])
def test_unknown_metadata_version_is_rejected(version):
    p = provider()
    msg = checkpoint_message(ITEM, p._checkpoint_provenance(MODEL))
    msg.metadata[METADATA_KEY]["version"] = version
    with pytest.raises(llm_errors.InvalidRequestError):
        p._build_payload(ChatRequest(model=MODEL, messages=[msg]))


@pytest.mark.parametrize(
    "key",
    [
        "input",
        "model",
        "instructions",
        "tools",
        "tool_choice",
        "stream",
        "store",
        "previous_response_id",
        "conversation",
        "context_management",
        "truncation",
    ],
)
def test_transport_overrides_rejected_for_compaction_and_replay(key):
    p = provider(extra_request_params={key: "conflicting"})
    with pytest.raises(llm_errors.InvalidRequestError):
        p._build_payload(request(), compaction=True)
    msg = checkpoint_message(ITEM, p._checkpoint_provenance(MODEL))
    with pytest.raises(llm_errors.InvalidRequestError):
        p._build_payload(ChatRequest(model=MODEL, messages=[msg]))


def test_payload_uses_actual_envelope_and_only_one_trigger_without_selection_policy():
    p = provider()
    req = request()
    before = req.model_dump_json()
    ordinary = p._build_payload(req)
    compact = p._build_payload(req, compaction=True)
    assert compact == {
        **ordinary,
        "input": [*ordinary["input"], {"type": "compaction_trigger"}],
    }
    assert compact["instructions"] == req.messages[0].content
    assert compact["tools"][0]["name"] == "lookup"
    assert req.model_dump_json() == before
    assert not p.supports_native_compaction()
    assert "native_compaction" not in p.get_info().capabilities
    assert not hasattr(p, "request_budget")


@pytest.mark.asyncio
async def test_disabled_by_default_and_no_auth_or_network():
    p = provider(experimental_compaction=False)
    p._ensure_valid_tokens = AsyncMock()
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.compact_checkpoint(request())
    p._ensure_valid_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_transport_operation_usage_and_raw_hooks_do_not_expose_ciphertext():
    p = provider(raw=True)
    mock = client_mock(events())
    with patch(
        "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
        return_value=mock,
    ):
        result = await p.compact_checkpoint(request())
    payload = mock.stream.call_args.kwargs["json"]
    assert payload["input"][-1] == {"type": "compaction_trigger"}
    assert (
        mock.stream.call_args.kwargs["headers"]["Authorization"]
        == "Bearer synthetic-token"
    )
    assert result.usage.input_tokens == 1000  # Includes cache; no admission discount.
    assert result.usage.cache_read_tokens == 600
    assert result.usage.cache_write_tokens == 20
    assert result.usage.reasoning_tokens == 50
    assert result.raw_usage == USAGE
    assert result.canonical_window is False and result.next_request_input_tokens is None
    calls = p._coordinator.hooks.emit.call_args_list
    assert [c.args[0] for c in calls] == ["llm:request", "llm:response"]
    assert all(c.args[1]["operation"] == "compaction" for c in calls)
    assert "synthetic\\nopaque" not in str(calls)
    assert "synthetic\\nopaque" not in repr(result)


@pytest.mark.asyncio
async def test_normal_completion_keeps_checkpoint_metadata_and_usage():
    p = provider()
    with patch(
        "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
        return_value=client_mock(events()),
    ):
        result = await p.complete(request())
    assert result.metadata[METADATA_KEY]["item"] == ITEM
    assert result.usage.cache_read_tokens == 600


@pytest.mark.asyncio
async def test_missing_usage_is_unavailable():
    p = provider()
    with patch(
        "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
        return_value=client_mock(events(usage=None)),
    ):
        result = await p.compact_checkpoint(request())
    assert result.usage is None and result.raw_usage == {}


@pytest.mark.asyncio
async def test_wrong_response_model_rejected():
    p = provider()
    with patch(
        "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
        return_value=client_mock(events(model="different")),
    ):
        with pytest.raises(llm_errors.LLMError):
            await p.compact_checkpoint(request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["cancel", "timeout", "transport", "entitlement", "context"]
)
async def test_failures_preserve_history_and_never_return_or_execute_tools(failure):
    p = provider()
    req = request()
    before = req.model_dump_json()
    errors = {
        "cancel": asyncio.CancelledError(),
        "timeout": httpx.ReadTimeout("synthetic"),
        "transport": httpx.RemoteProtocolError("synthetic"),
    }
    mock = client_mock(
        events()[:1],
        error=errors.get(failure),
        status=403 if failure == "entitlement" else 200,
    )
    if failure == "context":
        mock = client_mock(
            [
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "code": "context_length_exceeded",
                            "message": "too many tokens",
                        }
                    },
                }
            ]
        )
    expected = asyncio.CancelledError if failure == "cancel" else llm_errors.LLMError
    with patch(
        "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
        return_value=mock,
    ):
        with pytest.raises(expected):
            await p.compact_checkpoint(req)
    assert req.model_dump_json() == before
    assert mock.stream.call_count == 1
    assert not any(
        c.args[0] == "llm:stream_tool_call"
        for c in p._coordinator.hooks.emit.call_args_list
    )


@pytest.mark.asyncio
async def test_three_cycles_manager_retains_history_suffix_and_restores_provider():
    original = request()
    snapshot = original.model_dump_json()
    history = deepcopy(original.messages)
    for cycle in range(3):
        p = provider()
        selected = original.model_copy(update={"messages": history})
        item = {**ITEM, "id": f"cmp-{cycle}"}
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
            return_value=client_mock(events(item=item)),
        ):
            result = await p.compact_checkpoint(selected)
        checkpoint = Message.model_validate_json(result.checkpoint.model_dump_json())
        # This is fixture-owned retention, intentionally absent from the provider.
        retained = [
            m.model_copy(deep=True)
            for m in history
            if m.role in {"system", "developer", "user"}
        ]
        suffix = Message(role="user", content=f"CURRENT OVERLAY {cycle}: verify facts")
        next_request = original.model_copy(
            update={"messages": [*retained, checkpoint, suffix]}
        )
        wire = provider()._build_payload(next_request)
        assert wire["input"][-2] == item
        assert wire["input"][-1]["content"][0]["text"] == suffix.content
        assert all(
            "CURRENT OVERLAY" not in json.dumps(x)
            for x in p._build_payload(selected, compaction=True)["input"]
        )
        assert wire["input"][0]["role"] == "developer"
        history = [
            *retained,
            checkpoint,
            Message(role="user", content=f"Synthetic turn {cycle}"),
        ]
        assert original.model_dump_json() == snapshot


@pytest.mark.parametrize("field", ["id", "model"])
def test_created_and_terminal_response_identity_must_agree(field):
    values = events()
    created = {
        "type": "response.created",
        "response": {"id": "resp-synthetic", "model": MODEL},
    }
    created["response"][field] = "different"
    with pytest.raises(SSEError):
        parse_sse_events(lines([created, *values]), require_compaction=True)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 10},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {**USAGE, "input_tokens_details": "malformed"},
        "malformed",
    ],
)
def test_malformed_operation_usage_is_not_a_measurement(usage):
    with pytest.raises(SSEError):
        parse_sse_events(lines(events(usage=usage)), require_compaction=True)


@pytest.mark.asyncio
async def test_full_history_fallback_remains_caller_owned():
    p = provider()
    req = request()
    saved = req.model_dump_json()
    bad = client_mock(events()[:1])
    good = client_mock(
        [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Synthetic full-history continuation",
                        }
                    ],
                },
            },
            events()[-1],
        ]
    )
    with patch(
        "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient",
        side_effect=[bad, good],
    ):
        with pytest.raises(llm_errors.LLMError):
            await p.compact_checkpoint(req)
        assert bad.stream.call_count == 1
        await p.complete(req)
    assert req.model_dump_json() == saved
    assert all(
        item.get("type") != "compaction_trigger"
        for item in good.stream.call_args.kwargs["json"]["input"]
    )


def test_checkpoint_replay_preserves_changed_current_envelope():
    p = provider()
    checkpoint = checkpoint_message(ITEM, p._checkpoint_provenance(MODEL))
    req = request().model_copy(
        update={
            "messages": [
                Message(role="system", content="New system envelope"),
                checkpoint,
                Message(role="user", content="New pending turn"),
            ],
            "tools": [
                ToolSpec(
                    name="new_tool",
                    description="New schema",
                    parameters={"type": "object"},
                )
            ],
        }
    )
    wire = p._build_payload(req)
    assert wire["instructions"] == "New system envelope"
    assert wire["tools"][0]["name"] == "new_tool"
    assert wire["input"][0] == ITEM


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_output_tokens", 1000),
        ("temperature", 0.1),
        ("top_p", 0.9),
        ("stop", ["STOP"]),
        ("conversation_id", "synthetic-conversation"),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_native_request_never_silently_drops_unsupported_envelope_fields(field, value):
    req = request().model_copy(update={field: value})
    with pytest.raises(llm_errors.InvalidRequestError):
        provider()._build_payload(req, compaction=True)


def test_native_tool_choice_is_preserved():
    req = request().model_copy(update={"tool_choice": "none"})
    assert provider()._build_payload(req, compaction=True)["tool_choice"] == "none"


@pytest.fixture(autouse=True)
def no_live_auth_in_compaction_tests(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Regression fixture must never read live credentials")

    monkeypatch.setattr(
        "amplifier_module_provider_openai_chatgpt.provider.load_tokens", denied
    )
