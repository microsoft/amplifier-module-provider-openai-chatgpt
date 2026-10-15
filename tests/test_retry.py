"""Request-only retry campaigns using real HTTPX with synthetic transport."""
import asyncio
import json
from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils import retry as core_retry

from amplifier_module_provider_openai_chatgpt import provider as module

pytestmark = pytest.mark.asyncio


def wire(*events):
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()


def success(text="OK"):
    return httpx.Response(200, content=wire(
        {"type": "response.output_item.done", "item": {"type": "message", "content": [
            {"type": "output_text", "text": text}]}},
        {"type": "response.completed", "response": {"id": "fixture", "usage": {
            "input_tokens": 3, "output_tokens": 1}}},
    ))


def failure(status=503, code=None, hint=None):
    headers = {} if hint is None else {"retry-after": str(hint)}
    return httpx.Response(status, headers=headers, json={"error": {
        **({"code": code} if code else {}), "message": "synthetic service failure"}})


def sse_failure(code="server_error", event_type="error", flat=False, hint=None):
    error = {"code": code, "message": "Our servers are currently overloaded. Please try again later."}
    if hint is not None:
        error["retry_after"] = hint
    event = {"type": event_type, **error} if flat else (
        {"type": event_type, "error": error} if event_type == "error" else {
            "type": event_type, "response": {"error": error}})
    return httpx.Response(200, content=wire(event))


@pytest.fixture
def harness(monkeypatch):
    sleeps = AsyncMock()
    monkeypatch.setattr(core_retry.asyncio, "sleep", sleeps)
    calls, events = [], []
    real_client = httpx.AsyncClient

    def install(responses):
        async def receive(request):
            calls.append(request)
            if callable(responses):
                result = responses(request)
                if hasattr(result, "__await__"):
                    result = await result
            else:
                result = responses.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: real_client(
            **kw, transport=httpx.MockTransport(receive)))

    def make(mode="chatgpt_codex", hooks=True, **config):
        async def emit(event, data):
            events.append((event, data))
        coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit)) if hooks else None
        p = module.ChatGPTProvider({"auth_mode": mode, "default_model": "fixture",
            "retry_jitter": False, **config}, coordinator)
        p._ensure_valid_tokens = AsyncMock()
        p._build_headers = lambda: {"authorization": "Bearer synthetic-credential"}
        return p

    req = ChatRequest(messages=[Message(role="user", content="synthetic request")])
    return SimpleNamespace(make=make, install=install, calls=calls, events=events, sleeps=sleeps, request=req)


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("kind", ["http503", "http429", "error", "response.failed", "response.incomplete", "flat", "connection", "timeout"])
async def test_transient_then_success_retains_one_logical_request(harness, mode, kind):
    h = harness
    first = {
        "http503": failure(), "http429": failure(429),
        "error": sse_failure(), "response.failed": sse_failure(event_type="response.failed"),
        "response.incomplete": sse_failure(event_type="response.incomplete"),
        "flat": sse_failure(flat=True), "connection": httpx.ConnectError("synthetic"),
        "timeout": httpx.ReadTimeout("synthetic"),
    }[kind]
    h.install([first, success()])
    p = h.make(mode)
    result = await p.complete(h.request)
    assert result.content[0].text == "OK"
    assert len(h.calls) == 2 and h.calls[0].content == h.calls[1].content
    assert h.calls[0].url == h.calls[1].url
    assert h.calls[0].headers["authorization"] == h.calls[1].headers["authorization"]
    assert p._ensure_valid_tokens.await_count == 1
    names = [name for name, _ in h.events]
    assert names.count("llm:request") == names.count("llm:response") == 1
    assert names.count("provider:retry") == 1
    assert not any(data.get("status") == "error" for _, data in h.events)
    assert h.sleeps.await_args_list[0].args == (1.0,)
    retry = next(data for name, data in h.events if name == "provider:retry")
    assert retry["attempt"] == 1 and retry["max_retries"] == 5
    assert "synthetic-credential" not in str(retry) and "synthetic request" not in str(retry)


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("retries", [0, 2, 5])
async def test_exhaustion_and_disable_bound_physical_requests(harness, mode, retries):
    h = harness
    h.install([failure() for _ in range(retries + 1)])
    with pytest.raises(errors.ProviderUnavailableError):
        await h.make(mode, max_retries=retries).complete(h.request)
    assert len(h.calls) == retries + 1
    assert h.sleeps.await_count == retries
    assert sum(name == "llm:response" for name, _ in h.events) == 1
    assert h.events[-1][1]["status"] == "error"


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("status,code,kind", [
    (429, "insufficient_quota", errors.QuotaExceededError),
    (503, "billing_hard_limit_reached", errors.QuotaExceededError),
    (503, "invalid_api_key", errors.AuthenticationError),
    (429, "context_length_exceeded", errors.ContextLengthError),
    (503, "content_policy_violation", errors.ContentFilterError),
    (503, "invalid_request_error", errors.InvalidRequestError),
    (429, "unknown_service_code", errors.LLMError),
    (503, "unknown_service_code", errors.LLMError),
    (403, None, errors.AccessDeniedError),
    (404, None, errors.NotFoundError),
    (400, None, errors.InvalidRequestError),
])
async def test_permanent_codes_override_transient_status(harness, mode, status, code, kind):
    h = harness
    h.install([failure(status, code, hint=1)])
    with pytest.raises(kind) as caught:
        await h.make(mode).complete(h.request)
    assert caught.value.retryable is False
    assert len(h.calls) == 1 and not h.sleeps.await_count


@pytest.mark.parametrize("code", ["subscription_sharing_usage_limit_exceeded", "chatpass_v2_bad_route", "subscription_sharing_invalid_user", "subscription_sharing_unsupported_capability"])
async def test_plan_permanent_codes_win_over_http503(harness, code):
    h = harness
    h.install([failure(503, code)])
    with pytest.raises(errors.LLMError) as caught:
        await h.make("chatgpt_plan").complete(h.request)
    assert not caught.value.retryable and len(h.calls) == 1
    assert caught.value.error_code == code


@pytest.mark.parametrize("hint", ["4", "http-date"])
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
async def test_server_hint_floor_even_when_core_jitters_down(harness, monkeypatch, hint, mode):
    h = harness
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    if hint == "http-date":
        hint = format_datetime(datetime.fromtimestamp(1004, UTC), usegmt=True)
    # Simulate the shared helper's jitter result below its server floor.
    monkeypatch.setattr(core_retry, "_rust_compute_delay", lambda *a, **kw: 2.0)
    h.install([failure(503, hint=hint), success()])
    await h.make(mode).complete(h.request)
    assert [call.args[0] for call in h.sleeps.await_args_list] == [2.0, 2.0]
    retry = next(data for name, data in h.events if name == "provider:retry")
    assert retry["delay"] == 4.0


@pytest.mark.parametrize("hint", [61, float("inf"), float("nan"), -1, "bad date"])
async def test_long_hint_declined_invalid_hints_ignored(harness, hint):
    h = harness
    h.install([failure(hint=hint), success()])
    if hint == 61:
        with pytest.raises(errors.ProviderUnavailableError) as caught:
            await h.make().complete(h.request)
        assert not caught.value.retryable and len(h.calls) == 1
    else:
        await h.make().complete(h.request)
        assert len(h.calls) == 2


async def test_sse_server_hint(harness):
    h = harness
    h.install([sse_failure(hint=4), success()])
    await h.make().complete(h.request)
    assert sum(call.args[0] for call in h.sleeps.await_args_list) >= 4


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("item", ["message", "reasoning", "function_call"])
@pytest.mark.parametrize("partial", [False, True])
async def test_first_block_start_fences_output_and_outer_retry(harness, mode, item, partial):
    h = harness
    events = [{"type": "response.output_item.added", "output_index": 0, "item": {"type": item, "name": "effect"}}]
    if partial:
        events.append({"type": "response.output_text.delta", "output_index": 0, "delta": "partial"})
    events.append({"type": "error", "error": {"code": "server_error", "message": "synthetic"}})
    h.install([httpx.Response(200, content=wire(*events))])
    with pytest.raises(errors.ProviderUnavailableError) as caught:
        await h.make(mode).complete(h.request)
    assert caught.value.retryable is False
    assert len(h.calls) == 1
    assert sum(name == "llm:stream_aborted" for name, _ in h.events) == 1


async def test_hook_effect_is_fenced_before_await(harness):
    h = harness
    p = h.make()
    async def emit(event, data):
        if event == "llm:stream_block_start":
            raise errors.LLMTimeoutError("unknown hook effects")
    p._coordinator.hooks.emit = emit
    h.install([httpx.Response(200, content=wire({"type": "response.output_item.added", "item": {"type": "function_call"}}))])
    with pytest.raises(errors.LLMTimeoutError) as caught:
        await p.complete(h.request)
    assert not caught.value.retryable and len(h.calls) == 1


async def test_buffered_request_can_retry_without_exposing_failed_attempt(harness):
    h = harness
    h.request.metadata = {"stream": False}
    h.install([sse_failure(), success()])
    await h.make().complete(h.request)
    assert len(h.calls) == 2
    assert not any(name.startswith("llm:stream") for name, _ in h.events)


async def test_auth_refresh_is_once_per_logical_request_not_per_campaign(harness):
    h = harness
    h.install([failure(401), failure(503), failure(401)])
    p = h.make()
    with pytest.raises(errors.AuthenticationError):
        await p.complete(h.request)
    assert len(h.calls) == 3 and p._ensure_valid_tokens.await_count == 2
    assert len({call.content for call in h.calls}) == 1


async def test_zero_transient_retries_keeps_codex_auth_refresh(harness):
    h = harness
    h.install([failure(401), success()])
    p = h.make(max_retries=0)
    await p.complete(h.request)
    assert len(h.calls) == 2 and not h.sleeps.await_count


async def test_cancellation_during_backoff_never_reissues(harness, monkeypatch):
    h = harness
    entered = asyncio.Event()
    async def wait(delay):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(core_retry.asyncio, "sleep", wait)
    h.install([failure()])
    task = asyncio.create_task(h.make().complete(h.request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(h.calls) == 1


async def test_concurrent_request_state_is_local(harness):
    h = harness
    counts = {}
    def receive(req):
        body = req.content
        counts[body] = counts.get(body, 0) + 1
        return failure() if counts[body] == 1 else success()
    h.install(receive)
    p = h.make()
    requests = [ChatRequest(messages=[Message(role="user", content=text)]) for text in ("one", "two")]
    await asyncio.gather(*(p.complete(req) for req in requests))
    assert list(counts.values()) == [2, 2]
    retries = [data for name, data in h.events if name == "provider:retry"]
    assert len({r["request_id"] for r in retries}) == 2
    assert all(r["attempt"] == 1 for r in retries)


@pytest.mark.parametrize("config", [
    {"max_retries": -1}, {"min_retry_delay": -1}, {"max_retry_delay": float("inf")},
    {"min_retry_delay": float("nan")}, {"min_retry_delay": 2, "max_retry_delay": 1},
])
async def test_invalid_retry_configuration_fails_at_construction(config):
    with pytest.raises(ValueError):
        module.ChatGPTProvider(config)


async def test_config_string_forms_and_defaults():
    p = module.ChatGPTProvider()
    assert p._retry_config.max_retries == 5
    assert p._retry_config.initial_delay == 1 and p._retry_config.max_delay == 60
    p = module.ChatGPTProvider({"max_retries": "2", "min_retry_delay": "0.5", "retry_jitter": "false"})
    assert p._retry_config.max_retries == 2 and p._retry_config.initial_delay == 0.5
    assert not p._retry_config.jitter


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
async def test_incomplete_without_transient_error_is_terminal(harness, reason):
    h = harness
    h.install([httpx.Response(200, content=wire({
        "type": "response.incomplete",
        "response": {"incomplete_details": {"reason": reason}},
    }))])
    with pytest.raises(errors.LLMError) as caught:
        await h.make().complete(h.request)
    assert not caught.value.retryable and len(h.calls) == 1


async def test_retry_config_is_captured_for_campaign(harness):
    h = harness
    p = h.make(max_retries=1, max_retry_delay=10)
    async def emit(event, data):
        h.events.append((event, data))
        if event == "provider:retry":
            # Simulate a host replacing configuration while this call waits.
            p._retry_config = module.RetryConfig(max_retries=0, max_delay=0)
    p._coordinator.hooks.emit = emit
    h.install([failure(), failure(hint=5)])
    with pytest.raises(errors.ProviderUnavailableError) as caught:
        await p.complete(h.request)
    assert caught.value.retryable is True  # Original limit10, not replacement0.
    assert len(h.calls) == 2
    assert next(data for name, data in h.events if name == "provider:retry")["max_retries"] == 1
