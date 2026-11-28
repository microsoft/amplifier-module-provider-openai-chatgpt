"""Real loopback sockets and trace contracts; no account or external service."""

import asyncio
import json
import socket
import sys
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import LLMTimeoutError, ProviderUnavailableError

from amplifier_module_provider_openai_chatgpt import _transport
from amplifier_module_provider_openai_chatgpt import provider as module


def configured(mode, timeout=None):
    provider = module.ChatGPTProvider(
        {
            "auth_mode": mode,
            "default_model": "fake",
            "timeout": timeout,
            "max_retries": 0,
        }
    )
    # Never read a credential file or enter any sign-in/refresh path.
    provider._ensure_valid_tokens = AsyncMock()
    provider._build_headers = dict
    return provider


def request():
    return ChatRequest(messages=[Message(role="user", content="fixture")])


@asynccontextmanager
async def quiet_server(phase):
    accepted, release = asyncio.Event(), asyncio.Event()
    handlers = set()
    body = (
        "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "fake",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        )
        + "\n\n"
    ).encode()
    headers = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\nConnection: close\r\n\r\n"
    )

    async def serve(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            size = next(
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(size)
            if phase == "body":
                writer.write(headers)
                await writer.drain()
            accepted.set()
            await release.wait()
            writer.write((headers if phase == "headers" else b"") + body)
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1], accepted, release
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 2)


def local_endpoint(monkeypatch, port):
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(module, "CHATGPT_CODEX_ENDPOINT", base + "/responses")
    monkeypatch.setattr(module.plan_auth, "RESOURCE", base)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("via_proxy", [False, True])
@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_each_model_connection_has_keepalive_and_allows_quiet_peer(
    monkeypatch,
    caplog,
    mode,
    via_proxy,
    phase,
):
    inspected, prior_events = [], []
    original = module.configure_request

    async def previous(event, info):
        prior_events.append(event)

    async def configure(outgoing):
        outgoing.extensions["trace"] = previous
        await original(outgoing)
        trace = outgoing.extensions["trace"]

        async def inspect(event, info):
            log_start = len(caplog.records)
            await trace(event, info)
            if event.endswith("connect_tcp.complete"):
                sock = info["return_value"].get_extra_info("socket")
                names = (
                    "SO_KEEPALIVE",
                    "TCP_KEEPIDLE",
                    "TCP_KEEPALIVE",
                    "TCP_KEEPINTVL",
                    "TCP_KEEPCNT",
                )
                constants = {name: getattr(socket, name, None) for name in names}
                idle_name = (
                    "TCP_KEEPIDLE"
                    if constants["TCP_KEEPIDLE"] is not None
                    else "TCP_KEEPALIVE"
                )
                readback = {}
                for name in ("SO_KEEPALIVE", idle_name, "TCP_KEEPINTVL", "TCP_KEEPCNT"):
                    if constants[name] is not None:
                        level = (
                            socket.SOL_SOCKET
                            if name == "SO_KEEPALIVE"
                            else socket.IPPROTO_TCP
                        )
                        try:
                            readback[name] = sock.getsockopt(level, constants[name])
                        except OSError as error:
                            readback[name] = str(error)
                print(
                    "W9_KEEPALIVE_SOCKET "
                    + json.dumps(
                        {
                            "platform": sys.platform,
                            "python": sys.version.split()[0],
                            "mode": mode,
                            "proxy": via_proxy,
                            "quiet_phase": phase,
                            "constants": constants,
                            "readback": readback,
                            "refused_options": [
                                row.getMessage()
                                for row in caplog.records[log_start:]
                                if row.name == _transport.__name__
                                and row.levelname == "WARNING"
                            ],
                        }
                    ),
                    flush=True,
                )
                inspected.append(list(readback.values()))

        outgoing.extensions["trace"] = inspect

    monkeypatch.setattr(module, "configure_request", configure)
    async with quiet_server(phase) as (port, accepted, release):
        local_endpoint(monkeypatch, port)
        if via_proxy:
            # Local target and local HTTP proxy. The proxy must be used, since
            # the target port is closed. No DNS or remote request is possible.
            local_endpoint(monkeypatch, 1)
            real_client = httpx.AsyncClient
            monkeypatch.setattr(
                module.httpx,
                "AsyncClient",
                lambda **kw: real_client(
                    **kw,
                    proxy=f"http://127.0.0.1:{port}",
                ),
            )
        provider = configured(mode)
        for _ in range(2):
            accepted.clear()
            release.clear()
            task = asyncio.create_task(provider.complete(request()))
            try:
                await asyncio.wait_for(accepted.wait(), 2)
                done, _ = await asyncio.wait({task}, timeout=0.05)
                assert not done, "A live quiet model must not be cut off"
                release.set()
                result = await asyncio.wait_for(task, 2)
                assert result.metadata["provider_response_completed"] is True
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    expected = [1]
    if hasattr(socket, "TCP_KEEPIDLE") or hasattr(socket, "TCP_KEEPALIVE"):
        expected.append(30)
    if hasattr(socket, "TCP_KEEPINTVL"):
        expected.append(10)
    if hasattr(socket, "TCP_KEEPCNT"):
        expected.append(3)
    assert len(inspected) == 2
    for values in inspected:
        # getsockopt returns a nonzero flag, not necessarily 1: Darwin returns
        # the SO_KEEPALIVE bit (8). Retain that raw value in the job log above.
        assert isinstance(values[0], int) and values[0] != 0
        assert values[1:] == expected[1:]
    assert sum(event.endswith("connect_tcp.complete") for event in prior_events) == 2
    assert any(
        event.endswith("receive_response_headers.started") for event in prior_events
    )
    print(
        "W9_KEEPALIVE_COMPLETE "
        + json.dumps(
            {
                "platform": sys.platform,
                "python": sys.version.split()[0],
                "mode": mode,
                "proxy": via_proxy,
                "quiet_phase": phase,
                "completed_requests": 2,
            }
        ),
        flush=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_explicit_read_timeout_still_ends_a_quiet_request(
    monkeypatch, mode, phase
):
    async with quiet_server(phase) as (port, accepted, release):
        local_endpoint(monkeypatch, port)
        with pytest.raises(LLMTimeoutError) as caught:
            await asyncio.wait_for(
                configured(mode, timeout=0.05).complete(request()), 2
            )
        assert accepted.is_set()
        assert caught.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize(
    "failure, expected",
    [
        (httpx.ReadError, ProviderUnavailableError),
        (httpx.ReadTimeout, LLMTimeoutError),
    ],
)
async def test_transport_and_timeout_error_classes_are_unchanged(
    monkeypatch, mode, failure, expected
):
    async def fail(outgoing):
        raise failure("fixture", request=outgoing)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real_client(
            **kw,
            transport=httpx.MockTransport(fail),
        ),
    )
    with pytest.raises(expected) as caught:
        await configured(mode).complete(request())
    assert caught.value.retryable is True
    assert caught.value.provider == "openai-chatgpt"


@pytest.mark.parametrize("idle_name", ["TCP_KEEPIDLE", "TCP_KEEPALIVE", None])
def test_platform_constants_and_missing_options(monkeypatch, idle_name):
    constants = {"SOL_SOCKET": 1, "SO_KEEPALIVE": 2, "IPPROTO_TCP": 3}
    if idle_name:
        constants.update({idle_name: 4, "TCP_KEEPINTVL": 5, "TCP_KEEPCNT": 6})
    monkeypatch.setattr(_transport, "socket", SimpleNamespace(**constants))
    expected = [(1, 2, 1)]
    if idle_name:
        expected += [(3, 4, 30), (3, 5, 10), (3, 6, 3)]
    assert _transport.keepalive_options() == expected


@pytest.mark.asyncio
async def test_unsupported_option_warns_once_and_keeps_other_options(caplog):
    calls = []
    options = _transport.keepalive_options()
    rejected = options[0]

    def set_option(*args):
        calls.append(args)
        if args == rejected:
            raise OSError("fixture: unsupported by this kernel")

    outgoing = httpx.Request("POST", "http://127.0.0.1/responses")
    await _transport.configure_request(outgoing)
    stream = SimpleNamespace(
        get_extra_info=lambda key: SimpleNamespace(setsockopt=set_option)
    )
    await outgoing.extensions["trace"](
        "connection.connect_tcp.complete", {"return_value": stream}
    )
    assert calls == options
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"TCP keepalive option {rejected[1]} is unavailable" in caplog.text


@pytest.mark.asyncio
async def test_trace_forwards_all_events_and_tolerates_missing_socket():
    seen, calls = [], []

    async def previous(event, info):
        seen.append((event, info))

    outgoing = httpx.Request(
        "POST", "http://127.0.0.1/responses", extensions={"trace": previous}
    )
    await _transport.configure_request(outgoing)
    stream = SimpleNamespace(
        get_extra_info=lambda key: SimpleNamespace(
            setsockopt=lambda *args: calls.append(args),
        )
    )
    events = [
        ("connection.connect_tcp.started", {}),
        ("connection.connect_tcp.complete", {"return_value": None}),
        (
            "connection.connect_tcp.complete",
            {"return_value": SimpleNamespace(get_extra_info=lambda key: None)},
        ),
        ("connection.connect_tcp.complete", {"return_value": stream}),
        ("proxy.connect_tcp.complete", {"return_value": stream}),
        ("http11.receive_response_headers.complete", {"return_value": stream}),
    ]
    for event, info in events:
        await outgoing.extensions["trace"](event, info)
    assert seen == events
    assert calls == _transport.keepalive_options() * 2
