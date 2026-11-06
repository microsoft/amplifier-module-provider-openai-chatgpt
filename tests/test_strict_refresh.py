"""Strict discovery refresh regressions using real files and mocked HTTP only.

These tests deliberately do not replace token loading, refresh, or validation.
The legacy refresh contract remains covered by test_oauth.py.

The cross-process guarantee covers cooperating provider writers using the same
resolved-path lock. An arbitrary external rename that ignores that advisory
lock is not fenced; detecting an earlier replacement is not such a guarantee.
"""

import asyncio
import base64
import json
import logging
import os
import selectors
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
from amplifier_core import llm_errors

from amplifier_module_provider_openai_chatgpt import models, oauth
from amplifier_module_provider_openai_chatgpt.provider import (
    CODEX_MODE,
    ChatGPTProvider,
)


_ASYNC_CLIENT = httpx.AsyncClient
_ACCOUNT_A = "synthetic-account-A"
_ACCOUNT_B = "synthetic-account-B"
_ACCESS_A = "SENTINEL-EXPIRED-ACCESS-A"
_REFRESH_A = "SENTINEL-REFRESH-A+with&reserved=characters"
_FRESH_ACCESS = "SENTINEL-FRESH-ACCESS-A"
_FRESH_REFRESH = "SENTINEL-FRESH-REFRESH-A"
_ACCESS_B = "SENTINEL-ACCESS-B"
_REFRESH_B = "SENTINEL-REFRESH-B"
_SECRETS = (
    _ACCESS_A,
    _REFRESH_A,
    _FRESH_ACCESS,
    _FRESH_REFRESH,
    _ACCESS_B,
    _REFRESH_B,
)
_LEAK = " / ".join(_SECRETS)


@pytest.fixture(autouse=True)
def isolate_external_io(monkeypatch, tmp_path):
    """An omitted transport cannot reach the network or default credentials."""

    def forbidden_client(*args, **kwargs):
        raise AssertionError("Every HTTP request must use the test MockTransport")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    monkeypatch.setattr(httpx, "Client", forbidden_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    default_path = tmp_path / "unused-default-credentials.json"
    monkeypatch.setattr(oauth, "TOKEN_FILE_PATH", str(default_path))
    yield
    assert not default_path.exists(), "Strict discovery wrote a default credential file"


def _tokens(account=_ACCOUNT_A, *, expired=True):
    return {
        "auth_mode": "oauth",
        "account_id": account,
        "access_token": _ACCESS_A if account == _ACCOUNT_A else _ACCESS_B,
        "refresh_token": _REFRESH_A if account == _ACCOUNT_A else _REFRESH_B,
        "plan_type": "plus" if account == _ACCOUNT_A else "pro",
        "expires_at": (
            "2000-01-01T00:00:00+00:00"
            if expired
            else "2999-01-01T00:00:00+00:00"
        ),
    }


def _write_tokens(path, tokens):
    # Preserve formatting too: failed refresh must not rewrite equivalent JSON.
    content = (json.dumps(tokens, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    return content


_CREDENTIAL_PROCESS = r"""
import json
import sys
from pathlib import Path

# Use the parent's actual imported package, including in installed-wheel checks,
# rather than accidentally importing another checkout from the child's cwd.
module_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(module_path.parent.parent))
from amplifier_module_provider_openai_chatgpt import oauth
from filelock import FileLock, Timeout
import httpx

assert Path(oauth.__file__).resolve() == module_path
path = Path(sys.argv[2])
tokens = json.loads(sys.argv[3])
mode = sys.argv[4]
default_path = path.parent / "unused-child-default-credentials.json"
oauth.TOKEN_FILE_PATH = str(default_path)

def forbidden_client(*args, **kwargs):
    raise AssertionError("Credential writer must not use HTTP")

httpx.AsyncClient = forbidden_client
httpx.Client = forbidden_client
lock_path = str(path.resolve()) + ".lock"
print("attempting", flush=True)

if mode == "hold":
    with FileLock(lock_path, timeout=0, mode=0o600):
        print("locked", flush=True)
        assert sys.stdin.readline().strip() == "release"
    print("released", flush=True)
else:
    assert mode in ("save", "serialize")
    if mode == "serialize":
        try:
            with FileLock(lock_path, timeout=0, mode=0o600):
                pass
        except Timeout:
            print("lock-busy", flush=True)
        else:
            raise AssertionError("Strict comparison/commit did not hold the lock")

        class ObservedLock(FileLock):
            def _acquire(self):
                super()._acquire()
                if not self.is_locked and not getattr(self, "reported", False):
                    self.reported = True
                    print("save-blocked", flush=True)

        # Observe the real save_tokens acquisition failing, without replacing
        # its blocking, retry, timeout or release behavior.
        oauth.FileLock = ObservedLock
        original_write = oauth._write_tokens_atomic

        def observe_write(record, destination):
            print(
                "before-save:" + json.dumps(json.loads(destination.read_bytes())),
                flush=True,
            )
            original_write(record, destination)

        oauth._write_tokens_atomic = observe_write

    oauth.save_tokens(tokens, str(path))
    print("saved", flush=True)

assert not default_path.exists()
"""


@pytest.fixture
def credential_process(tmp_path):
    """Own every synthetic writer, including cleanup on failed assertions."""
    children = []

    def spawn(path, mode):
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _CREDENTIAL_PROCESS,
                str(Path(oauth.__file__).resolve()),
                str(path),
                json.dumps(_tokens(_ACCOUNT_B, expired=False)),
                mode,
            ],
            cwd=tmp_path,
            env={**os.environ, "HOME": str(tmp_path), "PYTHONIOENCODING": "utf-8"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # No buffered read-ahead: each readiness check matches the next line.
            bufsize=0,
        )
        children.append(child)
        return child

    yield spawn
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()


def _process_line(child):
    # Pipe handshakes establish ordering; timeouts only bound a broken protocol.
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=10), "Credential subprocess did not respond"
        line = child.stdout.readline()
    assert line, f"Credential subprocess exited early ({child.poll()})"
    return line.decode("utf-8").rstrip("\n")


def _provider(path, *, injected=False):
    config = {
        "auth_mode": CODEX_MODE,
        "token_file_path": str(path),
        "default_model": "configured-selection-not-in-inventory",
        "model": "another-selection-not-in-inventory",
        "models_client_version": "strict-refresh-fixture",
    }
    if injected:
        return ChatGPTProvider(config, tokens=_tokens())
    # File-backed instances must not accidentally become explicitly injected.
    return ChatGPTProvider(config)


def _id_token(account):
    payload = {
        "https://api.openai.com/profile": {"account_id": account},
        "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return "eyJhbGciOiJub25lIn0." + encoded.decode() + "."


def _refreshed(*, with_id_token=False, rotate_refresh=True):
    payload = {"access_token": _FRESH_ACCESS, "expires_in": 3600}
    if rotate_refresh:
        payload["refresh_token"] = _FRESH_REFRESH
    if with_id_token:
        payload["id_token"] = _id_token(_ACCOUNT_A)
    return payload


def _install_transport(monkeypatch, handler):
    requests = []

    async def record(request):
        requests.append(request)
        return await handler(request)

    def client(**kwargs):
        return _ASYNC_CLIENT(
            transport=httpx.MockTransport(record), trust_env=False, **kwargs
        )

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return requests


def _assert_refresh(request):
    assert request.method == "POST"
    assert str(request.url) == oauth.OAUTH_TOKEN_URL
    assert request.headers["content-type"].startswith(
        "application/x-www-form-urlencoded"
    )
    assert parse_qs(request.content.decode(), strict_parsing=True) == {
        "grant_type": ["refresh_token"],
        "refresh_token": [_REFRESH_A],
        "client_id": [oauth.OAUTH_CLIENT_ID],
    }


def _assert_catalog(request):
    assert request.method == "GET"
    assert str(request.url.copy_with(query=None)) == models.MODELS_ENDPOINT
    assert dict(request.url.params) == {"client_version": "strict-refresh-fixture"}
    assert request.headers["authorization"] == f"Bearer {_FRESH_ACCESS}"
    assert request.headers["chatgpt-account-id"] == _ACCOUNT_A
    assert request.content == b""


def _catalog_response():
    return httpx.Response(
        200,
        json={
            "models": [
                {
                    "slug": "account-A-live-model",
                    "display_name": "Account A live model",
                    "visibility": "list",
                    "supported_in_api": True,
                    "context_window": 64000,
                    "max_output_tokens": 8000,
                }
            ]
        },
    )


def _assert_result(result):
    assert set(result) == {"version", "source", "models"}
    assert result["version"] == 1
    assert result["source"] == "provider-api"
    assert [model.id for model in result["models"]] == ["account-A-live-model"]


def _assert_one_refresh(requests):
    assert len(requests) == 1, (
        "Failure must not fetch a catalog or retry another account"
    )
    _assert_refresh(requests[0])


def _assert_safe_error(
    error, caplog, expected, *, retryable, status=None, retry_after=None
):
    # Exact type matters: LLMError must not silently accept an AuthenticationError.
    assert type(error) is expected
    assert error.provider == "openai-chatgpt"
    assert error.retryable is retryable
    assert error.status_code == status
    assert error.retry_after == retry_after
    diagnostics = "\n".join(
        (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
            caplog.text,
        )
    )
    for secret in (*_SECRETS, _id_token(_ACCOUNT_A), _id_token(_ACCOUNT_B)):
        assert secret not in diagnostics
    # A safe message alone is insufficient if Python renders the unsafe cause.
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__


@pytest.mark.asyncio
@pytest.mark.parametrize("with_id_token", [False, True], ids=["no-id", "same-id"])
@pytest.mark.parametrize("change", ["replace", "remove"])
async def test_file_changed_during_refresh_fails_without_catalog_or_overwrite(
    monkeypatch, tmp_path, caplog, with_id_token, change
):
    path = tmp_path / "credentials.json"
    _write_tokens(path, _tokens())
    provider = _provider(path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        _assert_refresh(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=_refreshed(with_id_token=with_id_token))

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    task = asyncio.create_task(provider.discover_models())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
        if change == "replace":
            replacement = tmp_path / "replacement.json"
            replacement_bytes = _write_tokens(
                replacement, _tokens(_ACCOUNT_B, expired=False)
            )
            replacement.replace(path)
        else:
            path.unlink()
        release.set()
        with pytest.raises(llm_errors.ProviderUnavailableError) as caught:
            await asyncio.wait_for(task, timeout=5)
        _assert_safe_error(
            caught.value, caplog, llm_errors.ProviderUnavailableError, retryable=True
        )
        _assert_one_refresh(requests)
        if change == "replace":
            assert path.read_bytes() == replacement_bytes
        else:
            assert not path.exists(), "Refresh resurrected a removed credential file"
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("symlink_alias", [False, True])
async def test_strict_commit_serializes_with_subprocess_save_tokens(
    monkeypatch, tmp_path, credential_process, symlink_alias
):
    path = tmp_path / "credentials.json"
    original_bytes = _write_tokens(path, _tokens())
    configured_path = tmp_path / "alias.json" if symlink_alias else path
    if symlink_alias:
        configured_path.symlink_to(path)
    provider = _provider(configured_path)
    original_write = oauth._write_tokens_atomic
    children = []
    committed = []

    def commit_with_competing_writer(record, destination):
        # Strict refresh has completed its final byte comparison, but has not
        # committed yet. Launch the other process inside that exact interval.
        assert destination == path
        assert not children, "Strict refresh attempted more than one commit"
        child = credential_process(configured_path, "serialize")
        children.append(child)
        assert _process_line(child) == "attempting"
        assert _process_line(child) == "lock-busy"
        # Unlike a probe alone, this proves the normal save_tokens call itself
        # attempted to acquire the held lock and could not enter its write.
        assert _process_line(child) == "save-blocked"
        assert path.read_bytes() == original_bytes
        original_write(record, destination)
        saved = json.loads(path.read_bytes())
        assert saved == record
        assert saved["account_id"] == _ACCOUNT_A
        assert saved["access_token"] == _FRESH_ACCESS
        assert saved["refresh_token"] == _FRESH_REFRESH
        committed.append(saved)
        # Returning lets strict refresh release the lock; only then can the
        # child's real save_tokens call enter its atomic writer.

    monkeypatch.setattr(oauth, "_write_tokens_atomic", commit_with_competing_writer)

    async def handle(request):
        if request.method == "POST":
            _assert_refresh(request)
            return httpx.Response(200, json=_refreshed())
        _assert_catalog(request)
        return _catalog_response()

    requests = _install_transport(monkeypatch, handle)
    _assert_result(await provider.discover_models())
    assert [request.method for request in requests] == ["POST", "GET"]
    assert len(children) == len(committed) == 1
    child = children[0]
    before_save = _process_line(child)
    assert before_save.startswith("before-save:")
    assert json.loads(before_save.removeprefix("before-save:")) == committed[0]
    assert _process_line(child) == "saved"
    assert child.wait(timeout=10) == 0
    expected = _tokens(_ACCOUNT_B, expired=False)
    final_bytes = path.read_bytes()
    assert json.loads(final_bytes) == expected
    assert final_bytes == json.dumps(expected).encode(), "Mixed A/B credential bytes"
    if symlink_alias:
        assert configured_path.is_symlink()
        assert configured_path.read_bytes() == final_bytes


@pytest.mark.asyncio
async def test_subprocess_save_during_refresh_post_is_preserved(
    monkeypatch, tmp_path, caplog, credential_process
):
    path = tmp_path / "credentials.json"
    _write_tokens(path, _tokens())
    provider = _provider(path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        _assert_refresh(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=_refreshed())

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    task = asyncio.create_task(provider.discover_models())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        child = credential_process(path, "save")
        assert _process_line(child) == "attempting"
        assert _process_line(child) == "saved"
        assert child.wait(timeout=10) == 0
        replacement_bytes = path.read_bytes()
        assert json.loads(replacement_bytes) == _tokens(_ACCOUNT_B, expired=False)
        assert not task.done(), "The replacement must precede the POST response"
        release.set()
        with pytest.raises(llm_errors.ProviderUnavailableError) as caught:
            await asyncio.wait_for(task, timeout=5)
        _assert_safe_error(
            caught.value, caplog, llm_errors.ProviderUnavailableError, retryable=True
        )
        _assert_one_refresh(requests)
        assert path.read_bytes() == replacement_bytes
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_subprocess_lock_contention_is_retryable_without_overwrite(
    monkeypatch, tmp_path, caplog, credential_process
):
    path = tmp_path / "credentials.json"
    original_bytes = _write_tokens(path, _tokens())
    provider = _provider(path)
    child = credential_process(path, "hold")
    assert _process_line(child) == "attempting"
    assert _process_line(child) == "locked"

    def forbidden_write(*args, **kwargs):
        raise AssertionError("Strict refresh must not write through a held lock")

    monkeypatch.setattr(oauth, "_write_tokens_atomic", forbidden_write)

    async def handle(request):
        _assert_refresh(request)
        return httpx.Response(200, json=_refreshed())

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(llm_errors.ProviderUnavailableError) as caught:
        await asyncio.wait_for(provider.discover_models(), timeout=5)
    _assert_safe_error(
        caught.value, caplog, llm_errors.ProviderUnavailableError, retryable=True
    )
    _assert_one_refresh(requests)
    assert path.read_bytes() == original_bytes
    # The child holds the lock until this explicit handshake, not for a sleep.
    child.stdin.write(b"release\n")
    child.stdin.flush()
    assert _process_line(child) == "released"
    assert child.wait(timeout=10) == 0
    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("with_id_token", [False, True], ids=["no-id", "same-id"])
@pytest.mark.parametrize(
    "rotate_refresh", [False, True], ids=["retain-refresh", "rotate"]
)
async def test_same_account_refresh_saves_credentials_before_catalog(
    monkeypatch, tmp_path, with_id_token, rotate_refresh
):
    path = tmp_path / "credentials.json"
    before_bytes = _write_tokens(path, _tokens())
    provider = _provider(path)
    started = datetime.now(tz=timezone.utc)

    async def handle(request):
        if request.method == "POST":
            _assert_refresh(request)
            return httpx.Response(
                200,
                json=_refreshed(
                    with_id_token=with_id_token, rotate_refresh=rotate_refresh
                ),
            )
        _assert_catalog(request)
        saved = json.loads(path.read_bytes())
        assert saved["access_token"] == _FRESH_ACCESS
        assert saved["account_id"] == _ACCOUNT_A
        return _catalog_response()

    requests = _install_transport(monkeypatch, handle)
    _assert_result(await provider.discover_models())
    finished = datetime.now(tz=timezone.utc)
    assert [request.method for request in requests] == ["POST", "GET"]
    assert path.read_bytes() != before_bytes
    saved = json.loads(path.read_bytes())
    assert saved["account_id"] == _ACCOUNT_A
    assert saved["access_token"] == _FRESH_ACCESS
    assert saved["refresh_token"] == (_FRESH_REFRESH if rotate_refresh else _REFRESH_A)
    assert saved["plan_type"] == "plus"
    expiry = datetime.fromisoformat(saved["expires_at"])
    assert started + timedelta(seconds=3600) <= expiry
    assert expiry <= finished + timedelta(seconds=3600)


@pytest.mark.asyncio
@pytest.mark.parametrize("injected", [False, True], ids=["file", "injected"])
async def test_refreshed_id_token_cannot_change_account(
    monkeypatch, tmp_path, caplog, injected
):
    path = tmp_path / "credentials.json"
    original_bytes = _write_tokens(path, _tokens())
    provider = _provider(path, injected=injected)

    async def handle(request):
        _assert_refresh(request)
        return httpx.Response(
            200, json={**_refreshed(), "id_token": _id_token(_ACCOUNT_B)}
        )

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(llm_errors.AuthenticationError) as caught:
        await provider.discover_models()
    _assert_safe_error(
        caught.value, caplog, llm_errors.AuthenticationError, retryable=False
    )
    _assert_one_refresh(requests)
    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,expected",
    [
        (httpx.ReadTimeout, llm_errors.LLMTimeoutError),
        (httpx.ConnectError, llm_errors.ProviderUnavailableError),
    ],
)
@pytest.mark.parametrize("injected", [False, True], ids=["file", "injected"])
async def test_refresh_transport_errors_are_typed_safe_and_leave_file_unchanged(
    monkeypatch, tmp_path, caplog, failure, expected, injected
):
    path = tmp_path / "credentials.json"
    disk = _tokens(_ACCOUNT_B, expired=False) if injected else _tokens()
    original_bytes = _write_tokens(path, disk)
    provider = _provider(path, injected=injected)

    async def handle(request):
        _assert_refresh(request)
        raise failure(_LEAK, request=request)

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(expected) as caught:
        await provider.discover_models()
    _assert_safe_error(caught.value, caplog, expected, retryable=True)
    _assert_one_refresh(requests)
    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code,expected,retryable,retry_after",
    [
        (429, "rate_limit_exceeded", llm_errors.RateLimitError, True, 7.0),
        (
            503, "temporarily_unavailable",
            llm_errors.ProviderUnavailableError, True, 7.0,
        ),
        (400, "invalid_grant", llm_errors.AuthenticationError, False, None),
        (400, {"code": "invalid_grant"}, llm_errors.AuthenticationError, False, None),
        (401, "invalid_token", llm_errors.AuthenticationError, False, None),
        (403, "access_denied", llm_errors.AccessDeniedError, False, None),
    ],
)
async def test_refresh_http_errors_preserve_type_status_retry_hint_not_secrets(
    monkeypatch, tmp_path, caplog, status, code, expected, retryable, retry_after
):
    path = tmp_path / "credentials.json"
    original_bytes = _write_tokens(path, _tokens())
    provider = _provider(path)

    async def handle(request):
        _assert_refresh(request)
        return httpx.Response(
            status,
            json={"error": code, "error_description": _LEAK},
            headers={"retry-after": "7"},
        )

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(expected) as caught:
        await provider.discover_models()
    _assert_safe_error(
        caught.value,
        caplog,
        expected,
        retryable=retryable,
        status=status,
        retry_after=retry_after,
    )
    _assert_one_refresh(requests)
    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "invalid-json",
        "not-object",
        "missing-access",
        "empty-access",
        "wrong-access-type",
        "newline-access",
        "unicode-access",
        "space-access",
        "empty-refresh",
        "wrong-refresh-type",
        "negative-expiry",
        "zero-expiry",
        "boolean-expiry",
        "text-expiry",
        "infinite-expiry",
        "overflow-expiry",
        "malformed-id",
        "wrong-id-type",
    ],
)
async def test_malformed_refresh_is_not_auth_success_and_does_not_mutate_file(
    monkeypatch, tmp_path, caplog, case
):
    path = tmp_path / "credentials.json"
    original_bytes = _write_tokens(path, _tokens())
    provider = _provider(path)
    payload = {**_refreshed(), "diagnostic": _LEAK}
    changes = {
        "empty-access": ("access_token", " "),
        "wrong-access-type": ("access_token", {"secret": _LEAK}),
        "newline-access": ("access_token", _FRESH_ACCESS + "\n"),
        "unicode-access": ("access_token", _FRESH_ACCESS + "\u00e9"),
        "space-access": ("access_token", _FRESH_ACCESS + " space"),
        "empty-refresh": ("refresh_token", ""),
        "wrong-refresh-type": ("refresh_token", [_LEAK]),
        "negative-expiry": ("expires_in", -1),
        "zero-expiry": ("expires_in", 0),
        "boolean-expiry": ("expires_in", True),
        "text-expiry": ("expires_in", _LEAK),
        "infinite-expiry": ("expires_in", float("inf")),
        "overflow-expiry": ("expires_in", 1e100),
        "malformed-id": ("id_token", _LEAK),
        "wrong-id-type": ("id_token", {"secret": _LEAK}),
    }
    if case == "invalid-json":
        content = ("not-json " + _LEAK).encode()
    else:
        if case == "not-object":
            payload = [_LEAK]
        elif case == "missing-access":
            del payload["access_token"]
        else:
            key, value = changes[case]
            payload[key] = value
        # Raw bytes allow intentionally non-finite JSON to reach our parser.
        content = json.dumps(payload).encode()

    async def handle(request):
        _assert_refresh(request)
        return httpx.Response(200, content=content)

    requests = _install_transport(monkeypatch, handle)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(llm_errors.LLMError) as caught:
        await provider.discover_models()
    _assert_safe_error(caught.value, caplog, llm_errors.LLMError, retryable=False)
    _assert_one_refresh(requests)
    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disk_exists", [False, True], ids=["absent-file", "other-account"]
)
@pytest.mark.parametrize("with_id_token", [False, True], ids=["no-id", "same-id"])
async def test_injected_expired_tokens_refresh_own_account_without_persisting(
    monkeypatch, tmp_path, disk_exists, with_id_token
):
    path = tmp_path / "credentials.json"
    if disk_exists:
        original_bytes = _write_tokens(
            path, {**_tokens(_ACCOUNT_B, expired=False), "auth_mode": "chatgpt_plan"}
        )
    provider = _provider(path, injected=True)

    async def handle(request):
        if request.method == "POST":
            _assert_refresh(request)
            return httpx.Response(200, json=_refreshed(with_id_token=with_id_token))
        _assert_catalog(request)
        return _catalog_response()

    requests = _install_transport(monkeypatch, handle)
    _assert_result(await provider.discover_models())
    assert [request.method for request in requests] == ["POST", "GET"]
    if disk_exists:
        assert path.read_bytes() == original_bytes
    else:
        assert not path.exists(), "Injected credentials must remain memory-only"


@pytest.mark.asyncio
@pytest.mark.parametrize("injected", [False, True], ids=["file", "injected"])
async def test_cancellation_during_refresh_leaves_credentials_and_skips_catalog(
    monkeypatch, tmp_path, injected
):
    path = tmp_path / "credentials.json"
    disk = _tokens(_ACCOUNT_B, expired=False) if injected else _tokens()
    original_bytes = _write_tokens(path, disk)
    provider = _provider(path, injected=injected)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        _assert_refresh(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=_refreshed())

    requests = _install_transport(monkeypatch, handle)
    task = asyncio.create_task(provider.discover_models())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert task.cancelled()
        _assert_one_refresh(requests)
        assert path.read_bytes() == original_bytes
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
