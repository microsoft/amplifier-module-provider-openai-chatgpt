"""Credential-free protocol tests for the documented ChatGPT plan flow."""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from amplifier_module_provider_openai_chatgpt import plan_auth as auth


def record(**overrides):
    return {
        "auth_mode": auth.MODE,
        "client_id": "oaiapp_test",
        "ext_agent_host_id": "urn:uuid:test",
        "issuer": auth.ISSUER,
        "subject": "user-1",
        "email": "same@example.com",
        "access_token": "access-secret",
        "refresh_token": "refresh-secret",
        "id_token": "id-secret",
        "token_type": "Bearer",
        "scopes": [auth.PLAN_SCOPE],
        "expires_at": time.time() + 3600,
        **overrides,
    }


def token_response(**overrides):
    return {
        "access_token": "new-access",
        "refresh_token": "rotated-refresh",
        "id_token": "new-id",
        "token_type": "Bearer",
        "scope": auth.PLAN_SCOPE,
        "expires_in": 3600,
        **overrides,
    }


async def callback_from_url(url, **overrides):
    query = parse_qs(urlsplit(url).query)
    params = {
        "state": query["state"][0],
        "code": "authorization-code",
        "client_id": "oaiapp_test",
        **overrides,
    }
    async with httpx.AsyncClient() as client:
        await client.get(query["redirect_uri"][0] + "?" + urlencode(params))
    return query


@pytest.mark.asyncio
async def test_registration_persisted_before_exchange_and_sensitive_fields_stay_local(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    urls, callbacks = [], []

    def notify(url):
        urls.append(url)
        callbacks.append(asyncio.create_task(callback_from_url(url)))

    async def exchange(data):
        assert (
            json.loads(path.with_name(path.name + ".registration").read_text())[
                "client_id"
            ]
            == "oaiapp_test"
        )
        assert not path.exists()
        assert data["client_id"] == "oaiapp_test"
        assert data["resource"] == auth.RESOURCE
        assert data["redirect_uri"].startswith("http://127.0.0.1:")
        assert data["redirect_uri"].endswith("/auth/callback")
        return token_response()

    verify = AsyncMock(return_value={"sub": "user-1", "email": "same@example.com"})
    monkeypatch.setattr(auth, "_token_request", exchange)
    monkeypatch.setattr(auth, "_verify_identity", verify)
    await auth.login(token_file_path=str(path), print_fn=notify)
    query = (await asyncio.gather(*callbacks))[0]
    assert query["client_id"] == [auth.DYNAMIC_CLIENT]
    assert query["agent_name_hint"] == [auth.DEFAULT_APP_NAME]
    assert query["code_challenge_method"] == ["S256"]
    assert set(query["scope"][0].split()) == set(auth.SCOPES.split())
    assert verify.call_args.args == ("new-id", "oaiapp_test", query["nonce"][0])
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert auth.auth_status(str(path))["plan_enabled"]
    assert not any(
        secret in urls[0] for secret in ("new-access", "new-id", "rotated-refresh")
    )


@pytest.mark.asyncio
async def test_failed_exchange_retains_issued_registration_for_next_attempt(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    callback_tasks = []
    urls = []

    def notify(url):
        urls.append(url)
        callback_tasks.append(asyncio.create_task(callback_from_url(url)))

    monkeypatch.setattr(
        auth,
        "_token_request",
        AsyncMock(side_effect=auth.PlanAuthError("expired", code="invalid_grant")),
    )
    for _ in range(2):
        with pytest.raises(auth.PlanAuthError):
            await auth.login(token_file_path=str(path), print_fn=notify)
    await asyncio.gather(*callback_tasks)
    assert parse_qs(urlsplit(urls[1]).query)["client_id"] == ["oaiapp_test"]
    assert "agent_name_hint" not in parse_qs(urlsplit(urls[1]).query)
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_values",
    [{"state": "wrong"}, {"error": "access_denied"}, {"client_id": "different"}],
)
async def test_bad_callback_never_replaces_active_profile(
    tmp_path, monkeypatch, callback_values
):
    path = tmp_path / "profile.json"
    auth._write(path, record())
    original = path.read_bytes()
    calls = []

    def notify(url):
        assert "id_token_hint" not in url
        calls.append(asyncio.create_task(callback_from_url(url, **callback_values)))

    exchange = AsyncMock()
    monkeypatch.setattr(auth, "_token_request", exchange)
    with pytest.raises(auth.PlanAuthError):
        await auth.login(token_file_path=str(path), print_fn=notify, timeout=0.1)
    await asyncio.gather(*calls)
    assert not exchange.called
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_identity_change_rejected_and_declined_plan_saved_as_disabled(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    auth._write(path, record())
    original = path.read_bytes()
    tasks = []

    def notify(url):
        tasks.append(asyncio.create_task(callback_from_url(url)))

    monkeypatch.setattr(
        auth,
        "_token_request",
        AsyncMock(return_value=token_response(scope="openid email")),
    )
    verify = AsyncMock(return_value={"sub": "different", "email": "same@example.com"})
    monkeypatch.setattr(auth, "_verify_identity", verify)
    with pytest.raises(auth.PlanAuthError, match="different ChatGPT account"):
        await auth.login(token_file_path=str(path), print_fn=notify)
    assert path.read_bytes() == original
    verify.return_value = {"sub": "user-1", "email": "same@example.com"}
    await auth.login(token_file_path=str(path), print_fn=notify)
    await asyncio.gather(*tasks)
    assert auth.auth_status(str(path))["authenticated"]
    assert not auth.auth_status(str(path))["plan_enabled"]
    with pytest.raises(auth.PlanAuthError, match="not enabled"):
        await auth.ensure_tokens(str(path))


@pytest.mark.asyncio
async def test_refresh_serialized_rotating_token_and_temporary_failure_preserved(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    auth._write(path, record(expires_at=0))

    async def exchange(data):
        await asyncio.sleep(0.03)
        assert data == {
            "grant_type": "refresh_token",
            "client_id": "oaiapp_test",
            "refresh_token": "refresh-secret",
            "resource": auth.RESOURCE,
        }
        return token_response(id_token=None)

    mocked = AsyncMock(side_effect=exchange)
    monkeypatch.setattr(auth, "_token_request", mocked)
    results = await asyncio.gather(*(auth.ensure_tokens(str(path)) for _ in range(3)))
    assert mocked.call_count == 1
    assert all(r["refresh_token"] == "rotated-refresh" for r in results)
    assert all(r["id_token"] == "id-secret" for r in results)
    auth._write(path, record(expires_at=0))
    before = path.read_bytes()
    mocked.side_effect = auth.PlanAuthError(
        "unavailable", code="temporarily_unavailable"
    )
    with pytest.raises(auth.PlanAuthError):
        await auth.ensure_tokens(str(path))
    assert path.read_bytes() == before
    mocked.side_effect = auth.PlanAuthError("revoked", code="invalid_grant")
    with pytest.raises(auth.PlanAuthError):
        await auth.ensure_tokens(str(path))
    saved = json.loads(path.read_text())
    assert saved["client_id"] == "oaiapp_test"
    assert saved["subject"] == "user-1"
    assert "access_token" not in saved and "refresh_token" not in saved


@pytest.mark.asyncio
async def test_jwt_signature_issuer_audience_expiry_and_nonce(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class Keys:
        def get_signing_key_from_jwt(self, token):
            class Key:
                pass

            result = Key()
            result.key = key.public_key()
            return result

    monkeypatch.setattr(auth, "_jwks_clients", {"https://auth.openai.com/jwks": Keys()})
    monkeypatch.setattr(
        auth,
        "_discovery",
        AsyncMock(
            return_value={
                "issuer": auth.ISSUER,
                "jwks_uri": "https://auth.openai.com/jwks",
            }
        ),
    )
    claims = {
        "iss": auth.ISSUER,
        "sub": "user-1",
        "aud": "oaiapp_test",
        "iat": time.time(),
        "exp": time.time() + 100,
        "nonce": "nonce",
    }
    good = jwt.encode(claims, key, algorithm="RS256")
    assert (await auth._verify_identity(good, "oaiapp_test", "nonce"))[
        "sub"
    ] == "user-1"
    for changed in (
        {"iss": "wrong"},
        {"aud": "wrong"},
        {"exp": time.time() - 100},
        {"nonce": "wrong"},
        {"sub": ""},
    ):
        with pytest.raises(auth.PlanAuthError):
            await auth._verify_identity(
                jwt.encode({**claims, **changed}, key, algorithm="RS256"),
                "oaiapp_test",
                "nonce",
            )
    with pytest.raises(auth.PlanAuthError):
        await auth._verify_identity(
            jwt.encode(claims, other, algorithm="RS256"), "oaiapp_test", "nonce"
        )


@pytest.mark.asyncio
async def test_import_preserves_remote_host_and_keeps_profiles_separate(
    tmp_path, monkeypatch
):
    source, dest = tmp_path / "source.json", tmp_path / "vm/profile.json"
    auth._write(source, record())
    host_file = tmp_path / "vm/host.json"
    host = await auth.host_id(str(host_file))
    monkeypatch.setattr(
        auth, "_verify_identity", AsyncMock(return_value={"sub": "user-1"})
    )
    status = await auth.import_credentials(
        str(source), token_file_path=str(dest), host_file_path=str(host_file)
    )
    assert status["plan_enabled"]
    assert json.loads(dest.read_text())["ext_agent_host_id"] == host
    auth._write(source, record(client_id="other-workspace"))
    with pytest.raises(auth.PlanAuthError):
        await auth.import_credentials(str(source), token_file_path=str(dest))


@pytest.mark.asyncio
async def test_logout_revokes_then_clears_tokens_retaining_identity(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    auth._write(path, record())
    monkeypatch.setattr(
        auth,
        "_discovery",
        AsyncMock(
            return_value={
                "issuer": auth.ISSUER,
                "revocation_endpoint": "https://auth.openai.com/revoke",
            }
        ),
    )
    seen = []

    def handle(request):
        seen.append(request)
        assert (
            b"refresh-secret" in request.content and b"oaiapp_test" in request.content
        )
        return httpx.Response(200)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    assert (await auth.logout(str(path)))["revocation_confirmed"]
    saved = json.loads(path.read_text())
    assert saved["client_id"] == "oaiapp_test"
    assert all(k not in saved for k in ("access_token", "refresh_token", "id_token"))
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_cross_process_refresh_uses_only_one_rotating_token(tmp_path):
    """Separate workers must reread disk after acquiring the profile lock."""
    import sys

    path, calls = tmp_path / "profile.json", tmp_path / "refreshes"
    auth._write(path, record(expires_at=0))
    script = """
import asyncio, pathlib, sys
from amplifier_module_provider_openai_chatgpt import plan_auth as auth
async def exchange(data):
    with open(sys.argv[2], "a") as handle:
        handle.write("refresh\\n")
    await asyncio.sleep(.2)
    return {"access_token":"new", "refresh_token":"rotated", "expires_in":3600, "token_type":"Bearer"}
auth._token_request = exchange
asyncio.run(auth.ensure_tokens(sys.argv[1]))
"""
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, str(path), str(calls)
        )
        for _ in range(3)
    ]
    assert await asyncio.gather(*(p.wait() for p in processes)) == [0, 0, 0]
    assert calls.read_text() == "refresh\n"
    assert json.loads(path.read_text())["refresh_token"] == "rotated"


@pytest.mark.asyncio
async def test_unknown_revocation_is_reported_after_local_tokens_cleared(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    auth._write(path, record())
    monkeypatch.setattr(
        auth, "_discovery", AsyncMock(side_effect=auth.PlanAuthError("unavailable"))
    )
    monkeypatch.setattr(auth.asyncio, "sleep", AsyncMock())
    assert await auth.logout(str(path)) == {"revocation_confirmed": False}
    assert not auth.auth_status(str(path))["authenticated"]
    assert json.loads(path.read_text())["subject"] == "user-1"


@pytest.mark.asyncio
async def test_missing_plan_permission_never_refreshes(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    auth._write(path, record(scopes=["openid"], expires_at=0))
    exchange = AsyncMock()
    monkeypatch.setattr(auth, "_token_request", exchange)
    with pytest.raises(auth.PlanAuthError, match="not enabled"):
        await auth.ensure_tokens(str(path))
    assert not exchange.called


@pytest.mark.asyncio
async def test_consent_does_not_block_active_inference_and_signout_wins(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    auth._write(path, record())
    url_ready = asyncio.Future()
    monkeypatch.setattr(
        auth, "_token_request", AsyncMock(return_value=token_response())
    )
    monkeypatch.setattr(
        auth, "_verify_identity", AsyncMock(return_value={"sub": "user-1"})
    )
    pending = asyncio.create_task(
        auth.login(token_file_path=str(path), print_fn=url_ready.set_result)
    )
    url = await url_ready
    # This used to wait for the whole consent timeout behind the profile lock.
    assert (await asyncio.wait_for(auth.ensure_tokens(str(path)), 0.5))[
        "access_token"
    ] == "access-secret"
    async with auth._Locked(path):
        auth._write(path, auth._without_tokens(record()))
    await callback_from_url(url)
    with pytest.raises(auth.PlanAuthError) as error:
        await pending
    assert error.value.code == "profile_changed"
    assert not auth.auth_status(str(path))["authenticated"]


@pytest.mark.asyncio
async def test_staged_login_preserves_source_and_reuses_registration_across_candidates(
    tmp_path, monkeypatch
):
    source = tmp_path / "active.json"
    candidate = tmp_path / "candidate.json"
    registration = tmp_path / "connection.registration"
    auth._write(source, record())
    before = source.read_bytes()
    tasks, urls = [], []

    def notify(url):
        urls.append(url)
        tasks.append(asyncio.create_task(callback_from_url(url)))

    monkeypatch.setattr(
        auth, "_token_request", AsyncMock(return_value=token_response())
    )
    monkeypatch.setattr(
        auth, "_verify_identity", AsyncMock(return_value={"sub": "user-1"})
    )
    await auth.login(
        token_file_path=str(candidate),
        source_token_file_path=str(source),
        registration_file_path=str(registration),
        print_fn=notify,
    )
    assert source.read_bytes() == before
    assert json.loads(candidate.read_text())["access_token"] == "new-access"
    assert json.loads(registration.read_text())["client_id"] == "oaiapp_test"
    assert "access_token" not in json.loads(registration.read_text())
    # A new connection's failed attempt retains its issued ID independently of
    # the disposable candidate. Reusing it doesn't register a duplicate app.
    candidate2 = tmp_path / "another-candidate.json"
    await auth.login(
        token_file_path=str(candidate2),
        registration_file_path=str(registration),
        print_fn=notify,
    )
    await asyncio.gather(*tasks)
    assert parse_qs(urlsplit(urls[-1]).query)["client_id"] == ["oaiapp_test"]
    assert source.read_bytes() == before


@pytest.mark.asyncio
async def test_cancelled_staged_login_leaves_cleanup_to_host_without_revocation(
    tmp_path, monkeypatch
):
    source = tmp_path / "active.json"
    candidate = tmp_path / "candidate.json"
    registration = tmp_path / "connection.registration"
    host = tmp_path / "host.json"
    auth._write(source, record())
    auth._write(
        registration,
        {
            "auth_mode": auth.MODE,
            "client_id": "oaiapp_test",
            "ext_agent_host_id": "urn:uuid:test",
        },
    )
    before = source.read_bytes(), registration.read_bytes()
    ready = asyncio.Event()
    exchange = AsyncMock(side_effect=AssertionError("No exchange before consent"))
    revoke = AsyncMock(side_effect=AssertionError("Never revoke copied credentials"))
    monkeypatch.setattr(auth, "_token_request", exchange)
    monkeypatch.setattr(auth, "logout", revoke)
    task = asyncio.create_task(
        auth.login(
            token_file_path=str(candidate),
            source_token_file_path=str(source),
            registration_file_path=str(registration),
            host_file_path=str(host),
            print_fn=lambda _: ready.set(),
        )
    )
    await asyncio.wait_for(ready.wait(), 2)
    assert candidate.read_bytes() == before[0]
    assert os.stat(candidate).st_mode & 0o777 == 0o600
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    exchange.assert_not_awaited()
    revoke.assert_not_awaited()
    # The host can now remove this inactive candidate. Identity records and
    # the source's existing renewable session remain intact.
    candidate.unlink()
    assert source.read_bytes() == before[0]
    assert registration.read_bytes() == before[1]
    assert host.exists()
