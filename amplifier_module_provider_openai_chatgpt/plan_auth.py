"""Sign in with ChatGPT for the public Responses API.

Each path is one account/workspace registration. This is deliberately separate
from the legacy Codex OAuth/device flow in oauth.py. No tokens cross modes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
from filelock import FileLock

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
MODE = "chatgpt_plan"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
DYNAMIC_CLIENT = "dynamic_agent_client"
DEFAULT_TOKEN_FILE = "~/.amplifier/chatgpt-plan/default.json"
TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)
_jwks_clients: dict[str, jwt.PyJWKClient] = {}


class PlanAuthError(RuntimeError):
    """Safe, bounded auth error: never includes credentials or response bodies."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "authentication_failed",
        status: int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().absolute()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as exc:
        raise PlanAuthError(
            "The ChatGPT credential record could not be read.", code="invalid_record"
        ) from exc
    if not isinstance(value, dict):
        raise PlanAuthError(
            "The ChatGPT credential record is invalid.", code="invalid_record"
        )
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise PlanAuthError(
            "A credential destination cannot be a symbolic link.", code="invalid_record"
        )
    fd, temporary = tempfile.mkstemp(prefix=".chatgpt-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _Locked:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = FileLock(str(path) + ".lock", timeout=240, mode=0o600)

    async def __aenter__(self):
        # Nonblocking polling also makes cancellation safe: no abandoned thread
        # can acquire a lock after the coroutine has been cancelled.
        from filelock import Timeout

        deadline = time.monotonic() + 240
        while True:
            try:
                self.lock.acquire(timeout=0)
                return self
            except Timeout:
                if time.monotonic() >= deadline:
                    raise PlanAuthError(
                        "ChatGPT credentials are busy. Try again shortly.",
                        code="profile_busy",
                    )
                await asyncio.sleep(0.05)

    async def __aexit__(self, *_):
        self.lock.release()


async def host_id(host_file_path: str) -> str:
    path = _path(host_file_path)
    async with _Locked(path):
        record = _read(path)
        if record.get("ext_agent_host_id"):
            return str(record["ext_agent_host_id"])
        value = "urn:uuid:" + str(uuid.uuid4())
        _write(path, {"ext_agent_host_id": value})
        return value


def _host_path(token_file: Path, host_file_path: str | None) -> str:
    return host_file_path or str(token_file.parent / "host.json")


def _assert_mode(record: dict[str, Any]) -> None:
    if record and record.get("auth_mode") != MODE:
        raise PlanAuthError(
            "This connection requires a ChatGPT plan credential record. Legacy Codex credentials cannot be used with the public API.",
            code="auth_mode_mismatch",
        )


def auth_status(token_file_path: str) -> dict[str, Any]:
    record = _read(_path(token_file_path))
    _assert_mode(record)
    connected = bool(
        record.get("access_token") and record.get("subject") and record.get("client_id")
    )
    return {
        "auth_mode": MODE,
        "authenticated": connected,
        "plan_enabled": connected and PLAN_SCOPE in record.get("scopes", []),
        "email": record.get("email"),
        "client_id": record.get("client_id"),
        "subject": record.get("subject"),
    }


async def _discovery() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(ISSUER + "/.well-known/openid-configuration")
            if response.status_code != 200:
                raise PlanAuthError(
                    "OpenAI identity discovery is unavailable. Try again later.",
                    code="temporarily_unavailable",
                    status=response.status_code,
                )
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PlanAuthError(
            "OpenAI identity discovery is unavailable. Try again later.",
            code="temporarily_unavailable",
        ) from exc
    if not isinstance(data, dict) or data.get("issuer") != ISSUER:
        raise PlanAuthError(
            "OpenAI identity discovery returned an unexpected issuer.",
            code="invalid_issuer",
        )
    return data


def _trusted_endpoint(value: Any) -> str:
    if not isinstance(value, str):
        raise PlanAuthError(
            "OpenAI identity discovery is missing an endpoint.",
            code="invalid_discovery",
        )
    url = urlsplit(value)
    if (
        url.scheme != "https"
        or url.netloc != "auth.openai.com"
        or url.fragment
        or url.username
    ):
        raise PlanAuthError(
            "OpenAI identity discovery returned an untrusted endpoint.",
            code="invalid_discovery",
        )
    return value


async def _verify_identity(
    id_token: str, client_id: str, nonce: str | None, *, allow_expired: bool = False
) -> dict[str, Any]:
    if not isinstance(id_token, str) or not id_token:
        raise PlanAuthError(
            "OpenAI did not return an ID token.", code="invalid_identity"
        )
    try:
        discovery = await _discovery()
        jwks_uri = _trusted_endpoint(discovery.get("jwks_uri"))
        jwks = _jwks_clients.setdefault(
            jwks_uri, jwt.PyJWKClient(jwks_uri, lifespan=300, timeout=20)
        )
        key = await asyncio.to_thread(jwks.get_signing_key_from_jwt, id_token)
        claims = jwt.decode(
            id_token,
            key.key,
            algorithms=["RS256", "ES256"],
            audience=client_id,
            issuer=ISSUER,
            leeway=5,
            options={
                "require": ["sub", "exp", "iat", "iss", "aud"],
                "verify_exp": not allow_expired,
            },
        )
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise ValueError("Missing subject")
        if nonce is not None and not secrets.compare_digest(
            str(claims.get("nonce", "")), nonce
        ):
            raise ValueError("Nonce mismatch")
        multiple_audiences = (
            isinstance(claims.get("aud"), list) and len(claims["aud"]) > 1
        )
        if (multiple_audiences or claims.get("azp") is not None) and claims.get(
            "azp"
        ) != client_id:
            raise ValueError("Authorized party mismatch")
        return claims
    except PlanAuthError:
        raise
    except jwt.PyJWKClientConnectionError as exc:
        raise PlanAuthError(
            "OpenAI identity verification is temporarily unavailable. Try again later.",
            code="temporarily_unavailable",
        ) from exc
    except Exception as exc:
        raise PlanAuthError(
            "The ChatGPT identity could not be verified. Sign in again.",
            code="invalid_identity",
        ) from exc


async def _token_request(data: dict[str, str]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            response = await client.post(TOKEN_URL, data=data)
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PlanAuthError(
            "OpenAI sign-in is temporarily unavailable. Your saved connection was preserved.",
            code="temporarily_unavailable",
        ) from exc
    if response.status_code != 200:
        if response.status_code >= 500 or response.status_code == 429:
            raise PlanAuthError(
                "OpenAI sign-in is temporarily unavailable. Your saved connection was preserved.",
                code="temporarily_unavailable",
                status=response.status_code,
            )
        code = body.get("error") if isinstance(body, dict) else None
        if not isinstance(code, str) or code not in TERMINAL_REFRESH_ERRORS | {
            "invalid_client",
            "access_denied",
            "invalid_request",
            "temporarily_unavailable",
        }:
            code = "token_exchange_failed"
        raise PlanAuthError(
            f"ChatGPT authorization failed ({code}). Try signing in again.",
            code=code,
            status=response.status_code,
        )
    if not isinstance(body, dict):
        raise PlanAuthError(
            "OpenAI returned an invalid credential response.", code="invalid_response"
        )
    return body


def _credentials(
    data: dict[str, Any],
    registration: dict[str, Any],
    claims: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = previous or {}
    token = data.get("access_token")
    lifetime = data.get("expires_in")
    if (
        not isinstance(token, str)
        or not token
        or isinstance(lifetime, bool)
        or not isinstance(lifetime, (int, float))
        or not 0 < lifetime <= 365 * 86400
    ):
        raise PlanAuthError(
            "OpenAI returned incomplete credentials.", code="invalid_response"
        )
    if str(data.get("token_type", "")).lower() != "bearer":
        raise PlanAuthError(
            "OpenAI returned an unsupported token type.", code="invalid_response"
        )
    scope = data.get("scope")
    scopes = scope.split() if isinstance(scope, str) else previous.get("scopes", [])
    return {
        "auth_mode": MODE,
        **registration,
        "issuer": ISSUER,
        "subject": claims["sub"],
        "email": claims.get("email"),
        "access_token": token,
        "refresh_token": data.get("refresh_token") or previous.get("refresh_token"),
        "id_token": data.get("id_token") or previous.get("id_token"),
        "token_type": "Bearer",
        "scopes": scopes,
        "expires_at": time.time() + lifetime,
        "saved_at": time.time(),
    }


async def login(
    *,
    token_file_path: str = DEFAULT_TOKEN_FILE,
    host_file_path: str | None = None,
    app_name: str = "Amplifier Unified",
    print_fn: Callable[[str], None] | None = None,
    timeout: float = 180,
    open_browser: bool = False,
    request_plan_permission: bool = False,
    source_token_file_path: str | None = None,
    registration_file_path: str | None = None,
) -> dict[str, Any]:
    """Loopback OAuth. print_fn receives a URL, never a token or ID-token hint.

    Apps running remotely should use the local CLI and protected import instead.
    A cancelled/failed attempt leaves the active credential record untouched.
    """
    path = _path(token_file_path)
    source_path = _path(source_token_file_path) if source_token_file_path else None
    pending_path = (
        _path(registration_file_path)
        if registration_file_path
        else path.with_name(path.name + ".registration")
    )
    if source_path == path or pending_path == path or pending_path == source_path:
        raise PlanAuthError(
            "Sign-in candidate, source, and registration paths must be separate.",
            code="invalid_record",
        )
    host = await host_id(_host_path(path, host_file_path))
    # Serialize consent attempts separately. Active inference/refresh must not
    # wait on a person spending several minutes in the system browser.
    async with _Locked(pending_path.with_name(pending_path.name + ".login")):
        if source_path:
            # Hosts can use a fresh candidate profile and switch their config
            # only after their own edit/cancel compare-and-swap succeeds.
            # Credentials stay exclusively inside this provider-owned helper.
            async with _Locked(source_path):
                source = _read(source_path)
                _assert_mode(source)
                async with _Locked(path):
                    if path.exists():
                        raise PlanAuthError(
                            "The sign-in candidate already exists. Choose a fresh candidate path.",
                            code="invalid_record",
                        )
                    if source:
                        _write(path, source)
        async with _Locked(path):
            current = _read(path)
            _assert_mode(current)
        registration = current if current.get("client_id") else _read(pending_path)
        client_id = registration.get("client_id") or DYNAMIC_CLIENT
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        result: asyncio.Future[dict[str, str]] = (
            asyncio.get_running_loop().create_future()
        )

        async def callback(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            status, message = (
                "400 Bad Request",
                "This sign-in response was not accepted. Return to the app.",
            )
            try:
                header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
                line = header.split(b"\r\n", 1)[0].decode("ascii")
                method, target, _ = line.split(" ", 2)
                parsed = urlsplit(target)
                params = parse_qs(parsed.query)
                if (
                    method == "GET"
                    and parsed.path == "/auth/callback"
                    and all(len(v) == 1 for v in params.values())
                    and secrets.compare_digest(params.get("state", [""])[0], state)
                ):
                    if not result.done():
                        result.set_result({k: v[0] for k, v in params.items()})
                    status, message = (
                        "200 OK",
                        "Sign-in response received. Return to the app to confirm the connection.",
                    )
                writer.write(
                    f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{message}".encode()
                )
                await writer.drain()
            except (
                TimeoutError,
                ValueError,
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                ConnectionError,
            ):
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(callback, "127.0.0.1", 0, limit=16384)
        try:
            port = server.sockets[0].getsockname()[1]
            redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
            params = {
                "client_id": client_id,
                "ext_agent_host_id": host,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
            if client_id == DYNAMIC_CLIENT:
                params["agent_name_hint"] = app_name
            # No id_token_hint: authorization URLs are surfaced through host UIs.
            # Returning sign-ins use the supported account selector instead.
            if request_plan_permission:
                params["prompt"] = "consent"
            url = AUTHORIZE_URL + "?" + urlencode(params)
            (print_fn or (lambda value: print(value, file=sys.stderr, flush=True)))(url)
            if open_browser:
                await asyncio.to_thread(webbrowser.open, url)
            response = await asyncio.wait_for(result, timeout)
        except TimeoutError as exc:
            raise PlanAuthError(
                "ChatGPT sign-in timed out. Your saved connection was preserved.",
                code="login_timeout",
            ) from exc
        finally:
            server.close()
            await server.wait_closed()
        if response.get("error"):
            raise PlanAuthError(
                "ChatGPT sign-in was not completed. Your saved connection was preserved.",
                code="access_denied",
            )
        issued_id = response.get("client_id", client_id)
        if (
            issued_id == DYNAMIC_CLIENT
            or not isinstance(issued_id, str)
            or not issued_id
            or (client_id != DYNAMIC_CLIENT and issued_id != client_id)
        ):
            raise PlanAuthError(
                "The ChatGPT registration did not match this connection.",
                code="invalid_client",
            )
        if not response.get("code"):
            raise PlanAuthError(
                "ChatGPT sign-in did not return an authorization code.",
                code="invalid_response",
            )
        # Persist issued registration BEFORE exchange, without replacing active tokens.
        saved_registration = {
            "auth_mode": MODE,
            "client_id": issued_id,
            "ext_agent_host_id": host,
        }
        _write(pending_path, saved_registration)
        data = await _token_request(
            {
                "grant_type": "authorization_code",
                "client_id": issued_id,
                "code": response["code"],
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            }
        )
        claims = await _verify_identity(data.get("id_token"), issued_id, nonce)
        if current.get("subject") and claims["sub"] != current["subject"]:
            raise PlanAuthError(
                "This sign-in belongs to a different ChatGPT account. Add a separate connection instead.",
                code="identity_mismatch",
            )
        tokens = _credentials(data, saved_registration, claims)
        async with _Locked(path):
            latest = _read(path)
            # Refreshes of the same renewable session may finish while consent
            # is open. A new authorization supersedes that session atomically.
            # Sign-out or a different imported profile must never be undone.
            identity_keys = ("auth_mode", "client_id", "subject")
            changed_identity = any(
                latest.get(k) != current.get(k) for k in identity_keys
            )
            signed_out = bool(
                current.get("access_token") and not latest.get("access_token")
            )
            if changed_identity or signed_out:
                raise PlanAuthError(
                    "This ChatGPT connection changed during sign-in. Start a new sign-in from its current state.",
                    code="profile_changed",
                )
            _write(path, tokens)
        if not registration_file_path:
            pending_path.unlink(missing_ok=True)
        return tokens


def _without_tokens(record: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in record.items()
        if k
        not in {
            "access_token",
            "refresh_token",
            "id_token",
            "expires_at",
            "saved_at",
            "scopes",
        }
    }


async def ensure_tokens(token_file_path: str) -> dict[str, Any]:
    """Read latest disk record under one cross-process lock; rotate atomically."""
    path = _path(token_file_path)
    async with _Locked(path):
        record = _read(path)
        _assert_mode(record)
        if (
            not record.get("client_id")
            or record.get("client_id") == DYNAMIC_CLIENT
            or not record.get("subject")
            or not record.get("access_token")
        ):
            raise PlanAuthError(
                "Continue with ChatGPT to connect this account.",
                code="sign_in_required",
            )
        if PLAN_SCOPE not in record.get("scopes", []):
            raise PlanAuthError(
                "ChatGPT plan usage is not enabled. Enable it in this connection's settings.",
                code="plan_permission_required",
            )
        if (
            isinstance(record.get("expires_at"), (int, float))
            and record["expires_at"] > time.time() + 60
        ):
            return record
        if not record.get("refresh_token"):
            raise PlanAuthError(
                "This ChatGPT session expired. Sign in again.", code="sign_in_required"
            )
        try:
            data = await _token_request(
                {
                    "grant_type": "refresh_token",
                    "client_id": record["client_id"],
                    "refresh_token": record["refresh_token"],
                    "resource": RESOURCE,
                }
            )
        except PlanAuthError as exc:
            if exc.code in TERMINAL_REFRESH_ERRORS:
                _write(path, _without_tokens(record))
            raise
        claims = {"sub": record["subject"], "email": record.get("email")}
        if data.get("id_token"):
            claims = await _verify_identity(data["id_token"], record["client_id"], None)
            if claims["sub"] != record["subject"]:
                raise PlanAuthError(
                    "The refreshed identity does not match this ChatGPT connection.",
                    code="identity_mismatch",
                )
        updated = _credentials(
            data,
            {k: record[k] for k in ("client_id", "ext_agent_host_id")},
            claims,
            record,
        )
        _write(path, updated)
        if PLAN_SCOPE not in updated["scopes"]:
            raise PlanAuthError(
                "ChatGPT plan usage is no longer enabled for this connection.",
                code="plan_permission_required",
            )
        return updated


async def logout(token_file_path: str) -> dict[str, bool]:
    path = _path(token_file_path)
    async with _Locked(path):
        record = _read(path)
        _assert_mode(record)
        if not record:
            return {"revocation_confirmed": True}
        confirmed = not bool(record.get("refresh_token"))
        if record.get("refresh_token"):
            for attempt in range(3):
                try:
                    endpoint = _trusted_endpoint(
                        (await _discovery()).get("revocation_endpoint")
                    )
                    async with httpx.AsyncClient(
                        timeout=15, follow_redirects=False
                    ) as client:
                        response = await client.post(
                            endpoint,
                            data={
                                "token": record["refresh_token"],
                                "token_type_hint": "refresh_token",
                                "client_id": record["client_id"],
                            },
                        )
                    if response.status_code == 200:
                        confirmed = True
                        break
                    if response.status_code < 500:
                        break
                except (httpx.HTTPError, PlanAuthError, ValueError):
                    pass
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
        _write(path, _without_tokens(record))
        return {"revocation_confirmed": confirmed}


async def import_credentials(
    source_file: str, *, token_file_path: str, host_file_path: str | None = None
) -> dict[str, Any]:
    """Import an SSH-transferred record, preserving the destination host identity."""
    path = _path(token_file_path)
    record = _read(_path(source_file))
    _assert_mode(record)
    claims = await _verify_identity(
        record.get("id_token"), record.get("client_id"), None, allow_expired=True
    )
    if claims["sub"] != record.get("subject"):
        raise PlanAuthError(
            "Transferred ChatGPT identity does not match its registration.",
            code="identity_mismatch",
        )
    host = await host_id(_host_path(path, host_file_path))
    async with _Locked(path):
        existing = _read(path)
        if existing and (existing.get("subject"), existing.get("client_id")) != (
            record.get("subject"),
            record.get("client_id"),
        ):
            raise PlanAuthError(
                "Choose a separate destination for this ChatGPT registration.",
                code="identity_mismatch",
            )
        _write(path, {**record, "ext_agent_host_id": host})
    return auth_status(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChatGPT plan sign-in (public Responses API)"
    )
    parser.add_argument(
        "action", choices=["login", "status", "logout", "import", "host-id"]
    )
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--host-file")
    parser.add_argument("--app-name", default="Amplifier Unified")
    parser.add_argument("--source-file")
    parser.add_argument("--enable-plan", action="store_true")
    args = parser.parse_args()

    async def run():
        if args.action == "login":
            await login(
                token_file_path=args.token_file,
                host_file_path=args.host_file,
                app_name=args.app_name,
                open_browser=True,
                request_plan_permission=args.enable_plan,
            )
            return auth_status(args.token_file)
        if args.action == "status":
            return auth_status(args.token_file)
        if args.action == "logout":
            return await logout(args.token_file)
        if args.action == "host-id":
            return {
                "ext_agent_host_id": await host_id(
                    _host_path(_path(args.token_file), args.host_file)
                )
            }
        if not args.source_file:
            parser.error("--source-file is required for import")
        return await import_credentials(
            args.source_file,
            token_file_path=args.token_file,
            host_file_path=args.host_file,
        )

    try:
        print(json.dumps(asyncio.run(run())))
    except PlanAuthError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
