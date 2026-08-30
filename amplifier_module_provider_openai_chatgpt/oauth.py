"""OAuth constants and PKCE helper functions for OpenAI ChatGPT subscription authentication.

Implements RFC 7636 PKCE (Proof Key for Code Exchange) and defines all
OAuth/device flow endpoints for OpenAI authentication.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OAuth issuer and endpoints
# ---------------------------------------------------------------------------

OAUTH_ISSUER = "https://auth.openai.com"
OAUTH_TOKEN_URL = f"{OAUTH_ISSUER}/oauth/token"

# ---------------------------------------------------------------------------
# Client identity and scopes
# ---------------------------------------------------------------------------

OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OAUTH_SCOPES = "openid profile email offline_access"

# ---------------------------------------------------------------------------
# Device code callback
# ---------------------------------------------------------------------------

DEVICE_CODE_CALLBACK_URL = f"{OAUTH_ISSUER}/deviceauth/callback"

# ---------------------------------------------------------------------------
# Device code flow endpoints
# ---------------------------------------------------------------------------

DEVICE_CODE_USERCODE_URL = f"{OAUTH_ISSUER}/api/accounts/deviceauth/usercode"
DEVICE_CODE_TOKEN_URL = f"{OAUTH_ISSUER}/api/accounts/deviceauth/token"
DEVICE_CODE_VERIFICATION_URL = f"{OAUTH_ISSUER}/codex/device"
DEVICE_CODE_POLL_INTERVAL = 5  # seconds between polling attempts
DEVICE_CODE_POLL_TIMEOUT_S = 900  # overall wall-clock bound on the poll loop (15 min)

# ---------------------------------------------------------------------------
# ChatGPT Codex API
# ---------------------------------------------------------------------------

CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

TOKEN_FILE_PATH = "~/.amplifier/openai-chatgpt-oauth.json"

# ---------------------------------------------------------------------------
# Token storage helpers
# ---------------------------------------------------------------------------


def save_tokens(tokens: dict, path: str | None = None) -> None:
    """Write tokens as JSON to disk with 0600 permissions.

    Creates parent directories if they do not exist.
    Defaults to TOKEN_FILE_PATH with ~ expansion when path is None.

    Args:
        tokens: Dictionary of token data to persist.
        path: Destination file path. Defaults to TOKEN_FILE_PATH.
    """
    if path is None:
        path = os.path.expanduser(TOKEN_FILE_PATH)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokens, f)

    os.chmod(path, 0o600)
    logger.debug("Tokens saved to %s", path)


def load_tokens(path: str | None = None) -> dict | None:
    """Read tokens from a JSON file on disk.

    Returns the parsed dict on success.
    Returns None if the file is missing, empty, or contains malformed JSON.

    Args:
        path: Source file path. Defaults to TOKEN_FILE_PATH.

    Returns:
        Token dict on success, None otherwise.
    """
    if path is None:
        path = os.path.expanduser(TOKEN_FILE_PATH)

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return None
        return json.loads(content)
    except FileNotFoundError:
        logger.debug("Token file not found: %s", path)
        return None
    except json.JSONDecodeError:
        logger.warning("Malformed JSON in token file: %s", path)
        return None


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------


def is_token_valid(tokens: dict | None) -> bool:
    """Check whether a token dict contains a valid, unexpired access token.

    Returns True only if ``tokens`` contains a non-empty ``access_token`` and
    an ``expires_at`` timestamp that is strictly in the future.

    Args:
        tokens: Token dict (typically loaded via :func:`load_tokens`) or None.

    Returns:
        True if the token exists and has not expired, False otherwise.
    """
    if tokens is None:
        return False

    if not tokens.get("access_token"):
        return False

    expires_at = tokens.get("expires_at")
    if not expires_at:
        return False

    try:
        expiry = datetime.fromisoformat(expires_at)
    except (ValueError, TypeError):
        return False

    # Treat timezone-naive datetimes as UTC.
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    return expiry > datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


async def refresh_tokens(refresh_token: str, path: str | None = None) -> dict | None:
    """Exchange a refresh token for new credentials.

    POSTs to OAUTH_TOKEN_URL with the grant_type=refresh_token flow.
    On success, persists the new token dict to disk and returns it.
    On failure, logs a warning and returns None.

    Args:
        refresh_token: The refresh token to exchange.
        path: Destination file path for token storage. Defaults to TOKEN_FILE_PATH.

    Returns:
        Token dict on success, None on failure.
    """
    data = urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OAUTH_CLIENT_ID,
        }
    ).encode("utf-8")

    headers = {
        "User-Agent": "amplifier-openai-chatgpt-provider/1.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OAUTH_TOKEN_URL, content=data, headers=headers)
            resp.raise_for_status()
            token_data = resp.json()
    except httpx.HTTPStatusError as exc:
        # Server responded with an error status -- carry the status/body so
        # callers can distinguish "refresh token revoked" (e.g. 400
        # invalid_grant) from "network down" (no response at all).
        body = exc.response.text[:500]
        logger.warning(
            "Failed to refresh tokens (HTTP %s): %s", exc.response.status_code, body
        )
        return None
    except Exception as exc:
        logger.warning("Failed to refresh tokens (network error): %s", exc)
        return None

    # Compute expires_at from the expires_in field in the response.
    expires_in = token_data.get("expires_in", 3600)
    expires_at = (
        datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)
    ).isoformat()

    # Extract account_id from the new id_token if present.
    account_id = None
    if "id_token" in token_data:
        account_id = extract_account_id(token_data["id_token"]) or None

    # Extract plan_type from the new id_token if present.
    plan_type = extract_plan_type(token_data.get("id_token", ""))

    # Fall back: preserve account_id and plan_type from any existing tokens stored on disk.
    if not account_id or not plan_type:
        existing = load_tokens(path)
        if not account_id:
            account_id = existing.get("account_id") if existing else None
        if not plan_type:
            plan_type = (existing or {}).get("plan_type", "")

    result = {
        "auth_mode": "oauth",
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token", refresh_token),
        "id_token": token_data.get("id_token"),
        "account_id": account_id,
        "plan_type": plan_type,
        "expires_at": expires_at,
    }

    save_tokens(result, path)
    return result


# ---------------------------------------------------------------------------
# Authorization code exchange
# ---------------------------------------------------------------------------


async def exchange_code_for_tokens(
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    token_file_path: str | None = None,
) -> dict:
    """Exchange an authorization code for OAuth tokens.

    POSTs to OAUTH_TOKEN_URL with the grant_type=authorization_code flow.
    On success, persists the token dict to disk and returns it.
    Raises on failure (does not catch exceptions).

    Args:
        code: The authorization code received from the OAuth redirect.
        code_verifier: The PKCE code verifier that matches the original challenge.
        redirect_uri: The redirect URI used in the original authorization request.
        token_file_path: Destination file path for token storage. Defaults to TOKEN_FILE_PATH.

    Returns:
        Token dict with auth_mode, access_token, refresh_token, id_token, account_id,
        plan_type, and expires_at.

    Raises:
        Exception: Any error from the HTTP request or response parsing propagates up.
    """
    data = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": redirect_uri,
        }
    ).encode("utf-8")

    headers = {
        "User-Agent": "amplifier-openai-chatgpt-provider/1.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OAUTH_TOKEN_URL, content=data, headers=headers)
            resp.raise_for_status()
            token_data = resp.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        raise RuntimeError(
            f"Token exchange failed (HTTP {exc.response.status_code}): {body}"
        ) from exc

    id_token = token_data.get("id_token", "")
    account_id = extract_account_id(id_token)
    plan_type = extract_plan_type(id_token)

    expires_in = token_data.get("expires_in", 3600)
    expires_at = (
        datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)
    ).isoformat()

    result = {
        "auth_mode": "oauth",
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "id_token": id_token,
        "account_id": account_id,
        "plan_type": plan_type,
        "expires_at": expires_at,
    }

    save_tokens(result, token_file_path)
    return result


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _decode_jwt_payload(id_token: str) -> dict | None:
    """Decode the payload segment of a JWT without signature verification.

    Args:
        id_token: A JWT string in the form ``header.payload.signature``.

    Returns:
        Decoded payload dict, or None on any failure (empty input, wrong
        number of segments, base64 decode error, or JSON parse error).
    """
    if not id_token:
        return None
    try:
        parts = id_token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # Add base64url padding so that len is a multiple of 4.
        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None


def extract_account_id(id_token: str) -> str:
    """Decode a JWT id_token and extract the account ID.

    Decodes the JWT payload segment without signature verification (the token
    was just received over HTTPS from the issuer).  Looks for ``account_id``
    inside the ``https://api.openai.com/profile`` custom claim first; falls
    back to the standard ``sub`` claim.

    Args:
        id_token: A JWT string in the form ``header.payload.signature``.

    Returns:
        The account ID string, or an empty string on any failure (empty input,
        malformed JWT, missing claims, decode errors).
    """
    payload = _decode_jwt_payload(id_token)
    if payload is None:
        return ""
    profile = payload.get("https://api.openai.com/profile")
    if isinstance(profile, dict) and profile.get("account_id"):
        return str(profile["account_id"])
    return str(payload.get("sub", ""))


def extract_plan_type(id_token: str) -> str:
    """Decode a JWT id_token and extract the ChatGPT plan type.

    Decodes the JWT payload segment without signature verification (the token
    was just received over HTTPS from the issuer).  Looks for
    ``chatgpt_plan_type`` inside the ``https://api.openai.com/auth`` custom
    claim.

    Args:
        id_token: A JWT string in the form ``header.payload.signature``.

    Returns:
        The plan type string (e.g. ``"pro"``), or an empty string on any
        failure (empty input, malformed JWT, missing claim, decode errors).
    """
    payload = _decode_jwt_payload(id_token)
    if payload is None:
        return ""
    auth_claim = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        return str(auth_claim.get("chatgpt_plan_type", ""))
    return ""


# ---------------------------------------------------------------------------
# PKCE helpers (RFC 7636)
# ---------------------------------------------------------------------------


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE (code_verifier, code_challenge) pair per RFC 7636.

    The verifier is a URL-safe random string of 43–128 characters.
    The challenge is BASE64URL(SHA256(verifier)) with no padding.

    Returns:
        A (code_verifier, code_challenge) tuple, both as ASCII strings.
    """
    # secrets.token_urlsafe(32) produces 43 URL-safe characters (base64url of 32 bytes)
    code_verifier = secrets.token_urlsafe(32)

    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    return code_verifier, code_challenge


# ---------------------------------------------------------------------------
# Device code flow
# ---------------------------------------------------------------------------


async def start_device_code_flow(
    *,
    print_fn: Callable[[str], None] | None = None,
    overall_timeout_s: float = DEVICE_CODE_POLL_TIMEOUT_S,
) -> dict:
    """Perform device code authorization flow and return authorization credentials.

    Step 1: POSTs to DEVICE_CODE_USERCODE_URL with client_id and scope to
    obtain a user_code, device_code, and polling interval.

    Step 2: Prints the verification URL and user code to the terminal so the
    user can authorize the device in a browser.

    Step 3: Polls DEVICE_CODE_TOKEN_URL until the user authorizes or an error
    occurs.  Handles these poll responses:

    - ``authorization_pending``: continue polling after sleeping *interval* seconds.
    - ``slow_down``: increase *interval* by 5 and continue polling.
    - ``expired_token``: raise :exc:`RuntimeError`.
    - any other error: raise :exc:`RuntimeError`.
    - success (no ``error`` key): return immediately.

    Returns:
        A dict with keys:

        - ``authorization_code``: the code returned by the authorization server.
        - ``code_verifier``: a freshly generated PKCE code verifier string for
          use when exchanging the authorization code for tokens.

    Raises:
        RuntimeError: If the device code expires or the server returns an
            unhandled error.
    """
    # Generate a PKCE pair; code_verifier is returned to the caller for later
    # token exchange via exchange_code_for_tokens().
    code_verifier, _ = generate_pkce_pair()

    # Step 1: Request a device code and user code from the authorization server.
    data = json.dumps(
        {
            "client_id": OAUTH_CLIENT_ID,
            "scope": OAUTH_SCOPES,
        }
    ).encode("utf-8")

    headers = {
        "User-Agent": "amplifier-openai-chatgpt-provider/1.0",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            DEVICE_CODE_USERCODE_URL, content=data, headers=headers
        )
        resp.raise_for_status()
        device_data = resp.json()

    user_code: str = device_data["user_code"]
    device_code: str = device_data.get("device_code") or device_data.get(
        "device_auth_id", ""
    )
    interval: int = int(device_data.get("interval", DEVICE_CODE_POLL_INTERVAL))

    # Step 2: Prompt the user to authorize via their browser.
    # Route through print_fn when given (e.g. ChatGPTProvider.login() wires this
    # to the app-cli's own output channel); otherwise print to stderr so the
    # message is visible even when the CLI UI has captured stdout (mount path).
    import sys

    def _emit(line: str) -> None:
        if print_fn is not None:
            print_fn(line)
        else:
            print(line, file=sys.stderr, flush=True)

    _emit(f"\n\nOpen this URL on any device: {DEVICE_CODE_VERIFICATION_URL}")
    _emit(f"Enter code: {user_code}\n")
    logger.warning(
        "OpenAI OAuth: visit %s and enter code: %s",
        DEVICE_CODE_VERIFICATION_URL,
        user_code,
    )

    # Step 3: Poll until authorized or an error occurs.
    device_auth_id = device_data.get("device_auth_id", device_code)

    poll_headers = {
        "User-Agent": "amplifier-openai-chatgpt-provider/1.0",
        "Content-Type": "application/json",
    }

    deadline = time.monotonic() + overall_timeout_s
    first_poll = True
    while True:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Device code flow timed out after {overall_timeout_s:.0f} seconds "
                "waiting for authorization. Please try again."
            )

        if not first_poll:
            await asyncio.sleep(interval)
        first_poll = False

        poll_data = json.dumps(
            {
                "client_id": OAUTH_CLIENT_ID,
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            }
        ).encode("utf-8")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    DEVICE_CODE_TOKEN_URL, content=poll_data, headers=poll_headers
                )
                resp.raise_for_status()
                result = resp.json()
            # Success — return the authorization code and PKCE verifier.
            # The response may contain authorization_code (for token exchange)
            # or tokens directly (access_token, refresh_token).
            if "authorization_code" in result:
                # Use the server's code_verifier if provided (device code flow),
                # otherwise fall back to the locally generated one.
                return {
                    "authorization_code": result["authorization_code"],
                    "code_verifier": result.get("code_verifier", code_verifier),
                }
            else:
                # Tokens returned directly — skip the exchange step.
                return {"tokens_direct": True, **result}
        except httpx.HTTPStatusError as e:
            body = e.response.json()
            error_code = body.get("error", {}).get("code", "")

            if error_code in (
                "deviceauth_authorization_unknown",
                "authorization_pending",
                "deviceauth_authorization_pending",
            ):
                continue  # User hasn't authorized yet; sleep then retry.
            elif error_code in ("slow_down", "deviceauth_slow_down"):
                interval += 5
                continue
            elif error_code in ("expired_token", "deviceauth_expired"):
                raise RuntimeError("Device code expired. Please try again.")
            else:
                raise RuntimeError(f"Device code flow error: {error_code} - {body}")


# ---------------------------------------------------------------------------
# Login orchestration (device code only)
# ---------------------------------------------------------------------------


async def login(
    *,
    token_file_path: str | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> dict:
    """Authenticate using device code flow.

    Uses device code flow only — appropriate for all environments including SSH.

    Args:
        token_file_path: Destination file path for token storage.
            Defaults to TOKEN_FILE_PATH.
        print_fn: Optional callable used to surface the verification URL and
            user code (e.g. an app-cli output channel). Defaults to None,
            which prints to stderr — see :func:`start_device_code_flow`.

    Returns:
        Token dict from exchange_code_for_tokens().

    Raises:
        RuntimeError: If authentication fails.
    """
    # During mount(), only use device code flow. Browser flow is inappropriate
    # here — it opens a browser during session startup which blocks the UI and
    # can launch on a physical display the user isn't looking at (e.g. SSH into
    # a Pi with HDMI). Device code works everywhere: the user opens the URL on
    # any device they have handy.
    try:
        result = await start_device_code_flow(print_fn=print_fn)
    except Exception as exc:
        raise RuntimeError(f"Device code authentication failed: {exc}") from exc

    if result.get("tokens_direct"):
        # Device code flow returned tokens directly — compute expires_at
        # from expires_in so the token is never stored with an empty timestamp.
        expires_in = result.get("expires_in", 3600)
        expires_at = (
            datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)
        ).isoformat()
        tokens = {
            "auth_mode": "oauth",
            "access_token": result.get("access_token", ""),
            "refresh_token": result.get("refresh_token", ""),
            "id_token": result.get("id_token", ""),
            "account_id": extract_account_id(result.get("id_token", "")),
            "plan_type": extract_plan_type(result.get("id_token", "")),
            "expires_at": result.get("expires_at") or expires_at,
        }
        save_tokens(tokens, token_file_path)
        return tokens

    # Use the appropriate redirect_uri based on flow type.
    # Device code flow uses {issuer}/deviceauth/callback.
    flow_redirect = result.get("redirect_uri", DEVICE_CODE_CALLBACK_URL)
    return await exchange_code_for_tokens(
        code=result["authorization_code"],
        code_verifier=result["code_verifier"],
        redirect_uri=flow_redirect,
        token_file_path=token_file_path,
    )
