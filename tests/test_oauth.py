"""Tests for oauth.py constants and structure."""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_jwt(payload_dict: dict) -> str:
    """Create an unsigned JWT string (header.payload.signature) for testing.

    The signature part is left empty — this mirrors how unsigned tokens are
    structured and is sufficient for decode-only tests.
    """
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        .rstrip(b"=")
        .decode()
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps(payload_dict).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}."


def _make_httpx_mock(*post_responses):
    """Return a mock for patching httpx.AsyncClient with sequenced POST responses.

    Each positional argument is either:
    - dict  → successful response; resp.json() returns that dict, raise_for_status() is a no-op
    - Exception → raised when resp.raise_for_status() is called

    All calls to httpx.AsyncClient() share the same mock client instance,
    and each successive call to client.post() consumes the next response spec.
    """
    resp_mocks = []
    for spec in post_responses:
        mock_resp = MagicMock()
        if isinstance(spec, Exception):
            mock_resp.raise_for_status.side_effect = spec
        else:
            mock_resp.raise_for_status = MagicMock()  # no-op for success
            mock_resp.json.return_value = spec
        resp_mocks.append(mock_resp)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=resp_mocks)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=mock_client)


def _make_httpx_status_error(
    body_dict: dict, status_code: int = 403
) -> httpx.HTTPStatusError:
    """Create an httpx.HTTPStatusError with a JSON body, as device code poll errors arrive."""
    from amplifier_module_provider_openai_chatgpt.oauth import DEVICE_CODE_TOKEN_URL

    request = httpx.Request("POST", DEVICE_CODE_TOKEN_URL)
    response = httpx.Response(status_code=status_code, json=body_dict, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


class TestConstants:
    """Verify all OAuth constant values."""

    def test_oauth_issuer(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_ISSUER

        assert OAUTH_ISSUER == "https://auth.openai.com"

    def test_oauth_token_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_TOKEN_URL

        assert OAUTH_TOKEN_URL == "https://auth.openai.com/oauth/token"

    def test_oauth_client_id(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_CLIENT_ID

        assert OAUTH_CLIENT_ID == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_oauth_scopes(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_SCOPES

        assert OAUTH_SCOPES == "openid profile email offline_access"

    def test_device_code_usercode_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_USERCODE_URL,
        )

        assert (
            DEVICE_CODE_USERCODE_URL
            == "https://auth.openai.com/api/accounts/deviceauth/usercode"
        )

    def test_device_code_token_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import DEVICE_CODE_TOKEN_URL

        assert (
            DEVICE_CODE_TOKEN_URL
            == "https://auth.openai.com/api/accounts/deviceauth/token"
        )

    def test_device_code_verification_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_VERIFICATION_URL,
        )

        assert DEVICE_CODE_VERIFICATION_URL == "https://auth.openai.com/codex/device"

    def test_device_code_poll_interval(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_POLL_INTERVAL,
        )

        assert DEVICE_CODE_POLL_INTERVAL == 5

    def test_chatgpt_codex_base_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            CHATGPT_CODEX_BASE_URL,
        )

        assert CHATGPT_CODEX_BASE_URL == "https://chatgpt.com/backend-api/codex"

    def test_token_file_path(self):
        from amplifier_module_provider_openai_chatgpt.oauth import TOKEN_FILE_PATH

        assert TOKEN_FILE_PATH == "~/.amplifier/openai-chatgpt-oauth.json"

    def test_device_code_callback_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_CALLBACK_URL,
        )

        assert DEVICE_CODE_CALLBACK_URL == "https://auth.openai.com/deviceauth/callback"

    def test_no_oauth_authorize_url(self):
        """OAUTH_AUTHORIZE_URL must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "OAUTH_AUTHORIZE_URL"), (
            "OAUTH_AUTHORIZE_URL should have been removed"
        )

    def test_no_oauth_callback_port(self):
        """OAUTH_CALLBACK_PORT must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "OAUTH_CALLBACK_PORT"), (
            "OAUTH_CALLBACK_PORT should have been removed"
        )

    def test_no_oauth_callback_url(self):
        """OAUTH_CALLBACK_URL must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "OAUTH_CALLBACK_URL"), (
            "OAUTH_CALLBACK_URL should have been removed"
        )

    def test_no_subscription_models(self):
        """SUBSCRIPTION_MODELS must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "SUBSCRIPTION_MODELS"), (
            "SUBSCRIPTION_MODELS should have been removed"
        )

    def test_no_start_browser_flow(self):
        """start_browser_flow() must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "start_browser_flow"), (
            "start_browser_flow should have been removed"
        )

    def test_no_is_ssh_session(self):
        """_is_ssh_session() must not exist (dead code, removed)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "_is_ssh_session"), (
            "_is_ssh_session should have been removed as dead code"
        )


class TestPKCE:
    """Verify generate_pkce_pair() behavior per RFC 7636."""

    def test_returns_tuple_of_two_strings(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        result = generate_pkce_pair()
        assert isinstance(result, tuple)
        assert len(result) == 2
        verifier, challenge = result
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_length(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        verifier, _ = generate_pkce_pair()
        assert 43 <= len(verifier) <= 128

    def test_verifier_url_safe_characters(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        verifier, _ = generate_pkce_pair()
        assert re.fullmatch(r"[A-Za-z0-9\-._~]+", verifier), (
            f"Verifier contains non-URL-safe characters: {verifier!r}"
        )

    def test_challenge_is_sha256_of_verifier(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        verifier, challenge = generate_pkce_pair()
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_each_call_returns_unique_pair(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        pair1 = generate_pkce_pair()
        pair2 = generate_pkce_pair()
        assert pair1[0] != pair2[0], "verifiers should be unique per call"
        assert pair1[1] != pair2[1], "challenges should be unique per call"


class TestSaveTokens:
    """Verify save_tokens() file-writing behavior."""

    def test_creates_file_with_correct_json_content(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens

        tokens = {"access_token": "abc", "expires_at": "2099-01-01T00:00:00+00:00"}
        path = str(tmp_path / "tokens.json")
        save_tokens(tokens, path=path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == tokens

    def test_file_has_0600_permissions(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens

        tokens = {"access_token": "abc"}
        path = str(tmp_path / "tokens.json")
        save_tokens(tokens, path=path)
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o600

    def test_creates_parent_directories_if_missing(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens

        tokens = {"access_token": "abc"}
        path = str(tmp_path / "nested" / "deep" / "tokens.json")
        save_tokens(tokens, path=path)
        assert os.path.isfile(path)


class TestLoadTokens:
    """Verify load_tokens() file-reading behavior."""

    def test_returns_dict_for_valid_file(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        tokens = {"access_token": "abc", "expires_at": "2099-01-01T00:00:00+00:00"}
        path = str(tmp_path / "tokens.json")
        with open(path, "w") as f:
            json.dump(tokens, f)
        result = load_tokens(path=path)
        assert result == tokens

    def test_returns_none_for_missing_file(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        path = str(tmp_path / "nonexistent.json")
        result = load_tokens(path=path)
        assert result is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        result = load_tokens(path=path)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        path = str(tmp_path / "empty.json")
        with open(path, "w") as f:
            f.write("")
        result = load_tokens(path=path)
        assert result is None


class TestIsTokenValid:
    """Verify is_token_valid() token validation behavior."""

    def test_valid_future_token_returns_true(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {"access_token": "abc", "expires_at": future}
        assert is_token_valid(tokens) is True

    def test_expired_token_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        past = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        tokens = {"access_token": "abc", "expires_at": past}
        assert is_token_valid(tokens) is False

    def test_none_tokens_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        assert is_token_valid(None) is False

    def test_missing_access_token_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {"expires_at": future}
        assert is_token_valid(tokens) is False

    def test_missing_expires_at_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        tokens = {"access_token": "abc"}
        assert is_token_valid(tokens) is False


class TestExtractAccountId:
    """Verify extract_account_id() JWT decoding behavior."""

    def test_extracts_account_id_from_profile_claim(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        jwt = _make_jwt({"https://api.openai.com/profile": {"account_id": "acc_123"}})
        assert extract_account_id(jwt) == "acc_123"

    def test_falls_back_to_sub_claim(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        jwt = _make_jwt({"sub": "sub_456"})
        assert extract_account_id(jwt) == "sub_456"

    def test_returns_empty_string_for_invalid_jwt(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        assert extract_account_id("not_a_valid_jwt") == ""

    def test_returns_empty_string_for_empty_string(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        assert extract_account_id("") == ""


class TestExtractPlanType:
    """Verify extract_plan_type() JWT decoding behavior."""

    def test_extract_plan_type_happy_path(self) -> None:
        """JWT with chatgpt_plan_type='pro' returns 'pro'."""
        from amplifier_module_provider_openai_chatgpt.oauth import extract_plan_type

        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_plan_type": "pro"}})
        assert extract_plan_type(jwt) == "pro"

    def test_extract_plan_type_missing_claim(self) -> None:
        """JWT with no auth claim returns empty string."""
        from amplifier_module_provider_openai_chatgpt.oauth import extract_plan_type

        jwt = _make_jwt({"sub": "user_abc", "email": "user@example.com"})
        assert extract_plan_type(jwt) == ""

    def test_extract_plan_type_malformed_jwt(self) -> None:
        """Malformed JWT (not three dot-separated parts) returns empty string."""
        from amplifier_module_provider_openai_chatgpt.oauth import extract_plan_type

        assert extract_plan_type("not_a_valid_jwt") == ""

    def test_extract_plan_type_empty_string(self) -> None:
        """Empty string input returns empty string."""
        from amplifier_module_provider_openai_chatgpt.oauth import extract_plan_type

        assert extract_plan_type("") == ""


class TestRefreshTokens:
    """Verify refresh_tokens() HTTP behavior."""

    def test_successful_refresh_returns_new_tokens_with_account_id(self, tmp_path):
        """account_id falls back to disk when id_token is absent from the response."""
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        # Write existing tokens so that refresh_tokens can fall back to account_id.
        token_path = str(tmp_path / "tokens.json")
        with open(token_path, "w") as f:
            json.dump({"account_id": "existing_acc", "access_token": "old"}, f)

        mock_async_client = _make_httpx_mock(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
                # no id_token in response — should fall back to disk account_id
            }
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
            mock_async_client,
        ):
            result = asyncio.run(refresh_tokens("test_refresh", path=token_path))

        assert result is not None
        assert result["auth_mode"] == "oauth"
        assert result["access_token"] == "new_access"
        assert result["account_id"] == "existing_acc"

    def test_account_id_extracted_from_id_token_on_refresh(self, tmp_path):
        """account_id is read from the new id_token JWT when present in the response."""
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        token_path = str(tmp_path / "tokens.json")
        # Old disk tokens have a stale account_id — the id_token should win.
        with open(token_path, "w") as f:
            json.dump({"account_id": "stale_acc", "access_token": "old"}, f)

        id_token_jwt = _make_jwt(
            {"https://api.openai.com/profile": {"account_id": "fresh_acc"}}
        )
        mock_async_client = _make_httpx_mock(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "id_token": id_token_jwt,
                "expires_in": 3600,
            }
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
            mock_async_client,
        ):
            result = asyncio.run(refresh_tokens("test_refresh", path=token_path))

        assert result is not None
        assert result["account_id"] == "fresh_acc"

    def test_refresh_includes_plan_type_from_new_id_token(self, tmp_path):
        """plan_type is extracted from the new id_token JWT when present in the refresh response."""
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        token_path = str(tmp_path / "tokens.json")
        with open(token_path, "w") as f:
            json.dump(
                {"account_id": "acct_1", "plan_type": "free", "access_token": "old"}, f
            )

        id_token_jwt = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_plan_type": "pro"}}
        )
        mock_async_client = _make_httpx_mock(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "id_token": id_token_jwt,
                "expires_in": 3600,
            }
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
            mock_async_client,
        ):
            result = asyncio.run(refresh_tokens("test_refresh", path=token_path))

        assert result is not None
        assert result["plan_type"] == "pro"

    def test_refresh_falls_back_to_disk_plan_type(self, tmp_path):
        """plan_type falls back to value stored on disk when id_token is absent from the response."""
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        token_path = str(tmp_path / "tokens.json")
        with open(token_path, "w") as f:
            json.dump(
                {"account_id": "acct_1", "plan_type": "plus", "access_token": "old"}, f
            )

        mock_async_client = _make_httpx_mock(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
                # no id_token — plan_type must come from disk
            }
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
            mock_async_client,
        ):
            result = asyncio.run(refresh_tokens("test_refresh", path=token_path))

        assert result is not None
        assert result["plan_type"] == "plus"

    def test_refresh_failure_http_401_returns_none(self, tmp_path):
        """HTTP error during refresh causes refresh_tokens to return None."""
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_TOKEN_URL

        request = httpx.Request("POST", OAUTH_TOKEN_URL)
        response = httpx.Response(401, content=b"Unauthorized", request=request)
        status_error = httpx.HTTPStatusError(
            "401 Unauthorized", request=request, response=response
        )

        mock_async_client = _make_httpx_mock(status_error)

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
            mock_async_client,
        ):
            result = asyncio.run(
                refresh_tokens("bad_refresh", path=str(tmp_path / "tokens.json"))
            )

        assert result is None


class TestDeviceCodeFlow:
    """Verify start_device_code_flow() behavior."""

    def test_requests_device_code_and_returns_auth_code(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        mock_async_client = _make_httpx_mock(
            # Step 1: usercode endpoint response
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5},
            # Poll 1: success
            {"authorization_code": "auth_code_xyz"},
        )

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
                mock_async_client,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = asyncio.run(start_device_code_flow())

        assert "authorization_code" in result
        assert result["authorization_code"] == "auth_code_xyz"
        assert "code_verifier" in result

    def test_handles_authorization_pending_then_success(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        mock_async_client = _make_httpx_mock(
            # Step 1: usercode endpoint response
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5},
            # Poll 1: authorization still pending (arrives as 4xx)
            _make_httpx_status_error(
                {"error": {"code": "deviceauth_authorization_unknown"}}
            ),
            # Poll 2: success
            {"authorization_code": "auth_code_xyz"},
        )
        mock_sleep = AsyncMock()

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
                mock_async_client,
            ),
            patch("asyncio.sleep", mock_sleep),
        ):
            result = asyncio.run(start_device_code_flow())

        assert result["authorization_code"] == "auth_code_xyz"
        # Poll-immediately-then-interval: no sleep before poll 1, one sleep
        # before poll 2 (after the pending response).
        assert mock_sleep.call_count == 1

    def test_expired_device_code_raises(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        mock_async_client = _make_httpx_mock(
            # Step 1: usercode endpoint response
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5},
            # Poll 1: device code expired
            _make_httpx_status_error({"error": {"code": "deviceauth_expired"}}),
        )

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
                mock_async_client,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="Device code expired"):
                asyncio.run(start_device_code_flow())


class TestLogin:
    """Verify login() orchestration, especially the direct-tokens path."""

    def test_expires_at_computed_from_expires_in_on_direct_token_path(self, tmp_path):
        """When tokens arrive directly (no code exchange), expires_at must be computed
        from expires_in, never left as an empty string."""
        from amplifier_module_provider_openai_chatgpt.oauth import login

        id_jwt = _make_jwt({"sub": "user_sub"})
        direct_tokens_result = {
            "tokens_direct": True,
            "access_token": "direct_access",
            "refresh_token": "direct_refresh",
            "id_token": id_jwt,
            "expires_in": 7200,
            # intentionally NO expires_at key — must be computed from expires_in
        }

        token_path = str(tmp_path / "tokens.json")
        before = datetime.now(tz=timezone.utc)

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.start_device_code_flow",
            new_callable=AsyncMock,
            return_value=direct_tokens_result,
        ):
            result = asyncio.run(login(token_file_path=token_path))

        after = datetime.now(tz=timezone.utc)

        assert result["access_token"] == "direct_access"
        assert result["expires_at"], "expires_at must not be empty"

        # Verify the timestamp is plausible: between now+7200s ± a few seconds.
        expiry = datetime.fromisoformat(result["expires_at"])
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        assert expiry >= before + timedelta(seconds=7190), (
            f"expires_at {expiry} should be ~2h in the future"
        )
        assert expiry <= after + timedelta(seconds=7210), (
            f"expires_at {expiry} is unexpectedly far in the future"
        )

    def test_direct_token_path_uses_server_expires_at_when_provided(self, tmp_path):
        """When the server supplies expires_at, it takes precedence over computed value."""
        from amplifier_module_provider_openai_chatgpt.oauth import login

        server_expires_at = "2099-12-31T23:59:59+00:00"
        direct_tokens_result = {
            "tokens_direct": True,
            "access_token": "direct_access",
            "refresh_token": "direct_refresh",
            "id_token": "",
            "expires_in": 3600,
            "expires_at": server_expires_at,
        }

        token_path = str(tmp_path / "tokens.json")

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.start_device_code_flow",
            new_callable=AsyncMock,
            return_value=direct_tokens_result,
        ):
            result = asyncio.run(login(token_file_path=token_path))

        assert result["expires_at"] == server_expires_at


# ---------------------------------------------------------------------------
# TestRefreshTokensWarningDetail -- refresh_tokens() warning carries status/body
# ---------------------------------------------------------------------------


class TestRefreshTokensWarningDetail:
    """Verify refresh_tokens() warnings let an operator tell a revoked
    refresh token (HTTP error with status + body) apart from a bare network
    failure (oauth.py refresh_tokens, previously discarded both alike)."""

    def test_http_status_error_warning_carries_status_and_body(self, tmp_path, caplog):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            OAUTH_TOKEN_URL,
            refresh_tokens,
        )

        request = httpx.Request("POST", OAUTH_TOKEN_URL)
        response = httpx.Response(400, json={"error": "invalid_grant"}, request=request)
        status_error = httpx.HTTPStatusError(
            "400 Bad Request", request=request, response=response
        )
        mock_async_client = _make_httpx_mock(status_error)

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
                mock_async_client,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = asyncio.run(
                refresh_tokens("bad_refresh", path=str(tmp_path / "tokens.json"))
            )

        assert result is None
        assert "400" in caplog.text
        assert "invalid_grant" in caplog.text

    def test_network_error_warning_is_distinguishable_from_http_error(
        self, tmp_path, caplog
    ):
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client = MagicMock(return_value=mock_client)

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
                mock_async_client,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = asyncio.run(
                refresh_tokens("bad_refresh", path=str(tmp_path / "tokens.json"))
            )

        assert result is None
        assert "network error" in caplog.text
        # Must not look like the HTTP-status-error message from the sibling test.
        assert "400" not in caplog.text


# ---------------------------------------------------------------------------
# TestDeviceCodePollTimeout -- overall wall-clock bound on the poll loop
# ---------------------------------------------------------------------------


class TestDeviceCodePollTimeout:
    """Verify start_device_code_flow() enforces an overall wall-clock bound
    instead of polling forever (oauth.py's poll loop was `while True` with
    no upper bound)."""

    def test_overall_timeout_raises_runtime_error(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        mock_async_client = _make_httpx_mock(
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5},
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
            mock_async_client,
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                # A negative overall_timeout_s guarantees the deadline is
                # already in the past by the poll loop's first wall-clock
                # check -- proves the bound is enforced deterministically,
                # without waiting, without mocking time.monotonic (which
                # would also corrupt asyncio's own internal clock), and
                # without needing to exhaust a sequence of mocked poll
                # responses.
                asyncio.run(start_device_code_flow(overall_timeout_s=-1.0))


# ---------------------------------------------------------------------------
# TestPrintFnRouting -- print_fn routes device-code output instead of stderr
# ---------------------------------------------------------------------------


class TestPrintFnRouting:
    """Verify start_device_code_flow()/login() route the verification URL
    and user code through print_fn when given, instead of stderr
    (oauth.py's device-code prompt, previously hardcoded to stderr)."""

    def test_print_fn_receives_url_and_code_instead_of_stderr(self, capsys):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        mock_async_client = _make_httpx_mock(
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5},
            {"authorization_code": "auth_code_xyz"},
        )
        captured: list[str] = []

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.httpx.AsyncClient",
                mock_async_client,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            asyncio.run(start_device_code_flow(print_fn=captured.append))

        assert any("codex/device" in line for line in captured)
        assert any("ABC-123" in line for line in captured)

        # Nothing should have gone to stderr when print_fn is given.
        captured_std = capsys.readouterr()
        assert captured_std.err == ""

    def test_login_passes_print_fn_through_to_start_device_code_flow(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import login

        direct_tokens_result = {
            "tokens_direct": True,
            "access_token": "direct_access",
            "refresh_token": "direct_refresh",
            "id_token": "",
            "expires_in": 3600,
        }
        token_path = str(tmp_path / "tokens.json")
        mock_flow = AsyncMock(return_value=direct_tokens_result)
        sentinel_print_fn = MagicMock()

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.start_device_code_flow",
            mock_flow,
        ):
            result = asyncio.run(
                login(token_file_path=token_path, print_fn=sentinel_print_fn)
            )

        assert result["access_token"] == "direct_access"
        mock_flow.assert_awaited_once_with(print_fn=sentinel_print_fn)
