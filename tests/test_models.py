"""Tests for models.py — model catalog fetch and conversion."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_get_mock(
    status_code: int,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    """Return a mock for patching httpx.AsyncClient with a single GET response.

    Args:
        status_code: HTTP status code the mock response reports.
        json_data: If provided, ``resp.json()`` returns this dict.
        text: Value for ``resp.text`` (used for non-200 error body inspection).

    Returns:
        A MagicMock that, when called, returns an async context manager
        whose ``__aenter__`` yields a client where ``client.get`` is an
        AsyncMock returning the mock response.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    if json_data is not None:
        mock_resp.json.return_value = json_data

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=mock_client)


def _make_entry(
    slug: str,
    *,
    visibility: str = "visible",
    supported_in_api: bool = True,
    display_name: str | None = None,
    context_window: int = 128000,
    speed_tiers: list[str] | None = None,
) -> dict:
    """Build a minimal model entry dict that matches the API shape."""
    return {
        "slug": slug,
        "display_name": display_name or slug.upper(),
        "context_window": context_window,
        "supported_in_api": supported_in_api,
        "visibility": visibility,
        "additional_speed_tiers": speed_tiers or [],
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
    }


# Synthetic contract fixtures only: these entries are deliberately fabricated
# test inputs, not captured ChatGPT OAuth catalog data.
@pytest.fixture
def synthetic_sol_luna_catalog() -> list[dict]:
    """Return synthetic backend-advertised GPT-6 catalog entries.

    Values intentionally differ from public direct-API documentation so these
    tests prove catalog pass-through rather than claim OAuth backend limits.
    """
    sol = _make_entry(
        "gpt-6-sol",
        display_name="Synthetic GPT-6 Sol",
        context_window=543_210,
        speed_tiers=["fast"],
    )
    sol["max_output_tokens"] = 54_321
    sol["supported_reasoning_levels"] = ["low", "medium"]

    luna = _make_entry(
        "gpt-6-luna",
        display_name="Synthetic GPT-6 Luna",
        context_window=321_098,
    )
    luna["max_output_tokens"] = 32_109
    luna["supported_reasoning_levels"] = ["low"]

    return [sol, luna]


# ---------------------------------------------------------------------------
# TestFetchModels — fetch_models()
# ---------------------------------------------------------------------------


class TestFetchModels:
    """Verify fetch_models() HTTP behaviour, filtering, and error handling."""

    @pytest.mark.asyncio
    async def test_fetch_models_filters_correctly(self) -> None:
        """Only models with visibility!='hide' AND supported_in_api=True pass through."""
        from amplifier_module_provider_openai_chatgpt.models import fetch_models

        json_data = {
            "models": [
                # Filtered out: visibility == "hide"
                _make_entry("hidden-model", visibility="hide"),
                # Filtered out: supported_in_api == False
                _make_entry("no-api-model", supported_in_api=False),
                # Passes: visible and API-supported
                _make_entry("valid-model"),
            ]
        }

        mock_async_client = _make_get_mock(200, json_data=json_data)

        with patch(
            "amplifier_module_provider_openai_chatgpt.models.httpx.AsyncClient",
            mock_async_client,
        ):
            result = await fetch_models(access_token="tok", account_id="acct")

        assert len(result) == 1
        assert result[0]["slug"] == "valid-model"

    @pytest.mark.asyncio
    async def test_fetch_models_sends_correct_request(self) -> None:
        """GET must target MODELS_ENDPOINT with client_version param and correct headers."""
        from amplifier_module_provider_openai_chatgpt.models import (
            MODELS_CLIENT_VERSION,
            MODELS_ENDPOINT,
            fetch_models,
        )

        mock_async_client = _make_get_mock(200, json_data={"models": []})

        with patch(
            "amplifier_module_provider_openai_chatgpt.models.httpx.AsyncClient",
            mock_async_client,
        ):
            await fetch_models(access_token="test-token", account_id="acct-789")

        mock_client = mock_async_client.return_value
        assert mock_client.get.called

        call = mock_client.get.call_args
        url = call.args[0]
        params = call.kwargs.get("params", {})
        headers = call.kwargs.get("headers", {})

        assert url == MODELS_ENDPOINT
        assert params.get("client_version") == MODELS_CLIENT_VERSION

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["ChatGPT-Account-Id"] == "acct-789"
        assert headers["OpenAI-Beta"] == "responses=v1"
        assert headers["OpenAI-Originator"] == "codex"
        assert headers["accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_fetch_models_raises_on_non_200(self) -> None:
        """Non-200 response raises ValueError containing the status code."""
        from amplifier_module_provider_openai_chatgpt.models import fetch_models

        mock_async_client = _make_get_mock(401, text="Unauthorized request")

        with patch(
            "amplifier_module_provider_openai_chatgpt.models.httpx.AsyncClient",
            mock_async_client,
        ):
            with pytest.raises(ValueError, match="401"):
                await fetch_models(access_token="bad-tok", account_id="acct")


# ---------------------------------------------------------------------------
# TestToModelInfos — to_model_infos()
# ---------------------------------------------------------------------------


class TestToModelInfos:
    """Verify to_model_infos() ModelInfo construction and fast-variant emission."""

    def test_to_model_infos_emits_fast_variant(self) -> None:
        """Entry with 'fast' in additional_speed_tiers produces two ModelInfos."""
        from amplifier_module_provider_openai_chatgpt.models import to_model_infos

        entries = [_make_entry("gpt-5.2", context_window=272000, speed_tiers=["fast"])]
        result = to_model_infos(entries)

        assert len(result) == 2
        ids = [m.id for m in result]
        assert "gpt-5.2" in ids
        assert "gpt-5.2-fast" in ids

        fast = next(m for m in result if m.id == "gpt-5.2-fast")
        assert "(fast)" in fast.display_name

    def test_to_model_infos_preserves_valid_output_limit_for_fast_variants(self) -> None:
        """Raw catalog output limits remain model-specific for both variants."""
        from amplifier_module_provider_openai_chatgpt.models import to_model_infos

        entry = _make_entry("gpt-5.2", context_window=272000, speed_tiers=["fast"])
        entry["max_output_tokens"] = 64_000

        result = to_model_infos([entry])

        assert {model.max_output_tokens for model in result} == {64_000}

    @pytest.mark.parametrize("max_output_tokens", [None, 0, True])
    def test_to_model_infos_uses_fallback_for_absent_or_invalid_output_limit(
        self, max_output_tokens: int | bool | None
    ) -> None:
        """Missing, non-positive, and boolean raw limits use the planning fallback."""
        from amplifier_module_provider_openai_chatgpt.models import (
            DEFAULT_MAX_OUTPUT_TOKENS,
            to_model_infos,
        )

        entry = _make_entry("gpt-5.2", speed_tiers=["fast"])
        if max_output_tokens is not None:
            entry["max_output_tokens"] = max_output_tokens

        result = to_model_infos([entry])

        assert {model.max_output_tokens for model in result} == {
            DEFAULT_MAX_OUTPUT_TOKENS
        }

    def test_to_model_infos_no_fast_when_absent(self) -> None:
        """Entry without 'fast' in speed_tiers produces exactly one ModelInfo."""
        from amplifier_module_provider_openai_chatgpt.models import to_model_infos

        entries = [_make_entry("gpt-4o", context_window=128000)]
        result = to_model_infos(entries)

        assert len(result) == 1
        assert result[0].id == "gpt-4o"

    def test_synthetic_gpt_6_catalog_preserves_advertised_limits_and_tiers(
        self, synthetic_sol_luna_catalog: list[dict]
    ) -> None:
        """Synthetic catalog metadata passes through without GPT-6 assumptions."""
        from amplifier_module_provider_openai_chatgpt.models import to_model_infos

        models_by_id = {
            model.id: model for model in to_model_infos(synthetic_sol_luna_catalog)
        }

        assert models_by_id["gpt-6-sol"].context_window == 543_210
        assert models_by_id["gpt-6-sol"].max_output_tokens == 54_321
        assert models_by_id["gpt-6-sol-fast"].context_window == 543_210
        assert models_by_id["gpt-6-sol-fast"].max_output_tokens == 54_321
        assert models_by_id["gpt-6-luna"].context_window == 321_098
        assert models_by_id["gpt-6-luna"].max_output_tokens == 32_109

        # Only the backend-advertised Sol fast tier is emitted. No GPT-6 model
        # absent from the synthetic catalog is inferred or promoted.
        assert set(models_by_id) == {"gpt-6-sol", "gpt-6-sol-fast", "gpt-6-luna"}

    def test_fallback_first_entry_is_gpt_56_sol(self) -> None:
        """FALLBACK_MODELS first entry must be gpt-5.6-sol (current flagship).

        Load-bearing beyond "which model is listed first": it is also the
        value ChatGPTProvider's "latest" sentinel resolves to when the live
        catalog can't be reached -- see
        test_config_hygiene.py::test_default_model_sentinel_and_fallback_first_entry_match.
        """
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        assert len(FALLBACK_MODELS) > 0, "FALLBACK_MODELS must not be empty"
        assert FALLBACK_MODELS[0]["slug"] == "gpt-5.6-sol", (
            f"Expected gpt-5.6-sol as first fallback entry, got {FALLBACK_MODELS[0]['slug']!r}"
        )

    def test_fallback_models_round_trip(self) -> None:
        """FALLBACK_MODELS through to_model_infos produces valid ModelInfo objects."""
        from amplifier_core import ModelInfo

        from amplifier_module_provider_openai_chatgpt.models import (
            FALLBACK_MODELS,
            to_model_infos,
        )

        result = to_model_infos(FALLBACK_MODELS)

        assert len(result) > 0
        for m in result:
            assert isinstance(m, ModelInfo)
            assert m.id, f"ModelInfo missing id: {m}"
            assert m.display_name, f"ModelInfo missing display_name: {m}"
            assert m.context_window > 0, f"ModelInfo missing context_window: {m}"
            assert m.max_output_tokens > 0, f"ModelInfo missing max_output_tokens: {m}"
