"""Model catalog fetch and conversion for the ChatGPT provider.

Single responsibility: fetch the live model catalog from the ChatGPT API,
filter it, expose a fallback list, and convert entries to ModelInfo objects.
"""

from typing import Any

import httpx

from amplifier_core import ModelInfo

from .oauth import CHATGPT_CODEX_BASE_URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_ENDPOINT = f"{CHATGPT_CODEX_BASE_URL}/models"

# FRAGILE: This relies on the ChatGPT backend treating any unknown high version
# as "give me everything". If version gating logic changes, catalog fetches will
# start returning 0 or filtered entries. Canonical list_models() keeps that
# successful empty response distinct from an error; it never uses FALLBACK_MODELS.
MODELS_CLIENT_VERSION = "99.99.99"

DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_MAX_OUTPUT_TOKENS = 128_000

# ---------------------------------------------------------------------------
# Fallback model catalog
# ---------------------------------------------------------------------------

# Ordered flagship-first to mirror the live catalog's own ordering (confirmed
# by the live payload's per-entry numeric `priority` field, which increases
# monotonically in array order -- see the "latest" resolution design note in
# provider.py above `_resolve_default_model`). FALLBACK_MODELS[0] is load-
# bearing: it is both the static fallback used when the live catalog can't be
# reached AND the value "latest" resolves to when unauthenticated.
FALLBACK_MODELS: list[dict[str, Any]] = [
    {
        "slug": "gpt-6.1-sol",
        "display_name": "GPT 6.1 Sol",
        "context_window": 272_000,
        "max_context_window": 272_000,
        "max_output_tokens": 128_000,
        "additional_speed_tiers": ["fast"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-6-luna",
        "display_name": "GPT 6 Luna",
        "context_window": 272_000,
        "max_context_window": 272_000,
        "max_output_tokens": 128_000,
        "additional_speed_tiers": ["fast"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-5.6-sol",
        "display_name": "GPT 5.6 Sol",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
        "additional_speed_tiers": ["fast"],
        "supported_reasoning_levels": ["none", "low", "medium", "high"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-5.6-terra",
        "display_name": "GPT 5.6 Terra",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
        "additional_speed_tiers": ["fast"],
        "supported_reasoning_levels": ["none", "low", "medium", "high"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-5.6-luna",
        "display_name": "GPT 5.6 Luna",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
        "additional_speed_tiers": ["fast"],
        "supported_reasoning_levels": ["none", "low", "medium", "high"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-5.5",
        "display_name": "GPT 5.5",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
        "additional_speed_tiers": ["fast"],
        "supported_reasoning_levels": ["none", "low", "medium", "high"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-5.4",
        "display_name": "GPT 5.4",
        # context_window: effective limit for standard-tier subscriptions;
        # max_context_window: full capacity available on higher-tier plans.
        "context_window": 272_000,
        "max_context_window": 1_050_000,
        "additional_speed_tiers": ["fast"],
        "supported_reasoning_levels": ["none", "low", "medium", "high"],
        "visibility": "list",
        "supported_in_api": True,
    },
    {
        "slug": "gpt-5.4-mini",
        "display_name": "GPT 5.4 mini",
        "context_window": 1_047_576,
        "max_context_window": 1_047_576,
        "additional_speed_tiers": [],
        "supported_reasoning_levels": ["none", "low", "medium"],
        "visibility": "list",
        "supported_in_api": True,
    },
]

# Sentinel value for ChatGPTProvider's `default_model` config: resolve to the
# live catalog's flagship model at first need instead of a hardcoded id that
# goes stale as new model generations ship. See provider.py's
# `_resolve_default_model` for the resolution precedence.
LATEST_MODEL_SENTINEL = "latest"


def is_variant_model_id(model_id: str) -> bool:
    """True if `model_id` is a speed/size variant, not a flagship base model.

    A variant is a synthetic `-fast` speed-tier id (added by
    :func:`to_model_infos`) or a `-mini` size tier that is part of the raw
    slug itself (e.g. ``gpt-5.4-mini``). "latest" resolution skips variants
    so it always lands on a flagship base model.
    """
    return model_id.endswith("-fast") or model_id.endswith("-mini")


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


async def fetch_models(
    *,
    access_token: str,
    account_id: str,
    timeout: float = 30.0,
    client_version: str | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Fetch the live model catalog from the ChatGPT API.

    GETs ``MODELS_ENDPOINT?client_version=MODELS_CLIENT_VERSION`` and
    returns the filtered list of model entry dicts, preserving the raw
    shape for future use.

    Args:
        access_token: OAuth Bearer access token.
        account_id: ChatGPT account ID (``ChatGPT-Account-Id`` header).
        timeout: HTTP request timeout in seconds (default 30.0).
        client_version: Catalog version-gating parameter.
        strict: Validate the response and raise typed HTTP errors for discovery.
            The default retains the legacy fetch/error contract.

    Returns:
        List of model entry dicts with ``visibility != "hide"`` and
        ``supported_in_api is True``.

    Raises:
        ValueError: For a non-200 status in legacy mode, or an invalid catalog
            in strict mode. Strict HTTP failures use kernel LLMError subtypes.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-Id": account_id,
        "OpenAI-Beta": "responses=v1",
        "OpenAI-Originator": "codex",
        "accept": "application/json",
    }
    params = {
        "client_version": (
            MODELS_CLIENT_VERSION if client_version is None else client_version
        )
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(MODELS_ENDPOINT, headers=headers, params=params)

    if resp.status_code != 200:
        if strict:
            from amplifier_core import llm_errors

            from .provider import ChatGPTProvider

            if resp.status_code == 401:
                raise llm_errors.AuthenticationError(
                    "ChatGPT model discovery authentication failed (HTTP 401)",
                    provider="openai-chatgpt",
                    status_code=401,
                    retryable=False,
                )
            ChatGPTProvider._raise_for_status(
                resp.status_code,
                resp.headers,
                resp.content.replace(access_token.encode(), b"[REDACTED]"),
                "openai-chatgpt",
            )
        body = resp.text[:500]
        raise ValueError(f"Models API error {resp.status_code}: {body}")

    data = resp.json()
    entries: list[dict[str, Any]] = (
        _validated_catalog_entries(data, codex=True)
        if strict
        else data.get("models", [])
    )

    # Filter: exclude hidden entries and those not exposed via the API.
    return [
        entry
        for entry in entries
        if entry.get("visibility") != "hide" and entry.get("supported_in_api") is True
    ]


def _validated_catalog_entries(
    data: Any, *, codex: bool = False
) -> list[dict[str, Any]]:
    """Reject malformed discovery responses rather than turn them into emptiness."""
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("ChatGPT model catalog must contain a models array")
    # These routes have no supported pagination request contract in this
    # adapter. Never report an explicitly partial response as a full catalog.
    if "has_more" in data and data["has_more"] is not False:
        raise ValueError("ChatGPT model catalog must be complete")
    if data.get("next_cursor") or data.get("next_page"):
        raise ValueError("ChatGPT model catalog pagination is unsupported")
    entries = data["models"]
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("slug"), str)
            or not entry["slug"].strip()
        ):
            raise ValueError("ChatGPT model catalog entry must contain a non-empty slug")
        if "visibility" in entry and not isinstance(entry["visibility"], str):
            raise ValueError("ChatGPT model catalog visibility must be a string")
        if codex and not isinstance(entry.get("supported_in_api"), bool):
            raise ValueError("Codex model catalog entry must declare API support")
    return entries


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def to_model_infos(
    entries: list[dict[str, Any]], *,
    include_speed_variants: bool = True, strict: bool = False
) -> list[ModelInfo]:
    """Convert raw model catalog entries to :class:`~amplifier_core.ModelInfo` objects.

    For each entry:

    - Emits one ``ModelInfo`` with ``id=slug``.
    - By default, if ``"fast"`` is in ``additional_speed_tiers``, also emits a
      synthetic ``{slug}-fast`` variant with
      ``display_name = "{display_name} (fast)"``.

    Args:
        entries: List of model entry dicts (as returned by :func:`fetch_models`
            or taken from :data:`FALLBACK_MODELS`).
        include_speed_variants: Emit legacy synthetic speed-tier IDs. Strict
            discovery disables this so only literal API slugs are returned.
        strict: Preserve unknown limits as zero (Core's numeric unknown value),
            rather than supplying the legacy inference fallback output limit.
            Reject malformed limits instead of accepting coerced values.

    Returns:
        Flat list of :class:`~amplifier_core.ModelInfo` objects.
    """
    result: list[ModelInfo] = []

    for entry in entries:
        slug: str = entry["slug"]
        display_name: str = entry.get("display_name") or slug
        context_window: int = entry.get("context_window", 0)
        raw_max_output_tokens = entry.get("max_output_tokens")
        max_output_tokens = (
            raw_max_output_tokens
            if isinstance(raw_max_output_tokens, int)
            and not isinstance(raw_max_output_tokens, bool)
            and raw_max_output_tokens > 0
            else DEFAULT_MAX_OUTPUT_TOKENS
        )
        if strict:
            def limit(key: str) -> int:
                value = entry.get(key)
                if value is None:
                    return 0
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("ChatGPT model catalog limit must be a nonnegative integer")
                return value

            context_window = limit("context_window")
            max_output_tokens = limit("max_output_tokens")

        result.append(
            ModelInfo(
                id=slug,
                display_name=display_name,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
            )
        )

        # Emit a synthetic -fast variant when the fast speed tier is supported.
        speed_tiers: list[str] = entry.get("additional_speed_tiers") or []
        if include_speed_variants and "fast" in speed_tiers:
            result.append(
                ModelInfo(
                    id=f"{slug}-fast",
                    display_name=f"{display_name} (fast)",
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                )
            )

    return result


async def fetch_plan_models(
    *, access_token: str, timeout: float = 30.0, strict: bool = False
) -> list[dict[str, Any]]:
    """Public account-specific catalog: no Codex headers or static fallbacks."""
    from .plan_auth import RESOURCE
    from .provider import _raise_plan_error

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        response = await client.get(
            RESOURCE + "/models",
            headers={
                "Authorization": f"Bearer {access_token}",
                "accept": "application/json",
            },
        )
    if response.status_code != 200:
        _raise_plan_error(
            response.status_code,
            response.headers,
            response.content,
            "openai-chatgpt",
            access_token=access_token,
        )
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise TypeError("ChatGPT plan model catalog is invalid")
    entries = _validated_catalog_entries(data) if strict else data["models"]
    return [
        dict(entry, additional_speed_tiers=[])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("visibility") == "list"
        and isinstance(entry.get("slug"), str)
        and entry["slug"]
    ]
