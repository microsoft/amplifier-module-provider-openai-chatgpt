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
# start returning 0 or filtered entries. Symptom: list_models() always returns
# FALLBACK_MODELS. Check this constant first when debugging catalog issues.
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
) -> list[dict[str, Any]]:
    """Fetch the live model catalog from the ChatGPT API.

    GETs ``MODELS_ENDPOINT?client_version=MODELS_CLIENT_VERSION`` and
    returns the filtered list of model entry dicts, preserving the raw
    shape for future use.

    Args:
        access_token: OAuth Bearer access token.
        account_id: ChatGPT account ID (``ChatGPT-Account-Id`` header).
        timeout: HTTP request timeout in seconds (default 30.0).

    Returns:
        List of model entry dicts with ``visibility != "hide"`` and
        ``supported_in_api is True``.

    Raises:
        ValueError: If the API returns a non-200 status code.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-Id": account_id,
        "OpenAI-Beta": "responses=v1",
        "OpenAI-Originator": "codex",
        "accept": "application/json",
    }
    params = {"client_version": MODELS_CLIENT_VERSION}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(MODELS_ENDPOINT, headers=headers, params=params)

    if resp.status_code != 200:
        body = resp.text[:500]
        raise ValueError(f"Models API error {resp.status_code}: {body}")

    data = resp.json()
    entries: list[dict[str, Any]] = data.get("models", [])

    # Filter: exclude hidden entries and those not exposed via the API.
    return [
        entry
        for entry in entries
        if entry.get("visibility") != "hide" and entry.get("supported_in_api") is True
    ]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def to_model_infos(entries: list[dict[str, Any]]) -> list[ModelInfo]:
    """Convert raw model catalog entries to :class:`~amplifier_core.ModelInfo` objects.

    For each entry:

    - Emits one ``ModelInfo`` with ``id=slug``.
    - If ``"fast"`` is in ``additional_speed_tiers``, also emits a synthetic
      ``{slug}-fast`` variant with ``display_name = "{display_name} (fast)"``.

    Args:
        entries: List of model entry dicts (as returned by :func:`fetch_models`
            or taken from :data:`FALLBACK_MODELS`).

    Returns:
        Flat list of :class:`~amplifier_core.ModelInfo` objects.
    """
    result: list[ModelInfo] = []

    for entry in entries:
        slug: str = entry["slug"]
        display_name: str = entry.get("display_name") or slug
        context_window: int = entry.get("context_window", 0)

        result.append(
            ModelInfo(
                id=slug,
                display_name=display_name,
                context_window=context_window,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )
        )

        # Emit a synthetic -fast variant when the fast speed tier is supported.
        speed_tiers: list[str] = entry.get("additional_speed_tiers") or []
        if "fast" in speed_tiers:
            result.append(
                ModelInfo(
                    id=f"{slug}-fast",
                    display_name=f"{display_name} (fast)",
                    context_window=context_window,
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                )
            )

    return result
