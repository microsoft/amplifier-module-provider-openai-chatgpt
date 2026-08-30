# Amplifier ChatGPT Subscription Provider Module

ChatGPT subscription auth provider for [Amplifier](https://github.com/microsoft/amplifier) -- uses raw HTTP + manual SSE against the ChatGPT backend API (`chatgpt.com/backend-api/codex/responses`).

## Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- A ChatGPT Plus/Pro/Team subscription with device code auth enabled in ChatGPT security settings

## Purpose

Connects Amplifier to the ChatGPT backend API using OAuth device code authentication. This is a separate module from `provider-openai` because the ChatGPT backend is a distinct, undocumented API surface that rejects many standard OpenAI API parameters and requires raw HTTP + manual SSE parsing (the OpenAI Python SDK's streaming accumulator does not work against it).

## Contract

| Field | Value |
|-------|-------|
| Module Type | Provider |
| Mount Point | `providers` |
| Entry Point | `amplifier_module_provider_openai_chatgpt:mount` |

## Configuration

```toml
[providers.provider-openai-chatgpt]
default_model = "latest"
```

### All Config Options

This provider has no `ConfigField`-based setup wizard -- `config_fields` is
deliberately empty (see `get_info()`), because login is a *flow* (OAuth
device-code), not a config *field* a wizard prompts for. The one field
this provider meaningfully exposes, `default_model`, is set by app-cli's
model-picker phase. See "Onboarding" below for the login step itself.
Every key below is a fully supported config key -- set it directly in
`settings.yaml`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_model` | str | `"latest"` | Model to use for inference. `"latest"` is a sentinel meaning "resolve dynamically" -- see "Default Model Resolution" below. Set an explicit model id (e.g. `"gpt-5.4"`) to pin one. |
| `raw` | bool | `false` | Include full request/response payloads in `llm:request`/`llm:response` hook events (for debugging) |
| `login_on_mount` | bool | `true` | Trigger interactive device code login if tokens are absent or expired. Set `false` for non-interactive environments. |
| `token_file_path` | str | `~/.amplifier/openai-chatgpt-oauth.json` | Path to the OAuth token JSON file |
| `timeout` | float | `300.0` | HTTP timeout in seconds for streaming requests |
| `models_cache_ttl` | float | `3600` | How long (seconds) to cache the live model catalog before re-fetching |
| `models_client_version` | str | `"99.99.99"` | Settings-only override for the model-catalog version-gating constant (see `models.py`'s `MODELS_CLIENT_VERSION` -- FRAGILE, relies on the ChatGPT backend treating any unknown high version as "give me everything") |
| `use_streaming` | bool | `true` | Set `false` to force non-streaming completions |
| `priority` | int | `100` | Read by the orchestrator's provider-selection logic |
| `extra_request_params` | dict | `{}` | Merged last into the Responses-API payload -- an escape hatch for any field not listed above. **Warning:** this backend enforces a strict payload schema and is known to reject unrecognized top-level fields (e.g. Chat-Completions-style params like `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs` are NOT accepted here) -- verify any new key against the live backend first. |

Boolean and numeric keys accept native types or the string forms a config
wizard writes (`"true"`/`"false"`); invalid numeric strings warn and fall
back to the default rather than crashing at mount. Unrecognized config
keys produce a mount-time warning (with a did-you-mean suggestion).

### Default Model Resolution

`default_model` defaults to `"latest"` -- a sentinel meaning "resolve
dynamically" instead of a hardcoded model id that goes stale as new model
generations ship. Resolution precedence:

1. **Explicit non-sentinel config value** (e.g. `default_model = "gpt-5.4"`)
   -- used verbatim, no resolution, no network call.
2. **Live-catalog resolution** -- the raw ChatGPT models-endpoint payload
   does not mark any entry as a "default"/"recommended"/"current" model
   (verified directly against the live payload), but the catalog IS ordered
   flagship-first (confirmed by the payload's own per-entry numeric
   `priority` field). "latest" resolves to the first catalog entry whose id
   is not a speed/size variant (i.e. does not end in `-fast` or `-mini`) --
   today that's `gpt-5.6-sol`.
3. **Static fallback** -- `FALLBACK_MODELS[0]` (also `gpt-5.6-sol` today),
   used when unauthenticated or when the live catalog can't be reached. No
   authentication error is raised for this fallback -- auth errors surface
   from actual requests (`complete()`/`list_models()`), never from resolving
   what model name to use.

Resolution happens **lazily**, at the first moment a concrete model name is
actually needed (the first `complete()` call), and is **cached** for the
provider instance's lifetime -- it does not re-fetch on every request. A
single `INFO`-level log line records what `"latest"` resolved to and why
(e.g. `default_model 'latest' resolved to 'gpt-5.6-sol' (live catalog)`).

`get_info()` never triggers a network call to answer this (app-cli's wizard
calls it eagerly, possibly before login): before resolution has happened it
reports something honest like `"latest (resolves lazily; falls back to
gpt-5.6-sol)"` rather than a bare `"latest"` that could be mistaken for a
real, pinned model id; after resolution it reports the concrete resolved id.

**Wizard interplay:** app-cli's model-picker prompts with the provider's
current `default_model` and shows the live catalog when authenticated. A
saved `default_model: latest` is a fully valid, supported settings.yaml
value -- if the picker's "current" model isn't found in the live catalog
list (which is expected for the literal string `"latest"`, since it is
never itself a catalog entry), app-cli's existing "not in list" handling
shows it as an explicit extra "(current)" choice rather than failing; the
provider resolves it the same way regardless of how it got there.

### Authentication

On first use (via `amplifier provider add openai-chatgpt` or
`amplifier provider login openai-chatgpt`), the provider initiates an
OAuth device code flow:

1. Displays a verification URL (`https://auth.openai.com/codex/device`) and a code in the terminal
2. You open the URL in a browser and enter the code
3. Tokens are cached to `~/.amplifier/openai-chatgpt-oauth.json` for subsequent use

Tokens auto-refresh silently **mid-session** when the access token expires
(a 4-step in-memory/disk/refresh fallback chain -- see "Features" below).
If the *refresh* token itself has expired, there is no automatic mid-session
recovery: the device code flow only runs again the next time this provider
mounts (a new session) or when you explicitly run
`amplifier provider login openai-chatgpt`. Call `auth_status()` at any time
to check the current state: `"authenticated"`, `"expired"`, or
`"unauthenticated"`.

Requires "Sign in with device code" to be enabled in your ChatGPT account security settings (Settings > Security).

Works in SSH/headless sessions -- the device code flow only requires a browser on any device, not the machine running Amplifier.

## Features

- OAuth device code authentication with PKCE (no API key needed)
- Raw httpx + manual SSE streaming (not the OpenAI SDK)
- Automatic token refresh with 4-step fallback chain
- Dynamic model catalog from live API (cached, with fallback)
- Subscription plan type detection from OAuth JWT
- Tool calling support
- Reasoning effort support (`low`/`medium`/`high` across all catalog models;
  `xhigh` is additionally accepted for `gpt-5.5-pro`-prefixed model IDs via a
  dedicated pre-flight validator)
- `-fast` model suffix support (e.g. `gpt-5.5-fast` -> `gpt-5.5` with `service_tier: "priority"`)
- Production routing matrix for all 13 Amplifier agent roles
- `llm:request`/`llm:response` hook events with optional raw payload inclusion

## Local Development

```bash
# Clone
git clone https://github.com/microsoft/amplifier-module-provider-openai-chatgpt.git
cd amplifier-module-provider-openai-chatgpt

# Install deps (including dev group: amplifier-core, pytest, ruff)
uv sync

# Run tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_sse.py -v

# Lint and format check
uv run ruff check .
uv run ruff format --check .
```

### Testing with Amplifier

Register the module, install it, and add it through the standard provider management flow:

```bash
# 1. Register the module source
amplifier module add provider-openai-chatgpt \
  --source /path/to/amplifier-module-provider-openai-chatgpt

# 2. Install the provider
amplifier provider install openai-chatgpt --force

# 3. Add the provider. `provider add` runs the OAuth device-code login step
#    as part of onboarding (there is no ConfigField wizard to fill in -- see
#    "All Config Options" above; login is a flow, not a field). You will see
#    a verification URL and a code to enter in a browser.
amplifier provider add openai-chatgpt

# 4. Already added but need to (re-)authenticate? Run login directly
#    instead of re-adding the provider:
amplifier provider login openai-chatgpt

# 5. Or use the management dashboard
amplifier provider manage
```

`login_on_mount` (default `true`) remains a runtime safety net: if a
session starts and this provider's tokens are missing or expired,
`mount()` triggers the same device-code flow automatically. Set it
`false` in non-interactive environments where a stuck login prompt
would hang session startup.

You can also wire it into a bundle directly with an inline `source:` field:

```markdown
---
bundle:
  name: test-openai-chatgpt
  version: 0.1.0

includes:
  - bundle: git+https://github.com/microsoft/amplifier-foundation@main

providers:
  - module: provider-openai-chatgpt
    source: /path/to/amplifier-module-provider-openai-chatgpt
    config:
      default_model: gpt-5.5
---

# Test: provider-openai-chatgpt
```

```bash
amplifier run --bundle ./test-chatgpt.md "Hello, can you hear me?"
```

## Routing Matrix

This module ships with a production routing matrix at `routing/openai-chatgpt.yaml` that maps all 13 Amplifier agent roles to the correct models. This is **required** for agent delegation to work -- without it, agents like `web-research`, `explorer`, and `zen-architect` will fail to resolve a provider.

To use it:

```bash
# Copy to your user routing directory
cp routing/openai-chatgpt.yaml ~/.amplifier/routing/

# Activate it
amplifier routing use openai-chatgpt

# Verify
amplifier routing show
```

The matrix uses two-tier fallback chains (gpt-5.5 -> gpt-5.4) so it works across subscription tiers. Role highlights:

| Role | Primary Model | Config |
|------|--------------|--------|
| `general`, `creative`, `writing`, `vision` | gpt-5.5 | -- |
| `fast` | gpt-?.?-mini* (glob) | -- |
| `coding` | gpt-?.?-codex* (glob) | -- |
| `reasoning`, `research`, `security-audit`, `critical-ops` | gpt-5.5 | `reasoning_effort: high` |
| `critique` | gpt-5.5 | `reasoning_effort: xhigh` |

See the matrix YAML header for full documentation on glob strategy, fallback philosophy, and differences from the standard `openai` routing matrix.

## Supported Models

The model catalog is fetched dynamically from the ChatGPT backend API at `GET /backend-api/codex/models`. Available models depend on your subscription tier. The catalog is cached for 1 hour (configurable via `models_cache_ttl`).

If the live API is unreachable (or `auth_status()` would say `"unauthenticated"`/`"expired"` -- see below), this module's built-in
`FALLBACK_MODELS` catalog (`models.py`) is used instead:

| Model | Context Window | Max Context Window | Speed Tiers | Reasoning Levels |
|-------|----------------|---------------------|-------------|------------------|
| gpt-5.6-sol | 1M | 1M | fast | none/low/medium/high |
| gpt-5.6-terra | 1M | 1M | fast | none/low/medium/high |
| gpt-5.6-luna | 1M | 1M | fast | none/low/medium/high |
| gpt-5.5 | 1M | 1M | fast | none/low/medium/high |
| gpt-5.4 | 272K | 1.05M | fast | none/low/medium/high |
| gpt-5.4-mini | ~1.05M | ~1.05M | -- | none/low/medium |

(`context_window` is the effective limit; `max_context_window` is the
full capacity available on higher-tier plans -- your live catalog may
differ; check `list_models()` for what your subscription actually exposes.)

Models with a "fast" speed tier support a `-fast` suffix (e.g. `gpt-5.5-fast`) which maps to `service_tier: "priority"` in the request. This consumes priority quota faster.

The fallback is only used when `list_models()` cannot reach or successfully
parse the live catalog and the failure is **not** an `AuthenticationError`
(an auth failure propagates instead of silently falling back -- see "Known
Limitations"). The fallback is not cached, so the next `list_models()` call
retries the live API.

## DTU Validation

This module includes a [Digital Twin Universe](https://github.com/microsoft/amplifier-bundle-digital-twin-universe) profile for end-to-end validation in an isolated container. The DTU environment provisions Amplifier with the provider, a pre-authenticated OAuth token, and the routing matrix -- then runs acceptance tests against the live ChatGPT backend API.

```bash
# Launch (requires Incus and a valid OAuth token on the host)
amplifier-digital-twin launch \
  .amplifier/digital-twin-universe/profiles/chatgpt-provider-reality-check.yaml \
  --var OAUTH_TOKEN_FILE=$HOME/.amplifier/openai-chatgpt-oauth.json

# Check readiness
amplifier-digital-twin check-readiness <id>

# Destroy when done
amplifier-digital-twin destroy <id>
```

See [docs/DTU_VALIDATION.md](docs/DTU_VALIDATION.md) for the full guide covering prerequisites, what's tested, what's excluded, and troubleshooting.

## Known Limitations

- **Automatic mid-session 401 recovery** -- if the access token expires mid-session, the provider performs one silent token refresh and retries the request automatically. A second consecutive 401 raises `AuthenticationError`.
- **No `response.incomplete` continuation** -- if a reasoning model hits its output limit, the partial response is lost. Auto-continuation is planned.
- **Streaming is mandatory** -- the ChatGPT backend requires `stream=True`. The provider always streams internally but returns a complete `ChatResponse` to the orchestrator.
- **No dedicated `response.content_part.delta` handling** -- the delta event types the ChatGPT backend actually emits (`response.output_text.delta` for text, `response.reasoning_text.delta`/`response.reasoning_summary_text.delta` for reasoning) are forwarded live via the standard `llm:stream_block_delta` contract. `response.content_part.delta` is a distinct event type this provider does not special-case.
- **`list_models()` does not mask auth failures** -- `AuthenticationError` from the catalog fetch propagates to the caller instead of silently substituting the fallback catalog; only non-auth failures (network errors, parse errors, ...) fall back.

## Dependencies

- `httpx` - HTTP client for raw API requests

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
