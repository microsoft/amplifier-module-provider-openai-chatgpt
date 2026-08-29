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
default_model = "gpt-5.5"
```

### All Config Options

This provider has no interactive setup wizard (no `ConfigField` prompts) --
it authenticates via OAuth device-code login on mount instead. Every key
below is a fully supported config key -- set it directly in `settings.yaml`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_model` | str | `"gpt-5.5"` | Model to use for inference |
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

### Authentication

On first use, the provider initiates an OAuth device code flow:

1. Displays a verification URL (`https://auth.openai.com/codex/device`) and a code in the terminal
2. You open the URL in a browser and enter the code
3. Tokens are cached to `~/.amplifier/openai-chatgpt-oauth.json` for subsequent use

Tokens auto-refresh silently when they expire. If the refresh token itself expires, the device code flow runs again.

Requires "Sign in with device code" to be enabled in your ChatGPT account security settings (Settings > Security).

Works in SSH/headless sessions -- the device code flow only requires a browser on any device, not the machine running Amplifier.

## Features

- OAuth device code authentication with PKCE (no API key needed)
- Raw httpx + manual SSE streaming (not the OpenAI SDK)
- Automatic token refresh with 4-step fallback chain
- Dynamic model catalog from live API (cached, with fallback)
- Subscription plan type detection from OAuth JWT
- Tool calling support
- Reasoning effort support (`low`/`medium`/`high`/`xhigh` on all gpt-5.x models)
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

# 3. Add the provider (no interactive config wizard -- this provider has no
#    ConfigField prompts; it authenticates via OAuth device-code login on
#    mount instead. Set any config keys directly in settings.yaml -- see
#    "All config keys" below)
amplifier provider add openai-chatgpt

# 4. Or use the management dashboard
amplifier provider manage
```

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

Example catalog for a **Plus** subscription (as of April 2026):

| Model | Context Window | Priority | Speed Tiers | Reasoning |
|-------|---------------|----------|-------------|-----------|
| gpt-5.5 | 272K | 0 (highest) | fast | low/med/high/xhigh |
| gpt-5.4 | 272K | 2 | fast | low/med/high/xhigh |
| gpt-5.4-mini | 272K | 4 | -- | low/med/high/xhigh |
| gpt-5.3-codex | 272K | 6 | -- | low/med/high/xhigh |
| gpt-5.2 | 272K | 10 | -- | low/med/high/xhigh |

Models with a "fast" speed tier support a `-fast` suffix (e.g. `gpt-5.5-fast`) which maps to `service_tier: "priority"` in the request. This consumes priority quota faster.

If the live API is unreachable, a minimal fallback catalog (gpt-5.2, gpt-5.2-codex, gpt-4o) is used. The fallback is not cached, so the next `list_models()` call retries the live API.

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
- **No `response.content_part.delta` handling** -- only `response.output_item.done` events are accumulated. Streaming delta forwarding is planned.

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
