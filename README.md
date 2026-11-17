# Amplifier ChatGPT Subscription Provider Module

Choose **ChatGPT Codex** (`chatgpt_codex`, default) for the existing device sign-in
and Codex-compatible backend, or **ChatGPT plan** (`chatgpt_plan`) for the public
Sign in with ChatGPT browser flow. Both modes are implemented by this provider
with separate credentials and endpoints, with no automatic mode or account
fallback. See [connection modes and plan sign-in](docs/CHATGPT_PLAN_SIGN_IN.md).

ChatGPT subscription auth provider for [Amplifier](https://github.com/microsoft/amplifier)
using raw HTTP and SSE. The configuration and behavior below describe the default
Codex mode unless a section explicitly says otherwise.

## Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- A ChatGPT Plus/Pro/Team subscription with device code auth enabled in ChatGPT security settings

## Purpose

Connects Amplifier to ChatGPT-backed inference using the explicitly selected
authentication and transport. Codex mode uses device authorization and
`chatgpt.com/backend-api/codex/responses`; plan mode uses browser authorization and
`api.openai.com/v1/responses`. The two routes have different request contracts.

## Contract

| Field | Value |
|-------|-------|
| Module Type | Provider |
| Mount Point | `providers` |
| Entry Point | `amplifier_module_provider_openai_chatgpt:mount` |

## Configuration

```toml
[providers.provider-openai-chatgpt]
auth_mode = "chatgpt_codex"
default_model = "latest"
```

### All Config Options

`get_info().config_fields` exposes the connection mode, an optional credential
path, and plan-only host identity/application-name fields through the kernel's
generic configuration contract. Login is a separate explicit flow. Model
selection and additional tuning can be supplied by the host or configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auth_mode` | str | `chatgpt_codex` | `chatgpt_codex` device sign-in or `chatgpt_plan` public browser sign-in. `legacy_codex` is a compatibility alias. |
| `host_file_path` | str | `host.json` beside plan credential file | Persistent plan host identity. Reuse the path on reconnect/import. |
| `app_name` | str | `Amplifier ChatGPT provider` | Actual host application name for plan sign-in; not an account or routing key. |
| `default_model` | str | `"latest"` | Model to use for inference. `"latest"` is a sentinel meaning "resolve dynamically" -- see "Default Model Resolution" below. Set an explicit model id (e.g. `"gpt-5.4"`) to pin one. |
| `raw` | bool | `false` | Include full request/response payloads in `llm:request`/`llm:response` hook events (for debugging) |
| `login_on_mount` | bool | deprecated | Accepted for compatibility; mounting never starts sign-in. Use explicit account setup to sign in. |
| `token_file_path` | str | Mode-specific | Codex: `~/.amplifier/openai-chatgpt-oauth.json`; plan: `~/.amplifier/chatgpt-plan/default.json`. Explicit blank paths are rejected. |
| `timeout` | float or null | `null` | No default generation read deadline; a number opts in. Connection/pool acquisition remains bounded to 5 seconds. |
| `max_retries` | int | `5` | Additional transient completion attempts; `0` disables them, not the existing one-time Codex 401 refresh. |
| `min_retry_delay` | float | `1` | Initial/base exponential backoff in seconds, not a jittered minimum. |
| `max_retry_delay` | float | `60` | Base backoff cap and maximum accepted server wait hint. |
| `retry_jitter` | bool | `true` | Apply Core's jitter to backoff. |
| `models_cache_ttl` | float | `3600` | Inference-only catalog cache lifetime; canonical `list_models()` always fetches fresh |
| `models_client_version` | str | `"99.99.99"` | Settings-only override for the model-catalog version-gating constant (see `models.py`'s `MODELS_CLIENT_VERSION` -- FRAGILE, relies on the ChatGPT backend treating any unknown high version as "give me everything") |
| `model_filters` | mapping | `{"include": ["gpt-6*"]}` | Immutable recommendation defaults for `list_models()`. Supports `include`, `exclude`, and `hide_dated`; does not restrict explicit inference. |
| `use_streaming` | bool | `true` | Set `false` to force non-streaming completions |
| `reasoning_effort` | str | *(unset)* | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Lands as `reasoning.effort`; an explicit request value wins. `none` or unset means "don't inject -- let the model decide". Invalid values fail at mount. |
| `priority` | int | `100` | Read by the orchestrator's provider-selection logic |
| `extra_request_params` | dict | `{}` | Additional wire parameters. Codex retains its existing pass-through behavior. Plan mode validates documented restrictions and rejects unsupported fields/tools instead of dropping them; see [plan limits](docs/CHATGPT_PLAN_SIGN_IN.md#plan-request-limits). |

Boolean and numeric keys accept native types or the string forms a config
wizard writes (`"true"`/`"false"`); invalid numeric strings warn and fall
back to the default rather than crashing at mount. Unrecognized config
keys produce a mount-time warning (with a did-you-mean suggestion).

### Fresh account model listing

Both modes implement `await provider.list_models(*, filters=None)` returning
Core `list[ModelInfo]`. Each call reads the current bound account and requests
its catalog. There is no catalog-cache or static-list fallback. A successful
empty catalog returns `[]`; authentication, transport, malformed or explicitly
partial responses raise typed errors. The candidate-only `discover_models()`
method has been removed in favor of this canonical method.

```python
await provider.list_models()                         # configured defaults: gpt-6*
await provider.list_models(filters=None)             # same defaults
await provider.list_models(filters={})               # all account-listed API models
await provider.list_models(filters={"include": ["gpt-5*"]})  # replace defaults
await provider.list_models(filters={"include": []})  # no recommendations
```

`include` and `exclude` accept lists of case-insensitive, OS-independent glob
patterns; exclusion wins. An explicit mapping **replaces**, not merges with,
the configured defaults. `{}` also disables configured dated suppression.
`hide_dated: true` hides a `YYYY-MM-DD` snapshot only when its matching undated
alias is selected. It defaults to false for this provider: all actual `gpt-6*`
tiers, previews and snapshots remain recommended, in endpoint order. This is
not an intelligence ranking. IDs preserve their exact endpoint bytes.
Endpoint visibility/API eligibility still applies; overrides do not invent
access to hidden or unsupported entries.

No synthetic `-fast` models are added, including when the endpoint advertises
speed tiers. A literal endpoint ID ending in `-fast` remains literal inventory.
Missing limits are reported as `0` (unknown), not a guessed context or output
budget; no capabilities are inferred. `get_model_list_info()` describes this
extension without I/O using version 1, source `provider-api`, `filters_keyword`,
`filters_schema`, `default_filters`, `unfiltered: {}`, and `override: replace`.
Configuration and returned metadata are independent snapshots; invalid filter
types or keys fail before authentication.

Listing does not alter an explicit inference model, request parameters, or
the existing inference-only `latest` resolution/cache. Older models remain
usable by exact supported ID. The legacy Codex `-fast` inference shorthand is
unchanged, but is not advertised as a separate account model. Plan inference
continues to pass exact IDs without that remapping.

### Completion retries

Both sign-in modes use Core's shared `RetryConfig` / `retry_with_backoff`
mechanism, like the OpenAI and Anthropic providers. Known overload/server errors,
temporary rate limits, and transport failures retry the **same prepared request**,
not a conversation turn or its tools. There is no account, model or mode fallback.
Authentication, hard quota/plan usage limits, invalid requests, context/policy
errors and unknown service codes do not retry automatically.

Retries stop as soon as any text, thinking or tool-use block hook is emitted,
including a block start. Buffered requests that emit no blocks may retry.
Cancellation interrupts requests and backoff. Healthy generation still has no
implicit read deadline. A lost response can have been generated and billed
remotely; retries do not promise exactly-once inference.

Numeric and HTTP-date `Retry-After` hints are honored as minimum waits; hints
above `max_retry_delay` decline retry rather than retry early. Invalid hints are
ignored. Core's jitter may increase a wait above the **base** backoff cap (up to
1.5 times it); this setting is not a total elapsed-time or spend budget.
`llm:request` and terminal `llm:response` each occur once per logical request;
`provider:retry` reports attempt, delay and error class without request bodies.
The four option names match the OpenAI provider. Retry delays must be finite,
non-negative and ordered (`min_retry_delay <= max_retry_delay`); invalid ranges
fail during provider construction rather than creating an unbounded wait.
With defaults, six attempts can incur up to 46.5 seconds of ordinary backoff
(or up to 450 seconds if every server hint is 60 seconds), plus time spent on
the requests. Explicit read timeouts apply per attempt, not to the campaign.

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
   the concrete result depends on the live account catalog.
3. **Static fallback** -- `FALLBACK_MODELS[0]` (`gpt-6.1-sol` in this refresh),
   used when unauthenticated or when the live catalog can't be reached. No
   authentication error is raised for this fallback -- auth errors surface
   from actual requests (`complete()`/`list_models()`), never from resolving
   what model name to use.

Resolution happens **lazily**, at the first moment a concrete model name is
actually needed (the first `complete()` call), and is **cached** for the
provider instance's lifetime -- it does not re-fetch on every request. A
single `INFO`-level log line records what `"latest"` resolved to and why
(e.g. `default_model 'latest' resolved to 'gpt-6.1-sol' (live catalog)`).

`get_info()` never triggers a network call to answer this (app-cli's wizard
calls it eagerly, possibly before login): before resolution has happened it
reports something honest like `"latest (resolves lazily; falls back to
gpt-6.1-sol)"` rather than a bare `"latest"` that could be mistaken for a
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

- Native ordered base64 PNG/JPEG user images and structured tool-result images
  in both connection modes; see [image input and transport evidence](#image-input-and-transport-evidence)
- OAuth device code authentication with PKCE (no API key needed)
- Raw httpx + manual SSE streaming (not the OpenAI SDK)
- Automatic token refresh with 4-step fallback chain
- Fresh account model recommendations with explicit filter overrides; inference-only cache/fallback remains separate
- Subscription plan type detection from OAuth JWT
- Tool calling support
- Reasoning effort support (`low`/`medium`/`high` across all catalog models;
  `xhigh` is additionally accepted for `gpt-5.5-pro`-prefixed model IDs via a
  dedicated pre-flight validator)
- `-fast` model suffix support (e.g. `gpt-5.5-fast` -> `gpt-5.5` with `service_tier: "priority"`)
- Production routing matrix for all 13 Amplifier agent roles
- `llm:request`/`llm:response` hook events with optional raw payload inclusion

### Image input and transport evidence

Both adapters convert Core `ImageBlock(source={"type": "base64",
"media_type": "image/png", "data": ...})` into Responses
`{"type": "input_image", "image_url": "data:image/png;base64,..."}`.
JPEG uses `image/jpeg`. Mixed text/image order, multiple images, original image
bytes and user history are preserved. No model-name pattern is taken as proof
of vision support: the selected account/model must accept images.

Call-bound images use the host's `Message(role="tool", tool_call_id=...,
content=[TextBlock(...), ImageBlock(...)])`, paired with a preceding assistant
tool call. They also accept `ToolResultBlock.output` containing a nonempty list of
explicit Core block dictionaries, for example `[{type: "text", text: "..."},
{type: "image", source: {type: "base64", media_type: "image/png", data: ...}}]`.
Only image-bearing lists of this strict shape become a native
`function_call_output.output` array of `input_text`/`input_image` parts. Call ID
and result position stay unchanged. Other JSON outputs stay JSON strings;
strings are never parsed or promoted, and no synthetic user message is added.

URL/file/path sources, GIF/WebP/other formats, and assistant/system/developer
image positions raise a nonretryable `InvalidRequestError` with
`error_code="unsupported_image_input"` and `error_param="messages.content.image"`.
Convert an unsupported format explicitly before attachment. Nothing is silently
dropped, fetched, uploaded, resized, re-encoded or removed from saved history.
Validation checks strict base64 plus the PNG/JPEG signature only;
it is **not** a full image decoder or raster-integrity/visual-safety certificate.
The host owns upload/decode resource policy and admission before saving.
Valid decoder-accepted trailers are preserved; this adapter does not pretend a
terminal-byte check establishes raster completeness.

`get_media_request_support_for_request(*, request, model, request_options=None)`
returns synchronous, read-only v2 indexed facts with exact `v`, `positions`
(`message_index`/`role`) and `mime_types` fields. It qualifies user positions and
actual paired raw-tool-message positions for the twelve exact image-input models
in the [immutable first-party Codex catalogue](https://github.com/openai/codex/blob/65a5410f72dac1f4b40862c34ef0a8b71b12834a/codex-rs/models-manager/models.json).
Unknown/unresolved models and input/model overrides have no qualified positions.
These facts are not authority to hydrate references; the host owns request,
account, generation and lifetime binding. Plan **conversion** supports native
images, but plan automatic-hydration positions remain empty until account/model
modality evidence is available for that separate route. The Codex catalogue is
not borrowed to invent plan capability or entitlement.

Participating requests reuse the actual `experimental.media.v1` capability's
synchronous `validate(request, purpose="provider-send", request_options=...)`
on the same private Core request immediately before **each** physical POST,
including after auth/model awaits, 401 refresh and retry backoff. Missing,
replaced, invalidated or changed witnesses/options/connection selections refuse
before sending. No preparation, hydration, account fallback, or tool rerun is
performed after fit. This optional structural boundary imports no Foundation
runtime code; legacy inline requests without media capability/references remain
compatible. The host owns actual candidate identity and lifetime, and must
qualify its installed composition separately.

Plan image requests check the public API's documented 1,500-image and 512 MB
total serialized-payload limits (MB is conservatively decimal here). These are
outer limits, not admission guarantees: account/model/context/patch limits can
be lower. No undocumented Codex byte/pixel ceiling is invented. Unsupported
history remains an explicit error until the host/user repairs that attachment;
the provider neither mutates history nor poisons subsequent independent text
requests. Host admission and original-chat recovery are separate integration work.

Native image URL/data values are omitted from diagnostic **copies before hooks**,
including raw request/response events and known exact echoes in HTTP/SSE errors.
Known echoes split across streaming text/thinking deltas are filtered locally;
redundant raw delta values and opaque function-call argument strings are omitted
on image-bearing requests (including JSON-escaped echoes). Actual wire
payloads, completed model output, reasoning and tool arguments are not rewritten.
The marker `[IMAGE REDACTED]` is not a retrievable media reference. Exact-value
redaction does not certify truncated, re-encoded or arbitrary
third-party traceback/local-variable capture.

Evidence reviewed 2026-11-17; offline MockTransport tests qualify adapter wire
serialization only, **not authenticated endpoint acceptance or visual inference**:

| Surface | `chatgpt_codex` | `chatgpt_plan` | Evidence / implementation |
|---|---|---|---|
| Inference + auth | Device OAuth, `chatgpt.com/backend-api/codex/responses` | Browser OAuth plan permission, `api.openai.com/v1/responses` | [Official Codex client route][codex-route]; [documented plan route][plan-inference]. These are distinct from Platform API-key billing; no fallback. |
| Text, local function tools, reasoning, SSE | Implemented | Implemented; functions through `additional_tools`, `store:false`, `stream:true` | Existing source/offline tests; [plan limitations][plan-limits] |
| Base64 user images | Implemented PNG/JPEG | Implemented PNG/JPEG | [Documented image shape][vision-doc]; [official-client image wire][codex-image]; offline tests, not live qualification |
| Native tool-result images | Implemented strict typed list | Implemented strict typed list | [Official-client Responses output schema][codex-tool-image]; public plan documents image/function inputs, exact combination still requires service qualification |
| Other image formats / URL sources / detail controls | Not translated | Not translated | General API documents WebP/nonanimated GIF, URLs, model-dependent detail; not proof of Codex backend parity |
| Files / custom or namespaced functions / WebSockets | Not translated | Legitimate separate backlog | Plan documents files (not uploads), custom/namespaced tools, same-connection WebSocket continuation. No feature enabled by this change. |
| Image generation, native computer use, hosted MCP, file search, Code Interpreter, audio/video, Files upload | Not established by this adapter | Explicitly unavailable in plan preview | [Plan restrictions][plan-limits]; do not infer support from generic Responses schema |
| Arbitrary-request input count / enforced output cap / persistent server conversation | Not added | Not added | Existing source limits; plan rejects unsupported parameters. No counter/auth crossover. |

Immediate reusable improvements are the native image conversion and
privacy-safe diagnostic copies. Model-aware detail, other image forms, inline
file conversion, custom tools and new transports remain separate research/
qualification tasks. This work does not adopt the larger private-media
preparation/auth-binding experiment in [draft PR #28](https://github.com/microsoft/amplifier-module-provider-openai-chatgpt/pull/28)
(`b0fae0992b92f9f86ce2abe7f65b1dd38468b1d2`), whose original-position tool image
and diagnostic-copy design was reviewed as prior work. Its held source and proof
remain separate; no draft branch or compaction experiment is imported.

[plan-inference]: https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference.md
[plan-limits]: https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations.md
[vision-doc]: https://developers.openai.com/api/docs/guides/images-vision.md
[codex-route]: https://github.com/openai/codex/blob/9a56c2b1aa994f696b144ba863200f4b73a74c0a/codex-rs/model-provider-info/src/lib.rs#L472-L490
[codex-image]: https://github.com/openai/codex/blob/9a56c2b1aa994f696b144ba863200f4b73a74c0a/codex-rs/protocol/src/models.rs#L940-L995
[codex-tool-image]: https://github.com/openai/codex/blob/9a56c2b1aa994f696b144ba863200f4b73a74c0a/codex-rs/protocol/src/models.rs#L2235-L2355

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

Module loading and software update validation work without account credentials.
Before listing models or making a request, the provider automatically reuses or
refreshes saved credentials. If that fails, an authentication error asks the host
to offer explicit sign-in. Mounting never opens a browser, refreshes credentials
or tests subscription access; `login_on_mount` is accepted but no longer used.

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

Prefer the routing-matrix bundle's canonical **`openai`** matrix. It routes both
OpenAI API and ChatGPT subscription backends with caller-consistent delegation.
This provider's `routing/openai-chatgpt.yaml` remains a compatibility matrix for
existing explicit selections, refreshed to Sol 6.1 / Luna 6. It is not required
when canonical routing is installed.

For new setups:

```bash
amplifier routing use openai

# Verify
amplifier routing show
```

The compatibility matrix pins Sol 6.1 for most roles and Luna 6 for fast/UI.
Runtime exact pins do not check entitlement, and a second exact pin would not
provide model-not-found fallback. Check the account's live catalog before use.
New model choices are an interim refresh, not a comparative quality evaluation.

| Role | Primary Model | Config |
|------|--------------|--------|
| Most roles | gpt-6.1-sol | Existing role effort retained |
| `fast` | gpt-6-luna | -- |
| `ui-coding` | gpt-6-luna | `reasoning_effort: max` |

See the matrix YAML header for full documentation on glob strategy, fallback philosophy, and differences from the standard `openai` routing matrix.

## Supported Models

The model catalog is fetched dynamically from the ChatGPT backend API at `GET /backend-api/codex/models`. Available models depend on your subscription tier. The catalog is cached for 1 hour (configurable via `models_cache_ttl`).

If the live API is unreachable (or `auth_status()` would say `"unauthenticated"`/`"expired"` -- see below), this module's built-in
`FALLBACK_MODELS` catalog (`models.py`) is used instead:

| Model | Context Window | Max Context Window | Speed Tiers | Reasoning Levels |
|-------|----------------|---------------------|-------------|------------------|
| gpt-6.1-sol | 272K | 272K conservative fallback | fast | Consult live catalog |
| gpt-6-luna | 272K | 272K conservative fallback | fast | Consult live catalog |
| gpt-5.6-sol | 1M | 1M | fast | none/low/medium/high |
| gpt-5.6-terra | 1M | 1M | fast | none/low/medium/high |
| gpt-5.6-luna | 1M | 1M | fast | none/low/medium/high |
| gpt-5.5 | 1M | 1M | fast | none/low/medium/high |
| gpt-5.4 | 272K | 1.05M | fast | none/low/medium/high |
| gpt-5.4-mini | ~1.05M | ~1.05M | -- | none/low/medium |

(`context_window` is the effective limit; `max_context_window` is the
full capacity available on higher-tier plans -- your live catalog may
differ; check `list_models()` for what your subscription actually exposes.)

The current fallback starts with Sol 6.1 and includes Luna 6. Its 272K GPT-6
input limits are conservative fallback values, not an entitlement claim. Older
entries remain for explicit legacy pins; they are not the default selection.
Catalog metadata does not invent vision capability tags. Plan mode continues
to require its account-specific live catalog and never borrows this Codex list.

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

- **No native arbitrary-request input counter** -- this raw HTTP/SSE adapter
  has no source-supported preflight count operation for
  `chatgpt.com/backend-api/codex/responses`, so it does not advertise
  `request_budget:provider_count`. Subscription OAuth credentials are never
  forwarded to the separate OpenAI Platform counter.
- **Advisory planning metadata is selected-model-only** -- `get_info()`
  reports `context_window` and `max_output_tokens` together only when both
  positive values are known for the configured/resolved model from the
  already-cached catalog, or the built-in catalog when no live catalog is
  cached. Unresolved `latest` and unknown models omit both. These values are
  for context-compaction planning, not backend admission proof or an enforced
  output reservation.
- **No request-level output cap** -- this backend omits or rejects
  `max_output_tokens`; the advisory `get_info()` metadata is never sent in a
  request payload and does not guarantee the backend accepts that output size.
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

### Cache usage reporting

When the backend reports `usage.input_tokens_details.cached_tokens`, both the
completed response and `llm:response` event retain that count. A reported zero is
a cache miss; omitted or invalid counts remain unknown. This reporting does not
enable public OpenAI cache controls on the Codex or ChatGPT plan endpoints, and
does not imply per-token subscription pricing.
