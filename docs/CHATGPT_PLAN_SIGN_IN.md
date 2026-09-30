# ChatGPT plan sign-in

The provider owns OAuth credentials, validated identity, refresh/revocation,
account-specific models, and Responses transport. The host app owns the account
picker, explicit connection IDs, consent UI, and safe status/error presentation.
Both transports are implemented by this provider; no application or peer module
is an implementation dependency.

## Two explicit modes

- `auth_mode: chatgpt_codex` **(default)**: Codex-compatible device authorization
  and `chatgpt.com/backend-api/codex/responses`. Existing configurations without
  `auth_mode` retain this behavior. `legacy_codex` remains an accepted alias;
  mounted metadata always reports the canonical `chatgpt_codex` name.
- `auth_mode: chatgpt_plan`: the documented Sign in with ChatGPT browser flow.
  Uses the selected account's ChatGPT plan permission and the public OpenAI
  Responses API at `api.openai.com/v1/responses`.

The kernel's `ProviderInfo.config_fields` contract exposes `auth_mode` as a
choice, so configuration hosts can present both modes without provider-specific
configuration logic. The optional `token_file_path` field has no shared default:
omitting it selects `~/.amplifier/openai-chatgpt-oauth.json` for Codex or
`~/.amplifier/chatgpt-plan/default.json` for plan mode. An explicitly empty path
is invalid. A custom path must belong to the selected mode. When changing modes,
omit the old path or choose a new file; login will not overwrite other-mode
credentials. The plan-only fields `host_file_path` and `app_name` are also
declared as optional configuration fields.

Credentials cannot cross modes. A failed new sign-in never replaces existing
credentials. When migrating an existing Codex connection, use a new plan file
and commit the app's configuration only after sign-in succeeds. There is no
fallback to an API key, another account, another billing path, or the legacy
backend when plan usage fails.

## Local sign-in

Install this provider in your host's Python environment, or install the standalone
command with `uv tool install git+https://github.com/microsoft/amplifier-module-provider-openai-chatgpt`.
Then run:

```sh
amplifier-chatgpt-auth login \
  --token-file ~/.amplifier/chatgpt-plan/personal.json \
  --host-file ~/.amplifier/chatgpt-plan/host.json \
  --app-name 'Your application name'
```

The command starts an HTTP listener on **127.0.0.1**, chooses an available port,
and opens the system browser. Its callback path is `/auth/callback`. A browser
on another computer cannot use this callback. Complete the authorization within
three minutes. Status output contains safe identity fields only, never tokens.

Use a distinct credential path for each account/workspace connection, even when
email addresses match. Reconnect using the same path and host file; the issued
client registration is reused. The command verifies the new identity matches
the saved account before replacing credentials. First registration saves its
issued client ID before code exchange, so an expired code does not lose it.

Configure this exact provider instance:

```yaml
module: provider-openai-chatgpt
config:
  auth_mode: chatgpt_plan
  token_file_path: ~/.amplifier/chatgpt-plan/personal.json
  host_file_path: ~/.amplifier/chatgpt-plan/host.json
  app_name: Your application name
  login_on_mount: false
```

A successful identity sign-in can have `plan_enabled: false`. In that case,
inference remains disabled. Only when the user explicitly chooses to enable plan
usage, run login again with `--enable-plan` to request consent. A routine reconnect
does not force consent. Users can review and manage usage at
[ChatGPT Settings → Usage](https://chatgpt.com/settings/usage).

## Self-hosted server

Complete OAuth on the computer running the browser. The documented flow does
not provide a new device grant for this mode. Transfer the selected protected
credential record over SSH, then import it into the remote instance's explicit
provider path:

```sh
# On the server, create its own stable identity before import:
amplifier-chatgpt-auth host-id --host-file ~/.amplifier/chatgpt-plan/host.json

# On the local machine, transfer to a private staging directory on your server.
# Replace YOUR_SERVER with your SSH host; do not put tokens in shell arguments.
ssh YOUR_SERVER 'umask 077; mkdir -p ~/.amplifier/chatgpt-plan/import'
scp ~/.amplifier/chatgpt-plan/personal.json YOUR_SERVER:~/.amplifier/chatgpt-plan/import/personal.json

# On the server, using the exact path configured for this provider instance:
amplifier-chatgpt-auth import \
  --source-file ~/.amplifier/chatgpt-plan/import/personal.json \
  --token-file ~/.amplifier/chatgpt-plan/personal.json \
  --host-file ~/.amplifier/chatgpt-plan/host.json
rm ~/.amplifier/chatgpt-plan/import/personal.json
```

The import validates the signed identity and retains the server's own host ID.
Let the server own subsequent refreshes of the transferred session; do not keep
using the same rotating refresh token from both machines. Transfer is a shared
credential session: host-specific revocation/attribution for transferred sessions
is not yet supported by OpenAI. Use separate sign-ins when independent sessions
are needed. Never upload the credential file through chat or paste it into logs.

## Host integration contract

From `amplifier_module_provider_openai_chatgpt.plan_auth`:

- `await login(token_file_path=..., host_file_path=..., app_name=...,
  print_fn=..., timeout=180, request_plan_permission=False)` returns credentials.
  Keep this result in the trusted runtime. `print_fn` receives an authorization
  URL without an ID-token hint; there is no callback URL/code paste workflow.
  The neutral default name is `Amplifier ChatGPT provider`. Hosts should pass
  their actual application name, either here or through provider config
  `app_name`, which `provider.login()` forwards. This name is not a routing key.
  Hosts that need cancellation/edit safety can pass `source_token_file_path`
  (the active plan profile), a fresh `token_file_path` candidate, and a stable
  per-connection `registration_file_path`. The provider snapshots source data
  under its own lock; it never modifies the active source file. The registration
  file retains only the issued client/host mapping across failed attempts.
  Commit the candidate path in host settings only after the attempt succeeds
  and the connection configuration still matches the initiating user action.
  A separate consent lock keeps inference/refresh available during browser wait.
- `auth_status(path)` returns only `auth_mode`, `authenticated`, `plan_enabled`,
  `plan_permission_granted`,
  `email`, `client_id`, and `subject`. The identity/profile selection remains a
  host decision. An expired renewable session can still report connected;
  `ensure_tokens` checks/refreshes the actual credentials before requests.
  Expired nonrenewable profiles report unauthenticated so explicit login can
  repair them. Explicit `provider.login()` requests plan consent again for a
  saved identity without that grant; it does not force consent for routine
  reconnects with a previously granted scope. No background operation asks for
  new consent.
  `provider.login()` returns true only when plan inference is ready, so generic
  login UIs do not announce identity-only consent as a usable connection.
- `await ensure_tokens(path)` checks permission and refreshes near expiry under
  a cross-process profile lock. Access token, rotating refresh token, scope, and
  expiry are saved together with an atomic owner-only file replacement.
- `await logout(path)` attempts revocation with bounded retries, clears tokens,
  retains registration and host mapping, and returns `revocation_confirmed`.
  If false, tell the user local sign-out succeeded but remote revocation could
  not be confirmed; offer ChatGPT Settings to disconnect the app.
- `await import_credentials(source_file, token_file_path=..., host_file_path=...)`
  imports a securely transferred profile without replacing the server host ID.
- `provider.get_info().defaults.auth_mode` identifies the mounted mode.

Plan workers never launch a browser on mount. Explicit account setup precedes
worker initialization. Model discovery uses only that account's `/v1/models`
entries with `visibility: list`, preserves server ordering/display names, and
never returns a static Codex catalog on failure. Requests use literal model slugs
and always `store: false`, `stream: true` at `/v1/responses`. A completed response
requires `response.completed`; incomplete/failed/interrupted streams remain
failures. Plan usage limit errors are not retried automatically and do not change
billing. Request IDs and structured error codes/parameters remain available for
host diagnostics.

## Plan request limits

Local function tools are sent in a developer-role `additional_tools` input item
before conversation history. Function names, arguments, and call IDs are kept
unchanged across the response and next tool-result request. Flat top-level
function tools are not sent to the plan route. Hosted `web_search` can remain
in top-level `tools`, subject to model and account policy. This adapter does not
currently translate custom or namespaced tool-call histories; those definitions
fail clearly instead of being dropped.

The typed plan adapter currently accepts text/thinking and function-call/result
history. Other typed content blocks, including images, are rejected explicitly
instead of silently disappearing. This adapter limit is narrower than the
service's model-dependent image/file support; adding that conversion is separate
work.

The plan preview rejects stored/background conversations and several ordinary
Responses parameters. Explicit unsupported parameters, including `temperature`,
`top_p`, `max_output_tokens`, `metadata`, and `previous_response_id`, raise a
nonretryable `InvalidRequestError` with `error_code` and `error_param`. This
applies to non-null request values and `extra_request_params` wire fields.
Default unset kernel fields are not treated as requests for those capabilities.
Callers that depend on enforced output caps, including some internal naming,
judging, and reduced-output recovery calls, cannot use this preview adapter with
`max_output_tokens` set. The provider rejects the request rather than remove a
spending constraint. Bundle/feature compatibility must be assessed separately.
The provider enforces `store: false` and `stream: true` and converts later system
messages to developer messages. Supplying tool lists in both a typed request and
`extra_request_params` is rejected rather than replacing a list silently.

Image generation, native computer use, hosted MCP/connectors, Code Interpreter,
file search, `tool_search`, and `programmatic_tool_calling` are unavailable on the
plan route. These restrictions are specific to `chatgpt_plan`; this change does
not alter the Codex transport or establish new voice, audio, or realtime support
for either connection.

## Validation boundaries and official sources

Tests cover protocol contracts, signed JWT rejection, callback binding, rotation,
account isolation, wire routes, catalogs, and terminal stream errors without real
credentials. A real user sign-in and completed inference are separate acceptance
steps; mocked tests do not establish account eligibility or plan availability.

- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Accounts and sessions](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Self-hosted VMs](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms)
- [Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [Additional tool input items](https://developers.openai.com/api/docs/guides/tools-tool-search#add-tools-at-a-specific-point-in-the-input)
