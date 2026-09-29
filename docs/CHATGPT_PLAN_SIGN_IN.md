# ChatGPT plan sign-in

The provider owns OAuth credentials, validated identity, refresh/revocation,
account-specific models, and Responses transport. The host app owns the account
picker, explicit connection IDs, consent UI, and safe status/error presentation.
Nothing here requires Amplifier app-cli or Codex to be installed.

## Two explicit modes

- `auth_mode: chatgpt_plan`: the documented Sign in with ChatGPT flow. Uses the
  selected account's ChatGPT plan permission and the public OpenAI Responses API.
- `auth_mode: legacy_codex`: the existing Codex-compatible device authorization
  and backend transport. Existing configurations without `auth_mode` retain this
  behavior. This compatibility mode is not the new ChatGPT plan flow.

Credentials cannot cross modes. A failed new sign-in never replaces existing
credentials. When migrating an existing legacy connection, use a new plan file
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
  --app-name 'Amplifier Unified'
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
  Hosts that need cancellation/edit safety can pass `source_token_file_path`
  (the active plan profile), a fresh `token_file_path` candidate, and a stable
  per-connection `registration_file_path`. The provider snapshots source data
  under its own lock; it never modifies the active source file. The registration
  file retains only the issued client/host mapping across failed attempts.
  Commit the candidate path in host settings only after the attempt succeeds
  and the connection configuration still matches the initiating user action.
  A separate consent lock keeps inference/refresh available during browser wait.
- `auth_status(path)` returns only `auth_mode`, `authenticated`, `plan_enabled`,
  `email`, `client_id`, and `subject`. The identity/profile selection remains a
  host decision. An expired renewable session can still report connected;
  `ensure_tokens` checks/refreshes the actual credentials before requests.
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
