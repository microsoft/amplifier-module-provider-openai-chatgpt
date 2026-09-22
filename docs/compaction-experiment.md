# Experimental ChatGPT checkpoints

The optional `experimental_compaction: true` provider setting enables a standalone `compact_checkpoint(ChatRequest)` mechanism and exact checkpoint replay. It is disabled by default. `supports_native_compaction()` always returns false: this experiment is not eligible for context-managed automatic native selection.

The method uses the existing OAuth-authenticated ChatGPT `/backend-api/codex/responses` stream, appending exactly one `compaction_trigger` to the caller-selected input. It requires a successful `response.completed` or `response.done` with completed status and exactly one valid encrypted checkpoint. Missing/duplicate checkpoints, malformed data, failed/incomplete/cancelled responses, mixed output, and inconsistent response identity are rejected. No model-requested tool executes and no user-visible streaming content is emitted during this operation.

The result is a `CompactionCheckpoint` with:

- `checkpoint`: an empty assistant `Message` carrying versioned `openai-chatgpt:compaction` metadata. The raw vendor item preserves unknown fields, nullable or absent IDs and encrypted content. Core `model_dump_json()` / `model_validate_json()` can persist it.
- `usage` and `raw_usage`: accounting for the compaction operation, including cache and reasoning details. Missing usage is unavailable. These are never next-request admission counts.
- `response_id` and `terminal_event`: operation receipt information.
- `continuation_kind="checkpoint_requires_retained_history"`, `canonical_window=False`, `next_request_input_tokens=None`.

The caller constructs the next `ChatRequest` from its selected retained messages, the checkpoint and the new suffix. The provider never chooses which messages to keep or discard, invents a summary instruction, reserves a context budget, installs a candidate, or falls back to another history. The canonical original transcript must remain available to the caller.

Replay requires this experiment to be enabled and exact provider/endpoint/model/account provenance. Account scope is a SHA-256 digest, never an OAuth token. A different model, route, account or metadata version fails closed. Dispatch uses a copy of persisted state. Raw diagnostic hooks redact encrypted content.

The native path rejects overrides of input, model, instructions, tools, tool choice, stream/store, previous response, conversation, context management or truncation in `extra_request_params`. It also rejects request fields this backend adapter cannot preserve: response format, temperature, top-p, max output tokens, stop and conversation ID; unsupported history content fails instead of disappearing. Request tools, instructions, supported content and tool choice are preserved. In particular there is no native summary output-token target or guaranteed output reservation.

## Qualification and limits

On 2026-09-22, an isolated synthetic probe succeeded on one existing subscription account using `gpt-5.6-sol`: a standalone compact, retained-history and checkpoint-only continuations, and three repeated compact/continue cycles beginning with 79 messages. Every cycle preserved the immutable codename, corrected port/color, unresolved requirement and current overlay. The observed terminal was `response.completed`. This is account/model-specific mechanism evidence; it is not a compatibility promise for other models or accounts, a near-limit admission proof, or a general checkpoint-only retention policy.

The retained-user-message fixture produced only modest reductions because most filler was in user messages. The small standalone fixture actually increased observed continuation input. A context manager must measure the complete candidate request and reject insufficient reductions. No authoritative subscription preflight counting endpoint was verified, and this provider intentionally offers no `request_budget` or ciphertext-token estimate. Automatic native selection stays disabled.

`tests/test_compaction.py` covers local malformed/failure/cancellation cases, no-network mismatches, envelope integrity, exact serialization, accounting, caller-owned fallback and three reconstruction cycles. Live entitlement failures, cancellation and incompatible backend model calls are not claimed: these were exercised locally, without changing an account or wasting unsupported requests.

Run the bounded live probe only as an explicit experiment:

```sh
.venv/bin/python scripts/qualify_compaction.py --output /absolute/path/results.json
```

It uses only disposable synthetic data and already-valid provider credentials held in memory, disables token refresh, never logs or saves ciphertext, performs at most ten model requests within four minutes, and stops at the first failure. `--standalone-only` checks the compact and two continuation variants in three requests. No saved conversations, host settings or installed providers are changed.

## Sources

- [Current official Codex trigger construction](https://github.com/openai/codex/blob/89a7298aff55db29f9cb2ad2430e3efe02b7a454/codex-rs/core/src/compact_remote_v2_attempt.rs#L78)
- [Current official collector and separately constructed continuation](https://github.com/openai/codex/blob/89a7298aff55db29f9cb2ad2430e3efe02b7a454/codex-rs/core/src/compact_remote_v2.rs#L441)
- [Public API compaction guide](https://developers.openai.com/api/docs/guides/compaction): public API semantics are not assumed to be a subscription-backend contract.
