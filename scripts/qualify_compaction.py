"""Bounded synthetic opt-in qualification; never logs or persists opaque state.

Usage: .venv/bin/python scripts/qualify_compaction.py --output /path/results.json
No login, token refresh, host changes, saved conversations or tool execution.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

from amplifier_core import llm_errors
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
)
from amplifier_module_provider_openai_chatgpt.oauth import (
    load_tokens,
    is_token_valid,
    TOKEN_FILE_PATH,
)
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

MODEL = "gpt-5.6-sol"


class ProbeProvider(ChatGPTProvider):
    async def _ensure_valid_tokens(self):
        # Qualification must not refresh or change the account's live store.
        if not is_token_valid(self._tokens):
            raise llm_errors.AuthenticationError(
                "Probe credential unavailable or expired", provider=self.name
            )


def synthetic_fixture(long=False):
    messages = [
        Message(
            role="system",
            content="This is a synthetic compaction qualification fixture. Never call a tool. Answer questions only from the fixture facts, respecting corrections.",
        ),
        Message(
            role="developer",
            content="Reply with one JSON object with codename, port, color, unresolved, overlay. Unresolved must be the exact phrase offline operation. Do not expose any internal state.",
        ),
        Message(
            role="user",
            content="Immutable codename ORCHID-LANTERN-624. Port is initially 8799. Color initially amber. Offline operation remains an unresolved requirement.",
        ),
    ]
    if long:
        for i in range(36):
            messages += [
                Message(
                    role="user",
                    content=f"Synthetic work item {i}: document a deterministic validation example. "
                    + "Archive this ordinary placeholder detail; it introduces no new project facts. "
                    * 14,
                ),
                Message(
                    role="assistant", content=f"Documented synthetic work item {i}."
                ),
            ]
            if i == 17:
                messages.append(
                    Message(
                        role="user",
                        content="Correction: the port is now 8801, superseding 8799.",
                    )
                )
        messages += [
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="fixture_lookup", name="lookup", input={"key": "fixture"}
                    )
                ],
            ),
            Message(
                role="tool",
                content=[
                    ToolResultBlock(
                        tool_call_id="fixture_lookup",
                        output="Synthetic paired tool result; no real tool was executed.",
                    )
                ],
            ),
            Message(
                role="user",
                content="Final correction: color is blue, superseding amber. Port stays 8801. Offline operation remains unresolved.",
            ),
        ]
    else:
        messages.append(
            Message(
                role="user",
                content="Correction: port 8801 and color blue. Offline operation remains unresolved.",
            )
        )
    return ChatRequest(
        model=MODEL,
        messages=messages,
        tools=[
            ToolSpec(
                name="lookup",
                description="Synthetic fixture tool that must never be called",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ],
    )


def safe_error(exc):
    message = str(exc).lower()
    return {
        "type": type(exc).__name__,
        "status": getattr(exc, "status_code", None),
        "context_limit": "context" in message
        and ("limit" in message or "length" in message),
        "mentions_trigger": "compaction_trigger" in message,
        "mentions_unsupported": any(
            x in message
            for x in [
                "unsupported",
                "not supported",
                "unknown variant",
                "invalid value",
            ]
        ),
        "missing_checkpoint": "exactly one checkpoint" in message,
        "model_mismatch": "model differs" in message,
        "missing_success_terminal": "successful terminal" in message,
    }


async def qualify(standalone_only=False):
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "endpoint": "https://chatgpt.com/backend-api/codex/responses",
        "model_calls": 0,
        "automatic_eligible": False,
        "admission_count": "unavailable",
        "saved_conversations_read": 0,
        "tools_executed": 0,
        "ciphertext_persisted": False,
        "cases": [],
    }
    tokens = load_tokens()
    report["credentials"] = (
        "valid-provider-credential"
        if is_token_valid(tokens)
        else "unavailable-or-expired"
    )
    if not is_token_valid(tokens):
        report["status"] = "blocked-credentials"
        return report
    tokens = deepcopy(tokens)
    tokens.pop("refresh_token", None)

    def make():
        return ProbeProvider(
            {
                "experimental_compaction": True,
                "default_model": MODEL,
                "timeout": 40,
                "raw": False,
                "token_file_path": "/nonexistent/qualification-no-refresh",
            },
            tokens=deepcopy(tokens),
        )

    async def call(name, operation, request):
        if report["model_calls"] >= 10:
            raise RuntimeError("Synthetic qualification request budget exhausted")
        report["model_calls"] += 1
        before = request.model_dump_json()
        try:
            value = await asyncio.wait_for(operation(request), timeout=45)
            case = {
                "case": name,
                "passed": True,
                "source_unchanged": request.model_dump_json() == before,
            }
            report["cases"].append(case)
            return value, case
        except BaseException as exc:
            report["cases"].append(
                {
                    "case": name,
                    "passed": False,
                    "error": safe_error(exc),
                    "source_unchanged": request.model_dump_json() == before,
                }
            )
            raise

    async def compact(name, req):
        result, case = await call(name, make().compact_checkpoint, req)
        usage = deepcopy(result.raw_usage)
        attribution = usage.pop("attribution", None)
        if isinstance(attribution, dict):
            case["usage_attribution"] = {
                "item_count": len(attribution.get("items", {})),
                "request_fields": attribution.get("request_fields", {}),
            }
        case.update(
            {
                "terminal_event": result.terminal_event,
                "operation_usage": usage,
                "canonical_window": result.canonical_window,
                "json_roundtrip": False,
            }
        )
        checkpoint = Message.model_validate_json(result.checkpoint.model_dump_json())
        case["json_roundtrip"] = checkpoint == result.checkpoint
        return checkpoint

    async def continuation(name, req, checkpoint, retain, cycle):
        # Experiment-owned retention policy: caller selects these roles. It is
        # intentionally not a generic or qualified provider continuation rule.
        retained = (
            [
                m.model_copy(deep=True)
                for m in req.messages
                if m.role in {"system", "developer", "user"}
            ]
            if retain
            else [
                m.model_copy(deep=True)
                for m in req.messages
                if m.role in {"system", "developer"}
            ]
        )
        overlay = f"probe-overlay-{cycle}"
        suffix = Message(
            role="user",
            content=f"Current overlay is {overlay}. Give all five fixture facts now.",
        )
        candidate = req.model_copy(update={"messages": [*retained, checkpoint, suffix]})
        response, case = await call(name, make().complete, candidate)
        text = "".join(getattr(b, "text", "") for b in response.content)
        expected = ["ORCHID-LANTERN-624", "8801", "blue", "offline operation", overlay]
        case["fact_checks"] = {
            "codename": expected[0] in text,
            "corrected_port": expected[1] in text and "8799" not in text,
            "corrected_color": expected[2] in text.lower()
            and "amber" not in text.lower(),
            "unresolved": expected[3] in text.lower(),
            "current_overlay": expected[4] in text,
        }
        case["tools_requested"] = bool(response.tool_calls)
        case["passed"] = all(case["fact_checks"].values()) and not response.tool_calls
        case["observed_continuation_usage"] = (
            response.usage.model_dump(exclude_none=True) if response.usage else None
        )
        # Observed usage is a post-request observation, NOT a preflight count.
        case["admission_proven"] = False
        if not case["passed"]:
            raise RuntimeError("Synthetic fact qualification failed")
        return [*retained, checkpoint, suffix, Message(role="assistant", content=text)]

    try:
        req = synthetic_fixture()
        checkpoint = await compact("standalone-trigger", req)
        await continuation(
            "standalone-retained-continuation",
            req,
            checkpoint,
            True,
            "standalone-retained",
        )
        await continuation(
            "standalone-checkpoint-only-continuation",
            req,
            checkpoint,
            False,
            "standalone-checkpoint-only",
        )
        if standalone_only:
            report["status"] = "standalone-passed-admission-unqualified"
            return report
        req = synthetic_fixture(long=True)
        report["long_fixture_message_count"] = len(req.messages)
        original = req.model_dump_json()
        for cycle in range(3):
            checkpoint = await compact(f"long-cycle-{cycle + 1}-compact", req)
            history = await continuation(
                f"long-cycle-{cycle + 1}-continue", req, checkpoint, True, cycle + 1
            )
            req = req.model_copy(
                update={
                    "messages": [
                        *history,
                        Message(
                            role="user",
                            content=f"New synthetic turn after cycle {cycle + 1}; existing corrections and unresolved requirement remain in force.",
                        ),
                    ]
                }
            )
        report["original_long_fixture_preserved"] = (
            synthetic_fixture(long=True).model_dump_json() == original
        )
        report["status"] = "synthetic-three-cycles-passed-admission-unqualified"
    except BaseException as exc:
        report["status"] = "blocked-or-failed-live-qualification"
        report["stop_reason"] = safe_error(exc)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--standalone-only", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    credential_path = Path(TOKEN_FILE_PATH).expanduser()
    before = (
        hashlib.sha256(credential_path.read_bytes()).hexdigest()
        if credential_path.exists()
        else None
    )
    try:
        report = asyncio.run(
            asyncio.wait_for(qualify(args.standalone_only), timeout=240)
        )
    except BaseException as exc:
        report = {"status": "probe-aborted", "error": safe_error(exc)}
    after = (
        hashlib.sha256(credential_path.read_bytes()).hexdigest()
        if credential_path.exists()
        else None
    )
    report["live_token_store_unchanged"] = before == after
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_calls": report.get("model_calls"),
                "cases": len(report.get("cases", [])),
                "live_token_store_unchanged": report["live_token_store_unchanged"],
            }
        )
    )


if __name__ == "__main__":
    main()
