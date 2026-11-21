"""Provider-local selectors, following OpenAI's canonical listing contract."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase


# Recommendation policy, not aliases or entitlement: only a literal eligible
# successor base in this account's catalog supersedes these known older IDs.
# Keep this explicit; previews, dated pins and other families are independent.
_KNOWN_SUCCESSORS = {
    "gpt-6-astra": "gpt-6.2-astra",
    "gpt-6-sol": "gpt-6.2-sol",
    "gpt-6.1-sol": "gpt-6.2-sol",
}


def superseded_recommendations(ids: list[str]) -> set[str]:
    """IDs to omit from built-in recommendations/latest, never explicit pins.

    The established Codex -fast shorthand follows its base's supersession.
    A successor -fast ID alone is not evidence that its base is available.
    No new IDs, suffix normalization, version ranking or metadata are created.
    """
    available = set(ids)
    return {
        variant
        for old, successor in _KNOWN_SUCCESSORS.items()
        if successor in available
        for variant in (old, f"{old}-fast")
        if variant in available
    }


@dataclass(frozen=True)
class ModelFilters:
    """An immutable snapshot, never a reference to caller-owned lists."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    hide_dated: bool = False

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ModelFilters":
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        if set(value) - {"include", "exclude", "hide_dated"}:
            raise ValueError("unknown model filter key")

        def patterns(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(key, default)
            if not isinstance(raw, (list, tuple)) or any(
                not isinstance(pattern, str) or not pattern.strip()
                for pattern in raw
            ):
                raise ValueError(f"model filters {key} must be a list of nonblank strings")
            return tuple(raw)

        hide_dated = value.get("hide_dated", False)
        if not isinstance(hide_dated, bool):
            raise TypeError("model filters hide_dated must be a boolean")
        return cls(patterns("include", ("*",)), patterns("exclude", ()), hide_dated)

    def select(self, ids: list[str]) -> list[str]:
        selected = {
            model_id
            for model_id in ids
            if any(fnmatchcase(model_id.casefold(), pattern.casefold()) for pattern in self.include)
            and not any(fnmatchcase(model_id.casefold(), pattern.casefold()) for pattern in self.exclude)
        }
        if self.hide_dated:
            selected = {
                model_id
                for model_id in selected
                if not (
                    (match := re.search(r"-\d{4}-\d{2}-\d{2}$", model_id))
                    and model_id[: match.start()] in selected
                )
            }
        return sorted(selected)

    def as_dict(self) -> dict:
        return {
            "include": list(self.include),
            "exclude": list(self.exclude),
            "hide_dated": self.hide_dated,
        }


def selector_schema() -> dict:
    """Return a fresh schema, so capability inspection cannot change policy."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": ["*"],
                "description": "Case-insensitive, OS-independent fnmatch patterns; [] includes none.",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": [],
                "description": "Case-insensitive, OS-independent fnmatch patterns; exclusion wins.",
            },
            "hide_dated": {
                "type": "boolean",
                "default": False,
                "description": "Hide YYYY-MM-DD suffixes only when the selected undated alias exists.",
            },
        },
    }