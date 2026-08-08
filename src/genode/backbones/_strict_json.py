from __future__ import annotations

import json


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid JSON number {value}.")


def loads_strict_json(text: str, *, label: str) -> object:
    if not isinstance(text, str):
        raise TypeError(f"{label} must be supplied as text.")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not strict JSON.") from exc


__all__ = ["loads_strict_json"]
