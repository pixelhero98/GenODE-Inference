from __future__ import annotations

from collections.abc import Sequence


def parse_csv(value: object) -> list[str]:
    """Parse a comma-separated command-line value, omitting empty fields."""

    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value: object, *, default: Sequence[int] = ()) -> list[int]:
    """Parse comma-separated integers, using ``default`` when no values are given."""

    parsed = [int(part) for part in parse_csv(value)]
    return parsed or [int(item) for item in default]


__all__ = ["parse_csv", "parse_int_csv"]
