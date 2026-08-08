from __future__ import annotations

from typing import Any, Mapping


def checkpoint_scope_from_row(row: Mapping[str, Any], *, empty_label: str = "") -> str:
    checkpoint_id = str(row.get("checkpoint_id", "") or "").strip()
    if checkpoint_id:
        return checkpoint_id
    for field in ("checkpoint_step", "train_steps", "otflow_train_steps"):
        value = str(row.get(field, "") or "").strip()
        if value:
            return f"checkpoint_step:{value}"
    return str(empty_label)
