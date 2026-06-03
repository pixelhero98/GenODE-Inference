from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from genode.gipo.policy import (
    context_id_from_row,
    context_pair_key,
    load_context_embedding_table,
    save_context_embedding_table,
)
from genode.gipo.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, EXPERIMENTAL_REVERSED_SCHEDULE_KEYS

FIXED_DIVERSITY_KEYS: tuple[str, ...] = tuple(BASELINE_SCHEDULE_KEYS) + tuple(EXPERIMENTAL_REVERSED_SCHEDULE_KEYS)
SUPPORT_DIVERSITY_KEYS: tuple[str, ...] = FIXED_DIVERSITY_KEYS + (SER_PTG_SCHEDULE_KEY,)


def _read_rows(paths: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as fh:
            rows.extend(dict(row) for row in csv.DictReader(fh))
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key_text = str(key)
            if key_text not in seen:
                seen.add(key_text)
                fields.append(key_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _merge_embeddings(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        table = load_context_embedding_table(path)
        for key, value in table.items():
            arr = np.asarray(value, dtype=np.float32)
            if key in merged and not np.allclose(merged[key], arr, atol=1e-5):
                raise ValueError(f"Conflicting context embedding for {key} from {path}")
            merged[key] = arr
    return merged


def _schedule_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["scheduler_key"]) for row in rows).items()))


def _validate_splits(label: str, rows: Sequence[Mapping[str, object]], *, allowed_locked: bool) -> None:
    split_values = {str(row.get("split_phase", row.get("split", ""))) for row in rows}
    if allowed_locked:
        if split_values != {"locked_test"}:
            raise ValueError(f"{label} must contain locked_test rows only, got {sorted(split_values)}")
    elif "locked_test" in split_values:
        raise ValueError(f"{label} includes locked_test rows.")


def _validate_complete_support(label: str, rows: Sequence[Mapping[str, object]], support_keys: Sequence[str]) -> None:
    support = tuple(str(key) for key in support_keys)
    observed = {str(row["scheduler_key"]) for row in rows}
    missing_keys = sorted(set(support) - observed)
    extra_keys = sorted(observed - set(support))
    if missing_keys or extra_keys:
        raise ValueError(f"{label} support mismatch: missing={missing_keys} extra={extra_keys}")
    grouped: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[context_pair_key(row, pair_on_seed=True)].append(row)
    bad: list[dict[str, object]] = []
    for key, group in grouped.items():
        counts = Counter(str(row["scheduler_key"]) for row in group)
        if any(counts.get(schedule, 0) != 1 for schedule in support):
            bad.append({"group": str(key), "counts": dict(sorted(counts.items()))})
            if len(bad) >= 8:
                break
    if bad:
        raise ValueError(f"{label} rows are missing one-row-per-support coverage: {bad}")


def merge_density_diversity_inputs(root: Path) -> dict:
    cal = root / "calibration_inputs"
    calibration_csvs = [
        cal / "fixed_train_tuning" / "fixed_train_context_rows.csv",
        cal / "fixed_validation_tuning" / "fixed_validation_context_rows.csv",
        cal / "ser_train_tuning" / "ser_train_context_rows.csv",
        cal / "ser_validation_tuning" / "ser_validation_context_rows.csv",
    ]
    locked_csvs = [
        cal / "fixed_locked_test" / "fixed_locked_context_rows.csv",
        cal / "ser_locked_test" / "ser_locked_context_rows.csv",
    ]
    calibration_npzs = [
        cal / "fixed_train_tuning" / "fixed_train_context_embeddings.npz",
        cal / "fixed_validation_tuning" / "fixed_validation_context_embeddings.npz",
        cal / "ser_train_tuning" / "ser_train_context_embeddings.npz",
        cal / "ser_validation_tuning" / "ser_validation_context_embeddings.npz",
    ]
    locked_npzs = [
        cal / "fixed_locked_test" / "fixed_locked_context_embeddings.npz",
        cal / "ser_locked_test" / "ser_locked_context_embeddings.npz",
    ]

    calibration_rows = _read_rows(calibration_csvs)
    locked_rows = _read_rows(locked_csvs)
    _validate_splits("calibration", calibration_rows, allowed_locked=False)
    _validate_splits("locked", locked_rows, allowed_locked=True)
    _validate_complete_support("calibration", calibration_rows, SUPPORT_DIVERSITY_KEYS)
    _validate_complete_support("locked", locked_rows, SUPPORT_DIVERSITY_KEYS)

    calibration_embeddings = _merge_embeddings(calibration_npzs)
    locked_embeddings = _merge_embeddings(locked_npzs)
    missing_calibration_embeddings = sorted({context_id_from_row(row) for row in calibration_rows} - set(calibration_embeddings))
    missing_locked_embeddings = sorted({context_id_from_row(row) for row in locked_rows} - set(locked_embeddings))
    if missing_calibration_embeddings:
        raise KeyError(f"Calibration embeddings missing contexts: {missing_calibration_embeddings[:8]}")
    if missing_locked_embeddings:
        raise KeyError(f"Locked embeddings missing contexts: {missing_locked_embeddings[:8]}")

    _write_rows(cal / "context_calibration_rows.csv", calibration_rows)
    _write_rows(cal / "locked_context_rows.csv", locked_rows)
    save_context_embedding_table(cal / "context_calibration_embeddings.npz", calibration_embeddings)
    save_context_embedding_table(cal / "locked_context_embeddings.npz", locked_embeddings)

    manifest = {
        "artifact": "density_diversity_calibration_inputs_manifest",
        "locked_test_used_for_selection": False,
        "fixed_diversity_schedule_keys": list(FIXED_DIVERSITY_KEYS),
        "support_schedule_keys": list(SUPPORT_DIVERSITY_KEYS),
        "reversed_schedule_keys": list(EXPERIMENTAL_REVERSED_SCHEDULE_KEYS),
        "calibration_rows": str(cal / "context_calibration_rows.csv"),
        "calibration_embeddings": str(cal / "context_calibration_embeddings.npz"),
        "locked_rows": str(cal / "locked_context_rows.csv"),
        "locked_embeddings": str(cal / "locked_context_embeddings.npz"),
        "fixed_locked_baseline_rows": str(cal / "fixed_locked_test" / "fixed_locked_rows.csv"),
        "ser_locked_comparator_rows": str(cal / "ser_locked_test" / "ser_locked_rows.csv"),
        "ser_schedule_summary": str(cal / "ser_reference" / "ser_ptg_schedule_summary.json"),
        "calibration_row_count": len(calibration_rows),
        "calibration_context_count": len({context_id_from_row(row) for row in calibration_rows}),
        "locked_row_count": len(locked_rows),
        "locked_context_count": len({context_id_from_row(row) for row in locked_rows}),
        "calibration_schedule_counts": _schedule_counts(calibration_rows),
        "locked_schedule_counts": _schedule_counts(locked_rows),
        "calibration_sources": [str(path) for path in calibration_csvs],
        "locked_sources": [str(path) for path in locked_csvs],
    }
    (cal / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge measured fixed/SER density-diversity calibration artifacts.")
    parser.add_argument(
        "--root",
        default=os.environ.get("GENODE_VERIFICATION_ROOT", "/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603"),
    )
    args = parser.parse_args()
    manifest = merge_density_diversity_inputs(Path(args.root))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
