from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from genode.gipo.policy import (
    context_id_from_row,
    load_context_embedding_table,
    save_context_embedding_table,
)


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


def merge_calibration_inputs(root: Path) -> dict:
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
    locked_in_calibration = [
        row for row in calibration_rows
        if str(row.get("split_phase", row.get("split", ""))) == "locked_test"
    ]
    if locked_in_calibration:
        raise ValueError(f"Calibration rows include locked_test rows: {len(locked_in_calibration)}")

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
        "artifact": "verification_calibration_inputs_manifest",
        "locked_test_used_for_selection": False,
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
        "calibration_sources": [str(path) for path in calibration_csvs],
        "locked_sources": [str(path) for path in locked_csvs],
    }
    (cal / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge reusable fixed/SER calibration row and embedding artifacts.")
    parser.add_argument(
        "--root",
        default=os.environ.get("GENODE_VERIFICATION_ROOT", "/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602"),
    )
    args = parser.parse_args()
    manifest = merge_calibration_inputs(Path(args.root))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
