from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from genode.gipo.policy import context_id_from_row, load_context_embedding_table, save_context_embedding_table


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


def _require_no_locked(label: str, rows: Sequence[Mapping[str, object]]) -> None:
    locked = [row for row in rows if str(row.get("split_phase", row.get("split", ""))) == "locked_test"]
    if locked:
        raise ValueError(f"{label} contains locked_test rows: {len(locked)}")


def _require_nfes(label: str, rows: Sequence[Mapping[str, object]], expected: set[int]) -> None:
    found = {int(row["target_nfe"]) for row in rows}
    if found != expected:
        raise ValueError(f"{label} target NFEs {sorted(found)} != expected {sorted(expected)}")


def _paths(section_root: Path, split: str, prefix: str) -> tuple[Path, Path, Path]:
    base = section_root / f"{prefix}_{split}"
    return (
        base / f"{prefix}_{split}_context_rows.csv",
        base / f"{prefix}_{split}_context_embeddings.npz",
        base / f"{prefix}_{split}_rows.csv",
    )


def merge_tfv1_inputs(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    standard_nfes = {int(value) for value in str(args.standard_nfes).split(",") if value}
    unseen_nfes = {int(value) for value in str(args.unseen_nfes).split(",") if value}

    standard = root / "standard_inputs"
    unseen = root / "unseen_nfe_inputs"
    cal = root / "calibration_inputs"
    cal.mkdir(parents=True, exist_ok=True)
    unseen.mkdir(parents=True, exist_ok=True)

    standard_context_csvs: list[Path] = []
    standard_embedding_npzs: list[Path] = []
    locked_context_csvs: list[Path] = []
    locked_embedding_npzs: list[Path] = []
    for split in ("train", "validation"):
        for prefix in ("fixed", "ser"):
            context_path, embedding_path, _ = _paths(standard, split, prefix)
            standard_context_csvs.append(context_path)
            standard_embedding_npzs.append(embedding_path)
    for prefix in ("fixed", "ser"):
        context_path, embedding_path, _ = _paths(standard, "locked", prefix)
        locked_context_csvs.append(context_path)
        locked_embedding_npzs.append(embedding_path)

    standard_rows = _read_rows(standard_context_csvs)
    locked_rows = _read_rows(locked_context_csvs)
    _require_no_locked("standard calibration", standard_rows)
    _require_nfes("standard calibration", standard_rows, standard_nfes)
    _require_nfes("standard locked", locked_rows, standard_nfes)

    standard_embeddings = _merge_embeddings(standard_embedding_npzs)
    locked_embeddings = _merge_embeddings(locked_embedding_npzs)
    if sorted({context_id_from_row(row) for row in standard_rows} - set(standard_embeddings)):
        raise KeyError("standard calibration embeddings are missing contexts")
    if sorted({context_id_from_row(row) for row in locked_rows} - set(locked_embeddings)):
        raise KeyError("standard locked embeddings are missing contexts")

    _write_rows(cal / "context_calibration_rows.csv", standard_rows)
    _write_rows(cal / "locked_context_rows.csv", locked_rows)
    save_context_embedding_table(cal / "context_calibration_embeddings.npz", standard_embeddings)
    save_context_embedding_table(cal / "locked_context_embeddings.npz", locked_embeddings)

    unseen_manifest: dict[str, object] = {}
    for split in ("train", "validation", "locked"):
        context_csvs = []
        embedding_npzs = []
        fixed_rows_path = None
        ser_rows_path = None
        for prefix in ("fixed", "ser"):
            context_path, embedding_path, rows_path = _paths(unseen, split, prefix)
            context_csvs.append(context_path)
            embedding_npzs.append(embedding_path)
            if prefix == "fixed":
                fixed_rows_path = rows_path
            else:
                ser_rows_path = rows_path
        rows = _read_rows(context_csvs)
        if split != "locked":
            _require_no_locked(f"unseen {split}", rows)
        _require_nfes(f"unseen {split}", rows, unseen_nfes)
        embeddings = _merge_embeddings(embedding_npzs)
        missing = sorted({context_id_from_row(row) for row in rows} - set(embeddings))
        if missing:
            raise KeyError(f"unseen {split} embeddings missing contexts: {missing[:8]}")
        _write_rows(unseen / f"{split}_context_rows.csv", rows)
        save_context_embedding_table(unseen / f"{split}_context_embeddings.npz", embeddings)
        unseen_manifest[f"{split}_context_rows"] = str(unseen / f"{split}_context_rows.csv")
        unseen_manifest[f"{split}_context_embeddings"] = str(unseen / f"{split}_context_embeddings.npz")
        unseen_manifest[f"{split}_fixed_reference_rows"] = str(fixed_rows_path)
        unseen_manifest[f"{split}_ser_reference_rows"] = str(ser_rows_path)
        unseen_manifest[f"{split}_context_row_count"] = len(rows)
        unseen_manifest[f"{split}_context_count"] = len({context_id_from_row(row) for row in rows})

    manifest = {
        "artifact": "gipo_teacher_student_tfv1_inputs_manifest",
        "locked_test_used_for_selection": False,
        "standard_target_nfe_values": sorted(standard_nfes),
        "unseen_target_nfe_values": sorted(unseen_nfes),
        "calibration_rows": str(cal / "context_calibration_rows.csv"),
        "calibration_embeddings": str(cal / "context_calibration_embeddings.npz"),
        "locked_rows": str(cal / "locked_context_rows.csv"),
        "locked_embeddings": str(cal / "locked_context_embeddings.npz"),
        "standard_ser_schedule_summary": str(standard / "ser_reference" / "ser_ptg_schedule_summary.json"),
        "unseen_ser_schedule_summary": str(unseen / "ser_reference" / "ser_ptg_schedule_summary.json"),
        "standard_calibration_row_count": len(standard_rows),
        "standard_calibration_context_count": len({context_id_from_row(row) for row in standard_rows}),
        "standard_locked_row_count": len(locked_rows),
        "standard_locked_context_count": len({context_id_from_row(row) for row in locked_rows}),
        **unseen_manifest,
    }
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "tfv1_inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge GIPO TFv1 continuous-v3 verification inputs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--standard_nfes", default="4,8,12")
    parser.add_argument("--unseen_nfes", default="6,10,14,16")
    return parser


def main() -> None:
    payload = merge_tfv1_inputs(build_argparser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
