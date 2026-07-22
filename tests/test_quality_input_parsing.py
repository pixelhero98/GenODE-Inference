from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from genode.distillation.evaluation import (
    read_candidate_catalog,
    read_quality_contexts,
    read_quality_protocol,
    read_quality_rows,
)
from genode.distillation import evaluation as evaluation_module


def test_candidate_catalog_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "candidate-catalog.json"
    path.write_text(
        '[{"method":"flow_map","method":"gipo"}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'method'"):
        read_candidate_catalog(path)


def test_pipeline_protocol_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(
        '{"protocol_hash":"' + "0" * 64 + '",'
        '"scenario_key":"cryptos","scenario_key":"electricity",'
        '"flow_map":{}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'scenario_key'"):
        read_quality_protocol(path, scenario_key="cryptos")


def test_quality_evaluation_rejects_input_changed_after_identity_capture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quality-rows.csv"
    path.write_text("before\n", encoding="utf-8")
    paths = {"quality_rows": path}
    identities = evaluation_module._capture_input_file_hashes(paths)

    path.write_text("after\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed while it was being read"):
        evaluation_module._require_input_files_unchanged(paths, identities)


def test_quality_rows_reject_duplicate_csv_headers(tmp_path: Path) -> None:
    path = tmp_path / "quality-rows.csv"
    path.write_text("split_phase,method,method\nlocked_test,gipo,fixed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="headers must be unique"):
        read_quality_rows(path)


def test_quality_contexts_reject_duplicate_npz_members(tmp_path: Path) -> None:
    path = tmp_path / "quality-contexts.npz"
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(["context-0"], dtype=np.str_))
    member = buffer.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("context_ids.npy", member)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("context_ids.npy", member)

    with pytest.raises(ValueError, match="duplicate archive members"):
        read_quality_contexts(path)
