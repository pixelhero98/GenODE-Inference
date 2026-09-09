from __future__ import annotations

import json
from pathlib import Path

import pytest

from genode.gico import train_gico
from genode.gico.policy import save_context_embedding_table
from tests.test_unified_gico_rewards import reference_evidence


@pytest.mark.parametrize("weight", [0.01, 0.05, 0.1])
def test_config_relative_paths_and_dry_run_validate_without_fitting(tmp_path, monkeypatch, weight):
    rows, contexts = reference_evidence()
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    save_context_embedding_table(tmp_path / "contexts.npz", contexts)
    location = tmp_path / "training.json"
    location.write_text(
        json.dumps(
            {
                "rows": "rows.jsonl",
                "contexts": "contexts.npz",
                "output": "artifact",
                "student_kind": "both",
                "teacher_score_weight": weight,
            }
        )
    )
    config = train_gico.load_config(location)
    assert Path(config["rows"]) == tmp_path / "rows.jsonl"
    assert Path(config["contexts"]) == tmp_path / "contexts.npz"
    assert Path(config["output"]) == tmp_path / "artifact"

    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run must not fit")

    monkeypatch.setattr(train_gico, "fit", forbidden)
    result = train_gico.run_config(config, dry_run=True)
    assert result["dry_run"] is True
    assert result["paired_cells"] > 0
    assert result["teacher_score_weight"] == weight
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"rows": "r", "contexts": "c", "output": "o", "obsolete_teacher_architecture": "mlp"},
    ],
)
def test_config_rejects_missing_or_retired_options(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="documented GICO options"):
        train_gico.load_config(path)


@pytest.mark.parametrize("option,value", [("--student-kind", "discrete"), ("--teacher-score-weight", "0.2")])
def test_cli_rejects_unsupported_choices(option, value):
    with pytest.raises(SystemExit):
        train_gico.build_argparser().parse_args(["--config", "training.json", option, value])
