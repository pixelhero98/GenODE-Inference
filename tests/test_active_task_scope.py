from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from genode.backbone_packages import FAMILY_SPECS
from genode.canonical_experiment_layout import CANONICAL_SCENARIO_KEYS, FORECAST_SCENARIO_KEYS, MOLECULE_SCENARIO_KEYS
from genode.data.otflow_experiment_plan import experiment_plan_by_key
from genode.pipeline import full_pipeline
from genode.pipeline.full_pipeline import _build_stage_commands, build_argparser, run_full_pipeline


def test_active_tasks_and_backbone_packages_are_exact():
    assert FORECAST_SCENARIO_KEYS == ("solar_energy_10m", "traffic_hourly", "weather_daily")
    assert MOLECULE_SCENARIO_KEYS == ("molecule_3d_set1", "molecule_3d_set2", "molecule_3d_set3")
    assert (*FORECAST_SCENARIO_KEYS, *MOLECULE_SCENARIO_KEYS) == CANONICAL_SCENARIO_KEYS
    assert tuple(experiment_plan_by_key()) == FORECAST_SCENARIO_KEYS
    assert set(FAMILY_SPECS) == {"temporal-extrapolation", "molecule-coord-generation"}


def test_pipeline_routes_explicit_gico_config_to_common_cli(tmp_path: Path):
    config = tmp_path / "training.json"
    config.write_text(json.dumps({"task": "forecast"}))
    args = build_argparser().parse_args(
        [
            "--scenario_key",
            "traffic_hourly",
            "--stages",
            "gico_training",
            "--gico-config",
            str(config),
            "--student-kind",
            "stochastic",
            "--teacher-score-weight",
            "0.05",
        ]
    )
    stages = _build_stage_commands(args, tmp_path)
    assert len(stages) == 1
    command = stages[0].commands[0]
    assert command[2] == "genode.gico.train_gico"
    assert command[3:] == ["--config", str(config), "--student-kind", "stochastic", "--teacher-score-weight", "0.05"]


def test_pipeline_requires_explicit_gico_config(tmp_path: Path):
    args = build_argparser().parse_args(["--scenario_key", "traffic_hourly", "--stages", "gico_training"])
    with pytest.raises(ValueError, match="requires --gico-config"):
        _build_stage_commands(args, tmp_path)


def test_pipeline_defaults_keep_locked_test_out_of_functional_validation(tmp_path: Path):
    args = build_argparser().parse_args(
        ["--scenario_key", "traffic_hourly", "--stages", "schedule_rows_seen", "--checkpoint_steps", "4000"]
    )
    stages = _build_stage_commands(args, tmp_path)
    commands = stages[0].commands
    assert len(commands) == 2
    assert [command[command.index("--split_phase") + 1] for command in commands] == [
        "train_tuning",
        "validation_tuning",
    ]
    assert all(command[command.index("--forecast_datasets") + 1] == "traffic_hourly" for command in commands)


def test_pipeline_forecast_exact_budget_and_dry_run_has_no_side_effects(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Dry run must not execute a workload")

    monkeypatch.setattr(full_pipeline.subprocess, "run", unexpected)
    destination = tmp_path / "absent"
    args = build_argparser().parse_args(
        [
            "--scenario_key",
            "traffic_hourly",
            "--stages",
            "backbone_training",
            "--run_root",
            str(destination),
            "--dry_run",
        ]
    )
    summary = run_full_pipeline(args)
    command = summary["commands"][0]["commands"][0]
    assert command[2] == "genode.training.train_backbone"
    assert command[command.index("--checkpoint_export_mode") + 1] == "exact_budget"
    assert not destination.exists()


def test_pipeline_molecule_member_paths_and_manifest_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(full_pipeline, "load_molecule_group_manifest", lambda *args: {})
    monkeypatch.setattr(
        full_pipeline,
        "trainable_molecule_group_members",
        lambda _: [
            {"stratum": "Dynamic_A", "processed_dir": "Dynamic_A"},
        ],
    )
    manifest = tmp_path / "custom-manifest.json"
    args = build_argparser().parse_args(
        [
            "--scenario_key",
            "molecule_3d_set1",
            "--stages",
            "backbone_training",
            "--molecule_group_root",
            str(tmp_path),
            "--backbone_manifest",
            str(manifest),
        ]
    )
    command = _build_stage_commands(args, tmp_path)[0].commands[0]
    assert command[2] == "genode.training.train_molecule_backbone"
    assert command[command.index("--processed_dir") + 1] == str(tmp_path / "molecule_3d_set1" / "Dynamic_A")
    assert command[command.index("--backbone_manifest") + 1] == str(manifest)
    assert "--no_prepare_data" in command


@pytest.mark.parametrize("outputs_complete", [True, False])
def test_pipeline_resume_checks_actual_schedule_outputs(tmp_path, monkeypatch, outputs_complete):
    args = build_argparser().parse_args(
        [
            "--scenario_key",
            "traffic_hourly",
            "--stages",
            "schedule_rows_seen",
            "--checkpoint_steps",
            "4000",
            "--split_phases",
            "validation_tuning",
            "--run_root",
            str(tmp_path),
            "--resume",
        ]
    )
    stage = _build_stage_commands(args, tmp_path)[0]
    manifest = tmp_path / stage.manifest_name
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "command_hash": hashlib.sha256(json.dumps(stage.commands, sort_keys=True).encode()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        full_pipeline.schedule_runner, "schedule_row_output_status", lambda *args: {"complete": outputs_complete}
    )
    commands = []
    monkeypatch.setattr(full_pipeline.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert run_full_pipeline(args)["status"] == "complete"
    assert commands == ([] if outputs_complete else stage.commands)


def test_pipeline_records_failure_and_stops_later_stages(tmp_path, monkeypatch):
    args = build_argparser().parse_args(
        [
            "--scenario_key",
            "traffic_hourly",
            "--stages",
            "schedule_rows_seen,schedule_rows_unseen",
            "--checkpoint_steps",
            "4000",
            "--split_phases",
            "validation_tuning",
            "--run_root",
            str(tmp_path),
        ]
    )
    attempted = []

    def fail(command, **kwargs):
        attempted.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(full_pipeline.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_full_pipeline(args)
    assert len(attempted) == 1
    assert json.loads((tmp_path / "schedule_rows_seen_manifest.json").read_text())["status"] == "failed"
    assert not (tmp_path / "schedule_rows_unseen_manifest.json").exists()


def test_explicit_locked_test_keeps_full_evaluation_budget(tmp_path):
    args = build_argparser().parse_args(
        [
            "--scenario_key",
            "traffic_hourly",
            "--stages",
            "schedule_rows_seen",
            "--checkpoint_steps",
            "4000",
            "--split_phases",
            "locked_test",
        ]
    )
    command = _build_stage_commands(args, tmp_path)[0].commands[0]
    assert command[command.index("--split_phase") + 1] == "locked_test"
    assert "--eval_windows_test" not in command
