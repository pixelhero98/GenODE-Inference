from __future__ import annotations

from pathlib import Path

from genode.data.otflow_paths import project_data_root

LONG_TERM_HEADERED_ECG_DATASET_KEY = "long_term_headered_ECG_records"
SLEEP_EDF_DATASET_KEY = "sleep_edf"
DEFAULT_LONG_TERM_ECG_MANIFEST_NAME = "manifest.json"
DEFAULT_SLEEP_EDF_NPZ_NAME = "sleep_edf_3ch_100hz_stage_conditioned.npz"
DEFAULT_SLEEP_EDF_METADATA_NAME = "sleep_edf_3ch_100hz_stage_conditioned.json"


def default_long_term_headered_ecg_dataset_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root).resolve() / LONG_TERM_HEADERED_ECG_DATASET_KEY


def default_long_term_headered_ecg_manifest_path(dataset_root: str | Path) -> Path:
    return default_long_term_headered_ecg_dataset_dir(dataset_root) / DEFAULT_LONG_TERM_ECG_MANIFEST_NAME


def default_sleep_edf_data_path() -> str:
    return str(project_data_root() / DEFAULT_SLEEP_EDF_NPZ_NAME)


def default_sleep_edf_metadata_path() -> str:
    return str(project_data_root() / DEFAULT_SLEEP_EDF_METADATA_NAME)


def sleep_edf_metadata_path_for_npz(npz_path: str | Path) -> Path:
    return Path(npz_path).expanduser().resolve().with_suffix(".json")


__all__ = [
    "DEFAULT_LONG_TERM_ECG_MANIFEST_NAME",
    "DEFAULT_SLEEP_EDF_METADATA_NAME",
    "DEFAULT_SLEEP_EDF_NPZ_NAME",
    "LONG_TERM_HEADERED_ECG_DATASET_KEY",
    "SLEEP_EDF_DATASET_KEY",
    "default_long_term_headered_ecg_dataset_dir",
    "default_long_term_headered_ecg_manifest_path",
    "default_sleep_edf_data_path",
    "default_sleep_edf_metadata_path",
    "sleep_edf_metadata_path_for_npz",
]
