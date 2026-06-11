from __future__ import annotations

from pathlib import Path

from genode.data.otflow_paths import project_data_root

LONG_TERM_ST_DATASET_KEY = "long_term_st"
DEFAULT_LONG_TERM_ECG_MANIFEST_NAME = "manifest.json"
DEFAULT_LONG_TERM_ST_DIR_NAME = "long_term_st_100hz_context_only"


def default_long_term_st_data_path() -> str:
    return str(project_data_root() / DEFAULT_LONG_TERM_ST_DIR_NAME)


def default_long_term_st_manifest_path(data_path: str | Path | None = None) -> Path:
    return Path(data_path or default_long_term_st_data_path()).expanduser().resolve() / DEFAULT_LONG_TERM_ECG_MANIFEST_NAME


__all__ = [
    "DEFAULT_LONG_TERM_ECG_MANIFEST_NAME",
    "DEFAULT_LONG_TERM_ST_DIR_NAME",
    "LONG_TERM_ST_DATASET_KEY",
    "default_long_term_st_data_path",
    "default_long_term_st_manifest_path",
]
