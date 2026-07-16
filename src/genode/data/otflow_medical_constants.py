from __future__ import annotations

from pathlib import Path

from genode.data.otflow_paths import long_term_st_data_path

LONG_TERM_ST_DATASET_KEY = "long_term_st"
LONG_TERM_ST_MANIFEST_NAME = "manifest.json"
LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ = 250.0
LONG_TERM_ST_SAMPLING_RATE_HZ = 100.0
LONG_TERM_ST_FREQUENCY_LABEL = "100_hz"
LONG_TERM_ST_CONTEXT_SECONDS = 120
LONG_TERM_ST_HORIZON_SECONDS = 30
LONG_TERM_ST_HISTORY_LEN = int(LONG_TERM_ST_CONTEXT_SECONDS * LONG_TERM_ST_SAMPLING_RATE_HZ)
LONG_TERM_ST_HORIZON_LEN = int(LONG_TERM_ST_HORIZON_SECONDS * LONG_TERM_ST_SAMPLING_RATE_HZ)
LONG_TERM_ST_STRIDE = LONG_TERM_ST_HORIZON_LEN


def long_term_st_manifest_path(data_path: str | Path | None = None) -> Path:
    return Path(data_path or long_term_st_data_path()).expanduser().resolve() / LONG_TERM_ST_MANIFEST_NAME


__all__ = [
    "LONG_TERM_ST_MANIFEST_NAME",
    "LONG_TERM_ST_DATASET_KEY",
    "LONG_TERM_ST_CONTEXT_SECONDS",
    "LONG_TERM_ST_STRIDE",
    "LONG_TERM_ST_FREQUENCY_LABEL",
    "LONG_TERM_ST_HISTORY_LEN",
    "LONG_TERM_ST_HORIZON_LEN",
    "LONG_TERM_ST_HORIZON_SECONDS",
    "LONG_TERM_ST_SAMPLING_RATE_HZ",
    "LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ",
    "long_term_st_manifest_path",
]
