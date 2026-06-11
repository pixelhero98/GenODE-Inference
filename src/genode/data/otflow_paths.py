
from __future__ import annotations

import os
from pathlib import Path


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (project_root() / raw).resolve()


def project_data_root() -> Path:
    return project_root() / "data"


def project_paper_dataset_root() -> Path:
    return project_root() / "paper_datasets"


def project_outputs_root() -> Path:
    return project_root() / "outputs"


def project_results_root() -> Path:
    return project_outputs_root()


def project_backbone_matrix_root() -> Path:
    return project_outputs_root() / "backbone_matrix"


def default_backbone_manifest_path() -> Path:
    return project_backbone_matrix_root() / "backbone_manifest.json"


def project_checkpoint_import_root() -> Path:
    return project_outputs_root() / "imported_backbones"


def project_medical_staging_root() -> Path:
    raw = str(os.environ.get("OTFLOW_MEDICAL_STAGING_ROOT", "") or "").strip()
    if not raw:
        raise RuntimeError("Set OTFLOW_MEDICAL_STAGING_ROOT to prepare raw medical datasets.")
    return Path(raw).expanduser().resolve()


def default_cryptos_data_path() -> str:
    return str(project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz")


def default_lobster_synthetic_profile_path() -> str:
    return str(project_data_root() / "lobster_synthetic" / "lobster_free_sample_profile_10.json")


def default_long_term_st_data_path() -> str:
    return str(project_data_root() / "long_term_st_100hz_context_only")
