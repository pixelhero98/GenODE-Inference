
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


def project_root() -> Path:
    configured = str(os.environ.get("GENODE_PROJECT_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def resolve_project_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (project_root() / raw).resolve()


def display_project_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    for root_name, root in (
        ("outputs", project_outputs_root()),
        ("data", project_data_root()),
        ("datasets", project_dataset_root()),
    ):
        try:
            rel = resolved.relative_to(Path(root).expanduser().resolve())
        except ValueError:
            continue
        return PurePosixPath(root_name, *rel.parts).as_posix()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return PurePosixPath("external", resolved.name).as_posix()


def project_data_root() -> Path:
    return project_root() / "data"


def project_dataset_root() -> Path:
    return project_root() / "datasets"


def project_outputs_root() -> Path:
    return project_root() / "outputs"


def project_backbone_matrix_root() -> Path:
    return project_outputs_root() / "backbone_matrix"


def backbone_manifest_path() -> Path:
    return project_root() / "backbone_manifest.json"


def cryptos_data_path() -> str:
    return str(project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz")


def lobster_synthetic_profile_path() -> str:
    return str(project_data_root() / "lobster_synthetic" / "lobster_free_sample_profile_10.json")


def long_term_st_data_path() -> str:
    return str(project_data_root() / "long_term_st_100hz_context_only")
