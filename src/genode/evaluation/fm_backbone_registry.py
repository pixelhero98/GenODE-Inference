from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from genode.data.otflow_experiment_plan import (
    PAPER_CONDITIONAL_GENERATION_DATASETS,
    PAPER_FORECAST_DATASETS,
    CONDITIONAL_GENERATION_FAMILY,
    FORECAST_FAMILY,
    experiment_plan_by_key,
)
from genode.data.otflow_medical_constants import (
    LONG_TERM_ST_DATASET_KEY,
    long_term_st_manifest_path,
)
from genode.data.molecule_xyz import (
    MOLECULE_GROUP_DATASET_KEYS,
    load_molecule_group_manifest,
    molecule_group_root as project_molecule_group_root,
)
from genode.data.otflow_paths import (
    backbone_manifest_path,
    lobster_synthetic_profile_path,
    display_project_path,
    project_backbone_matrix_root,
    project_data_root,
    project_outputs_root,
    project_paper_dataset_root,
    project_root,
)
from genode.path_safety import (
    MANIFEST_PARENT_PATH_BASE,
    resolve_manifest_path_base,
    resolve_portable_relative_path,
)

MOLECULE_FAMILY = "molecule_3d_coordinate_generation"
BACKBONE_NAME_OTFLOW = "otflow"
BACKBONE_NAME_OTFLOW_MOLECULE = "otflow_molecule_3d"
DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE = "transformer"
DEFAULT_MOLECULE_VARIANT = "ar_h1"
DEFAULT_SEED = 0
TEMPORAL_ROLLOUT_MODE = "non_ar"
MOLECULE_ROLLOUT_MODE = "autoregressive"
TRAIN_BUDGET_STEPS: Tuple[int, ...] = (4000, 8000, 12000, 16000, 20000)
STANDARD_ARTIFACT_SUMMARY_NAME = "artifact_summary.json"
MANIFEST_VERSION = "fm_backbone_manifest"
AUDIT_VERSION = "fm_backbone_manifest_check"
IMPORTED_EXTERNAL_SOURCE_KIND = "imported_external"
_MANIFEST_ARTIFACT_PATH_FIELDS = ("checkpoint_path", "summary_path", "metadata_path")
_MANIFEST_ROOT_PATH_FIELDS = (
    "matrix_root",
    "otflow_reuse_root",
    "imported_backbone_root",
    "molecule_group_root",
    "molecule_backbone_root",
)

ACTIVE_FORECAST_BACKBONE_BUDGETS: Mapping[str, Tuple[int, ...]] = {
    "solar_energy_10m": (4000, 8000, 12000, 16000, 20000),
    "traffic_hourly": (4000, 8000, 12000, 16000, 20000),
    "weather_daily": (4000, 8000, 12000, 16000, 20000),
}
ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS: Mapping[str, Tuple[int, ...]] = {
    "cryptos": (4000, 8000, 12000, 16000, 20000),
    "lobster_synthetic": (4000, 8000, 12000, 16000, 20000),
    LONG_TERM_ST_DATASET_KEY: (4000, 8000, 12000, 16000, 20000),
}


@dataclass(frozen=True)
class BackboneArtifactSpec:
    backbone_name: str
    benchmark_family: str
    dataset_key: str
    train_steps: int
    train_budget_label: str
    checkpoint_id: str
    checkpoint_path: str
    summary_path: str
    status: str
    seed: int = DEFAULT_SEED
    checkpoint_budget_steps: Optional[int] = None
    effective_train_steps: Optional[int] = None
    checkpoint_export_protocol: Optional[str] = None
    source_kind: str = "planned"
    metadata_path: Optional[str] = None
    field_network_type: Optional[str] = None
    notes: Optional[str] = None
    model_cond_dim: Optional[int] = None
    compatibility_error: Optional[str] = None
    member_key: Optional[str] = None
    stratum: Optional[str] = None
    atom_count: Optional[int] = None
    formula: Optional[str] = None
    source_zip_name: Optional[str] = None
    trajectory_count: Optional[int] = None
    variant: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def train_budget_label(train_steps: int) -> str:
    steps = int(train_steps)
    for value in TRAIN_BUDGET_STEPS:
        if int(value) == steps:
            return f"{int(value) // 1000}k"
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    return f"{steps}_steps"


def _metadata_int(metadata: Mapping[str, Any] | None, key: str) -> Optional[int]:
    if metadata is None:
        return None
    value = metadata.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _metadata_str(metadata: Mapping[str, Any] | None, key: str) -> Optional[str]:
    if metadata is None:
        return None
    value = str(metadata.get(key, "") or "").strip()
    return value or None


def project_otflow_reuse_root() -> Path:
    return project_outputs_root() / "shared_backbones" / "otflow_fullhorizon_seed0"


def project_imported_otflow_backbone_root() -> Path:
    return project_outputs_root() / "imported_backbones" / "otflow"


def _project_display_path(path: str | Path, *, path_base: Path | None = None) -> str:
    if path_base is not None:
        try:
            relative = Path(path).resolve().relative_to(Path(path_base).resolve())
        except ValueError as exc:
            raise ValueError(
                "Refusing to serialize manifest path outside the validated path base: "
                f"{display_project_path(path)}"
            ) from exc
        return PurePosixPath(*relative.parts).as_posix()
    return display_project_path(path)


def _manifest_path_base(target_path: Path, roots: Sequence[Path]) -> Path:
    resolved_target_parent = target_path.parent.resolve()
    resolved_roots = tuple(Path(root).resolve() for root in roots)
    outside = [root for root in resolved_roots if not root.is_relative_to(resolved_target_parent)]
    if outside:
        raise ValueError(
            "Backbone manifest roots must be contained by the manifest parent; "
            f"outside roots={[display_project_path(path) for path in outside]}."
        )
    return resolved_target_parent


def build_backbone_checkpoint_id(
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int = DEFAULT_SEED,
    field_network_type: Optional[str] = None,
    member_key: Optional[str] = None,
    stratum: Optional[str] = None,
) -> str:
    if str(benchmark_family) == FORECAST_FAMILY:
        family_token = "temporal_extrapolation"
    elif str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        family_token = "temporal_conditional_generation"
    elif str(benchmark_family) == MOLECULE_FAMILY:
        family_token = "molecule_3d_coordinate_generation"
    else:
        raise ValueError(f"Unsupported benchmark_family={benchmark_family}")
    parts = [str(dataset_key), str(backbone_name), family_token]
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY and field_network_type:
        parts.append(str(field_network_type))
    if str(benchmark_family) == MOLECULE_FAMILY:
        parts.append(str(member_key or stratum or "molecule_stratum"))
    parts.extend([train_budget_label(int(train_steps)), f"seed{int(seed)}"])
    return "_".join(parts)


def _forecast_artifact_root(matrix_root: Path, backbone_name: str, dataset_key: str, train_steps: int) -> Path:
    return matrix_root / str(backbone_name) / FORECAST_FAMILY / train_budget_label(int(train_steps)) / str(dataset_key)


def _conditional_generation_artifact_root(matrix_root: Path, backbone_name: str, dataset_key: str, train_steps: int) -> Path:
    return (
        matrix_root
        / str(backbone_name)
        / CONDITIONAL_GENERATION_FAMILY
        / train_budget_label(int(train_steps))
        / str(dataset_key)
        / DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
    )


def project_molecule_backbone_root() -> Path:
    return project_outputs_root() / "molecule_3d_backbones"


def _molecule_artifact_root(
    molecule_backbone_root: Path,
    *,
    dataset_key: str,
    member_key: str,
    stratum: str,
    train_steps: int,
    variant: str = DEFAULT_MOLECULE_VARIANT,
) -> Path:
    return (
        molecule_backbone_root
        / str(dataset_key)
        / str(member_key)
        / str(stratum)
        / str(variant)
        / f"{int(train_steps)}_steps"
    )


def expected_artifact_root(
    matrix_root: str | Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    molecule_backbone_root: str | Path | None = None,
    member_key: str = "",
    stratum: str = "",
    variant: str = DEFAULT_MOLECULE_VARIANT,
) -> Path:
    root = Path(matrix_root).resolve()
    if str(benchmark_family) == FORECAST_FAMILY:
        return _forecast_artifact_root(root, str(backbone_name), str(dataset_key), int(train_steps))
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        return _conditional_generation_artifact_root(root, str(backbone_name), str(dataset_key), int(train_steps))
    if str(benchmark_family) == MOLECULE_FAMILY:
        if not str(member_key) or not str(stratum):
            raise ValueError("Molecule artifact roots require non-empty member_key and stratum.")
        molecule_root = Path(molecule_backbone_root or project_molecule_backbone_root()).resolve()
        return _molecule_artifact_root(
            molecule_root,
            dataset_key=str(dataset_key),
            member_key=str(member_key),
            stratum=str(stratum),
            train_steps=int(train_steps),
            variant=str(variant),
        )
    raise ValueError(f"Unsupported benchmark_family={benchmark_family}")


def _expected_materialized_paths(
    matrix_root: str | Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    molecule_backbone_root: str | Path | None = None,
    member_key: str = "",
    stratum: str = "",
    variant: str = DEFAULT_MOLECULE_VARIANT,
) -> Dict[str, Path]:
    artifact_root = expected_artifact_root(
        matrix_root,
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        molecule_backbone_root=molecule_backbone_root,
        member_key=str(member_key),
        stratum=str(stratum),
        variant=str(variant),
    )
    checkpoint_name = "model.pt"
    metadata_name = "checkpoint_metadata.json"
    return {
        "artifact_root": artifact_root,
        "checkpoint_path": artifact_root / checkpoint_name,
        "summary_path": artifact_root / STANDARD_ARTIFACT_SUMMARY_NAME,
        "metadata_path": artifact_root / metadata_name,
    }


def _existing_summary_path(
    artifact_root: Path,
    *,
    preferred: Path,
    benchmark_family: str,
    backbone_name: str,
) -> Path:
    candidates = [preferred]
    candidates.append(artifact_root / "checkpoint_metadata.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred


def _safe_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_cond_dim(metadata: Mapping[str, Any]) -> int:
    split_stats = dict(metadata.get("split_stats", {}) or {})
    if "cond_dim" in split_stats:
        return int(split_stats.get("cond_dim") or 0)
    return int(metadata.get("cond_dim") or 0)


def _checkpoint_signature(checkpoint_path: Path) -> Tuple[Optional[Dict[str, int | str]], Optional[str]]:
    try:
        import torch

        from genode.evaluation.otflow_evaluation_support import load_checkpoint_model

        model, cfg = load_checkpoint_model(Path(checkpoint_path), torch.device("cpu"))
        del model
    except Exception as exc:
        return None, f"Unable to load OTFlow checkpoint: {type(exc).__name__}: {exc}"
    signature = {
        "model_cond_dim": int(getattr(cfg.model, "cond_dim", 0) or 0),
        "history_len": int(getattr(cfg, "history_len")),
        "future_block_len": int(getattr(cfg.model, "future_block_len", 1)),
        "prediction_horizon": int(getattr(cfg, "prediction_horizon", 1)),
        "rollout_mode": str(getattr(cfg.model, "rollout_mode", "")),
        "train_steps": int(getattr(cfg.train, "steps", 0) or 0),
        "field_network_type": str(getattr(cfg.model, "fu_net_type", "")),
    }
    return signature, None


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if value is None or str(value).strip() == "":
        raise KeyError(key)
    return value


def _otflow_artifact_compatibility(
    metadata: Optional[Mapping[str, Any]],
    checkpoint_path: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    field_network_type: Optional[str],
) -> Tuple[Optional[int], str, Optional[str]]:
    if not metadata:
        return None, "invalid", "Missing checkpoint metadata."
    signature, error = _checkpoint_signature(checkpoint_path)
    if error is not None:
        return None, "invalid", error
    assert signature is not None
    model_cond_dim = int(signature["model_cond_dim"])
    spec = experiment_plan_by_key()[str(dataset_key)]
    errors: List[str] = []

    required_checks = (
        ("dataset_key", str(dataset_key), str),
        ("benchmark_family", str(benchmark_family), str),
        ("train_steps", int(train_steps), int),
        ("checkpoint_budget_steps", int(train_steps), int),
        ("effective_train_steps", int(train_steps), int),
        ("checkpoint_export_protocol", "exact_budget_step_state", str),
        ("history_len", int(spec.history_len), int),
        ("future_block_len", int(spec.future_block_len), int),
    )
    for key, expected, caster in required_checks:
        try:
            observed = caster(_metadata_value(metadata, key))
        except (KeyError, TypeError, ValueError):
            errors.append(f"metadata.{key} is missing or invalid")
            continue
        if observed != expected:
            errors.append(f"metadata.{key}={observed!r} != expected {expected!r}")

    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        try:
            observed_field = str(_metadata_value(metadata, "field_network_type"))
        except KeyError:
            errors.append("metadata.field_network_type is missing or invalid")
        else:
            if observed_field != str(field_network_type or DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE):
                errors.append(
                    f"metadata.field_network_type={observed_field!r} != expected {str(field_network_type or DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE)!r}"
                )

    cond_dim = _metadata_cond_dim(metadata)
    if int(model_cond_dim) != int(cond_dim):
        errors.append(f"metadata cond_dim={int(cond_dim)} != checkpoint model.cond_dim={int(model_cond_dim)}")
    if int(signature["train_steps"]) != int(train_steps):
        errors.append(f"checkpoint train.steps={int(signature['train_steps'])} != expected {int(train_steps)}")
    if int(signature["history_len"]) != int(spec.history_len):
        errors.append(f"checkpoint history_len={int(signature['history_len'])} != expected {int(spec.history_len)}")
    if int(signature["future_block_len"]) != int(spec.future_block_len):
        errors.append(
            f"checkpoint future_block_len={int(signature['future_block_len'])} != expected {int(spec.future_block_len)}"
        )
    if str(signature.get("rollout_mode", "")).strip().lower() != TEMPORAL_ROLLOUT_MODE:
        errors.append(
            f"checkpoint rollout_mode={str(signature.get('rollout_mode', ''))!r} != expected {TEMPORAL_ROLLOUT_MODE!r}"
        )
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY and str(signature["field_network_type"]) != str(
        field_network_type or DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
    ):
        errors.append(
            f"checkpoint field_network_type={str(signature['field_network_type'])!r} != expected {str(field_network_type or DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE)!r}"
        )

    if errors:
        return model_cond_dim, "invalid", "; ".join(errors)
    return model_cond_dim, "ready", None


def _safe_manifest_token(value: Any, *, label: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"Molecule group manifest member has empty {label}.")
    posix = PurePosixPath(token)
    if posix.is_absolute() or ".." in posix.parts or "/" in token or "\\" in token:
        raise ValueError(f"Molecule group manifest member has unsafe {label}={token!r}.")
    return token


def _molecule_manifest_members(
    *,
    molecule_group_root: str | Path | None = None,
    dataset_keys: Sequence[str] = MOLECULE_GROUP_DATASET_KEYS,
) -> List[Dict[str, Any]]:
    members: List[Dict[str, Any]] = []
    for dataset_key in tuple(str(key) for key in dataset_keys):
        manifest_path = (
            project_molecule_group_root()
            if molecule_group_root is None
            else Path(molecule_group_root).expanduser().resolve()
        ) / dataset_key / "group_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_molecule_group_manifest(dataset_key, manifest_path.parent.parent)
        for raw_member in manifest.get("strata", []):
            member = dict(raw_member)
            stratum = _safe_manifest_token(member.get("stratum"), label="stratum")
            if stratum.startswith("Direct_"):
                continue
            if not bool(member.get("trainable", True)) or bool(member.get("mixed_shape", False)):
                continue
            member_key = _safe_manifest_token(member.get("member_key"), label="member_key")
            source_zip_name = _safe_manifest_token(member.get("source_zip_name"), label="source_zip_name")
            if Path(source_zip_name).name != source_zip_name:
                raise ValueError("Molecule group manifest source_zip_name must be a file name, not a path.")
            members.append(
                {
                    "dataset_key": dataset_key,
                    "member_key": member_key,
                    "stratum": stratum,
                    "source_zip_name": source_zip_name,
                    "trajectory_count": int(member.get("trajectory_count", member.get("xyz_count", 0)) or 0),
                    "atom_count": int(member["atom_count"]),
                    "formula": str(member.get("formula", "")),
                }
            )
    return sorted(members, key=lambda row: (str(row["dataset_key"]), str(row["member_key"])))


def _molecule_artifact_compatibility(
    metadata: Optional[Mapping[str, Any]],
    checkpoint_path: Path,
    *,
    dataset_key: str,
    member_key: str,
    stratum: str,
    train_steps: int,
    atom_count: int,
    formula: str,
    source_zip_name: str,
    variant: str,
) -> Tuple[str, Optional[str]]:
    if not metadata:
        return "invalid", "Missing checkpoint metadata."
    errors: List[str] = []
    try:
        from genode.evaluation.molecule_metrics import load_molecule_checkpoint_payload

        load_molecule_checkpoint_payload(
            checkpoint_path,
            expected_identity=f"molecule backbone artifact {dataset_key}/{member_key}/{train_steps}",
            expected_dataset_key=str(dataset_key),
            expected_stratum=str(stratum),
        )
    except Exception as exc:
        errors.append(f"Unable to load molecule checkpoint: {type(exc).__name__}: {exc}")
    required_checks = (
        ("dataset_key", str(dataset_key), str),
        ("member_key", str(member_key), str),
        ("stratum", str(stratum), str),
        ("benchmark_family", MOLECULE_FAMILY, str),
        ("backbone_name", BACKBONE_NAME_OTFLOW_MOLECULE, str),
        ("variant", str(variant), str),
        ("train_steps", int(train_steps), int),
        ("history_len", 16, int),
        ("future_block_len", 1, int),
    )
    for key, expected, caster in required_checks:
        try:
            observed = caster(_metadata_value(metadata, key))
        except (KeyError, TypeError, ValueError):
            errors.append(f"metadata.{key} is missing or invalid")
            continue
        if observed != expected:
            errors.append(f"metadata.{key}={observed!r} != expected {expected!r}")
    cfg = dict(metadata.get("cfg", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    observed_rollout = str(metadata.get("rollout_mode", model_cfg.get("rollout_mode", "")))
    if observed_rollout.strip().lower() != MOLECULE_ROLLOUT_MODE:
        errors.append(
            f"metadata.rollout_mode={observed_rollout!r} != expected {MOLECULE_ROLLOUT_MODE!r}"
        )
    split_stats = dict(metadata.get("split_stats", {}) or {})
    if int(split_stats.get("atom_count", atom_count)) != int(atom_count):
        errors.append(f"metadata.split_stats.atom_count={split_stats.get('atom_count')!r} != expected {int(atom_count)!r}")
    if str(split_stats.get("formula", formula)) != str(formula):
        errors.append(f"metadata.split_stats.formula={split_stats.get('formula')!r} != expected {str(formula)!r}")
    if str(metadata.get("source_zip_name", "")) != str(source_zip_name):
        errors.append(f"metadata.source_zip_name={metadata.get('source_zip_name')!r} != expected {str(source_zip_name)!r}")
    if errors:
        return "invalid", "; ".join(errors)
    return "ready", None


def _existing_matrix_artifact(
    matrix_root: Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int,
    path_base: Path | None = None,
) -> Optional[BackboneArtifactSpec]:
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    if not paths["checkpoint_path"].exists():
        return None
    metadata = _safe_json(paths["metadata_path"])
    field_network_type = None if metadata is None else metadata.get("field_network_type")
    expected_field_network_type = (
        DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
        if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY
        else None
    )
    model_cond_dim, status, compatibility_error = _otflow_artifact_compatibility(
        metadata,
        paths["checkpoint_path"],
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        field_network_type=str(field_network_type or expected_field_network_type) if (field_network_type or expected_field_network_type) else None,
    )
    checkpoint_id = None if metadata is None else metadata.get("checkpoint_id")
    summary_path = _existing_summary_path(
        paths["artifact_root"],
        preferred=paths["summary_path"],
        benchmark_family=str(benchmark_family),
        backbone_name=str(backbone_name),
    )
    return BackboneArtifactSpec(
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        train_budget_label=str((metadata or {}).get("train_budget_label", train_budget_label(int(train_steps)))),
        checkpoint_id=str(
            checkpoint_id
            or build_backbone_checkpoint_id(
                backbone_name=str(backbone_name),
                benchmark_family=str(benchmark_family),
                dataset_key=str(dataset_key),
                train_steps=int(train_steps),
                seed=int(seed),
                field_network_type=field_network_type if field_network_type else None,
            )
        ),
        checkpoint_path=_project_display_path(paths["checkpoint_path"], path_base=path_base),
        summary_path=_project_display_path(summary_path, path_base=path_base),
        status=str(status),
        seed=int(seed),
        checkpoint_budget_steps=_metadata_int(metadata, "checkpoint_budget_steps"),
        effective_train_steps=_metadata_int(metadata, "effective_train_steps"),
        checkpoint_export_protocol=_metadata_str(metadata, "checkpoint_export_protocol"),
        source_kind="matrix_output",
        metadata_path=_project_display_path(paths["metadata_path"], path_base=path_base),
        field_network_type=None if field_network_type is None else str(field_network_type),
        notes=compatibility_error,
        model_cond_dim=model_cond_dim,
        compatibility_error=compatibility_error,
    )


def _existing_molecule_artifact(
    matrix_root: Path,
    *,
    molecule_backbone_root: Path,
    member: Mapping[str, Any],
    train_steps: int,
    seed: int,
    variant: str = DEFAULT_MOLECULE_VARIANT,
    path_base: Path | None = None,
) -> Optional[BackboneArtifactSpec]:
    dataset_key = str(member["dataset_key"])
    stratum = str(member["stratum"])
    member_key = str(member["member_key"])
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
        benchmark_family=MOLECULE_FAMILY,
        dataset_key=dataset_key,
        train_steps=int(train_steps),
        molecule_backbone_root=molecule_backbone_root,
        member_key=member_key,
        stratum=stratum,
        variant=str(variant),
    )
    if not paths["checkpoint_path"].exists():
        return None
    metadata = _safe_json(paths["metadata_path"])
    status, compatibility_error = _molecule_artifact_compatibility(
        metadata,
        paths["checkpoint_path"],
        dataset_key=dataset_key,
        member_key=member_key,
        stratum=stratum,
        train_steps=int(train_steps),
        atom_count=int(member["atom_count"]),
        formula=str(member.get("formula", "")),
        source_zip_name=str(member.get("source_zip_name", "")),
        variant=str(variant),
    )
    if not paths["summary_path"].exists():
        summary_error = f"Missing required {STANDARD_ARTIFACT_SUMMARY_NAME}."
        compatibility_error = summary_error if not compatibility_error else f"{compatibility_error}; {summary_error}"
        status = "invalid"
    checkpoint_id = None if metadata is None else metadata.get("checkpoint_id")
    return BackboneArtifactSpec(
        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
        benchmark_family=MOLECULE_FAMILY,
        dataset_key=dataset_key,
        train_steps=int(train_steps),
        train_budget_label=str((metadata or {}).get("train_budget_label", train_budget_label(int(train_steps)))),
        checkpoint_id=str(
            checkpoint_id
            or build_backbone_checkpoint_id(
                backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                benchmark_family=MOLECULE_FAMILY,
                dataset_key=dataset_key,
                train_steps=int(train_steps),
                seed=int(seed),
                member_key=member_key,
                stratum=stratum,
            )
        ),
        checkpoint_path=_project_display_path(paths["checkpoint_path"], path_base=path_base),
        summary_path=_project_display_path(paths["summary_path"], path_base=path_base),
        status=status,
        seed=int(seed),
        checkpoint_budget_steps=_metadata_int(metadata, "checkpoint_budget_steps"),
        effective_train_steps=_metadata_int(metadata, "effective_train_steps"),
        checkpoint_export_protocol=_metadata_str(metadata, "checkpoint_export_protocol"),
        source_kind="molecule_backbone_output",
        metadata_path=_project_display_path(paths["metadata_path"], path_base=path_base),
        notes=compatibility_error,
        compatibility_error=compatibility_error,
        member_key=member_key,
        stratum=stratum,
        atom_count=int(member["atom_count"]),
        formula=str(member.get("formula", "")),
        source_zip_name=str(member.get("source_zip_name", "")),
        trajectory_count=int(member.get("trajectory_count", 0)),
        variant=str(variant),
    )


def _existing_otflow_reuse_artifact(
    reuse_root: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int,
    path_base: Path | None = None,
) -> Optional[BackboneArtifactSpec]:
    if int(train_steps) != 20000:
        return None
    if str(benchmark_family) == FORECAST_FAMILY:
        artifact_root = reuse_root / FORECAST_FAMILY / str(dataset_key)
        checkpoint_path = artifact_root / "model.pt"
        metadata_path = artifact_root / "checkpoint_metadata.json"
        summary_path = artifact_root / STANDARD_ARTIFACT_SUMMARY_NAME
        field_network_type = None
    elif str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        artifact_root = (
            reuse_root
            / CONDITIONAL_GENERATION_FAMILY
            / str(dataset_key)
            / DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
        )
        checkpoint_path = artifact_root / "model.pt"
        metadata_path = artifact_root / "checkpoint_metadata.json"
        summary_path = artifact_root / "checkpoint_metadata.json"
        field_network_type = DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
    else:
        raise ValueError(f"Unsupported benchmark_family={benchmark_family}")
    if not checkpoint_path.exists():
        return None
    metadata = _safe_json(metadata_path) or {}
    model_cond_dim, status, compatibility_error = _otflow_artifact_compatibility(
        metadata,
        checkpoint_path,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        field_network_type=field_network_type,
    )
    resolved_steps = int(metadata.get("train_steps", 20000))
    if resolved_steps != 20000:
        return None
    return BackboneArtifactSpec(
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=resolved_steps,
        train_budget_label=str(metadata.get("train_budget_label", train_budget_label(resolved_steps))),
        checkpoint_id=build_backbone_checkpoint_id(
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=str(benchmark_family),
            dataset_key=str(dataset_key),
            train_steps=resolved_steps,
            seed=int(seed),
            field_network_type=field_network_type,
        ),
        checkpoint_path=_project_display_path(checkpoint_path, path_base=path_base),
        summary_path=_project_display_path(
            summary_path if summary_path.exists() else metadata_path,
            path_base=path_base,
        ),
        status=str(status),
        seed=int(seed),
        checkpoint_budget_steps=_metadata_int(metadata, "checkpoint_budget_steps"),
        effective_train_steps=_metadata_int(metadata, "effective_train_steps"),
        checkpoint_export_protocol=_metadata_str(metadata, "checkpoint_export_protocol"),
        source_kind="reused_shared_20k",
        metadata_path=_project_display_path(metadata_path, path_base=path_base),
        field_network_type=field_network_type,
        notes=compatibility_error,
        model_cond_dim=model_cond_dim,
        compatibility_error=compatibility_error,
    )


def _imported_source_dir(
    imported_root: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
) -> Path:
    return (
        imported_root
        / str(benchmark_family)
        / str(dataset_key)
        / train_budget_label(int(train_steps))
    )


def _rewrite_normalized_json_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    summary_path: Path,
    normalized_from: Path,
) -> Dict[str, Any]:
    data = dict(payload)
    if "checkpoint_path" in data:
        data["checkpoint_path"] = _project_display_path(checkpoint_path)
    if "checkpoint_metadata_path" in data:
        data["checkpoint_metadata_path"] = _project_display_path(metadata_path)
    if "metadata_path" in data:
        data["metadata_path"] = _project_display_path(metadata_path)
    if "summary_path" in data:
        data["summary_path"] = _project_display_path(summary_path)
    data["normalized_from"] = _project_display_path(normalized_from)
    return data


def _copy_json_with_rewritten_paths(
    src_path: Path,
    dst_path: Path,
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    summary_path: Path,
) -> None:
    payload = _safe_json(src_path)
    if payload is None:
        shutil.copy2(src_path, dst_path)
        return
    rewritten = _rewrite_normalized_json_payload(
        payload,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        summary_path=summary_path,
        normalized_from=src_path.parent,
    )
    dst_path.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")


def _normalize_imported_artifact(
    imported_root: Path,
    matrix_root: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
) -> bool:
    source_dir = _imported_source_dir(
        imported_root,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    if not source_dir.exists():
        return False
    source_checkpoint = source_dir / "model.pt"
    source_metadata = source_dir / "checkpoint_metadata.json"
    if not source_checkpoint.exists() or not source_metadata.exists():
        return False
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    artifact_root = paths["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, paths["checkpoint_path"])
    for name in (STANDARD_ARTIFACT_SUMMARY_NAME, "transfer_record.json"):
        source_file = source_dir / name
        if not source_file.exists():
            continue
        if source_file.suffix.lower() == ".json":
            summary_target = (
                paths["summary_path"]
                if name == STANDARD_ARTIFACT_SUMMARY_NAME
                else artifact_root / name
            )
            _copy_json_with_rewritten_paths(
                source_file,
                summary_target,
                checkpoint_path=paths["checkpoint_path"],
                metadata_path=paths["metadata_path"],
                summary_path=paths["summary_path"],
            )
        else:
            shutil.copy2(source_file, artifact_root / name)
    _copy_json_with_rewritten_paths(
        source_metadata,
        paths["metadata_path"],
        checkpoint_path=paths["checkpoint_path"],
        metadata_path=paths["metadata_path"],
        summary_path=paths["summary_path"],
    )
    if not paths["summary_path"].exists():
        payload = _safe_json(paths["metadata_path"]) or {}
        summary_payload = _rewrite_normalized_json_payload(
            payload,
            checkpoint_path=paths["checkpoint_path"],
            metadata_path=paths["metadata_path"],
            summary_path=paths["summary_path"],
            normalized_from=source_dir,
        )
        paths["summary_path"].write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return True


def normalize_imported_backbone_artifacts(
    *,
    matrix_root: str | Path | None = None,
    imported_root: str | Path | None = None,
    budget_steps: Sequence[int] = TRAIN_BUDGET_STEPS,
) -> Dict[str, Any]:
    resolved_matrix_root = Path(matrix_root or project_backbone_matrix_root()).resolve()
    resolved_import_root = Path(imported_root or project_imported_otflow_backbone_root()).resolve()
    normalized: List[Dict[str, Any]] = []
    if not resolved_import_root.exists():
        return {
            "matrix_root": _project_display_path(resolved_matrix_root),
            "imported_root": _project_display_path(resolved_import_root),
            "normalized_count": 0,
            "normalized": normalized,
        }
    requested_steps = {int(value) for value in budget_steps}
    for benchmark_family, budget_map in (
        (FORECAST_FAMILY, ACTIVE_FORECAST_BACKBONE_BUDGETS),
        (CONDITIONAL_GENERATION_FAMILY, ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS),
    ):
        for dataset_key, active_steps in budget_map.items():
            for train_steps in active_steps:
                if int(train_steps) not in requested_steps:
                    continue
                if not _normalize_imported_artifact(
                    resolved_import_root,
                    resolved_matrix_root,
                    benchmark_family=str(benchmark_family),
                    dataset_key=str(dataset_key),
                    train_steps=int(train_steps),
                ):
                    continue
                normalized.append(
                    {
                        "benchmark_family": str(benchmark_family),
                        "dataset_key": str(dataset_key),
                        "train_steps": int(train_steps),
                    }
                )
    return {
        "matrix_root": _project_display_path(resolved_matrix_root),
        "imported_root": _project_display_path(resolved_import_root),
        "normalized_count": int(len(normalized)),
        "normalized": normalized,
    }


def _planned_artifact(
    matrix_root: Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int,
    path_base: Path | None = None,
) -> BackboneArtifactSpec:
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    field_network_type = (
        DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
        if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY
        else None
    )
    return BackboneArtifactSpec(
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        train_budget_label=train_budget_label(int(train_steps)),
        checkpoint_id=build_backbone_checkpoint_id(
            backbone_name=str(backbone_name),
            benchmark_family=str(benchmark_family),
            dataset_key=str(dataset_key),
            train_steps=int(train_steps),
            seed=int(seed),
            field_network_type=field_network_type,
        ),
        checkpoint_path=_project_display_path(paths["checkpoint_path"], path_base=path_base),
        summary_path=_project_display_path(paths["summary_path"], path_base=path_base),
        status="missing",
        seed=int(seed),
        source_kind="planned",
        metadata_path=_project_display_path(paths["metadata_path"], path_base=path_base),
        field_network_type=field_network_type,
    )


def _planned_molecule_artifact(
    matrix_root: Path,
    *,
    molecule_backbone_root: Path,
    member: Mapping[str, Any],
    train_steps: int,
    seed: int,
    variant: str = DEFAULT_MOLECULE_VARIANT,
    path_base: Path | None = None,
) -> BackboneArtifactSpec:
    dataset_key = str(member["dataset_key"])
    stratum = str(member["stratum"])
    member_key = str(member["member_key"])
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
        benchmark_family=MOLECULE_FAMILY,
        dataset_key=dataset_key,
        train_steps=int(train_steps),
        molecule_backbone_root=molecule_backbone_root,
        member_key=member_key,
        stratum=stratum,
        variant=str(variant),
    )
    return BackboneArtifactSpec(
        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
        benchmark_family=MOLECULE_FAMILY,
        dataset_key=dataset_key,
        train_steps=int(train_steps),
        train_budget_label=train_budget_label(int(train_steps)),
        checkpoint_id=build_backbone_checkpoint_id(
            backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
            benchmark_family=MOLECULE_FAMILY,
            dataset_key=dataset_key,
            train_steps=int(train_steps),
            seed=int(seed),
            member_key=member_key,
            stratum=stratum,
        ),
        checkpoint_path=_project_display_path(paths["checkpoint_path"], path_base=path_base),
        summary_path=_project_display_path(paths["summary_path"], path_base=path_base),
        status="missing",
        seed=int(seed),
        source_kind="planned",
        metadata_path=_project_display_path(paths["metadata_path"], path_base=path_base),
        member_key=member_key,
        stratum=stratum,
        atom_count=int(member["atom_count"]),
        formula=str(member.get("formula", "")),
        source_zip_name=str(member.get("source_zip_name", "")),
        trajectory_count=int(member.get("trajectory_count", 0)),
        variant=str(variant),
    )


def _iter_target_specs(
    *,
    matrix_root: Path,
    molecule_backbone_root: Path,
    molecule_group_root: str | Path | None,
    seed: int,
    budget_steps: Sequence[int],
    path_base: Path | None = None,
) -> Iterable[BackboneArtifactSpec]:
    requested_steps = {int(value) for value in budget_steps}
    for dataset_key in tuple(PAPER_FORECAST_DATASETS):
        for train_steps in ACTIVE_FORECAST_BACKBONE_BUDGETS.get(str(dataset_key), ()):
            if int(train_steps) not in requested_steps:
                continue
            yield _planned_artifact(
                matrix_root,
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=FORECAST_FAMILY,
                dataset_key=str(dataset_key),
                train_steps=int(train_steps),
                seed=int(seed),
                path_base=path_base,
            )
    for dataset_key in tuple(PAPER_CONDITIONAL_GENERATION_DATASETS):
        for train_steps in ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS.get(str(dataset_key), ()):
            if int(train_steps) not in requested_steps:
                continue
            yield _planned_artifact(
                matrix_root,
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                dataset_key=str(dataset_key),
                train_steps=int(train_steps),
                seed=int(seed),
                path_base=path_base,
            )
    for member in _molecule_manifest_members(molecule_group_root=molecule_group_root):
        for train_steps in TRAIN_BUDGET_STEPS:
            if int(train_steps) not in requested_steps:
                continue
            yield _planned_molecule_artifact(
                matrix_root,
                molecule_backbone_root=molecule_backbone_root,
                member=member,
                train_steps=int(train_steps),
                seed=int(seed),
                path_base=path_base,
            )


def materialize_backbone_manifest(
    *,
    matrix_root: str | Path | None = None,
    otflow_reuse_root: str | Path | None = None,
    imported_backbone_root: str | Path | None = None,
    molecule_group_root: str | Path | None = None,
    molecule_backbone_root: str | Path | None = None,
    budget_steps: Sequence[int] = TRAIN_BUDGET_STEPS,
    seed: int = DEFAULT_SEED,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    target_path = Path(write_path or backbone_manifest_path()).resolve()
    target_is_external = not target_path.is_relative_to(project_root().resolve())
    external_root = target_path.parent
    resolved_matrix_root = Path(
        matrix_root
        or (external_root / "matrix" if target_is_external else project_backbone_matrix_root())
    ).resolve()
    resolved_reuse_root = Path(
        otflow_reuse_root
        or (external_root / "shared_backbones" if target_is_external else project_otflow_reuse_root())
    ).resolve()
    resolved_import_root = Path(
        imported_backbone_root
        or (external_root / "imported_backbones" if target_is_external else project_imported_otflow_backbone_root())
    ).resolve()
    resolved_molecule_backbone_root = Path(
        molecule_backbone_root
        or (external_root / "molecule_3d_backbones" if target_is_external else project_molecule_backbone_root())
    ).resolve()
    resolved_molecule_group_root = Path(
        molecule_group_root
        or (external_root / "molecule_3d" if target_is_external else project_molecule_group_root())
    ).resolve()
    manifest_path_base = _manifest_path_base(
        target_path,
        (
            resolved_matrix_root,
            resolved_reuse_root,
            resolved_import_root,
            resolved_molecule_backbone_root,
            resolved_molecule_group_root,
        ),
    )
    artifacts: List[Dict[str, Any]] = []
    ready_count = 0
    molecule_member_by_key = {
        (str(member["dataset_key"]), str(member["member_key"]), str(member["stratum"])): member
        for member in _molecule_manifest_members(molecule_group_root=resolved_molecule_group_root)
    }
    for planned in _iter_target_specs(
        matrix_root=resolved_matrix_root,
        molecule_backbone_root=resolved_molecule_backbone_root,
        molecule_group_root=resolved_molecule_group_root,
        seed=int(seed),
        budget_steps=budget_steps,
        path_base=manifest_path_base,
    ):
        if str(planned.benchmark_family) == MOLECULE_FAMILY:
            member = molecule_member_by_key[
                (str(planned.dataset_key), str(planned.member_key), str(planned.stratum))
            ]
            resolved = _existing_molecule_artifact(
                resolved_matrix_root,
                molecule_backbone_root=resolved_molecule_backbone_root,
                member=member,
                train_steps=int(planned.train_steps),
                seed=int(seed),
                path_base=manifest_path_base,
            )
        else:
            resolved = _existing_matrix_artifact(
                resolved_matrix_root,
                backbone_name=str(planned.backbone_name),
                benchmark_family=str(planned.benchmark_family),
                dataset_key=str(planned.dataset_key),
                train_steps=int(planned.train_steps),
                seed=int(seed),
                path_base=manifest_path_base,
            )
            if resolved is None and str(planned.backbone_name) == BACKBONE_NAME_OTFLOW:
                resolved = _existing_otflow_reuse_artifact(
                    resolved_reuse_root,
                    benchmark_family=str(planned.benchmark_family),
                    dataset_key=str(planned.dataset_key),
                    train_steps=int(planned.train_steps),
                    seed=int(seed),
                    path_base=manifest_path_base,
                )
        artifact = planned if resolved is None else resolved
        if str(artifact.status) == "ready":
            ready_count += 1
        artifacts.append(artifact.to_dict())
    payload = {
        "version": MANIFEST_VERSION,
        "path_base": MANIFEST_PARENT_PATH_BASE,
        "seed": int(seed),
        "train_budget_steps": [int(value) for value in budget_steps],
        "matrix_root": _project_display_path(resolved_matrix_root, path_base=manifest_path_base),
        "otflow_reuse_root": _project_display_path(resolved_reuse_root, path_base=manifest_path_base),
        "imported_backbone_root": _project_display_path(resolved_import_root, path_base=manifest_path_base),
        "molecule_group_root": _project_display_path(resolved_molecule_group_root, path_base=manifest_path_base),
        "molecule_backbone_root": _project_display_path(
            resolved_molecule_backbone_root,
            path_base=manifest_path_base,
        ),
        "temporal_artifact_count": int(
            sum(1 for row in artifacts if str(row.get("benchmark_family")) in {FORECAST_FAMILY, CONDITIONAL_GENERATION_FAMILY})
        ),
        "molecule_stratum_count": int(len(molecule_member_by_key)),
        "molecule_artifact_count": int(
            sum(1 for row in artifacts if str(row.get("benchmark_family")) == MOLECULE_FAMILY)
        ),
        "artifact_count": int(len(artifacts)),
        "ready_count": int(ready_count),
        "missing_count": int(len(artifacts) - ready_count),
        "artifacts": artifacts,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _validated_manifest_path_base(manifest_path: Path, path_base: str) -> Path:
    return resolve_manifest_path_base(manifest_path, path_base)


def _resolve_manifest_relative_path(manifest_path: Path, value: Any, *, path_base: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    base = _validated_manifest_path_base(manifest_path, path_base)
    return str(resolve_portable_relative_path(base, value, label="Backbone manifest path"))


def load_backbone_manifest(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    path_base = str(payload.get("path_base", "") or "").strip()
    if not path_base:
        raise ValueError("Backbone manifest requires path_base='manifest_parent'.")
    _validated_manifest_path_base(resolved, path_base)
    for artifact in payload.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        for field in _MANIFEST_ARTIFACT_PATH_FIELDS:
            if field in artifact:
                artifact[field] = _resolve_manifest_relative_path(resolved, artifact[field], path_base=path_base)
    for field in _MANIFEST_ROOT_PATH_FIELDS:
        if field in payload:
            payload[field] = _resolve_manifest_relative_path(resolved, payload[field], path_base=path_base)
    return payload


def find_backbone_artifact(
    manifest_payload: Mapping[str, Any],
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    status: str = "ready",
    member_key: str | None = None,
    stratum: str | None = None,
) -> Dict[str, Any]:
    if str(benchmark_family) == MOLECULE_FAMILY and member_key is None and stratum is None:
        raise ValueError("Molecule backbone artifact lookup requires member_key or stratum to avoid ambiguous group matches.")
    matches: List[Dict[str, Any]] = []
    for artifact in manifest_payload.get("artifacts", []):
        if (
            str(artifact.get("backbone_name")) == str(backbone_name)
            and str(artifact.get("benchmark_family")) == str(benchmark_family)
            and str(artifact.get("dataset_key")) == str(dataset_key)
            and int(artifact.get("train_steps", -1)) == int(train_steps)
            and str(artifact.get("status")) == str(status)
            and (member_key is None or str(artifact.get("member_key", "")) == str(member_key))
            and (stratum is None or str(artifact.get("stratum", "")) == str(stratum))
        ):
            matches.append(dict(artifact))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Backbone artifact lookup is ambiguous; add a narrower molecule member_key or stratum filter."
        )
    raise KeyError(
        "No matching backbone artifact found for "
        f"{backbone_name}/{benchmark_family}/{dataset_key}/{int(train_steps)} with status={status}"
    )


def build_runtime_probe(
    *,
    dataset_root: str | Path | None = None,
    lobster_profile_path: str | Path | None = None,
    long_term_st_path: str | Path | None = None,
    molecule_group_root: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_dataset_root = Path(dataset_root or project_paper_dataset_root()).resolve()
    resolved_lobster_profile_path = Path(lobster_profile_path or lobster_synthetic_profile_path()).resolve()
    resolved_long_term_st_path = Path(long_term_st_path or long_term_st_manifest_path().parent).resolve()
    resolved_molecule_group_root = Path(molecule_group_root or project_molecule_group_root()).resolve()
    monash_root = resolved_dataset_root / "monash"
    import_names = ("numpy", "torch", "wfdb")
    imports = {
        name: bool(importlib.util.find_spec(name) is not None)
        for name in import_names
    }
    forecast_dataset_presence = {
        str(dataset_key): bool((monash_root / str(dataset_key) / "manifest.json").exists())
        for dataset_key in PAPER_FORECAST_DATASETS
    }
    dataset_presence = {
        "monash_manifests": forecast_dataset_presence,
        LONG_TERM_ST_DATASET_KEY: bool(long_term_st_manifest_path(resolved_long_term_st_path).exists()),
        "cryptos_npz": bool((project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz").exists()),
        "lobster_synthetic_profile": bool(resolved_lobster_profile_path.exists()),
        "molecule_group_manifests": {
            str(dataset_key): bool((resolved_molecule_group_root / str(dataset_key) / "group_manifest.json").exists())
            for dataset_key in MOLECULE_GROUP_DATASET_KEYS
        },
    }
    return {
        "dataset_root": _project_display_path(resolved_dataset_root),
        "lobster_synthetic_profile_name": str(resolved_lobster_profile_path.name),
        "long_term_st_prepared_dir": str(resolved_long_term_st_path.name),
        "molecule_group_root": _project_display_path(resolved_molecule_group_root),
        "imports": imports,
        "dataset_presence": dataset_presence,
    }


def build_backbone_readiness_audit(
    *,
    matrix_root: str | Path | None = None,
    otflow_reuse_root: str | Path | None = None,
    imported_backbone_root: str | Path | None = None,
    molecule_group_root: str | Path | None = None,
    molecule_backbone_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    lobster_profile_path: str | Path | None = None,
    long_term_st_path: str | Path | None = None,
    budget_steps: Sequence[int] = TRAIN_BUDGET_STEPS,
    seed: int = DEFAULT_SEED,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    normalization = normalize_imported_backbone_artifacts(
        matrix_root=matrix_root,
        imported_root=imported_backbone_root,
        budget_steps=budget_steps,
    )
    manifest = materialize_backbone_manifest(
        matrix_root=matrix_root,
        otflow_reuse_root=otflow_reuse_root,
        imported_backbone_root=imported_backbone_root,
        molecule_group_root=molecule_group_root,
        molecule_backbone_root=molecule_backbone_root,
        budget_steps=budget_steps,
        seed=int(seed),
        write_path=write_path,
    )
    readiness = {
        "version": AUDIT_VERSION,
        "manifest_path": _project_display_path(Path(write_path or backbone_manifest_path()).resolve()),
        "manifest": manifest,
        "normalization": normalization,
        "runtime_probe": build_runtime_probe(
            dataset_root=dataset_root,
            lobster_profile_path=lobster_profile_path,
            long_term_st_path=long_term_st_path,
            molecule_group_root=molecule_group_root,
        ),
    }
    return readiness


__all__ = [
    "ACTIVE_FORECAST_BACKBONE_BUDGETS",
    "ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS",
    "AUDIT_VERSION",
    "BACKBONE_NAME_OTFLOW",
    "BACKBONE_NAME_OTFLOW_MOLECULE",
    "MOLECULE_ROLLOUT_MODE",
    "DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE",
    "DEFAULT_MOLECULE_VARIANT",
    "DEFAULT_SEED",
    "IMPORTED_EXTERNAL_SOURCE_KIND",
    "MANIFEST_VERSION",
    "MOLECULE_FAMILY",
    "STANDARD_ARTIFACT_SUMMARY_NAME",
    "TRAIN_BUDGET_STEPS",
    "BackboneArtifactSpec",
    "build_backbone_checkpoint_id",
    "build_backbone_readiness_audit",
    "expected_artifact_root",
    "find_backbone_artifact",
    "load_backbone_manifest",
    "materialize_backbone_manifest",
    "normalize_imported_backbone_artifacts",
    "project_imported_otflow_backbone_root",
    "project_molecule_backbone_root",
    "project_otflow_reuse_root",
    "train_budget_label",
]
