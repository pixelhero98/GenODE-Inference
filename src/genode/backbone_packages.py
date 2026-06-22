from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from genode.canonical_experiment_layout import (
    CANONICAL_CHECKPOINT_STEPS,
    CONDITIONAL_GENERATION_SCENARIO_KEYS,
    FORECAST_SCENARIO_KEYS,
    MOLECULE_SCENARIO_KEYS,
    SCENARIO_FAMILY_MOLECULE,
)
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY
from genode.data.otflow_paths import project_root, resolve_project_path

PACKAGE_SCHEMA_VERSION = "genode_backbone_package_v1"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
PACKAGED_BACKBONE_MANIFEST = "outputs/backbone_matrix/backbone_manifest.json"
PATH_BASE_FROM_BACKBONE_MANIFEST = "../.."
MANIFEST_VERSION = "fm_backbone_manifest"
MOLECULE_FAMILY = SCENARIO_FAMILY_MOLECULE
TRAIN_BUDGET_STEPS = CANONICAL_CHECKPOINT_STEPS

PATH_FIELDS = ("checkpoint_path", "summary_path", "metadata_path")
ROOT_FIELDS = (
    "matrix_root",
    "otflow_reuse_root",
    "imported_backbone_root",
    "molecule_group_root",
    "molecule_backbone_root",
)
LOCAL_PATH_MARKERS = (
    "C" + ":/",
    "C" + ":\\",
    "/" + "home" + "/",
    "/" + "scratch" + "/",
    "/" + "projects" + "/",
    "/" + "Users" + "/",
)
REDACTED_LOCAL_PATH = "<local_path_redacted>"
MIN_CHECKPOINT_SIZE_BYTES = 1024


@dataclass(frozen=True)
class BackbonePackageFamily:
    key: str
    benchmark_family: str
    scenarios: Sequence[str]
    zip_name: str
    data_roots: Sequence[str]
    expected_artifact_count: int


FAMILY_SPECS: Mapping[str, BackbonePackageFamily] = {
    "temporal-extrapolation": BackbonePackageFamily(
        key="temporal-extrapolation",
        benchmark_family=FORECAST_FAMILY,
        scenarios=FORECAST_SCENARIO_KEYS,
        zip_name="genode_temporal_extrapolation_backbones_datasets.zip",
        data_roots=tuple(f"paper_datasets/monash/{scenario}" for scenario in FORECAST_SCENARIO_KEYS),
        expected_artifact_count=len(FORECAST_SCENARIO_KEYS) * len(TRAIN_BUDGET_STEPS),
    ),
    "temporal-generation": BackbonePackageFamily(
        key="temporal-generation",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        scenarios=CONDITIONAL_GENERATION_SCENARIO_KEYS,
        zip_name="genode_temporal_generation_backbones_datasets.zip",
        data_roots=(
            "data/cryptos_binance_spot_monthly_1s_l10.npz",
            "data/lobster_synthetic",
            "data/long_term_st_100hz_context_only",
        ),
        expected_artifact_count=len(CONDITIONAL_GENERATION_SCENARIO_KEYS) * len(TRAIN_BUDGET_STEPS),
    ),
    "molecule-coord-generation": BackbonePackageFamily(
        key="molecule-coord-generation",
        benchmark_family=MOLECULE_FAMILY,
        scenarios=MOLECULE_SCENARIO_KEYS,
        zip_name="genode_molecule_coord_generation_backbones_datasets.zip",
        data_roots=("data/molecule_3d",),
        expected_artifact_count=len(MOLECULE_SCENARIO_KEYS) * 6 * len(TRAIN_BUDGET_STEPS),
    ),
}
EXPECTED_FULL_BACKBONE_ARTIFACT_COUNT = sum(int(spec.expected_artifact_count) for spec in FAMILY_SPECS.values())


def _as_posix(path: str | Path) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).as_posix()


def _safe_rel(path: str | Path) -> PurePosixPath:
    rel = PurePosixPath(_as_posix(path))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0].endswith(":"):
        raise ValueError(f"Unsafe package-relative path: {path!r}")
    return rel


def _strip_known_prefix(path_text: str) -> str:
    text = _as_posix(path_text)
    parts = PurePosixPath(text).parts
    if len(parts) >= 2 and parts[0] == "genode" and parts[1] in {"outputs", "data", "paper_datasets"}:
        return PurePosixPath(*parts[1:]).as_posix()
    for marker in ("outputs", "data", "paper_datasets"):
        if marker in parts:
            return PurePosixPath(*parts[parts.index(marker) :]).as_posix()
    return text


def _source_path(source_root: Path, path_text: str | Path) -> Path:
    raw = Path(path_text)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(source_root / raw)
        stripped = _strip_known_prefix(str(path_text))
        candidates.append(source_root / stripped)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _package_rel_for_path(path_text: str | Path) -> str:
    stripped = _strip_known_prefix(str(path_text))
    _safe_rel(stripped)
    return stripped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _guard_package_delete(package_root: Path, output_dir: Path, expected_basename: str) -> None:
    resolved = package_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved == resolved.anchor or resolved.parent == resolved:
        raise ValueError(f"Refusing to delete unsafe package staging directory: {resolved}")
    if _is_relative_to(resolved, resolved_output):
        return
    if resolved.name == expected_basename:
        return
    raise ValueError(
        f"Refusing to delete package staging directory outside output_dir without expected basename {expected_basename!r}: {resolved}"
    )


def _resolve_manifest_relative_path(manifest_path: Path, value: Any, *, path_base: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    raw = Path(value)
    if raw.is_absolute():
        return str(raw)
    base = (manifest_path.parent / str(path_base)).resolve()
    return str((base / _strip_known_prefix(value)).resolve())


def load_portable_backbone_manifest(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    path_base = str(payload.get("path_base", "") or "").strip()
    if path_base:
        for artifact in payload.get("artifacts", []):
            if not isinstance(artifact, dict):
                continue
            for field in PATH_FIELDS:
                if field in artifact:
                    artifact[field] = _resolve_manifest_relative_path(resolved, artifact[field], path_base=path_base)
        for field in ROOT_FIELDS:
            if field in payload:
                payload[field] = _resolve_manifest_relative_path(resolved, payload[field], path_base=path_base)
    return payload


def _copy_json_or_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() != ".json":
        shutil.copy2(src, dst)
        return
    payload = json.loads(src.read_text(encoding="utf-8"))
    rewritten = _rewrite_json_paths(payload)
    dst.write_text(json.dumps(rewritten, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_json_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _rewrite_json_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_json_paths(item) for item in value]
    if isinstance(value, str):
        text = _as_posix(value)
        if any(marker in text for marker in ("outputs/", "data/", "paper_datasets/", "genode/outputs/")):
            stripped = _strip_known_prefix(text)
            try:
                _safe_rel(stripped)
            except ValueError:
                return value
            return stripped
        if _contains_local_marker(text):
            stripped = _strip_known_prefix(text)
            if stripped != text:
                try:
                    _safe_rel(stripped)
                except ValueError:
                    return REDACTED_LOCAL_PATH
                return stripped
            return REDACTED_LOCAL_PATH
    return value


def _copy_tree_or_file(source_root: Path, package_root: Path, rel_path: str) -> List[Dict[str, Any]]:
    src = _source_path(source_root, rel_path)
    if not src.exists():
        raise FileNotFoundError(f"Package source path is missing: {rel_path} resolved to {src}")
    copied: List[Dict[str, Any]] = []
    if src.is_file():
        dst = package_root / _safe_rel(rel_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(_file_record(package_root, dst, role="dataset"))
        return copied
    for file_path in sorted(path for path in src.rglob("*") if path.is_file()):
        relative_tail = file_path.relative_to(src)
        dst = package_root / _safe_rel(rel_path) / relative_tail
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dst)
        copied.append(_file_record(package_root, dst, role="dataset"))
    return copied


def _file_record(package_root: Path, path: Path, *, role: str) -> Dict[str, Any]:
    rel = path.relative_to(package_root).as_posix()
    return {
        "path": rel,
        "role": role,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _artifact_belongs_to_family(artifact: Mapping[str, Any], spec: BackbonePackageFamily) -> bool:
    return (
        str(artifact.get("benchmark_family")) == str(spec.benchmark_family)
        and str(artifact.get("dataset_key")) in set(str(scenario) for scenario in spec.scenarios)
        and str(artifact.get("status")) == "ready"
    )


def _scenario_artifact_counts(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for artifact in artifacts:
        scenario = str(artifact.get("dataset_key", ""))
        counts[scenario] = counts.get(scenario, 0) + 1
    return counts


def _artifact_identity(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    family = str(artifact.get("benchmark_family", ""))
    if family == MOLECULE_FAMILY:
        return (
            family,
            str(artifact.get("dataset_key", "")),
            str(artifact.get("member_key", "")),
            str(artifact.get("stratum", "")),
            str(artifact.get("variant", "")),
            str(int(artifact.get("train_steps", -1))),
        )
    return (
        family,
        str(artifact.get("dataset_key", "")),
        str(int(artifact.get("train_steps", -1))),
    )


def _validate_artifact_grid(artifacts: Sequence[Mapping[str, Any]], spec: BackbonePackageFamily) -> List[str]:
    errors: List[str] = []
    if len(artifacts) != int(spec.expected_artifact_count):
        errors.append(
            f"Package family {spec.key!r} has {len(artifacts)} artifacts; expected {int(spec.expected_artifact_count)}."
        )
    identities = [_artifact_identity(artifact) for artifact in artifacts]
    if len(set(identities)) != len(identities):
        errors.append(f"Package family {spec.key!r} contains duplicate artifact identities.")
    expected_steps = {int(step) for step in TRAIN_BUDGET_STEPS}
    if spec.benchmark_family == MOLECULE_FAMILY:
        members_by_scenario: Dict[str, set[tuple[str, str, str]]] = {str(scenario): set() for scenario in spec.scenarios}
        steps_by_member: Dict[tuple[str, str, str, str], set[int]] = {}
        for artifact in artifacts:
            scenario = str(artifact.get("dataset_key", ""))
            member = (
                str(artifact.get("member_key", "")),
                str(artifact.get("stratum", "")),
                str(artifact.get("variant", "")),
            )
            members_by_scenario.setdefault(scenario, set()).add(member)
            steps_by_member.setdefault((scenario, *member), set()).add(int(artifact.get("train_steps", -1)))
        for scenario in spec.scenarios:
            member_count = len(members_by_scenario.get(str(scenario), set()))
            if member_count != 6:
                errors.append(f"Scenario {scenario} has {member_count} molecule members; expected 6.")
        for member, observed_steps in sorted(steps_by_member.items()):
            if observed_steps != expected_steps:
                errors.append(f"Molecule member {'/'.join(member)} has train steps {sorted(observed_steps)}; expected {sorted(expected_steps)}.")
    else:
        steps_by_scenario: Dict[str, set[int]] = {str(scenario): set() for scenario in spec.scenarios}
        for artifact in artifacts:
            steps_by_scenario.setdefault(str(artifact.get("dataset_key", "")), set()).add(int(artifact.get("train_steps", -1)))
        for scenario in spec.scenarios:
            observed_steps = steps_by_scenario.get(str(scenario), set())
            if observed_steps != expected_steps:
                errors.append(f"Scenario {scenario} has train steps {sorted(observed_steps)}; expected {sorted(expected_steps)}.")
    return errors


def _normalize_artifact_for_package(
    artifact: Mapping[str, Any],
    *,
    source_root: Path,
    package_root: Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    normalized = dict(artifact)
    records: List[Dict[str, Any]] = []
    for field in PATH_FIELDS:
        raw_value = str(artifact.get(field, "") or "").strip()
        if not raw_value:
            continue
        rel = _package_rel_for_path(raw_value)
        src = _source_path(source_root, raw_value)
        if not src.exists():
            raise FileNotFoundError(f"Artifact {artifact.get('checkpoint_id', '')} missing {field}: {src}")
        dst = package_root / _safe_rel(rel)
        _copy_json_or_file(src, dst)
        normalized[field] = rel
        records.append(_file_record(package_root, dst, role=f"artifact:{field}"))
    return normalized, records


def _normalize_manifest_for_package(
    source_manifest: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    spec: BackbonePackageFamily,
) -> Dict[str, Any]:
    payload = {key: value for key, value in source_manifest.items() if key != "artifacts"}
    payload.update(
        {
            "version": MANIFEST_VERSION,
            "package_schema_version": PACKAGE_SCHEMA_VERSION,
            "package_family": spec.key,
            "path_base": PATH_BASE_FROM_BACKBONE_MANIFEST,
            "matrix_root": "outputs/backbone_matrix",
            "otflow_reuse_root": "outputs/shared_backbones/otflow_fullhorizon_seed0",
            "imported_backbone_root": "outputs/imported_backbones/otflow",
            "molecule_group_root": "data/molecule_3d",
            "molecule_backbone_root": "outputs/molecule_3d_backbones",
            "temporal_artifact_count": int(
                sum(1 for row in artifacts if str(row.get("benchmark_family")) in {FORECAST_FAMILY, CONDITIONAL_GENERATION_FAMILY})
            ),
            "molecule_artifact_count": int(sum(1 for row in artifacts if str(row.get("benchmark_family")) == MOLECULE_FAMILY)),
            "artifact_count": int(len(artifacts)),
            "ready_count": int(len(artifacts)),
            "missing_count": 0,
            "artifacts": [dict(row) for row in artifacts],
        }
    )
    for field in ROOT_FIELDS:
        if field in payload and isinstance(payload[field], str):
            payload[field] = _strip_known_prefix(str(payload[field]))
    return payload


def _read_git_commit(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def package_backbone_family(
    *,
    family: str,
    source_root: str | Path | None = None,
    output_dir: str | Path,
    stage_dir: str | Path | None = None,
    overwrite: bool = False,
    make_zip: bool = True,
) -> Dict[str, Any]:
    spec = FAMILY_SPECS[str(family)]
    resolved_source = Path(source_root or project_root()).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    package_root = Path(stage_dir).expanduser().resolve() if stage_dir else resolved_output / spec.zip_name.removesuffix(".zip")
    if package_root.exists():
        if not overwrite:
            raise FileExistsError(f"Package staging directory already exists: {package_root}")
        _guard_package_delete(package_root, resolved_output, spec.zip_name.removesuffix(".zip"))
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    source_manifest_path = _source_path(resolved_source, PACKAGED_BACKBONE_MANIFEST)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if int(source_manifest.get("ready_count", -1)) != int(source_manifest.get("artifact_count", -2)):
        raise ValueError(
            "Source backbone manifest is not fully ready; expected ready_count == artifact_count before packaging."
        )
    if int(source_manifest.get("artifact_count", -1)) != int(EXPECTED_FULL_BACKBONE_ARTIFACT_COUNT):
        raise ValueError(
            f"Source backbone manifest has artifact_count={source_manifest.get('artifact_count')}; "
            f"expected {EXPECTED_FULL_BACKBONE_ARTIFACT_COUNT} canonical artifacts before packaging."
        )
    artifacts = [row for row in source_manifest.get("artifacts", []) if _artifact_belongs_to_family(row, spec)]
    if not artifacts:
        raise ValueError(f"No ready artifacts found for package family {family!r}.")
    artifact_grid_errors = _validate_artifact_grid(artifacts, spec)
    if artifact_grid_errors:
        raise ValueError("Source backbone manifest is incomplete for this family:\n- " + "\n- ".join(artifact_grid_errors))

    records: List[Dict[str, Any]] = []
    normalized_artifacts: List[Dict[str, Any]] = []
    for artifact in artifacts:
        normalized, artifact_records = _normalize_artifact_for_package(
            artifact,
            source_root=resolved_source,
            package_root=package_root,
        )
        normalized_artifacts.append(normalized)
        records.extend(artifact_records)
    data_records: List[Dict[str, Any]] = []
    for rel_path in spec.data_roots:
        data_records.extend(_copy_tree_or_file(resolved_source, package_root, rel_path))
    records.extend(data_records)

    packaged_manifest = _normalize_manifest_for_package(source_manifest, normalized_artifacts, spec=spec)
    backbone_manifest_path = package_root / PACKAGED_BACKBONE_MANIFEST
    backbone_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    backbone_manifest_path.write_text(json.dumps(packaged_manifest, indent=2, sort_keys=True), encoding="utf-8")
    records.append(_file_record(package_root, backbone_manifest_path, role="backbone_manifest"))

    package_manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "family": spec.key,
        "benchmark_family": spec.benchmark_family,
        "scenarios": list(spec.scenarios),
        "source_commit": _read_git_commit(resolved_source),
        "backbone_manifest": PACKAGED_BACKBONE_MANIFEST,
        "artifact_count": int(len(normalized_artifacts)),
        "expected_artifact_count": int(spec.expected_artifact_count),
        "artifacts_by_scenario": _scenario_artifact_counts(normalized_artifacts),
        "data_roots": list(spec.data_roots),
        "dataset_root_count": int(len(spec.data_roots)),
        "files": sorted(records, key=lambda row: str(row["path"])),
    }
    package_manifest_path = package_root / PACKAGE_MANIFEST_NAME
    package_manifest_path.write_text(json.dumps(package_manifest, indent=2, sort_keys=True), encoding="utf-8")

    zip_path = resolved_output / spec.zip_name
    zip_manifest_path = zip_path.with_suffix(zip_path.suffix + ".manifest.json")
    if make_zip:
        resolved_output.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            if not overwrite:
                raise FileExistsError(f"Package zip already exists: {zip_path}")
            zip_path.unlink()
        if zip_manifest_path.exists():
            if not overwrite:
                raise FileExistsError(f"Package zip manifest already exists: {zip_manifest_path}")
            zip_manifest_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for path in sorted(file_path for file_path in package_root.rglob("*") if file_path.is_file()):
                zf.write(path, path.relative_to(package_root).as_posix())
        zip_manifest = {
            **package_manifest,
            "zip_name": zip_path.name,
            "zip_size_bytes": int(zip_path.stat().st_size),
            "zip_sha256": _sha256_file(zip_path),
        }
        zip_manifest_path.write_text(json.dumps(zip_manifest, indent=2, sort_keys=True), encoding="utf-8")
        package_manifest = zip_manifest
    return {
        "status": "complete",
        "package_root": str(package_root),
        "zip_path": str(zip_path) if make_zip else "",
        "zip_manifest_path": str(zip_manifest_path) if make_zip else "",
        "manifest": package_manifest,
    }


def _contains_local_marker(value: str) -> bool:
    text = value.replace("\\", "/")
    return any(marker.lower().replace("\\", "/") in text.lower() for marker in LOCAL_PATH_MARKERS)


def _iter_json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_strings(item)
    elif isinstance(value, str):
        yield value


def _validate_file_record(package_root: Path, record: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    rel = str(record.get("path", ""))
    try:
        safe = _safe_rel(rel)
    except ValueError as exc:
        return [str(exc)]
    path = package_root / safe
    if not path.exists():
        errors.append(f"Missing file listed in package manifest: {rel}")
        return errors
    expected_size = int(record.get("size_bytes", -1))
    if expected_size >= 0 and int(path.stat().st_size) != expected_size:
        errors.append(f"Size mismatch for {rel}")
    expected_hash = str(record.get("sha256", "") or "")
    if expected_hash and _sha256_file(path) != expected_hash:
        errors.append(f"SHA256 mismatch for {rel}")
    return errors


def _known_text_checkpoint_header(header: bytes) -> bool:
    stripped = header.lstrip().lower()
    return (
        stripped.startswith(b"version https://git-lfs.github.com/spec")
        or stripped.startswith(b"<!doctype html")
        or stripped.startswith(b"<html")
    )


def _checkpoint_label(artifact: Mapping[str, Any]) -> str:
    parts = [
        str(artifact.get("checkpoint_id", "") or "").strip(),
        str(artifact.get("benchmark_family", "") or "").strip(),
        str(artifact.get("dataset_key", "") or "").strip(),
        str(artifact.get("train_steps", "") or "").strip(),
    ]
    return "/".join(part for part in parts if part)


def _validate_artifact_checkpoint_integrity(artifact: Mapping[str, Any], checkpoint_path: Path) -> List[str]:
    label = _checkpoint_label(artifact)
    errors: List[str] = []
    if not checkpoint_path.exists():
        return [f"Provided artifact {label} is missing checkpoint_path: {checkpoint_path}"]
    if not checkpoint_path.is_file():
        return [f"Provided artifact {label} checkpoint_path is not a file: {checkpoint_path}"]
    size_bytes = int(checkpoint_path.stat().st_size)
    if size_bytes < MIN_CHECKPOINT_SIZE_BYTES:
        errors.append(
            f"Provided artifact {label} checkpoint is too small to be valid: "
            f"{checkpoint_path} has {size_bytes} bytes."
        )
    with checkpoint_path.open("rb") as fh:
        header = fh.read(256)
    if _known_text_checkpoint_header(header):
        errors.append(f"Provided artifact {label} checkpoint looks like text or a pointer: {checkpoint_path}")
    if errors:
        return errors
    if (
        str(artifact.get("backbone_name", "")) == "otflow"
        and str(artifact.get("benchmark_family", "")) in {FORECAST_FAMILY, CONDITIONAL_GENERATION_FAMILY}
    ):
        try:
            from genode.evaluation.otflow_evaluation_support import load_otflow_checkpoint_payload

            load_otflow_checkpoint_payload(
                checkpoint_path,
                expected_identity=f"provided backbone artifact {label}",
            )
        except Exception as exc:
            errors.append(f"Provided artifact {label} checkpoint is not loadable: {exc}")
    return errors


def validate_backbone_package(
    package_root: str | Path,
    *,
    expected_family: str | None = None,
    require_clean_paths: bool = True,
) -> Dict[str, Any]:
    root = Path(package_root).expanduser().resolve()
    errors: List[str] = []
    package_manifest_path = root / PACKAGE_MANIFEST_NAME
    if not package_manifest_path.exists():
        errors.append(f"Missing {PACKAGE_MANIFEST_NAME}")
        return {"status": "failed", "errors": errors}
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    if str(package_manifest.get("schema_version")) != PACKAGE_SCHEMA_VERSION:
        errors.append("Unexpected package schema version.")
    family = str(package_manifest.get("family", ""))
    if expected_family and family != str(expected_family):
        errors.append(f"Package family {family!r} != expected {expected_family!r}.")
    if family not in FAMILY_SPECS:
        errors.append(f"Unknown package family {family!r}.")
        spec = None
    else:
        spec = FAMILY_SPECS[family]
        if list(package_manifest.get("scenarios", [])) != list(spec.scenarios):
            errors.append("Package scenarios do not match the canonical family scenarios.")
        if int(package_manifest.get("expected_artifact_count", -1)) != int(spec.expected_artifact_count):
            errors.append("Package expected artifact count does not match the canonical family.")
        if list(package_manifest.get("data_roots", [])) != list(spec.data_roots):
            errors.append("Package data roots do not match the canonical family data roots.")
        file_paths = [str(record.get("path", "")) for record in package_manifest.get("files", [])]
        if len(set(file_paths)) != len(file_paths):
            errors.append("Package manifest contains duplicate file records.")
        listed_paths = set(file_paths) | {PACKAGE_MANIFEST_NAME}
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        unlisted_paths = sorted(actual_paths - listed_paths)
        if unlisted_paths:
            errors.append(f"Package contains files missing from hash manifest: {unlisted_paths[:10]}")
        for data_root in spec.data_roots:
            if not any(path == data_root or path.startswith(f"{data_root}/") for path in file_paths):
                errors.append(f"Package data root has no listed files: {data_root}")
    for record in package_manifest.get("files", []):
        errors.extend(_validate_file_record(root, record))
    backbone_manifest_path = root / PACKAGED_BACKBONE_MANIFEST
    if not backbone_manifest_path.exists():
        errors.append(f"Missing packaged backbone manifest: {PACKAGED_BACKBONE_MANIFEST}")
    else:
        raw_backbone_manifest = json.loads(backbone_manifest_path.read_text(encoding="utf-8"))
        if str(raw_backbone_manifest.get("path_base", "")) != PATH_BASE_FROM_BACKBONE_MANIFEST:
            errors.append("Packaged backbone manifest has an unexpected path_base.")
        for artifact in raw_backbone_manifest.get("artifacts", []):
            if not isinstance(artifact, Mapping):
                continue
            for field in PATH_FIELDS:
                raw_path = str(artifact.get(field, "") or "")
                if raw_path:
                    try:
                        _safe_rel(raw_path)
                    except ValueError as exc:
                        errors.append(f"Artifact {artifact.get('checkpoint_id', '')} has unsafe {field}: {exc}")
        backbone_manifest = load_portable_backbone_manifest(backbone_manifest_path)
        if int(backbone_manifest.get("ready_count", -1)) != int(backbone_manifest.get("artifact_count", -2)):
            errors.append("Packaged backbone manifest is not fully ready.")
        if spec is not None:
            artifacts = list(backbone_manifest.get("artifacts", []))
            if int(package_manifest.get("artifact_count", -1)) != len(artifacts):
                errors.append("Package artifact count does not match packaged backbone manifest.")
            errors.extend(_validate_artifact_grid(artifacts, spec))
            observed_counts = _scenario_artifact_counts(artifacts)
            for scenario in spec.scenarios:
                if observed_counts.get(str(scenario), 0) <= 0:
                    errors.append(f"Package has no ready artifacts for scenario: {scenario}")
            for artifact in artifacts:
                if not _artifact_belongs_to_family(artifact, spec):
                    errors.append(f"Unexpected artifact in package: {artifact.get('checkpoint_id', artifact)}")
                for field in PATH_FIELDS:
                    path_value = str(artifact.get(field, "") or "")
                    if not path_value:
                        continue
                    if Path(path_value).is_absolute():
                        resolved = Path(path_value)
                    else:
                        resolved = root / _safe_rel(_strip_known_prefix(path_value))
                    if not resolved.exists():
                        errors.append(f"Artifact {artifact.get('checkpoint_id', '')} missing {field}: {path_value}")
                    elif field == "checkpoint_path":
                        errors.extend(_validate_artifact_checkpoint_integrity(artifact, resolved))
    if require_clean_paths:
        for json_path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON {json_path.relative_to(root).as_posix()}: {exc}")
                continue
            for text in _iter_json_strings(payload):
                if _contains_local_marker(text):
                    errors.append(f"Local path marker found in {json_path.relative_to(root).as_posix()}: {text}")
                    break
    return {
        "status": "complete" if not errors else "failed",
        "family": family,
        "package_root": str(root),
        "errors": errors,
        "artifact_count": int(package_manifest.get("artifact_count", 0) or 0),
        "file_count": int(len(package_manifest.get("files", []) or [])),
    }


def _resolve_loaded_artifact_path(path_value: str) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return resolve_project_path(str(path_value))


def validate_provided_backbone_manifest(
    manifest_path: str | Path,
    *,
    scenario_key: str = "",
    benchmark_family: str = "",
) -> Dict[str, Any]:
    resolved_manifest = resolve_project_path(str(manifest_path))
    errors: List[str] = []
    if not resolved_manifest.exists():
        return {"status": "failed", "errors": [f"Provided backbone manifest is missing: {resolved_manifest}"]}
    manifest = load_portable_backbone_manifest(resolved_manifest)
    matching_artifacts = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if str(artifact.get("status")) == "ready"
        and (not scenario_key or str(artifact.get("dataset_key")) == str(scenario_key))
        and (not benchmark_family or str(artifact.get("benchmark_family")) == str(benchmark_family))
    ]
    if not matching_artifacts:
        errors.append(f"No ready provided backbone artifacts match scenario={scenario_key!r}, family={benchmark_family!r}.")
    expected_steps = {int(step) for step in TRAIN_BUDGET_STEPS}
    observed_steps = {int(artifact.get("train_steps", -1)) for artifact in matching_artifacts}
    if matching_artifacts and not expected_steps.issubset(observed_steps):
        errors.append(f"Provided backbone artifacts have train steps {sorted(observed_steps)}; expected at least {sorted(expected_steps)}.")
    if benchmark_family == MOLECULE_FAMILY:
        members: Dict[tuple[str, str, str], set[int]] = {}
        for artifact in matching_artifacts:
            member = (
                str(artifact.get("member_key", "")),
                str(artifact.get("stratum", "")),
                str(artifact.get("variant", "")),
            )
            members.setdefault(member, set()).add(int(artifact.get("train_steps", -1)))
        if matching_artifacts and len(members) != 6:
            errors.append(f"Provided molecule scenario has {len(members)} members; expected 6.")
        for member, member_steps in sorted(members.items()):
            if member_steps != expected_steps:
                errors.append(f"Provided molecule member {'/'.join(member)} has train steps {sorted(member_steps)}; expected {sorted(expected_steps)}.")
    for artifact in matching_artifacts:
        for field in PATH_FIELDS:
            value = str(artifact.get(field, "") or "")
            if not value:
                continue
            resolved = _resolve_loaded_artifact_path(value)
            if not resolved.exists():
                errors.append(f"Provided artifact {artifact.get('checkpoint_id', '')} is missing {field}: {value}")
            elif field == "checkpoint_path":
                errors.extend(_validate_artifact_checkpoint_integrity(artifact, resolved))
    return {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "manifest_path": str(resolved_manifest),
        "artifact_count": int(len(matching_artifacts)),
    }


def apply_backbone_package_to_args(args: argparse.Namespace) -> argparse.Namespace:
    raw_root = str(getattr(args, "backbone_package_root", "") or "").strip()
    if not raw_root:
        return args
    package_root = resolve_project_path(raw_root)
    validation = validate_backbone_package(package_root)
    if validation["status"] != "complete":
        raise ValueError("Invalid backbone package:\n- " + "\n- ".join(validation["errors"]))
    family = str(validation.get("family", ""))
    spec = FAMILY_SPECS[family]
    scenario = str(getattr(args, "scenario_key", "") or getattr(args, "dataset", ""))
    if scenario and scenario not in set(spec.scenarios):
        raise ValueError(f"Scenario {scenario!r} is not included in backbone package family {family!r}.")
    args.backbone_manifest = str(package_root / PACKAGED_BACKBONE_MANIFEST)
    args.shared_backbone_root = str(package_root / "outputs" / "shared_backbones" / "otflow_fullhorizon_seed0")
    args.dataset_root = str(package_root / "paper_datasets")
    args.cryptos_path = str(package_root / "data" / "cryptos_binance_spot_monthly_1s_l10.npz")
    args.lobster_synthetic_profile_path = str(package_root / "data" / "lobster_synthetic" / "lobster_free_sample_profile_10.json")
    args.long_term_st_path = str(package_root / "data" / "long_term_st_100hz_context_only")
    args.molecule_group_root = str(package_root / "data" / "molecule_3d")
    args.molecule_backbone_root = str(package_root / "outputs" / "molecule_3d_backbones")
    return args


def backbone_package_protocol_payload(args: argparse.Namespace) -> Dict[str, Any]:
    raw_root = str(getattr(args, "backbone_package_root", "") or "").strip()
    if not raw_root:
        return {"use_provided_backbones": bool(getattr(args, "use_provided_backbones", False))}
    package_root = resolve_project_path(raw_root)
    package_manifest_path = package_root / PACKAGE_MANIFEST_NAME
    payload = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    return {
        "use_provided_backbones": True,
        "backbone_package_root": str(package_root),
        "backbone_package_schema": str(payload.get("schema_version", "")),
        "backbone_package_family": str(payload.get("family", "")),
        "backbone_package_source_commit": str(payload.get("source_commit", "")),
        "backbone_package_artifact_count": int(payload.get("artifact_count", 0) or 0),
    }


def package_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Package a portable GenODE backbone family with its processed data.")
    parser.add_argument("--family", choices=tuple(FAMILY_SPECS), required=True)
    parser.add_argument("--source_root", default=str(project_root()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage_dir", default="")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no_zip", action="store_true", default=False)
    args = parser.parse_args(argv)
    summary = package_backbone_family(
        family=str(args.family),
        source_root=str(args.source_root),
        output_dir=str(args.output_dir),
        stage_dir=str(args.stage_dir) or None,
        overwrite=bool(args.overwrite),
        make_zip=not bool(args.no_zip),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def validate_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a portable GenODE backbone package.")
    parser.add_argument("package_root")
    parser.add_argument("--expected_family", choices=tuple(FAMILY_SPECS), default="")
    parser.add_argument("--allow_local_paths", action="store_true", default=False)
    args = parser.parse_args(argv)
    summary = validate_backbone_package(
        args.package_root,
        expected_family=str(args.expected_family) or None,
        require_clean_paths=not bool(args.allow_local_paths),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "complete":
        raise SystemExit(1)


__all__ = [
    "FAMILY_SPECS",
    "PACKAGE_MANIFEST_NAME",
    "PACKAGE_SCHEMA_VERSION",
    "PACKAGED_BACKBONE_MANIFEST",
    "apply_backbone_package_to_args",
    "backbone_package_protocol_payload",
    "load_portable_backbone_manifest",
    "package_backbone_family",
    "validate_backbone_package",
    "validate_provided_backbone_manifest",
]
