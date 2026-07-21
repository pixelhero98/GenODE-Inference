from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from genode.experiment_layout import (
    REFERENCE_CHECKPOINT_STEPS,
    CONDITIONAL_GENERATION_SCENARIO_KEYS,
    FORECAST_SCENARIO_KEYS,
    MOLECULE_SCENARIO_KEYS,
    SCENARIO_FAMILY_MOLECULE,
)
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY
from genode.data.otflow_paths import display_project_path, project_root, resolve_project_path
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW, BACKBONE_NAME_OTFLOW_MOLECULE
from genode.path_safety import (
    MANIFEST_PARENT_PATH_BASE,
    first_link_or_reparse_component,
    is_link_or_reparse_point,
    portable_relative_path,
    resolve_manifest_path_base,
    resolve_portable_relative_path,
)
from genode.provenance import file_sha256

PACKAGE_SCHEMA_VERSION = "genode_backbone_package"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
PACKAGED_BACKBONE_MANIFEST = "backbone_manifest.json"
MANIFEST_VERSION = "fm_backbone_manifest"
MOLECULE_FAMILY = SCENARIO_FAMILY_MOLECULE
TRAIN_BUDGET_STEPS = REFERENCE_CHECKPOINT_STEPS

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
        data_roots=tuple(f"datasets/monash/{scenario}" for scenario in FORECAST_SCENARIO_KEYS),
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
    return portable_relative_path(path, label="Package-relative path")


def _strip_known_prefix(path_text: str) -> str:
    text = _as_posix(path_text)
    parts = PurePosixPath(text).parts
    if len(parts) >= 2 and parts[0] == "genode" and parts[1] in {"outputs", "data", "datasets"}:
        return PurePosixPath(*parts[1:]).as_posix()
    for marker in ("outputs", "data", "datasets"):
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
            try:
                indirect = first_link_or_reparse_component(candidate, root=source_root)
            except ValueError:
                indirect = None
            if indirect is not None:
                raise ValueError(f"Refusing to package linked or reparse-point source path: {indirect}")
            resolved = candidate.resolve()
            if not resolved.is_relative_to(source_root.resolve()):
                raise ValueError(f"Refusing to package source path outside source_root: {candidate}")
            return resolved
    return candidates[-1].resolve()


def _package_rel_for_path(path_text: str | Path) -> str:
    stripped = _strip_known_prefix(str(path_text))
    _safe_rel(stripped)
    return stripped


def _resolve_manifest_relative_path(manifest_path: Path, value: Any, *, path_base: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    base = resolve_manifest_path_base(manifest_path, path_base)
    return str(resolve_portable_relative_path(base, value, label="Portable backbone manifest path"))


def load_portable_backbone_manifest(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    loaded = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("Portable backbone manifest root must be a JSON object.")
    payload = dict(loaded)
    path_base = str(payload.get("path_base", "") or "").strip()
    resolve_manifest_path_base(resolved, path_base)
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


def _is_json_path_field(key: str) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in {*PATH_FIELDS, *ROOT_FIELDS, "path", "manifest", "backbone_manifest"}
        or normalized.endswith(("_path", "_root", "_dir", "_file"))
    )


def _rewrite_json_paths(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(item_key): _rewrite_json_paths(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_json_paths(item, key=key) for item in value]
    if isinstance(value, str):
        if not _is_json_path_field(key):
            return value
        text = _as_posix(value)
        if any(marker in text for marker in ("outputs/", "data/", "datasets/", "genode/outputs/")):
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
        _copy_json_or_file(src, dst)
        copied.append(_file_record(package_root, dst, role="dataset"))
        return copied
    entries = sorted(src.rglob("*"))
    indirect_entries = [path for path in entries if is_link_or_reparse_point(path)]
    if indirect_entries:
        raise ValueError(f"Refusing to package linked or reparse-point source entry: {indirect_entries[0]}")
    for file_path in (path for path in entries if path.is_file()):
        if not file_path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError(f"Refusing to package source entry outside source_root: {file_path}")
        relative_tail = file_path.relative_to(src)
        dst = package_root / _safe_rel(rel_path) / relative_tail
        _copy_json_or_file(file_path, dst)
        copied.append(_file_record(package_root, dst, role="dataset"))
    return copied


def _file_record(package_root: Path, path: Path, *, role: str) -> Dict[str, Any]:
    rel = path.relative_to(package_root).as_posix()
    return {
        "path": rel,
        "role": role,
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
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
            "path_base": MANIFEST_PARENT_PATH_BASE,
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


def _remove_output_path(path: Path, *, output_dir: Path) -> None:
    absolute = path.absolute()
    if absolute.parent != output_dir.absolute():
        raise ValueError(f"Refusing to remove package output outside output_dir: {absolute}")
    if not path.exists() and not path.is_symlink():
        return
    if is_link_or_reparse_point(path):
        raise ValueError(f"Refusing to remove linked or reparse-point package output: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _commit_package_outputs(
    replacements: Sequence[tuple[Path, Path]],
    *,
    removals: Sequence[Path] = (),
    output_dir: Path,
    token: str,
) -> None:
    backups: List[tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        final_paths = [final_path for _, final_path in replacements]
        for final_path in (*final_paths, *removals):
            if not final_path.exists():
                continue
            if final_path.parent.absolute() != output_dir.absolute():
                raise ValueError(f"Refusing to replace package output outside output_dir: {final_path}")
            if is_link_or_reparse_point(final_path):
                raise ValueError(f"Refusing to replace linked or reparse-point package output: {final_path}")
            backup = output_dir / f".backup-{token}-{len(backups)}"
            os.replace(final_path, backup)
            backups.append((backup, final_path))
        for temporary, final_path in replacements:
            os.replace(temporary, final_path)
            installed.append(final_path)
    except Exception:
        for final_path in reversed(installed):
            _remove_output_path(final_path, output_dir=output_dir)
        for backup, final_path in reversed(backups):
            os.replace(backup, final_path)
        raise
    for backup, _ in backups:
        _remove_output_path(backup, output_dir=output_dir)


def package_backbone_family(
    *,
    family: str,
    source_root: str | Path | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
    make_zip: bool = True,
) -> Dict[str, Any]:
    spec = FAMILY_SPECS[str(family)]
    resolved_source = Path(source_root or project_root()).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    package_root = resolved_output / spec.zip_name.removesuffix(".zip")
    zip_path = resolved_output / spec.zip_name
    zip_manifest_path = zip_path.with_suffix(zip_path.suffix + ".manifest.json")
    existing_outputs = [package_root, zip_path, zip_manifest_path]
    if not overwrite:
        existing = [path for path in existing_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Package output already exists: {existing[0]}")

    token = uuid.uuid4().hex
    build_root = Path(tempfile.mkdtemp(prefix=".pkg-", dir=resolved_output))
    temporary_zip = resolved_output / f".archive-{token}.tmp"
    temporary_zip_manifest = resolved_output / f".archive-manifest-{token}.tmp"
    package_manifest: Dict[str, Any] = {}
    try:
        source_manifest_path = _source_path(resolved_source, PACKAGED_BACKBONE_MANIFEST)
        source_manifest = load_portable_backbone_manifest(source_manifest_path)
        if int(source_manifest.get("ready_count", -1)) != int(source_manifest.get("artifact_count", -2)):
            raise ValueError(
                "Source backbone manifest is not fully ready; expected ready_count == artifact_count before packaging."
            )
        if int(source_manifest.get("artifact_count", -1)) != int(EXPECTED_FULL_BACKBONE_ARTIFACT_COUNT):
            raise ValueError(
                f"Source backbone manifest has artifact_count={source_manifest.get('artifact_count')}; "
                f"expected {EXPECTED_FULL_BACKBONE_ARTIFACT_COUNT} required artifacts before packaging."
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
                package_root=build_root,
            )
            normalized_artifacts.append(normalized)
            records.extend(artifact_records)
        for rel_path in spec.data_roots:
            records.extend(_copy_tree_or_file(resolved_source, build_root, rel_path))

        packaged_manifest = _normalize_manifest_for_package(source_manifest, normalized_artifacts, spec=spec)
        backbone_manifest_file = build_root / PACKAGED_BACKBONE_MANIFEST
        backbone_manifest_file.write_text(json.dumps(packaged_manifest, indent=2, sort_keys=True), encoding="utf-8")
        records.append(_file_record(build_root, backbone_manifest_file, role="backbone_manifest"))
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
        (build_root / PACKAGE_MANIFEST_NAME).write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validation = validate_backbone_package(build_root, expected_family=spec.key)
        if validation["status"] != "complete":
            raise ValueError("Generated backbone package failed self-validation:\n- " + "\n- ".join(validation["errors"]))

        replacements: List[tuple[Path, Path]] = [(build_root, package_root)]
        if make_zip:
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in sorted(file_path for file_path in build_root.rglob("*") if file_path.is_file()):
                    archive.write(path, path.relative_to(build_root).as_posix())
            package_manifest = {
                **package_manifest,
                "zip_name": zip_path.name,
                "zip_size_bytes": int(temporary_zip.stat().st_size),
                "zip_sha256": file_sha256(temporary_zip),
            }
            temporary_zip_manifest.write_text(
                json.dumps(package_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            replacements.extend(((temporary_zip, zip_path), (temporary_zip_manifest, zip_manifest_path)))
        removals = () if make_zip else (zip_path, zip_manifest_path)
        _commit_package_outputs(
            replacements,
            removals=removals,
            output_dir=resolved_output,
            token=token,
        )
    finally:
        for temporary in (build_root, temporary_zip, temporary_zip_manifest):
            if temporary.exists():
                _remove_output_path(temporary, output_dir=resolved_output)
    return {
        "status": "complete",
        "package_root": display_project_path(package_root),
        "zip_path": display_project_path(zip_path) if make_zip else "",
        "zip_manifest_path": display_project_path(zip_manifest_path) if make_zip else "",
        "manifest": package_manifest,
    }


def _contains_local_marker(value: str) -> bool:
    text = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    embedded_absolute = bool(
        re.search(r"(?:^|[\s'\"(=])(?:[A-Za-z]:[/\\]|\\\\[^\s\\]+\\[^\s\\]+|/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)", value)
    )
    return (
        PurePosixPath(text).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in PurePosixPath(text).parts
        or any(marker.lower().replace("\\", "/") in text.lower() for marker in LOCAL_PATH_MARKERS)
        or embedded_absolute
    )


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
        path = resolve_portable_relative_path(
            package_root,
            rel,
            label="Package manifest file path",
            reject_links=True,
        )
    except ValueError as exc:
        return [str(exc)]
    if not path.exists():
        errors.append(f"Missing file listed in package manifest: {rel}")
        return errors
    if not path.is_file():
        errors.append(f"Package manifest entry is not a regular file: {rel}")
        return errors
    try:
        expected_size = int(record.get("size_bytes", -1))
    except (TypeError, ValueError):
        expected_size = -1
    if expected_size < 0:
        errors.append(f"Missing or invalid size_bytes for {rel}")
    elif int(path.stat().st_size) != expected_size:
        errors.append(f"Size mismatch for {rel}")
    expected_hash = str(record.get("sha256", "") or "")
    if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash.lower()):
        errors.append(f"Missing or invalid SHA256 for {rel}")
    elif file_sha256(path) != expected_hash.lower():
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


def validate_backbone_artifact_checkpoint(artifact: Mapping[str, Any], checkpoint_path: Path) -> List[str]:
    """Return integrity errors for a declared backbone checkpoint artifact."""
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
    elif (
        str(artifact.get("backbone_name", "")) == BACKBONE_NAME_OTFLOW_MOLECULE
        and str(artifact.get("benchmark_family", "")) == MOLECULE_FAMILY
    ):
        try:
            from genode.evaluation.molecule_metrics import load_molecule_checkpoint_payload

            load_molecule_checkpoint_payload(
                checkpoint_path,
                expected_identity=f"provided molecule backbone artifact {label}",
                expected_dataset_key=str(artifact.get("dataset_key", "")),
                expected_stratum=str(artifact.get("stratum", "")),
            )
        except Exception as exc:
            errors.append(f"Provided artifact {label} checkpoint is not loadable: {exc}")
    return errors


def validate_backbone_package(
    package_root: str | Path,
    *,
    expected_family: str | None = None,
) -> Dict[str, Any]:
    requested_root = Path(package_root).expanduser().absolute()
    errors: List[str] = []
    if is_link_or_reparse_point(requested_root):
        return {
            "status": "failed",
            "errors": [f"Package root may not be a symlink, junction, or reparse point: {requested_root}"],
        }
    root = requested_root.resolve()
    try:
        entries = sorted(root.rglob("*")) if root.exists() else []
    except OSError as exc:
        entries = []
        errors.append(f"Unable to enumerate package contents safely: {exc}")
    for entry in entries:
        try:
            resolved_entry = entry.resolve(strict=True)
        except OSError as exc:
            errors.append(f"Unable to resolve package entry {entry}: {exc}")
            continue
        if not resolved_entry.is_relative_to(root):
            errors.append(
                "Package entry resolves outside the package root: "
                f"{entry.relative_to(root).as_posix()}"
            )
    indirect_entries = [path for path in entries if is_link_or_reparse_point(path)]
    if indirect_entries:
        errors.append(
            "Package contains a symlink, junction, or reparse point: "
            f"{indirect_entries[0].relative_to(root).as_posix()}"
        )
    package_manifest_path = root / PACKAGE_MANIFEST_NAME
    if not package_manifest_path.exists():
        errors.append(f"Missing {PACKAGE_MANIFEST_NAME}")
        return {"status": "failed", "errors": errors}
    if is_link_or_reparse_point(package_manifest_path):
        errors.append(f"{PACKAGE_MANIFEST_NAME} may not be a symlink, junction, or reparse point.")
        return {"status": "failed", "errors": errors}
    try:
        loaded_package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_package_manifest, Mapping):
            raise ValueError("package manifest root must be a JSON object")
        package_manifest = dict(loaded_package_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {PACKAGE_MANIFEST_NAME}: {exc}")
        package_manifest = {}
    if str(package_manifest.get("schema_version")) != PACKAGE_SCHEMA_VERSION:
        errors.append("Unexpected package schema version.")
    raw_file_records = package_manifest.get("files", [])
    if not isinstance(raw_file_records, list):
        errors.append("Package manifest files must be a list.")
        file_records: List[Mapping[str, Any]] = []
    else:
        file_records = []
        for index, record in enumerate(raw_file_records):
            if not isinstance(record, Mapping):
                errors.append(f"Package manifest files[{index}] must be an object.")
                continue
            file_records.append(record)
    family = str(package_manifest.get("family", ""))
    if expected_family and family != str(expected_family):
        errors.append(f"Package family {family!r} != expected {expected_family!r}.")
    if family not in FAMILY_SPECS:
        errors.append(f"Unknown package family {family!r}.")
        spec = None
    else:
        spec = FAMILY_SPECS[family]
        if list(package_manifest.get("scenarios", [])) != list(spec.scenarios):
            errors.append("Package scenarios do not match the reference family scenarios.")
        if int(package_manifest.get("expected_artifact_count", -1)) != int(spec.expected_artifact_count):
            errors.append("Package expected artifact count does not match the reference family.")
        if list(package_manifest.get("data_roots", [])) != list(spec.data_roots):
            errors.append("Package data roots do not match the reference family data roots.")
        file_paths = [str(record.get("path", "")) for record in file_records]
        if len(set(file_paths)) != len(file_paths):
            errors.append("Package manifest contains duplicate file records.")
        listed_paths = set(file_paths) | {PACKAGE_MANIFEST_NAME}
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in entries
            if path.is_file() and not is_link_or_reparse_point(path)
        }
        unlisted_paths = sorted(actual_paths - listed_paths)
        if unlisted_paths:
            errors.append(f"Package contains files missing from hash manifest: {unlisted_paths[:10]}")
        for data_root in spec.data_roots:
            if not any(path == data_root or path.startswith(f"{data_root}/") for path in file_paths):
                errors.append(f"Package data root has no listed files: {data_root}")
    for record in file_records:
        errors.extend(_validate_file_record(root, record))
    backbone_manifest_path = root / PACKAGED_BACKBONE_MANIFEST
    if not backbone_manifest_path.exists():
        errors.append(f"Missing packaged backbone manifest: {PACKAGED_BACKBONE_MANIFEST}")
    else:
        try:
            loaded_raw_manifest = json.loads(backbone_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_raw_manifest, Mapping):
                raise ValueError("backbone manifest root must be a JSON object")
            raw_backbone_manifest = dict(loaded_raw_manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid packaged backbone manifest JSON: {exc}")
            raw_backbone_manifest = {}
        if str(raw_backbone_manifest.get("path_base", "")) != MANIFEST_PARENT_PATH_BASE:
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
        try:
            backbone_manifest = load_portable_backbone_manifest(backbone_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid packaged backbone manifest paths: {exc}")
            backbone_manifest = {}
        if int(backbone_manifest.get("ready_count", -1)) != int(backbone_manifest.get("artifact_count", -2)):
            errors.append("Packaged backbone manifest is not fully ready.")
        if spec is not None:
            raw_artifacts = backbone_manifest.get("artifacts", [])
            if not isinstance(raw_artifacts, list):
                errors.append("Packaged backbone manifest artifacts must be a list.")
                artifacts: List[Mapping[str, Any]] = []
            else:
                artifacts = []
                for index, artifact in enumerate(raw_artifacts):
                    if not isinstance(artifact, Mapping):
                        errors.append(f"Packaged backbone manifest artifacts[{index}] must be an object.")
                        continue
                    artifacts.append(artifact)
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
                        resolved = Path(path_value).resolve()
                        if not resolved.is_relative_to(root):
                            errors.append(
                                f"Artifact {artifact.get('checkpoint_id', '')} has {field} outside package root: {path_value}"
                            )
                            continue
                        indirect = first_link_or_reparse_component(Path(path_value), root=root)
                        if indirect is not None:
                            errors.append(
                                f"Artifact {artifact.get('checkpoint_id', '')} has {field} through a link or reparse point: {path_value}"
                            )
                            continue
                    else:
                        try:
                            resolved = resolve_portable_relative_path(
                                root,
                                _strip_known_prefix(path_value),
                                label=f"Artifact {field}",
                                reject_links=True,
                            )
                        except ValueError as exc:
                            errors.append(f"Artifact {artifact.get('checkpoint_id', '')} has unsafe {field}: {exc}")
                            continue
                    if not resolved.exists():
                        errors.append(f"Artifact {artifact.get('checkpoint_id', '')} missing {field}: {path_value}")
                    elif field == "checkpoint_path":
                        errors.extend(validate_backbone_artifact_checkpoint(artifact, resolved))
    for json_path in (path for path in entries if path.suffix.lower() == ".json" and path.is_file() and not is_link_or_reparse_point(path)):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid JSON {json_path.relative_to(root).as_posix()}: {exc}")
            continue
        for text in _iter_json_strings(payload):
            if _contains_local_marker(text):
                errors.append(f"Local path marker found in {json_path.relative_to(root).as_posix()}: {text}")
                break
    return {
        "status": "complete" if not errors else "failed",
        "family": family,
        "package_root": display_project_path(root),
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
    try:
        manifest = load_portable_backbone_manifest(resolved_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "errors": [f"Provided backbone manifest is invalid: {exc}"],
            "manifest_path": display_project_path(resolved_manifest),
            "artifact_count": 0,
        }
    candidate_artifacts = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if str(artifact.get("status")) == "ready"
        and (not scenario_key or str(artifact.get("dataset_key")) == str(scenario_key))
        and (not benchmark_family or str(artifact.get("benchmark_family")) == str(benchmark_family))
    ]
    expected_backbone_name = BACKBONE_NAME_OTFLOW_MOLECULE if benchmark_family == MOLECULE_FAMILY else BACKBONE_NAME_OTFLOW
    wrong_backbone_names = sorted(
        {
            str(artifact.get("backbone_name", ""))
            for artifact in candidate_artifacts
            if str(artifact.get("backbone_name", "")) != expected_backbone_name
        }
    )
    if wrong_backbone_names:
        errors.append(
            f"Provided backbone artifacts for scenario={scenario_key!r}, family={benchmark_family!r} "
            f"use backbone names {wrong_backbone_names}; expected {expected_backbone_name!r}."
        )
    matching_artifacts = [
        artifact
        for artifact in candidate_artifacts
        if str(artifact.get("backbone_name", "")) == expected_backbone_name
    ]
    if not matching_artifacts:
        errors.append(
            f"No ready provided backbone artifacts match scenario={scenario_key!r}, "
            f"family={benchmark_family!r}, backbone={expected_backbone_name!r}."
        )
    expected_steps = {int(step) for step in TRAIN_BUDGET_STEPS}
    observed_steps = {int(artifact.get("train_steps", -1)) for artifact in matching_artifacts}
    if matching_artifacts and not expected_steps.issubset(observed_steps):
        errors.append(f"Provided backbone artifacts have train steps {sorted(observed_steps)}; expected at least {sorted(expected_steps)}.")
    lookup_counts: Dict[tuple[Any, ...], int] = {}
    for artifact in matching_artifacts:
        key = (
            str(artifact.get("backbone_name", "")),
            str(artifact.get("benchmark_family", "")),
            str(artifact.get("dataset_key", "")),
            int(artifact.get("train_steps", -1)),
            str(artifact.get("member_key", "")) if benchmark_family == MOLECULE_FAMILY else "",
            str(artifact.get("stratum", "")) if benchmark_family == MOLECULE_FAMILY else "",
        )
        lookup_counts[key] = int(lookup_counts.get(key, 0)) + 1
    duplicate_keys = {key: count for key, count in lookup_counts.items() if count != 1}
    if duplicate_keys:
        first_key, first_count = next(iter(sorted(duplicate_keys.items(), key=lambda item: repr(item[0]))))
        errors.append(f"Provided backbone manifest has duplicate runtime lookup key {first_key} count={first_count}.")
    if matching_artifacts and benchmark_family != MOLECULE_FAMILY:
        for step in sorted(expected_steps):
            count = sum(1 for artifact in matching_artifacts if int(artifact.get("train_steps", -1)) == int(step))
            if count != 1:
                errors.append(f"Provided temporal scenario has {count} ready {expected_backbone_name} artifacts for train_steps={step}; expected 1.")
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
                errors.extend(validate_backbone_artifact_checkpoint(artifact, resolved))
    return {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "manifest_path": display_project_path(resolved_manifest),
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
    scenario = str(getattr(args, "scenario_key", ""))
    if scenario and scenario not in set(spec.scenarios):
        raise ValueError(f"Scenario {scenario!r} is not included in backbone package family {family!r}.")
    args.backbone_manifest = str(package_root / PACKAGED_BACKBONE_MANIFEST)
    args.shared_backbone_root = str(package_root / "outputs" / "shared_backbones" / "otflow_fullhorizon_seed0")
    args.dataset_root = str(package_root / "datasets")
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
        "backbone_package_root": display_project_path(package_root),
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
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args(argv)
    summary = package_backbone_family(
        family=str(args.family),
        source_root=str(args.source_root),
        output_dir=str(args.output_dir),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def validate_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a portable GenODE backbone package.")
    parser.add_argument("package_root")
    parser.add_argument("--expected_family", choices=tuple(FAMILY_SPECS), default="")
    args = parser.parse_args(argv)
    summary = validate_backbone_package(
        args.package_root,
        expected_family=str(args.expected_family) or None,
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
    "validate_backbone_artifact_checkpoint",
    "validate_provided_backbone_manifest",
]
