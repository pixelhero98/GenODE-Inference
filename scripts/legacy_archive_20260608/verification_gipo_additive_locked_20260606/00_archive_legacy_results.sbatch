#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_archive
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_OUTPUTS_ROOT=${GENODE_OUTPUTS_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs}
GENODE_ROOT=${GENODE_ROOT:-${GENODE_OUTPUTS_ROOT}/verification_gipo_additive_locked_20260606}
GENODE_LEGACY_ARCHIVE=${GENODE_LEGACY_ARCHIVE:-${GENODE_OUTPUTS_ROOT}/legacy_archive_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_LEGACY_ARCHIVE}/links"
source "${GENODE_ENV_DIR}/bin/activate"

GENODE_OUTPUTS_ROOT="${GENODE_OUTPUTS_ROOT}" \
GENODE_ROOT="${GENODE_ROOT}" \
GENODE_LEGACY_ARCHIVE="${GENODE_LEGACY_ARCHIVE}" \
python - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

outputs_root = Path(os.environ["GENODE_OUTPUTS_ROOT"])
canonical_root = Path(os.environ["GENODE_ROOT"]).resolve()
archive_root = Path(os.environ["GENODE_LEGACY_ARCHIVE"])
links_root = archive_root / "links"
archive_root.mkdir(parents=True, exist_ok=True)
links_root.mkdir(parents=True, exist_ok=True)

entries = []
for path in sorted(outputs_root.glob("verification_gipo*")):
    resolved = path.resolve()
    if resolved == canonical_root:
        continue
    if archive_root in resolved.parents:
        continue
    link = links_root / path.name
    if not link.exists():
        link.symlink_to(path, target_is_directory=True)
    entries.append(
        {
            "name": path.name,
            "path": str(path),
            "archive_link": str(link),
            "is_dir": path.is_dir(),
            "non_destructive": True,
        }
    )

manifest = {
    "artifact": "gipo_legacy_archive_manifest",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "archive_root": str(archive_root),
    "canonical_root": str(canonical_root),
    "method": "non_destructive_symlink_manifest",
    "entry_count": len(entries),
    "entries": entries,
}
(archive_root / "legacy_archive_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
