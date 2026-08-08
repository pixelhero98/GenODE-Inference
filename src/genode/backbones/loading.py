from __future__ import annotations

import math
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from torch import nn

from genode.path_safety import is_link_or_reparse_point

from .adapter import CanonicalNoiseToDataAdapter
from .checkpoint import verify_checkpoint_binding
from .protocol import ImageBackboneManifest
from .registry import ImageBackboneSpec, get_image_backbone_spec
from .rfpp_factory import build_rfpp_native_model


class UserSuppliedRFPPFactory(Protocol):
    def __call__(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        spec: ImageBackboneSpec,
    ) -> nn.Module: ...


GitRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run_git(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def verify_user_supplied_rfpp_source_root(
    source_root: str | Path,
    spec: ImageBackboneSpec,
    *,
    timeout: float = 10.0,
    _git_runner: GitRunner = _run_git,
) -> Path:
    """Validate a clean user-supplied RF++ checkout without importing from it."""

    raw_root = Path(source_root).expanduser()
    if is_link_or_reparse_point(raw_root):
        raise ValueError("RF++ source root must not be a symlink, junction, or reparse point.")
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("RF++ source root must be a directory.")
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("Git verification timeout must be positive.")

    config_path = root.joinpath(*spec.source_config_path.split("/"))
    if is_link_or_reparse_point(config_path) or not config_path.is_file():
        raise ValueError(f"RF++ source root is missing regular config {spec.source_config_path!r}.")

    try:
        revision_result = _git_runner(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            timeout=float(timeout),
        )
        status_result = _git_runner(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            timeout=float(timeout),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Could not verify the user-supplied RF++ Git checkout.") from exc
    if revision_result.returncode != 0:
        raise ValueError("RF++ source root is not a readable Git checkout.")
    revision = revision_result.stdout.strip()
    if revision != spec.source_revision:
        raise ValueError(
            f"RF++ source revision mismatch: expected {spec.source_revision}, got {revision or '<empty>'}."
        )
    if status_result.returncode != 0:
        raise ValueError("Could not inspect the RF++ source worktree.")
    if status_result.stdout.strip():
        raise ValueError("RF++ source worktree has tracked modifications.")
    return root


def load_verified_image_backbone(
    manifest: ImageBackboneManifest,
    *,
    checkpoint_path: str | Path,
    source_root: str | Path,
    factory: UserSuppliedRFPPFactory | None = None,
    source_verification_timeout: float = 10.0,
) -> CanonicalNoiseToDataAdapter:
    """Verify external inputs, then use the built-in or caller-provided RF++ factory."""

    selected_factory = build_rfpp_native_model if factory is None else factory
    if not callable(selected_factory):
        raise TypeError("factory must be callable.")
    spec = get_image_backbone_spec(manifest.model_key)
    verified_checkpoint = verify_checkpoint_binding(
        manifest.model_key,
        checkpoint_path,
        manifest.checkpoint,
    )
    verified_source = verify_user_supplied_rfpp_source_root(
        source_root,
        spec,
        timeout=source_verification_timeout,
    )
    native_model = selected_factory(
        source_root=verified_source,
        checkpoint_path=verified_checkpoint,
        spec=spec,
    )
    if not isinstance(native_model, nn.Module):
        raise TypeError("User-supplied RF++ factory must return a torch.nn.Module.")
    native_model.eval()
    native_model.requires_grad_(False)
    adapter = CanonicalNoiseToDataAdapter(native_model, manifest)
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


__all__ = [
    "UserSuppliedRFPPFactory",
    "load_verified_image_backbone",
    "verify_user_supplied_rfpp_source_root",
]
