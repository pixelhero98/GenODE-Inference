from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from genode.data.otflow_paths import project_root
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, TRANSFER_SCHEDULE_KEYS, build_schedule_grid, schedule_display_name
from genode.schedule_transfer.otflow_signal_traces import NATIVE_INFO_GROWTH_TRACE_KEY

PROJECT_ROOT = project_root()
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "native_info_growth_hardness"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures"
DEFAULT_INPUT_JSON = DEFAULT_RESULTS_DIR / "native_info_growth_payload.json"
DEFAULT_PNG = DEFAULT_FIGURE_DIR / "native_info_growth_schedule_trace.png"
DEFAULT_PDF = DEFAULT_FIGURE_DIR / "native_info_growth_schedule_trace.pdf"

DATASET_ORDER = (
    "solar_energy_10m",
    "traffic_hourly",
    "weather_daily",
)
SCHEDULE_ORDER = BASELINE_SCHEDULE_KEYS
NATIVE_HARDNESS_TRACE_KEY = NATIVE_INFO_GROWTH_TRACE_KEY
PAPER_FACING_TRACE_NAME = "native_info_growth"


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def validate_time_grid(grid: Sequence[float], *, name: str = "time_grid") -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional grid with at least two nodes.")
    if abs(float(arr[0])) > 1e-8 or abs(float(arr[-1]) - 1.0) > 1e-8:
        raise ValueError(f"{name} must start at 0.0 and end at 1.0.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def normalize_trace(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Native hardness trace must be a non-empty one-dimensional sequence.")
    fill = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    arr = np.clip(arr, 0.0, None)
    mean = max(float(np.mean(arr)), 1e-12)
    return [float(x) for x in (arr / mean).tolist()]


def schedule_node_summary(scheduler_key: str, macro_steps: int) -> Dict[str, Any]:
    key = str(scheduler_key).strip().lower()
    if key not in SCHEDULE_ORDER:
        raise ValueError(f"Unsupported active schedule {scheduler_key!r}.")
    grid = build_schedule_grid(key, int(macro_steps))
    if grid is None:
        raise ValueError(f"Could not build active schedule {key!r}.")
    arr = validate_time_grid(grid, name=f"{key}_grid")
    widths = np.diff(arr)
    return {
        "scheduler_key": key,
        "scheduler_name": schedule_display_name(key),
        "macro_steps": int(macro_steps),
        "time_grid": [float(x) for x in arr.tolist()],
        "min_step": float(np.min(widths)),
        "max_step": float(np.max(widths)),
        "step_ratio": float(np.max(widths) / max(float(np.min(widths)), 1e-12)),
        "is_transfer_schedule": key in TRANSFER_SCHEDULE_KEYS,
    }


def synthetic_payload(*, macro_steps: int = 10) -> Dict[str, Any]:
    reference_grid = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    mid = 0.5 * (reference_grid[:-1] + reference_grid[1:])
    trace = 0.45 + 0.35 * np.sin(np.pi * mid) ** 2 + 0.20 * mid
    return {
        "artifact": "native_info_growth_hardness_payload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_facing_trace": PAPER_FACING_TRACE_NAME,
        "native_trace_key": NATIVE_HARDNESS_TRACE_KEY,
        "reference_time_grid": [float(x) for x in reference_grid.tolist()],
        "native_info_growth_trace": normalize_trace(trace),
        "schedule_nodes": [schedule_node_summary(key, int(macro_steps)) for key in SCHEDULE_ORDER],
    }


def load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_payload(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_figure(payload: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    ref_grid = validate_time_grid(payload["reference_time_grid"], name="reference_time_grid")
    trace = np.asarray(payload["native_info_growth_trace"], dtype=np.float64)
    if trace.size != ref_grid.size - 1:
        raise ValueError("native_info_growth_trace must have one value per reference interval.")
    mid = 0.5 * (ref_grid[:-1] + ref_grid[1:])
    fig, (ax_trace, ax_nodes) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2]})
    ax_trace.plot(mid, trace, color="#2F6B52", linewidth=2.0)
    ax_trace.set_ylabel("Info-growth")
    ax_trace.grid(True, axis="y", color="#DDDDDD", linewidth=0.8)
    rows = list(payload.get("schedule_nodes", []))
    y_positions = np.arange(len(rows), dtype=np.float64)
    for y, row in zip(y_positions, rows):
        grid = validate_time_grid(row["time_grid"], name=f"{row['scheduler_key']}_grid")
        color = "#1B4E9B" if bool(row.get("is_transfer_schedule")) else "#666666"
        ax_nodes.vlines(grid, y - 0.34, y + 0.34, color=color, linewidth=1.0)
    ax_nodes.set_yticks(y_positions)
    ax_nodes.set_yticklabels([str(row["scheduler_name"]) for row in rows])
    ax_nodes.set_xlabel("Flow time")
    ax_nodes.set_xlim(0.0, 1.0)
    ax_nodes.grid(True, axis="x", color="#E5E5E5", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_payload(payload: Mapping[str, Any], *, png_path: Path = DEFAULT_PNG, pdf_path: Path = DEFAULT_PDF, dpi: int = 300) -> Dict[str, str]:
    fig = build_figure(payload)
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the native info-growth hardness trace figure.")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synthetic", help="Write a lightweight native info-growth payload.")
    synth.add_argument("--out-json", type=Path, default=DEFAULT_INPUT_JSON)
    synth.add_argument("--macro-steps", type=int, default=10)
    plot = sub.add_parser("plot", help="Render a native info-growth payload.")
    plot.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    plot.add_argument("--png", type=Path, default=DEFAULT_PNG)
    plot.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    plot.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    args = build_argparser().parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_payload(macro_steps=int(args.macro_steps))
        write_payload(payload, Path(args.out_json))
        return {"json": str(args.out_json), "payload": payload}
    if args.command == "plot":
        payload = load_payload(Path(args.input_json))
        return plot_payload(payload, png_path=Path(args.png), pdf_path=Path(args.pdf), dpi=int(args.dpi))
    raise ValueError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    main()
