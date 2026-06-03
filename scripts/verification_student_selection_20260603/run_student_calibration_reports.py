from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from genode.gipo.report_locked_test import report_gipo_locked_test


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_reports(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    manifest = _read_json(root / "selection_inputs" / "student_selection_inputs_manifest.json")
    completed: list[dict] = []
    for run in list(manifest["runs"]):
        run_id = str(run["run_id"])
        for split_name, rows_key in (("context_disjoint", "context_rows"), ("series_disjoint", "series_rows")):
            out_dir = root / "calibration_student_reports" / run_id / split_name
            summary_path = out_dir / f"{split_name}_gipo_policy_summary.json"
            if summary_path.exists():
                completed.append({"run_id": run_id, "split_name": split_name, "status": "skipped_existing"})
                continue
            report_args = Namespace(
                gipo_student_checkpoint=str(run["student_checkpoint"]),
                training_summary=str(run["training_summary"]),
                context_rows=str(run[rows_key]),
                context_embeddings_npz=str(run["calibration_embeddings"]),
                split_phase=split_name,
                selection_mode="calibration",
                report_label=split_name,
                out_dir=str(out_dir),
                baseline_rows=str(run["fixed_reference_rows"]),
                comparator_rows=str(run["ser_reference_rows"]),
                dataset=str(args.dataset),
                dataset_root=str(args.dataset_root),
                shared_backbone_root=str(args.shared_backbone_root),
                backbone_manifest=str(args.backbone_manifest),
                output_root=str(root),
                otflow_train_steps=int(args.otflow_train_steps),
                device=str(args.device),
                seeds=str(args.seeds),
                solver_names=str(args.solver_names),
                target_nfe_values=str(args.target_nfe_values),
                num_eval_samples=int(args.num_eval_samples),
                forecast_eval_batch_size=int(args.forecast_eval_batch_size),
            )
            report_gipo_locked_test(report_args)
            completed.append({"run_id": run_id, "split_name": split_name, "status": "reported"})
    payload = {"artifact": "student_calibration_report_status", "locked_test_used_for_selection": False, "completed": completed}
    status_path = root / "summary" / "student_calibration_report_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibration-only frozen-student context reports.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="solar_energy_10m")
    parser.add_argument("--dataset_root", default="/projects/b35z/genode/paper_datasets")
    parser.add_argument("--shared_backbone_root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix")
    parser.add_argument("--backbone_manifest", default="/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json")
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=64)
    return parser


def main() -> None:
    payload = run_reports(build_argparser().parse_args())
    print(json.dumps({"reported": len(payload["completed"]), "locked_test_used_for_selection": False}, sort_keys=True))


if __name__ == "__main__":
    main()
