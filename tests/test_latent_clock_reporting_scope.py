import json

from genode.latent_clock.artifacts import write_new_jsonl
from genode.latent_clock.protocol import NOISE_SEEDS


def test_report_completes_without_external_audit_files(tmp_path, monkeypatch):
    from genode.latent_clock import accounting, report

    def forbidden(*args):
        raise AssertionError("Cancelled external audits must not be read")

    monkeypatch.setattr(report, "_audit_summary", forbidden)
    monkeypatch.setattr(report, "_density_plots", lambda *args: None)
    monkeypatch.setattr(accounting, "write_cost_ledger", lambda *args: None)
    for name, value in (("candidate", 1.0), ("anchor", 0.0)):
        write_new_jsonl(
            tmp_path / f"{name}.jsonl",
            [
                {
                    "prompt_id": str(i),
                    "noise_seed": seed,
                    "split": "locked_test",
                    "preference": value,
                    "alignment": value,
                }
                for i in range(512)
                for seed in NOISE_SEEDS
            ],
        )
    spec = {
        "results_root": str(tmp_path),
        "methods": [],
        "include_external_audits": False,
        "evaluation_scope": {"split": "locked_test", "excluded_audits": ["GenEval", "VisionReward"]},
        "comparisons": [
            {
                "name": "candidate vs anchor",
                "nfe": 4,
                "budget": "100",
                "candidate": [{"path": str(tmp_path / "candidate.jsonl")}],
                "anchor": [{"path": str(tmp_path / "anchor.jsonl")}],
            }
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    report.report_comparisons(str(path), str(tmp_path / "report"))
    complete = json.loads((tmp_path / "report/complete.json").read_text())
    assert complete["external_audits_included"] is False
    summary = json.loads((tmp_path / "report/summary.json").read_text())
    assert {row["metric"] for row in summary["comparisons"]} == {"preference", "alignment"}
    assert all(row["prompt_count"] == 512 for row in summary["comparisons"])
