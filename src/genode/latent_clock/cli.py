from __future__ import annotations

import argparse
import json

from genode.latent_clock.artifacts import read_jsonl, sha256_file, write_new_json
from genode.latent_clock.clocks import REFERENCE_CLOCK_KEYS, reference_clocks
from genode.latent_clock.protocol import build_prompt_splits, estimate_reward_scales, validate_budget
from genode.latent_clock.rewards import score_image_manifest


def _prepare_splits(args: argparse.Namespace) -> None:
    write_new_json(args.output, build_prompt_splits(args.coco_annotations, seed=args.seed))


def _reference_clocks(args: argparse.Namespace) -> None:
    clocks = [
        {
            "clock_key": clock.key,
            "target_nfe": clock.target_nfe,
            "nodes": list(clock.nodes),
            "density_mass": list(clock.density_mass),
            "transferred": clock.key.startswith(("ays_", "gits_", "ots_")),
        }
        for nfe in args.nfes
        for clock in reference_clocks(nfe)
    ]
    write_new_json(
        args.output,
        {
            "protocol": "genode_latent_t2i_reference_clocks_density64_v2",
            "support_keys": list(REFERENCE_CLOCK_KEYS),
            "target_backbone": args.target_backbone,
            "clocks": clocks,
        },
    )


def _freeze_scales(args: argparse.Namespace) -> None:
    scales = estimate_reward_scales(read_jsonl(args.rows))
    write_new_json(
        args.output,
        {
            "protocol": "genode_latent_t2i_reward_scales_v1",
            "preference": scales.preference,
            "alignment": scales.alignment,
            "source_sha256": sha256_file(args.rows),
        },
    )


def _validate_budget(args: argparse.Namespace) -> None:
    validate_budget(args.budget, read_jsonl(args.ledger))
    print(json.dumps({"status": "valid", "budget": str(args.budget)}))


def _score(args: argparse.Namespace) -> None:
    score_image_manifest(args.images, args.output, device=args.device)


def _collect(args: argparse.Namespace) -> None:
    from genode.latent_clock.collection import collect

    collect(runtime_config=args.runtime_config, plan_path=args.plan, output=args.output)


def _prepare_collection(args: argparse.Namespace) -> None:
    from genode.latent_clock.collection import prepare_collection

    prepare_collection(
        manifest_path=args.manifest,
        output=args.output,
        phase=args.phase,
        nfes=args.nfes,
        budget=args.budget,
        method=args.method,
        checkpoint=args.checkpoint,
        freeze_path=args.freeze,
        policy_kind=args.policy_kind,
        clock_seed=args.clock_seed,
    )


def _parity(args: argparse.Namespace) -> None:
    from genode.latent_clock.collection import sampler_parity

    sampler_parity(runtime_config=args.runtime_config, output=args.output)


def _prepare_gico(args: argparse.Namespace) -> None:
    from genode.latent_clock.gico import prepare_gico

    print(
        json.dumps(
            prepare_gico(
                rows_paths=args.rows,
                embeddings_paths=args.embeddings,
                manifest_path=args.manifest,
                task=args.task,
                nfes=tuple(args.nfes),
                output=args.output,
            )
        )
    )


def _fit_gico(args: argparse.Namespace) -> None:
    from genode.latent_clock.gico import fit_gico

    print(
        json.dumps(
            fit_gico(
                config_path=args.config,
                policy_kind=args.policy_kind,
                dry_run=args.dry_run,
            )
        )
    )


def _fit_search(args: argparse.Namespace) -> None:
    from genode.latent_clock.search import fit_search

    fit_search(
        method=args.method,
        runtime_config=args.runtime_config,
        manifest_path=args.manifest,
        scales_path=args.scales,
        anchors_path=args.anchors,
        scorer_python=args.scorer_python,
        nfe=args.nfe,
        budget=args.budget,
        seed=args.seed,
        output=args.output,
    )


def _serve_scores(args: argparse.Namespace) -> None:
    from genode.latent_clock.search import serve_scores

    serve_scores()


def _fit_ld3(args: argparse.Namespace) -> None:
    from genode.latent_clock.ld3 import fit_ld3

    fit_ld3(
        runtime_config=args.runtime_config,
        manifest_path=args.manifest,
        nfe=args.nfe,
        seed=args.seed,
        output=args.output,
    )


def _freeze_methods(args: argparse.Namespace) -> None:
    from genode.latent_clock.report import freeze_methods

    freeze_methods(registry_path=args.registry, manifest_path=args.manifest, output=args.output)


def _report(args: argparse.Namespace) -> None:
    from genode.latent_clock.report import report_comparisons

    report_comparisons(args.specification, args.output)


def _prepare_geneval(args: argparse.Namespace) -> None:
    from genode.latent_clock.collection import prepare_geneval

    prepare_geneval(
        source=args.source,
        output=args.output,
        method=args.method,
        nfe=args.nfe,
        checkpoint=args.checkpoint,
        freeze_path=args.freeze,
        policy_kind=args.policy_kind,
        clock_seed=args.clock_seed,
    )


def _export_geneval(args: argparse.Namespace) -> None:
    from genode.latent_clock.collection import export_geneval

    export_geneval(images_path=args.images, output=args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genode-latent-clock")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare-splits")
    split.add_argument("--coco-annotations", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=48271)
    split.set_defaults(run=_prepare_splits)
    clocks = commands.add_parser("reference-clocks")
    clocks.add_argument("--nfes", type=int, nargs="+", default=[4, 6, 8])
    clocks.add_argument("--target-backbone", required=True)
    clocks.add_argument("--output", required=True)
    clocks.set_defaults(run=_reference_clocks)
    scale = commands.add_parser("freeze-reward-scales")
    scale.add_argument("--rows", required=True)
    scale.add_argument("--output", required=True)
    scale.set_defaults(run=_freeze_scales)
    budget = commands.add_parser("validate-budget")
    budget.add_argument("--budget", choices=("25", "50", "100"), required=True)
    budget.add_argument("--ledger", required=True)
    budget.set_defaults(run=_validate_budget)
    score = commands.add_parser("score-images")
    score.add_argument("--images", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.set_defaults(run=_score)
    prepare = commands.add_parser("prepare-collection")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--phase", choices=("pilot", "calibration", "validation", "locked_test"), required=True)
    prepare.add_argument("--nfes", nargs="+", type=int, required=True)
    prepare.add_argument("--budget", choices=("25", "50", "100"), default="100")
    prepare.add_argument("--method", default="support")
    prepare.add_argument("--checkpoint")
    prepare.add_argument("--freeze")
    prepare.add_argument("--policy-kind", choices=("deterministic", "stochastic"), default="deterministic")
    prepare.add_argument("--clock-seed", type=int, default=0)
    prepare.set_defaults(run=_prepare_collection)
    collect = commands.add_parser("collect")
    collect.add_argument("--runtime-config", required=True)
    collect.add_argument("--plan", required=True)
    collect.add_argument("--output", required=True)
    collect.set_defaults(run=_collect)
    parity = commands.add_parser("sampler-parity")
    parity.add_argument("--runtime-config", required=True)
    parity.add_argument("--output", required=True)
    parity.set_defaults(run=_parity)
    evidence = commands.add_parser("prepare-gico")
    evidence.add_argument("--rows", nargs="+", required=True)
    evidence.add_argument("--embeddings", nargs="+", required=True)
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument("--task", choices=("sana", "sd15"), required=True)
    evidence.add_argument("--nfes", nargs="+", type=int, required=True)
    evidence.add_argument("--output", required=True)
    evidence.set_defaults(run=_prepare_gico)
    fit = commands.add_parser("fit-gico")
    fit.add_argument("--config", required=True)
    fit.add_argument("--policy-kind", choices=("deterministic", "stochastic", "both"))
    fit.add_argument("--dry-run", action="store_true")
    fit.set_defaults(run=_fit_gico)
    search = commands.add_parser("fit-search")
    for name in ("runtime-config", "manifest", "scales", "anchors", "scorer-python", "output"):
        search.add_argument("--" + name, required=True)
    search.add_argument("--method", choices=("bo", "pg"), required=True)
    search.add_argument("--nfe", type=int, choices=(4, 6, 8), required=True)
    search.add_argument("--budget", choices=("25", "50", "100"), required=True)
    search.add_argument("--seed", type=int, required=True)
    search.set_defaults(run=_fit_search)
    commands.add_parser("serve-scores").set_defaults(run=_serve_scores)
    ld3 = commands.add_parser("fit-ld3")
    for name in ("runtime-config", "manifest", "output"):
        ld3.add_argument("--" + name, required=True)
    ld3.add_argument("--nfe", type=int, choices=(4, 6, 8), required=True)
    ld3.add_argument("--seed", type=int, required=True)
    ld3.set_defaults(run=_fit_ld3)
    freeze = commands.add_parser("freeze-methods")
    for name in ("registry", "manifest", "output"):
        freeze.add_argument("--" + name, required=True)
    freeze.set_defaults(run=_freeze_methods)
    report = commands.add_parser("report")
    report.add_argument("--specification", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(run=_report)
    geneval = commands.add_parser("prepare-geneval")
    for name in ("source", "output", "method", "freeze"):
        geneval.add_argument("--" + name, required=True)
    geneval.add_argument("--checkpoint")
    geneval.add_argument("--policy-kind", choices=("deterministic", "stochastic"), default="deterministic")
    geneval.add_argument("--clock-seed", type=int, default=0)
    geneval.add_argument("--nfe", type=int, required=True)
    geneval.set_defaults(run=_prepare_geneval)
    export = commands.add_parser("export-geneval")
    export.add_argument("--images", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(run=_export_geneval)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
