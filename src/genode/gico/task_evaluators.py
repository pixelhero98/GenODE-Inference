"""Frozen forecasting, molecular and text-to-image execution for selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from genode.gico.rewards import MOLECULE_METRICS, TASK_METRICS
from genode.latent_clock.artifacts import sha256_file
from genode.latent_clock.rewards import _module_fingerprint


def tensor_digest(value):
    array = (
        value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.ascontiguousarray(value)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def verify_file(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"Selection asset checksum mismatch: {path}")


class FrozenModules:
    def bind_modules(self, modules):
        self.modules = modules
        for module in modules:
            module.eval().requires_grad_(False)
        self.fingerprints = [_module_fingerprint(m) for m in modules]

    def verify_frozen(self):
        for module, digest in zip(self.modules, self.fingerprints, strict=True):
            if (
                module.training
                or any(p.requires_grad or p.grad is not None for p in module.parameters())
                or _module_fingerprint(module) != digest
            ):
                raise ValueError("Selection changed frozen generator or scorer weights/state.")


class ReplayPolicy:
    """Feed the already sampled member clocks through the native sequence runtime."""

    def __init__(self, row, clocks, context):
        self.row, self.clocks, self.context, self.cursor = row, clocks, context, 0
        self.artifact_sha256 = row.get("selection_checkpoint_id", "uniform")
        self.policy_kind = row.get("policy_kind", "deterministic")
        self.metadata = {
            "task": row["task"],
            "backbone": row["backbone"],
            "molecular_feature_maps": {"reference": row.get("molecule_feature_map")},
        }

    def density(self, context, solver, nfe, **_):
        if (
            solver != self.row["solver"]
            or nfe != self.row["nfe"]
            or not np.array_equal(np.asarray(context, dtype=np.float32), np.asarray(self.context, dtype=np.float32))
        ):
            raise ValueError("Executed native context or solver/NFE differs from selection evidence.")
        if self.cursor >= len(self.clocks):
            raise ValueError("Runtime requested more clocks than the declared ensemble.")
        clock = self.clocks[self.cursor]
        self.cursor += 1
        return np.asarray(clock["density_mass"], dtype=np.float64)


class SequenceEvaluator(FrozenModules):
    def __init__(self, config):
        from genode.data.otflow_forecast_data import build_monash_forecast_splits
        from genode.evaluation.molecule_metrics import load_molecule_checkpoint_splits
        from genode.evaluation.otflow_evaluation_support import load_checkpoint_model

        self.config, self.device = config, torch.device(config["device"])
        self.molecule = config["task"].startswith("molecule_")
        self.loaded = {}
        for name, source in config["backbones"].items():
            verify_file(source["checkpoint"], source["checkpoint_sha256"])
            if self.molecule:
                loaded = load_molecule_checkpoint_splits(
                    checkpoint_path=source["checkpoint"],
                    dataset_key=config["task"],
                    stratum=source["stratum"],
                    processed_dir=source["processed_dir"],
                    rollout_steps=source["rollout_steps"],
                    stride_eval=source["stride_eval"],
                    device=self.device,
                )
            else:
                model, cfg = load_checkpoint_model(Path(source["checkpoint"]), self.device)
                splits = build_monash_forecast_splits(
                    dataset_root=Path(source["dataset_root"]),
                    dataset_key=config["task"],
                    cfg=cfg,
                    history_len=cfg.history_len,
                    horizon=cfg.prediction_horizon,
                    stride_train=1,
                    time_feature_mode=source["time_feature_mode"],
                )
                loaded = {"model": model, "cfg": cfg, "splits": splits}
            self.loaded[name] = loaded
        self.bind_modules([v["model"] for v in self.loaded.values()])

    def measure(self, row, clocks, context, case, output):
        from genode.solver_protocol import solver_macro_steps

        loaded = self.loaded[case["backbone"]]
        source = self.config["backbones"][case["backbone"]]
        if row["backbone"] != source["backbone_id"] or case["reference_id"] != row["reference_id"]:
            raise ValueError("Sequence checkpoint/reference identity differs from the panel.")
        ds = loaded["splits"]["val"]
        index = case["example_idx"]
        policy = ReplayPolicy(row, clocks, context)
        steps = solver_macro_steps(row["solver"], row["nfe"])
        if self.molecule:
            from genode.evaluation.molecule_metrics import evaluate_molecule_rollout_schedule

            target = ds.eval_item(index)["future_coords"]
            if tensor_digest(target) != case["target_sha256"]:
                raise ValueError("Molecular held-out target changed.")
            seed = row["seed"]
            if "collection_seed" in row:
                seeds = row["sample_seeds"]
                if seeds != list(range(seed, seed + row["ensemble_size"])):
                    raise ValueError("Molecular collection member seeds differ from the planned complete ensemble.")
                # The native evaluator adds this example offset internally.
                seed -= 10000 * index
            result = evaluate_molecule_rollout_schedule(
                model=loaded["model"],
                ds=ds,
                cfg=loaded["cfg"],
                solver_key=row["solver"],
                runtime_nfe=steps,
                target_nfe=row["nfe"],
                time_grid=clocks[0]["time_grid"],
                rollout_steps=source["rollout_steps"],
                sample_count=row["ensemble_size"],
                example_indices=[index],
                seed=seed,
                scheduler_key=row["schedule_key"],
                split_phase="validation_tuning",
                checkpoint_id=row["backbone"],
                dataset_key=row["task"],
                member_key=case["backbone"],
                stratum=source["stratum"],
                device=self.device,
                policy=policy,
            )
            metrics = {k: result["molecule_" + k] for k in MOLECULE_METRICS}
        else:
            from genode.evaluation.otflow_evaluation_support import _parse_forecast_batch, evaluate_forecast_schedule

            _, target, future, _ = _parse_forecast_batch(ds[index])
            complete = target[None] if future is None else torch.cat((target[None], future))
            if tensor_digest(complete) != case["target_sha256"]:
                raise ValueError("Forecast held-out target changed.")
            seeds = case["sample_seed_values"]
            if "collection_seed" in row and seeds != row["sample_seeds"]:
                raise ValueError("Forecast case seeds differ from the planned complete ensemble.")
            if seeds != list(range(seeds[0], seeds[0] + row["ensemble_size"])):
                raise ValueError("Forecast physical member seeds must be complete and consecutive.")
            if case["collection_batch_size"] != 1:
                raise ValueError("Forecast selection requires evidence collected one context per batch.")
            result = evaluate_forecast_schedule(
                loaded["model"],
                ds,
                loaded["cfg"],
                solver_name=row["solver"],
                runtime_nfe=steps,
                target_nfe=row["nfe"],
                time_grid=clocks[0]["time_grid"],
                num_eval_samples=row["ensemble_size"],
                seed=seeds[0],
                dataset_key=row["task"],
                split_phase="validation_tuning",
                checkpoint_id=row["backbone"],
                example_indices=[index],
                batch_size=1,
                return_per_example_rows=True,
                policy=policy,
            )
            metrics = {k: result["forecast_" + k] for k in ("crps", "mase")}
        if policy.cursor != len(clocks):
            raise ValueError("Runtime did not execute every sampled ensemble clock.")
        output.mkdir()
        (output / "metrics.json").write_text(json.dumps(metrics, allow_nan=False), encoding="utf-8")
        return metrics


class TextEvaluator:
    def __init__(self, config):
        from genode.latent_clock.runtime import load_runtime

        self.config = config
        self.runtime = load_runtime(config["generator_config"])

    def verify_frozen(self):
        self.runtime.verify_frozen()

    def measure(self, row, clocks, context, case, output):
        from genode.latent_clock.clocks import Clock
        from genode.latent_clock.collection import image_tensor_to_pil
        from genode.latent_clock.gico import runtime_binding

        runtime = self.runtime
        if (
            row["ensemble_size"] != 1
            or row["backbone_binding"] != runtime_binding(runtime)
            or row["solver"] != runtime.adapter.solver_key
        ):
            raise ValueError("Text selection requires the exact single-image frozen runtime binding.")
        from genode.latent_clock.artifacts import canonical_sha256

        scorer = self.config["scorer"]
        base_protocol = canonical_sha256(
            {
                "clock_protocol": "shared_density64_v1",
                "runtime": runtime_binding(runtime),
                "postprocess": "decoded_clamp_round_uint8_v1",
            }
        )
        scorer_protocol = canonical_sha256(
            {
                "scorer_versions": scorer["versions"],
                "asset_manifest_sha256": scorer["asset_manifest_sha256"],
                "weight_fingerprints": scorer["weight_fingerprints"],
            }
        )
        if row["measurement_protocol"] != canonical_sha256([base_protocol, scorer_protocol]):
            raise ValueError("Text scoring protocol differs from the frozen selection evidence.")
        reference = canonical_sha256(
            {"prompt_id": row["context_id"], **{key: case[key] for key in ("image_id", "caption_id", "prompt")}}
        )
        if reference != row["reference_id"]:
            raise ValueError("Text prompt/reference identity changed.")
        native = runtime.adapter.encode_context(row["context_id"], case["prompt"])
        if not np.array_equal(native.embedding, np.asarray(context, dtype=np.float32)):
            raise ValueError("Native prompt embedding differs from the selection panel.")
        if case["reference_id"] != row["reference_id"]:
            raise ValueError("Prompt reference identity differs from its panel.")
        clock = Clock("policy", row["nfe"], tuple(clocks[0]["time_grid"]), "gico", tuple(clocks[0]["density_mass"]))
        image, trace = runtime.adapter.sample(noise_seed=row["seed"], context=native, clock=clock)
        if trace.realized_nfe != row["nfe"]:
            raise ValueError("Text runtime exceeded the declared NFE.")
        output.mkdir()
        image_path = output / "image.png"
        image_tensor_to_pil(image).save(image_path)
        request = {"prompt": case["prompt"], "image": str(image_path.resolve()), **self.config["scorer"]}
        request_path = output / "score-request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        subprocess.run(
            [request["python"], "-m", "genode.gico.task_evaluators", str(request_path.resolve())], check=True
        )
        return json.loads(request_path.with_suffix(".result.json").read_text(encoding="utf-8"))


def load_task_evaluator(config):
    task = config["task"]
    if task not in TASK_METRICS:
        raise ValueError("Unsupported selection task.")
    if task in ("sana", "sd15"):
        return TextEvaluator(config)
    if task in ("cifar10", "imagenet64"):
        from genode.gico.image_evaluator import ImageEvaluator

        return ImageEvaluator(config)
    return SequenceEvaluator(config)


def score_request(path):
    from genode.latent_clock.rewards import FrozenDualScorer

    request = json.loads(Path(path).read_text(encoding="utf-8"))
    scorer = FrozenDualScorer(device=request["device"], asset_manifest=request["asset_manifest"])
    if (
        list(scorer._fingerprints) != request["weight_fingerprints"]
        or scorer.versions != request["versions"]
        or scorer.asset_manifest_sha256 != request["asset_manifest_sha256"]
    ):
        raise ValueError("Text scorer differs from the pinned selection protocol.")
    preference, alignment = scorer.score(request["prompt"], request["image"])
    scorer.verify_frozen()
    Path(path).with_suffix(".result.json").write_text(
        json.dumps(
            {
                "preference": preference,
                "alignment": alignment,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import sys

    score_request(sys.argv[1])
