from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genode.latent_clock.artifacts import sha256_file, write_new_json
from genode.latent_clock.protocol import NOISE_SEEDS
from genode.latent_clock.runtime import load_runtime


@dataclass(frozen=True)
class LD3Schedule:
    integration_times: tuple[float, ...]
    evaluation_times: tuple[float, ...]
    provenance: dict
    order: int = 2

    def __post_init__(self) -> None:
        first, second = np.asarray(self.integration_times), np.asarray(self.evaluation_times)
        if first.ndim != 1 or first.shape != second.shape or len(first) < 3 or self.order != 2:
            raise ValueError("LD3 requires two complete same-sized grids and configured iPNDM order 2.")
        if not np.isfinite(first).all() or not np.isfinite(second).all() or not np.all(np.diff(first) < 0):
            raise ValueError("LD3 integration grid must descend and both time arrays must be finite.")
        if np.any(first <= 0) or np.any(first > 1) or np.any(second <= 0) or np.any(second > 1):
            raise ValueError("LD3 times must be in (0,1].")
        if not self.provenance:
            raise ValueError("LD3 schedules require source and checkpoint provenance.")

    @property
    def nfe(self) -> int:
        return len(self.integration_times) - 1


def checkpointed_ld3_model(runtime):
    """Use the pinned upstream --low_gpu path without changing inference sampling."""
    from models.latent_diff import model_wrapper

    model = runtime.modules["sd15_generator_text_vae"]
    return model_wrapper(
        lambda x, t, c: model.apply_model(x, t, c),
        runtime.adapter.noise_schedule,
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=runtime.metadata["cfg"],
        use_checkpoint=True,
    )


def fit_ld3(*, runtime_config: str, manifest_path: str, nfe: int, seed: int, output: str) -> None:
    import torch

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime(runtime_config)
    if runtime.metadata["backbone"] != "sd15":
        raise ValueError("This LD3 protocol requires the frozen SD1.5 EMA backbone.")
    if runtime.adapter.solver_key != "ipndm":
        raise ValueError("The official LD3 baseline requires explicit runtime solver='ipndm', not 'ipndm_v'.")
    from dataset import LD3Dataset
    from trainer import LD3Trainer, ModelConfig, TrainingConfig
    from utils import PRIOR_TIMESTEPS, set_seed_everything

    set_seed_everything(seed)
    adapter = runtime.adapter
    manifest = json.loads(Path(manifest_path).read_text())
    # The original 25 training / 25 holdout prompt recipe, each evaluated with
    # both matched seeds. These teacher trajectories have a separate cost.
    prompts = [r for r in manifest["records"] if r["split"] == "calibration"][:50]
    if len(prompts) != 50:
        raise ValueError("LD3 needs 50 distinct calibration prompts for disjoint training and holdout panels.")
    teacher_nfe = nfe + 1
    sigma = torch.tensor(PRIOR_TIMESTEPS["sd"][teacher_nfe], device="cuda")
    teacher_times = adapter.noise_schedule.inverse_lambda(-sigma.log()).float()
    counters = {
        "teacher": {"module_forwards": 0, "cfg_sample_equivalents": 0},
        "optimization": {"module_forwards": 0, "cfg_sample_equivalents": 0},
    }
    phase = "teacher"

    def count_forward(module, args):
        counters[phase]["module_forwards"] += 1
        counters[phase]["cfg_sample_equivalents"] += int(args[0].shape[0])

    model = runtime.modules["sd15_generator_text_vae"]
    hook = model.model.register_forward_pre_hook(count_forward)
    data = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for prompt in prompts:
        context = adapter.encode_context(prompt["prompt_id"], prompt["prompt"])
        condition, uncondition = adapter._opaque_contexts[context.context_id]
        for noise_seed in NOISE_SEEDS:
            latent = adapter.latent_factory(noise_seed, context)
            with torch.no_grad():
                target = adapter.solver.sample_simple(
                    adapter.model_fn,
                    adapter.noise_schedule.prior_transformation(latent),
                    teacher_times,
                    teacher_times,
                    order=2,
                    condition=condition,
                    unconditional_condition=uncondition,
                )
                image = adapter.decoder(target)
            data.append(
                (
                    latent[0].detach().cpu().clone(),
                    image[0].detach().cpu().clone(),
                    condition[0].detach().cpu().clone(),
                    uncondition[0].detach().cpu().clone(),
                )
            )
    torch.cuda.synchronize()
    teacher_seconds = time.perf_counter() - start

    def dataset(rows):
        latents, images, conditions, unconditions = zip(*rows, strict=True)
        return LD3Dataset(
            [x.clone() for x in latents],
            [x.clone() for x in latents],
            list(images),
            list(conditions),
            list(unconditions),
        )

    training = TrainingConfig(
        train_data=dataset(data[:50]),
        valid_data=dataset(data[50:]),
        train_batch_size=1,
        valid_batch_size=1,
        lr_time_1=0.005,
        lr_time_2=0.001 / nfe,
        shift_lr=12.0 / nfe,
        prior_timesteps=PRIOR_TIMESTEPS["sd"][nfe],
        match_prior=True,
        loss_type="LPIPS",
        visualize=False,
    )
    configuration = ModelConfig(
        net=checkpointed_ld3_model(runtime) if nfe >= 8 else adapter.model_fn,
        decoding_fn=adapter.decoder,
        noise_schedule=adapter.noise_schedule,
        solver=adapter.solver,
        solver_name="ipndm",
        order=2,
        steps=nfe,
        prior_bound=0.001 * 64 * 64 * 4 / (nfe * nfe),
        resolution=64,
        channels=4,
        time_mode="time",
        snapshot_path=str(destination),
        device="cuda",
    )
    phase = "optimization"
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer = LD3Trainer(configuration, training)
    trainer.loss_fn.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"phase": "ld3_optimization", "nfe": nfe, "official_low_gpu": nfe >= 8}), flush=True)
    try:
        trainer.train(2, 3)
    except BaseException as exc:
        with (destination / "failed-attempts.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "job_id": os.environ.get("GENODE_JOB_ID"),
                        "error": repr(exc),
                        "phase": phase,
                        "counts": counters,
                        "teacher_trajectories": len(data),
                        "teacher_gpu_seconds": teacher_seconds,
                        "optimization_gpu_seconds": time.perf_counter() - start,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "official_low_gpu": nfe >= 8,
                    }
                )
                + "\n"
            )
        raise
    torch.cuda.synchronize()
    optimization_seconds = time.perf_counter() - start
    hook.remove()
    runtime.verify_frozen()
    checkpoint = destination / "best_v2.pt"
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)["best_t_steps"].numpy()
    if len(value) != 2 * (nfe + 1):
        raise RuntimeError("LD3 checkpoint does not contain the requested two complete time arrays.")
    provenance = {
        "source_revision": runtime.metadata["source_revision"],
        "backbone_revision": adapter.backbone_revision,
        "recipe": "official_SD15_iPNDM_order2_LPIPS",
        "checkpoint_sha256": sha256_file(checkpoint),
        "teacher_nfe": teacher_nfe,
        "teacher_schedule": "official_GITS_SD",
        "train_prompt_ids": [p["prompt_id"] for p in prompts[:25]],
        "holdout_prompt_ids": [p["prompt_id"] for p in prompts[25:]],
        "noise_seeds": list(NOISE_SEEDS),
        "optimizer_seed": seed,
        "cfg": 7.5,
        "teacher_trajectories": 100,
        "teacher_gpu_seconds": teacher_seconds,
        "optimization_gpu_seconds": optimization_seconds,
        "counts": counters,
        "official_low_gpu": nfe >= 8,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_model": torch.cuda.get_device_name(),
        "budget_matched_to_gico": False,
    }
    schedule = LD3Schedule(tuple(value[: nfe + 1].tolist()), tuple(value[nfe + 1 :].tolist()), provenance)
    write_new_json(
        destination / "schedule.json",
        {
            "integration_times": list(schedule.integration_times),
            "evaluation_times": list(schedule.evaluation_times),
            "order": 2,
            "provenance": provenance,
        },
    )
