from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genode.latent_clock.adapters.ipndm import IPNDMAdapter
from genode.latent_clock.adapters.sana import SanaFlowEulerAdapter
from genode.latent_clock.artifacts import implementation_sha256
from genode.latent_clock.clocks import Clock
from genode.latent_clock.rewards import _module_fingerprint


@dataclass
class FrozenRuntime:
    adapter: Any
    modules: dict[str, Any]
    fingerprints: dict[str, str]
    native_sample: Any
    native_clock: Any
    metadata: dict[str, Any]

    def verify_frozen(self) -> None:
        for name, module in self.modules.items():
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} is not frozen in evaluation mode.")
            if _module_fingerprint(module) != self.fingerprints[name]:
                raise RuntimeError(f"Frozen weights changed: {name}")


def _freeze(modules: dict[str, Any]) -> dict[str, str]:
    for module in modules.values():
        module.eval().requires_grad_(False)
    return {key: _module_fingerprint(module) for key, module in modules.items()}


def _source_revision(path: Path, expected: str) -> str:
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected:
        raise ValueError(f"External source revision mismatch at {path}: {revision}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise ValueError(f"External source checkout is dirty at {path}.")
    return revision


def load_runtime(config_path: str | Path) -> FrozenRuntime:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen generator loading and generation require CUDA.")
    config = json.loads(Path(config_path).read_text())
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(config["source"]).resolve(strict=True)
    revision = _source_revision(source, config["source_revision"])
    sys.path.insert(0, str(source))
    assets = json.loads(Path(config["asset_manifest"]).read_text())
    if config["backbone"] == "sana":
        runtime = _load_sana(source, revision, assets)
    elif config["backbone"] == "sd15":
        runtime = _load_sd15(source, revision, assets)
    else:
        raise ValueError(f"Unknown backbone {config['backbone']!r}")
    runtime.metadata["implementation_sha256"] = implementation_sha256()
    return runtime


def _load_sana(source: Path, revision: str, assets: dict) -> FrozenRuntime:
    import app.sana_pipeline as sana_pipeline
    import torch
    from app.sana_pipeline import SanaPipeline
    from diffusion import FlowEuler
    from diffusion.model.builder import vae_decode
    from diffusion.model.utils import prepare_prompt_ar
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text_snapshot = assets["sana_text"]["snapshot"]

    def local_text_encoder(name: str, device: Any) -> tuple[Any, Any]:
        if name != "gemma-2-2b-it":
            raise ValueError(f"The pinned SANA-600M runtime refuses text encoder {name!r}.")
        tokenizer = AutoTokenizer.from_pretrained(text_snapshot, local_files_only=True)
        tokenizer.padding_side = "right"
        encoder = (
            AutoModelForCausalLM.from_pretrained(text_snapshot, torch_dtype=torch.bfloat16, local_files_only=True)
            .get_decoder()
            .to(device)
        )
        return tokenizer, encoder

    sana_pipeline.get_tokenizer_and_text_encoder = local_text_encoder
    config_text = (source / "configs/sana_config/512ms/Sana_600M_img512.yaml").read_text()
    original_vae = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
    if config_text.count(original_vae) != 1:
        raise ValueError("Pinned SANA config has an unexpected VAE reference.")
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as stream:
        stream.write(config_text.replace(original_vae, assets["sana_vae"]["snapshot"]))
        resolved_config = stream.name
    try:
        pipeline = SanaPipeline(resolved_config)
    finally:
        Path(resolved_config).unlink(missing_ok=True)
    checkpoint = Path(assets["sana"]["snapshot"]) / "checkpoints/Sana_600M_512px_MultiLing.pth"
    pipeline.from_pretrained(str(checkpoint))
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    modules = {"generator": pipeline.model, "text_encoder": pipeline.text_encoder, "vae": pipeline.vae}
    fingerprints = _freeze(modules)
    backbone_revision = assets["sana"]["revision"]

    @torch.inference_mode()
    def encode(prompt: str) -> tuple:
        text = prepare_prompt_ar(prompt, pipeline.base_ratios, device=pipeline.device, show=False)[0].strip()
        max_length = pipeline.config.text_encoder.model_max_length
        chi = pipeline.config.text_encoder.chi_prompt
        if chi:
            prefix = "\n".join(chi)
            max_length = len(pipeline.tokenizer.encode(prefix)) + max_length - 2
            text = prefix + text
        token = pipeline.tokenizer(
            [text], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).to(pipeline.device)
        indices = [0] + list(range(-pipeline.config.text_encoder.model_max_length + 1, 0))
        condition = pipeline.text_encoder(token.input_ids, token.attention_mask)[0][:, None][:, :, indices]
        condition = condition.to(pipeline.weight_dtype)
        mask = token.attention_mask[:, indices]
        uncondition = pipeline.null_caption_embs[:, None].to(pipeline.weight_dtype)
        pooled = ((condition[:, 0].float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))[0]
        kwargs = {
            "data_info": {
                "img_hw": torch.tensor([[512.0, 512.0]], device=pipeline.device),
                "aspect_ratio": torch.tensor([[1.0]], device=pipeline.device),
            },
            "mask": mask,
        }
        return condition, uncondition, pooled.cpu().numpy(), kwargs

    def latent(seed: int, context: Any) -> Any:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        return torch.randn(
            1,
            pipeline.config.vae.vae_latent_dim,
            pipeline.latent_size,
            pipeline.latent_size,
            generator=generator,
            device=pipeline.device,
        )

    def decode(value: Any) -> Any:
        return vae_decode(pipeline.config.vae.vae_type, pipeline.vae, value.to(pipeline.vae_dtype))

    sampler = FlowEuler(pipeline.model, None, None, cfg_scale=4.5, flow_shift=3.0, apg=None)
    adapter = SanaFlowEulerAdapter(
        sampler=sampler,
        context_encoder=encode,
        latent_factory=latent,
        decoder=decode,
        backbone_revision=backbone_revision,
    )

    def native_clock(nfe: int) -> Clock:
        sampler.scheduler.set_timesteps(nfe, device=pipeline.device)
        nodes = 1.0 - sampler.scheduler.sigmas.double().cpu().numpy()
        return Clock("native", nfe, tuple(nodes), "native_sana_shift_3")

    @torch.inference_mode()
    def native_sample(context: Any, seed: int, nfe: int) -> Any:
        condition, uncondition, kwargs = adapter._opaque_contexts[context.context_id]
        sampler.condition, sampler.uncondition, sampler.model_kwargs = condition, uncondition, kwargs
        return decode(sampler.sample(latent(seed, context), steps=nfe))

    return FrozenRuntime(
        adapter,
        modules,
        fingerprints,
        native_sample,
        native_clock,
        {
            "backbone": "sana",
            "source_revision": revision,
            "cfg": 4.5,
            "pag": False,
            "resolution": 512,
            "flow_shift": 3.0,
            "assets": assets["sana"]["revision"],
        },
    )


def _load_sd15(source: Path, revision: str, assets: dict) -> FrozenRuntime:
    from types import SimpleNamespace

    # The pinned LD3 vendored tree omits package __init__ files; its legacy
    # find_packages() install is empty. Import the original namespace directly.
    sys.path.insert(0, str(source / "src/taming-transformers"))
    import torch
    from models.latent_diff import get_pretrained_conditioned_ldm_model
    from samplers.ipndm import iPNDM

    args = SimpleNamespace(
        config=str(source / "configs/stable-diffusion/v1-inference.yaml"),
        ckp_path=str(Path(assets["sd15"]["snapshot"]) / "v1-5-pruned-emaonly.ckpt"),
        scale=7.5,
        low_gpu=False,
        H=512,
        W=512,
        f=8,
        C=4,
    )
    model_fn, model, decoder, schedule, *_ = get_pretrained_conditioned_ldm_model(args)
    modules = {"sd15_generator_text_vae": model}
    fingerprints = _freeze(modules)

    @torch.inference_mode()
    def encode(prompt: str) -> tuple:
        condition = model.get_learned_conditioning([prompt])
        uncondition = model.get_learned_conditioning([""])
        return condition, uncondition, condition.float().mean(dim=1)[0].cpu().numpy()

    def latent(seed: int, context: Any) -> Any:
        return torch.randn(1, 4, 64, 64, generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda")

    solver = iPNDM(schedule)
    adapter = IPNDMAdapter(
        model_fn=model_fn,
        decoder=decoder,
        solver=solver,
        noise_schedule=schedule,
        context_encoder=encode,
        latent_factory=latent,
        backbone_revision=assets["sd15"]["revision"],
    )

    def native_clock(nfe: int) -> Clock:
        times = solver.get_time_steps("time_uniform", schedule.T, schedule.eps, nfe, "cuda").double().cpu().numpy()
        nodes = (times[0] - times) / (times[0] - times[-1])
        return Clock("native", nfe, tuple(nodes), "ld3_time_uniform")

    @torch.inference_mode()
    def native_sample(context: Any, seed: int, nfe: int) -> Any:
        condition, uncondition = adapter._opaque_contexts[context.context_id]
        times = solver.get_time_steps("time_uniform", schedule.T, schedule.eps, nfe, "cuda")
        value = solver.sample_simple(
            model_fn,
            schedule.prior_transformation(latent(seed, context)),
            times,
            times,
            order=2,
            condition=condition,
            unconditional_condition=uncondition,
        )
        return decoder(value)

    return FrozenRuntime(
        adapter,
        modules,
        fingerprints,
        native_sample,
        native_clock,
        {
            "backbone": "sd15",
            "source_revision": revision,
            "cfg": 7.5,
            "order": 2,
            "resolution": 512,
            "assets": assets["sd15"]["revision"],
        },
    )
