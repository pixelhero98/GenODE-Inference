"""Verified frozen BezierFlow generator; its two learned grids remain frozen."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from genode.gico.clocks import materialize
from genode.gico.kid_objective import KID_OBJECTIVE, cubic_kid
from genode.image_comparators.fid import edm_uint8, load_detector, seeded_noise
from genode.image_comparators.frozen_clock import warp_frozen_grids
from genode.image_comparators.reflow import ReflowVelocity, load_reflow, validate_executed_model_times
from genode.latent_clock.artifacts import canonical_sha256, sha256_file
from genode.latent_clock.rewards import _module_fingerprint


class FrozenBezierGenerator:
    def __init__(self, config):
        from omegaconf import OmegaConf

        self.config = config
        self.device = config["device"]
        self.nfe = int(config["scope"]["nfe"])
        if self.nfe not in (4, 6, 8, 10):
            raise ValueError("Frozen BezierFlow supports NFE 4, 6, 8 or 10.")
        self.assets = Path(config["assets"])
        bf = Path(config["bezier_source"])
        for key in ("bezier", "reflow"):
            root = config[key + "_source"]
            if (
                subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
                != config["source_revisions"][key]
            ):
                raise ValueError("Upstream revision changed")
            if subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip():
                raise ValueError("Upstream source is dirty")
        for name, digest in config["asset_sha256"].items():
            if sha256_file(self.assets / name) != digest:
                raise ValueError("Asset hash mismatch: " + name)
        if sha256_file(config["frozen_bezier"]) != config["frozen_bezier_sha256"]:
            raise ValueError("Frozen BezierFlow state changed")
        settings = OmegaConf.to_container(
            OmegaConf.load(bf / "configs/reflow/cifar10_rf_gaussian_ddpmpp.yml"),
            resolve=True,
        )
        self.net = load_reflow(
            config["reflow_source"],
            self.assets / "reflow_1.pth",
            settings,
            device=self.device,
        )
        if _module_fingerprint(self.net) != config["ema_sha256"]:
            raise ValueError("EMA differs")
        self.velocity = ReflowVelocity(self.net)
        sys.path.insert(0, str(bf))
        import noise_schedulers as ns
        from samplers.euler import Euler
        from samplers.general_solver import ODESolver

        self.state = torch.load(config["frozen_bezier"], map_location=self.device, weights_only=True)
        from genode.image_comparators.reflow import validate_learned_grids

        validate_learned_grids(
            self.state["grid1"].cpu().numpy(),
            self.state["grid2"].cpu().numpy(),
            self.nfe,
            allow_repeated_model_times=True,
        )
        original = ns.NoiseScheduleRF()
        self.schedule = ns.NoiseSchedulerSI(p_order=32, orig_sched=original).to(self.device)
        self.schedule.load_state_dict(self.state["scheduler_params"])
        self.schedule.eval().requires_grad_(False)
        self.transform_hash = _module_fingerprint(self.schedule)
        self.model = ns.make_interpolant_converter(
            original,
            self.schedule,
            self.velocity.bezier,
            ODESolver(original, "v_prediction", "v"),
        )
        self.solver = Euler(self.schedule, "v_prediction", "v")
        self.detector = load_detector(self.assets / "inception-2015-12-05.pkl", device=self.device)
        self.detector_hash = _module_fingerprint(self.detector)
        self.times = []
        self.forwards = 0

        def observe(module, inputs):
            self.forwards += len(inputs[0])
            if not torch.all(inputs[1] == inputs[1][0]):
                raise ValueError("Global sampler has inconsistent model times")
            self.times.append(float(inputs[1][0]))

        self.net.register_forward_pre_hook(observe)
        self.binding = {
            "backbone": f"reflow1-frozen-bezier-n{self.nfe}",
            "checkpoint_sha256": config["asset_sha256"]["reflow_1.pth"],
            "transform_sha256": config["frozen_bezier_sha256"],
            "clock_protocol": "frozen-two-grid-index-warp-v1",
        }
        self.objective = {
            "protocol": KID_OBJECTIVE,
            "generator": self.binding,
            "features": {
                "weights_sha256": config["asset_sha256"]["inception-2015-12-05.pkl"],
                "implementation_sha256": canonical_sha256(
                    {
                        "kid": sha256_file(Path(sys.modules[cubic_kid.__module__].__file__)),
                        "decoding": sha256_file(Path(sys.modules[edm_uint8.__module__].__file__)),
                    }
                ),
                "input_protocol": "edm-uint8",
                "estimator": "unbiased-cubic",
            },
        }

    @torch.no_grad()
    def sample(self, noise, mass=None, original=False):
        if original:
            first, second = self.state["grid1"], self.state["grid2"]
        else:
            grid = materialize(mass, "euler", self.nfe)
            first, second = warp_frozen_grids(grid, self.state["grid1"].cpu(), self.state["grid2"].cpu())
            first, second = [
                torch.tensor(x, dtype=self.state["grid1"].dtype, device=self.device) for x in (first, second)
            ]
            if torch.any(torch.diff(first) <= 0):
                raise ValueError("Realized integration grid degenerates at solver precision")
        self.times.clear()
        before = self.forwards
        result = self.solver.sample_simple(self.model, noise, first, second, NFEs=self.nfe)
        validate_executed_model_times(self.times, self.nfe, allow_repeated_model_times=True)
        if self.forwards - before != len(noise) * self.nfe or not torch.isfinite(result).all():
            raise ValueError("Incorrect NFE or nonfinite image")
        return result

    @torch.no_grad()
    def features(self, mass, seeds):
        chunks = []
        for start in range(0, len(seeds), self.config["sampling_batch"]):
            image = self.sample(
                seeded_noise(seeds[start : start + self.config["sampling_batch"]], device=self.device),
                mass,
            )
            chunks.append(self.detector(edm_uint8(image), return_features=True).double().cpu().numpy())
        return np.concatenate(chunks)

    def verify_frozen(self):
        for model, digest in (
            (self.net, self.config["ema_sha256"]),
            (self.schedule, self.transform_hash),
            (self.detector, self.detector_hash),
        ):
            if _module_fingerprint(model) != digest or any(
                p.requires_grad or p.grad is not None for p in model.parameters()
            ):
                raise ValueError("Frozen generator or scorer changed")
