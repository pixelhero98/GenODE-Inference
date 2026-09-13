"""Native image LPIPS/KID and frozen BezierFlow KID selection execution."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from genode.gico.image_objective import IMAGE_OBJECTIVE, lpips_values, validate_image_objective
from genode.gico.kid_objective import KID_OBJECTIVE, cubic_kid
from genode.gico.task_evaluators import FrozenModules, tensor_digest, verify_file
from genode.image_comparators.fid import edm_uint8, load_detector, seeded_noise
from genode.latent_clock.artifacts import canonical_sha256, sha256_file


def lpips_implementation_identity():
    import lpips

    from genode.gico import image_objective

    root = Path(lpips.__file__).parent
    return canonical_sha256(
        {
            "wrapper": sha256_file(Path(image_objective.__file__)),
            "lpips": {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*.py"))},
        }
    )


def feature_implementation_identity():
    import sys

    return canonical_sha256(
        {
            "kid": sha256_file(Path(sys.modules[cubic_kid.__module__].__file__)),
            "decoding": sha256_file(Path(sys.modules[edm_uint8.__module__].__file__)),
        }
    )


class ImageEvaluator(FrozenModules):
    def __init__(self, config):
        self.config, self.device, self.objective = config, config["device"], config["objective"]
        validate_image_objective(self.objective)
        self.bezier = None
        if self.objective["protocol"] == KID_OBJECTIVE:
            from genode.image_comparators.bezier import FrozenBezierGenerator

            self.bezier = FrozenBezierGenerator({**config["bezier"], "device": self.device})
            if self.bezier.objective != self.objective:
                raise ValueError("Frozen BezierFlow objective differs from the measured evidence.")
            self.detector = self.bezier.detector
            self.binding = self.bezier.binding
            self.bind_modules([self.bezier.net, self.bezier.schedule, self.detector])
            return
        from genode.backbones.loading import load_verified_image_backbone
        from genode.backbones.protocol import ImageBackboneManifest
        from genode.gico.image_conditional_context import native_contexts

        manifest = ImageBackboneManifest.from_manifest_dict(
            json.loads(Path(config["backbone_manifest"]).read_text(encoding="utf-8"))
        )
        self.model = load_verified_image_backbone(
            manifest, checkpoint_path=config["checkpoint"], source_root=config["source_root"]
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        self.table, self.binding = native_contexts(self.model)
        modules = [self.model]
        if self.objective["protocol"] == IMAGE_OBJECTIVE:
            import lpips

            verify_file(config["lpips_checkpoint"], self.objective["lpips"]["weights_sha256"])
            if (
                importlib.metadata.version("lpips") != self.objective["lpips"]["version"]
                or lpips_implementation_identity() != self.objective["lpips"]["implementation_sha256"]
            ):
                raise ValueError("LPIPS package version or implementation differs from the frozen objective.")
            # Full user-supplied state avoids implicit VGG or LPIPS downloads.
            self.lpips = lpips.LPIPS(net="vgg", pnet_rand=True, pretrained=False).to(self.device)
            self.lpips.load_state_dict(
                torch.load(config["lpips_checkpoint"], map_location=self.device, weights_only=True)
            )
            modules.append(self.lpips)
        else:
            self.binding = {**self.binding, "backbone": manifest.model_key, "clock_protocol": "density64-euler"}
            verify_file(config["feature_checkpoint"], self.objective["features"]["weights_sha256"])
            if self.objective["features"]["implementation_sha256"] != feature_implementation_identity():
                raise ValueError("KID feature/decoding implementation differs from the evidence.")
            self.detector = load_detector(config["feature_checkpoint"], device=self.device)
            modules.append(self.detector)
        self.bind_modules(modules)

    @torch.no_grad()
    def measure(self, row, clocks, context, case, output):
        from genode.benchmarks.image.noise import generate_seeded_image_noise
        from genode.solvers.euler import integrate_euler

        if (
            row["image_objective"] != self.objective
            or row["backbone_binding"] != self.binding
            or row["solver"] != "euler"
        ):
            raise ValueError("Image selection objective or backbone/context binding changed.")
        label = row.get("class_id")
        native = [0.0] if self.bezier else self.table[label if label is not None else 0]
        if not np.array_equal(np.asarray(native, dtype=np.float32), np.asarray(context, dtype=np.float32)):
            raise ValueError("Image selection context differs from the frozen native embedding.")
        lpips = self.objective["protocol"] == IMAGE_OBJECTIVE
        seeds = [row["seed"]] if lpips else row["sample_block"]["seeds"]
        noises = (
            seeded_noise(seeds, device=self.device)
            if self.bezier
            else generate_seeded_image_noise(row["task"], seeds).values.to(self.device)
        )
        expected_noise = [row["target"]["noise_sha256"]] if lpips else row["sample_block"]["noise_sha256"]
        if [tensor_digest(x) for x in noises] != expected_noise or len(clocks) != len(seeds):
            raise ValueError("Image selection generation noise differs from the paired sample block.")
        images = []
        for noise, clock in zip(noises, clocks, strict=True):
            if self.bezier:
                if row["nfe"] != self.bezier.nfe:
                    raise ValueError("Frozen BezierFlow NFE differs from the requested clock.")
                image = self.bezier.sample(noise[None], clock["density_mass"])
            else:
                labels = None if label is None else torch.tensor([label], device=self.device)
                image = integrate_euler(
                    lambda x, t, labels=labels: self.model(x, t, labels),
                    noise[None],
                    target_nfe=row["nfe"],
                    time_grid=torch.tensor(clock["time_grid"], dtype=noise.dtype, device=self.device),
                ).final_state
            images.append(image)
        images = torch.cat(images)
        if lpips:
            target = np.load(case["target"], allow_pickle=False)
            if tensor_digest(target) != row["target"]["image_sha256"]:
                raise ValueError("Paired LPIPS target image changed.")
            metrics = {
                "lpips": float(lpips_values(self.lpips, images, torch.as_tensor(target, device=self.device)).mean())
            }
        else:
            verify_file(case["reference_features"], case["reference_features_sha256"])
            if case["reference_block"] != row["reference_block"]:
                raise ValueError("KID reference indices/dataset/class differ from the frozen paired block.")
            reference = np.load(case["reference_features"], allow_pickle=False)
            if len(reference) != len(row["reference_block"]["indices"]):
                raise ValueError("KID reference feature block is incomplete.")
            features = self.detector(edm_uint8(images), return_features=True).double().cpu().numpy()
            metrics = {"kid": cubic_kid(features, reference)}
        output.mkdir()
        np.save(output / "images.npy", images.cpu().numpy(), allow_pickle=False)
        (output / "metrics.json").write_text(json.dumps(metrics, allow_nan=False), encoding="utf-8")
        return metrics
