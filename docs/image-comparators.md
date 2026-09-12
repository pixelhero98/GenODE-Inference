# ReFlow image comparison bridge

The loader accepts only the SHA-256-verified official `reflow_1.pth` checkpoint
(`27d1463f573556765d380b1983664d24e00a194853e0778f5af18fcdf34500de`).
Its original training payload contains NumPy optimizer scalars, so it needs the
original pickle reader after this exact-file check. Parameters and EMA tensors
are still restored with strict name, shape and tensor-count validation.

`genode.image_comparators` provides common checkpoint loading, a counted Euler
sampler and EDM Inception FID accumulation. It does not implement or rename
LD3 or BézierFlow training. Their schedules and transformations must come from
the respective upstream methods.

The integration targets [BézierFlow](https://github.com/KAIST-Visual-AI-Group/BezierFlow)
revision `63ccd10454919a61804536a2c59609256910cf02` and
[RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)
revision `5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6`.
Record these revisions, the model configuration and checkpoint SHA-256 in each
experiment manifest. All methods must load the same `reflow_1.pth` EMA weights.

The loader imports the original NCSN++ implementation directly from its checkout,
strictly restores the model parameters and buffers, validates every EMA tensor,
then freezes the EMA network. It does not import the original TensorFlow-based
training checkpoint utility or restore an optimizer. RF's absolute `models`,
`op` and `sde_lib` imports are scoped during initialization to preserve another
method's module namespace. Initialize runtimes before starting other threads.
The original CUDA operators and their build requirements remain required;
there is no substitute network or silent operator fallback.

The common RF convention follows BézierFlow: progress zero to one maps to
model time `0.0001` to `0.9999`, and continuous model times are multiplied by
`1000` before the network call. The original standalone RF sampler's `999`
multiplier is a different protocol and must not be mixed into this comparison.
Normalized GICO clocks can use `reflow_grid` to map onto this common horizon.
Euler makes precisely one network call per interval and validates the executed
float32 grid. `ReflowVelocity.bezier` exposes the upstream solver signature;
its call counter also counts evaluations inside a transformed velocity field.

Two-time BézierFlow and LD3 schedules keep integration nodes strictly increasing,
but their published offset clipping can produce equal adjacent model times.
These still evaluate distinct evolving states and each call counts toward NFE.
Use `allow_repeated_model_times=True` in both learned-grid and executed-time
validation for those adapters; Base Euler and GICO retain the strict default.
Do not jitter or project a learned schedule to pass validation. When constructing
upstream training configurations directly, preserve the authors' hyperparameter
adjustments, including dividing the model-time learning rate by NFE. Record the
resolved values, fitting allowance and checkpoint-selection rule for each method.

FID uses only the official EDM `inception-2015-12-05.pkl` detector with its
matching `cifar10-32x32.npz` reference moments. Verify and record both asset
hashes before loading the trusted pickle. The pinned BézierFlow checkout
contains the required StyleGAN `torch_utils` and `dnnlib` imports. Conversion
from centered images uses `clip(255 * (x + 1) / 2, 0, 255).to(uint8)`.
Every method must use identical image seeds, detector, reference moments and
sample count. `evaluate_moments` counts the actual final partial batch, and
per-image noise seeds make pairing independent of batch boundaries. Covariance
uses the observed count minus one. Do not combine a second Inception extractor
with the EDM reference statistics or divide by a padded batch count.

Ten-thousand-image FID is a preliminary screening estimate, distinct from the
usual 50,000-image result. Report the actual count, training costs, seeds and
which datasets supplied training/calibration/validation evidence. No held-out
NFE claim follows from fitting a separate model at every evaluated NFE.
