# Distributional GICO on frozen BézierFlow

The common GICO trainer accepts explicit `paired-cifar-kid-v1` CIFAR evidence
in addition to paired LPIPS fidelity evidence. The image-fidelity CLI remains
LPIPS-only. KID evidence goes through the common `genode.gico.training.fit`
interface with global teacher and student conditioning.

The reward is `(KID_uniform - KID_candidate) / frozen_reward_scale`. KID uses
the unbiased cubic-kernel estimator, including legitimate negative estimates;
it is never clamped or converted to a log ratio. Repeat measurements are paired
and averaged before constructing labels. Calibration uses training data only,
balanced across training NFEs. Temperatures use raw KID-improvement units.

Each row records `image_objective.protocol`, the frozen `generator` binding,
feature-weight and implementation hashes, `sample_block` (all generated seeds
and noise hashes), and `reference_block` (dataset hash and individual real-image
indices). `reference_identity(reference_block)` constructs the reference ID.
Pairing requires the complete blocks to agree. Training, selection and test
provenance checks detect partial overlaps as well as renamed panels.

For the frozen BézierFlow adapter, `warp_frozen_grids` interpolates both learned
grids by their normalized node index using the density-decoded clock. Uniform
density recovers both original grids exactly. The learned transformation, base
network, endpoints and Euler update rule remain frozen. This is a schedule
composition with the frozen two-time sampler, not a refit of BézierFlow. Verify
uniform image parity, finite outputs, realized grids and actual backbone calls
in the native runtime before collecting evidence.

Teacher ranking/regression, regret-based teacher selection, student distillation
plus teacher-score chasing, and held-out measured-utility student selection use
the shared implementations. Held-out utility is paired KID improvement for this
objective. Inference uses only the frozen student and the composed clock.

KID and LPIPS objectives have separate versioned identities and calibrations.
Neither historical evidence from another generator nor a fidelity policy may
be relabeled as distributional evidence. Improved KID during selection does not
establish improved FID or generalization; evaluate frozen choices separately.
