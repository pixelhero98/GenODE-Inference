# genODE Lessons

- Split identity must be explicit in schedule evaluation. Validation, calibration, and locked-test rows should carry `split_phase` through protocol hashes and output names so reusable row artifacts cannot be confused.
- Forecast schedule evaluation should batch examples through the OTFlow sampler when possible. Full split evaluation is otherwise dominated by per-example Python overhead.
- Train-tuning sample fractions should be interpreted against the validation-normalized train reference universe, not raw stride-window dataset length.
- Context rewards must be paired inside exact `(dataset, solver, NFE, context_id, seed)` groups. Never build rewards by aggregating across series, context, seed, or solver/NFE cells.
- GIPO should train only from measured fixed/SER supervision rows. Reject BO/candidate schedules and duplicate supervision rows before teacher or student target construction.
- Teacher and student schedule representations must match. Store canonical density mass, feed train-normalized log density to the teacher, and derive solver grids only by inverse CDF.
- Teacher checkpoint selection should use context-disjoint and series-disjoint calibration diagnostics, while locked-test rows remain reporting-only.
- Locked-test reporting must only apply the frozen density policy and should never tune thresholds, choose checkpoints, or refit density metadata from locked-test metrics.
- GPU Slurm jobs should request explicit CPU and memory resources instead of relying on cluster defaults.
- Git ignore rules for local data and outputs should be anchored at the repository root so source packages such as `src/genode/data` are never hidden.
- Deprecated CLI flags must be removed from tests, docs, and Slurm examples in the same cleanup as the parser change; stale argparse inputs otherwise turn canonicalization into a test failure.
- The `/projects/b35z/genode/envs/genode-py311` environment is x86_64. Use `b35z.macs3.isambard` for login-side checks; `b35z.3.isambard` exposes an aarch64 Python ABI and will fail importing the x86_64 NumPy wheel.
- Slurm heredoc Python only sees exported shell variables. Pass experiment roots as CLI args or `export` them before inline Python; otherwise final merge/manifest steps can fail after expensive calibration work.
