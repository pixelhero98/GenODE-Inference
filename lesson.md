# genODE Lessons

- Split identity must be explicit in schedule evaluation. Validation, calibration, and locked-test rows should carry `split_phase` through protocol hashes and output names so reusable row artifacts cannot be confused.
- Forecast schedule evaluation should batch examples through the OTFlow sampler when possible. Full split evaluation is otherwise dominated by per-example Python overhead.
- Train-tuning sample fractions should be interpreted against the validation-normalized train reference universe, not raw stride-window dataset length.
- Context rewards must be paired inside exact `(dataset, solver, NFE, context_id, seed)` groups. Never build rewards by aggregating across series, context, seed, or solver/NFE cells.
- Context-conditional support selection should train only on measured fixed/SER support rows. Reject BO/candidate schedules and duplicate support rows before teacher or guard construction.
- Teacher checkpoint selection should match the deployed support-choice policy. Use context-disjoint and series-disjoint top-1/top-2 support diagnostics with pairwise/Spearman sanity constraints rather than scalar loss alone.
- Student support policies should report both pre-guard argmax behavior and final guarded deployment behavior. Locked-test reporting must only apply the frozen calibration guard.
- Calibration guards should record explicit holdout provenance and row hashes. A guard built from locked-test rows or missing context/series holdout provenance is invalid.
- GPU Slurm jobs should request explicit CPU and memory resources instead of relying on cluster defaults.
- Git ignore rules for local data and outputs should be anchored at the repository root so source packages such as `src/genode/data` are never hidden.
