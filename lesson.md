# genODE Lessons

- Symptom: full-protocol validation/test rows could have mixed split phases if the schedule runner reused its original locked-test-only path. Cause: split selection was hard-coded in the runner even though lower-level evaluation helpers already supported validation versus test datasets. Fix: add a `--split_phase` CLI choice, include it in the protocol hash, and test both hash separation and mocked val/test dispatch. Prevention: when adding an output root for a new protocol split, make the split an explicit part of the run configuration and protocol hash.
- Symptom: the student schedule evaluator could accidentally look like a seventh baseline. Cause: learned schedules share much of the row schema with fixed schedules. Fix: keep `conditional_opd_student` in the evaluator module only, assert it is not in `BASELINE_SCHEDULE_KEYS`, and write `student_rows.*` separately from baseline rows. Prevention: learned methods should use separate registry keys and comparison summaries rather than extending fixed baseline registries.
- Symptom: full SF evaluation with one example per sampler call was correct but would take many hours for 216 baseline cells per split. Cause: `evaluate_forecast_schedule` looped serially over examples even though `OTFlow.sample_future` accepts batched histories. Fix: add `forecast_eval_batch_size` to the runner/evaluator protocol and batch forecast examples while preserving full split and sample counts. Prevention: full-split evaluation protocols should expose batch size in the protocol hash whenever sampler batching changes runtime semantics.
- Symptom: multi-schedule OPD ablation comparisons reported `observed_student_rows` larger than `expected_student_rows` and non-empty missing student cells even when every ablation row existed. Cause: the comparison summary still assumed the selected single-student key `conditional_opd_student_selected`. Fix: derive student schedule keys from the evaluated rows, compute expected/missing cells across all student schedules, and report per-student cell comparisons. Prevention: evaluator summaries that accept schedule-summary files must treat learned schedule keys as a set, not a singleton.
- Symptom: the clean OPD spec rewrite passed the new checker but broke the older Section 15 doc test. Cause: the canonical HTML lost the historical `section-15-baselines` anchor and exact compatibility wording. Fix: restore the anchor and baseline wording while keeping the clean one-teacher method text. Prevention: when rewriting canonical docs, keep stable anchors used by tests and downstream review scripts.
- Symptom: smoke runs with `--seeds 0` trained the OPD teacher from cached seed-1/2 validation rows. Cause: trainer row cleaning checked required seeds but did not filter extra seeds before seed-mean aggregation. Fix: filter baseline, reference, and generated candidate rows by requested seeds in the direct and MLP-flow trainers. Prevention: seed completeness checks should be paired with explicit seed filtering before aggregation.
- Symptom: candidate-pool tests collapsed temperature/noise/Dirichlet variants into fewer schedules than expected. Cause: a uniform source grid is invariant under temperature and can duplicate deterministic perturbations after full-family grid hashing. Fix: test candidate generation with a nonuniform source grid and keep deduplication by full-family grid hash. Prevention: variant-generation tests should avoid symmetry cases unless they explicitly verify deduplication.
- Symptom: Train20 could still compare locked-test students against validation-derived SER rows by default. Cause: the orchestrator generated train-derived SER for initialization but kept an unrelated default comparator path. Fix: default to evaluating the train-derived SER summary under the Train20 output root and only use external comparator rows when explicitly supplied. Prevention: when changing reference data lineage, update train, validation, and comparison defaults together.
- Symptom: perturbation/random candidates could inherit parent pseudo-utility and bias exploit selection. Cause: candidate-pool generation copied `utility` fields while altering grids. Fix: load each direct-student teacher checkpoint and score every complete candidate family before exploit/diverse/random selection. Prevention: any operation that changes a schedule grid must invalidate or recompute surrogate scores.
- Symptom: Train20 train-tuning evaluation failed before writing rows with `ValueError: Seed must be between 0 and 2**32 - 1`. Cause: generated chunk/sample seeds were passed directly to NumPy's global RNG seed API, which rejects large integers. Fix: normalize non-negative seeds centrally in `seed_all` and `_temporary_eval_seed`, and include train-tuning sampling args in the runner protocol hash. Prevention: generated deterministic seeds should pass through a single normalizer before touching bounded RNG seed APIs.

## 2026-05-26 Train20 forecast-only train_tuning conditional dispatch
- Symptom: Train20 baseline train-tuning completed all 216 rows, then failed with `ValueError: train_tuning split is only supported for forecast schedule evaluation` even though the command passed an empty conditional-generation dataset list.
- Cause: the diffusion-flow runner parsed `--conditional_generation_datasets ""` as an empty list but still called the conditional-generation phase, whose train-tuning guard raises before there is any dataset work to do.
- Fix: dispatch forecast and conditional-generation phases only when their parsed dataset lists are non-empty, while preserving the explicit train-tuning error for real conditional-generation datasets.
- Prevention: keep a regression test for forecast-only train-tuning execution with empty conditional datasets, plus a paired test that non-empty conditional train-tuning still raises.

## 2026-05-26 Train20 schedule-role cleanup
- Symptom: a diagnostic low-fraction Train20 run precomputed all fixed schedules on train tuning and would have used SER-PTG metric rows as teacher supervision, mixing final-comparison baselines with teacher demos.
- Cause: fixed baseline, teacher-demo, SER initialization, validation, and locked-test comparison roles were all routed through broad baseline/SER defaults.
- Fix: split roles explicitly (`uniform,late_power_3` teacher fixed demos; SER-PTG init only; all fixed baselines only for locked-test comparison) and keep diagnostic roots out of canonical outputs.
- Prevention: keep tests that assert no all-six train precompute, no SER metric rows in teacher training, no fixed validation dependency, and no automatic SER locked-test comparator.

## 2026-05-26 Train20 eval fraction denominator
- Symptom: `--eval_train_fraction 0.10` selected about 1.44M San Francisco train windows per row, making Train20 schedule evaluation slower than a backbone run.
- Cause: the old sampler interpreted the fraction over all stride-1 train forecast windows, while validation/test use one holdout window per series.
- Fix: added validation-normalized train-tuning sampling with explicit sampler metadata and protocol-hash fields.
- Prevention: Train20 runs must set `--train_tuning_sampling_mode validation_normalized` when percentages are intended as raw split proportions.

## 2026-05-27 BO candidate schedule loading
- Symptom: candidate-pool generation failed when it needed to load observed schedules from a previous BO round, raising `NameError: load_schedules is not defined`.
- Cause: `bo_candidate_pool.py` reused the shared schedule-summary loader for observed BO schedules but did not import it from `candidate_pool.py`.
- Fix: import `load_schedules` and verify candidate-pool tests plus full unittest discovery.
- Prevention: round-dependent BO candidate-pool tests should exercise observed schedule summaries, not only round-0 warmup generation.

## 2026-05-31 V4.3 source package ignore guard
- Symptom: preparing the public source tree showed `src/genode/data` as ignored.
- Cause: `.gitignore` used a broad `data/` rule that also matched package directories named `data`.
- Fix: anchor local dataset/output ignores to the repository root.
- Prevention: after adding ignore rules, run `git status --ignored` and check that source package directories are not hidden.

## 2026-05-31 Slurm GPU memory request
- Symptom: a forecast backbone training job can be OOM-killed before reaching the first checkpoint when the cluster default memory request is too small.
- Cause: some Slurm GPU partitions allocate modest memory per CPU unless the job asks for memory explicitly.
- Fix: generic GPU Slurm examples request explicit CPU and memory resources for backbone training and downstream inference.
- Prevention: GPU training scripts should set CPU and memory requests deliberately instead of relying on site defaults.

## 2026-05-31 V4.3 pooled calibration denominator
- Symptom: pooled forecast calibration can evaluate far too many train windows when the calibration fraction is applied to raw stride-window dataset length.
- Cause: raw train windows are not the same unit as the validation-normalized train universe used by the Train20 forecast protocol.
- Fix: allocate V4.3 pooled counts from the validation-normalized train reference universe plus validation, while sampling train indices over the actual train dataset range.
- Prevention: pooled forecast calibration tests must include a large raw train split regression so 20% cannot accidentally mean 20% of all stride-1 train windows.
