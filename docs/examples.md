# Portable examples

After installing GenODE, run the source distribution's asset-free decoder example:

```bash
python examples/decode_clock.py
```

It materializes one reference density for Euler and Heun at the same denoiser-call budget. It verifies endpoints and macro-step counts without loading a generator or making a quality claim.

With an canonical `genode-gico` policy artifact and its frozen native context:

```python
from genode.gico.policy import load_context_embedding_table, load_policy

contexts = load_context_embedding_table("contexts.npz")
policy = load_policy("policy", policy_kind="stochastic")
context_id = next(iter(contexts))
grid = policy.materialize(
    contexts[context_id],
    "euler",
    8,
    seed=412,
    request_id=f"{context_id}:euler:8:seed:31:member:0:replicate:0",
)
print(grid)
```

Use `deterministic` to load the deterministic policy. Reuse the sampled grid through a complete generated trajectory; change request identity for each new member or replicate. The native generator and its exact solver must consume this grid.

Use the README's fitting configuration with a [completed collection manifest](collection.md). Fitting defaults to deterministic; request `both` or `stochastic` explicitly. Neither policy needs a generator/scorer during fitting or selection. Then run:

```bash
genode-train-gico --config train.json --dry-run
genode-train-gico --config train.json --policy-kind both
genode-report-gico-locked-test --artifact policy --policy-kind stochastic --rows test.jsonl --contexts test-contexts.npz --output test-report.json
```

These commands require independently collected assets and complete paired evidence. Reduced `purpose: functional` panels are integration fixtures; they cannot substantiate benchmark results.
