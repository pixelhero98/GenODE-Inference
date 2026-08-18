# Contributing to GenODE

Thank you for helping improve GenODE. Bug reports, documentation corrections,
tests, and focused code changes are welcome through GitHub issues and pull
requests.

## Development setup

GenODE requires Python 3.11 or newer. Create an isolated environment, then run:

```bash
python -m pip install -e ".[medical,test]"
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src tests
python -m pytest -q
python -m pip check
git diff --check
```

Keep changes small, add regression tests for behavioral fixes, and update the
README when a public interface changes. Preserve strict artifact validation,
semantic identities, deterministic behavior, and portable relative inputs.

Do not commit credentials, private or cluster paths, datasets, generated
results, logs, trained weights, or checkpoints. External code, model assets,
and reference data must have clear provenance and compatible terms recorded in
`THIRD_PARTY_NOTICES.md` when applicable.

By submitting a contribution, you agree that it may be distributed under the
repository's MIT License.
