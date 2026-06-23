# GenODE Lessons

- 2026-06-23: `context_sample_count` must cap global SER/schedule workload before expensive trace or inference calls. A per-seed/member cap made molecule SER run `256 * 3 seeds * 6 members = 4608` traces and triggered OOMs; tests should include multi-seed/multi-member cases and assert total selected work, not only per-group metadata.
