# Third-party notices

GenODE's reference-clock catalog records the exact upstream repository,
revision, source location, and license for every transferred schedule. The
implementations remain source-specific references; using their normalized
nodes with a GenODE backbone does not reproduce or imply the upstream model or
optimizer.

## AYS constants

The AYS Stable Diffusion 1.5 reference nodes are taken from Hugging Face
Diffusers at revision `50e7158093710f9c1b4ea9ff100137a91c9228f3`.

- Project: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Sources: `src/diffusers/schedulers/scheduling_utils.py` for the AYS nodes and
  `src/diffusers/schedulers/scheduling_ddim.py` for the scaled-linear beta
  realization used to derive the finite SD1.5 log-sigma terminal
- Pinned scheduler parameters: `num_train_timesteps=1000`,
  `beta_start=0.00085`, `beta_end=0.012`, `beta_schedule=scaled_linear`
- License: Apache License 2.0

## GITS example nodes

The GITS CIFAR-10 reference nodes are taken from diff-sampler at revision
`68d5ce427f261962b89ce3b0ee8f6b29f0577328`.

- Project: [diff-sampler](https://github.com/zju-pi/diff-sampler)
- Source: the pre-specified CIFAR-10 `t_steps` example in `gits-main/README.md`
- License: Apache License 2.0

The complete Apache License 2.0 text is included in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## OTS objective

The linear-VP OTS objective follows DM-NonUniform at revision
`95d4ac6b8a3d1d389ab63a197e1b05d8512b6a99`.

- Project: [DM-NonUniform](https://github.com/scxue/DM-NonUniform)
- Source: `step_optim.py`
- License notice:

```text
MIT License

Copyright (c) 2024 scxue

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## FlowTS power clock

The FlowTS power-clock formula and released exponent are taken from FMTS at
revision `1ec35fb1d3d89d91a1607a9f949a515347d54c8c`.

- Project: [FlowTS](https://github.com/UNITES-Lab/FlowTS)
- Source: `FMTS/Models/interpretable_diffusion/FMTS.py` and `FMTS/run.sh`
- Note: the `XXX` holder below is reproduced verbatim from `FMTS/LICENSE` at the pinned revision.
- License notice:

```text
MIT License

Copyright (c) 2024 XXX

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Image backbone integration

GenODE can bind externally supplied RF++/EDM image backbones at pinned source
revisions. This repository does not vendor the upstream network implementation,
datasets, or checkpoint weights. Their upstream licenses and dataset terms
continue to apply; the registry records those constraints, and generated image
checkpoint archives must not be redistributed without separate permission.
