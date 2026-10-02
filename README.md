# AI Experiments

Personal AI development platform and reproducible experiments. The MacBook is the control plane; GPU-heavy work can move to RunPod as projects require it.

## Current project

[`projects/embedding-retrieval-benchmark`](projects/embedding-retrieval-benchmark) compares dense, lexical, and hybrid retrieval on a fixed corpus. Start with its [README](projects/embedding-retrieval-benchmark/README.md).

## Repository layout

- `docs/`: platform architecture and operating plans.
- `projects/`: self-contained AI experiments.
- `workspace/`: future working area for notebooks, agents, and experiments; this will be the default remote Jupyter directory.

GPU lifecycle commands will be added alongside the projects that need them, rather than pre-building unused automation.
