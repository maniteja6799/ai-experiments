# Personal AI Development Platform

## Intent

`/Users/teja/teja/projects/ai` is the long-lived platform repository. A MacBook Air provides the control plane; ephemeral RunPod GPUs provide CUDA compute when a project needs it. The architecture should stay reproducible, modular, and cost-aware without forcing premature infrastructure.

## Working model

Projects live in `projects/` and own their code, environments, notebooks, data instructions, and results. The `workspace/` directory is the active working area for notebooks, AI agents, experiments, RAG systems, and model prototypes. When remote Jupyter is introduced, it will start in the synced or mounted remote equivalent of `workspace/`.

## Platform principles

- Build a thin vertical slice first: provision, connect, sync, run a workload, inspect it, and tear it down.
- Use Git for code and small fixtures; use a RunPod volume for large datasets, model caches, and generated artifacts.
- Keep secrets in macOS Keychain and out of Git.
- Begin with direct SSH and a RunPod PyTorch image. Add Tailscale, custom images, and provider abstraction only when they solve an active need.
- Prefer a Python Typer CLI behind Make targets once GPU workflows are required.

## Deferred GPU interface

When a project needs it, add the smallest useful set of commands:

```text
make doctor
make gpu-up
make gpu-status
make connect
make sync
make jupyter
make vllm
make gpu-down
```

The corresponding CLI will own provider configuration, SSH connection data, port forwarding, synchronization, and lifecycle state. Commands should be added only with tests and project documentation that exercise them.

## Delivery sequence

1. Create self-contained, reproducible experiments with pinned dependencies and clear reports.
2. Add a shared GPU bootstrap only when an experiment exceeds local capability or needs CUDA.
3. Add batch jobs and serving endpoints as workloads demand them.
4. Add broader MLOps, agents, observability, and multi-provider capabilities when there is a concrete consumer.
