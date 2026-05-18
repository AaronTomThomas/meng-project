# Archived Task Study

Archived task-study experiments live under:

```text
experiments.archive.task_study
```

The maintained archived router package is:

```text
experiments.archive.task_study.gradient_proxy_router
```

See [gradient_proxy_router/README.md](gradient_proxy_router/README.md) for the
full build, analysis, train, eval, and robustness commands.

## Quick Commands

Build the default router dataset:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli build-router-dataset \
  --device cuda
```

Analyze deployable feature predictability:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.analyze_router_predictability \
  --dataset_paths outputs/deployable_routers/gradient_proxy_router/router_dataset.pt \
  --feature_key sequence_only_scalar_features_gradient_state \
  --default_action soft \
  --candidate_actions nondefault \
  --models ridge,mlp \
  --output_path outputs/deployable_routers/gradient_proxy_router/predictability_sequence_features.json \
  --device cuda
```

Train the archived router:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli sweep \
  --device cuda
```

Evaluate a trained checkpoint:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli eval \
  --checkpoint_path outputs/deployable_routers/gradient_proxy_router/focused_sweep/YOUR_CHECKPOINT.pt \
  --device cuda
```
