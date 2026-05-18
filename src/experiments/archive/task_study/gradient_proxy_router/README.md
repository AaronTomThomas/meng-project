# Archived Gradient Proxy Router

This archived experiment trains a sparse router from deployable features to
first-order gain targets, then evaluates the resulting switch policy against
exact downstream costs.

The current module path is:

```text
experiments.archive.task_study.gradient_proxy_router
```

## CLI

Show the public commands:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli --help
```

Public commands:

```text
build-router-dataset
sweep
eval
layer-robustness
```

## Dataset

Build the canonical router-ready dataset:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli build-router-dataset \
  --device cuda
```

Default output:

```text
outputs/deployable_routers/gradient_proxy_router/router_dataset.pt
```

Useful overrides:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli build-router-dataset \
  --layer_idx 11 \
  --output_path outputs/task_study/layer11_router_dataset.pt \
  --device cuda
```

The builder also writes deterministic intermediate artifacts beside
`--output_path`:

```text
*.combiner.pt
*.gradient_state.pt
```

## Predictability Analysis

Run the diagnostic with base/current-position features:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.analyze_router_predictability \
  --dataset_paths outputs/deployable_routers/gradient_proxy_router/router_dataset.pt \
  --feature_key features \
  --default_action soft \
  --candidate_actions nondefault \
  --models ridge,mlp \
  --output_path outputs/deployable_routers/gradient_proxy_router/predictability_base_features.json \
  --device cuda
```

Run the diagnostic with the actual deployable router features:

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

For the canonical single-switch setting:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.analyze_router_predictability \
  --dataset_paths outputs/deployable_routers/gradient_proxy_router/router_dataset.pt \
  --feature_key sequence_only_scalar_features_gradient_state \
  --default_action soft \
  --candidate_actions window_soft \
  --group_ids 1 \
  --models ridge,mlp \
  --output_path outputs/deployable_routers/gradient_proxy_router/predictability_group1_window_soft.json \
  --device cuda
```

Supported `--feature_key` values:

```text
features
sequence_only_scalar_features_gradient_state
```

## Train

Train the canonical sparse router sweep:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli sweep \
  --device cuda
```

Default dataset:

```text
outputs/deployable_routers/gradient_proxy_router/router_dataset.pt
```

Default output directory:

```text
outputs/deployable_routers/gradient_proxy_router/focused_sweep
```

To use another dataset or output directory:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli sweep \
  --dataset_path outputs/task_study/layer11_router_dataset.pt \
  --output_dir outputs/deployable_routers/gradient_proxy_router/layer11_sweep \
  --device cuda
```

After training, inspect:

```bash
cat outputs/deployable_routers/gradient_proxy_router/focused_sweep/best_checkpoint.json
```

## Evaluate

Evaluate a checkpoint. The checkpoint stores its dataset path, so
`--dataset_path` is only needed when overriding that path.

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli eval \
  --checkpoint_path outputs/deployable_routers/gradient_proxy_router/focused_sweep/YOUR_CHECKPOINT.pt \
  --device cuda
```

Evaluate and write JSON:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli eval \
  --checkpoint_path outputs/deployable_routers/gradient_proxy_router/focused_sweep/YOUR_CHECKPOINT.pt \
  --output_json outputs/deployable_routers/gradient_proxy_router/focused_sweep/eval_test.json \
  --device cuda
```

You can also call the eval module directly:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.eval \
  --checkpoint_path outputs/deployable_routers/gradient_proxy_router/focused_sweep/YOUR_CHECKPOINT.pt \
  --device cuda
```

## Robustness

Run the archived layer 4 / layer 11 robustness check:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli layer-robustness \
  --layers 4,11 \
  --seeds 0,1,2 \
  --output_root outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11_repro \
  --dataset_device cuda \
  --sweep_device cuda \
  --bootstrap_samples 500 \
  --bootstrap_seed 0
```

If GPU memory is tight, keep dataset construction on GPU and run sweeps on CPU:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.archive.task_study.gradient_proxy_router.cli layer-robustness \
  --layers 4,11 \
  --seeds 0,1,2 \
  --output_root outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11_repro \
  --dataset_device cuda \
  --sweep_device cpu \
  --bootstrap_samples 500 \
  --bootstrap_seed 0
```

Canonical archived result:

```text
outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11_repro/summary.json
```

Summary from that artifact:

- layer 4, 3 seeds: mean gain `0.001773`, positive `3/3`
- layer 4 best seed: mean cost `3.574021`, gain `0.002142`, switch rate `0.0146`
- layer 11, 3 seeds: mean gain `0.000535`, positive `3/3`
- layer 11 best seed: mean cost `3.575526`, gain `0.000637`, switch rate `0.0195`

This is marginal evidence for a small deployable signal, strongest at layer 4.
