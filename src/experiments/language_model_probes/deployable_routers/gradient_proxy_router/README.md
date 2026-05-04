# Gradient Proxy Router

Curated deployable router that trains a sparse action policy from smoother
first-order gain targets, then validates an abstention threshold against exact
costs.

## Result To Cite

Use the layer robustness result as the canonical artifact:

```text
outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11/summary.json
```

Summary:

- layer 4, 3 seeds: mean gain `0.002978`, positive `3/3`
- layer 4 best seed: mean cost `3.697862`, gain `0.004342`, switch rate `0.1465`
- layer 11, 3 seeds: mean gain `-0.000208`, positive `0/3`

This supports a small, layer-dependent deployable signal. It does not support a
claim that routing is solved.

## Core Idea

For each candidate learner action `a`, train a compact predictor for:

```text
first_order_gains_a(x)
```

Optionally mix in a small exact pairwise target:

```text
c_soft(x) - c_a(x)
```

Deployment is sparse:

1. predict candidate gains
2. choose the largest predicted gain
3. select a threshold on validation using a switch-budget grid
4. abstain to `soft` below threshold

## Dataset Pipeline

Build exact-cost learner-combiner rows:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.language_model_probes.deployable_routers.gradient_proxy_router.cli build-combiner \
  --model_name openai-community/gpt2 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split validation \
  --text_field text \
  --max_texts 200 \
  --block_size 96 \
  --batch_size 4 \
  --max_chunks 64 \
  --layer_idx 4 \
  --head_indices all \
  --min_context 16 \
  --position_stride 1 \
  --replace_mode multi_head_single_pos_shared \
  --head_group_size 6 \
  --head_group_strategy contiguous \
  --seed 0 \
  --output_path outputs/learner_combiner/group_size_sweep/combiner_dataset_group6.pt
```

Add first-order gain targets:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.language_model_probes.deployable_routers.gradient_proxy_router.cli add-gradient-state \
  --base_dataset_path outputs/learner_combiner/group_size_sweep/combiner_dataset_group6.pt \
  --output_path outputs/sensitivity_gradient_router/sweep/gradient_state_dataset_group6.pt \
  --device cuda
```

Add compact temporal features:

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.language_model_probes.deployable_routers.gradient_proxy_router.cli add-sequence-features \
  --base_dataset_path outputs/sensitivity_gradient_router/sweep/gradient_state_dataset_group6.pt \
  --output_path outputs/influence_router/group6_sequence_only_scalar_dataset.pt \
  --base_feature_key features_gradient_state \
  --output_feature_key sequence_only_scalar_features_gradient_state \
  --window_size 4 \
  --summary_stats mean,std,max,delta_to_mean \
  --sequence_feature_patterns attn_,topk_recency,pred_norm_,pred_teacher_,pred_cos_,pred_l2_,absolute_position,normalized_position,context_length \
  --include_history_fraction
```

## Robustness Check

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.language_model_probes.deployable_routers.gradient_proxy_router.cli layer-robustness \
  --layers 4,11 \
  --seeds 0,1,2 \
  --output_root outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11 \
  --sweep_device cpu
```

## Evaluate A Regenerated Checkpoint

The committed/kept evidence is the JSON summary above. If the large `.pt`
checkpoint and dataset artifacts are not present locally, rerun the robustness
check first.

```bash
UV_CACHE_DIR=/tmp/uv_cache \
PYTHONPATH=src \
uv run python -m experiments.language_model_probes.deployable_routers.gradient_proxy_router.cli eval \
  --checkpoint_path outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11/layer4/seed0/distill_w0.080_b05.pt \
  --dataset_path outputs/influence_router/group6_sequence_only_scalar_dataset.pt \
  --split test \
  --device cpu
```
