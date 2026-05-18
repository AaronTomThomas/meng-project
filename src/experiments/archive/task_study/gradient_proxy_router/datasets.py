"""Offline dataset generation for learner-combination experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.gpt2_probe_utils import (
    add_shared_cli_args,
    build_candidate_assignments,
    build_head_groups,
    candidate_name,
    continue_from_modified_block_gpt2,
    head_slice,
    load_and_pack_texts,
    mean_next_token_nll,
    parse_head_indices,
    run_to_block_and_cache_tensors,
)
from experiments.language_model_probes.probe_utils import LearnerRegistry


BASE_LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]
LEARNER_REGISTRY = LearnerRegistry(BASE_LEARNERS)
FIRST_ORDER_GAIN_KEY = "first_order_gains"
ROUTER_FEATURE_KEY = "sequence_only_scalar_features_gradient_state"
DEFAULT_ROUTER_DATASET_PATH = "outputs/deployable_routers/gradient_proxy_router/router_dataset.pt"
SEQUENCE_WINDOW_SIZE = 4
SEQUENCE_SUMMARY_STATS = ("mean", "std", "max", "delta_to_mean")
SEQUENCE_FEATURE_PATTERNS = (
    "attn_",
    "topk_recency",
    "pred_norm_",
    "pred_teacher_",
    "pred_cos_",
    "pred_l2_",
    "absolute_position",
    "normalized_position",
    "context_length",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class RouterDatasetBuildConfig:
    beta_soft: float = 6.0
    window_size: int = 16
    k_knn_mean: int = 4
    ridge_lambda: float = 1e-1
    k_linear_local: int = 16
    k_sharp: int = 2
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "validation"
    text_field: str = "text"
    max_texts: int = 200
    block_size: int = 96
    batch_size: int = 4
    max_chunks: int = 64
    layer_idx: int = 4
    head_indices: str = "all"
    min_context: int = 16
    position_stride: int = 1
    replace_mode: str = "multi_head_single_pos_shared"
    head_group_size: int = 6
    head_group_strategy: str = "contiguous"
    manual_head_groups: str = ""
    max_head_groups: int = 0
    seed: int = 0
    cache_dir: str = "outputs/head_counterfactual_cache"
    output_dir: str = "outputs/head_counterfactual_results"
    save_results: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: str = "outputs/learner_combiner/combiner_dataset.pt"
    val_frac: float = 0.2
    test_frac: float = 0.2
    rebuild: bool = False


TOP_K_MASSES = (1, 3, 5)
RECENCY_TOP_K = 4


def split_chunk_ids(num_chunks: int, val_frac: float, test_frac: float, seed: int) -> dict[str, set[int]]:
    chunk_ids = list(range(num_chunks))
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)
    n_val = min(max(int(round(num_chunks * val_frac)), 1), max(num_chunks - 2, 1))
    n_test = min(max(int(round(num_chunks * test_frac)), 1), max(num_chunks - n_val - 1, 1))
    val = set(chunk_ids[:n_val])
    test = set(chunk_ids[n_val : n_val + n_test])
    train = set(chunk_ids[n_val + n_test :])
    if not train:
        train = set(chunk_ids) - val
    return {"train": train, "val": val, "test": test}


def extract_teacher_group_output(
    zcat_teacher: torch.Tensor,
    group: Sequence[int],
    pos: int,
    head_dim: int,
) -> torch.Tensor:
    return torch.cat([zcat_teacher[0, pos, head_slice(head_idx, head_dim)] for head_idx in group], dim=0)


def predict_group_outputs_for_shared_actions(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: Sequence[int],
    pos: int,
    cfg: RouterDatasetBuildConfig,
) -> dict[str, torch.Tensor]:
    pred_map: dict[str, torch.Tensor] = {}
    for learner_name in BASE_LEARNERS:
        parts: list[torch.Tensor] = []
        for head_idx in group:
            pred = LEARNER_REGISTRY.predict(
                learner_name,
                q[:, head_idx, pos, :],
                k[:, head_idx, : pos + 1, :],
                v[:, head_idx, : pos + 1, :],
                cfg,
            )[0]
            parts.append(pred.reshape(-1))
        pred_map[learner_name] = torch.cat(parts, dim=0)
    return pred_map


@torch.no_grad()
def compute_pointwise_action_costs(
    model,
    block,
    input_ids: torch.Tensor,
    x_in: torch.Tensor,
    zcat_teacher: torch.Tensor,
    group: Sequence[int],
    pos: int,
    pred_map: dict[str, torch.Tensor],
    head_dim: int,
    layer_idx: int,
) -> torch.Tensor:
    action_names = list(pred_map)
    zcat_batch = zcat_teacher.repeat(len(action_names), 1, 1)
    x_rep = x_in.repeat(len(action_names), 1, 1)
    ids_rep = input_ids.repeat(len(action_names), 1)
    for action_idx, action_name in enumerate(action_names):
        pred = pred_map[action_name]
        for local_idx, head_idx in enumerate(group):
            start = local_idx * head_dim
            zcat_batch[action_idx, pos, head_slice(head_idx, head_dim)] = pred[start : start + head_dim]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_batch,
        layer_idx=layer_idx,
    )
    return mean_next_token_nll(logits, ids_rep, [pos]).detach()


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    x = x.float().reshape(-1)
    if x.numel() <= 1:
        return x.new_zeros(())
    return x.std(unbiased=False)


def _stats(prefix: str, values: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], list[str]]:
    x = torch.stack([v.float().reshape(()) for v in values])
    return [x.mean(), _safe_std(x), x.max()], [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_max"]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=-1)[0]


@torch.no_grad()
def compute_router_feature_vector(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: Sequence[int],
    pos: int,
    teacher_output: torch.Tensor,
    learner_predictions: dict[str, torch.Tensor],
    learner_names: Sequence[str],
    seq_len: int,
    group_id: int,
) -> tuple[torch.Tensor, list[str]]:
    values: list[torch.Tensor] = []
    names: list[str] = []
    head_dim = int(q.shape[-1])
    eps = 1e-8

    attn_entropy: list[torch.Tensor] = []
    attn_top1: list[torch.Tensor] = []
    attn_gap: list[torch.Tensor] = []
    q_norm: list[torch.Tensor] = []
    k_norm: list[torch.Tensor] = []
    v_norm: list[torch.Tensor] = []
    recency: list[torch.Tensor] = []
    topk_masses = {k_top: [] for k_top in TOP_K_MASSES}

    for head_idx in group:
        q_h = q[0, head_idx, pos, :].float()
        k_ctx = k[0, head_idx, : pos + 1, :].float()
        v_ctx = v[0, head_idx, : pos + 1, :].float()
        scores = torch.matmul(k_ctx, q_h) / math.sqrt(float(head_dim))
        probs = torch.softmax(scores, dim=0)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)

        attn_entropy.append(-(probs * torch.log(probs.clamp_min(eps))).sum())
        attn_top1.append(sorted_probs[0])
        attn_gap.append(sorted_probs[0] - sorted_probs[1] if sorted_probs.numel() > 1 else sorted_probs[0])
        q_norm.append(q_h.norm())
        k_norm.append(k_ctx.norm(dim=-1).mean())
        v_norm.append(v_ctx.norm(dim=-1).mean())

        topk = min(RECENCY_TOP_K, sorted_idx.numel())
        rel_age = (float(pos) - sorted_idx[:topk].float()) / float(max(pos, 1)) if topk else q_h.new_zeros(1)
        recency.append(rel_age.mean())

        for k_top in TOP_K_MASSES:
            top_n = min(int(k_top), sorted_probs.numel())
            topk_masses[k_top].append(sorted_probs[:top_n].sum())

    for prefix, vals in [
        ("attn_entropy", attn_entropy),
        ("attn_top1_mass", attn_top1),
        ("attn_top12_gap", attn_gap),
        ("q_norm", q_norm),
        ("k_ctx_norm", k_norm),
        ("v_ctx_norm", v_norm),
        ("topk_recency", recency),
    ]:
        stat_values, stat_names = _stats(prefix, vals)
        values.extend(stat_values)
        names.extend(stat_names)

    for k_top, vals in topk_masses.items():
        stat_values, stat_names = _stats(f"attn_top{k_top}_mass", vals)
        values.extend(stat_values)
        names.extend(stat_names)

    teacher = teacher_output.float().reshape(-1)
    values.append(teacher.norm())
    names.append("teacher_norm")
    for learner_name in learner_names:
        pred = learner_predictions[learner_name].float().reshape(-1)
        delta = pred - teacher
        values.extend([pred.norm(), delta.norm(), _cosine(pred, teacher), pred.abs().mean(), pred.abs().max()])
        names.extend(
            [
                f"pred_norm_{learner_name}",
                f"pred_teacher_l2_{learner_name}",
                f"pred_teacher_cos_{learner_name}",
                f"pred_abs_mean_{learner_name}",
                f"pred_abs_max_{learner_name}",
            ]
        )

    for left_idx, left_name in enumerate(learner_names):
        left = learner_predictions[left_name].float().reshape(-1)
        for right_name in learner_names[left_idx + 1 :]:
            right = learner_predictions[right_name].float().reshape(-1)
            values.extend([(left - right).norm(), _cosine(left, right)])
            names.extend([f"pred_l2_{left_name}__{right_name}", f"pred_cos_{left_name}__{right_name}"])

    scalar_ref = q.new_tensor
    values.extend(
        [
            scalar_ref(float(pos)),
            scalar_ref(float(pos) / float(max(seq_len - 1, 1))),
            scalar_ref(float(pos + 1)),
            scalar_ref(float(group_id)),
            scalar_ref(float(len(group))),
            scalar_ref(float(group[0])),
            scalar_ref(float(group[-1])),
        ]
    )
    names.extend(
        [
            "absolute_position",
            "normalized_position",
            "context_length",
            "group_id",
            "group_size",
            "group_start_head",
            "group_end_head",
        ]
    )

    return torch.stack([v.float().reshape(()) for v in values]), names


def dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    """Small metadata summary for logs and README examples."""

    split_sizes = {
        split: int(indices.numel())
        for split, indices in dataset["split_indices"].items()
    }
    summary = {
        "num_examples": int(dataset["features"].shape[0]),
        "feature_dim": int(dataset["features"].shape[1]),
        "num_actions": int(dataset["costs"].shape[1]),
        "split_sizes": split_sizes,
        "action_names": list(dataset["action_names"]),
        "layer_idx" : int(dataset["layer_idx"])
    }
    if "teacher_outputs" in dataset:
        summary["prediction_dim"] = int(dataset["teacher_outputs"].shape[1])
    return summary


@torch.no_grad()
def build_combiner_dataset(
    cfg: RouterDatasetBuildConfig,
) -> dict[str, Any]:
    """Build a dataset that preserves learner predictions and teacher outputs."""

    set_seed(cfg.seed)
    ensure_dir(Path(cfg.output_path).parent)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    chunks = load_and_pack_texts(cfg, tokenizer, text_field=cfg.text_field).cpu()
    block_tensors = run_to_block_and_cache_tensors(model, chunks, cfg)

    attn_module = model.transformer.h[cfg.layer_idx].attn
    head_dim = attn_module.head_dim
    n_heads = attn_module.num_heads
    block = model.transformer.h[cfg.layer_idx]

    selected_heads = parse_head_indices(cfg.head_indices, n_heads)
    head_groups = build_head_groups(
        selected_heads=selected_heads,
        group_size=cfg.head_group_size,
        strategy=cfg.head_group_strategy,
        manual_head_groups=cfg.manual_head_groups,
        max_head_groups=cfg.max_head_groups,
        seed=cfg.seed,
    )
    assignments = build_candidate_assignments(
        replace_mode=cfg.replace_mode,
        learner_names=["soft", "sharp", "window_soft", "weighted_linear"],
        group_size=cfg.head_group_size,
    )
    action_names = [assignment[0] for assignment in assignments]
    candidate_names = [candidate_name(assignment) for assignment in assignments]

    seq_len = int(chunks.shape[1])
    positions = list(range(cfg.min_context, seq_len - 1, cfg.position_stride))
    if not positions:
        raise ValueError("No combiner positions available with the current min_context/stride settings.")

    feature_rows: list[torch.Tensor] = []
    cost_rows: list[torch.Tensor] = []
    best_action_rows: list[int] = []
    teacher_rows: list[torch.Tensor] = []
    pred_rows: list[torch.Tensor] = []
    chunk_ids: list[int] = []
    group_ids: list[int] = []
    pos_rows: list[int] = []
    feature_names: list[str] | None = None

    num_chunks = int(chunks.shape[0])
    total_rows = num_chunks * len(head_groups) * len(positions)
    row_counter = 0

    for chunk_id in range(num_chunks):
        input_ids = chunks[chunk_id : chunk_id + 1].to(cfg.device)
        x_in = block_tensors["x_in"][chunk_id : chunk_id + 1].to(cfg.device)
        q = block_tensors["q"][chunk_id : chunk_id + 1].to(cfg.device)
        k = block_tensors["k"][chunk_id : chunk_id + 1].to(cfg.device)
        v = block_tensors["v"][chunk_id : chunk_id + 1].to(cfg.device)
        zcat_teacher = block_tensors["zcat_teacher"][chunk_id : chunk_id + 1].to(cfg.device)

        for group_id, group in enumerate(head_groups):
            for pos in positions:
                teacher_output = extract_teacher_group_output(
                    zcat_teacher=zcat_teacher,
                    group=group,
                    pos=pos,
                    head_dim=head_dim,
                )
                pred_map = predict_group_outputs_for_shared_actions(
                    q=q,
                    k=k,
                    v=v,
                    group=group,
                    pos=pos,
                    cfg=cfg,
                )
                feature_vector, current_feature_names = compute_router_feature_vector(
                    q=q,
                    k=k,
                    v=v,
                    group=group,
                    pos=pos,
                    teacher_output=teacher_output,
                    learner_predictions=pred_map,
                    learner_names=action_names,
                    seq_len=seq_len,
                    group_id=group_id,
                )
                costs = compute_pointwise_action_costs(
                    model=model,
                    block=block,
                    input_ids=input_ids,
                    x_in=x_in,
                    zcat_teacher=zcat_teacher,
                    group=group,
                    pos=pos,
                    pred_map=pred_map,
                    head_dim=head_dim,
                    layer_idx=cfg.layer_idx,
                )

                if feature_names is None:
                    feature_names = current_feature_names

                learner_predictions = torch.stack(
                    [pred_map[action_name].float().reshape(-1) for action_name in action_names],
                    dim=0,
                )
                feature_rows.append(feature_vector.cpu())
                teacher_rows.append(teacher_output.float().cpu())
                pred_rows.append(learner_predictions.cpu())
                cost_rows.append(costs.float().cpu())
                best_action_rows.append(int(costs.argmin().item()))
                chunk_ids.append(chunk_id)
                group_ids.append(group_id)
                pos_rows.append(pos)
                row_counter += 1

        if chunk_id % 4 == 0 or chunk_id == num_chunks - 1:
            print(f"[build] processed chunk {chunk_id + 1}/{num_chunks} rows={row_counter}/{total_rows}")

    if feature_names is None:
        raise RuntimeError("Feature extraction produced no rows.")

    split_chunks = split_chunk_ids(
        num_chunks=num_chunks,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    split_indices = {
        split: torch.nonzero(
            torch.tensor([chunk_id in split_chunk_ids_set for chunk_id in chunk_ids], dtype=torch.bool),
            as_tuple=False,
        ).squeeze(-1).long()
        for split, split_chunk_ids_set in split_chunks.items()
    }

    dataset = {
        "features": torch.stack(feature_rows, dim=0),
        "teacher_outputs": torch.stack(teacher_rows, dim=0),
        "learner_predictions": torch.stack(pred_rows, dim=0),
        "costs": torch.stack(cost_rows, dim=0),
        "best_action": torch.tensor(best_action_rows, dtype=torch.long),
        "chunk_ids": torch.tensor(chunk_ids, dtype=torch.long),
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "positions": torch.tensor(pos_rows, dtype=torch.long),
        "feature_names": feature_names,
        "action_names": action_names,
        "layer_idx" : cfg.layer_idx,
        "candidate_names": candidate_names,
        "split_indices": split_indices,
        "metadata": {
            "config": asdict(cfg),
            "feature_config": {"top_k_masses": list(TOP_K_MASSES), "recency_top_k": RECENCY_TOP_K},
            "head_groups": [list(group) for group in head_groups],
            "assignments": [list(assignment) for assignment in assignments],
            "evaluation_scope": "pointwise_local_next_token_nll",
            "format_version": 1,
        },
    }
    return dataset


def build_or_load_combiner_dataset(
    cfg: RouterDatasetBuildConfig,
) -> dict[str, Any]:
    """Load a cached dataset unless `--rebuild` forces regeneration."""

    dataset_path = Path(cfg.output_path)
    if dataset_path.exists() and not cfg.rebuild:
        print(f"[cache] loading combiner dataset from {dataset_path}")
        return torch.load(dataset_path, map_location="cpu")
    dataset = build_combiner_dataset(cfg)
    torch.save(dataset, dataset_path)
    print(f"[cache] saved combiner dataset to {dataset_path}")
    return dataset


def build_combiner_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI for offline combiner dataset generation."""

    parser = build_dataset_pipeline_arg_parser(
        description="Build a cached learner-combiner dataset for GPT-2 probes.",
        default_output_path="outputs/learner_combiner/combiner_dataset.pt",
    )
    return parser


def remove_parser_options(parser: argparse.ArgumentParser, option_names: Sequence[str]) -> None:
    """Remove shared parser options that are not part of this public workflow."""

    for option_name in option_names:
        action = parser._option_string_actions.pop(option_name, None)
        if action is None:
            continue
        parser._actions = [existing for existing in parser._actions if existing is not action]
        for group in parser._action_groups:
            group._group_actions = [
                existing for existing in group._group_actions if existing is not action
            ]
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)


def build_dataset_pipeline_arg_parser(description: str, default_output_path: str) -> argparse.ArgumentParser:
    """Construct CLI args shared by the high-level dataset pipeline commands."""

    parser = argparse.ArgumentParser(description=description)
    add_shared_cli_args(
        parser,
        include_text_field=True,
        replace_mode_choices=["multi_head_single_pos_shared"],
        default_replace_mode="multi_head_single_pos_shared",
        default_position_stride=1,
        head_group_strategy_choices=["contiguous", "random", "manual"],
    )
    parser.set_defaults(
        head_group_size=6,
        replace_mode="multi_head_single_pos_shared",
        output_dir="outputs/head_counterfactual_results",
        save_results=False,
    )
    remove_parser_options(parser, ["--replace_mode", "--output_dir", "--save_results"])
    parser.add_argument("--output_path", type=str, default=default_output_path)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def parse_combiner_args(argv: Sequence[str] | None = None) -> RouterDatasetBuildConfig:
    """Parse CLI args into typed config objects."""

    args = build_combiner_arg_parser().parse_args(argv)
    return RouterDatasetBuildConfig(**vars(args))


def combiner_main(argv: Sequence[str] | None = None) -> None:
    """Build exact-cost learner-combiner rows."""

    cfg = parse_combiner_args(argv)
    dataset = build_or_load_combiner_dataset(cfg)
    print(json.dumps(dataset_summary(dataset), indent=2))


def derived_dataset_path(output_path: str | Path, suffix: str) -> Path:
    """Return a deterministic sidecar path beside a final dataset artifact."""

    output_path = Path(output_path)
    return output_path.with_suffix(f".{suffix}.pt")


def build_or_load_final_dataset(
    output_path: str | Path,
    rebuild: bool,
    required_keys: Sequence[str],
) -> dict[str, Any] | None:
    """Load a final pipeline artifact when caching is allowed."""

    output_path = Path(output_path)
    if output_path.exists() and not rebuild:
        print(f"[cache] loading final dataset from {output_path}")
        dataset = torch.load(output_path, map_location="cpu")
        missing = [key for key in required_keys if key not in dataset]
        if missing:
            raise KeyError(
                f"Cached dataset at {output_path} is missing required keys {missing}. "
                "Pass --rebuild or choose a different --output_path."
            )
        return dataset
    return None


def print_pipeline_summary(dataset: dict[str, Any], output_path: str | Path) -> None:
    """Print a compact summary for a high-level dataset build."""

    summary = dataset_summary(dataset)
    summary["output_path"] = str(output_path)
    if FIRST_ORDER_GAIN_KEY in dataset:
        summary["first_order_gain_dim"] = int(dataset[FIRST_ORDER_GAIN_KEY].shape[1])
    if ROUTER_FEATURE_KEY in dataset:
        summary["router_feature_key"] = ROUTER_FEATURE_KEY
        summary["router_feature_dim"] = int(dataset[ROUTER_FEATURE_KEY].shape[1])
    print(json.dumps(summary, indent=2))


def build_diagnostic_dataset(
    cfg: RouterDatasetBuildConfig,
) -> dict[str, Any]:
    """Build a two-stage dataset for layer-signal diagnostics."""

    final_path = Path(cfg.output_path)
    cached = build_or_load_final_dataset(
        final_path,
        rebuild=cfg.rebuild,
        required_keys=("features", "costs", FIRST_ORDER_GAIN_KEY),
    )
    if cached is not None:
        return cached

    combiner_path = derived_dataset_path(final_path, "combiner")
    combiner_cfg = RouterDatasetBuildConfig(**{**asdict(cfg), "output_path": str(combiner_path)})
    build_or_load_combiner_dataset(combiner_cfg)
    return add_gradient_state_dataset(
        base_dataset_path=combiner_path,
        output_path=final_path,
        device=cfg.device,
    )


def build_router_dataset(
    cfg: RouterDatasetBuildConfig,
) -> dict[str, Any]:
    """Build the full canonical gradient-proxy router dataset."""

    final_path = Path(cfg.output_path)
    cached = build_or_load_final_dataset(
        final_path,
        rebuild=cfg.rebuild,
        required_keys=("costs", FIRST_ORDER_GAIN_KEY, ROUTER_FEATURE_KEY),
    )
    if cached is not None:
        return cached

    combiner_path = derived_dataset_path(final_path, "combiner")
    gradient_path = derived_dataset_path(final_path, "gradient_state")
    combiner_cfg = RouterDatasetBuildConfig(**{**asdict(cfg), "output_path": str(combiner_path)})
    build_or_load_combiner_dataset(combiner_cfg)
    add_gradient_state_dataset(
        base_dataset_path=combiner_path,
        output_path=gradient_path,
        device=cfg.device,
    )
    return add_sequence_state_features(
        base_dataset_path=gradient_path,
        output_path=final_path,
        seed=cfg.seed,
        final_router_artifact=True,
    )


def parse_pipeline_args(
    argv: Sequence[str] | None,
    description: str,
    default_output_path: str,
) -> RouterDatasetBuildConfig:
    args = build_dataset_pipeline_arg_parser(
        description=description,
        default_output_path=default_output_path,
    ).parse_args(argv)
    return RouterDatasetBuildConfig(**vars(args))


def router_dataset_main(argv: Sequence[str] | None = None) -> None:
    """Build the full router-ready dataset in one command."""

    cfg = parse_pipeline_args(
        argv,
        description="Build the full canonical gradient-proxy router dataset.",
        default_output_path=DEFAULT_ROUTER_DATASET_PATH,
    )
    dataset = build_router_dataset(cfg)
    print_pipeline_summary(dataset, cfg.output_path)


def diagnostic_dataset_main(argv: Sequence[str] | None = None) -> None:
    """Build the two-stage diagnostic dataset in one command."""

    cfg = parse_pipeline_args(
        argv,
        description="Build a gradient-proxy diagnostic dataset with base features and first-order gains.",
        default_output_path="outputs/task_study/diagnostic_dataset.pt",
    )
    dataset = build_diagnostic_dataset(cfg)
    print_pipeline_summary(dataset, cfg.output_path)



# Gradient-state dataset augmentation.
def build_gradient_state_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add first-order gains to a deployable combiner dataset.")
    parser.add_argument("--base_dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def token_nll_vector(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )
    return losses.view(shift_labels.shape[0], shift_labels.shape[1])


def continue_from_modified_block_gpt2_with_grad(
    model,
    block,
    x_in: torch.Tensor,
    zcat_mod: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    attn_out = block.attn.c_proj(zcat_mod)
    attn_out = block.attn.resid_dropout(attn_out)
    x = x_in + attn_out
    mlp_out = block.mlp(block.ln_2(x))
    x = x + mlp_out
    for next_block in model.transformer.h[layer_idx + 1 :]:
        out = next_block(x, use_cache=False)
        x = out[0] if isinstance(out, tuple) else out
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def compute_position_gradients(
    model,
    block,
    x_in: torch.Tensor,
    input_ids: torch.Tensor,
    zcat_teacher: torch.Tensor,
    positions: Sequence[int],
    layer_idx: int,
) -> dict[int, torch.Tensor]:
    with torch.enable_grad():
        z_var = zcat_teacher.clone().detach().requires_grad_(True)
        logits = continue_from_modified_block_gpt2_with_grad(
            model=model,
            block=block,
            x_in=x_in,
            zcat_mod=z_var,
            layer_idx=layer_idx,
        )
        per_token_nll = token_nll_vector(logits, input_ids)[0]
        grads: dict[int, torch.Tensor] = {}
        for grad_idx, pos in enumerate(positions):
            grad = torch.autograd.grad(
                per_token_nll[pos],
                z_var,
                retain_graph=grad_idx < len(positions) - 1,
                create_graph=False,
                allow_unused=False,
            )[0]
            grads[int(pos)] = grad[0, pos, :].detach()
    return grads


def add_gradient_state_dataset(
    base_dataset_path: str | Path,
    output_path: str | Path,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    base_dataset_path = Path(base_dataset_path)
    output_path = Path(output_path)
    base = torch.load(base_dataset_path, map_location="cpu")
    metadata = dict(base["metadata"])
    cfg_dict = dict(metadata["config"])
    layer_idx = int(cfg_dict["layer_idx"])
    head_groups = [list(group) for group in metadata["head_groups"]]

    print("[setup] loading tokenizer/model")
    tokenizer = AutoTokenizer.from_pretrained(cfg_dict["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg_dict["model_name"]).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    cfg_obj = argparse.Namespace(**cfg_dict)
    chunks = load_and_pack_texts(cfg_obj, tokenizer, text_field=cfg_dict.get("text_field", "text")).to(device)
    block_tensors = run_to_block_and_cache_tensors(model, chunks, cfg_obj)
    block = model.transformer.h[layer_idx]
    head_dim = int(block.attn.head_dim)

    num_rows = int(base["costs"].shape[0])
    num_actions = int(base["costs"].shape[1])
    first_order_gains = torch.empty(num_rows, num_actions, dtype=torch.float32)
    gradient_norms = torch.empty(num_rows, dtype=torch.float32)
    linearized_costs = torch.empty_like(first_order_gains)
    base_costs = base["costs"][:, 0].float()

    row_by_chunk: dict[int, list[int]] = {}
    for row_idx, chunk_id in enumerate(base["chunk_ids"].tolist()):
        row_by_chunk.setdefault(int(chunk_id), []).append(row_idx)

    for chunk_counter, (chunk_id, row_indices) in enumerate(sorted(row_by_chunk.items()), start=1):
        input_ids = chunks[chunk_id : chunk_id + 1]
        x_in = block_tensors["x_in"][chunk_id : chunk_id + 1].to(device)
        zcat_teacher = block_tensors["zcat_teacher"][chunk_id : chunk_id + 1].to(device)
        positions = sorted({int(base["positions"][row].item()) for row in row_indices})
        grad_by_pos = compute_position_gradients(
            model=model,
            block=block,
            x_in=x_in,
            input_ids=input_ids,
            zcat_teacher=zcat_teacher,
            positions=positions,
            layer_idx=layer_idx,
        )

        for row in row_indices:
            group = head_groups[int(base["group_ids"][row].item())]
            pos = int(base["positions"][row].item())
            grad_full = grad_by_pos[pos]
            grad_group = torch.cat([grad_full[head_slice(head_idx, head_dim)] for head_idx in group], dim=0).cpu()
            teacher_output = base["teacher_outputs"][row].float()
            predictions = base["learner_predictions"][row].float()
            deltas = predictions - teacher_output.unsqueeze(0)
            gains = -(deltas * grad_group.unsqueeze(0)).sum(dim=1)
            first_order_gains[row] = gains
            gradient_norms[row] = grad_group.norm()
            linearized_costs[row] = base_costs[row] - gains

        if chunk_counter % 4 == 0 or chunk_counter == len(row_by_chunk):
            print(f"[build] processed chunk {chunk_counter}/{len(row_by_chunk)} rows={len(row_indices)}")

    out = dict(base)
    out["first_order_gains"] = first_order_gains
    out["gradient_norms"] = gradient_norms
    out["linearized_costs"] = linearized_costs
    metadata["gradient_state_format_version"] = 1
    metadata["gradient_state_source"] = "deployable_gradient_proxy_gradient_state_dataset"
    out["metadata"] = metadata

    ensure_dir(output_path.parent)
    torch.save(out, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "base_dataset_path": str(base_dataset_path),
                "num_examples": num_rows,
                "num_actions": num_actions,
                "layer_idx": layer_idx,
                "target": FIRST_ORDER_GAIN_KEY,
            },
            indent=2,
        )
    )
    return out


def gradient_state_main(argv: list[str] | None = None) -> None:
    args = build_gradient_state_arg_parser().parse_args(argv)
    add_gradient_state_dataset(
        base_dataset_path=args.base_dataset_path,
        output_path=args.output_path,
        device=args.device,
    )


# Sequence-state feature augmentation.
def build_sequence_state_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add short-window sequence-state features to a routing dataset.")
    parser.add_argument("--base_dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_feature_names(dataset: dict, feature_key: str, feature_dim: int) -> list[str]:
    explicit_key = feature_key.replace("features", "feature_names", 1)
    if explicit_key in dataset:
        names = list(dataset[explicit_key])
    elif feature_key == "features" and "feature_names" in dataset:
        names = list(dataset["feature_names"])
    else:
        names = [f"{feature_key}_{idx}" for idx in range(feature_dim)]
    if len(names) != feature_dim:
        raise ValueError(f"Feature name count mismatch for '{feature_key}': {len(names)} != {feature_dim}")
    return names


def _window_summary_rows(
    seq_features: torch.Tensor,
) -> torch.Tensor:
    seq_features = seq_features.float()
    seq_len, feature_dim = seq_features.shape

    prev_mean = torch.zeros(seq_len, feature_dim, dtype=torch.float32)
    prev_std = torch.zeros(seq_len, feature_dim, dtype=torch.float32)
    prev_max = torch.zeros(seq_len, feature_dim, dtype=torch.float32)
    prefix_sum = torch.cat([torch.zeros(1, feature_dim, dtype=torch.float32), seq_features.cumsum(dim=0)], dim=0)
    prefix_sq = torch.cat([torch.zeros(1, feature_dim, dtype=torch.float32), seq_features.square().cumsum(dim=0)], dim=0)

    for idx in range(seq_len):
        start = max(0, idx - SEQUENCE_WINDOW_SIZE)
        count = idx - start
        if count <= 0:
            continue
        mean = (prefix_sum[idx] - prefix_sum[start]) / float(count)
        mean_sq = (prefix_sq[idx] - prefix_sq[start]) / float(count)
        prev_mean[idx] = mean
        prev_std[idx] = (mean_sq - mean.square()).clamp_min(0.0).sqrt()
        prev_max[idx] = seq_features[start:idx].max(dim=0).values

    history_fraction = torch.tensor(
        [min(idx, SEQUENCE_WINDOW_SIZE) / float(SEQUENCE_WINDOW_SIZE) for idx in range(seq_len)],
        dtype=torch.float32,
    )
    return torch.cat([prev_mean, prev_std, prev_max, seq_features - prev_mean, history_fraction.unsqueeze(-1)], dim=-1)


def _sequence_feature_names(base_feature_names: list[str]) -> list[str]:
    return (
        [f"prev_mean__{name}" for name in base_feature_names]
        + [f"prev_std__{name}" for name in base_feature_names]
        + [f"prev_max__{name}" for name in base_feature_names]
        + [f"curr_minus_prev_mean__{name}" for name in base_feature_names]
        + ["prev_history_fraction"]
    )


def _selected_sequence_feature_indices(
    base_feature_names: list[str],
) -> list[int]:
    indices = [
        idx
        for idx, name in enumerate(base_feature_names)
        if any(pattern in name for pattern in SEQUENCE_FEATURE_PATTERNS)
    ]
    if not indices:
        raise ValueError("No base features matched the requested sequence-feature selection.")
    return indices


def add_sequence_state_features(
    base_dataset_path: str | Path,
    output_path: str | Path,
    seed: int = 0,
    final_router_artifact: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    base_dataset_path = Path(base_dataset_path)
    output_path = Path(output_path)
    dataset = torch.load(base_dataset_path, map_location="cpu")
    base_feature_key = "features"
    if base_feature_key not in dataset:
        raise KeyError(f"Unknown base feature key '{base_feature_key}'")

    base_features = dataset[base_feature_key]
    base_feature_names = _resolve_feature_names(dataset, base_feature_key, int(base_features.shape[1]))
    sequence_feature_indices = _selected_sequence_feature_indices(
        base_feature_names=base_feature_names,
    )
    selected_feature_names = [base_feature_names[idx] for idx in sequence_feature_indices]
    seq_feature_names = _sequence_feature_names(selected_feature_names)
    selected_base_features = base_features[:, torch.tensor(sequence_feature_indices, dtype=torch.long)]

    chunk_ids = dataset["chunk_ids"].long()
    group_ids = dataset["group_ids"].long()
    positions = dataset["positions"].long()
    order = sorted(
        range(base_features.shape[0]),
        key=lambda idx: (
            int(chunk_ids[idx].item()),
            int(group_ids[idx].item()),
            int(positions[idx].item()),
        ),
    )

    sequence_blocks = torch.zeros(
        base_features.shape[0],
        len(seq_feature_names),
        dtype=torch.float16 if base_features.dtype == torch.float16 else torch.float32,
    )

    def flush_sequence(rows: list[int]) -> None:
        if not rows:
            return
        row_tensor = torch.tensor(rows, dtype=torch.long)
        sequence_blocks[row_tensor] = _window_summary_rows(
            selected_base_features[row_tensor],
        ).to(sequence_blocks.dtype)

    seq_rows: list[int] = []
    seq_key: tuple[int, int] | None = None
    for row_idx in order:
        key = (int(chunk_ids[row_idx].item()), int(group_ids[row_idx].item()))
        if seq_key is None:
            seq_key = key
        if key != seq_key:
            flush_sequence(seq_rows)
            seq_rows = []
            seq_key = key
        seq_rows.append(row_idx)

    flush_sequence(seq_rows)

    metadata = dict(dataset["metadata"])
    metadata["sequence_state_feature_format_version"] = 1
    metadata["sequence_state_feature_config"] = {
        "base_feature_key": base_feature_key,
        "window_size": SEQUENCE_WINDOW_SIZE,
        "summary_stats": list(SEQUENCE_SUMMARY_STATS),
        "sequence_feature_patterns": list(SEQUENCE_FEATURE_PATTERNS),
        "num_sequence_base_features": int(len(selected_feature_names)),
        "include_history_fraction": True,
    }
    router_features = sequence_blocks.float().to(base_features.dtype)

    if final_router_artifact:
        out = {
            "features": dataset["features"],
            "feature_names": dataset["feature_names"],
            ROUTER_FEATURE_KEY: router_features,
            "sequence_state_feature_names": seq_feature_names,
            "sequence_state_selected_base_feature_names": selected_feature_names,
            "costs": dataset["costs"],
            FIRST_ORDER_GAIN_KEY: dataset[FIRST_ORDER_GAIN_KEY],
            "best_action": dataset["best_action"],
            "chunk_ids": dataset["chunk_ids"],
            "group_ids": dataset["group_ids"],
            "positions": dataset["positions"],
            "action_names": dataset["action_names"],
            "candidate_names": dataset["candidate_names"],
            "split_indices": dataset["split_indices"],
            "layer_idx": dataset["layer_idx"],
            "metadata": metadata,
        }
    else:
        out = dict(dataset)
        out["sequence_state_feature_names"] = seq_feature_names
        out["sequence_state_selected_base_feature_names"] = selected_feature_names
        out[ROUTER_FEATURE_KEY] = router_features
        out["metadata"] = metadata

    ensure_dir(output_path.parent)
    torch.save(out, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "base_dataset_path": str(base_dataset_path),
                "base_feature_key": base_feature_key,
                "output_feature_key": ROUTER_FEATURE_KEY,
                "window_size": SEQUENCE_WINDOW_SIZE,
                "summary_stats": list(SEQUENCE_SUMMARY_STATS),
                "sequence_feature_patterns": list(SEQUENCE_FEATURE_PATTERNS),
                "num_sequence_base_features": int(len(selected_feature_names)),
                "include_history_fraction": True,
                "input_dim": int(base_features.shape[1]),
                "augmented_dim": int(out[ROUTER_FEATURE_KEY].shape[1]),
            },
            indent=2,
        )
    )
    return out


def sequence_state_main(argv: list[str] | None = None) -> None:
    args = build_sequence_state_arg_parser().parse_args(argv)
    add_sequence_state_features(
        base_dataset_path=args.base_dataset_path,
        output_path=args.output_path,
        seed=args.seed,
    )
