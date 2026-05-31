from __future__ import annotations

from pathlib import Path

import torch

from experiments.archive.task_study.gradient_proxy_router import cli
from experiments.archive.task_study.gradient_proxy_router import datasets


def test_public_dataset_commands_are_registered() -> None:
    assert "build-router-dataset" in cli.COMMANDS
    assert "sweep" in cli.COMMANDS
    assert "eval" in cli.COMMANDS
    assert "layer-robustness" in cli.COMMANDS
    assert "build-diagnostic-dataset" not in cli.COMMANDS
    assert "build-combiner" not in cli.COMMANDS
    assert "add-gradient-state" not in cli.COMMANDS
    assert "add-sequence-features" not in cli.COMMANDS


def test_build_router_dataset_runs_all_stages(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []
    output_path = tmp_path / "router_dataset.pt"

    def fake_combiner(cfg: datasets.RouterDatasetBuildConfig):
        calls.append(("combiner", cfg.output_path, ""))
        return {}

    def fake_gradient(base_dataset_path, output_path, device):
        calls.append(("gradient", str(base_dataset_path), str(output_path)))
        return {}

    def fake_sequence(base_dataset_path, output_path, seed, final_router_artifact=False):
        calls.append(("sequence", str(base_dataset_path), str(output_path)))
        assert final_router_artifact is True
        return {"ok": True}

    monkeypatch.setattr(datasets, "build_or_load_combiner_dataset", fake_combiner)
    monkeypatch.setattr(datasets, "add_gradient_state_dataset", fake_gradient)
    monkeypatch.setattr(datasets, "add_sequence_state_features", fake_sequence)

    cfg = datasets.RouterDatasetBuildConfig(output_path=str(output_path), device="cpu", rebuild=False)
    assert datasets.build_router_dataset(cfg) == {"ok": True}

    assert calls == [
        ("combiner", str(tmp_path / "router_dataset.combiner.pt"), ""),
        (
            "gradient",
            str(tmp_path / "router_dataset.combiner.pt"),
            str(tmp_path / "router_dataset.gradient_state.pt"),
        ),
        (
            "sequence",
            str(tmp_path / "router_dataset.gradient_state.pt"),
            str(output_path),
        ),
    ]


def test_build_diagnostic_dataset_runs_two_stages(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []
    output_path = tmp_path / "diagnostic_dataset.pt"

    def fake_combiner(cfg: datasets.RouterDatasetBuildConfig):
        calls.append(("combiner", cfg.output_path, ""))
        return {}

    def fake_gradient(base_dataset_path, output_path, device):
        calls.append(("gradient", str(base_dataset_path), str(output_path)))
        return {"ok": True}

    monkeypatch.setattr(datasets, "build_or_load_combiner_dataset", fake_combiner)
    monkeypatch.setattr(datasets, "add_gradient_state_dataset", fake_gradient)

    cfg = datasets.RouterDatasetBuildConfig(output_path=str(output_path), device="cpu", rebuild=False)
    assert datasets.build_diagnostic_dataset(cfg) == {"ok": True}

    assert calls == [
        ("combiner", str(tmp_path / "diagnostic_dataset.combiner.pt"), ""),
        (
            "gradient",
            str(tmp_path / "diagnostic_dataset.combiner.pt"),
            str(output_path),
        ),
    ]


def test_add_sequence_state_final_router_artifact_keeps_only_runtime_and_analysis_keys(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "gradient_state.pt"
    output_path = tmp_path / "router_dataset.pt"
    dataset = {
        "features": torch.randn(3, 2),
        "feature_names": ["attn_alpha", "pred_norm_x"],
        "costs": torch.randn(3, 2),
        datasets.FIRST_ORDER_GAIN_KEY: torch.randn(3, 2),
        "best_action": torch.tensor([0, 1, 0]),
        "chunk_ids": torch.tensor([0, 0, 0]),
        "group_ids": torch.tensor([0, 0, 0]),
        "positions": torch.tensor([1, 2, 3]),
        "action_names": ["soft", "window_soft"],
        "candidate_names": ["soft", "window_soft"],
        "split_indices": {"train": torch.tensor([0]), "val": torch.tensor([1]), "test": torch.tensor([2])},
        "layer_idx": 4,
        "metadata": {},
        "teacher_outputs": torch.randn(3, 4),
        "learner_predictions": torch.randn(3, 2, 4),
        "linearized_costs": torch.randn(3, 2),
        "gradient_norms": torch.randn(3),
    }
    torch.save(dataset, base_path)

    monkeypatch.setattr(datasets, "_selected_sequence_feature_indices", lambda base_feature_names: [0, 1])

    out = datasets.add_sequence_state_features(
        base_dataset_path=base_path,
        output_path=output_path,
        seed=0,
        final_router_artifact=True,
    )

    assert set(out) == {
        "features",
        "feature_names",
        datasets.ROUTER_FEATURE_KEY,
        "sequence_state_feature_names",
        "sequence_state_selected_base_feature_names",
        "costs",
        datasets.FIRST_ORDER_GAIN_KEY,
        "best_action",
        "chunk_ids",
        "group_ids",
        "positions",
        "action_names",
        "candidate_names",
        "split_indices",
        "layer_idx",
        "metadata",
    }
    assert set(torch.load(output_path, map_location="cpu", weights_only=False)) == set(out)
