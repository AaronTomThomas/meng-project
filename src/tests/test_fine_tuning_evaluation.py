from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
import pytest

from experiments.router_development.attention_adapter.fine_tuning_evaluation.config import FineTuneEvalConfig
from experiments.router_development.attention_adapter.fine_tuning_evaluation.compare import aggregate_rows, comparison_rows, validate_fairness
from experiments.router_development.attention_adapter.fine_tuning_evaluation.datasets import LoadedTaskData
from experiments.router_development.attention_adapter.fine_tuning_evaluation.evaluate import (
    _GlueVerbalizerCollator,
    _encode_prompt_target,
    score_candidates,
)
from experiments.router_development.attention_adapter.fine_tuning_evaluation import evaluate as evaluate_module
from experiments.router_development.attention_adapter.fine_tuning_evaluation import datasets as datasets_module
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import TASKS, format_example
from experiments.router_development.attention_adapter.fine_tuning_evaluation import train as train_module
from experiments.router_development.attention_adapter.fine_tuning_evaluation import cli as cli_module


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.inverse: dict[int, str] = {0: "<pad>", 1: "<eos>"}

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> SimpleNamespace:
        del add_special_tokens
        tokens = text.strip().split() or [text]
        ids = [self._id(token) for token in tokens]
        if truncation and max_length is not None:
            ids = ids[-max_length:]
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for idx in ids:
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(self.inverse[idx])
        return " ".join(pieces)

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse[idx] = token
        return self.vocab[token]


class SpyLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, next_token_id: int = 2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.next_token_id = next_token_id
        self.seen: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.seen.append(input_ids.detach().cpu().clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size, device=input_ids.device)
        logits[..., self.next_token_id] = 10.0
        return logits


class TinyTrainModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.proj = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


@dataclass
class ListDataset:
    rows: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return [row[idx] for row in self.rows]
        return self.rows[idx]

    def select(self, indices):
        return ListDataset([self.rows[idx] for idx in indices])

    def shuffle(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.rows), generator=generator).tolist()
        return self.select(order)

    @property
    def column_names(self) -> list[str]:
        names: set[str] = set()
        for row in self.rows:
            names.update(row)
        return sorted(names)


class RawSplits(dict):
    pass


def split_details() -> dict[str, dict[str, object]]:
    return {
        "train": {
            "source_split": "train",
            "num_examples": 1,
            "has_labels": True,
        },
        "validation": {
            "source_split": "validation",
            "num_examples": 1,
            "has_labels": True,
        },
        "test": {
            "source_split": "test",
            "num_examples": 1,
            "has_labels": False,
            "source_kind": "official",
        },
    }


def test_format_example_keeps_tasks_registry_separate() -> None:
    row = format_example(TASKS["sst2"], {"sentence": "sharp and funny", "label": 1})

    assert row == {"prompt": "Sentence: sharp and funny\nSentiment:", "target": " positive"}


def test_rte_format_example_uses_premise_hypothesis_prompt() -> None:
    row = format_example(
        TASKS["rte"],
        {
            "sentence1": "The cat sat on the mat.",
            "sentence2": "An animal sat down.",
            "label": "entailment",
        },
    )

    assert row == {
        "prompt": "Premise: The cat sat on the mat.\nHypothesis: An animal sat down.\nEntailment:",
        "target": " entailment",
    }


def test_sst2_collator_masks_prompt_without_training_on_eos() -> None:
    tokenizer = TinyTokenizer()
    batch = _GlueVerbalizerCollator(tokenizer, TASKS["sst2"], max_length=8, target_max_length=2)(
        [{"sentence": "a compact movie", "label": 1}]
    )

    labels = batch["labels"][0].tolist()
    assert tokenizer.eos_token_id not in [label for label in labels if label != -100]
    assert labels.count(-100) > 0


def test_collator_can_train_on_eos_when_task_requests_it() -> None:
    tokenizer = TinyTokenizer()
    task = replace(TASKS["sst2"], add_eos_to_target=True)
    batch = _GlueVerbalizerCollator(tokenizer, task, max_length=8, target_max_length=2)(
        [{"sentence": "a compact movie", "label": 1}]
    )

    labels = batch["labels"][0].tolist()
    assert labels[-1] == tokenizer.eos_token_id


def test_shared_encoder_reserves_room_for_target_on_long_prompts() -> None:
    tokenizer = TinyTokenizer()

    encoded = _encode_prompt_target(
        tokenizer,
        "one two three four five six",
        " positive",
        max_length=5,
        target_max_length=2,
        add_eos=False,
    )

    assert len(encoded["input_ids"]) == 5
    assert encoded["labels"].count(-100) == 4
    assert encoded["labels"][-1] != -100
    assert tokenizer.eos_token_id not in encoded["labels"]


def test_candidate_scoring_uses_task_eos_policy() -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel(next_token_id=tokenizer.eos_token_id)

    metrics, rows = score_candidates(
        model,
        [{"sentence": "plain", "label": 1}],
        tokenizer,
        TASKS["sst2"],
        max_length=16,
        target_max_length=2,
        device=torch.device("cpu"),
    )

    assert metrics["accuracy"] in {0.0, 1.0}
    assert rows[0]["reference"] == "positive"
    assert all(tokenizer.eos_token_id not in seen[0].tolist() for seen in model.seen)


def test_candidate_scoring_can_include_eos_when_task_requests_it() -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel(next_token_id=tokenizer.eos_token_id)
    task = replace(TASKS["sst2"], add_eos_to_target=True)

    score_candidates(
        model,
        [{"sentence": "plain", "label": 1}],
        tokenizer,
        task,
        max_length=16,
        target_max_length=2,
        device=torch.device("cpu"),
    )

    assert all(tokenizer.eos_token_id in seen[0].tolist() for seen in model.seen)


def test_candidate_scoring_rejects_unknown_score_normalization() -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel()
    task = replace(TASKS["sst2"], score_normalization="unsupported")

    with pytest.raises(ValueError, match="score_normalization"):
        score_candidates(
            model,
            [{"sentence": "plain", "label": 1}],
            tokenizer,
            task,
            max_length=16,
            target_max_length=2,
            device=torch.device("cpu"),
        )


def test_candidate_scoring_supports_sum_logprob(monkeypatch) -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel()
    task = replace(TASKS["sst2"], score_normalization="sum_logprob")
    monkeypatch.setattr(evaluate_module, "masked_lm_loss", lambda logits, labels: (torch.tensor(6.0), 3))

    _, rows = score_candidates(
        model,
        [{"sentence": "plain", "label": 1}],
        tokenizer,
        task,
        max_length=16,
        target_max_length=2,
        device=torch.device("cpu"),
    )

    assert set(rows[0]["candidate_logprobs_per_token"].values()) == {-6.0}


def test_run_trains_with_selection_validation_and_checkpoint(monkeypatch, tmp_path) -> None:
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(train_module.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr(
        train_module,
        "load_task_data",
        lambda task, cfg: LoadedTaskData(
            train=ListDataset([{"sentence": "train sample", "label": 1}]),
            val=ListDataset([{"sentence": "selection sample", "label": 1}]),
            report_val=ListDataset([{"sentence": "validation sample", "label": 0}]),
            test=None,
            split_names={"train": "train.tsv", "selection": "dev.tsv", "val": "dev.tsv", "test": None},
            split_details=split_details(),
        ),
    )
    monkeypatch.setattr(
        train_module,
        "build_model",
        lambda cfg, device: (TinyTrainModel().to(device), "gpt2", []),
    )
    cfg = FineTuneEvalConfig(
        model_name_or_path="tiny",
        method="full_finetune",
        task="sst2",
        output_dir=str(tmp_path),
        do_train=True,
        do_eval=False,
        epochs=2,
        batch_size=1,
        eval_batch_size=1,
        device="cpu",
    )

    metrics = train_module.run(cfg)

    assert metrics["best_epoch"] >= 1
    assert metrics["best_validation_loss"] is not None
    assert metrics["selection_metric"] == "accuracy"
    assert metrics["manifest"]["training"]["effective_batch_size"] == 1
    task_manifest = metrics["manifest"]["task"]
    assert task_manifest["evaluation_protocol"] == "decoder_lm_verbalized_classification"
    assert task_manifest["prompt_template"] == "Sentence: {sentence}\\nSentiment:"
    assert task_manifest["candidate_verbalizers"] == {"negative": " negative", "positive": " positive"}
    assert task_manifest["candidate_score_normalization"] == "mean_token_logprob"
    assert task_manifest["target_eos"] is False
    assert task_manifest["candidate_tokenizations"] == {
        "negative": tokenizer(" negative", add_special_tokens=False).input_ids,
        "positive": tokenizer(" positive", add_special_tokens=False).input_ids,
    }
    assert (tmp_path / "best_checkpoint.pt").exists()
    assert "validation_accuracy" in metrics


def test_split_policy_preserves_unlabeled_official_sst2_test(monkeypatch) -> None:
    raw = RawSplits(
        {
            "train": ListDataset([{"sentence": "train", "label": 1}]),
            "validation": ListDataset([{"sentence": "val", "label": 0}]),
            "test": ListDataset([{"sentence": "test", "label": -1}]),
        }
    )
    monkeypatch.setattr(datasets_module, "_load_local_glue_task", lambda data_dir, task: raw)
    cfg = FineTuneEvalConfig("tiny", "zero_shot", "sst2", "unused")

    data = datasets_module.load_task_data(TASKS["sst2"], cfg)

    assert data.split_names["test"] == "test.tsv"
    assert data.split_details["test"]["source_kind"] == "official"
    assert data.split_details["test"]["has_labels"] is False


def test_split_policy_preserves_unlabeled_official_rte_test(monkeypatch) -> None:
    raw = RawSplits(
        {
            "train": ListDataset([{"sentence1": "train premise", "sentence2": "train hypothesis", "label": "entailment"}]),
            "validation": ListDataset(
                [{"sentence1": "val premise", "sentence2": "val hypothesis", "label": "not_entailment"}]
            ),
            "test": ListDataset([{"sentence1": "test premise", "sentence2": "test hypothesis", "idx": 7}]),
        }
    )
    monkeypatch.setattr(datasets_module, "_load_local_glue_task", lambda data_dir, task: raw)
    cfg = FineTuneEvalConfig("tiny", "zero_shot", "rte", "unused")

    data = datasets_module.load_task_data(TASKS["rte"], cfg)

    assert data.split_names["test"] == "test.tsv"
    assert data.split_details["test"]["source_kind"] == "official"
    assert data.split_details["test"]["has_labels"] is False


def test_rte_tsv_loader_handles_labeled_and_unlabeled_rows(tmp_path) -> None:
    train_path = tmp_path / "train.tsv"
    test_path = tmp_path / "test.tsv"
    train_path.write_text(
        "index\tsentence1\tsentence2\tlabel\n"
        "3\tPremise text\tHypothesis text\tentailment\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "index\tsentence1\tsentence2\n"
        "4\tTest premise\tTest hypothesis\n",
        encoding="utf-8",
    )

    train = datasets_module._load_rte_tsv(train_path, has_labels=True)
    test = datasets_module._load_rte_tsv(test_path, has_labels=False)

    assert train[0] == {
        "sentence1": "Premise text",
        "sentence2": "Hypothesis text",
        "idx": 3,
        "index": 3,
        "label": "entailment",
    }
    assert test[0] == {
        "sentence1": "Test premise",
        "sentence2": "Test hypothesis",
        "idx": 4,
        "index": 4,
    }


def test_selection_split_can_be_carved_from_train_without_consuming_validation(monkeypatch) -> None:
    raw = RawSplits(
        {
            "train": ListDataset([{"sentence": f"train {idx}", "label": idx % 2} for idx in range(10)]),
            "validation": ListDataset([{"sentence": "official val", "label": 1}]),
            "test": ListDataset([{"sentence": "official test", "label": -1}]),
        }
    )
    monkeypatch.setattr(datasets_module, "_load_local_glue_task", lambda data_dir, task: raw)
    cfg = FineTuneEvalConfig("tiny", "zero_shot", "sst2", "unused", selection_split_from_train=0.2)

    data = datasets_module.load_task_data(TASKS["sst2"], cfg)

    assert len(data.train) == 8
    assert len(data.val) == 2
    assert len(data.report_val) == 1
    assert data.split_names["selection"] == "train_selection"
    assert data.split_names["val"] == "dev.tsv"
    assert data.split_details["selection"]["is_train_derived"] is True


def test_zero_shot_and_full_finetune_have_explicit_trainable_counts(monkeypatch) -> None:
    monkeypatch.setattr(train_module.AutoModelForCausalLM, "from_pretrained", lambda _: TinyTrainModel())

    zero_shot, _, _ = train_module.build_model(
        FineTuneEvalConfig("tiny", "zero_shot", "sst2", "unused", device="cpu"),
        torch.device("cpu"),
    )
    full_finetune, _, _ = train_module.build_model(
        FineTuneEvalConfig("tiny", "full_finetune", "sst2", "unused", device="cpu"),
        torch.device("cpu"),
    )

    assert sum(p.numel() for p in zero_shot.parameters() if p.requires_grad) == 0
    assert sum(p.numel() for p in full_finetune.parameters() if p.requires_grad) == sum(p.numel() for p in full_finetune.parameters())


def test_unlabeled_official_classification_test_outputs_predictions_without_accuracy() -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel(next_token_id=tokenizer.eos_token_id)

    metrics, rows = score_candidates(
        model,
        [{"sentence": "official test", "label": -1, "idx": 42}],
        tokenizer,
        TASKS["sst2"],
        max_length=16,
        target_max_length=2,
        device=torch.device("cpu"),
    )

    assert "accuracy" not in metrics
    assert metrics["predicted_examples"] == 1.0
    assert rows[0]["idx"] == 42
    assert rows[0]["reference"] is None
    assert rows[0]["prediction"] in {"negative", "positive"}


def test_rte_candidate_scoring_outputs_official_labels() -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel(next_token_id=tokenizer.eos_token_id)

    labeled_metrics, labeled_rows = score_candidates(
        model,
        [{"sentence1": "dev premise", "sentence2": "dev hypothesis", "label": "entailment"}],
        tokenizer,
        TASKS["rte"],
        max_length=32,
        target_max_length=4,
        device=torch.device("cpu"),
    )
    metrics, rows = score_candidates(
        model,
        [{"sentence1": "official test premise", "sentence2": "official test hypothesis", "idx": 42}],
        tokenizer,
        TASKS["rte"],
        max_length=32,
        target_max_length=4,
        device=torch.device("cpu"),
    )

    assert "accuracy" in labeled_metrics
    assert labeled_rows[0]["reference"] == "entailment"
    assert "accuracy" not in metrics
    assert metrics["predicted_examples"] == 1.0
    assert rows[0]["idx"] == 42
    assert rows[0]["reference"] is None
    assert rows[0]["prediction"] in {"entailment", "not_entailment"}


def test_candidate_scoring_prefers_official_index_when_idx_missing() -> None:
    tokenizer = TinyTokenizer()
    model = SpyLogitModel(next_token_id=tokenizer.eos_token_id)

    _, rows = score_candidates(
        model,
        [{"sentence": "official test", "index": 1729}],
        tokenizer,
        TASKS["sst2"],
        max_length=16,
        target_max_length=2,
        device=torch.device("cpu"),
    )

    assert rows[0]["idx"] == 1729


def test_test_output_writers(tmp_path) -> None:
    train_module.write_test_outputs(
        tmp_path,
        TASKS["sst2"],
        [{"idx": 7, "prediction": "positive"}, {"idx": 8, "prediction": "negative"}],
    )

    assert (tmp_path / "test_predictions.csv").read_text().splitlines() == [
        "idx,prediction",
        "7,positive",
        "8,negative",
    ]
    assert (tmp_path / "submissions" / "SST-2.tsv").read_text().splitlines() == [
        "index\tprediction",
        "7\tpositive",
        "8\tnegative",
    ]


def test_rte_test_output_writer_uses_glue_submission_name_and_labels(tmp_path) -> None:
    train_module.write_test_outputs(
        tmp_path,
        TASKS["rte"],
        [{"idx": 7, "prediction": "entailment"}, {"idx": 8, "prediction": "not_entailment"}],
    )

    assert (tmp_path / "submissions" / "RTE.tsv").read_text().splitlines() == [
        "index\tprediction",
        "7\tentailment",
        "8\tnot_entailment",
    ]


def test_cli_accepts_rte_task() -> None:
    args = cli_module.build_arg_parser().parse_args(
        [
            "--model_name_or_path",
            "tiny",
            "--method",
            "zero_shot",
            "--task",
            "rte",
            "--output_dir",
            "unused",
        ]
    )

    assert args.task == "rte"


def test_comparison_aggregation_uses_validation_report_split_and_checks_fairness() -> None:
    base_manifest = {
        "config": {
            "model_name_or_path": "tiny",
            "task": "sst2",
            "max_length": 16,
            "target_max_length": 2,
            "epochs": 1,
            "lr": 0.1,
            "weight_decay": 0.0,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_batch_size": 1,
            "selection_split_from_train": 0.1,
        },
        "training": {"seed": 0, "effective_batch_size": 1},
        "model": {"model_family": "gpt2", "layer_indices": [1]},
        "task": {"main_metric": "accuracy"},
        "splits": split_details(),
    }
    rows = [
        {
            "task": "sst2",
            "method": "akaza_freez",
            "validation_accuracy": 0.75,
            "trainable_parameters": 10,
            "total_parameters": 100,
            "best_epoch": 1,
            "selection_metric": "accuracy",
            "best_selection_score": 0.75,
            "split_details": split_details(),
            "manifest": base_manifest,
        },
        {
            "task": "sst2",
            "method": "akaza_freez",
            "validation_accuracy": 1.0,
            "trainable_parameters": 10,
            "total_parameters": 100,
            "best_epoch": 1,
            "selection_metric": "accuracy",
            "best_selection_score": 1.0,
            "split_details": split_details(),
            "manifest": {**base_manifest, "training": {"seed": 1, "effective_batch_size": 1}},
        },
    ]

    validate_fairness(rows)
    per_run = comparison_rows(rows)
    aggregate = aggregate_rows(per_run)

    assert per_run[0]["report_split"] == "validation"
    assert aggregate[0]["n"] == 2
    assert aggregate[0]["mean"] == 0.875


def test_fairness_check_allows_separate_full_finetune_baseline_group() -> None:
    peft_manifest = {
        "config": {
            "model_name_or_path": "tiny",
            "task": "sst2",
            "max_length": 16,
            "target_max_length": 2,
            "epochs": 1,
            "lr": 0.1,
            "weight_decay": 0.0,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_batch_size": 1,
            "selection_split_from_train": 0.1,
        },
        "training": {"seed": 0, "effective_batch_size": 1},
        "model": {"model_family": "gpt2", "layer_indices": [1], "method": "akaza_freez"},
        "task": {"main_metric": "accuracy"},
        "splits": split_details(),
    }
    base_manifest = {
        **peft_manifest,
        "config": {**peft_manifest["config"], "lr": 0.01},
        "model": {"model_family": "gpt2", "layer_indices": [], "method": "full_finetune"},
    }

    validate_fairness(
        [
            {"task": "sst2", "method": "akaza_freez", "manifest": peft_manifest},
            {"task": "sst2", "method": "full_finetune", "manifest": base_manifest},
        ]
    )
