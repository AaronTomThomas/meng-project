# Downstream Fine-Tuning Evaluation

This package fine-tunes causal language models on downstream tasks with the existing `attention_adapter` PEFT methods, then evaluates task performance. It is for task fine-tuning performance comparisons, not zero-shot model ability.

Supported datasets:

- SST-2: `load_dataset("nyu-mll/glue", "sst2")`
- BoolQ: `load_dataset("google/boolq")`
- E2E NLG: `load_dataset("tuetschek/e2e_nlg")`

SST-2 and BoolQ are evaluated with candidate conditional log-likelihood accuracy. E2E NLG reports held-out causal-LM loss and saves generated predictions/references; simple BLEU and ROUGE-L are computed without extra dependencies.

## Examples

SST-2 AKAZA:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method akaza_freez \
  --task sst2 \
  --output_dir outputs/downstream/sst2_akaza \
  --do_train --do_eval \
  --batch_size 4 --eval_batch_size 8 \
  --epochs 3 --lr 3e-4
```

SST-2 LoRA:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method lora \
  --task sst2 \
  --output_dir outputs/downstream/sst2_lora \
  --do_train --do_eval \
  --batch_size 4 --eval_batch_size 8 \
  --epochs 3 --lr 3e-4
```

BoolQ AKAZA:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method akaza_freez \
  --task boolq \
  --output_dir outputs/downstream/boolq_akaza \
  --do_train --do_eval \
  --batch_size 2 --eval_batch_size 4 \
  --epochs 3 --lr 3e-4 \
  --max_length 512
```

E2E NLG AKAZA:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method akaza_freez \
  --task e2e_nlg \
  --output_dir outputs/downstream/e2e_akaza \
  --do_train --do_eval --generate \
  --batch_size 4 --eval_batch_size 8 \
  --epochs 3 --lr 3e-4 \
  --target_max_length 96
```

Evaluation-only from a checkpoint:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method akaza_freez \
  --task sst2 \
  --output_dir outputs/downstream/sst2_akaza_eval \
  --checkpoint_path outputs/downstream/sst2_akaza/best_checkpoint.pt \
  --do_eval
```

For smoke tests, add small limits such as:

```bash
--max_train_examples 32 --max_val_examples 32 --max_test_examples 32
```

## Outputs

Each run writes:

- `config.json`: resolved CLI configuration.
- `metrics.json`: parameter counts, split names, validation/test loss, and task metrics.
- `best_checkpoint.pt`: trainable parameters in the same trainable-state style used by the existing adapter trainer, written for training runs.
- `predictions.jsonl`: classification candidate predictions or E2E generated outputs/references.
- `training_log.jsonl`: per-epoch training and validation loss rows when training is enabled.
