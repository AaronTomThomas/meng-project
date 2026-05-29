# Downstream GLUE-Style Adapter Evaluation Suite

This directory contains a downstream evaluation suite for comparing fine-tuning and adapter methods on GLUE-style natural language understanding tasks using decoder-only causal language models.

The suite currently supports a subset of GLUE. At present, only SST-2 is active, but the code is structured around task specifications so that additional GLUE tasks can be added incrementally.

The evaluation protocol uses causal language-model verbalizers rather than classifier heads. Each classification example is converted into a prompt, and the model predicts by scoring a small set of candidate label strings. The predicted class is the verbalizer with the highest conditional log-likelihood.

This is therefore a causal-LM verbalizer evaluation suite, not a standard classifier-head GLUE fine-tuning setup.

## Current Task Coverage

Currently active:

```text
sst2    binary sentiment classification from GLUE SST-2
```

Future tasks may include RTE, MRPC, QNLI, or other GLUE tasks, but these are not active yet.

## Data

The suite uses local official GLUE files. For SST-2, the expected files are:

```text
glue_data/SST-2/train.tsv
glue_data/SST-2/dev.tsv
glue_data/SST-2/test.tsv
```

The splits are used as follows:

```text
train.tsv    training data
dev.tsv      checkpoint selection and local validation reporting
test.tsv     official hidden-label test prediction split
```

If the files are missing, the suite calls the GLUE downloader helper automatically. SST-2 can also be downloaded explicitly with:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.download_glue_data \
  --data_dir glue_data
```

For SST-2, the current prompt template is:

```text
Sentence: {sentence}
Sentiment:
```

The verbalized label candidates are:

```text
negative -> " negative"
positive -> " positive"
```

The official SST-2 test labels are hidden, so local metrics do not include test accuracy. The test split is used only to write prediction files. For local thesis reporting, use validation accuracy on `dev.tsv`.

## Evaluation Protocol

For each example, the suite formats the input as a prompt and scores each candidate verbalizer under the causal language model.

For SST-2, the comparison is:

```text
P(" positive" | prompt)
```

against:

```text
P(" negative" | prompt)
```

EOS is not appended to the SST-2 label target, so candidate scoring compares only the label verbalizers themselves.

The main reported metric for SST-2 is validation accuracy on `dev.tsv`.

## Selection and Reporting

The main thesis runs use the standard GLUE-style local evaluation protocol:

```text
train.tsv    train the model
dev.tsv      select the best checkpoint and report validation accuracy
test.tsv     generate hidden-label prediction files
```

This corresponds to:

```bash
--selection_split_from_train 0.0
```

When `--selection_split_from_train 0.0` is used, the official SST-2 development split is used both for checkpoint selection and for local validation reporting.

This is the standard setup when official test labels are hidden. The hidden-label test split is not used for training, checkpoint selection, or hyperparameter tuning.

## Methods

The suite currently supports:

```text
zero_shot        frozen pretrained model evaluation
full_finetune    full-model fine-tuning
akaza_freez      AKAZA/FreeZ attention-output correction
lora             LoRA baseline
loreft           LoReFT baseline
reft             alias/compatibility option for ReFT-style runs
```

For the main thesis comparison, use:

```text
akaza_freez
lora
loreft
full_finetune
```

Use `zero_shot` as a separate frozen-model reference point.

## Main GPT-2 Thesis Runs

Run all commands from the project root.

First set the shared environment variables and common arguments:

```bash
export PYTHONPATH=src
export ROOT=outputs/downstream/thesis_gpt2/sst2
export MODEL=openai-community/gpt2

export COMMON="--model_name_or_path ${MODEL} \
  --task sst2 \
  --model_family gpt2 \
  --device cuda \
  --do_train --do_eval \
  --batch_size 4 \
  --eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --epochs 10 \
  --patience 3 \
  --eval_every 1 \
  --weight_decay 0.0 \
  --selection_split_from_train 0.0 \
  --max_length 256 \
  --target_max_length 4 \
  --grad_clip 1.0"
```

## AKAZA/FreeZ

```bash
for seed in 0 1 2 3 4; do
  uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
    ${COMMON} \
    --method akaza_freez \
    --output_dir "${ROOT}/akaza_freez/seed_${seed}" \
    --seed "${seed}" \
    --lr 3e-4 \
    --layer_indices 6,7,8,9,10,11 \
    --bottleneck_dim 4 \
    --adapter_dropout 0.05 \
    --output_scale 1.0
done
```

## LoRA

```bash
for seed in 0 1 2 3 4; do
  uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
    ${COMMON} \
    --method lora \
    --output_dir "${ROOT}/lora/seed_${seed}" \
    --seed "${seed}" \
    --lr 3e-4 \
    --layer_indices 6,7,8,9,10,11 \
    --peft_target_profile attn_c_proj \
    --lora_rank 4 \
    --lora_alpha 4 \
    --lora_dropout 0.05 \
    --lora_bias none
done
```

## LoReFT

```bash
for seed in 0 1 2 3 4; do
  uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
    ${COMMON} \
    --method loreft \
    --output_dir "${ROOT}/loreft/seed_${seed}" \
    --seed "${seed}" \
    --lr 3e-4 \
    --layer_indices 6,7,8,9,10,11 \
    --reft_rank 4 \
    --reft_dropout 0.05 \
    --reft_output_scale 1.0 \
    --reft_position_mode all
done
```

## Full Fine-Tuning

Full fine-tuning uses a smaller learning rate because all base-model parameters are updated.

```bash
for seed in 0 1 2 3 4; do
  uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
    ${COMMON} \
    --method full_finetune \
    --output_dir "${ROOT}/full_finetune/seed_${seed}" \
    --seed "${seed}" \
    --lr 3e-5
done
```

## Zero-Shot Evaluation

Run zero-shot separately from the trained aggregate comparison.

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method zero_shot \
  --task sst2 \
  --model_family gpt2 \
  --device cuda \
  --output_dir "${ROOT}/zero_shot/seed_0" \
  --seed 0 \
  --do_eval \
  --selection_split_from_train 0.0 \
  --max_length 256 \
  --target_max_length 4
```

## Smoke Test

Before running the full experiment grid, run a small smoke test to check that data loading, model wrapping, training, evaluation, checkpointing, and prediction writing all work.

```bash
export PYTHONPATH=src

uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method akaza_freez \
  --task sst2 \
  --model_family gpt2 \
  --device cuda \
  --output_dir outputs/downstream/smoke/sst2_akaza \
  --do_train --do_eval \
  --max_train_examples 32 \
  --max_val_examples 32 \
  --max_test_examples 32 \
  --selection_split_from_train 0.0 \
  --epochs 1 \
  --patience 1 \
  --eval_every 1 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --lr 3e-4 \
  --weight_decay 0.0 \
  --max_length 256 \
  --target_max_length 4 \
  --layer_indices 6,7,8,9,10,11 \
  --bottleneck_dim 4 \
  --adapter_dropout 0.05 \
  --output_scale 1.0
```

Expected outputs include:

```text
config.json
manifest.json
metrics.json
training_log.jsonl
best_checkpoint.pt
predictions.jsonl
test_predictions.csv
submissions/SST-2.tsv
```

## Aggregating Results

After all trained runs finish, aggregate the main comparison:

```bash
PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.compare \
  ${ROOT}/akaza_freez/seed_*/metrics.json \
  ${ROOT}/lora/seed_*/metrics.json \
  ${ROOT}/loreft/seed_*/metrics.json \
  ${ROOT}/full_finetune/seed_*/metrics.json \
  --output_csv ${ROOT}_comparison_summary.csv \
  --aggregate_json ${ROOT}_comparison_summary.json
```

Inspect the summary CSV:

```bash
cat ${ROOT}_comparison_summary.csv
```

Or print the aggregate JSON in a compact format:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/downstream/thesis_gpt2/sst2_comparison_summary.json")
data = json.loads(path.read_text())

for row in data["aggregates"]:
    print(
        row["method"],
        "n=", row["n"],
        "mean=", row["mean"],
        "std=", row["std"],
        "stderr=", row["stderr"],
        "params=", row["trainable_parameters"],
    )
PY
```

## Evaluation From a Checkpoint

Example AKAZA checkpoint evaluation:

```bash
uv run python -m experiments.router_development.attention_adapter.fine_tuning_evaluation.cli \
  --model_name_or_path openai-community/gpt2 \
  --method akaza_freez \
  --task sst2 \
  --model_family gpt2 \
  --device cuda \
  --output_dir outputs/downstream/sst2_akaza_eval \
  --checkpoint_path outputs/downstream/thesis_gpt2/sst2/akaza_freez/seed_0/best_checkpoint.pt \
  --do_eval \
  --selection_split_from_train 0.0 \
  --max_length 256 \
  --target_max_length 4 \
  --layer_indices 6,7,8,9,10,11 \
  --bottleneck_dim 4 \
  --adapter_dropout 0.05 \
  --output_scale 1.0
```

## Output Files

Each run writes:

```text
config.json              raw CLI/config values
manifest.json            reproducibility metadata
metrics.json             final metrics
training_log.jsonl       per-epoch training and validation metrics
best_checkpoint.pt       best selected trainable parameters
predictions.jsonl        per-example predictions
test_predictions.csv     simple CSV of hidden-test predictions
submissions/SST-2.tsv    GLUE-format SST-2 submission file
```

The manifest records the model, method, task, layer indices, trainable parameter count, tokenizer metadata, split details, random seed, batch size, learning rate, sequence length, target length, package versions, and git metadata.

## Reporting Notes

For the thesis table, report:

```text
method
trainable parameters
validation accuracy mean
validation accuracy standard deviation
validation accuracy standard error
number of seeds
```

Use `validation_accuracy` from `metrics.json` as the main score.

Do not report local official SST-2 test accuracy, because the official test labels are hidden. The test split should only be used for writing prediction files.

The zero-shot result should be reported separately from the seed-averaged trained comparison.

The full fine-tuning baseline should be reported separately from PEFT methods in parameter-count tables, because it updates all base-model parameters.

The development split is used for checkpoint selection and local validation reporting. This is the standard local protocol for GLUE-style experiments when hidden test labels are unavailable. Avoid presenting the validation result as hidden-test performance unless the generated `SST-2.tsv` file is submitted to the official GLUE evaluation server.

This suite is intended to provide a compact downstream classification sanity check alongside the main language-modelling and PEFT experiments.
