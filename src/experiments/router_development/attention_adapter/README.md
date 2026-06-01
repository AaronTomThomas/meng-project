# Attention Adapter

Training entrypoints for the AKAZA/FreeZ attention adapter experiments, LoRA baselines, and canonical PyReFT LoReFT baselines. The runs here train on packed language-modeling text chunks. Downstream task fine-tuning lives in `fine_tuning_evaluation/`.

IA3 is intentionally not included in this package.

## CLI Help

Each model/method pair has its own subcommand:

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli --help
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-akaza --help
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-lora --help
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-loreft --help
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli pythia-akaza --help
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli pythia-lora --help
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli pythia-loreft --help
```

LoReFT uses `pyreft.ReftConfig`, `pyreft.LoreftIntervention`, and `pyreft.get_reft_model` directly. `pyreft` is imported at runtime because the current upstream package metadata conflicts with this repo's locked dependency set; install it in a compatible environment before running `*-loreft` commands. The legacy `--reft_output_scale` flag is still accepted for config compatibility, but canonical PyReFT controls the intervention math.

The training CLI limits dataset size by packed chunks: `--max_train_chunks`, `--max_val_chunks`, and `--max_test_chunks`. It does not expose `--max_*_texts` flags.

## Smoke Test

Small CPU run for imports, argument wiring, dataset loading, training, and checkpoint writing:

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-akaza \
  --device cpu \
  --layer_indices 6 \
  --block_size 96 \
  --batch_size 2 \
  --max_train_chunks 8 \
  --max_val_chunks 4 \
  --max_test_chunks 4 \
  --epochs 1 \
  --patience 1 \
  --bottleneck_dim 4 \
  --adapter_dropout 0.05 \
  --output_scale 0.05 \
  --output_path outputs/attention_adapter/smoke_gpt2_akaza.pt
```

## GPT-2 Runs

### GPT-2 AKAZA/FreeZ

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-akaza \
  --device cuda \
  --layer_indices 6,7,8,9,10,11 \
  --block_size 96 \
  --batch_size 4 \
  --max_train_chunks 2048 \
  --max_val_chunks 512 \
  --max_test_chunks 512 \
  --epochs 500 \
  --patience 3 \
  --bottleneck_dim 32 \
  --adapter_dropout 0.05 \
  --output_scale 1.0 \
  --output_path outputs/attention_adapter/gpt2_akaza.pt
```

### GPT-2 LoRA

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-lora \
  --device cuda \
  --layer_indices 6,7,8,9,10,11 \
  --block_size 96 \
  --batch_size 4 \
  --max_train_chunks 2048 \
  --max_val_chunks 512 \
  --max_test_chunks 512 \
  --epochs 500 \
  --patience 30 \
  --peft_target_profile attn_c_proj \
  --lora_rank 4 \
  --lora_alpha 4 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --output_path outputs/attention_adapter/gpt2_lora_attn_c_proj.pt
```

### GPT-2 LoReFT

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-loreft \
  --device cuda \
  --layer_indices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --block_size 96 \
  --batch_size 4 \
  --max_train_chunks 2048 \
  --max_val_chunks 512 \
  --max_test_chunks 512 \
  --epochs 500 \
  --patience 30 \
  --reft_rank 4 \
  --reft_dropout 0.05 \
  --reft_position_mode all \
  --output_path outputs/attention_adapter/gpt2_loreft_all_layers.pt
```

## Pythia/GPT-NeoX Runs

### Pythia/GPT-NeoX AKAZA/FreeZ

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli pythia-akaza \
  --device cuda \
  --layer_indices 10,11,12,13,14,15 \
  --block_size 96 \
  --batch_size 1 \
  --max_train_chunks 512 \
  --max_val_chunks 128 \
  --max_test_chunks 128 \
  --epochs 80 \
  --patience 3 \
  --bottleneck_dim 1 \
  --adapter_dropout 0.05 \
  --output_scale 1.0 \
  --output_path outputs/attention_adapter/pythia_akaza.pt
```

### Pythia/GPT-NeoX LoRA

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli pythia-lora \
  --device cuda \
  --layer_indices 10,11,12,13,14,15 \
  --block_size 96 \
  --batch_size 1 \
  --max_train_chunks 512 \
  --max_val_chunks 128 \
  --max_test_chunks 128 \
  --epochs 80 \
  --patience 3 \
  --peft_target_profile attn_dense \
  --lora_rank 4 \
  --lora_alpha 2 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --output_path outputs/attention_adapter/pythia_lora_attn_dense.pt
```

### Pythia/GPT-NeoX LoReFT

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli pythia-loreft \
  --device cuda \
  --layer_indices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --block_size 96 \
  --batch_size 1 \
  --max_train_chunks 512 \
  --max_val_chunks 128 \
  --max_test_chunks 128 \
  --epochs 80 \
  --patience 30 \
  --reft_rank 4 \
  --reft_dropout 0.05 \
  --reft_position_mode all \
  --output_path outputs/attention_adapter/pythia_loreft_all_layers.pt
```

## Dataset Overrides

The commands above use WikiText-2 raw by default:

```text
--dataset_name wikitext
--dataset_config wikitext-2-raw-v1
--text_field text
--train_split train
--val_split validation
--test_split test
```

For another Hugging Face text dataset, pass the dataset arguments explicitly. For example, Penn Treebank via the generated Parquet revision:

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.cli gpt2-akaza \
  --device cuda \
  --dataset_name ptb-text-only/ptb_text_only \
  --dataset_config "" \
  --dataset_revision refs/convert/parquet \
  --text_field sentence \
  --train_split train \
  --val_split validation \
  --test_split test \
  --layer_indices 6,7,8,9,10,11 \
  --block_size 96 \
  --batch_size 4 \
  --max_train_chunks 2048 \
  --max_val_chunks 512 \
  --max_test_chunks 512 \
  --epochs 500 \
  --patience 30 \
  --bottleneck_dim 4 \
  --adapter_dropout 0.05 \
  --output_scale 0.05 \
  --output_path outputs/attention_adapter/gpt2_akaza_ptb.pt
```
