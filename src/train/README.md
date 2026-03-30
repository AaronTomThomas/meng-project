# Training Harness

This is a reusable PyTorch DDP training entrypoint for experiments that follow the standard supervised pattern:
- build dataset and dataloader
- build model
- compute loss
- backpropagate
- evaluate periodically
- save checkpoints
- optionally generate samples

## High-Level

Codebase is split into:
- `src/train/train.py` - Generic training loop, DDP setup, eval, logging, checkpointing
- `src/experiments` - Experiment-specific logic
  - dataset construction
  - model construction
  - optimizer construction
  - loss computation
  - sample generation
  - startup printing

Each experiment should implement the common experiment API defined in `src/experiments/base.py`.

## Checkpoint Contents

Checkpoints store:
```python
{
    "model": ddp_model.module.state_dict(),
    "optimizer": optimizer.state_dict(),
    "args": vars(args),
    "meta": meta,
    "global_step": global_step,
    "epoch": epoch,
    "experiment": args.experiment,
}
```

## Usage

Run training from the repo root with `torchrun` through `uv`:

## Adding a new Experiment

- Create a new experiment file in src/experiments/
- Implement the interface from src/experiments/base.py
- Register it in src/experiments/__init__.py
- Run it with --experiment <name>

### Smoke-test
```bash
uv run torchrun --nproc_per_node=1 src/train/train.py \
  --experiment tiny_shakespeare \
  --epochs 1 \
  --max_steps 500 \
  --batch_size 4 \
  --block_size 32 \
  --max_seq_length 32 \
  --n_layers 2 \
  --d_model 64 \
  --n_heads 4 \
  --d_ff 128 \
  --log_every 50 \
  --eval_every 100 \
  --eval_batches 5 \
  --gen_tokens 30 \
  --gen_prompt "ROMEO:" \
  --save_path checkpoints/test_ckpt.pt
```

