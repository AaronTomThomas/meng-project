import argparse 
import math
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments import get_experiment

def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

@torch.no_grad()
def ddp_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    ddp_setup(rank, world_size)

    try:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        if rank == 0:
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
        
        experiment = get_experiment(args.experiment)    
        dataset, sampler, loader, meta = experiment.build_data(args=args, rank=rank, world_size=world_size)

        if rank == 0:
            experiment.print_startup_info(dataset, meta, args)
        model = experiment.build_model(args=args, meta=meta).to(device)

        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

        optimizer = experiment.build_optimizer(args=args, model=ddp_model)
        scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

        @torch.no_grad()
        def evaluate(num_batches=None):
            ddp_model.eval()
            total = torch.tensor(0.0, device=device)
            count = torch.tensor(0.0, device=device)

            for i, batch in enumerate(loader):
                if num_batches is not None and i >= num_batches:
                    break

                loss = experiment.compute_loss(model=ddp_model.module, batch=batch, device=device)
                total += loss.detach()
                count += 1.0

            mean_loss = total / torch.clamp(count, min=1.0)
            mean_loss = ddp_mean(mean_loss)
            ddp_model.train()
            return mean_loss.item()
    
        ddp_model.train()
        global_step = 0
        start_time = time.time()
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)

            for batch in loader:
                global_step += 1

                if args.max_steps is not None and global_step > args.max_steps:
                    if rank == 0:
                        print(f"Reached max_steps={args.max_steps}, stopping training.")
                    return
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=args.fp16):
                    loss = experiment.compute_loss(
                        model=ddp_model,
                        batch=batch,
                        device=device,
                    )

                    scaler.scale(loss).backward()

                    if args.grad_clip is not None and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            ddp_model.parameters(),
                            max_norm=args.grad_clip,
                        )

                    scaler.step(optimizer)
                    scaler.update()

                    if rank == 0 and (global_step % args.log_every == 0):
                        elapsed = time.time() - start_time
                        print(
                            f"[epoch {epoch} step {global_step}] "
                            f"loss={loss.item():.4f} time={elapsed:.1f}s"
                        )

                    if global_step % args.eval_every == 0:
                        eval_loss = evaluate(num_batches=args.eval_batches)
                        if rank == 0:
                            ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
                            print(f"  eval_loss={eval_loss:.4f} ppl={ppl:.2f}")
                            experiment.generate_sample(
                                model=ddp_model.module,
                                dataset=dataset,
                                device=device,
                                args=args,
                                global_step=global_step,
                            )
                        ckpt = {
                            "model": ddp_model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "args": vars(args),
                            "meta": meta,
                            "global_step": global_step,
                            "epoch": epoch,
                            "experiment": args.experiment,
                        }
                        torch.save(ckpt, args.save_path)
    finally:
        ddp_cleanup()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, default="tiny_shakespeare")
    repo_root = Path(__file__).resolve().parent.parent.parent

    parser.add_argument(
        "--data_path",
        type=str,
        default=repo_root / "src" / "data" / "tiny_shakespeare.txt",
    )

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="checkpoints/ckpt.pt")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=256)

    parser.add_argument("--gen_prompt", type=str, default="ROMEO:")
    parser.add_argument("--gen_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)

    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())