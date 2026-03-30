from experiments.base import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from data.tiny_shakespeare import CharDataset
from models.transformer import TransformerDecoder


class TinyShakespeareExperiment(Experiment):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def build_data(self, args, rank, world_size):
        dataset = CharDataset(args.data_path, block_size=args.block_size)

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )

        meta = {
            "vocab_size": dataset.vocab_size,
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        }

        return dataset, sampler, loader, meta

    def print_startup_info(self, dataset, meta, args):
        print("data_path:", args.data_path)
        print("len(dataset):", len(dataset))
        print("vocab_size:", meta["vocab_size"])
        print("vocab sample:", list(dataset.stoi.items())[:20])

        if hasattr(dataset, "text"):
            print("text[:200]:", repr(dataset.text[:200]))

    def build_model(self, args, meta):
        return TransformerDecoder(
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            vocab_size=meta["vocab_size"],
            dropout=args.dropout,
            max_seq_length=args.max_seq_length,
        )

    def build_optimizer(self, args, model):
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def compute_loss(self, model, batch, device):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        B, T, V = logits.shape

        loss = self.criterion(
            logits.reshape(B * T, V),
            y.reshape(B * T),
        )
        return loss

    @torch.no_grad()
    def generate_sample(self, model, dataset, device, args, global_step):
        prompt = args.gen_prompt

        try:
            idx = dataset.encode_text(prompt).unsqueeze(0).to(device)
        except KeyError as e:
            print(f"[sample skipped @ step {global_step}] prompt contains OOV char: {e}")
            return

        top_k = None if args.top_k is None or args.top_k <= 0 else args.top_k

        model.eval()
        out = model.generate(
            idx=idx,
            max_new_tokens=args.gen_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )
        sample = dataset.decode_tokens(out[0].detach().cpu())

        print("\n" + "=" * 80)
        print(f"[sample @ step {global_step}]")
        print(sample)
        print("=" * 80 + "\n")