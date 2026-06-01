from __future__ import annotations

from typing import Dict, Sequence

from experiments.router_development.attention_adapter.adapters.base import AdapterModel
from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig
import torch
from torch import nn

REFT_METHODS = {"loreft"}
REFT_POSITION_MODES = {"all", "prefix", "suffix", "prefix_suffix"}


def _import_pyreft():
    try:
        import pyreft
    except ImportError as exc:
        raise ImportError(
            "Canonical LoReFT runs require pyreft. Install pyreft in an environment "
            "whose transitive dependencies are compatible with this project."
        ) from exc
    return pyreft


def reft_positions(
    *,
    seq_len: int,
    mode: str,
    prefix_positions: int,
    suffix_positions: int,
) -> list[int]:
    if mode == "all":
        return list(range(seq_len))
    if mode == "prefix":
        n = prefix_positions if prefix_positions > 0 else seq_len
        return list(range(min(n, seq_len)))
    if mode == "suffix":
        n = suffix_positions if suffix_positions > 0 else seq_len
        return list(range(max(0, seq_len - n), seq_len))
    if mode == "prefix_suffix":
        if prefix_positions <= 0 and suffix_positions <= 0:
            return list(range(seq_len))
        positions = set(range(min(prefix_positions, seq_len)))
        if suffix_positions > 0:
            positions.update(range(max(0, seq_len - suffix_positions), seq_len))
        return sorted(positions)
    raise ValueError(f"Unknown reft_position_mode={mode!r}; choices={sorted(REFT_POSITION_MODES)}")


class ResidualReFTAdapter(AdapterModel):
    """Canonical PyReFT LoReFT residual-stream adapter."""

    def __init__(self, *, model: nn.Module, cfg: AdapterFineTuneConfig, layer_indices: Sequence[int]):
        super().__init__()

        if cfg.method not in REFT_METHODS:
            raise ValueError(
                f"ResidualReFTAdapter expected one of {sorted(REFT_METHODS)}, "
                f"got {cfg.method!r}"
            )
        if getattr(cfg, "reft_position_mode", "all") not in REFT_POSITION_MODES:
            raise ValueError(
                f"Unknown reft_position_mode={getattr(cfg, 'reft_position_mode', None)!r}; "
                f"choices={sorted(REFT_POSITION_MODES)}"
            )

        pyreft = _import_pyreft()

        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
            hidden_size = int(model.config.n_embd)
            self.arch_name = "gpt2"
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            layers = model.gpt_neox.layers
            hidden_size = int(model.config.hidden_size)
            self.arch_name = "pythia"
        else:
            raise ValueError(
                "Canonical PyReFT adapter only supports GPT-2-style "
                "model.transformer.h or Pythia/GPT-NeoX-style model.gpt_neox.layers."
            )

        base_param = next(model.parameters())
        if base_param.device.type == "cpu" and base_param.dtype in {torch.float16, torch.bfloat16}:
            model.float()
            base_param = next(model.parameters())

        n_layers = len(layers)
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")

        rank = int(cfg.reft_rank)
        if rank <= 0:
            raise ValueError("--reft_rank must be positive")
        if rank > hidden_size:
            raise ValueError(f"--reft_rank={rank} cannot exceed hidden_size={hidden_size}")

        intervention_dtype = base_param.dtype
        representations = [
            {
                "layer": layer_idx,
                "component": "block_output",
                "low_rank_dimension": rank,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=hidden_size,
                    low_rank_dimension=rank,
                    dropout=getattr(cfg, "reft_dropout", 0.05),
                    dtype=intervention_dtype,
                ),
            }
            for layer_idx in self.layer_indices
        ]
        reft_config = pyreft.ReftConfig(representations=representations)
        self.reft_model = pyreft.get_reft_model(
            model,
            reft_config,
            set_device=False,
            disable_model_grads=True,
        )

        for p in self.reft_model.model.parameters():
            p.requires_grad_(False)
        self._initialise_loreft_noop()
        for p in self.reft_model.interventions.parameters():
            p.requires_grad_(True)

    def _initialise_loreft_noop(self) -> None:
        """Set PyReFT LoReFT source map equal to its rotation so h' == h at init."""
        with torch.no_grad():
            for module in self.reft_model.interventions.modules():
                if not hasattr(module, "learned_source") or not hasattr(module, "rotate_layer"):
                    continue
                source = module.learned_source
                rotate_weight = module.rotate_layer.weight.detach()
                source.weight.copy_(rotate_weight.T.to(device=source.weight.device, dtype=source.weight.dtype))
                if source.bias is not None:
                    source.bias.zero_()

    @property
    def device(self) -> torch.device:
        return next(self.reft_model.parameters()).device

    def _unit_locations(self, seq_len: int) -> Dict[str, list[int]]:
        positions = reft_positions(
            seq_len=seq_len,
            mode=getattr(self.cfg, "reft_position_mode", "all"),
            prefix_positions=getattr(self.cfg, "reft_prefix_positions", 0),
            suffix_positions=getattr(self.cfg, "reft_suffix_positions", 0),
        )
        return {"base": positions}

    def set_peft_train_mode(self) -> None:
        self.reft_model.model.eval()
        self.reft_model.interventions.train()

    def set_peft_eval_mode(self) -> None:
        self.reft_model.model.eval()
        self.reft_model.interventions.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        _base_outputs, intervened_outputs = self.reft_model(
            {"input_ids": input_ids},
            unit_locations=self._unit_locations(input_ids.shape[1]),
            use_cache=False,
        )
        return intervened_outputs.logits

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        del input_ids
        params = [p for p in self.reft_model.interventions.parameters() if p.requires_grad]
        if not params:
            return {}
        vec = torch.cat([p.detach().reshape(-1).float().cpu() for p in params])
        return {
            "peft_param_abs_mean": float(vec.abs().mean().item()),
            "peft_param_l2_rms": float(vec.pow(2).mean().sqrt().item()),
        }
