from typing import Any, Dict, Sequence

from experiments.router_development.attention_adapter.adapters.akaza_adapters import AdapterModel
from experiments.router_development.attention_adapter.adapters.utils import delta_stats
from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig
import torch
from torch import nn
import torch.nn.functional as F

REFT_INTERVENTION_METHODS = {"loreft"}
REFT_POSITION_MODES = {"all", "prefix", "suffix", "prefix_suffix"}

def extract_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    if hasattr(output, "hidden_states"):
        return output.hidden_states
    raise TypeError(f"Unsupported block output type: {type(output)}")


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    if hasattr(output, "hidden_states"):
        output.hidden_states = hidden
        return output
    raise TypeError(f"Unsupported block output type: {type(output)}")


class ReFTIntervention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        method: str,
        dropout: float,
        output_scale: float,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("--reft_rank must be positive")
        if rank > hidden_size:
            raise ValueError(f"--reft_rank={rank} cannot exceed hidden_size={hidden_size}")
        if method not in REFT_INTERVENTION_METHODS:
            raise ValueError(f"Unknown ReFT method={method!r}; choices={sorted(REFT_INTERVENTION_METHODS)}")

        self.method = method
        self.output_scale = float(output_scale)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        basis = torch.zeros(rank, hidden_size)
        basis[:, :rank] = torch.eye(rank)

        self.r = nn.Parameter(basis.clone())
        self.w = nn.Parameter(basis.clone())
        self.bias = nn.Parameter(torch.zeros(rank))

    def projected_r(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.r.transpose(0, 1), mode="reduced")
        return q.transpose(0, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_f = hidden.float()

        r = self.projected_r().float()
        w = self.w.float()

        source = F.linear(hidden_f, w, self.bias.float())
        base = F.linear(hidden_f, r)
        delta_low_rank = self.dropout(source - base)
        delta = F.linear(delta_low_rank, r.transpose(0, 1))

        return hidden + (self.output_scale * delta).to(dtype=hidden.dtype, device=hidden.device)
    

def reft_position_mask(
    *,
    seq_len: int,
    mode: str,
    prefix_positions: int,
    suffix_positions: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    if mode == "all":
        mask[:] = True
    elif mode == "prefix":
        n = prefix_positions if prefix_positions > 0 else seq_len
        mask[: min(n, seq_len)] = True
    elif mode == "suffix":
        n = suffix_positions if suffix_positions > 0 else seq_len
        mask[max(0, seq_len - n):] = True
    elif mode == "prefix_suffix":
        if prefix_positions <= 0 and suffix_positions <= 0:
            mask[:] = True
        else:
            mask[: min(prefix_positions, seq_len)] = True
            if suffix_positions > 0:
                mask[max(0, seq_len - suffix_positions):] = True
    else:
        raise ValueError(
            f"Unknown reft_position_mode={mode!r}; "
            f"choices={sorted(REFT_POSITION_MODES)}"
        )

    return mask.view(1, seq_len, 1)

class ResidualReFTAdapter(AdapterModel):
    """
    Residual-stream ReFT adapter.

    Supports LoReFT residual-stream interventions.

    GPT-2:
      hooks model.transformer.h[layer_idx]

    Pythia/GPT-NeoX:
      hooks model.gpt_neox.layers[layer_idx]
    """

    def __init__(self, *, model: nn.Module, cfg: AdapterFineTuneConfig, layer_indices: Sequence[int]):
        super().__init__()

        if str(cfg.method) != "loreft":
            raise ValueError(
                "ResidualReFTAdapter expected method='loreft', "
                f"got {cfg.method!r}"
            )

        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)
        self._latest_deltas: Dict[int, torch.Tensor] = {}

        for p in self.model.parameters():
            p.requires_grad_(False)

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
            hidden_size = int(model.config.n_embd)
            self.arch_name = "gpt2"
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            self.layers = model.gpt_neox.layers
            hidden_size = int(model.config.hidden_size)
            self.arch_name = "pythia"
        else:
            raise ValueError(
                "ResidualReFTAdapter only supports GPT-2-style "
                "model.transformer.h or Pythia/GPT-NeoX-style model.gpt_neox.layers."
            )

        n_layers = len(self.layers)
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")

        self.interventions = nn.ModuleDict(
            {
                str(layer_idx): ReFTIntervention(
                    hidden_size=hidden_size,
                    rank=cfg.reft_rank,
                    method=cfg.method,
                    dropout=getattr(cfg, "reft_dropout", 0.05),
                    output_scale=getattr(cfg, "reft_output_scale", 1.0),
                )
                for layer_idx in self.layer_indices
            }
        )

        for p in self.interventions.parameters():
            p.requires_grad_(True)

    @property
    def device(self) -> torch.device:
        return next(self.interventions.parameters()).device

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        self.interventions.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()
        self.interventions.eval()

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> Any:
            hidden = extract_hidden(output)
            edited = self.interventions[str(layer_idx)](hidden)

            mask = reft_position_mask(
                seq_len=hidden.shape[1],
                mode=getattr(self.cfg, "reft_position_mode", "all"),
                prefix_positions=getattr(self.cfg, "reft_prefix_positions", 0),
                suffix_positions=getattr(self.cfg, "reft_suffix_positions", 0),
                device=hidden.device,
            )

            new_hidden = torch.where(mask, edited, hidden)
            self._latest_deltas[layer_idx] = (new_hidden - hidden).detach()

            return replace_hidden(output, new_hidden)

        return hook

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        self._latest_deltas.clear()

        handles: list[torch.utils.hooks.RemovableHandle] = []

        try:
            for layer_idx in self.layer_indices:
                layer = self.layers[layer_idx]
                handles.append(layer.register_forward_hook(self._make_layer_hook(layer_idx)))

            return self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}

        self.set_peft_eval_mode()
        _ = self(input_ids)

        return delta_stats(self._latest_deltas)


class ZSpaceLoReFTAdapter(AdapterModel):
    """
    Pre-output-projection attention-output LoReFT adapter.

    GPT-2:
      hooks model.transformer.h[layer_idx].attn.c_proj and edits its input z.

    Pythia/GPT-NeoX:
      hooks model.gpt_neox.layers[layer_idx].attention.dense and edits its input z.
    """

    def __init__(self, *, model: nn.Module, cfg: AdapterFineTuneConfig, layer_indices: Sequence[int]):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)
        self._latest_deltas: Dict[int, torch.Tensor] = {}

        for p in self.model.parameters():
            p.requires_grad_(False)

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
            hidden_size = int(model.config.n_embd)
            self.arch_name = "gpt2"
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            self.layers = model.gpt_neox.layers
            hidden_size = int(model.config.hidden_size)
            self.arch_name = "pythia"
        else:
            raise ValueError(
                "ZSpaceLoReFTAdapter only supports GPT-2-style "
                "model.transformer.h or Pythia/GPT-NeoX-style model.gpt_neox.layers."
            )

        n_layers = len(self.layers)
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")

        self.interventions = nn.ModuleDict(
            {
                str(layer_idx): ReFTIntervention(
                    hidden_size=hidden_size,
                    rank=cfg.reft_rank,
                    method="loreft",
                    dropout=getattr(cfg, "reft_dropout", 0.05),
                    output_scale=getattr(cfg, "reft_output_scale", 1.0),
                )
                for layer_idx in self.layer_indices
            }
        )

        for p in self.interventions.parameters():
            p.requires_grad_(True)

    @property
    def device(self) -> torch.device:
        return next(self.interventions.parameters()).device

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        self.interventions.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()
        self.interventions.eval()

    def _make_projection_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            z = inputs[0]
            edited = self.interventions[str(layer_idx)](z)

            mask = reft_position_mask(
                seq_len=z.shape[1],
                mode=getattr(self.cfg, "reft_position_mode", "all"),
                prefix_positions=getattr(self.cfg, "reft_prefix_positions", 0),
                suffix_positions=getattr(self.cfg, "reft_suffix_positions", 0),
                device=z.device,
            )

            new_z = torch.where(mask, edited, z)
            self._latest_deltas[layer_idx] = (new_z - z).detach()
            return (new_z,) + inputs[1:]

        return hook

    def _projection_module(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch_name == "gpt2":
            return layer.attn.c_proj
        if self.arch_name == "pythia":
            return layer.attention.dense
        raise RuntimeError(f"Unsupported arch_name={self.arch_name!r}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        self._latest_deltas.clear()

        handles: list[torch.utils.hooks.RemovableHandle] = []

        try:
            for layer_idx in self.layer_indices:
                projection = self._projection_module(layer_idx)
                handles.append(projection.register_forward_pre_hook(self._make_projection_pre_hook(layer_idx)))

            return self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}

        self.set_peft_eval_mode()
        _ = self(input_ids)

        return delta_stats(self._latest_deltas)
