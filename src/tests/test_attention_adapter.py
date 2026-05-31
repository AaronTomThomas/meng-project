from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

from experiments.router_development.attention_adapter.adapters.akaza_adapters import GPT2AKAZAAdapter, PythiaAKAZAAdapter
from experiments.router_development.attention_adapter.config import AKAZAFreeZConfig, AdapterMethod, config_from_values
from experiments.router_development.attention_adapter.trainer import TrainableParameters


def _gpt2_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        n_layer=3,
        n_head=2,
        n_embd=16,
        n_positions=32,
        vocab_size=50,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval()


def _pythia_model() -> GPTNeoXForCausalLM:
    config = GPTNeoXConfig(
        num_hidden_layers=3,
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        vocab_size=50,
        max_position_embeddings=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return GPTNeoXForCausalLM(config).eval()


def _cfg(model_family: str) -> AKAZAFreeZConfig:
    return AKAZAFreeZConfig(
        model_family=model_family,
        bottleneck_dim=2,
        adapter_dropout=0.0,
        output_scale=0.05,
    )


def _input_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 7, 3, 4, 11, 5, 2, 9, 6, 8, 10, 12],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        ],
        dtype=torch.long,
    )


def _make_adapter_nonzero(adapter: torch.nn.Module) -> None:
    with torch.no_grad():
        for module in adapter.adapters.values():
            module.up.weight.fill_(0.01)
            module.up.bias.fill_(0.01)


def test_gpt2_zero_delta_matches_baseline_and_hooks_are_transient() -> None:
    torch.manual_seed(0)
    model = _gpt2_model()
    wrapped = GPT2AKAZAAdapter(model=model, cfg=_cfg("gpt2"), layer_indices=[1, 2])
    input_ids = _input_ids()

    with torch.no_grad():
        baseline_before = model(input_ids, use_cache=False).logits
        wrapped_logits = wrapped(input_ids)
        baseline_after = model(input_ids, use_cache=False).logits

    torch.testing.assert_close(wrapped_logits, baseline_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(baseline_after, baseline_before, rtol=0.0, atol=0.0)


def test_pythia_zero_delta_matches_baseline_and_hooks_are_transient() -> None:
    torch.manual_seed(0)
    model = _pythia_model()
    wrapped = PythiaAKAZAAdapter(model=model, cfg=_cfg("pythia"), layer_indices=[1, 2])
    input_ids = _input_ids()

    with torch.no_grad():
        baseline_before = model(input_ids, use_cache=False).logits
        wrapped_logits = wrapped(input_ids)
        baseline_after = model(input_ids, use_cache=False).logits

    torch.testing.assert_close(wrapped_logits, baseline_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(baseline_after, baseline_before, rtol=0.0, atol=0.0)


def test_nonzero_attention_adapters_change_logits_without_editing_base_model() -> None:
    torch.manual_seed(0)
    input_ids = _input_ids()

    for model in [_gpt2_model(), _pythia_model()]:
        family = "gpt2" if isinstance(model, GPT2LMHeadModel) else "pythia"
        wrapped = (
            GPT2AKAZAAdapter(model=model, cfg=_cfg(family), layer_indices=[1, 2])
            if family == "gpt2"
            else PythiaAKAZAAdapter(model=model, cfg=_cfg(family), layer_indices=[1, 2])
        )
        _make_adapter_nonzero(wrapped)

        with torch.no_grad():
            baseline_before = model(input_ids, use_cache=False).logits
            wrapped_logits = wrapped(input_ids)
            baseline_after = model(input_ids, use_cache=False).logits

        assert not torch.allclose(wrapped_logits, baseline_before)
        torch.testing.assert_close(baseline_after, baseline_before, rtol=0.0, atol=0.0)


def test_gradients_only_reach_attention_adapter_params() -> None:
    torch.manual_seed(0)
    input_ids = _input_ids()

    for model in [_gpt2_model(), _pythia_model()]:
        family = "gpt2" if isinstance(model, GPT2LMHeadModel) else "pythia"
        wrapped = (
            GPT2AKAZAAdapter(model=model, cfg=_cfg(family), layer_indices=[1, 2])
            if family == "gpt2"
            else PythiaAKAZAAdapter(model=model, cfg=_cfg(family), layer_indices=[1, 2])
        )
        wrapped.set_peft_train_mode()

        logits = wrapped(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()

        assert any(p.grad is not None for p in wrapped.adapters.parameters())
        assert all(p.grad is None for p in wrapped.model.parameters())


def test_trainable_state_load_rejects_missing_adapter_tensors() -> None:
    torch.manual_seed(0)
    model = _gpt2_model()
    wrapped = GPT2AKAZAAdapter(model=model, cfg=_cfg("gpt2"), layer_indices=[1, 2])
    scope = TrainableParameters(params=[], frozen_before_training={}, check_frozen=False)
    state = scope.trainable_state_dict(wrapped)
    dropped_name = next(iter(state))
    del state[dropped_name]

    with pytest.raises(KeyError, match="missing trainable parameter names"):
        scope.load_trainable_state_dict(wrapped, state)


def test_direft_is_not_an_active_adapter_method() -> None:
    assert "direft" not in {method.value for method in AdapterMethod}
    with pytest.raises(ValueError, match="Unknown method"):
        config_from_values("direft")
