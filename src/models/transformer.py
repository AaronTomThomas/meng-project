import math 
import torch

import torch.nn as nn
from torch import Tensor

from models.learners import SoftmaxKernelLearner


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        L = x.size(1)
        return x + self.pe[:L, :].unsqueeze(0)


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class AttentionHead(nn.Module):
    def __init__(self, dk, dv, d_model):
        super().__init__()
        self.Q = nn.Linear(d_model, dk)
        self.K = nn.Linear(d_model, dk)
        self.V = nn.Linear(d_model, dv)
        self.learner = SoftmaxKernelLearner(dk)

    def forward(self, x_q: Tensor, x_kv: Tensor | None = None, attn_mask=None, return_weights=False):
        if x_kv is None:
            x_kv = x_q  # self-attn

        q = self.Q(x_q)     # [B, Lq, dk]
        k = self.K(x_kv)    # [B, Lk, dk]
        v = self.V(x_kv)    # [B, Lk, dv]
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, Lq, Lk] -> broadcast over batch
        out, A = self.learner(q, k, v, attn_mask)  # expect A: [B, Lq, Lk]
        return (out, A) if return_weights else out




class MHA(nn.Module):
    def __init__(self, d_model, dk, dv, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dk, dv, d_model) for _ in range(n_heads)])
        self.Wo = nn.Linear(n_heads * dv, d_model)

    def forward(self, x_q: Tensor, x_kv: Tensor | None = None, attn_mask=None, return_weights=False):
        if return_weights:
            outs, weights = [], []
            for head in self.heads:
                out, A = head(x_q, x_kv=x_kv, attn_mask=attn_mask, return_weights=True)
                outs.append(out)
                weights.append(A)
            y = self.Wo(torch.cat(outs, dim=-1))      # [B, Lq, d_model]
            A_all = torch.stack(weights, dim=1)       # [B, n_heads, Lq, Lk]
            return y, A_all
        else:
            outs = [head(x_q, x_kv=x_kv, attn_mask=attn_mask) for head in self.heads]
            return self.Wo(torch.cat(outs, dim=-1))
    


class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x : Tensor):
        inner = self.activation(self.l1(x))
        return self.l2(inner)


class EncoderLayer(nn.Module): 
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MHA(d_model, dk=d_model//n_heads, dv=d_model//n_heads, n_heads=n_heads)
        self.ff = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask=None, return_weights=False):
        # pre-LN
        if return_weights:
            attn_out, A = self.self_attn(self.norm1(x), attn_mask=attn_mask, return_weights=True)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x, A
        else:
            x = x + self.dropout(self.self_attn(self.norm1(x), attn_mask=attn_mask))
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x
        

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        dk = d_model // n_heads
        dv = d_model // n_heads

        self.self_attn  = MHA(d_model, dk=dk, dv=dv, n_heads=n_heads)
        self.cross_attn = MHA(d_model, dk=dk, dv=dv, n_heads=n_heads)

        self.ff = FeedForwardSubLayer(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor | None = None, tgt_mask=None, src_mask=None, return_weights=False):
        # 1) masked self-attn
        if return_weights:
            sa_out, A_sa = self.self_attn(self.norm1(x), attn_mask=tgt_mask, return_weights=True)
        else:
            sa_out = self.self_attn(self.norm1(x), attn_mask=tgt_mask)
        x = x + self.dropout(sa_out)

        # 2) cross-attn (optional if memory provided)
        A_ca = None
        if memory is not None:
            if return_weights:
                ca_out, A_ca = self.cross_attn(self.norm2(x), x_kv=memory, attn_mask=src_mask, return_weights=True)
            else:
                ca_out = self.cross_attn(self.norm2(x), x_kv=memory, attn_mask=src_mask)
            x = x + self.dropout(ca_out)

        # 3) FFN
        ff_out = self.ff(self.norm3(x))
        x = x + self.dropout(ff_out)

        return (x, A_sa, A_ca) if return_weights else x

def causal_mask(L: int, device=None, dtype=None):
    m = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)  # block future



class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # optional tie
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(self, x: Tensor, memory: Tensor | None = None, tgt_mask=None, src_mask=None, return_weights=False):
        # tgt_idx: [B, Lt]
        B, Lt = x.shape
        x = self.embedding(x)
        x = self.positional_encoding(x)

        if tgt_mask is None:
            tgt_mask = causal_mask(Lt, device=x.device, dtype=x.dtype)

        if not return_weights:
            for layer in self.layers:
                x = layer(x, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask)
            x = self.final_norm(x)
            return self.lm_head(x)  # [B, Lt, vocab]

        layer_sa, layer_ca = [], []
        for layer in self.layers:
            x, A_sa, A_ca = layer(x, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask, return_weights=True)
            layer_sa.append(A_sa)
            layer_ca.append(A_ca)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        # sa: [n_layers, B, n_heads, Lt, Lt]
        # ca: [n_layers, B, n_heads, Lt, Ls] or None per layer
        return logits, torch.stack(layer_sa, dim=0), layer_ca

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        max_ctx = self.positional_encoding.pe.size(0)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_ctx:]  # crop context

            logits = self(idx_cond)       # [B,T,V]
            logits = logits[:, -1, :]     # last step

            if temperature is None or temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size, dropout, max_seq_length):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for i in range(n_layers)])
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.final_norm = nn.LayerNorm(d_model)
        

    def forward(self, x: Tensor, attn_mask=None, return_weights=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        if not return_weights:
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask)
            return self.final_norm(x)

        layer_weights = []
        for layer in self.layers:
            x, A = layer(x, attn_mask=attn_mask, return_weights=True)
            layer_weights.append(A)  # each A: [B, n_heads,L,L]
        # stack: [num_layers, B, n_heads, L, L]
        return self.final_norm(x), torch.stack(layer_weights, dim=0)

