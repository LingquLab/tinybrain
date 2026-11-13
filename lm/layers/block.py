import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import Attention, LayerCache
from .engram import EngramGate
from .feedforward import FeedForward
from .hyper import read_streams, write_streams
from .moe import MoE
from .norm import RMSNorm


class Block(nn.Module):
    """一层：（可选的记忆注入）→ 注意力 → 前馈。两个子层各自通过多流残差读写 n 条流。"""

    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.index = index
        self.engram = EngramGate(cfg.engram_dim, cfg.dim, cfg.n_streams) if index in cfg.engram_layers else None
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.window_of(index))
        self.attn_norm, self.ffn_norm = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        if cfg.moe_experts:
            self.ffn = MoE(cfg.dim, cfg.moe_experts, cfg.moe_top_k, int(cfg.moe_hidden_mult * cfg.dim), int(cfg.moe_shared_mult * cfg.dim),
                           cfg.moe_latent_dim, cfg.moe_balance_rate)
        else:
            self.ffn = FeedForward(cfg.dim, int(cfg.ffn_mult * cfg.dim))

    def forward(self, x, cos, sin, maps, memory=None, cache: LayerCache | None = None):
        """x: (B, T, n, d)；maps: 全模型的 (pre, post, res)，这一层用其中第 2i 和 2i+1 组。返回 (x, 均衡损失)。"""
        pre, post, res = maps
        a, f = 2 * self.index, 2 * self.index + 1
        if self.engram is not None:
            x = self.engram(x, memory)
        x = write_streams(x, self.attn(self.attn_norm(read_streams(x, pre[a])), cos, sin, cache), res[a], post[a])
        out = self.ffn(self.ffn_norm(read_streams(x, pre[f])))
        out, balance_loss = out if isinstance(out, tuple) else (out, x.new_zeros(()))
        return write_streams(x, out, res[f], post[f]), balance_loss
