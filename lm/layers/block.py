import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import Attention, LayerCache
from .engram import EngramGate
from .feedforward import FeedForward
from .hyper import read_normalized_streams, write_streams
from .moe import MoE
from .norm import RMSNorm


class Block(nn.Module):
    """一层：（可选的记忆注入）→ 注意力 → 前馈。两个子层各自通过多流残差读写 n 条流。"""

    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.index = index
        self.engram = EngramGate(cfg.engram_dim, cfg.dim, cfg.n_streams) if index in cfg.engram_layers else None
        self.long = cfg.is_long(index)  # 看全文的层：位置编码可以单独设置（global_rope、global_rope_base）
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.window_of(index), rope=cfg.uses_rope(index), sink=cfg.attn_sink)
        self.attn_norm, self.ffn_norm = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        if cfg.is_moe_layer(index):
            self.ffn = MoE(cfg.dim, cfg.moe_experts, cfg.moe_top_k, int(cfg.moe_hidden_mult * cfg.dim), int(cfg.moe_shared_mult * cfg.dim),
                           cfg.moe_latent_dim, cfg.moe_balance_rate, cfg.moe_packed, cfg.moe_clip)
        else:
            self.ffn = FeedForward(cfg.dim, int(cfg.ffn_mult * cfg.dim))

    def forward(self, x, cos, sin, maps, memory=None, cache: LayerCache | None = None, return_counts=False, docs=None):
        """maps 可传全模型的系数，或只传这一层的两组系数。return_counts 时额外返回本层的命中数。
        docs 是 DocMask（每个位置属于哪篇文档），只在开了 doc_eos 时传。"""
        pre, post, res = maps
        a, f = (0, 1) if pre.size(0) == 2 else (2 * self.index, 2 * self.index + 1)
        if self.engram is not None:
            x = self.engram(x, memory)
        x = write_streams(x, self.attn(read_normalized_streams(x, pre[a], self.attn_norm), cos, sin, cache, docs), res[a], post[a])
        value = read_normalized_streams(x, pre[f], self.ffn_norm)
        counts = None
        if isinstance(self.ffn, MoE):
            out, balance_loss, counts = self.ffn(value, return_counts=True)
            if not return_counts and cache is None:
                self.ffn.record_load(counts)
        else:
            out, balance_loss = self.ffn(value), x.new_zeros(())
        x = write_streams(x, out, res[f], post[f])
        return (x, balance_loss, counts) if return_counts else (x, balance_loss)
