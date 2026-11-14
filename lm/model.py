"""语言模型本体：词向量和 Engram 记忆 → 入口分成多条流 → 若干层 → 出口合成一条 → 输出层。"""
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .checkpointing import checkpoint_block
from .layers.attention import LayerCache
from .layers.block import Block
from .layers.engram import EngramMemory
from .layers.hyper import HyperConnections
from .layers.norm import RMSNorm
from .layers.rotary import rotary_tables
from .layers.streams import StreamEntry, StreamExit
from .loss import linear_cross_entropy


class Output(NamedTuple):
    logits: torch.Tensor | None      # (B, T, 词表大小)；给了 targets 时默认省略（只算损失），要的话传 return_logits=True
    loss: torch.Tensor | None        # 给了 targets 才有：交叉熵加上均衡损失
    balance_loss: torch.Tensor       # MoE 的均衡损失（已乘权重）；没有 MoE 时是 0


class DecodeState:
    """逐词生成时要记住的东西：每层注意力的键值缓存、已经走到第几个位置、最近几个词（Engram 查表用）。"""

    def __init__(self, n_layers: int):
        self.layers = [LayerCache() for _ in range(n_layers)]
        self.pos = 0
        self.history = None


class LanguageModel(nn.Module):
    loss_chunk = 4096  # 算损失时每块多少个词

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.checkpoint_policy = "full"
        self.checkpoint_block = checkpoint_block
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.embed_norm = RMSNorm(cfg.dim)
        self.memory = None
        if cfg.engram_orders:
            self.memory = EngramMemory(cfg.vocab_size, cfg.engram_orders, cfg.engram_heads, cfg.engram_head_dim, cfg.engram_rows)
            # 每个阶数的记忆各自投影成一份「来源」，和词向量一起交给入口
            self.memory_proj = nn.ModuleList(nn.Linear(cfg.engram_heads * cfg.engram_head_dim, cfg.dim, bias=False) for _ in cfg.engram_orders)
            self.memory_norms = nn.ModuleList(RMSNorm(cfg.dim) for _ in cfg.engram_orders)
        self.entry = StreamEntry(cfg.dim, cfg.n_streams, cfg.stream_key_dim)
        self.hyper = HyperConnections(2 * cfg.n_layers, cfg.n_streams, cfg.sinkhorn_iters)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.exit = StreamExit(cfg.dim, cfg.stream_key_dim)
        self.final_norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        cos, sin = rotary_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """矩阵按输入宽度的均匀分布；每个子层往残差流里写的那个投影从零开始；词向量和记忆表标准差 1（后面紧跟归一化）；
        归一化的系数保持 1。"""
        for name, p in self.named_parameters():
            if p.ndim == 1 and "norm" in name:
                continue
            leaf = name.rsplit(".", 2)[-2] if name.endswith(".weight") else name.rsplit(".", 1)[-1]
            if name in ("embed.weight", "memory.table.weight"):
                nn.init.normal_(p, std=1.0)
            elif name == "head.weight":
                nn.init.normal_(p, std=0.001)
            elif leaf in ("proj", "value") or ".w_out." in name or (leaf == "up" and "ffn" in name):
                nn.init.zeros_(p)
            elif leaf in ("pre_logit", "post_logit", "res_logit", "query"):
                continue  # 各自的模块里已经设好了
            elif p.ndim >= 2:
                bound = math.sqrt(3.0 / p.size(-1))
                nn.init.uniform_(p, -bound, bound)
            else:
                raise RuntimeError(f"参数 {name} 没有规定怎么初始化")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, state: DecodeState | None = None,
                return_logits: bool | None = None) -> Output:
        """idx: (B, T) 的词编号。targets 里等于 -1 的位置不算损失。state 只在逐词生成时传。
        return_logits 不写时：没给 targets 就返回整张输出，给了 targets 就只算损失（省下「词数 × 词表」那么大的一份显存）。"""
        B, T = idx.shape
        pos = state.pos if state is not None else 0
        if pos + T > self.cfg.max_seq_len:
            raise ValueError(f"序列长度 {pos + T} 超过了上限 {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[pos:pos + T], self.rope_sin[pos:pos + T]

        sources, memory = [self.embed_norm(self.embed(idx))], None
        if self.memory is not None:
            per_order = self.memory(idx, state.history if state is not None else None)  # (B, T, 阶数, 宽)
            sources += [norm(proj(per_order[:, :, o])) for o, (proj, norm) in enumerate(zip(self.memory_proj, self.memory_norms))]
            memory = per_order.flatten(2)
        x = self.entry(torch.stack(sources, dim=2))  # (B, T, 流数, d)

        maps = self.hyper.mappings(x.dtype)
        balance = x.new_zeros((), dtype=torch.float32)
        for i, block in enumerate(self.blocks):
            # MoE 前向会更新路由偏置，不能直接重算；带生成缓存时也不能重算。
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled() and state is None and not self.cfg.moe_experts:
                layer_maps = tuple(m[2 * i:2 * i + 2] for m in maps)
                x, b = self.checkpoint_block(block, x, cos, sin, layer_maps, memory, self.checkpoint_policy)
            else:
                x, b = block(x, cos, sin, maps, memory, state.layers[i] if state is not None else None)
            balance = balance + b.float()
        balance = self.cfg.moe_balance_loss * balance

        hidden = self.final_norm(self.exit(x))
        want_logits = targets is None if return_logits is None else return_logits
        logits = self.head(hidden).float() if want_logits else None
        loss = None
        if targets is not None:
            if logits is None:  # CUDA 上装了 Liger 用融合的输出层加交叉熵，否则分块算；两者结果和一次算完相同
                loss = linear_cross_entropy(hidden, self.head.weight, targets, chunk=self.loss_chunk) + balance
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1) + balance
        if state is not None:
            state.pos = pos + T
            if self.memory is not None and self.memory.history_len:
                past = state.history if state.history is not None else idx.new_full((B, self.memory.history_len), self.memory.pad_id)
                state.history = torch.cat([past, idx], dim=1)[:, -self.memory.history_len:]
        return Output(logits, loss, balance)

    def new_state(self) -> DecodeState:
        return DecodeState(self.cfg.n_layers)

    def param_breakdown(self) -> dict:
        """参数量按部件细分，每个参数恰好算进一项。"""
        rules = [("embed.weight", "词向量"), ("head.weight", "输出层"), ("memory.table", "记忆表"), ("norm", "归一化系数"), ("hyper.", "多流残差系数"),
                 (".attn.", "注意力"), (".ffn.shared.", "共享专家"), (".ffn.w_", "路由专家"), (".ffn.router", "专家路由器"), (".ffn.", "前馈层"),
                 (".engram.", "记忆注入（值和键的投影）"), ("memory_proj", "记忆来源投影"), ("entry.", "流入口"), ("exit.", "流出口")]
        if self.cfg.moe_latent_dim:
            rules[9] = (".ffn.", "专家的压缩和还原")
        counts = {label: 0 for _, label in rules}
        for name, p in self.named_parameters():
            counts[next(label for key, label in rules if key in name)] += p.numel()
        return {k: v for k, v in counts.items() if v}

    def param_report(self) -> dict:
        """参数量分几块数：主干、词向量、输出层、记忆表；有 MoE 时另外给出每个词实际用到的主干参数量。"""
        counts = {"词向量": 0, "输出层": 0, "记忆表": 0, "主干": 0}
        experts = 0
        for name, p in self.named_parameters():
            key = {"embed.weight": "词向量", "head.weight": "输出层", "memory.table.weight": "记忆表"}.get(name, "主干")
            counts[key] += p.numel()
            if ".w_in." in name or ".w_out." in name:
                experts += p.numel()
        counts["合计"] = sum(counts.values())
        if self.cfg.moe_experts:
            counts["主干（每个词实际用到）"] = counts["主干"] - experts + experts * self.cfg.moe_top_k // self.cfg.moe_experts
        return counts
