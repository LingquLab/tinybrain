"""快速生成：键值缓存预先按最大长度分配好，每一步原地写入，形状完全固定。

CUDA 上逐词那一步用 torch.compile(mode="reduce-overhead") 编译，它会把整步录成 CUDA graph，
省掉 Python 和逐个内核启动的开销（gpt-fast 的做法）。CPU、MPS 上不编译，结果和 CUDA 上一样。

只用模型各层的参数，不改 lm 里的代码；测试保证它和普通前向的输出一致。
滑动窗口的层用遮罩实现：缓存仍按全长分配，只让查询看窗口里的那些位置。
混合专家的模型每步各专家分到的词数不同，没法录成固定的图，所以这里不编译它们（仍然用静态缓存）。
"""
import torch
import torch.nn.functional as F

from lm import LanguageModel
from lm.layers.hyper import read_streams, write_streams
from lm.layers.rotary import apply_rotary

from .generate import sample_next


class StaticState:
    """一批序列（每行长度相同）的固定形状缓存。"""

    def __init__(self, model: LanguageModel, batch: int, dtype: torch.dtype, device: torch.device):
        cfg = model.cfg
        shape = (batch, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        m = model.memory.history_len if model.memory is not None else 0
        self.history = torch.full((batch, m), model.memory.pad_id if model.memory is not None else 0, dtype=torch.long, device=device)
        self.positions = torch.arange(cfg.max_seq_len, device=device)


def _attention(attn, h, cos, sin, k_cache, v_cache, pos, positions):
    """h: (B, T, d)；pos: 这 T 个词的位置（长度 T 的张量）。把新的 k、v 写进缓存，查询看缓存里能看的位置。"""
    B, T, _ = h.shape
    q = attn.q(h).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k(h).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v(h).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary(attn.q_norm(q), cos, sin), apply_rotary(attn.k_norm(k), cos, sin)
    k_cache.index_copy_(2, pos, k.to(k_cache.dtype))
    v_cache.index_copy_(2, pos, v.to(v_cache.dtype))
    visible = positions[None, :] <= pos[:, None]                      # (T, 最大长度)
    if attn.window is not None:
        visible = visible & (positions[None, :] > pos[:, None] - attn.window)
    out = F.scaled_dot_product_attention(q, k_cache.to(q.dtype), v_cache.to(q.dtype), attn_mask=visible, enable_gqa=True)
    return attn.proj(out.transpose(1, 2).reshape(B, T, -1))


def forward_static(model: LanguageModel, idx: torch.Tensor, pos: torch.Tensor, state: StaticState) -> torch.Tensor:
    """idx: (B, T)，pos: (T,) 这几个词的位置。返回最后一个位置的 logits (B, 词表)，并更新 state。"""
    cos, sin = model.rope_cos[pos], model.rope_sin[pos]
    sources, memory = [model.embed_norm(model.embed(idx))], None
    if model.memory is not None:
        per_order = model.memory(idx, state.history)
        sources += [norm(proj(per_order[:, :, o])) for o, (proj, norm) in enumerate(zip(model.memory_proj, model.memory_norms))]
        memory = per_order.flatten(2)
        m = state.history.size(1)
        if m:
            state.history.copy_(torch.cat([state.history, idx], dim=1)[:, -m:])
    x = model.entry(torch.stack(sources, dim=2))
    pre, post, res = model.hyper.mappings(x.dtype)
    for i, block in enumerate(model.blocks):
        a, f = 2 * i, 2 * i + 1
        if block.engram is not None:
            x = block.engram(x, memory)
        h = _attention(block.attn, block.attn_norm(read_streams(x, pre[a])), cos, sin, state.k[i], state.v[i], pos, state.positions)
        x = write_streams(x, h, res[a], post[a])
        out = block.ffn(block.ffn_norm(read_streams(x, pre[f])))
        out = out[0] if isinstance(out, tuple) else out
        x = write_streams(x, out, res[f], post[f])
    return model.head(model.final_norm(model.exit(x[:, -1:]))).float()[:, -1]


class FastGenerator:
    """同一个模型反复生成时复用：编译过的那一步只编一次。"""

    def __init__(self, model: LanguageModel, compile: bool | None = None):
        self.model = model
        device = next(model.parameters()).device
        moe = bool(model.cfg.moe_experts)
        self.compiled = (device.type == "cuda" and not moe) if compile is None else compile
        self.step = torch.compile(forward_static, mode="reduce-overhead", fullgraph=False) if self.compiled else forward_static

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                 eos_id=None, generator: torch.Generator | None = None) -> torch.Tensor:
        """和 inference.generate.generate 一样的用法和结果。"""
        model, device = self.model, idx.device
        was_training = model.training
        model.eval()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        B, T = idx.shape
        state = StaticState(model, B, dtype, device)
        stops = None if eos_id is None else torch.tensor([eos_id] if isinstance(eos_id, int) else list(eos_id), device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with autocast:
            logits = forward_static(model, idx, torch.arange(T, device=device), state)  # 提示整段一次算完（不编译）
            for t in range(min(max_new_tokens, model.cfg.max_seq_len - T)):
                step = sample_next(logits, temperature, top_k, top_p, generator)
                idx = torch.cat([idx, step], dim=1)
                if stops is not None:
                    done |= torch.isin(step.squeeze(1), stops)
                    if done.all():
                        break
                if T + t + 1 >= model.cfg.max_seq_len:
                    break
                logits = self.step(model, step, torch.tensor([T + t], device=device), state).clone()
        model.train(was_training)
        return idx


def fast_generate(model: LanguageModel, idx: torch.Tensor, max_new_tokens: int, **kw) -> torch.Tensor:
    return FastGenerator(model).generate(idx, max_new_tokens, **kw)
