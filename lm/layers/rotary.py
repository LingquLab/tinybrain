"""旋转位置编码（RoPE）：把每个头的向量两两配对，按位置转一个角度。"""
import torch

try:
    from liger_kernel.ops.rope import LigerRopeFunction
except ImportError:
    LigerRopeFunction = None


def rotary_tables(head_dim: int, seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状都是 (seq_len, head_dim // 2)。"""
    freqs = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, head_dim)；cos、sin: (T, head_dim // 2)，是这 T 个位置对应的那一段。"""
    half = x.size(-1) // 2
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)  # 半精度下不能让表把结果抬回单精度
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


def apply_rotary_qk(q, k, cos, sin):
    # With T=1 the transposed contiguous tensor can have a singleton stride
    # smaller than a batch row. Liger uses that stride to locate the next row.
    batched_decode = q.size(0) > 1 and q.size(-2) == 1
    if q.is_cuda and q.dtype != torch.float64 and LigerRopeFunction is not None and not batched_decode:
        # 本项目用负角度约定；Liger 的 Llama 约定取相反的 sin。
        return LigerRopeFunction.apply(q, k, cos.to(q.dtype), -sin.to(q.dtype), None, 1, q.size(-1))
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
