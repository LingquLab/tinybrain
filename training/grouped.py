"""CUDA 分组线性层。ends[e] 是专家 e 最后一行之后的位置，始终留在设备上。"""
import torch
import triton
import triton.language as tl

from .segments import linear  # 保留 training.grouped.linear 入口


@triton.jit
def _project(X, W, Ends, Y, D: tl.constexpr, H: tl.constexpr, E: tl.constexpr,
             S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, P: tl.constexpr):
    tile = tl.program_id(0)
    ids = tl.arange(0, P)
    ends = tl.load(Ends + ids, ids < E, 0)
    # 每个专家预留一个边界块，网格大小只依赖总行数，允许专家为空。
    e = tl.sum(((ids < E) & (tile >= ends // 64 + ids + 1)).to(tl.int32), 0)
    start = tl.load(Ends + e - 1, e > 0, 0)
    stop = tl.load(Ends + e)
    first = start + (tile - start // 64 - e) * 64
    if first < stop:
        row = first + tl.arange(0, 64)
        col = tl.program_id(1) * 64 + tl.arange(0, 64)
        acc = tl.full((64, 64), 0, tl.float32)
        for i in range(tl.cdiv(D, 32)):
            k = i * 32 + tl.arange(0, 32)
            x = tl.load(X + row[:, None] * D + k[None, :], (row[:, None] < stop) & (k[None, :] < D), 0)
            w = tl.load(W + e * S0 + k[:, None] * S2 + col[None, :] * S1,
                        (k[:, None] < D) & (col[None, :] < H), 0)
            acc = tl.dot(x, w, acc, input_precision="ieee")
        tl.store(Y + row[:, None] * H + col[None, :], acc, (row[:, None] < stop) & (col[None, :] < H))


@triton.jit
def _weight(G, X, Ends, DW, H: tl.constexpr, D: tl.constexpr):
    e = tl.program_id(2)
    start = tl.load(Ends + e - 1, e > 0, 0)
    stop = tl.load(Ends + e)
    h = tl.program_id(0) * 64 + tl.arange(0, 64)
    d = tl.program_id(1) * 64 + tl.arange(0, 64)
    acc = tl.full((64, 64), 0, tl.float32)
    for i in range(tl.cdiv(stop - start, 32)):
        row = start + i * 32 + tl.arange(0, 32)
        g = tl.load(G + row[None, :] * H + h[:, None], (row[None, :] < stop) & (h[:, None] < H), 0)
        x = tl.load(X + row[:, None] * D + d[None, :], (row[:, None] < stop) & (d[None, :] < D), 0)
        acc = tl.dot(g, x, acc, input_precision="ieee")
    tl.store(DW + (e * H + h[:, None]) * D + d[None, :], acc, (h[:, None] < H) & (d[None, :] < D))


def project(x: torch.Tensor, w: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    rows, d = x.shape
    e, h, _ = w.shape
    y = x.new_empty((rows, h))
    _project[(rows // 64 + e, triton.cdiv(h, 64))](
        x.contiguous(), w, ends, y, d, h, e, *w.stride(), triton.next_power_of_2(e), num_warps=4, num_stages=2)
    return y


def weight_grad(g: torch.Tensor, x: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    h, d = g.size(1), x.size(1)
    dw = x.new_empty((ends.numel(), h, d))
    _weight[(triton.cdiv(h, 64), triton.cdiv(d, 64), ends.numel())](
        g.contiguous(), x.contiguous(), ends, dw, h, d, num_warps=4, num_stages=2)
    return dw
