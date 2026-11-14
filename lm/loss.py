"""输出层加交叉熵。CUDA 上装了 Liger 就用它的融合内核；否则一块一块地过输出层。两者结果都和一次算完相同。

整张输出是「词数 × 词表」那么大：16 × 2048 个词、32768 的词表，单精度一份就是 4 GB，一次算完时前向和反向里会同时有好几份。
分块以后同时存在的只有一块的输出。
"""
import torch
import torch.nn.functional as F

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
except ImportError:
    LigerFusedLinearCrossEntropyFunction = None


def linear_cross_entropy(hidden, weight, targets, chunk: int = 4096, chunk_mem_const: int = 8):
    """hidden: (..., d)，weight: (词表, d)，targets 里等于 -1 的位置不算。返回平均损失。"""
    hidden = hidden.reshape(-1, hidden.size(-1))
    targets = targets.reshape(-1)
    if hidden.is_cuda and LigerFusedLinearCrossEntropyFunction is not None:
        return LigerFusedLinearCrossEntropyFunction.apply(
            hidden, weight, targets, None, None, -1, 0.0, 0.0, "mean", None, False,
            None, False, False, False, None, None, chunk_mem_const
        )[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        logits = F.linear(hidden[start:start + chunk], weight).float()
        total = total + F.cross_entropy(logits, targets[start:start + chunk], ignore_index=-1, reduction="sum")
    return total / (targets != -1).sum()
