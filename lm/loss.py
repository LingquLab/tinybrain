"""输出层加交叉熵。

CUDA 上装了 Liger 就用它的融合内核；否则一块一块地过输出层，每块用下面的稳定写法算。

稳定写法（Barron、Gnanapragasam、Tschannen，A Fix for Softmax and Cross-Entropy，arXiv 2610.31642 v3）：
标准实现的梯度是 softmax − onehot。正确答案的领先幅度 m = z_t − LSE_{j≠t} z_j 超过精度的临界值时
（FP32 约 16.6，BF16 约 5.5），正确项的 softmax 舍入成 1，梯度变成 0，其余项的梯度还在，加起来不再是 0；
Adam 会把这种本该抵消的微小梯度放大成损失的尖峰。改成
    m = −LSE_{j≠t}(z_j − z_t)，   损失 = softplus(−m)
对它求导，正确项的梯度是 −sigmoid(−m)，在 m 很大时也能表示，所有梯度之和恰好为 0。PyTorch 对这个式子自动求出的就是这组梯度。
论文 v3 的勘误：除了正确答案其余都是负无穷时，旧写法的梯度是 NaN；这里把差值的下限夹在 −1e4（e^−1e4 在 FP32 里就是 0）。

整张输出是「词数 × 词表」那么大：16 × 2048 个词、32768 的词表，单精度一份就是 4 GB。分块以后同时存在的只有一块。
"""
import torch
import torch.nn.functional as F

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
except ImportError:
    LigerFusedLinearCrossEntropyFunction = None

FP32_CLIFF = 16.6  # FP32 下正确答案领先超过它，标准实现的正确项梯度就舍入成 0


def _margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """每个位置上正确答案的领先幅度 m = z_t − LSE_{j≠t} z_j。targets 必须都是合法的编号。"""
    logits = logits.float()
    t = targets.unsqueeze(-1)
    diff = logits - logits.gather(-1, t)
    others = diff.scatter(-1, t, float("-inf")).clamp(min=-1e4)
    return -torch.logsumexp(others, dim=-1)


def token_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """每个位置给目标词的对数概率 log p_t = −softplus(−m)，形状同 targets。和 log_softmax 再取目标一样，梯度在饱和时也对。"""
    return -F.softplus(-_margin(logits, targets))


def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1, reduction: str = "mean") -> torch.Tensor:
    """和 F.cross_entropy(logits, targets, ignore_index=ignore_index) 一样，只是正确答案把握极大时梯度不会算错。"""
    logits, targets = logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    valid = targets != ignore_index
    loss = -token_log_probs(logits, targets.clamp(min=0)) * valid
    total = loss.sum()
    return total / valid.sum() if reduction == "mean" else total


def margin_saturation(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1, cliff: float = FP32_CLIFF) -> float:
    """有多大比例的目标词，领先幅度已经超过精度的临界值（标准实现在这些位置上梯度会算错）。"""
    with torch.no_grad():
        logits, targets = logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        valid = targets != ignore_index
        if not valid.any():
            return 0.0
        m = _margin(logits[valid], targets[valid])
        return float((m > cliff).float().mean())


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
        logits = F.linear(hidden[start:start + chunk], weight)
        total = total + stable_cross_entropy(logits, targets[start:start + chunk], reduction="sum")
    return total / (targets != -1).sum()
