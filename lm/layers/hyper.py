"""多流残差连接（mHC，静态版）。

普通残差只有一条流：x ← x + F(x)。这里有 n 条流，每个子层（注意力、前馈各算一个）有三组可学的系数：
  pre  (n)    读：把 n 条流加权合成一份输入，取值在 0 到 1 之间；
  post (n)    写：子层的输出按多大比例加回每条流，取值在 0 到 2 之间；
  res  (n×n)  混：流和流之间怎么互相搬运，是双随机矩阵（每行每列的和都是 1），信号总量不放大也不缩小。
更新式：x ← res·x + post ⊗ F(pre·x)。静态版的系数不看输入，整个模型所有子层的系数放在一起一次算完。
"""
import torch
import torch.nn as nn


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """把 exp(logits) 反复按列、按行归一，得到（近似的）双随机矩阵。在对数域里算，不怕数值溢出。"""
    log = logits.float()
    for _ in range(iters):
        log = log - log.logsumexp(dim=-2, keepdim=True)
        log = log - log.logsumexp(dim=-1, keepdim=True)
    return log.exp()


class HyperConnections(nn.Module):
    def __init__(self, n_sublayers: int, n_streams: int, iters: int = 20):
        super().__init__()
        self.iters = iters
        # 起点等价于普通残差：流各走各的（res 接近单位阵），第 s 个子层主要读第 s mod n 条流，输出写回所有流。
        focus = torch.eye(n_streams)[torch.arange(n_sublayers) % n_streams]
        self.pre_logit = nn.Parameter(4.0 * focus - 2.0)
        self.post_logit = nn.Parameter(torch.zeros(n_sublayers, n_streams))
        self.res_logit = nn.Parameter(5.0 * torch.eye(n_streams).repeat(n_sublayers, 1, 1))

    def mappings(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (pre, post, res)，形状 (子层数, n)、(子层数, n)、(子层数, n, n)。"""
        pre = torch.sigmoid(self.pre_logit.float())
        post = 2.0 * torch.sigmoid(self.post_logit.float())
        return pre.to(dtype), post.to(dtype), sinkhorn(self.res_logit, self.iters).to(dtype)


def read_streams(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """x: (B, T, n, d)，pre: (n) → (B, T, d)"""
    return torch.einsum("btnd,n->btd", x, pre)


def write_streams(x: torch.Tensor, out: torch.Tensor, res: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    """x: (B, T, n, d)，out: (B, T, d)，res: (n, n)，post: (n) → (B, T, n, d)"""
    return torch.einsum("mn,btnd->btmd", res, x) + post[:, None] * out[:, :, None, :]
