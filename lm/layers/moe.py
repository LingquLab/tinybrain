"""混合专家前馈层（只有大模型用）。

每个词由一个所有词共用的「共享专家」加上从 E 个专家里挑出的 top-k 个来处理。
  打分：sqrt(softplus(线性))，非负；选中的 k 个分数归一后当权重。
  均衡：每个专家有一个不参与求导的偏置，只在「挑谁」时加上——哪个专家活儿多就把它的偏置调低一点；
        另外返回一个很轻的均衡损失。偏置不在前向里改：每个真实训练前向累加命中数，优化器走完一步再由 apply_balance 调一次。
        这样激活重计算把前向再跑一遍时，挑出来的专家和第一次一模一样，梯度才对。
  LatentMoE：moe_latent_dim 大于 0 时，先把输入压到这个宽度，路由和专家都在压缩后的空间里算，最后再升回来。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import dispatch
from .feedforward import FeedForward


class MoE(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, hidden: int, shared_hidden: int, latent_dim: int = 0,
                 balance_rate: float = 1e-3, packed: bool = False, cap: float = 0.0):
        super().__init__()
        self.n_experts, self.top_k, self.balance_rate = n_experts, top_k, balance_rate
        width = latent_dim or dim
        self.packed, self.cap = packed, cap
        self.shared = FeedForward(dim, shared_hidden, cap)
        self.down = nn.Linear(dim, width, bias=False) if latent_dim else None
        self.up = nn.Linear(width, dim, bias=False) if latent_dim else None
        self.router = nn.Linear(width, n_experts, bias=False)
        if packed:
            self.w_in = nn.Parameter(torch.empty(n_experts, hidden, width))
            self.w_out = nn.Parameter(torch.zeros(n_experts, width, hidden))
        else:
            self.w_in = nn.ParameterList(nn.Parameter(torch.empty(hidden, width)) for _ in range(n_experts))
            self.w_out = nn.ParameterList(nn.Parameter(torch.zeros(width, hidden)) for _ in range(n_experts))
        self.register_buffer("balance_bias", torch.zeros(n_experts))
        self.register_buffer("pending_load", torch.zeros(n_experts, dtype=torch.int64), persistent=False)

    def forward(self, x: torch.Tensor, return_counts: bool = False):
        """默认返回输出和辅助损失；检查点调用者另外取计数，在重算区域外累计。"""
        shape = x.shape
        z = (self.down(x) if self.down is not None else x).reshape(-1, self.router.in_features)
        N, E, k = z.size(0), self.n_experts, self.top_k
        scores = torch.sqrt(F.softplus(self.router(z).float()))             # (N, E)
        chosen = (scores.detach() + self.balance_bias).topk(k, dim=-1).indices  # (N, k)
        gates = scores.gather(-1, chosen)
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-9)

        y, counts = dispatch(z, chosen, gates, self.w_in, self.w_out, self.cap)
        if self.up is not None:
            y = self.up(y)

        load = counts.float() / (max(N, 1) * k)  # 每个专家实际接了多大比例的活，均衡时是 1/E
        share = (scores / (scores.sum(-1, keepdim=True) + 1e-9)).sum(0) / max(N, 1)  # 每个专家平均拿到多大比例的分数
        balance_loss = E * (load * share).sum()
        out = self.shared(x) + y.view(shape)
        if return_counts:
            return out, balance_loss, counts
        self.record_load(counts)
        return out, balance_loss

    def record_load(self, counts):
        if self.training and torch.is_grad_enabled():
            self.pending_load.add_(counts.detach())

    @torch.no_grad()
    def apply_balance(self) -> None:
        """整数比较避免各个 micro-batch 大小不同时平均比例造成偏差。"""
        delta = self.pending_load.sum() - self.n_experts * self.pending_load
        self.balance_bias.add_(delta.sign().to(self.balance_bias.dtype), alpha=self.balance_rate)
        self.pending_load.zero_()
