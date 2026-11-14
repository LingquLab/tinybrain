"""混合专家前馈层（只有大模型用）。

每个词由一个所有词共用的「共享专家」加上从 E 个专家里挑出的 top-k 个来处理。
  打分：sqrt(softplus(线性))，非负；选中的 k 个分数归一后当权重。
  均衡：每个专家有一个不参与求导的偏置，只在「挑谁」时加上——哪个专家活儿多就把它的偏置调低一点；
        另外返回一个很轻的均衡损失。偏置不在前向里改：前向只记下这一批的负载，优化器走完一步再由 apply_balance 调一次。
        这样激活重计算把前向再跑一遍时，挑出来的专家和第一次一模一样，梯度才对。
  LatentMoE：moe_latent_dim 大于 0 时，先把输入压到这个宽度，路由和专家都在压缩后的空间里算，最后再升回来。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import FeedForward


class MoE(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, hidden: int, shared_hidden: int, latent_dim: int = 0,
                 balance_rate: float = 1e-3):
        super().__init__()
        self.n_experts, self.top_k, self.balance_rate = n_experts, top_k, balance_rate
        width = latent_dim or dim
        self.shared = FeedForward(dim, shared_hidden)
        self.down = nn.Linear(dim, width, bias=False) if latent_dim else None
        self.up = nn.Linear(width, dim, bias=False) if latent_dim else None
        self.router = nn.Linear(width, n_experts, bias=False)
        # 每个专家两个矩阵，各自是一个二维参数（优化器按矩阵处理）；输出那个从零开始，和普通前馈层一样
        self.w_in = nn.ParameterList(nn.Parameter(torch.empty(hidden, width)) for _ in range(n_experts))
        self.w_out = nn.ParameterList(nn.Parameter(torch.zeros(width, hidden)) for _ in range(n_experts))
        self.register_buffer("balance_bias", torch.zeros(n_experts))
        self.pending_load = None  # 最近一次训练前向记下的负载，等 apply_balance 用

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (输出, 均衡损失)。"""
        shape = x.shape
        z = (self.down(x) if self.down is not None else x).reshape(-1, self.router.in_features)
        N, E, k = z.size(0), self.n_experts, self.top_k
        scores = torch.sqrt(F.softplus(self.router(z).float()))             # (N, E)
        chosen = (scores.detach() + self.balance_bias).topk(k, dim=-1).indices  # (N, k)
        gates = scores.gather(-1, chosen)
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-9)

        # 把 (词, 专家) 对按专家排好序，每个专家处理连续的一段
        expert = chosen.reshape(-1)
        order = expert.argsort(stable=True)
        token = torch.arange(N, device=z.device).repeat_interleave(k)[order]
        counts = torch.bincount(expert, minlength=E)
        inputs, outputs, start = z[token], [], 0
        for e, count in enumerate(counts.tolist()):
            if count:
                seg = inputs[start:start + count]
                outputs.append(F.relu(seg @ self.w_in[e].T).square() @ self.w_out[e].T)
                start += count
        routed = torch.cat(outputs) * gates.reshape(-1)[order, None].to(inputs.dtype)
        y = torch.zeros(N, z.size(-1), dtype=routed.dtype, device=z.device).index_add_(0, token, routed)
        if self.up is not None:
            y = self.up(y)

        load = counts.float() / (N * k)  # 每个专家实际接了多大比例的活，均衡时是 1/E
        if self.training:
            self.pending_load = load.detach()  # 重计算时同一批再跑一遍，记下的是同样的负载
        share = (scores / (scores.sum(-1, keepdim=True) + 1e-9)).mean(0)  # 每个专家平均拿到多大比例的分数
        balance_loss = E * (load * share).sum()
        return self.shared(x) + y.view(shape), balance_loss

    @torch.no_grad()
    def apply_balance(self) -> None:
        """按最近一次训练前向的负载调一次偏置：活儿多的专家偏置降一点。每个优化器步调用一次（见 LanguageModel.update_balance）。"""
        if self.pending_load is not None:
            self.balance_bias += self.balance_rate * torch.sign(1.0 / self.n_experts - self.pending_load)
            self.pending_load = None
