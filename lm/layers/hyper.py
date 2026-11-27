"""多流残差连接（mHC，静态版）。

普通残差只有一条流：x ← x + F(x)。这里有 n 条流，每个子层（注意力、前馈各算一个）有三组可学的系数：
  pre  (n)    读：把 n 条流加权合成一份输入，取值在 0 到 1 之间；
  post (n)    写：子层的输出按多大比例加回每条流，取值在 0 到 2 之间；
  res  (n×n)  混：流和流之间怎么互相搬运，是双随机矩阵（每行每列的和都是 1），信号总量不放大也不缩小。
更新式：x ← res·x + post ⊗ F(pre·x)。静态版的系数不看输入，整个模型所有子层的系数放在一起一次算完。
"""
import torch
import torch.nn as nn

try:
    from liger_kernel.ops.mhc import LigerMHCPreFunction, mhc_pre_fwd, mhc_pre_bwd
    from training.mhc import write_mhc
except ImportError:
    LigerMHCPreFunction = write_mhc = None


class _MHCPre(torch.autograd.Function):
    """复用 Liger 内核，直接写入输入精度的缓冲区，省去大块 FP32 临时张量。"""
    @staticmethod
    def forward(ctx, x, pre):
        ctx.shape = x.shape
        x = x.contiguous().view(-1, *x.shape[-2:])
        pre = pre.reshape(-1, x.size(1)).float().contiguous()
        out = torch.empty((x.size(0), x.size(2)), device=x.device, dtype=x.dtype)
        mhc_pre_fwd(x, pre, out=out)
        ctx.save_for_backward(x, pre)
        return out.view(*ctx.shape[:-2], x.size(2))

    @staticmethod
    def backward(ctx, grad):
        x, pre = ctx.saved_tensors
        grad = grad.contiguous().view(-1, x.size(2))
        dx, dp = mhc_pre_bwd(x, pre, grad, out_grad_x=torch.empty_like(x))
        return dx.view(ctx.shape), dp.view(*ctx.shape[:-1])


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
    if LigerMHCPreFunction is not None and x.is_cuda and x.dtype != torch.float64:
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        # expand 的反向归约回同一组静态系数，保留原模型的参数化方式。
        return _MHCPre.apply(x.to(dtype), pre.expand(*x.shape[:-2], x.size(-2)))
    return torch.einsum("btnd,n->btd", x, pre)


def read_normalized_streams(x, pre, norm):
    """编译时融合读流和 RMSNorm，避免把合流结果往显存写回再读出。"""
    if not (x.is_cuda and x.dtype != torch.float64 and torch.compiler.is_compiling()):
        return norm(read_streams(x, pre))
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    x = x.to(dtype)
    # 展开后再归约，与静态系数的参数化一致；显式保留读流处的精度转换。
    pre = pre.expand(*x.shape[:-2], x.size(-2))
    mixed = (x.float() * pre.float().unsqueeze(-1)).sum(-2).to(dtype).float()
    eps = norm.eps if norm.eps is not None else torch.finfo(dtype).eps
    return (mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps) * norm.weight.float()).to(dtype)


def write_streams(x: torch.Tensor, out: torch.Tensor, res: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    """x: (B, T, n, d)，out: (B, T, d)，res: (n, n)，post: (n) → (B, T, n, d)"""
    if write_mhc is not None and x.is_cuda and x.dtype != torch.float64:
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        if out.dtype == post.dtype == dtype:
            n = x.size(-2)
            return write_mhc(
                x.to(dtype), out, post.expand(*x.shape[:-2], n), res.expand(*x.shape[:-2], n, n))
    return torch.einsum("mn,btnd->btmd", res, x) + post[:, None] * out[:, :, None, :]
