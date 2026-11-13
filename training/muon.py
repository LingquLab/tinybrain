"""Muon：给矩阵参数用的优化器。

每一步先做动量，再把更新量「正交化」——用几次 Newton-Schulz 迭代把它的奇异值都推到 1 附近，
这样每个方向走的步子一样大，不会被少数几个大的方向主导。只适用于二维以上的参数。
形状相同的参数叠在一起一次算完（GPU 上比一个个算快得多）。
"""
import torch


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """对最后两维做 Newton-Schulz 迭代，前面的维度都当成批。返回和 g 同形状、奇异值大致在 1 附近的矩阵。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    return x.mT if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))
        for group in self.param_groups:
            if any(p.ndim < 2 for p in group["params"]):
                raise ValueError("Muon 只能用于二维以上的参数")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = {}
            for p in group["params"]:
                if p.grad is not None:
                    buckets.setdefault(p.shape, []).append(p)
            for shape, params in buckets.items():
                grads = torch.stack([p.grad for p in params])
                key = (shape, len(params))
                buf = self.state[params[0]].get(key)
                if buf is None:
                    buf = self.state[params[0]][key] = torch.zeros_like(grads)
                buf.lerp_(grads, 1 - group["momentum"])
                update = grads.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                half = update.is_cuda  # CUDA 上用半精度算正交化：快，而且只要方向大致对就行
                update = orthogonalize(update.bfloat16() if half else update, group["ns_steps"]).to(grads.dtype)
                scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
                for p, u in zip(params, update):
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(u, alpha=-group["lr"] * scale)
        return loss
