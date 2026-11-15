"""Muon：给矩阵参数用的优化器。

每一步先做动量，再把更新量「正交化」——用几次 Newton-Schulz 迭代把它的奇异值都推到 1 附近，
这样每个方向走的步子一样大，不会被少数几个大的方向主导。只适用于二维以上的参数。
形状相同的参数叠在一起一次算完（GPU 上比一个个算快得多）；动量按参数各存一份，
所以一步怎么走和「这一步还有哪些参数一起走」无关——多卡训练时每张卡只更新自己管的那几个矩阵（training/distributed.py）。
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
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 momentum_dtype: torch.dtype | None = None):
        """momentum_dtype：动量用什么精度存，默认和梯度一样；显存紧时用 torch.bfloat16，省一半。"""
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))
        self.momentum_dtype = momentum_dtype
        for group in self.param_groups:
            if any(p.ndim < 2 for p in group["params"]):
                raise ValueError("Muon 只能用于二维以上的参数")

    def _momentum(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p.grad, dtype=self.momentum_dtype or p.grad.dtype)
        return state["momentum_buffer"]

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
                bufs = [self._momentum(p) for p in params]
                torch._foreach_lerp_(bufs, [p.grad.to(bufs[0].dtype) for p in params], 1 - group["momentum"])
                grads = torch.stack([p.grad for p in params])
                momentum = torch.stack(bufs).to(grads.dtype)
                update = grads.lerp(momentum, group["momentum"]) if group["nesterov"] else momentum
                half = update.is_cuda  # CUDA 上用半精度算正交化：快，而且只要方向大致对就行
                update = orthogonalize(update.bfloat16() if half else update, group["ns_steps"]).to(grads.dtype)
                scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
                for p, u in zip(params, update):
                    lr = group["lr"]
                    if group["weight_decay"]:
                        p.mul_(1 - lr * group["weight_decay"])
                    if isinstance(lr, torch.Tensor):
                        p.add_(u * (-lr * scale))
                    else:
                        p.add_(u, alpha=-lr * scale)
        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        """也读以前的存档：那时同形状的矩阵共用一份叠起来的动量，存在这组第一个矩阵名下，键是（形状，个数），这里拆回每个参数一份。
        PyTorch 加载时会把状态转成参数的精度，动量要是本来存成 bfloat16，再转回去。"""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            same_shape = {}
            for p in group["params"]:
                same_shape.setdefault(tuple(p.shape), []).append(p)
            for p in group["params"]:
                state = self.state.get(p, {})
                for key in [k for k in state if isinstance(k, tuple)]:
                    shape, n = key
                    stacked = state.pop(key)
                    peers = same_shape[tuple(shape)]
                    block = peers[peers.index(p):peers.index(p) + n]  # 当时一起走的那几个（都有梯度时就是这组里同形状的全部）
                    if len(block) == n:
                        for q, buf in zip(block, stacked.unbind(0)):
                            self.state[q]["momentum_buffer"] = buf.clone()
            if self.momentum_dtype is not None:
                for p in group["params"]:
                    if "momentum_buffer" in self.state.get(p, {}):
                        self.state[p]["momentum_buffer"] = self.state[p]["momentum_buffer"].to(self.momentum_dtype)
