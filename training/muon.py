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
        self.expert_counts = {}  # 参数 -> 当前优化器步的全局路由计数，不写入存档
        for group in self.param_groups:
            if any(p.ndim < 2 for p in group["params"]):
                raise ValueError("Muon 只能用于二维以上的参数")

    def _momentum(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p.grad, dtype=self.momentum_dtype or p.grad.dtype)
        return state["momentum_buffer"]

    @staticmethod
    def _direction(grad, momentum, group):
        update = grad.lerp(momentum, group["momentum"]) if group["nesterov"] else momentum
        work = update.bfloat16() if update.is_cuda else update
        return orthogonalize(work, group["ns_steps"]).to(grad.dtype)

    @staticmethod
    def _apply(p, update, group):
        lr = group["lr"]
        scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        if group["weight_decay"]:
            p.mul_(1 - lr * group["weight_decay"])
        if isinstance(lr, torch.Tensor):
            p.add_(update * (-lr * scale))
        else:
            p.add_(update, alpha=-lr * scale)

    def _step_bank(self, p, group):
        buf = self._momentum(p)
        counts = self.expert_counts.get(p)
        # 每次处理若干专家；归一化和 NS 始终只看最后两维。
        chunk = max(1, 2**22 // p[0].numel())
        for i in range(0, p.size(0), chunk):
            sl = slice(i, i + chunk)
            w, g, v = p[sl], p.grad[sl], buf[sl]
            new = v.lerp(g.to(v.dtype), 1 - group["momentum"])
            active = None if counts is None else (counts[sl] > 0)[:, None, None]
            v.copy_(new if active is None else torch.where(active, new, v))
            update = self._direction(g, v.to(g.dtype), group)
            if active is None:
                self._apply(w, update, group)
            else:
                # 零梯度不等于未命中：空专家的动量、权重衰减都必须跳过。
                candidate = w.clone()
                self._apply(candidate, update, group)
                w.copy_(torch.where(active, candidate, w))

    def _step_list(self, params, group):
        bufs = [self._momentum(p) for p in params]
        torch._foreach_lerp_(bufs, [p.grad.to(bufs[0].dtype) for p in params], 1 - group["momentum"])
        grads = torch.stack([p.grad for p in params])
        update = self._direction(grads, torch.stack(bufs).to(grads.dtype), group)
        for p, u in zip(params, update):
            self._apply(p, u, group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim == 3:
                    self._step_bank(p, group)
                else:
                    buckets.setdefault(p.shape, []).append(p)
            for params in buckets.values():
                chunk = max(1, 2**22 // params[0].numel())
                for i in range(0, len(params), chunk):
                    self._step_list(params[i:i + chunk], group)
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
