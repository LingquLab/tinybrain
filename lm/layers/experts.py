"""把 top-k 个槽按专家排序，算完后按原槽顺序加权求和。"""
import torch

from .feedforward import relu_square


class _Rows(torch.autograd.Function):
    """按槽读行；反向用逆排列收回，再合并同一 token 的 k 份梯度。"""

    @staticmethod
    def forward(ctx, x, index, inverse, repeats):
        ctx.save_for_backward(inverse)
        ctx.repeats = repeats
        return x.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        inverse, = ctx.saved_tensors
        rows = grad.index_select(0, inverse)
        if ctx.repeats > 1:
            rows = rows.reshape(-1, ctx.repeats, grad.size(1)).sum(1, dtype=grad.dtype)
        return rows, None, None, None


def dispatch(z, chosen, gates, w_in, w_out, cap):
    n, k = chosen.shape
    ids = chosen.flatten()
    order = ids.argsort(stable=True)
    counts = torch.zeros(len(w_in), device=z.device, dtype=torch.int32)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
    inverse = torch.empty_like(order).scatter(0, order, torch.arange(ids.numel(), device=z.device))
    inputs = _Rows.apply(z, order // k, inverse, k)

    if z.is_cuda:
        from training.grouped import linear
        wi = w_in if torch.is_tensor(w_in) else torch.stack(tuple(w_in))
        wo = w_out if torch.is_tensor(w_out) else torch.stack(tuple(w_out))
        ends = counts.cumsum(0, dtype=torch.int32)
        parts = linear(relu_square(linear(inputs, wi, ends), cap), wo, ends)
    else:
        # CPU/MPS 对照路径只算命中的专家。unbind 让三维参数只做一次反向拼接。
        wi = w_in.unbind() if torch.is_tensor(w_in) else w_in
        wo = w_out.unbind() if torch.is_tensor(w_out) else w_out
        pieces = inputs.split(counts.tolist())
        outputs = [relu_square(x @ a.T, cap) @ b.T for x, a, b in zip(pieces, wi, wo) if x.size(0)]
        parts = torch.cat(outputs) if outputs else inputs

    slots = _Rows.apply(parts, inverse, order, 1).reshape(n, k, z.size(1))
    weighted = slots * gates.to(z.dtype).unsqueeze(-1)
    return weighted.sum(1, dtype=weighted.dtype), counts
