"""分段 GEMM 的算子边界。设备实现可以读分段；编译器只需要固定的输出形状。"""
import torch


def _sizes(ends):
    stops = ends.tolist()
    return [stop - start for start, stop in zip([0, *stops], stops)]


@torch.library.custom_op("tinybrain::expert_linear", mutates_args=())
def _linear(x: torch.Tensor, w: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    if x.is_cuda:
        from .grouped import project
        return project(x, w, ends)
    return torch.cat([part @ weight.T for part, weight in zip(x.split(_sizes(ends)), w.unbind())])


@_linear.register_fake
def _(x, w, ends):
    return x.new_empty((x.size(0), w.size(1)))


@torch.library.custom_op("tinybrain::expert_dw", mutates_args=())
def _dw(g: torch.Tensor, x: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    if x.is_cuda:
        from .grouped import weight_grad
        return weight_grad(g, x, ends)
    sizes = _sizes(ends)
    return torch.stack([dy.T @ part for dy, part in zip(g.split(sizes), x.split(sizes))])


@_dw.register_fake
def _(g, x, ends):
    return x.new_empty((ends.numel(), g.size(1), x.size(1)))


def _save(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    x, w, ends = ctx.saved_tensors
    grad = grad.to(w.dtype)
    return _linear(grad, w.mT, ends), _dw(grad, x, ends), None


_linear.register_autograd(_backward, setup_context=_save)


def linear(x, w, ends):
    if x.is_cuda and x.dtype == torch.float64:
        # tl.dot 不支持双精度。诊断时用设备上的掩码选行，避免读回边界。
        rows = torch.arange(x.size(0), device=x.device, dtype=ends.dtype)
        owner = torch.searchsorted(ends, rows, right=True)
        y = x.new_zeros((x.size(0), w.size(1)))
        for i, weight in enumerate(w.unbind()):
            y = torch.where((owner == i)[:, None], x @ weight.T, y)
        return y
    # square 在 CUDA 自动混合精度下可返回 FP32，两次投影都在这里恢复 GEMM 精度。
    device = x.device.type
    dtype = x.dtype
    if dtype != torch.float64 and torch.is_autocast_enabled(device):
        dtype = torch.get_autocast_dtype(device)
    return _linear(x.to(dtype), w.to(dtype), ends)
