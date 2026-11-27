"""让 selective checkpoint 看到无副作用的算子，CUDA 内核只写新建的输出。"""
import torch
from liger_kernel.ops.mhc import mhc_post_res_bwd, mhc_post_res_fwd


def _options(width):
    return dict(block_n=32, block_c=128, num_warps=4, num_stages=2) if width >= 512 else {}


@torch.library.custom_op("tinybrain::mhc_post", mutates_args=())
def _post(x: torch.Tensor, value: torch.Tensor, post: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    return mhc_post_res_fwd(x, value, post, res, out=torch.empty_like(x), **_options(x.size(-1)))


@_post.register_fake
def _(x, value, post, res):
    return torch.empty_like(x)


@torch.library.custom_op("tinybrain::mhc_post_grad", mutates_args=())
def _post_grad(x: torch.Tensor, value: torch.Tensor, post: torch.Tensor, res: torch.Tensor,
               grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return mhc_post_res_bwd(x, value, post, res, grad.contiguous(), out_grad_x=torch.empty_like(x),
                            out_grad_f=torch.empty_like(value), **_options(x.size(-1)))


@_post_grad.register_fake
def _(x, value, post, res, grad):
    return torch.empty_like(x), torch.empty_like(value), torch.empty_like(post), torch.empty_like(res)


def _save(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    return _post_grad(*ctx.saved_tensors, grad)


_post.register_autograd(_backward, setup_context=_save)


def write_mhc(x, value, post, res):
    streams, width = x.shape[-2:]
    flat = x.contiguous().view(-1, streams, width)
    value = value.contiguous().view(-1, width)
    post = post.reshape(-1, streams).float().contiguous()
    res = res.reshape(-1, streams, streams).float().contiguous()
    return _post(flat, value, post, res).view(x.shape)
