"""只重算廉价操作，避免整层激活重计算重复执行矩阵乘法和注意力。"""
import functools

import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts


_SAVED_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
}


def _policy(ctx, op, *args, **kwargs):
    if op in _SAVED_OPS or str(op) in ("flex_attention", "cudnn.sdpa_fwd.default"):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def checkpoint_block(block, x, cos, sin, maps, memory, policy="full"):
    if policy == "selective":
        return checkpoint(block, x, cos, sin, maps, memory, use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, _policy))
    return checkpoint(block, x, cos, sin, maps, memory, use_reentrant=False)
