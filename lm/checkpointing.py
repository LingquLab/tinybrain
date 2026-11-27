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
    if op in _SAVED_OPS or op == torch.ops.aten.topk.default or str(op) in ("flex_attention", "cudnn.sdpa_fwd.default"):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def checkpoint_block(block, x, cos, sin, maps, memory, policy="full"):
    options = {}
    if policy == "selective":
        options["context_fn"] = functools.partial(create_selective_checkpoint_contexts, _policy)
    x, loss, counts = checkpoint(block, x, cos, sin, maps, memory, return_counts=True,
                                 use_reentrant=False, **options)
    # 重算只会再次调用 block，不会执行下面这次记账。
    if counts is not None:
        block.ffn.record_load(counts)
    return x, loss
