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


def _routing_policy(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.topk.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def block_body(block, x, cos, sin, maps, memory, policy, docs=None, mask=None):
    """编译边界含 checkpoint 调用；只返回张量结果，不修改模块的统计缓冲区。"""
    if policy == "off":
        return block(x, cos, sin, maps, memory, return_counts=True, docs=docs, mask=mask)
    options = {}
    if policy == "selective":
        options["context_fn"] = functools.partial(create_selective_checkpoint_contexts, _policy)
    elif getattr(block.ffn, "packed", False):
        # full 也保留 top-k，避免前向和反向重算在接近并列的专家之间换路。
        options["context_fn"] = functools.partial(create_selective_checkpoint_contexts, _routing_policy)
    return checkpoint(block, x, cos, sin, maps, memory, return_counts=True, docs=docs, mask=mask,
                      use_reentrant=False, **options)


def checkpoint_block(block, x, cos, sin, maps, memory, policy="full", docs=None, mask=None, run=None):
    body = block_body if run is None else run
    x, loss, counts = body(block, x, cos, sin, maps, memory, policy, docs, mask)
    # 重算只会再次调用 block，不会执行下面这次记账。
    if counts is not None:
        block.ffn.record_load(counts)
    return x, loss
