"""用 torchrun 启动两个 rank，检查 ZeRO-1 梯度归属、裁剪和更新后的参数广播。"""
import argparse
import math

import torch
from torch import nn

from training.distributed import init_distributed


def check(device_name, tilexr):
    zero, device = init_distributed(device_name, tilexr=tilexr)
    try:
        if zero.world != 2:
            raise ValueError("此测试需要两个 rank")
        # 两个参数大小相同，按注册顺序分别归 rank 0 和 1。
        model = nn.ParameterList([
            nn.Parameter(torch.full((3,), float(zero.rank + 1), device=device))
            for _ in range(2)
        ])
        model.register_buffer("indices", torch.tensor([2**40 + zero.rank], device=device))
        zero.setup(model)
        assert model.indices.item() == 2**40
        for p in model:
            torch.testing.assert_close(p, torch.ones_like(p))
        loss = model[0].sum() * (zero.rank + 1)
        if zero.rank == 1:
            loss = loss + model[1].sum() * 4
        loss.backward()
        torch.testing.assert_close(zero.mean(loss).cpu(), torch.tensor(10.5))
        zero.reduce_gradients()
        assert model[1 - zero.rank].grad is None
        owned = model[zero.rank]
        expected = (1.5, 2.0)[zero.rank]
        torch.testing.assert_close(owned.grad, torch.full_like(owned, expected))
        norm = zero.clip_gradients(1.0)
        reference_norm = math.sqrt(3 * (1.5**2 + 2.0**2))
        torch.testing.assert_close(norm.cpu(), torch.tensor(reference_norm))
        with torch.no_grad():
            owned.sub_(owned.grad)
        zero.broadcast_parameters()
        for p, grad in zip(model, (1.5, 2.0)):
            torch.testing.assert_close(p, torch.full_like(p, 1 - grad / (reference_norm + 1e-6)))
        assert zero.sum_floats(2**25 + zero.rank) == [2**26 + 1]
        print(f"rank={zero.rank} device={device} tilexr={tilexr} PASS", flush=True)
    finally:
        zero.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--tilexr", action="store_true")
    args = parser.parse_args()
    check(args.device, args.tilexr)
