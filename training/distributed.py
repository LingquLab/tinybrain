"""多卡训练：ZeRO-1（优化器状态按参数分到各张卡上）。用 torchrun 启动就是多卡，单卡的用法不变：

    torchrun --standalone --nproc-per-node 8 -m training.train_pretrain --preset medium --data data/tokens/base --batch-size 2 ...

每张卡上有一份完整的模型，各自算 --batch-size × --grad-accum 个片段；一步一共 --batch-size × --grad-accum × 卡数 个片段。
第几步、第几个微批、第几张卡取哪些片段，只由种子和这三个序号决定，合起来正好是单卡同一步取的那些片段。
所以只要「每步的片段数」一样，多卡和单卡训出来的是同一个模型（只差浮点加法的顺序），存档也能在单卡和多卡之间互相接着训。

ZeRO-1：每个参数归一张卡管（从大到小贪心地分，各卡管的总量差不多）。一步里：
  1. 各卡各自前向、反向，得到自己那份数据的梯度；
  2. 每个参数的梯度求平均，只送到管它的那张卡上；
  3. 梯度裁剪：各卡算自己管的那部分梯度的平方和，加起来就是总范数，各卡按同一个比例缩小；
  4. 各卡只更新自己管的参数。优化器状态（Adam 的两个矩、Muon 的动量）只在管它的卡上有，每张卡只占原来的 1/卡数，这就是省显存的地方；
  5. 更新好的参数从管它的卡广播给所有卡。
通信量和普通的数据并行一样多（汇总一次、广播一次）。存档时把各卡的优化器状态合成一份，格式和单卡存的完全一样。
MoE 的路由偏置按各卡记下的负载的平均来调，各卡的偏置始终一样。
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from lm.layers.moe import MoE

from .trainer_utils import pick_device

BUCKET = 32 * 2**20  # 一次通信最多多少个数（32M 个 float32 是 128 MB）


def assign_owners(params, world: int) -> dict[int, int]:
    """每个参数归哪张卡管：从大到小，每次给目前总量最小的卡（一样小时取编号小的）。返回 {id(参数): 卡号}。"""
    loads, owners = [0] * world, {}
    for p in sorted(params, key=lambda p: -p.numel()):  # sorted 是稳定的：一样大的按模型里的顺序
        rank = min(range(world), key=lambda r: loads[r])
        owners[id(p)] = rank
        loads[rank] += p.numel()
    return owners


def shard_optimizer_state(opt: torch.optim.Optimizer, state_dict: dict, owners: dict[int, int], rank: int) -> dict:
    """从完整的优化器状态（存档里的）只留下这张卡管的那些参数的部分。"""
    params = [p for group in opt.param_groups for p in group["params"]]
    keep = {i for i, p in enumerate(params) if owners[id(p)] == rank}
    return {**state_dict, "state": {i: s for i, s in state_dict["state"].items() if i in keep}}


def _buckets(tensors: list[torch.Tensor]):
    """按精度分开，再切成一段段不超过 BUCKET 个数的，每段一起通信。"""
    by_dtype = {}
    for t in tensors:
        by_dtype.setdefault(t.dtype, []).append(t)
    for group in by_dtype.values():
        bucket, size = [], 0
        for t in group:
            if bucket and size + t.numel() > BUCKET:
                yield bucket
                bucket, size = [], 0
            bucket.append(t)
            size += t.numel()
        if bucket:
            yield bucket


def _to_cpu(state_dict: dict) -> dict:
    return {"state": {i: {k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()} for i, s in state_dict["state"].items()},
            "param_groups": state_dict["param_groups"]}


class Zero:
    """这个进程在多卡训练里的位置。单卡时 world 是 1，训练脚本照常走单卡的路，这里的通信都用不上。"""

    def __init__(self, rank: int = 0, world: int = 1, device: torch.device | None = None, cpu_group=None):
        self.rank, self.world, self.device, self.cpu_group = rank, world, device, cpu_group
        self.params, self.owners = [], {}

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def main(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

    def main_first(self, fn):
        """先让 0 号卡做（比如第一次切词，可能要几个小时），别的卡等它做完再做（这时直接用它做好的）。"""
        if not self.enabled:
            return fn()
        result = fn() if self.main else None
        dist.monitored_barrier(group=self.cpu_group, timeout=timedelta(days=2))
        return result if self.main else fn()

    def setup(self, model: torch.nn.Module) -> None:
        """分好每个参数归谁管，再把 0 号卡的参数和缓冲区广播给所有卡（各卡从完全一样的模型出发）。"""
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.owners = assign_owners(self.params, self.world)
        if self.enabled:
            with torch.no_grad():
                for t in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(t.data, src=0)

    def owns(self, p: torch.Tensor) -> bool:
        return self.owners.get(id(p), 0) == self.rank

    def shard(self, opt: torch.optim.Optimizer, state_dict: dict) -> dict:
        """加载存档时：只留这张卡管的参数的优化器状态。"""
        return shard_optimizer_state(opt, state_dict, self.owners, self.rank) if self.enabled else state_dict

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        """各卡的值取平均（比如损失）。"""
        if not self.enabled:
            return t
        t = t.detach().clone()
        dist.all_reduce(t)
        return t / self.world

    def sum_floats(self, *values: float) -> list[float]:
        if not self.enabled:
            return list(values)
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(t)
        return t.tolist()

    @torch.no_grad()
    def reduce_gradients(self) -> None:
        """每个参数的梯度求平均，只送到管它的卡上；别的卡上这个参数的梯度清掉（用不上了）。
        某张卡上没有梯度的参数（比如这批数据一个词都没选中的专家）当 0 参与平均；所有卡上都没有的，平均完仍然没有，优化器照旧跳过它。"""
        used = torch.tensor([p.grad is not None for p in self.params], dtype=torch.int32, device=self.device)
        dist.all_reduce(used)
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        for owner in range(self.world):
            grads = [p.grad for p in self.params if self.owners[id(p)] == owner]
            for bucket in _buckets(grads):
                flat = _flatten_dense_tensors(bucket)
                dist.reduce(flat, dst=owner)
                if owner == self.rank:
                    flat /= self.world
                    for g, f in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                        g.copy_(f)
        for p, n in zip(self.params, used.tolist()):
            if not n or not self.owns(p):
                p.grad = None

    @torch.no_grad()
    def clip_gradients(self, max_norm: float) -> torch.Tensor | None:
        """多卡版的梯度裁剪（在 reduce_gradients 之后）：总范数由各卡管的那部分合起来算，和单卡在全部梯度上算的一样；max_norm 为 0 时不裁。"""
        if not max_norm:
            return None
        grads = [p.grad for p in self.params if p.grad is not None]
        square = torch.zeros((), dtype=torch.float32, device=self.device)
        if grads:
            square = torch.stack([g.float().pow(2).sum() for g in grads]).sum()
        dist.all_reduce(square)
        norm = square.sqrt()
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        if grads:
            torch._foreach_mul_(grads, scale)
        return norm

    @torch.no_grad()
    def sync_balance(self, model: torch.nn.Module) -> None:
        """先跨卡求整步命中数之和；各卡 token 数不等时也使用同一个分母。"""
        layers = [m for m in model.modules() if isinstance(m, MoE)]
        if not layers:
            return
        counts = torch.cat([m.pending_load for m in layers])
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        for m, part in zip(layers, counts.split([m.n_experts for m in layers])):
            m.pending_load.copy_(part)

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        """更新好的参数从管它的卡广播给所有卡。"""
        for owner in range(self.world):
            mine = [p for p in self.params if self.owners[id(p)] == owner]
            for bucket in _buckets(mine):
                if owner == self.rank:
                    flat = _flatten_dense_tensors([p.data for p in bucket])
                else:
                    flat = torch.empty(sum(p.numel() for p in bucket), dtype=bucket[0].dtype, device=bucket[0].device)
                dist.broadcast(flat, src=owner)
                if owner != self.rank:
                    for p, f in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                        p.data.copy_(f)

    def gather_optimizer_states(self, optimizers: list) -> list | None:
        """存档用：各卡的优化器状态合成一份，格式和单卡一样。所有卡都要调用；结果只在 0 号卡上有，别的卡返回 None。
        单卡时直接返回优化器本身（save_checkpoint 两种都收）。走 CPU 上的 gloo 通信，不占显存。"""
        if not self.enabled:
            return optimizers
        mine = [_to_cpu(opt.state_dict()) for opt in optimizers]
        everyone = [None] * self.world if self.main else None
        dist.gather_object(mine, everyone, dst=0, group=self.cpu_group)
        if not self.main:
            return None
        merged = []
        for i, base in enumerate(mine):
            state = {}
            for theirs in everyone:
                state.update(theirs[i]["state"])
            merged.append({"state": dict(sorted(state.items())), "param_groups": base["param_groups"]})
        return merged


def init_distributed(device_name: str = "auto") -> tuple[Zero, torch.device]:
    """torchrun 启动时（环境变量里 WORLD_SIZE 大于 1）建好进程组：CUDA 用 NCCL，每个进程一张卡；CPU 用 gloo（测试用）。否则就是单卡。"""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return Zero(), pick_device(device_name)
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if device_name.startswith("cuda") or (device_name == "auto" and torch.cuda.is_available()):
        device, backend = torch.device("cuda", local), "nccl"
        torch.cuda.set_device(device)
    elif device_name in ("auto", "cpu"):
        device, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError(f"多卡训练只支持 CUDA（NCCL）和 CPU（gloo），不支持 {device_name}")
    dist.init_process_group(backend, timeout=timedelta(minutes=30))
    cpu_group = dist.new_group(backend="gloo") if backend == "nccl" else None
    return Zero(rank, world, device, cpu_group), device
