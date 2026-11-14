"""各个训练脚本（train_*.py）共用的东西：设备和混合精度、学习率曲线、优化器、存档。

每个阶段自己的流程（读什么数据、损失怎么算、循环怎么走）都写在它自己的脚本里，这里只放大家都一样的部分。
"""
import contextlib
import math
import os
from pathlib import Path

import torch

from lm import LanguageModel, ModelConfig

from .muon import Muon


# ---------- 设备

def pick_device(name: str = "auto") -> torch.device:
    """auto：有 CUDA 用 CUDA，其次是苹果的 MPS，都没有就用 CPU。"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def setup_device(device: torch.device) -> None:
    """CUDA 上允许 TF32：矩阵乘法用稍低的内部精度换速度，对训练没有可见影响。"""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def autocast(device: torch.device):
    """CUDA 上前向用 bfloat16（参数仍是单精度）；其他设备不变。"""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compile_model(model: torch.nn.Module, device: torch.device, enabled: bool = True, seq_len: int | None = None,
                  autotune: bool = False) -> torch.nn.Module:
    """CUDA 上编译前向；激活重计算时逐层编译，保留层间的释放边界。MoE 不编译。"""
    if device.type == "cuda":
        from lm.loss import LigerFusedLinearCrossEntropyFunction
        if LigerFusedLinearCrossEntropyFunction is None:
            raise RuntimeError("CUDA 加速需要 liger-kernel>=0.8.5；Linux 安装 .[cuda]，Windows 配好 triton-windows 后用 --no-deps 安装 liger-kernel。")
        for block in model.blocks:
            block.attn.prepare(seq_len or block.attn._flex_length or model.cfg.max_seq_len, device)
    if not enabled or device.type != "cuda" or getattr(getattr(model, "cfg", None), "moe_experts", 0):
        return model
    options = {"triton.cudagraphs": False, "max_autotune": autotune}
    if getattr(model, "gradient_checkpointing", False):
        # 整图 AOT 反向会把多层 mHC 的布局转换、重算结果同时保留。
        # checkpoint 放在编译区域外，让每层反向结束后即可释放临时激活。
        from lm.checkpointing import checkpoint_block
        for module in (model.memory, model.entry, model.exit):
            if module is not None:
                module.compile(fullgraph=True, dynamic=False, options=options)
        model.hyper.mappings = torch.compile(model.hyper.mappings, fullgraph=True, dynamic=False, options=options)
        model.checkpoint_block = torch.compile(checkpoint_block, dynamic=False, options=options)
        return model
    return torch.compile(model, dynamic=False, options=options)


def peak_memory_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def count(n: float) -> str:
    """词数写成人看的样子：41.9K、1.3M、2.05B。"""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.3g}{unit}" if n < 100 * size else f"{n / size:,.0f}{unit}"
    return str(int(n))


# ---------- 学习率

def lr_scale(step: int, total_steps: int, warmup_steps: int, final_frac: float = 0.1) -> float:
    """第 step 步（从 0 数）的学习率是起步值的几倍：先线性预热，再按余弦降到 final_frac。"""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr_scale(optimizers: list[torch.optim.Optimizer], scale: float) -> None:
    """把所有组的学习率设成各自起步值的 scale 倍。"""
    for opt in optimizers:
        for group in opt.param_groups:
            value = group["base_lr"] * scale
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = value


# ---------- 优化器：矩阵始终用 Muon，其余（词向量、输出层、记忆表、归一化系数和零碎的标量）用 Adam

SCALARS = ("pre_logit", "post_logit", "res_logit", "query")  # 形状是矩阵、其实是一组系数的参数
MUON_KINDS = ("custom", "native")  # custom：自己写的（training/muon.py）；native：PyTorch 自带的 torch.optim.Muon（2.9 起）


def split_parameters(model: LanguageModel) -> dict[str, list]:
    """把可训练参数分成五类：embed（词向量）、head（输出层）、engram（记忆表）、scalar（系数和向量）、matrix（矩阵）。"""
    named = {"embed.weight": "embed", "head.weight": "head", "memory.table.weight": "engram"}
    groups = {"embed": [], "head": [], "engram": [], "scalar": [], "matrix": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in named:
            groups[named[name]].append(p)
        elif p.ndim < 2 or name.rsplit(".", 1)[-1] in SCALARS:
            groups["scalar"].append(p)
        else:
            groups["matrix"].append(p)
    return groups


def build_optimizers(model: LanguageModel, kind: str = "custom", matrix_lr: float = 0.02, embed_lr: float = 0.3, head_lr: float = 0.004,
                     engram_lr: float = 0.3, scalar_lr: float = 0.01, weight_decay: float = 0.0, momentum: float = 0.95,
                     adam_betas: tuple = (0.8, 0.95), adam_eps: float = 1e-10) -> list[torch.optim.Optimizer]:
    """返回 [Adam, Muon]，每个参数恰好归一组。Adam 那几组的学习率按 (dim/768)^-0.5 缩放；权重衰减只作用在矩阵上。
    每组记下 base_lr，学习率调度按比例缩放它。这些学习率都是起步值，还没有在我们的数据上调过。"""
    if kind not in MUON_KINDS:
        raise ValueError(f"没有 {kind!r} 这种 Muon 路径，可选：{'、'.join(MUON_KINDS)}")
    groups = split_parameters(model)
    scale = (model.cfg.dim / 768) ** -0.5
    spec = [("embed", embed_lr), ("head", head_lr), ("engram", engram_lr), ("scalar", scalar_lr)]
    on_cuda = next(model.parameters()).is_cuda  # CUDA 上用融合版的 Adam：一次调用更新所有参数
    adam = torch.optim.Adam([dict(params=groups[k], lr=lr * scale, name=k) for k, lr in spec if groups[k]],
                            betas=adam_betas, eps=adam_eps, weight_decay=0.0, fused=on_cuda)
    matrices = [dict(params=groups["matrix"], name="matrix")]
    if kind == "native":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(f"这版 PyTorch（{torch.__version__}）没有自带的 Muon，要 2.9 以上；或者用 --optimizer custom")
        muon = torch.optim.Muon(matrices, lr=matrix_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, ns_steps=5,
                                adjust_lr_fn="original")
    else:
        muon = Muon(matrices, lr=matrix_lr, momentum=momentum, weight_decay=weight_decay)
    optimizers = [adam, muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def add_optimizer_args(parser, lr_mult: float = 1.0) -> None:
    """每个训练脚本都有的那组优化器参数。lr_mult 是这个脚本的默认总倍数（SFT、RL 比预训练小）。"""
    g = parser.add_argument_group("优化器（矩阵用 Muon，其余用 Adam；学习率都是起步值，乘上 --lr-mult 和学习率曲线）")
    g.add_argument("--optimizer", choices=MUON_KINDS, default="custom", help="用哪个 Muon：custom 自己写的，native 是 PyTorch 自带的")
    g.add_argument("--lr-mult", type=float, default=lr_mult, help="所有学习率再乘这个数")
    g.add_argument("--matrix-lr", type=float, default=0.02, help="Muon：注意力、前馈、专家、各种投影")
    g.add_argument("--embed-lr", type=float, default=0.3, help="词向量")
    g.add_argument("--head-lr", type=float, default=0.004, help="输出层")
    g.add_argument("--engram-lr", type=float, default=0.3, help="Engram 记忆表（不做权重衰减）")
    g.add_argument("--scalar-lr", type=float, default=0.01, help="归一化系数、多流残差系数、流入口出口的查询向量")
    g.add_argument("--weight-decay", type=float, default=0.0, help="只作用在 Muon 管的矩阵上")


def optimizers_from_args(model: LanguageModel, args) -> list[torch.optim.Optimizer]:
    return build_optimizers(model, args.optimizer, matrix_lr=args.matrix_lr, embed_lr=args.embed_lr, head_lr=args.head_lr,
                            engram_lr=args.engram_lr, scalar_lr=args.scalar_lr, weight_decay=args.weight_decay)


def step_optimizers(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)


# ---------- 存档：模型参数、优化器状态、走到第几步、模型配置。先写临时文件再改名，中途断电不会留下半个存档。

def save_checkpoint(path: Path, model: LanguageModel, optimizers: list, step: int, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers], step=step, model_config=model.cfg.to_json(), **extra)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_model(path: Path, device: torch.device) -> LanguageModel:
    """只要模型（接着训练、生成、评测时用）。"""
    state = load_checkpoint(path, device)
    model = LanguageModel(ModelConfig.from_json(state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model


def resume(path: Path, model: LanguageModel, optimizers: list, device: torch.device) -> int:
    """从 path 的存档接着训：恢复模型和优化器，返回已经走完的步数；没有存档返回 0。"""
    if not Path(path).exists():
        return 0
    state = load_checkpoint(path, device)
    model.load_state_dict(state["model"])
    for opt, opt_state in zip(optimizers, state["optimizers"]):
        opt.load_state_dict(opt_state)
    return state["step"]
