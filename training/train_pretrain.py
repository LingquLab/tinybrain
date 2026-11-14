"""预训练，也用来退火。

    python -m training.train_pretrain --data data/tokens/nano --out out/small/pretrain --preset small --seq-len 4096 --batch-size 8 --epochs 1

    # 退火：从预训练的存档出发，九成预训练数据 + 一成对话数据（整段算损失），学习率从 0.1 倍降到 0
    python -m training.train_pretrain --init out/small/pretrain/ckpt.pt --data data/tokens/nano:0.9,data/tokens/chat-all:0.1 \\
        --out out/small/anneal --steps 2000 --warmup-steps 0 --lr-mult 0.1 --final-lr-frac 0

按轮数训（--epochs，默认 1）：数据切成不重叠的片段，每轮打乱一次依次取完，一轮里每个片段恰好看一遍；也可以直接给 --steps。
日志里每一行都写着已经训了多少词（tokens）、多少轮（epoch），不用自己乘。
输出目录里有 config.json（这次的全部参数）、log.jsonl（每隔几步一行）、ckpt.pt（最新的存档）。
中断后加 --resume 接着跑：每一步用哪批数据只由种子和步数决定，所以接着跑和一口气跑完结果一样。
全部参数见文件末尾的 parse_args。
"""
import argparse
import ast
import json
import math
import time
from pathlib import Path

import torch

from lm import PRESETS, LanguageModel, ModelConfig, preset

from .dataset import open_mixed
from .trainer_utils import (add_optimizer_args, autocast, compile_model, count, load_checkpoint, lr_scale, optimizers_from_args, pick_device, resume,
                            save_checkpoint, set_lr_scale, setup_device, step_optimizers)

MODEL_ARGS = ("dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len", "window", "moe_experts")  # 命令行上能直接改的模型设置


def parse_overrides(pairs: list[str]) -> dict:
    """--set 的值：["window=512", "engram_layers=(1,3)"] → 字典，值按 Python 字面量读。"""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--set 要写成 名字=值，收到的是 {pair!r}")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"--set {key} 的值 {value!r} 读不懂") from None
    return out


def build_model(args, vocab_size: int, device: torch.device) -> tuple[LanguageModel, str]:
    """新建模型（预设 + 命令行上改的设置），或者从 --init 的存档出发（设置取自存档）。返回模型和一句说明。"""
    changes = {k: getattr(args, k) for k in MODEL_ARGS if getattr(args, k) is not None} | parse_overrides(args.set)
    if args.init:
        if changes:
            raise ValueError(f"从存档出发（--init）时模型设置取自存档，不能再改：{'、'.join(changes)}")
        state = load_checkpoint(Path(args.init), device)
        model = LanguageModel(ModelConfig.from_json(state["model_config"])).to(device)
        model.load_state_dict(state["model"])
        origin = f"从 {args.init} 出发"
    else:
        model = LanguageModel(preset(args.preset, vocab_size, **changes)).to(device)
        origin = args.preset + ("（改了 " + "、".join(changes) + "）" if changes else "")
    if model.cfg.vocab_size != vocab_size:
        raise ValueError(f"模型的词表是 {model.cfg.vocab_size}，数据是用 {vocab_size} 的词表切的")
    if args.seq_len > model.cfg.max_seq_len:
        raise ValueError(f"片段长度 {args.seq_len} 超过了模型的上限 {model.cfg.max_seq_len}")
    return model, origin


@torch.no_grad()
def evaluate(model, data, args, device: torch.device) -> float:
    """验证集上的平均损失（不含 MoE 的均衡损失）。每次用同一批片段，前后才可比。"""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(args.eval_batches):
        x, y = data.batch(args.batch_size, args.seq_len, seed=1_000_003 * (i + 1), device=device)
        with autocast(device):
            out = model(x, y, return_logits=False)
        total += (out.loss - out.balance_loss).item()
    model.train(was_training)
    return total / args.eval_batches


def total_steps(args, windows: int) -> int:
    """给了 --steps 就用它；否则按 --epochs 算：每步用 batch_size × grad_accum 个片段。"""
    if args.steps:
        return args.steps
    return max(1, math.ceil(args.epochs * windows / (args.batch_size * args.grad_accum)))


def train(args) -> dict:
    """跑完训练，返回 {"step", "loss", "val_loss", "tokens"}。"""
    device = pick_device(args.device)
    setup_device(device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 数据
    train_data, val_data = open_mixed(args.data, "train"), open_mixed(args.data, "val")
    if not train_data:
        raise FileNotFoundError(f"{args.data} 里没有训练数据")

    # 模型和优化器
    model, origin = build_model(args, train_data.vocab_size, device)
    model.gradient_checkpointing = device.type == "cuda" and not model.cfg.moe_experts
    model.checkpoint_policy = "selective"
    windows = train_data.windows(args.seq_len)
    steps = total_steps(args, windows)
    rows_per_step = args.batch_size * args.grad_accum
    optimizers = optimizers_from_args(model, args)
    start_step = resume(out / "ckpt.pt", model, optimizers, device) if args.resume else 0
    (out / "config.json").write_text(json.dumps({"train": vars(args), "steps": steps, "model": json.loads(model.cfg.to_json()), "data": train_data.meta},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    forward = compile_model(model, device, enabled=not args.no_compile, seq_len=args.seq_len, autotune=args.autotune)
    graph_step = None
    if args.cuda_graph:
        if device.type != "cuda" or model.cfg.moe_experts:
            raise ValueError("--cuda-graph 要求 CUDA 和普通前馈模型")
        from .cuda_graph import OptimizerGraph
        graph_step = OptimizerGraph(optimizers)
        print("已启用优化器图重放；首步预热和捕获。", flush=True)
    report = model.param_report()
    tokens_per_step = rows_per_step * args.seq_len
    print(f"设备 {device}；模型 {origin}，共 {report['合计'] / 1e6:.1f}M 参数（主干 {report['主干'] / 1e6:.1f}M）；"
          f"数据 {count(train_data.n_tokens)} 词（每轮 {windows} 个片段）；共 {steps} 步 = {count(steps * tokens_per_step)} 词 = "
          f"{steps * rows_per_step / windows:.2f} 轮" + (f"，从第 {start_step} 步接着跑" if start_step else ""), flush=True)

    # 训练循环
    model.train()
    smooth, val_loss, loss_value = None, None, float("nan")
    last_time, last_step = time.perf_counter(), start_step
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for step in range(start_step, steps):
            scale = lr_scale(step, steps, args.warmup_steps, args.final_lr_frac)
            set_lr_scale(optimizers, scale * args.lr_mult)
            loss_value = 0.0
            for micro in range(args.grad_accum):  # 攒几个批次的梯度再更新一次
                x, y = train_data.ordered_batch(args.batch_size, args.seq_len, index=step * args.grad_accum + micro, seed=args.seed, device=device)
                with autocast(device):
                    loss = forward(x, y, return_logits=False).loss / args.grad_accum
                loss.backward()
                loss_value += loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"第 {step} 步的损失变成了 {loss_value}，训练停下。上一个存档在 {out / 'ckpt.pt'}")
            if graph_step is None:
                step_optimizers(optimizers)
            else:
                graph_step()
            smooth = loss_value if smooth is None else 0.9 * smooth + 0.1 * loss_value

            # 记录、验证、存档
            done = step + 1
            record = None
            if done % args.log_every == 0 or done == steps:
                now = time.perf_counter()
                speed = (done - last_step) * tokens_per_step / max(now - last_time, 1e-9)
                last_time, last_step = now, done
                epoch = done * rows_per_step / windows
                record = dict(step=done, tokens=done * tokens_per_step, epoch=round(epoch, 4), loss=round(loss_value, 4), smooth_loss=round(smooth, 4),
                              lr_scale=round(scale, 4), tokens_per_s=round(speed))
                print(f"第 {done:>7} 步（{epoch:.3f} 轮，{count(done * tokens_per_step)} 词）  损失 {loss_value:.4f}（平滑 {smooth:.4f}）  "
                      f"学习率×{scale:.3f}  每秒 {speed:,.0f} 词", flush=True)
            if val_data and (done % args.eval_every == 0 or done == steps):
                val_loss = evaluate(forward, val_data, args, device)
                record = {**(record or dict(step=done)), "val_loss": round(val_loss, 4)}
                print(f"第 {done:>7} 步  验证集损失 {val_loss:.4f}", flush=True)
                last_time = time.perf_counter()  # 验证花的时间不算进训练速度
            if record:
                log.write(json.dumps(record) + "\n")
                log.flush()
            if done % args.save_every == 0 or done == steps:
                save_checkpoint(out / "ckpt.pt", model, optimizers, done, train_config=vars(args))
    return dict(step=steps, loss=loss_value, val_loss=val_loss, tokens=(steps - start_step) * tokens_per_step)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="预训练（也用来退火）")
    g = p.add_argument_group("数据和输出")
    g.add_argument("--data", required=True, help="准备好的词数据目录（tools.prepare 写的）；可以写成「目录A:0.9,目录B:0.1」按比例混合")
    g.add_argument("--out", required=True, help="输出目录")
    g.add_argument("--seq-len", type=int, default=2048, help="每个片段多少个词")
    g.add_argument("--batch-size", type=int, default=8, help="每个批次多少个片段")

    g = p.add_argument_group("模型（新建时用；--init 时设置取自存档，不能改）")
    g.add_argument("--preset", choices=list(PRESETS), default="small")
    g.add_argument("--dim", type=int, default=None, help="隐藏层宽度")
    g.add_argument("--n-layers", type=int, default=None)
    g.add_argument("--n-heads", type=int, default=None)
    g.add_argument("--n-kv-heads", type=int, default=None)
    g.add_argument("--max-seq-len", type=int, default=None, help="模型能处理的最长序列")
    g.add_argument("--window", type=int, default=None, help="滑动窗口注意力的窗口")
    g.add_argument("--moe-experts", type=int, default=None, help="专家数，0 是不用 MoE")
    g.add_argument("--set", nargs="*", default=[], metavar="名字=值", help="lm/config.py 里其他的设置，比如 engram_layers=(1,5)")
    g.add_argument("--init", default="", help="从这个存档的模型参数出发（退火用），优化器从头开始")

    g = p.add_argument_group("训练")
    g.add_argument("--epochs", type=float, default=1.0, help="把数据过几遍（可以是小数）；没给 --steps 时按它算")
    g.add_argument("--steps", type=int, default=0, help="一共更新多少次；给了就不看 --epochs")
    g.add_argument("--grad-accum", type=int, default=1, help="攒几个批次的梯度再更新一次（显存不够时用它凑大批次）")
    g.add_argument("--warmup-steps", type=int, default=100)
    g.add_argument("--final-lr-frac", type=float, default=0.1, help="学习率最后降到起步值的几分之几")
    g.add_argument("--seed", type=int, default=0)
    add_optimizer_args(p)

    g = p.add_argument_group("记录和存档")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=200, help="每隔多少步在验证集上量一次；没有验证集就不量")
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument("--save-every", type=int, default=500)
    g.add_argument("--resume", action="store_true", help="从输出目录里的存档接着跑")

    g = p.add_argument_group("硬件")
    g.add_argument("--device", default="auto", help="auto、cuda、mps 或 cpu")
    g.add_argument("--no-compile", action="store_true", help="CUDA 上也不用 torch.compile")
    g.add_argument("--cuda-graph", action="store_true", help="捕获融合 Adam 和 Muon 更新图；只支持 CUDA 普通前馈模型、自定义 Muon")
    g.add_argument("--autotune", action="store_true", help="为固定形状搜索更快的矩阵和注意力内核；首次启动编译更久")
    return p.parse_args(argv)


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
