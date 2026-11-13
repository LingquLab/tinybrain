"""SFT：从一个存档出发，在打包好的对话数据上接着训，只在回答上算损失。三档的通用 SFT 和三个领域专家都用它。

    python -m tools.prepare --chat --input data/sft_small.parquet --out data/tokens/sft-small --seq-len 2048
    python -m training.train_sft --init out/small/anneal/ckpt.pt --data data/tokens/sft-small --out out/small/sft --epochs 2
    python -m training.train_sft --init out/medium/sft/ckpt.pt --data data/tokens/expert-math --out out/medium/expert_math --epochs 1   # 领域专家

和预训练不同的地方：
  - 一定要 --init，模型设置取自存档；
  - 数据是 tools.prepare --chat 打包的对话，一行一个片段，片段长度就是打包时的 --seq-len；遮罩为 0 的位置（提示、填充）不算损失；
  - 按轮数（--epochs）或步数（--steps）训；
  - 学习率默认是预训练的 0.1 倍、预热短、最后降到 0（还没在我们的模型上调过）。
全部参数见文件末尾的 parse_args。
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .dataset import PackedData, open_data
from .trainer_utils import (add_optimizer_args, autocast, compile_model, count, load_model, lr_scale, optimizers_from_args, pick_device, resume,
                            save_checkpoint, set_lr_scale, setup_device, step_optimizers)


def open_chat_data(directory: str) -> tuple[PackedData, PackedData]:
    data = open_data(directory, "train")
    if not isinstance(data, PackedData):
        raise ValueError(f"{directory} 不是对话数据；先用 python -m tools.prepare --chat 准备")
    if not data:
        raise ValueError(f"{directory} 里没有训练数据")
    return data, PackedData(directory, "val")


def total_steps(args, data: PackedData) -> int:
    """给了 --steps 就用它；否则按 --epochs 算：每轮把全部行打乱后过一遍。"""
    if args.steps:
        return args.steps
    return max(1, math.ceil(args.epochs * data.rows / (args.batch_size * args.grad_accum)))


@torch.no_grad()
def evaluate(model, data: PackedData, args, device: torch.device) -> float:
    """验证集上回答部分的平均损失。每次用同一批行，前后才可比。"""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(args.eval_batches):
        x, y = data.batch(args.batch_size, data.row_len - 1, seed=1_000_003 * (i + 1), device=device)
        with autocast(device):
            out = model(x, y)
        total += (out.loss - out.balance_loss).item()
    model.train(was_training)
    return total / args.eval_batches


def train(args) -> dict:
    """跑完 SFT，返回 {"step", "loss", "val_loss", "tokens"}。tokens 只算回答部分的词。"""
    device = pick_device(args.device)
    setup_device(device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 数据
    train_data, val_data = open_chat_data(args.data)
    seq_len = train_data.row_len - 1
    steps = total_steps(args, train_data)

    # 模型和优化器
    model = load_model(Path(args.init), device)
    if model.cfg.vocab_size != train_data.vocab_size:
        raise ValueError(f"模型的词表是 {model.cfg.vocab_size}，数据是用 {train_data.vocab_size} 的词表切的")
    if seq_len > model.cfg.max_seq_len:
        raise ValueError(f"数据是按片段长度 {seq_len} 打包的，超过了模型的上限 {model.cfg.max_seq_len}")
    optimizers = optimizers_from_args(model, args)
    start_step = resume(out / "ckpt.pt", model, optimizers, device) if args.resume else 0
    (out / "config.json").write_text(json.dumps({"sft": vars(args), "steps": steps, "model": json.loads(model.cfg.to_json()), "data": train_data.meta},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    forward = compile_model(model, device, enabled=not args.no_compile)
    print(f"设备 {device}；从 {args.init} 出发；{train_data.rows} 行对话（每行 {seq_len} 词）；"
          f"每步 {args.batch_size * args.grad_accum} 行，共 {steps} 步（约 {steps * args.batch_size * args.grad_accum / train_data.rows:.2f} 轮）", flush=True)

    # 训练循环
    model.train()
    smooth, val_loss, loss_value, learned = None, None, float("nan"), 0
    last_time = time.perf_counter()
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for step in range(start_step, steps):
            scale = lr_scale(step, steps, args.warmup_steps, args.final_lr_frac)
            set_lr_scale(optimizers, scale * args.lr_mult)
            loss_value = 0.0
            for micro in range(args.grad_accum):
                x, y = train_data.ordered_batch(args.batch_size, seq_len, index=step * args.grad_accum + micro, seed=args.seed, device=device)
                learned += int((y != -1).sum())
                with autocast(device):
                    loss = forward(x, y).loss / args.grad_accum
                loss.backward()
                loss_value += loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"第 {step} 步的损失变成了 {loss_value}，训练停下。上一个存档在 {out / 'ckpt.pt'}")
            step_optimizers(optimizers)
            smooth = loss_value if smooth is None else 0.9 * smooth + 0.1 * loss_value

            # 记录、验证、存档
            done = step + 1
            record = None
            if done % args.log_every == 0 or done == steps:
                now = time.perf_counter()
                epoch = done * args.batch_size * args.grad_accum / train_data.rows
                record = dict(step=done, epoch=round(epoch, 4), answer_tokens=learned, loss=round(loss_value, 4), smooth_loss=round(smooth, 4),
                              lr_scale=round(scale, 4), seconds=round(now - last_time, 2))
                last_time = now
                print(f"第 {done:>6} 步（{epoch:.3f} 轮，回答 {count(learned)} 词）  损失 {loss_value:.4f}（平滑 {smooth:.4f}）  学习率×{scale:.3f}", flush=True)
            if val_data and (done % args.eval_every == 0 or done == steps):
                val_loss = evaluate(forward, val_data, args, device)
                record = {**(record or dict(step=done)), "val_loss": round(val_loss, 4)}
                print(f"第 {done:>6} 步  验证集损失 {val_loss:.4f}", flush=True)
            if record:
                log.write(json.dumps(record) + "\n")
                log.flush()
            if done % args.save_every == 0 or done == steps:
                save_checkpoint(out / "ckpt.pt", model, optimizers, done, sft_config=vars(args))
    return dict(step=steps, loss=loss_value, val_loss=val_loss, tokens=learned)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SFT：只在回答上算损失")
    g = p.add_argument_group("数据和输出")
    g.add_argument("--init", required=True, help="从这个存档出发（退火之后的，或者通用 SFT 之后的）")
    g.add_argument("--data", required=True, help="tools.prepare --chat 打包好的对话数据目录")
    g.add_argument("--out", required=True, help="输出目录")
    g.add_argument("--batch-size", type=int, default=16, help="每个批次多少行")

    g = p.add_argument_group("训练")
    g.add_argument("--epochs", type=float, default=1.0, help="把数据过几遍（没给 --steps 时用）")
    g.add_argument("--steps", type=int, default=0, help="一共更新多少次；给了就不看 --epochs")
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--warmup-steps", type=int, default=50)
    g.add_argument("--final-lr-frac", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    add_optimizer_args(p, lr_mult=0.1)

    g = p.add_argument_group("记录和存档")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=100)
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument("--save-every", type=int, default=200)
    g.add_argument("--resume", action="store_true")

    g = p.add_argument_group("硬件")
    g.add_argument("--device", default="auto")
    g.add_argument("--no-compile", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
