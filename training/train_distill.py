"""MOPD（多老师在线蒸馏）：学生自己生成回答，按每道题的领域找对应的专家当老师，在学生写的每个位置上最小化 KL(学生 ‖ 老师)。

    python -m training.train_distill --init out/medium/sft/ckpt.pt --prompts data/mopd_medium.parquet --out out/medium/mopd \\
        --teachers 代码=out/medium/expert-code/ckpt.pt 数学=out/medium/expert-math/ckpt.pt 文学=out/medium/expert-lit/ckpt.pt

专家和学生用同一个分词器，所以 KL 是在整个词表上精确算的，不是只靠采样出的那一个词去估。只有一个老师时就是普通的在线蒸馏。
--merge-init：先把学生换成「SFT 模型 + 各专家相对它的改动的平均」（任务算术，CA-MOPD 的做法），再开始蒸馏。

提示数据（parquet，一行一道）：messages（不含回答）、domain（对应 --teachers 里的名字）。
全部参数见文件末尾的 parse_args。
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from inference.template import ChatFormat
from inference.tokenizer import DEFAULT_MODEL, Tokenizer

from .rollout import collate, keep_fitting, sample_group
from .trainer_utils import (add_optimizer_args, autocast, load_model, optimizers_from_args, pick_device, save_checkpoint, set_lr_scale,
                            setup_device, step_optimizers)


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """每个位置上的 KL(学生 ‖ 老师) = Σ p学生 (log p学生 − log p老师)，(B, T)。"""
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    return (s.exp() * (s - t)).sum(-1)


@torch.no_grad()
def merge_into(student, experts: list) -> None:
    """任务算术：学生 ← 学生 + 平均(专家 − 学生)。"""
    params = dict(student.named_parameters())
    for name, p in params.items():
        delta = sum(dict(e.named_parameters())[name] - p for e in experts) / len(experts)
        p.add_(delta)


def train(cfg) -> dict:
    """跑完蒸馏，返回最后一步的统计。cfg 是 parse_args 的结果（teachers 已经是 {领域: 存档}）。"""
    device = pick_device(cfg.device)
    setup_device(device)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.teachers:
        raise ValueError("至少要有一个老师：--teachers 领域=存档")
    prompts = pq.read_table(cfg.prompts).to_pylist()
    unknown = sorted({p.get("domain") for p in prompts} - set(cfg.teachers))
    if unknown:
        raise ValueError(f"这些领域没有对应的老师：{unknown}；老师有 {sorted(cfg.teachers)}")
    student = load_model(Path(cfg.init), device)
    teachers = {d: load_model(Path(p), device).eval() for d, p in cfg.teachers.items()}
    for name, t in teachers.items():
        if t.cfg.vocab_size != student.cfg.vocab_size:
            raise ValueError(f"老师 {name} 的词表和学生不一样，没法算全词表 KL")
        t.requires_grad_(False)
    if cfg.merge_init:
        merge_into(student, list(teachers.values()))
    fmt = ChatFormat(Tokenizer(DEFAULT_MODEL))
    prompts = keep_fitting(student, fmt, prompts, cfg.max_new, "MOPD 提示")
    optimizers = optimizers_from_args(student, cfg)
    (out / "config.json").write_text(json.dumps({"distill": vars(cfg), "model": json.loads(student.cfg.to_json())}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed) if device.type == "cpu" else None
    print(f"设备 {device}；学生 {cfg.init}；老师 {', '.join(cfg.teachers)}；{len(prompts)} 道提示，共 {cfg.steps} 步", flush=True)
    stats = {}
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for step in range(cfg.steps):
            t0 = time.perf_counter()
            set_lr_scale(optimizers, cfg.lr_mult * min(1.0, (step + 1) / max(1, cfg.warmup_steps)))
            batch = [prompts[i] for i in rng.choice(len(prompts), size=min(cfg.prompts_per_step, len(prompts)), replace=False)]
            by_domain = {}
            for p in batch:
                by_domain.setdefault(p["domain"], []).extend(sample_group(student, fmt, p["messages"], cfg.samples_per_prompt, cfg.max_new,
                                                                          cfg.temperature, None, gen))
            n_tokens = sum(sum(e.learn) for eps in by_domain.values() for e in eps)
            student.train()
            total, per_domain = 0.0, {}
            for domain, episodes in by_domain.items():
                kl_sum = 0.0
                for start in range(0, len(episodes), cfg.micro_batch):
                    x, y, learn = collate(episodes[start:start + cfg.micro_batch], device)
                    with autocast(device):
                        s_logits = student(x).logits
                        with torch.no_grad():
                            t_logits = teachers[domain](x).logits
                    kl = reverse_kl(s_logits, t_logits) * learn
                    loss = kl.sum() / max(1, n_tokens)
                    loss.backward()
                    kl_sum += kl.sum().item()
                total += kl_sum
                per_domain[domain] = round(kl_sum / max(1, sum(sum(e.learn) for e in episodes)), 4)
            if not math.isfinite(total):
                raise RuntimeError(f"第 {step} 步的 KL 变成了 {total}")
            step_optimizers(optimizers)
            stats = dict(step=step + 1, kl=round(total / max(1, n_tokens), 4), by_domain=per_domain, response_tokens=n_tokens, step_s=round(time.perf_counter() - t0, 1))
            log.write(json.dumps(stats, ensure_ascii=False) + "\n")
            log.flush()
            if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
                print(f"第 {step + 1:>5} 步  KL {stats['kl']:.4f}（" + "，".join(f"{d} {v:.3f}" for d, v in per_domain.items()) + f"）  {stats['step_s']}s", flush=True)
            if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
                save_checkpoint(out / "ckpt.pt", student, optimizers, step + 1, distill_config=vars(cfg))
    return stats


def parse_teachers(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        domain, sep, path = pair.partition("=")
        if not sep or not domain or not path:
            raise ValueError(f"--teachers 要写成 领域=存档，收到的是 {pair!r}")
        out[domain] = path
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MOPD：多老师在线蒸馏")
    g = p.add_argument_group("模型、提示和输出")
    g.add_argument("--init", required=True, help="学生从这个存档出发（通用 SFT 模型）")
    g.add_argument("--prompts", required=True, help="提示数据（parquet），每行有 messages 和 domain")
    g.add_argument("--out", required=True, help="输出目录")
    g.add_argument("--teachers", nargs="+", required=True, metavar="领域=存档", help="每个领域的老师，比如 数学=out/medium/expert_math/ckpt.pt")
    g.add_argument("--merge-init", action="store_true", help="先把学生换成 SFT 模型 + 各专家改动的平均（任务算术），再开始蒸馏")

    g = p.add_argument_group("采样")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--prompts-per-step", type=int, default=8)
    g.add_argument("--samples-per-prompt", type=int, default=4)
    g.add_argument("--max-new", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--micro-batch", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("训练")
    g.add_argument("--warmup-steps", type=int, default=5)
    add_optimizer_args(p, lr_mult=0.05)

    g = p.add_argument_group("记录、存档和硬件")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--save-every", type=int, default=50)
    g.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    args.teachers = parse_teachers(args.teachers)
    return args


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
