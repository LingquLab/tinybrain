"""在当前机器上量一量训练速度和显存：python -m tools.bench small --batch 8 --seq 8192

用随机词跑若干步「前向 + 反向 + 优化器」，报告每秒处理多少词、显存峰值。拿到新显卡先跑它。
"""
import argparse
import time

import torch

from lm import PRESETS, LanguageModel, preset

from training.trainer_utils import autocast, build_optimizers, compile_model, peak_memory_gb, pick_device, setup_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="训练速度和显存测试")
    parser.add_argument("preset", choices=list(PRESETS))
    parser.add_argument("--vocab", type=int, default=32768, help="词表大小（默认是正式分词器的 32768）")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3, help="前几步不计时（编译、缓存预热）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--optimizer", choices=["custom", "native"], default="custom", help="矩阵用哪个 Muon：custom 是自己写的，native 是 PyTorch 自带的")
    parser.add_argument("--generate", action="store_true", help="改成量生成速度：每批 --batch 行，提示长 --seq，各写 256 个词")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    setup_device(device)
    cfg = preset(args.preset, args.vocab, max_seq_len=max(args.seq, 1024))
    model = LanguageModel(cfg).to(device)
    optimizers = build_optimizers(model, args.optimizer)
    step_fn = compile_model(model, device, enabled=not args.no_compile)
    if args.generate:
        return bench_generate(model, args.vocab, args.batch, args.seq, device, args.no_compile)
    print(f"设备 {device}，{args.preset}，优化器 {args.optimizer}，参数 {model.param_report()['合计'] / 1e6:.1f}M，批 {args.batch} × 长 {args.seq}")

    elapsed = 0.0
    for step in range(args.warmup + args.steps):
        idx = torch.randint(0, args.vocab, (args.batch, args.seq + 1), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast(device):
            loss = step_fn(idx[:, :-1], idx[:, 1:]).loss
        loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= args.warmup:
            elapsed += time.perf_counter() - start
    speed = args.steps * args.batch * args.seq / elapsed
    peak = peak_memory_gb(device)
    print(f"每秒 {speed:,.0f} 词，每步 {1000 * elapsed / args.steps:.0f} 毫秒" + (f"，显存峰值 {peak:.1f} GB" if peak is not None else ""))
    return speed


def bench_generate(model, vocab: int, batch: int, prompt_len: int, device, no_compile: bool, new_tokens: int = 256) -> float:
    """普通路径和快速路径各生成一次（快速路径先热身一次，让它编译好），报告每秒生成多少词。"""
    from inference.fast import FastGenerator
    from inference.generate import generate

    idx = torch.randint(0, vocab, (batch, prompt_len), device=device)
    fast = FastGenerator(model, compile=None if not no_compile else False)
    fast.generate(idx, 8, temperature=0)  # 热身
    results = {}
    for name, fn in (("普通", lambda: generate(model, idx, new_tokens, temperature=0, fast=False)), ("快速", lambda: fast.generate(idx, new_tokens, temperature=0))):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        results[name] = batch * new_tokens / (time.perf_counter() - start)
    print(f"设备 {device}，每批 {batch} 行：普通路径每秒 {results['普通']:,.0f} 词，快速路径每秒 {results['快速']:,.0f} 词"
          + ("（已编译）" if fast.compiled else "（没编译）"))
    return results["快速"]


if __name__ == "__main__":
    main()
