"""在当前机器上量一量训练速度和显存：tinybrain bench small --batch 8 --seq 8192

用随机词跑若干步「前向 + 反向 + 优化器」，报告每秒处理多少词、显存峰值。拿到新显卡先跑它。
CUDA 使用 Liger 融合损失和 mHC 等算子，以及 FlexAttention 原生窗口/GQA。
--checkpoint-policy full 改为整层重算，更省显存，但会重复执行昂贵算子。
--no-checkpoint 关闭激活重计算；--full-logits 恢复完整输出张量。
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
    parser.add_argument("--autotune", action="store_true", help="搜索固定形状下更快的矩阵和注意力内核")
    parser.add_argument("--cuda-graph", action="store_true", help="捕获融合 Adam 和自定义 Muon 的更新图")
    parser.add_argument("--no-checkpoint", action="store_true", help="关闭 CUDA 普通前馈模型的逐层激活重计算")
    parser.add_argument("--checkpoint-policy", choices=["full", "selective"], default="selective", help="selective 保存昂贵算子结果；full 整层重算，更省显存")
    parser.add_argument("--full-logits", action="store_true", help="保留完整 logits；默认分块计算相同的交叉熵以节省显存")
    parser.add_argument("--optimizer", choices=["custom", "native"], default="custom", help="矩阵用哪个 Muon：custom 是自己写的，native 是 PyTorch 自带的")
    parser.add_argument("--generate", action="store_true", help="改成量生成速度：每批 --batch 行，提示长 --seq，各写 256 个词")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    setup_device(device)
    cfg = preset(args.preset, args.vocab, max_seq_len=max(args.seq, 1024))
    model = LanguageModel(cfg).to(device)
    model.gradient_checkpointing = device.type == "cuda" and not args.no_checkpoint and not cfg.moe_experts and not args.generate
    model.checkpoint_policy = args.checkpoint_policy
    optimizers = build_optimizers(model, args.optimizer)
    step_fn = compile_model(model, device, enabled=not args.no_compile, seq_len=args.seq, autotune=args.autotune)
    if args.generate:
        return bench_generate(model, args.vocab, args.batch, args.seq, device, args.no_compile)
    graph_step = None
    if args.cuda_graph:
        if device.type != "cuda" or cfg.moe_experts:
            raise ValueError("--cuda-graph 要求 CUDA 普通前馈模型")
        from training.cuda_graph import OptimizerGraph
        graph_step = OptimizerGraph(optimizers)
    name = f"{torch.cuda.get_device_name(device)}，" if device.type == "cuda" else ""
    print(f"设备 {device}，{name}PyTorch {torch.__version__}，{args.preset}，优化器 {args.optimizer}，参数 {model.param_report()['合计'] / 1e6:.1f}M，"
          f"批 {args.batch} × 长 {args.seq}，{'已编译' if step_fn is not model else '没编译'}")
    print(f"激活重计算：{args.checkpoint_policy if model.gradient_checkpointing else '关'}；输出层：{'完整 logits' if args.full_logits else '融合交叉熵'}；注意力：{'FlexAttention' if device.type == 'cuda' else 'SDPA'}")

    elapsed = 0.0
    for step in range(args.warmup + args.steps):
        idx = torch.randint(0, args.vocab, (args.batch, args.seq + 1), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast(device):
            loss = step_fn(idx[:, :-1], idx[:, 1:], return_logits=args.full_logits).loss
        loss.backward()
        if graph_step is None:
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
        else:
            graph_step()
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
