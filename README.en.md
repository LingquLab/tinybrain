# tinybrain

[![License: MIT](https://img.shields.io/badge/license-MIT-2140d6)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-2140d6)](pyproject.toml)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-e9482b)](https://pytorch.org)
[![Dataset ModelScope](https://img.shields.io/badge/dataset-ModelScope-ffd21f)](https://www.modelscope.cn/datasets/caiwangzsnm/tinybrain-dataset)
[![Video Bilibili](https://img.shields.io/badge/video-Bilibili-fb7299)](https://www.bilibili.com/video/BV1kqsj6EEtT/)
[![Model in training](https://img.shields.io/badge/model-in%20training-lightgrey)](#status)
[![GitHub stars](https://img.shields.io/github/stars/cai114514114514/tinybrain?color=15965a)](https://github.com/cai114514114514/tinybrain/stargazers)

**A Chinese-first small language model, built from scratch — starting from the data.**

[中文](README.md) · [Dataset on ModelScope](https://www.modelscope.cn/datasets/caiwangzsnm/tinybrain-dataset) · [Video (Bilibili)](https://www.bilibili.com/video/BV1kqsj6EEtT/)

tinybrain is a from-scratch language model project in the spirit of [nanochat](https://github.com/karpathy/nanochat) and [MiniMind](https://github.com/jingyaogong/minimind): every line is plain PyTorch you can read. What it adds:

- **The data is built from scratch too.** 26 public sources are downloaded, cleaned, deduplicated across sources, and split by mixture and quality into four nested tiers (1B / 5B / 20B / 60B tokens). That's the **tinybrain dataset**, and every step of it is in `dataset/`.
- **Recent architecture pieces.** Multi-stream residuals (mHC), Engram lookup memory, 3:1 sliding-window attention, an optional mixture of experts, and the Muon optimizer.
- **The whole post-training stack.** SFT → domain experts → multi-teacher on-policy distillation (MOPD) → RL (RLVR, agent RL, RLAIF), with 10 task types and their own graders.
- **Chinese first.** The tokenizer, data mixture and grading rubrics are designed around Chinese, with classical Chinese, poetry and Lu Xun's complete works included on purpose.

## Status

2026-11-15. The model, data pipeline, four pretraining tiers, the full post-training stack and evaluation are written and tested, and the whole chain has been run end to end on a 23M-param demo model. **No trained weights have been released yet.**

- **medium** (499M) is pretraining: validation loss 2.75 at about 1.06B tokens, where small was at 3.01.
- **small** (167M) reached 2.85 at about 3.3B tokens (step 100,400) and is paused on the flat part of its curve.
- Current checkpoints only continue text; they can't chat yet. `tinybrain serve` opens a chat page whose "continue" mode shows what they have learned. Post-training, weights and benchmark scores come after medium finishes.

## What it costs

| Hardware | Model | Data | One pretraining pass |
|---|---|---|---|
| RTX 5070 Ti 16 GB | small (64M trunk, 167M with embeddings and memory tables) | nano (1B tokens) | ~4 h |
| RTX 5070 Ti 16 GB | small | mini (5B tokens) | ~20 h |
| RTX 5070 Ti 16 GB | small | base (20B tokens) | ~3.4 days |
| M4 Max | demo model (23M) | 90M tokens + full post-training | 198 min total |

Measured on the 5070 Ti: 16 × 2048 per step, ~71k tokens/s, ~12.5 GiB peak memory.

## Quick start

```bash
git clone https://github.com/cai114514114514/tinybrain && cd tinybrain
pip install -e ".[data]"        # Python 3.11+, PyTorch 2.5+; installs the `tinybrain` command
pip install -e ".[cuda,fast]"   # optional: CUDA kernels (Liger) and fast tokenization

modelscope download --dataset caiwangzsnm/tinybrain-dataset pretrain_nano.parquet posttrain/sft_small.parquet --local_dir data
mv data/posttrain/* data/

tinybrain pretrain     # data/pretrain_nano.parquet → out/small/pretrain
tinybrain sft          # continues from the previous stage → out/small/sft
tinybrain generate     # continue a few built-in prompts
tinybrain chat         # chat (--tools enables a calculator and Python)
tinybrain serve        # chat page at http://127.0.0.1:8001/ (continue mode for base checkpoints), plus an OpenAI-compatible API
```

Each stage writes to `out/<preset>/<stage>/` and starts from the previous stage's checkpoint, so no paths are needed. Use `--preset medium` for a bigger model and `--data data/pretrain_mini.parquet` for more data. `tinybrain demo` runs every stage on the 23M demo model. `tinybrain <command> --help` lists all options.

## Model

```
token ids ─┬─ embeddings ───────────────┐
           └─ Engram lookup ─┬─ 2-gram ─┤  three "sources"
                             └─ 3-gram ─┘
                                  │ entry: each stream attends over the sources
                                  ▼
                        n = 4 residual streams
                                  │
        ┌────────── layer i (× L) ──────────────┐
        │ [layer 1, middle layer] Engram gate   │
        │ read → RMSNorm → attention → write    │
        │ read → RMSNorm → FFN / MoE → write    │
        └───────────────────────────────────────┘
                                  │ exit: attention merges the 4 streams
                                  ▼
                       RMSNorm → output head → logits
```

- **mHC.** The residual is widened to 4 streams. Each sublayer reads a weighted mix of the streams, writes its output back with learned weights, and mixes the streams with a doubly stochastic matrix (Sinkhorn, 20 log-domain iterations), so the total signal is neither amplified nor shrunk across layers. tinybrain uses the static variant: all coefficients are input-independent and computed once. `lm/layers/hyper.py`
- **Engram.** A hashed lookup over 2- and 3-grams (8 hash heads each). The retrieved vectors are used as extra input sources and gated into the streams before layer 1 (and a middle layer for medium/large). A lookup costs memory, not compute, so the tables are large: 45% of trunk + memory parameters in small/medium. `lm/layers/engram.py`
- **Attention.** GQA, QK-norm, RoPE, context 8192; 3 of every 4 layers see only the last 1024 tokens. On CUDA this is FlexAttention with a block-sparse mask (or cuDNN with `--attention-backend cudnn`). `lm/layers/attention.py`
- **FFN / MoE.** ReLU² FFNs. The large preset keeps layer 1 dense and uses 2 shared + 48 routed quarter-width experts with top-8 routing, sqrt-softplus scores, bias-based load balancing and a 1e-4 balance loss. `lm/layers/moe.py`

| Preset | Width × layers | Trunk | Memory tables | Embeddings + head | Total |
|---|---|---|---|---|---|
| small | 768 × 10 | 64M | 53M | 50M | 167M |
| medium | 1152 × 16 | 233M | 190M | 75M | 499M |
| large | 1280 × 20 | 889M (266M active) | 207M | 84M | 1.18B |

## Tokenizer

SentencePiece BPE, 32,768 tokens, trained on a ~35M-token sample of the mixture. No normalization, byte fallback, digits split, lossless decoding. Average characters per token: Chinese articles 1.70, encyclopedia 2.16, classical Chinese 1.11, code 2.84, math 3.09, papers 3.40, English web 4.16, overall 2.38.

## Dataset

| Tier | Tokens | Documents | Size |
|---|---|---|---|
| nano | 0.99B | 540k | 2.1 GB |
| mini | 4.96B | 3.48M | 10.3 GB |
| base | 20.19B | 13.19M | 40.6 GB |
| full | 60.26B | 42.98M | 117 GB |

Mixture for nano/mini/base: Chinese articles 37%, encyclopedias 15.5%, code 12%, math 10%, papers 8%, classical Chinese and public-domain books 6%, English web 5%, English books 4%, synthetic 2.25%, law 0.25%. full is everything left after dedup.

How it's made: per-document cleaning (mojibake, punctuation, traditional→simplified, length/language/ad/repetition filters, phone numbers, ID numbers and e-mails replaced by placeholders; court judgments excluded because they contain personal details); cross-source dedup with sentence fingerprints (2 of the 3 longest sentences must match); quality-ranked tier selection; verification of counts, duplicates, nesting and shuffling. Each row carries its source's license; there is no single license for the mixture.

## Training

- Next-token cross-entropy, computed in a form that stays correct when the model is very confident (softplus of the margin; Barron et al., arXiv 2610.31642).
- Muon for all matrices (Nesterov momentum, 5 Newton–Schulz iterations), Adam for embeddings, head, memory tables and scalars.
- Cosine or WSD learning-rate schedules; WSD runs can branch a cooldown from any checkpoint.
- Epoch-based data order that is fully determined by the seed, so `--resume` gives the same result as an uninterrupted run.
- Multi-GPU with ZeRO-1: launch with `torchrun --standalone --nproc-per-node 8 -m training.train_pretrain ...`. Each parameter is owned by one GPU, which receives its averaged gradient, updates it and broadcasts it back, so optimizer state is split across GPUs. Data order depends only on the seed and the step, so with the same rows per step (`--batch-size × --grad-accum × GPUs`) a multi-GPU run trains the same model as a single-GPU run, and checkpoints resume across the two. Pretraining only for now.
- CUDA: Liger fused cross-entropy, RMSNorm, RoPE and ReLU², FlexAttention or cuDNN attention, per-layer compile, selective recompute, optional CUDA graphs for the optimizer step. medium fits a 16 GB card with micro-batch 4 × accumulation 4 (~26k tokens/s).

## Post-training

- **SFT** on 18 external sources plus self-made literature data (poems on a given title, classical↔modern translation, Lu Xun continuations); loss only on answers.
- **Domain experts** (math, code, literature) branched from the SFT model.
- **MOPD:** the student writes its own answers; the expert for each domain scores every position, and the student minimizes the reverse KL over the full vocabulary.
- **RL:** Dr. GRPO, fully on-policy (one update per batch), with 10 task types: math, code (sandboxed tests), logic (reasoning-gym), multiple choice, instruction following, reading (F1), poetry cloze, tool use, LLM-judge (Qwen3.8-27B), and classical poetry (tonal pattern + rhyme + judge).

Details: [docs/post-training.md](docs/post-training.md) (Chinese).

## License and data

Code: [MIT](LICENSE). Each document in the dataset carries its source's license. C-Eval and CMMLU are non-commercial and are only downloaded locally for evaluation; they are not redistributed.

## References

nanochat, MiniMind, Hyper-Connections (ByteDance), mHC (DeepSeek), Engram (DeepSeek), Muon (Keller Jordan), GRPO (DeepSeekMath), Dr. GRPO, MOPD, reasoning-gym. Every data source is listed in `dataset/sources.py`.
