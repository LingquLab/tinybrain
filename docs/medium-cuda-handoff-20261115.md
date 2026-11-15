# Medium CUDA 优化合并交接（2026-11-15）

本分支从 `883d57c` 出发，尚未合入 `main`。交接时远端 `main` 是 `9f80bf3`，其中已有 WSD 调度和其他诊断改动。请按功能合并，保留双方修改；本分支没有实现或启用 WSD。

## 本分支的修改

- `lm/layers/attention.py`、`lm/checkpointing.py`：可选 NVIDIA cuDNN Frontend 原生因果/滑动窗口 GQA 前向与反向，选择性重计算保存官方 SDPA 算子的结果。默认仍为 FlexAttention。`window_left=window` 对应原来的 `q-k < window`；反向需要 `return_lse=True`。
- `lm/layers/hyper.py`、`block.py`：CUDA 编译路径融合 mHC 读流与 RMSNorm。普通路径继续复用 Liger；保留 FP64 普通路径。
- `lm/layers/feedforward.py`：编译路径暴露原生 ReLU²，供 Inductor 联合搜索 GEMM 与激活/反向 epilogue。普通 CUDA 路径继续使用 Liger。
- `training/train_pretrain.py`：增加显存分配预算、checkpoint policy、注意力后端和独立验证微批参数；每个完整训练更新只读回一次 loss。
- `training/trainer_utils.py`：编译前配置注意力后端，提前注册官方算子。
- `tools/bench.py`：支持梯度累积、cuDNN 后端和显存预算；报告完整更新的吞吐，以及 allocated/reserved 显存。
- `inference/generate.py`、`fast.py`、`eval/suite.py`、`eval/benchmarks.py`：默认温度 0.8、严格 top-k 50，设备匹配的随机数生成器，CLI 种子及评测参数记录。预训练原始文本提示加一个 EOS 前缀，并在生成 EOS 时停止。选择题仍按候选答案概率评分。
- `pyproject.toml`、`README.md`：补充可选依赖和复现命令。

没有修改官方库源码，没有改变模型参数名称或优化器状态格式。Q/K/V 权重仍分别保存在检查点中。

## 当前训练及恢复约束

正在运行的 medium 是完整的 498,868,768 参数模型：宽 1152、16 层、序列长 2048，微批 4、累积 4，每次更新处理 32,768 tokens。参数、梯度、Adam 状态和 Muon 动量都在 GPU；没有 CPU offload、FP8 或优化器图重放。使用 BF16 autocast、FP32 参数和状态。

当前命令：

```bash
python -m training.train_pretrain --data data/tokens/base --out out/medium/pretrain --preset medium --seq-len 2048 --batch-size 4 --grad-accum 4 --epochs 1 --autotune --attention-backend cudnn --gpu-memory-fraction 0.85 --checkpoint-policy selective --eval-batch-size 16 --eval-micro-batch-size 4 --log-every 50
```

原调度为 100 步 warmup，随后按整轮 615,948 步做 cosine，终值为峰值的 0.1 倍。82,850 步目前只是讨论中的阶段目标，现有调度在该步仍约为峰值的 96.05%。用户尚未给出 WSD 的稳定段/衰减段长度，也没有要求现在切换调度或停止训练。

合并 WSD 后若要续训，需要显式恢复当前检查点的模型、优化器和步数，避免重新 warmup，以及学习率在切换点意外跳变。请保留本分支的全部性能和验证参数。合并代码本身不会把新调度注入正在运行的 Python 进程。

验证配置固定为每批抽 16 条，再拆成微批 4；20 批、2048 长度、种子 `1_000_003 * (i+1)`。这保持与原 small 训练相同的验证片段。不要改成直接抽微批 4，否则验证数据会变，历史 loss 无法直接比较。

## 与 main 合并时重点检查

1. `training/train_pretrain.py`：同时保留 main 的 WSD/退火恢复逻辑，以及本分支的后端、显存预算、验证微批和 loss 同步修改。显存上限要在模型分配之前设置。
2. `training/trainer_utils.py`：main 的 LR 调度与本分支的 `compile_model(..., attention_backend=...)` 修改都需要保留。
3. `inference/generate.py`：main 已有 EOS 前缀相关修改。合并后应只加一个前缀，并保留严格 top-k、设备匹配 RNG 和非贪心默认参数。
4. README 的训练现状和 CUDA 说明需与 main 的新文档一起整理。

## 本机验证证据

环境：RTX 5070 Ti 16GB / SM120，Windows，Python 3.14，PyTorch 2.13.0+cu132，triton-windows 3.7.1.post27，Liger 0.8.5，cuDNN Frontend 1.32.3，CuTeDSL 4.8.1。

- 30 个计时更新、3 步预热的正式 bench：26,090 tokens/s，1.256 秒/完整更新；allocated 峰值 11.29 GiB，reserved 峰值 13.33 GiB。真实训练约 25,500–26,000 tokens/s。
- Windows 稳态 30 次采样：专用显存峰值 12.44 GiB，共享显存 82 MiB。这是采样区间的峰值。
- 使用更新过 13 步的同一 medium 权重，在真实 2048 长度数据上比较原实现与优化组合：loss 9.14612198 → 9.14611530，差 -6.68e-6；所有参数梯度合计相对 L2 差异 0.1442%。最大单个参数相对差异为 `hyper.res_logit` 的 3.74%，该参数的最大绝对梯度差约 6.22e-6。BF16 不保证逐位一致；这些检查不能证明长期训练质量完全等同。
- 8 项 CUDA 回归和 2 项 CPU 训练检查通过，包括窗口边界/GQA 梯度、融合读流梯度、融合损失、RoPE、RMSNorm、QKV 投影及全模型重计算梯度。`tests/` 按现有仓库规则仅保存在本机，本次没有强行发布。
- 训练已经超过 12,000 步，检查点保存、固定验证和 TensorBoard 持续工作。11500 步的 CPU FP32 逐词缓存检查：下一词 argmax 一致率 100%，最大 logit 差约 1.41e-5。

性能复现：

```bash
python -m tools.bench medium --batch 4 --grad-accum 4 --seq 2048 --steps 30 --autotune --attention-backend cudnn --gpu-memory-fraction 0.85
```

已尝试但未并入的方案：全面原生 mHC 写流融合、不同激活缓存策略、48 组 Liger mHC 启动配置、微批 8 加 FFN 重算、编译 Muon、融合 Newton-Schulz 乘加、优化器图与固定梯度缓冲。没有得到明确的整步收益，部分图方案超出显存预算。全范围 GEMM 搜索编译约 10 分钟仍未完成第一步，已中断，不能据此判断其速度。不要把这些实验选项当作已经支持的训练参数。

## 生成质量与 loss

固定验证 loss 的同 token 对比：

| 步数 | small | medium |
| --- | ---: | ---: |
| 1000 | 4.0561 | 3.9537 |
| 4000 | 3.4970 | 3.3533 |
| 11600 | 3.2135 | 3.0475 |

CPU FP32 原始文本续写，温度 0.8、top-k 50、种子 0/1，同一组 14 个样本：平均重复四元组占比从 1000 步的 26.1% 降到 4000 步的 19.1%，再到 11500 步的 11.8%；长度上限内生成 EOS 的次数为 0、0、1。小样本只用于观察，不是正式能力基准。11500 步仍答错首都、续不出原诗，部分提示严重重复，长文会跳题；不能把 loss 改善直接等同于可靠生成能力。

完整数据、模型/优化器存档、TensorBoard 事件和 profiler trace 均留在本机 `data/`、`out/`，未纳入本次上传。
