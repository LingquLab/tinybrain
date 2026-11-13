# tinybrain

从零搭一个小型中文语言模型，外加一套自己做预训练数据集的流水线。思路上参考 MiniMind 和 nanochat，结构上换成了更新的几样东西。

**现状**：模型、预训练、整套后训练（SFT、MOPD、RLVR、agent RL、RLAIF）和评测都写完并有测试，在一个 25M 的演示模型上整条链跑通过；数据清洗在进行中；分词器定为 32768 词的 SentencePiece（`tokenizer/tokenizer.model`）。CUDA 上的路径（半精度、编译、分块窗口注意力）目前只在 CPU 上验证过数值正确，还没有在显卡上跑过。

## 目录

```
lm/                    只放模型的定义（一个文件一个概念）
  config.py              所有尺寸和开关；small / medium / large 三档预设
  model.py               整个模型的组装和前向
  generation.py          带缓存的逐词生成、采样
  layers/
    attention.py           分组查询注意力 + QK 归一化 + 旋转位置编码 + 滑动窗口
    feedforward.py         ReLU² 前馈层
    moe.py                 混合专家（共享专家、sqrt(softplus) 打分、偏置均衡、可选 LatentMoE）
    hyper.py               多流残差连接（mHC，静态）
    engram.py              Engram：按 2、3 元词组查表的记忆
    streams.py             多条流的入口和出口（用注意力）
    block.py               一层 = 记忆注入 + 注意力 + 前馈
    norm.py / rotary.py    归一化、位置编码
training/              训练，和模型定义分开；每个阶段一个脚本，从头读到尾就是这个阶段的全部，参数都在文件末尾的 parse_args 里
  train_pretrain.py      预训练（也用于退火：--init 从存档出发，--data 按比例混合几份数据）
  train_sft.py           SFT：从存档出发，只在回答上算损失，按轮数或步数训（通用 SFT 和领域专家都用它）
  train_distill.py       MOPD：多老师在线蒸馏
  train_rl.py            GRPO（Dr. GRPO）：RLVR、agent RL、RLAIF 共用
  trainer_utils.py       几个脚本共用的：设备和混合精度、学习率曲线、优化器（各组学习率都能在命令行上改）、存档
  dataset.py             训练数据的读写（预训练的长串、打包好的对话、按比例混合）
  muon.py                自己写的 Muon（另一条路径是 PyTorch 自带的，--optimizer native）
  rollout.py             采样一组回答、带工具的多轮对话（RL、MOPD、评测共用）
  rewards/               数学、代码（沙箱）、诗词格律、评审模型、工具环境
inference/             推理
  tokenizer.py           读正式分词器
  template.py            对话格式和工具调用
  generate.py            生成；CUDA 上自动走 fast.py
  fast.py                预分配的静态缓存 + torch.compile（CUDA graph）
  chat.py                命令行里和模型对话
eval/                  评测：留出文档上各类别的损失、题目正确率、评审打分、固定提问的回答；C-Eval、CMMLU、GSM8K
tools/                 辅助工具，都是命令行
  report.py              参数量明细和各部件占比
  bench.py               训练速度、生成速度（--generate）
  tokenizer.py           按配比抽样本、训练 SentencePiece 分词器、量压缩率
  prepare.py             把文本或对话变成训练用的数据
  overfit.py             过拟合检查
  demo.py                在 25M 的小模型上把整条链跑一遍
  judge_check.py         用好回答和故意弄坏的回答检查评审模型和评分细则
tokenizer/             正式分词器（tokenizer.model、tokenizer.vocab）
dataset/               数据流水线（tinybrain dataset）
  sources.py             所有来源一张表：出自哪里、许可证、怎么下载、怎么取正文
  download.py            下载原始数据
  clean.py               单篇文档的清洗规则（含切书）
  pretrain.py            按来源批量清洗预训练数据
  mix.py                 去重、配比、分四档写出、检查、生成数据集说明
  sft.py                 SFT 语料
  rl.py                  RL 题目、评测题、MOPD 提示
  build.py               总入口：按顺序跑各步，开跑前检查输入
docs/                  设计文档（post-training.md：后训练流程；rlaif-rubric.md：评审的评分细则）
```

## 模型

| 部件 | 做法 |
|---|---|
| 残差 | 4 条流的 mHC：每个子层有读、写、混三组静态系数，混合矩阵是双随机的 |
| 入口 / 出口 | 每条流用注意力从「词向量、二元记忆、三元记忆」里各取所需；出口用注意力把 4 条流合成一条 |
| 记忆 | Engram：2、3 元词组各用 8 个哈希查表；入口用一次，第 2 层前再带门控注入一次 |
| 注意力 | 分组查询、QK 归一化、RoPE；三层只看最近 1024 个词，第四层看全文，上下文 8192 |
| 前馈 | ReLU²；large 档换成混合专家 |
| 归一化 | 带可学系数的 RMSNorm，输出投影零初始化 |
| 优化器 | 矩阵始终用 Muon（自己写的或 PyTorch 自带的，二选一），词向量、输出层、记忆表、各种系数用 Adam |

三档预设（词表 32768）：

| 档 | 主干 | 记忆表 | 词向量 + 输出层 | 合计 | 说明 |
|---|---|---|---|---|---|
| small | 64M | 53M（每个哈希 10 万行 × 宽 32） | 50M | 167M | 768 宽 × 10 层 |
| medium | 233M | 190M（25 万行 × 宽 48） | 75M | 499M | 1152 宽 × 16 层 |
| large | 556M，每个词用到 294M | 455M（44 万行 × 宽 64） | 84M | 1.09B | 1280 宽 × 20 层，前馈层是 24 选 8 的四分之一宽专家加一个共享专家 |

三档的记忆表各不相同，大小统一按一个比例定：记忆表占「主干 + 记忆表」的 45%（`lm/config.py` 里的 `ENGRAM_SHARE`，词向量和输出层不算在内）。
记忆表查一次只取十几行，不花算力，只占显存。

## 用法

```bash
pip install -e ".[data]"
python -m tools.report small                       # 参数量明细
python -m tools.bench small --batch 8 --seq 8192   # 当前机器上的训练速度和显存
python -m tools.bench small --optimizer native     # 换成 PyTorch 自带的 Muon 比一比
```

```python
import torch
from inference import generate
from lm import LanguageModel, preset
from training.trainer_utils import build_optimizers

model = LanguageModel(preset("small", vocab_size=32768))
optimizers = build_optimizers(model, "custom")  # 或 "native"；各组学习率是关键字参数，比如 matrix_lr=0.02
idx = torch.randint(0, 32768, (2, 512))
loss = model(idx[:, :-1], idx[:, 1:]).loss
text = generate(model, idx[:, :8], max_new_tokens=32, top_k=50)
```

## 分词器

`tokenizer/tokenizer.model`：SentencePiece BPE，32768 词，在按配比抽出的约 3500 万 token 样本上训练（每个来源读 3 个分片，现洗）。
不做归一化、字节兜底、数字逐位拆开、字符覆盖率 0.99995，解码和原文一字不差；0–3 是未知、开头、结束、填充，另留 8 个控制符给对话格式。

在留出文档上，一个词平均几个字：中文文章 1.70、百科 2.16、古文 1.11、代码 2.84、数学 3.09、论文 3.40、英文网页 4.16，合计 2.38。

```bash
python -m tools.tokenizer sample --tokens 40000000      # 重新抽样本
python -m tools.tokenizer train --vocab 32768           # 重新训练
python -m tools.tokenizer eval tokenizer/tokenizer.model
```

## 预训练

```bash
# 1. 把文本变成词数据
python -m tools.prepare --input "data/mix/out/nano/*.parquet" --out data/tokens/nano
# 2. 训练；中断后加 --resume 接着跑
python -m training.train_pretrain --data data/tokens/nano --out out/small-nano --preset small --seq-len 4096 --batch-size 8 --epochs 1
python -m training.train_pretrain --help     # 全部参数：模型（--dim、--n-layers……）、各组学习率（--matrix-lr、--embed-lr……）、记录和存档
```

输出目录里是 `config.json`（全部设置）、`log.jsonl`（损失、速度、验证集损失）、`ckpt.pt`（最新存档）。
按轮数训（`--epochs`，可以是小数；也可以直接给 `--steps`）：数据切成不重叠的片段，每轮打乱一次依次取完，一轮里每个片段恰好看一遍。
日志每一行都写着已经训了多少词（`tokens`）和多少轮（`epoch`），开训时也会先打印「共 N 步 = 多少词 = 几轮」。
目前是单卡、最简单的版本：预热加余弦的学习率、梯度累积、定期验证和存档；中断后 `--resume` 接着跑，结果和一口气跑完一样。

## 后训练

完整设计见 [docs/post-training.md](docs/post-training.md)。

```
small        ：预训练(mini 5B) → 退火 → SFT（small 专用的大语料）→ agent RL（基础工具调用）→ RLAIF
medium/large ：预训练(base 20B / full ~70B) → 退火 → SFT（各自的语料）
                 → 领域专家（代码、数学：SFT + RLVR；高质量文学：SFT）→ MOPD → agent RL → RLAIF
```

- **每档都能单独跑通**，不依赖别的尺寸。
- **退火**：预训练最后 10% 的步数，数据里一成是对话格式，所有 token 算损失。
- **MOPD**：学生自己生成，对应领域的专家逐 token 打分，最小化全词表上的反向 KL。
- **RL**：一个 GRPO 训练器（Dr. GRPO 修正），奖励来自判分器（RLVR）、工具环境（agent RL）、评审模型 Qwen3.8-27B（RLAIF，评分细则见 `docs/rlaif-rubric.md`）。
- **鲁迅全集**做成续写题，原文当参考给评审，专门补通用语言能力。

```bash
python -m dataset.build sft rl                                                 # 后训练数据（SFT 语料、RL 题目、评测题、MOPD 提示）
python -m tools.prepare --chat --input data/sft_small.parquet --out data/tokens/sft-small --seq-len 2048
python -m training.train_sft --init out/small/anneal/ckpt.pt --data data/tokens/sft-small --out out/small/sft --epochs 2
python -m training.train_rl --init out/small/sft/ckpt.pt --tasks data/rl_small.parquet --out out/small/rl --judge-url http://localhost:8000/v1
python -m training.train_distill --init out/medium/sft/ckpt.pt --prompts data/mopd_prompts.parquet --out out/medium/mopd \
    --teachers 代码=out/medium/expert_code/ckpt.pt 数学=out/medium/expert_math/ckpt.pt 文学=out/medium/expert_literature/ckpt.pt 通用=out/medium/sft/ckpt.pt
python -m eval.benchmarks prepare                                              # 下载标准基准（只在本地评测用）
python -m eval.suite out/small/rl/ckpt.pt --tasks data/eval_tasks.parquet --bench ceval cmmlu gsm8k
python -m eval.benchmarks run out/small/pretrain/ckpt.pt --shots 5           # 基座模型：不套对话格式，带例题
python -m inference.chat out/small/rl/ckpt.pt --tools
python -m tools.demo --judge-url http://127.0.0.1:8080/v1     # 小模型上整条链跑一遍
```

## 数据集

一份以中文为主的混合预训练集，分四档，大档包含小档（token 数按正式分词器算）：

| 档 | 大小 | 怎么挑 | 适合 |
|---|---|---|---|
| nano | 1B | 按配比，每个来源取质量最高的 | 跑通流程、做小实验 |
| mini | 5B | 同上 | small（167M） |
| base | 20B | 同上 | medium（499M） |
| full | 去重后的全部，60.26B | 不挑，比例跟着池子走 | large（1.09B） |

做出来是 `data/pretrain_nano.parquet`、`_mini`、`_base`、`_full` 四个文件（2.1 / 10.3 / 40.6 / 117 GB），每个都是完整的一档、内部打乱。
token 数是正式分词器的真实 token（抽样实测）。每个来源的出处、许可证和各档用量见生成的数据集说明（`python -m dataset.mix card`）。

```bash
python -m dataset.build --check              # 每一步要什么、有没有、会产出什么
python -m dataset.build                      # 从头做：download → clean → mix → sft → rl
python -m dataset.download wiki_zh --list    # 只看某个来源会下哪些文件
python -m dataset.pretrain audit <来源>       # 抽查洗出来的结果
```

来源、各自的许可、下载方式和取正文的方式都登记在 `dataset/sources.py` 里；配比在 `dataset/mix.py` 的 `MIX` 里。
