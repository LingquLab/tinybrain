# 后训练流程

本文是预训练之后整条链的设计：从基座模型到能对话、能评测、能发布的三档模型（small / medium / large）。
写于 2026-11-12。

**目前写好的**：所有阶段都有代码和测试，并在一个 25M 的演示模型上从头到尾跑通（`python -m tools.demo`）：
预训练、退火、SFT、三个领域专家、RLVR、MOPD、agent RL、RLAIF、评测。真正的三档模型要等预训练数据和 PRO 6000。

## 三档的流程

```
small        ：① 预训练(mini 5B) → ② 退火 → ③ SFT（small 专用的大语料）→ agent RL（基础工具调用）→ RLAIF
medium/large ：① 预训练(base 20B / full ~70B) → ② 退火 → ③ SFT（各自的语料）
                 → ④ 领域专家 ┬ 代码：SFT + RLVR（跑测试）
                             ├ 数学：SFT + RLVR（对答案）
                             └ 高质量文学：SFT（古文、诗词、鲁迅）
                 → ⑤ MOPD（三个专家合回一个模型；通用题的老师是 ③ 自己）→ agent RL → RLAIF
每个阶段的产物都过一遍 ⑧ 评测，最后 ⑨ 发布
```

| | small（167M） | medium（499M） | large（1.09B） |
|---|---|---|---|
| ① 预训练数据 | mini，5B | base，20B | full，去重后的全部，约 70B |
| ② 退火 | 预训练最后 10% 的步数；数据里一成是对话格式 | 同 small | 同 small |
| ③ SFT 语料 | 单独一份大语料，2048 个词以内，不带推理过程 | 单独一份，4096 以内，推理过程写成 `<think>` | 单独一份，8192 以内，全部数据 |
| ④ 领域专家 | 不做 | 代码、数学、高质量文学三个 | 同 medium |
| ⑤ MOPD | 不做 | 有 | 有 |
| RLVR | 不做 | 在代码、数学专家里做 | 同 medium |
| agent RL | SFT 之后 | MOPD 之后 | 同 medium |
| RLAIF | 最后一步 | 最后一步 | 同 medium |

**原则**：
- **每档都能单独跑通，不依赖别的尺寸。** 这套东西要发布给别人用，只想训 small 的人不需要先训 large。
- **small 做简单**：一份大的 SFT 语料打底，再用 agent RL 和 RLAIF 补上最基本的工具调用和对齐。
- **medium、large 用 MOPD 加 RLVR**：领域只分三个——最好的代码、数学、高质量文学。
- **通用语言能力单独重视**：文学专家，加上用鲁迅全集做的 RL（续写给原文当参考，评审按写作细则打分）。现在的模型普遍偏重编程，这一块要补。
- **RLAIF 都放在最后**：先长能力，再做通用对齐。
- **对照组**：medium、large 各留一个「全部数据混在一起做 SFT、花同样算力」的版本，检验专家加 MOPD 到底有没有用。

## 对话格式

用分词器里预留的控制符（`tokenizer/tokenizer.model` 里的 `<|im_start|>`、`<|im_end|>`）。它们的编号是固定的；正文里出现同样的字符串，只会被切成普通文字，不会变成控制符。

```
<|im_start|>system
你是……<|im_end|>
<|im_start|>user
床前明月光的下一句是什么？<|im_end|>
<|im_start|>assistant
疑是地上霜。<|im_end|>
```

工具调用沿用同一套格式：模型在回答里写出调用，工具的返回作为 `tool` 角色的一轮放回对话。

| 阶段 | 哪些 token 算损失 |
|---|---|
| ② 退火 | 全部 |
| ③ ④ SFT | 只有 assistant 的回答，以及回答后面的 `<|im_end|>` |
| ⑤ MOPD、各种 RL | 只有模型自己生成的部分（工具返回的内容不算） |

## 各阶段

### ① 预训练（已完成）

`training/train_pretrain.py`。下一个词预测，所有 token 都算损失。数据是 `data/pretrain_{nano,mini,base,full}.parquet` 里的一档，用 `tools.prepare` 切词。

### ② 退火（中期训练）

- **做什么**：预训练学习率衰减的最后一段（默认最后 10% 的步数）换数据接着训。用 `training.train_pretrain --init` 从预训练存档出发。
- **数据**：预训练这一档里质量分最高的部分，加上一部分对话格式的数据（从这一档的 SFT 语料里抽），整段都算损失。比例 90 : 10（对话一成）。
- **为什么**：基座本身先学会对话的格式，后面 SFT 用的数据和步数都能少一些。
- **混着读**：`--data 预训练目录:0.9,对话目录:0.1`。

### ③ SFT

- **代码**：`tools.prepare --chat` 打包，`training.train_sft --init` 训练，只在回答上算损失（已写好）。
- **三份语料**：small、medium、large 各一份，分别单独做。
  - small 的是一份大而全的语料：短样本为主，覆盖面要广，因为后面没有专家来补。
  - medium、large 的语料要给后面的专家留出领域数据。
- **自我认知**：每份语料都加一小份自己写的问答：模型叫什么、谁做的、能做什么。
- **来源**：附录里的 20 个候选数据集，抽样看过质量后再定每份语料用哪些。

### ④ 领域专家（medium、large）

从 ③ 的模型复制出三份，每份只在一个领域上继续练（`training.train_sft --init ③`）。它在自己的领域要比 ③ 强，别的领域变差不要紧。

| 专家 | 数据（`data/sft_expert_*.parquet`） | 之后 |
|---|---|---|
| 代码 | opc-sft、Nemotron 竞赛编程（8192 以内） | RLVR（跑测试用例） |
| 数学 | NuminaMath-CoT、Nemotron 数学 | RLVR（对最终答案） |
| 高质量文学 | 古籍库的文白互译、诗词（按题写诗）、鲁迅续写 | — |

### ⑤ MOPD（medium、large）

做法参考 MOPD（ICML 2026），小米 MiMo-V2-Flash、DeepSeek-V4 用的也是这个思路。

1. **学生**：③ 的通用模型。
2. **路由**：每条提示带一个领域标签，按标签找同名的专家当老师（代码、数学、文学、通用；通用题的老师是 ③ 自己，起到不让学生跑偏的作用）。没有单独专家的领域按路由表交给别的老师：逻辑 → 数学；阅读、诗词、写作、古文 → 文学；指令跟随、科学、工具调用 → 通用（`training/train_distill.py` 的 `ROUTES`，`--route 领域=老师` 可以改）。提示是 SFT 语料里的问题加上全部 RL 题。
3. **生成**：学生自己采样出回答。
4. **打分**：把「提示 + 学生的回答」送进那个专家，得到专家在每个位置的概率分布。
5. **损失**：每个回答位置上，学生对老师的反向 KL，即 KL(学生 ‖ 老师)。专家和学生同尺寸、用同一个词表，所以算的是全词表上的精确值。
6. **可选开关**：先把几个专家按任务算术合成一份，当学生的起点（CA-MOPD 的做法）。

显存方面，large 用 bf16 时五个专家加一个学生不到 15 GB。

### ⑥ RL：一套训练器，奖励可插拔

RLVR、agent RL、RLAIF 用同一个训练器（`training/train_rl.py`），只是题目和奖励不同。题库由 `dataset/rl.py` 生成：

| 题型（reward） | 判分 | 来源 | 数量 | 用在 |
|---|---|---|---|---|
| math | 对最终答案 | Big-Math-RL-Verified（挑 Llama 8B 解题率 0.2–0.95 的）、DeepScaleR（竞赛题）、GSM8K_zh（中文） | 约 3 万 | medium、large |
| code | 沙箱里跑测试用例 | DeepCoder-Preview 的训练集；评测用它的 Codeforces 部分 | 约 2.4 万 | medium、large |
| logic | reasoning-gym 自带的判分 | reasoning-gym 现场生成：算术、数列、24 点、Countdown、骑士与无赖、小数独等 33 种 | 约 1 万 | medium、large |
| choice | 选项字母对不对 | ARC、OpenScience | 约 1.2 万 | medium、large |
| instruct | 程序检查约束（字数、段落、编号、关键词、开头结尾、JSON……），满足的比例 | RLVR-IFeval（英文）；给留出的中文问题加中文约束 | 约 1 万 | 三档 |
| reading | 和参考答案按字算 F1 | CMRC2018 | 约 1 万 | 三档 |
| cloze | 和标准答案完全一样 | 从诗词里出：给作者、题目、上句，写下句 | 约 5000 | 三档 |
| agent | 答案对不对；工具调用格式坏一次扣 0.1 | 程序生成：大数运算、百分比、取模、日期、在维基资料里检索（一跳、两跳） | 约 5000 | 三档 |
| judge | 评审模型按评分细则打分 | 留出的通用问题；鲁迅续写（带原文参考） | 约 7000 | 三档 |
| poetry | 格律 × 0.4 + 评审 × 0.6 | 真实诗题 | 约 2000 | 三档 |

评测题从每种题里先留出一份（共约 700 道），和 RL、SFT 都不重复；RL 和评测用到的通用问题来自 SFT 留出的 2%（`dataset/sft.py`），模型在 SFT 时没见过。

- **算法**：GRPO，用 Dr. GRPO 的修正，去掉原版偏向长回答的偏差。
- **流程**：每条提示采样一组回答，组内比较奖励，算每个回答的优势值，再做策略梯度。

| | 用在哪 | 奖励从哪来 | 要准备的 |
|---|---|---|---|
| **RLVR** | medium、large 的数学专家（math、logic）、代码专家（code）；MOPD 之后再做一轮 choice | 程序判分 | 上表的程序判分题；代码沙箱 |
| **agent RL** | small 在 SFT 之后（同时练 instruct、reading、cloze）；medium、large 在 MOPD 之后 | 环境和程序判分 | 自己做的基础工具（计算器、跑 Python、在给定资料里检索）。large 以后接小米开源的环境（见下） |
| **RLAIF** | 三档的最后一步 | 评审模型按中文评分细则打分（有用、准确、安全、格式） | 评审模型、评分细则、judge 和 poetry 题 |

- **题目难度**：RLVR 和 agent RL 都按通过率挑题，留下通过率在 10%–90% 的，并逐步加难。
- **small 上的 RL**：小模型 RL 的证据本来就弱（MobileLLM-R1：十亿参数以下，SFT 之后再加 RL 没有明显提升）。所以 small 只用 RL 磨最基础的东西：调对工具、格式对、回答得体。前提是 SFT 语料里有足够的工具调用示范。

**agent RL 的环境**：
- 第一版用自己做的基础工具环境（`training/rewards/tools.py`），任务由 `dataset/rl.py` 生成，small 也能练。
- 小米开源了 MiMo-V2.6 的 RL 环境（`XiaomiMiMo/MiMo-V2.6-RL-oss`，Apache-2.0：软件工程、漏洞复现、知识工作、网页开发、作曲）、
  训练框架（他们 fork 的 verl）和智能体外壳（mimoagent、uni-agent）。这些任务对我们的尺寸太难，
  而且 verl 要求 Hugging Face 格式和 vLLM 推理，我们的结构两样都不支持。所以先不接训练框架；以后给模型做一个
  OpenAI 兼容的服务接口，让 mimoagent 驱动它，在「知识工作」这类环境上给 large 做评测和收集轨迹。

**RLAIF 的评审模型**：
- **用 Qwen3.8-27B**：2026-08 开放权重，Apache-2.0，ModelScope 上有。PRO 6000 上用 vLLM 跑官方 FP8 版本（约 27 GB）；
  Mac 上用 mlx-lm 跑 `mlx-community/Qwen3.8-27B-4bit`（约 15 GB，`python -m mlx_lm.server --model …`），每条打分约 2.7 秒。mlx_lm.server 一次只算一个请求，所以在 Mac 上要加 `--judge-workers 1`（或 2）和较长的 `--judge-timeout`，否则排队的请求会超时、全部记成解析失败。备选是 Qwen3.6-27B。
- **打分时不思考**：它默认会想很久，打分时用不思考的模式，只输出分数和一句理由。
- **和分词器无关**：评审只看文字，所以用外部模型没问题，别人复现时下载同一个评审模型就行。
- **防钻空子**：小模型容易学会讨好评审。训练时同时盯着奖励曲线和固定提问的实际回答；奖励涨了回答却变差，就停下来改评分细则。

### 研究线：啊哈时刻对照实验（large，可选）

**问题**：小模型 RL 里出现的反思行为，是 RL 逼出来的，还是预训练里早就埋着的？我们的预训练数据是自己做的，所以能做这个对照。它不进发布的配方。

**设置**：在 large 上做（medium 也做一次，small 当下限），RL 设置完全相同，只改 ② 退火时额外加的数据：

| 组 | 退火时额外加的数据 |
|---|---|
| A | 不加 |
| B | 1%–2% 带反思的推理过程（「等等，不对，我检查一下」） |
| C | 和 B 一样多的推理过程，但删掉反思的部分 |

**任务**：凑数游戏（Countdown）、多位数四则运算、解简单方程、古诗格律检查。都能自动判分，难度可调。

**记录**：
- 正确率、回答长度、反思用语出现的频率；
- 第 0 步（RL 之前）就量一次，免得把基座里本来就有的反思误当成新长出来的；
- 反思出现的那些回答，正确率是不是更高。

### ⑧ 评测

每个阶段的产物都评一遍，结果汇总成一张表。

| 类型 | 内容 | 说明 |
|---|---|---|
| 基础能力 | 各类别留出集上的损失（中文、代码、数学、古文……） | 检查后训练有没有把预训练学到的东西训坏 |
| 知识 | C-Eval（验证集）、CMMLU；MMLU 待接 | 已接（`eval/benchmarks.py`）：比较「答案：」后面四个字母的概率，不让它生成 |
| 数学 | GSM8K；MATH-500 待接 | GSM8K 已接：贪心生成，判分和数学 RLVR 同一套 |
| 代码 | HumanEval、MBPP | |
| 指令跟随 | IFEval | |
| 工具调用 | 自己的工具任务集（和 agent RL 的任务分开，不重叠） | |
| 古文诗词 | 待定，自己出一套 | |
| 对话 | 固定的一组提问，每个阶段生成一遍，放在一起人工看 | |

标准基准的数据用 `python -m eval.benchmarks prepare` 下载到 `data/benchmarks/`（不进仓库）。C-Eval 是 CC BY-NC-SA 4.0、CMMLU 是 CC BY-NC 4.0，只在本地评测用，不能随数据集发布。

### ⑨ 发布

- **模型**：导出 safetensors、配置和分词器，每档写一份模型卡。
- **数据**：预训练和后训练数据都发到 Hugging Face，写清每个来源的许可证。
- **许可证要求**：见附录。

## 代码和产物的位置

```
training/
  train_pretrain.py  ① ②（--init 从存档出发；--data 可以写成「目录A:0.9,目录B:0.1」按比例混合）
  train_sft.py       ③ ④（--epochs 或 --steps）
  train_distill.py   ⑤ MOPD（--teachers 领域=存档 …；--merge-init 先做任务算术）
  train_rl.py        ⑥ GRPO 训练器，奖励可插拔
  trainer_utils.py   共用：设备、学习率曲线、优化器、存档
  rollout.py       采样一组回答、带工具的多轮对话
  rewards/         math、code（沙箱跑测试）、poetry（格律）、judge（评审模型）、tools（工具环境）
inference/
  template.py      对话格式和工具调用
  generate.py      生成；CUDA 上自动走 fast.py
  fast.py          预分配的静态缓存 + torch.compile（CUDA graph）
  chat.py          命令行里和模型对话（--tools 打开工具）
eval/suite.py      ⑧：留出文档上各类别的损失、题目正确率、评审打分、固定提问的回答，可选加上标准基准
eval/benchmarks.py ⑧：C-Eval、CMMLU、GSM8K 的下载、转换和打分
dataset/sft.py     后训练数据：三份 SFT 语料、三个专家的数据
dataset/rl.py      RL 题目、评测题、MOPD 提示
tools/demo.py      在 25M 的小模型上把整条链跑一遍
out/<档>/<阶段>/    每个阶段的存档
```

## 数据文件

预训练数据最后只留 `data/pretrain_{nano,mini,base,full}.parquet` 四个文件（这条只管预训练）。后训练数据放在旁边：
- `data/sft_{small,medium,large}.parquet`：三份 SFT 语料；
- `data/sft_expert_{math,code,literature}.parquet`：三个专家的数据；
- `data/mopd_prompts.parquet`：MOPD 的提示，带领域；
- `data/sft_heldout.parquet`：从 SFT 留出的 2% 问题，给 RL 和评测出题；
- `data/rl_{small,medium,large}.parquet`：RL 题目（题型见 ⑥）；
- `data/eval_tasks.parquet`：评测题，和 RL 题目、SFT 都不重叠。

原料在 `data/inputs/`（`python -m dataset.download sft rl extra`）：`sft/`（SFT 来源，每个前 10 万条）、`rl/`（RL 题目的原始数据）、`gujilab_translate.jsonl`。

## 还缺的

1. 外部 SFT 数据现在是每个来源前 10 万条（`data/inputs/sft/`，`python -m dataset.download sft`），全量的下载和转换还没做；UltraData-SFT-2605 要登录申请，没拿到。
2. 代码 RLVR 的题目来自 DeepCoder-Preview（带测试用例），接好了，演示里跑通了；题目还偏难，小模型基本拿不到分。
3. OpenAI 兼容的服务接口已经有了（`inference/serve.py`），还没真的接小米的 mimoagent 跑一遍。
4. CUDA 上的快速生成（CUDA graph）要在 PRO 6000 上用 `python -m tools.bench … --generate` 实测。
5. 标准基准已经接了 C-Eval、CMMLU、GSM8K；MMLU、MATH-500、HumanEval、MBPP、IFEval 还没接。

## 还没定的（文档里先按默认值写）

1. **退火的长度**：默认最后 10% 的步数（对话数据占一成已定）。
2. **三份 SFT 语料各放哪些数据集**：见下面附录的取舍。
3. **agent RL 的工具和任务**：先用自己做的，见上。
4. **RLAIF 的评分细则**：初稿在 `docs/rlaif-rubric.md`；用好回答和弄坏的回答做过一次自动检查（通用、古文、诗词分得清好坏，结果在细则文档里），还没和人工打分对比。
5. **古文诗词的评测**：内容待定。

## 附录：SFT 数据的取舍（2026-11-13 抽样后）

每个来源先抽 1000 条看指标和内容，用得上的再取前 10 万条（具体文件见 `dataset/sources.py` 的 `SFT`）。

| 数据集 | 用在 | 说明 |
|---|---|---|
| opencsg/smoltalk-chinese | 三档 | 中文多轮，质量稳定；回答偏列表式 |
| Mxode/Chinese-Instruct | 三档 | 中文，详细 |
| m-a-p/COIG-CQIA | 三档 | 人工整理的中文问答，偏短；许可证待确认 |
| deepctrl/deepctrl-sft-data（中文部分） | 三档 | MiniMind 的数据来源；许可证待确认 |
| Karasumikura/Ying-K1-SFT-Chinese-NonThink-102K | 三档 | 中文，Markdown 很多 |
| BAAI/Infinity-Instruct | 三档 | 抽到的是英文部分；部分答案来自 GPT-4 |
| HuggingFaceTB/smol-smoltalk、allenai/Dolci-Instruct-SFT | 三档 | 英文通用 |
| OpenCoder-LLM/opc-sft-stage2 | 三档、代码专家 | 短的 Python 题 |
| AI-MO/NuminaMath-CoT | 三档、数学专家、数学 RL 题 | 解题过程短；答案由 GPT-4o 改写 |
| Team-ACE/ToolACE | 三档 | 调用写成文字，构建时转成结构化的调用 |
| Congliu/Chinese-DeepSeek-R1-Distill-data-110k | 三档 | small 只留回答，medium、large 带 `<think>` |
| zake7749/OpenScience-Chinese-Reasoning-SFT | medium、large | 繁体，构建时转简体 |
| openbmb/UltraData-SFT-Agent-2609（工具调用部分） | medium、large | 英文客服多轮工具调用 |
| nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 | medium、large | 同一提示多个回答，按提示去重 |
| a-m-team/AM-Qwen3-Distilled | large | 很长，只留 8192 以内的 |
| nvidia/Nemotron-SFT-Math-v4、Nemotron-SFT-Competitive-Programming-v2 | large、专家 | 大多超过 8192，只留短的 |
| 自制：古籍库文白互译、按题写诗、鲁迅续写 | 三档、文学专家 | 来自我们自己洗好的数据 |
| 自制：自我认知 | 三档 | 名字待定 |
| ~~open-thoughts/OpenThoughts3-1.2M~~ | 不用 | 中位长度 1.5 万个词，超过上下文 |
| ~~openbmb/UltraData-SFT-2605~~ | 暂缺 | 要登录申请 |
| ~~UltraData-SFT-Agent 的通用智能体部分~~ | 不用 | 工具返回里有二进制乱码 |

## 参考

- MOPD: Multi-Teacher On-Policy Distillation — https://arxiv.org/abs/2606.30406
- CA-MOPD — https://arxiv.org/abs/2610.26950
- DeepSeek-V4 技术报告 — https://arxiv.org/abs/2606.19348
- MobileLLM-R1 — https://arxiv.org/abs/2509.24945
- SimpleRL-Zoo — https://arxiv.org/abs/2503.18892
- Understanding R1-Zero-Like Training（Dr. GRPO） — https://arxiv.org/abs/2503.20783
- There May Not be Aha Moment in R1-Zero-like Training — https://sail.sea.com/blog/articles/62
- Qwen3.8-27B — https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/
