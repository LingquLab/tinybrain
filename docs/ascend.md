# 昇腾 A5 推理和预训练

从 tinybrain 源码目录运行。先加载服务器上的 CANN 环境，使用已安装配套 PyTorch 和 torch_npu 的 Python 3.11+ 环境。推理读取发布包里的 safetensors 和分词器，准备训练数据时还需要 PyArrow。[TorchNPU 版本对应表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)

```bash
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())'
python -m pip install -e '.[artifact]' pyarrow
```

## 单卡推理

按 README 的下载方法取得 Small Chat RC1 的完整目录，保留清单列出的全部文件。以下命令在当前可见的第 0 张 NPU 上加载发布权重，用 KV cache 逐词续写。交互式对话可用第二条命令。

```bash
tinybrain generate tinybrain-small --device npu:0 --prompt '床前明月光，' --temperature 0 --max-new 8
tinybrain chat tinybrain-small --device npu:0
```

NPU 路径使用 eager PyTorch 和 FP32 参数。单卡推理没有跨 rank 通信。

## 训练并重新加载存档

已有切词数据时，可直接将 `--data` 指向含 `meta.json` 和 `train-*.bin` 的目录。以下命令从头训练 small 两步，验证前向、反向、优化器更新和存档读取。训练质量需要在正式语料和训练步数下评估。

```bash
tinybrain pretrain --device npu:0 --preset small --data data/tokens/nano \
  --seq-len 32 --batch-size 1 --steps 2 --eval-batches 2 --out out/a5-single
tinybrain generate out/a5-single/ckpt.pt --device npu:0 \
  --prompt '床前明月光，' --temperature 0 --max-new 8
```

## 双卡 ZeRO-1 与 TileXR

同一台服务器上，每个 torchrun 进程按 `LOCAL_RANK` 使用一张卡。省略 `--tilexr` 时设备集合通信使用 HCCL。启用 TileXR 时，使用与本机 SoC 匹配的 collectives 安装和同一份源码里的 Python 适配器。本项目对照的 TileXR 版本为 [79a29a4](https://github.com/LingquLab/TileXR/tree/79a29a4c5a31a6e92b0bce37b56401bc86564eda)。

```bash
# 在 TileXR 源码目录内构建
git submodule update --init --recursive
source scripts/common_env.sh
cmake -S . -B build-a5 -DTILEXR_BUILD_COLLECTIVES=ON \
  -DTILEXR_COLLECTIVES_SOC_TYPE=Ascend950 -DCMAKE_INSTALL_PREFIX="$PWD/install-a5"
cmake --build build-a5 -j8 && cmake --install build-a5
export TILEXR_INSTALL_PREFIX="$PWD/install-a5"
export PYTHONPATH="$PWD/integrations/vllm_ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TILEXR_INSTALL_PREFIX/lib64:$TILEXR_INSTALL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# 单节点示例。为并行作业分配不同的基础端口并预留 communicator 偏移端口。
export TILEXR_COMM_ID=127.0.0.1:29688
```

回到 tinybrain 源码目录，确认本次使用的两张物理卡空闲，并确保两个进程看到相同顺序的设备。先检查两个 rank 的归约和广播结果，再训练模型。

```bash
torchrun --standalone --nproc-per-node 2 tests/zero_collectives.py --device npu --tilexr
torchrun --standalone --nproc-per-node 2 -m training.train_pretrain \
  --device npu --tilexr --preset small --data data/tokens/nano \
  --seq-len 32 --batch-size 1 --steps 2 --eval-batches 2 --out out/a5-tilexr
```

| 数据 | 启用 `--tilexr` 后的操作 |
| --- | --- |
| 初始参数和 buffer、更新后的参数桶 | TileXR Broadcast，从初始化 rank 或参数所有者广播 |
| 损失、梯度使用标记、梯度平方和、MoE 负载 | TileXR AllReduce SUM |
| 梯度桶 | HCCL Reduce 到参数所有者，由所有者除以 rank 数 |
| 验证指标、优化器存档和启动元数据 | Gloo；验证指标保留 FP64，优化器状态在 CPU 上汇总 |

该适配器按单节点 2–8 个 rank 建组，8 是当前 native Broadcast 的 rank 上限。已有通信提交失败时会抛出错误。比较单卡和双卡存档时，固定种子、数据和每步总片段数，例如单卡用 `--grad-accum 2`，双卡用 `--grad-accum 1`。
