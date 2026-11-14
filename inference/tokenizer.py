"""正式分词器的读取。推理、对话格式、训练数据准备都从这里拿；怎么训练分词器在 tools/tokenizer.py。"""
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
# 正式用的分词器（2026-11-12 定）：SentencePiece BPE，32768 词，字符覆盖率 0.99995
DEFAULT_MODEL = ROOT / "tokenizer" / "tokenizer.model"
# 同一个分词器的 HF 格式（python -m tools.tokenizer export-hf 从 .model 导出，逐词一致），给 gigatoken 这类快速实现用
DEFAULT_JSON = DEFAULT_MODEL.with_suffix(".json")
VOCAB_SIZE = 32768
UNK, BOS, EOS, PAD = 0, 1, 2, 3
CONTROL = ["<|im_start|>", "<|im_end|>"] + [f"<|reserved_{i}|>" for i in range(6)]


class Tokenizer:
    """训练好的分词器。encode 可以一次编一批（多线程）；decode 和原文一字不差。"""

    def __init__(self, path=DEFAULT_MODEL):
        self.path = Path(path)
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.vocab_size, self.eos_id = self.sp.get_piece_size(), self.sp.eos_id()

    def encode(self, text, num_threads: int = 1):
        return self.sp.encode(text, num_threads=num_threads) if isinstance(text, list) else self.sp.encode(text)

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids])

    def control_id(self, symbol: str) -> int:
        i = self.sp.piece_to_id(symbol)
        if i == UNK:
            raise KeyError(f"分词器里没有控制符 {symbol}")
        return i
