"""训练数据在硬盘上的样子，以及怎么读、怎么写（tools/prepare.py 写，train_*.py 读）。

一个目录：
    meta.json        {"vocab_size": 词表大小, "dtype": "uint16" 或 "uint32", "eos_id": 文档结束符, "tokenizer": 用的什么分词器}
    train-00000.bin  一长串词编号，文档和文档之间隔一个结束符
    val-00000.bin    留作验证的那部分，格式一样

训练按「轮」走（ordered_batch）：把数据切成不重叠的等长片段，每轮按种子打乱一次，依次取完——一轮里每个片段恰好一次。
第几批取哪些片段只由种子和批次序号决定，所以中断后接着跑和一口气跑完一样。验证集用 batch()：按种子随机截，每次同一批。
文件用内存映射取得长度；训练片段按小块读取，避免随机访问大语料时映射页面逐渐驻留在进程内存中。

对话数据（SFT）是另一种格式，meta.json 里 "format": "packed"：
    train.ids.bin、train.mask.bin   一行行等长的词编号和损失遮罩（每行 row_len 个），一段对话不会跨行
    val.ids.bin、val.mask.bin       验证集，格式一样
训练时每轮把所有行打乱一遍依次取；遮罩为 0 的位置，目标记成 -1，不算损失。open_data 按 meta 自动选对读法。
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from inference.tokenizer import PAD


class EpochOrder:
    """n 个片段的轮流顺序：第 k 个取的是第 k // n 轮的打乱顺序里第 k % n 个。只缓存当前这一轮的打乱。"""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.perm = n, seed, None, None

    def take(self, start: int, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.int64)
        for i, k in enumerate(range(start, start + count)):
            epoch, pos = divmod(k, self.n)
            if epoch != self.epoch:
                self.epoch, self.perm = epoch, np.random.default_rng([self.seed, epoch]).permutation(self.n)
            out[i] = self.perm[pos]
        return out


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else t.to(device)


class ShardWriter:
    """把一批批词编号接着写进 <目录>/<split>-00000.bin、-00001.bin……每个文件大约 shard_tokens 个词。"""

    def __init__(self, directory: Path, split: str, dtype: str, shard_tokens: int = 100_000_000):
        self.directory, self.split, self.dtype, self.shard_tokens = Path(directory), split, np.dtype(dtype), shard_tokens
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index, self.in_shard, self.total, self.file = 0, 0, 0, None

    def write(self, ids) -> None:
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() > np.iinfo(self.dtype).max):
            raise ValueError(f"词编号超出了 {self.dtype} 能表示的范围")
        if self.file is None or self.in_shard >= self.shard_tokens:
            self.close()
            self.file = open(self.directory / f"{self.split}-{self.index:05d}.bin", "wb")
            self.index, self.in_shard = self.index + 1, 0
        self.file.write(ids.astype(self.dtype).tobytes())
        self.in_shard += ids.size
        self.total += ids.size

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def write_meta(directory: Path, vocab_size: int, dtype: str, eos_id: int, tokenizer: str, **extra) -> None:
    meta = dict(vocab_size=vocab_size, dtype=dtype, eos_id=eos_id, tokenizer=tokenizer, **extra)
    (Path(directory) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


class TokenData:
    """一个目录里某个 split（train 或 val）的全部词。batch() 随机截出一批等长的片段。"""

    def __init__(self, directory: Path, split: str = "train"):
        directory = Path(directory)
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"{directory} 里没有 meta.json，这不是一个准备好的训练数据目录")
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.vocab_size = self.meta["vocab_size"]
        self.files = [f for f in sorted(directory.glob(f"{split}-*.bin")) if f.stat().st_size]
        self.shards = [np.memmap(f, dtype=self.meta["dtype"], mode="r") for f in self.files]
        self._dtype = np.dtype(self.meta["dtype"])
        self._readers = OrderedDict()
        self.n_tokens = sum(len(s) for s in self.shards)
        self._orders = {}

    def _read_window(self, shard: int, start: int, length: int) -> np.ndarray:
        """Read only this window, keeping at most 16 shard handles open."""
        reader = self._readers.pop(shard, None)
        if reader is None:
            if len(self._readers) >= 16:
                _, oldest = self._readers.popitem(last=False)
                oldest.close()
            reader = open(self.files[shard], "rb")
        self._readers[shard] = reader
        reader.seek(start * self._dtype.itemsize)
        block = reader.read(length * self._dtype.itemsize)
        if len(block) != length * self._dtype.itemsize:
            raise IOError(f"Short token read from {self.files[shard]} at token {start}")
        return np.frombuffer(block, dtype=self._dtype)

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self):
        for reader in getattr(self, "_readers", {}).values():
            reader.close()

    def windows(self, seq_len: int) -> int:
        """一轮有多少个片段：每个文件切成不重叠的 seq_len 段（每段多带一个词当最后一个目标），片段不跨文件。"""
        return sum((len(s) - 1) // seq_len for s in self.shards)

    def ordered_batch(self, batch_size: int, seq_len: int, index: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """训练用的第 index 批（从 0 数）：按轮依次取，一轮里每个片段恰好一次。返回 (输入, 目标)。"""
        import torch

        key = (seq_len, seed)
        if key not in self._orders:
            per_file = np.array([(len(s) - 1) // seq_len for s in self.shards])
            if not per_file.sum():
                raise ValueError(f"没有哪个数据文件比一个片段（{seq_len + 1} 个词）长")
            self._orders[key] = (EpochOrder(int(per_file.sum()), seed), np.concatenate([[0], np.cumsum(per_file)]))
        order, first = self._orders[key]
        rows = []
        for w in order.take(index * batch_size, batch_size):
            f = int(np.searchsorted(first, w, side="right")) - 1
            start = int(w - first[f]) * seq_len
            rows.append(self._read_window(f, start, seq_len + 1))
        block = _to_device(torch.from_numpy(np.stack(rows).astype(np.int64)), device)
        return block[:, :-1], block[:, 1:]

    def __bool__(self) -> bool:
        return bool(self.shards)

    def batch(self, batch_size: int, seq_len: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (输入, 目标)，都是 (batch_size, seq_len)；目标是输入往后错一位。同一个 seed 给出同一批。"""
        import torch

        room = np.array([len(s) - seq_len for s in self.shards], dtype=np.float64)  # 每个文件里有多少个可用的起点
        if (room <= 0).all():
            raise ValueError(f"没有哪个数据文件比一个片段（{seq_len + 1} 个词）长")
        room = np.clip(room, 0, None)
        rng = np.random.default_rng(seed)
        which = rng.choice(len(self.shards), size=batch_size, p=room / room.sum())
        rows = []
        for i in which:
            start = int(rng.integers(0, int(room[i])))
            rows.append(self._read_window(int(i), start, seq_len + 1))
        block = _to_device(torch.from_numpy(np.stack(rows).astype(np.int64)), device)
        return block[:, :-1], block[:, 1:]


class PackedData:
    """打包好的对话数据（tools.prepare --chat 写的）。一行就是一个片段；返回 (输入, 目标)，不算损失的位置目标是 -1。"""

    def __init__(self, directory: Path, split: str = "train"):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.vocab_size, self.row_len = self.meta["vocab_size"], self.meta["row_len"]
        ids_path, mask_path = directory / f"{split}.ids.bin", directory / f"{split}.mask.bin"
        if ids_path.exists() and ids_path.stat().st_size:
            self.ids = np.memmap(ids_path, dtype=self.meta["dtype"], mode="r").reshape(-1, self.row_len)
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r").reshape(-1, self.row_len)
        else:
            self.ids = self.mask = None
        self.rows = 0 if self.ids is None else len(self.ids)
        self.n_tokens = self.rows * (self.row_len - 1)
        self._orders = {}

    def __bool__(self) -> bool:
        return self.rows > 0

    def windows(self, seq_len: int) -> int:
        return self.rows

    def _check(self, seq_len: int) -> None:
        if seq_len != self.row_len - 1:
            raise ValueError(f"这份对话数据是按片段长度 {self.row_len - 1} 打包的，训练时 --seq-len 也要用这个值（收到 {seq_len}）")

    def _rows(self, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        ids = torch.from_numpy(self.ids[rows].astype(np.int64))
        mask = torch.from_numpy(self.mask[rows].astype(np.bool_))
        x, y = ids[:, :-1], ids[:, 1:].masked_fill(~mask[:, 1:], -1)
        return _to_device(x, device), _to_device(y, device)

    def batch(self, batch_size: int, seq_len: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """随机挑几行（验证用）。"""
        self._check(seq_len)
        return self._rows(np.sort(np.random.default_rng(seed).integers(0, self.rows, size=batch_size)), device)

    def ordered_batch(self, batch_size: int, seq_len: int, index: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """训练用的第 index 批：每轮把所有行打乱一遍依次取。"""
        self._check(seq_len)
        if seed not in self._orders:
            self._orders[seed] = EpochOrder(self.rows, seed)
        return self._rows(self._orders[seed].take(index * batch_size, batch_size), device)


def open_data(directory: Path, split: str = "train"):
    """按 meta.json 里的 format 选读法：packed 是对话数据，否则是预训练用的一长串词。"""
    meta = Path(directory) / "meta.json"
    if not meta.exists():
        raise FileNotFoundError(f"{directory} 里没有 meta.json，这不是一个准备好的训练数据目录")
    fmt = json.loads(meta.read_text(encoding="utf-8")).get("format")
    return PackedData(directory, split) if fmt == "packed" else TokenData(directory, split)


class MixedData:
    """几份数据按比例混着读：每一行先按权重抽一份数据，再从里面取一段。几份数据的词表必须一样。"""

    def __init__(self, parts: list[tuple[object, float]]):
        parts = [(d, w) for d, w in parts if d and w > 0]
        if not parts:  # 比如几份数据都没有验证集
            self.parts, self.vocab_size, self.meta, self.n_tokens = [], None, {}, 0
            return
        vocab = {d.vocab_size for d, _ in parts}
        if len(vocab) != 1:
            raise ValueError(f"几份数据的词表不一样：{sorted(vocab)}")
        self.parts = [d for d, _ in parts]
        self.weights = np.array([w for _, w in parts], dtype=np.float64) / sum(w for _, w in parts)
        self.vocab_size = vocab.pop()
        self.meta = dict(self.parts[0].meta, mixed_weights=[float(w) for w in self.weights])
        self.n_tokens = sum(d.n_tokens for d in self.parts)

    def __bool__(self) -> bool:
        return bool(self.parts)

    def windows(self, seq_len: int) -> int:
        return sum(d.windows(seq_len) for d in self.parts)

    def ordered_batch(self, batch_size: int, seq_len: int, index: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """按比例混合的数据没有自然的「一轮」，每批都按权重随机抽（由种子和批次序号决定）；轮数按总片段数折算。"""
        return self.batch(batch_size, seq_len, seed=seed * 1_000_000_007 + index, device=device)

    def batch(self, batch_size: int, seq_len: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        which = np.random.default_rng(seed).choice(len(self.parts), size=batch_size, p=self.weights)
        xs, ys = [], []
        for i, part in enumerate(self.parts):
            n = int((which == i).sum())
            if n:
                x, y = part.batch(n, seq_len, seed=seed * 7919 + i, device=device)
                xs.append(x)
                ys.append(y)
        return torch.cat(xs), torch.cat(ys)


def open_mixed(spec: str, split: str = "train"):
    """「目录」或「目录A:0.9,目录B:0.1」。只有一个目录时就是 open_data。"""
    if "," not in spec and ":" not in spec.replace('\\', '/').rsplit("/", 1)[-1]:
        return open_data(spec, split)
    parts = []
    for item in spec.split(","):
        path, _, weight = item.rpartition(":")
        if not path:
            raise ValueError(f"混合数据要写成 目录:权重,目录:权重，收到的是 {spec!r}")
        parts.append((open_data(path, split), float(weight)))
    return MixedData(parts)


def pack(examples, row_len: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """SFT 数据：把一段段 (词编号, 遮罩) 按顺序拼成长度为 row_len 的行：一段对话不会被拆到两行，放不下就另起一行，行尾用填充补齐（不算损失）。
    比 row_len 还长的对话丢掉。返回 (词编号, 遮罩, 统计)，前两个形状都是 (行数, row_len)。"""
    rows_ids, rows_mask, cur_ids, cur_mask = [], [], [], []
    stats = dict(examples=0, dropped_too_long=0, tokens=0, target_tokens=0)

    def close():
        pad = row_len - len(cur_ids)
        rows_ids.append(cur_ids + [PAD] * pad)
        rows_mask.append(cur_mask + [0] * pad)

    for ids, mask in examples:
        if len(ids) > row_len:
            stats["dropped_too_long"] += 1
            continue
        if len(cur_ids) + len(ids) > row_len:
            close()
            cur_ids, cur_mask = [], []
        cur_ids, cur_mask = cur_ids + list(ids), cur_mask + list(mask)
        stats["examples"] += 1
        stats["tokens"] += len(ids)
        stats["target_tokens"] += sum(mask)
    if cur_ids:
        close()
    n = len(rows_ids)
    stats["rows"] = n
    stats["fill"] = stats["tokens"] / max(1, n * row_len)  # 行里有多少是真内容（其余是填充）
    shape = (n, row_len)
    return (np.array(rows_ids, dtype=np.int64).reshape(shape), np.array(rows_mask, dtype=np.uint8).reshape(shape), stats)
