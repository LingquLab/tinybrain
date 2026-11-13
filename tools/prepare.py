"""把文本变成训练用的词数据目录。

    python -m tools.prepare --input "data/mix/out/nano/*.parquet" --out data/tokens/nano

--tokenizer 默认是仓库里的正式分词器 tokenizer/tokenizer.model；也可以换成别的 SentencePiece .model 文件，或者 bytes：
把 UTF-8 的每个字节当成一个词（0 是文档结束符，字节 b 记作 b+1，词表 257），只用来在没有分词器时跑通流程。
用 .model 时会把它复制进输出目录，数据目录自己就能说清楚是用哪个分词器切的。

对话数据（SFT）加 --chat：读 parquet 的 messages 列（每行一段对话，[{role, content}, …]），按 inference.template 的格式编码，
打包成长度为 --seq-len + 1 的行：

    python -m tools.prepare --chat --input data/sft.parquet --out data/tokens/sft-small --seq-len 2048
"""
import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from training.dataset import ShardWriter, write_meta

EOS = 0


def encode_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + 1


def decode_bytes(ids) -> str:
    return bytes(int(i) - 1 for i in ids if i != EOS).decode("utf-8", errors="replace")


def load_encoder(tokenizer: str, threads: int = 4):
    """返回 (一次编一批的函数, 词表大小, 结束符编号, 分词器的说明)。"""
    if tokenizer == "bytes":
        return (lambda texts: [encode_bytes(t) for t in texts]), 257, EOS, "bytes"
    if not tokenizer.endswith(".model") or not Path(tokenizer).exists():
        raise FileNotFoundError(f"--tokenizer 要么是 bytes，要么是一个存在的 SentencePiece .model 文件，收到的是 {tokenizer!r}")
    from inference.tokenizer import Tokenizer

    tok = Tokenizer(tokenizer)
    return (lambda texts: tok.encode(texts, num_threads=threads)), tok.vocab_size, tok.eos_id, "sentencepiece"


def default_tokenizer() -> str:
    from inference.tokenizer import DEFAULT_MODEL

    return str(DEFAULT_MODEL)


def prepare(inputs: list[str], out: str, tokenizer: str | None = None, val_every: int = 1000, shard_tokens: int = 100_000_000, threads: int = 4) -> dict:
    """读 parquet 文件的 text 列，每篇后面接一个结束符。每 val_every 篇里拿一篇放进验证集。返回各部分的词数。"""
    tokenizer = tokenizer or default_tokenizer()
    encode, vocab_size, eos, kind = load_encoder(tokenizer, threads)
    dtype = "uint16" if vocab_size <= 1 << 16 else "uint32"
    files = sorted(f for pattern in inputs for f in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有找到输入文件：{inputs}")
    out = Path(out)
    writers = {"train": ShardWriter(out, "train", dtype, shard_tokens), "val": ShardWriter(out, "val", dtype, shard_tokens)}
    n_docs = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["text"]):
            for ids in encode(batch.column("text").to_pylist()):
                split = "val" if val_every and n_docs % val_every == val_every - 1 else "train"
                writers[split].write(np.append(np.asarray(ids, dtype=np.int64), eos))
                n_docs += 1
    for w in writers.values():
        w.close()
    extra = {}
    if kind == "sentencepiece":
        shutil.copyfile(tokenizer, out / "tokenizer.model")
        extra["tokenizer_file"] = "tokenizer.model"
    write_meta(out, vocab_size, dtype, eos, kind, docs=n_docs, **extra)
    return {"docs": n_docs, "train": writers["train"].total, "val": writers["val"].total}


def prepare_chat(inputs: list[str], out: str, seq_len: int, tokenizer: str | None = None, val_every: int = 1000,
                 train_on: str = "assistant", chunk: int = 20_000) -> dict:
    """对话数据：编码、按 seq_len + 1 打包，写出 train/val 的 .ids.bin 和 .mask.bin。返回各部分的统计。"""
    from inference.template import ChatFormat
    from inference.tokenizer import Tokenizer
    from training.dataset import pack

    path = tokenizer or default_tokenizer()
    tok = Tokenizer(path)
    fmt = ChatFormat(tok)
    files = sorted(f for pattern in inputs for f in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有找到输入文件：{inputs}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = "uint16" if tok.vocab_size <= 1 << 16 else "uint32"
    handles = {s: (open(out / f"{s}.ids.bin", "wb"), open(out / f"{s}.mask.bin", "wb")) for s in ("train", "val")}
    pending = {"train": [], "val": []}
    stats = {s: dict(examples=0, dropped_too_long=0, tokens=0, target_tokens=0, rows=0) for s in ("train", "val")}
    n = 0

    def flush(split):
        ids, mask, st = pack(pending[split], seq_len + 1)
        handles[split][0].write(ids.astype(dtype).tobytes())
        handles[split][1].write(mask.tobytes())
        for k in stats[split]:
            stats[split][k] += st[k]
        pending[split] = []

    for path_ in files:
        for batch in pq.ParquetFile(path_).iter_batches(batch_size=1000, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                split = "val" if val_every and n % val_every == val_every - 1 else "train"
                pending[split].append(fmt.encode(messages, train_on))
                n += 1
                if len(pending[split]) >= chunk:
                    flush(split)
    for split in pending:
        if pending[split]:
            flush(split)
        for h in handles[split]:
            h.close()
    shutil.copyfile(path, out / "tokenizer.model")
    write_meta(out, tok.vocab_size, dtype, tok.eos_id, "sentencepiece", format="packed", row_len=seq_len + 1, train_on=train_on,
               tokenizer_file="tokenizer.model", stats=stats)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="把文本准备成训练用的词数据")
    parser.add_argument("--input", nargs="+", required=True, help="parquet 文件，可以写通配符")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=None, help="SentencePiece 的 .model 文件，或者 bytes；默认是 tokenizer/tokenizer.model")
    parser.add_argument("--val-every", type=int, default=1000, help="每多少篇里拿一篇放进验证集；0 表示不留")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chat", action="store_true", help="输入是对话（messages 列），打包成 SFT 用的数据")
    parser.add_argument("--seq-len", type=int, default=2048, help="--chat 时每行的片段长度")
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant", help="--chat 时哪些词算损失")
    args = parser.parse_args(argv)
    if args.chat:
        stats = prepare_chat(args.input, args.out, args.seq_len, args.tokenizer, args.val_every, args.train_on)
        for split, st in stats.items():
            print(f"{split}：{st['examples']} 段对话打成 {st['rows']} 行，要学的词 {st['target_tokens'] / 1e6:.2f}M，"
                  f"太长丢掉 {st['dropped_too_long']} 段")
        return
    stats = prepare(args.input, args.out, args.tokenizer, args.val_every, threads=args.threads)
    print(f"{stats['docs']} 篇；训练 {stats['train'] / 1e6:.1f}M 词，验证 {stats['val'] / 1e6:.2f}M 词 → {args.out}")


if __name__ == "__main__":
    main()
