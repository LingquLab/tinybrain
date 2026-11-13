"""预训练数据的批量清洗：把 data/raw/<来源>/ 里的原始文件洗成 data/clean/<来源>/ 里的 parquet。来源的定义在 dataset/sources.py。

    python -m dataset.pretrain clean fw22_4_5 --workers 10 [--limit 100]
    python -m dataset.pretrain audit fw22_4_5 [--docs 30]

clean 可以重复运行，洗过的文件会跳过。audit 用来人工检查：随机抽几篇看开头和结尾，
统计最常重复出现的行（多半是没洗掉的网页外壳），以及各种可疑痕迹出现的比例。
"""
import argparse
import collections
import csv
import fnmatch
import gzip
import io
import json
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.clean import clean_with_reason
from dataset.sources import CLEAN, PRETRAIN, RAW

DROPPED_PER_FILE = 15  # 每个原始文件留几篇被丢掉的文档，供人工检查


RAW_SUFFIXES = (".parquet", ".jsonl.gz", ".json.gz", ".jsonl", ".json", ".zst", ".csv", ".txt")

SCHEMA = pa.schema([("text", pa.string()), ("quality", pa.float32()), ("heuristic", pa.float32()), ("est_tokens", pa.int32()),
                    ("n_chars", pa.int32()), ("sigs", pa.list_(pa.uint64())), ("row", pa.int32())])


def iter_rows(path: Path):
    """逐行读原始文件：parquet、csv、每行一个 JSON 的文本（可以是 gz、zst 压缩的），或者整个当成一行的 txt。"""
    name = path.name.lower()
    if name.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return
    if name.endswith(".txt"):
        yield {"text": path.read_text(encoding="utf-8")}
        return
    if name.endswith(".parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500):
            yield from batch.to_pylist()
        return
    if name.endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    elif name.endswith(".zst"):
        import zstandard

        f = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8")
    else:
        f = open(path, encoding="utf-8")
    with f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def raw_files(name: str) -> list[Path]:
    """这个来源要洗的原始文件：认得的格式，不是下划线开头的清单，没在下载中，并且符合来源自己限定的文件名（files 通配，skip 正则排除）。"""
    pattern, skip = PRETRAIN[name].get("files", "*"), PRETRAIN[name].get("skip")
    return sorted(p for p in (RAW / PRETRAIN[name].get("raw", name)).iterdir()
                  if p.name.lower().endswith(RAW_SUFFIXES) and not p.name.startswith("_")
                  and fnmatch.fnmatch(p.name, pattern) and not (skip and re.search(skip, p.name))
                  and not p.with_name(p.name + ".aria2").exists())


WRITE_EVERY = 50_000  # 攒够这么多篇就写出一组，大文件（比如十几 GB 的百科）也不会全压在内存里


def process_file(job) -> dict:
    """洗一个原始文件，写出一个 parquet 和一份被丢文档的小样。返回统计。"""
    name, in_path, out_path = job
    spec = PRETRAIN[name]
    stats = collections.Counter()
    cols = {field: [] for field in SCHEMA.names}
    dropped = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")

    def flush():
        writer.write_table(pa.table(cols, schema=SCHEMA))
        for v in cols.values():
            v.clear()

    rows = iter_rows(Path(in_path))
    for i, row in enumerate(spec["regroup"](rows) if spec.get("regroup") else rows):
        raws = spec["text"](row)
        score = spec["score"](row) if spec.get("score") else None
        for raw in [raws] if isinstance(raws, str) else raws:  # 一行可以给出多篇，比如一本书切成的各篇
            doc, reason = clean_with_reason(raw, spec["kind"], native_score=score, markdown=spec.get("markdown", False))
            stats["读入"] += 1
            stats["读入字符"] += len(raw)
            if doc is None:
                stats["丢:" + reason] += 1
                if len(dropped) < DROPPED_PER_FILE:
                    dropped.append({"reason": reason, "text": raw[:1500]})
                continue
            stats["保留"] += 1
            stats["保留字符"] += doc["n_chars"]
            stats["保留token"] += doc["est_tokens"]
            for field in SCHEMA.names:
                cols[field].append(i if field == "row" else doc[field])
            if len(cols["text"]) >= WRITE_EVERY:
                flush()
    flush()
    writer.close()
    tmp.rename(out_path)  # 写完整了才改名，中断不会留下半个文件
    (out_path.parent / "_dropped").mkdir(exist_ok=True)
    with open(out_path.parent / "_dropped" / (out_path.stem + ".jsonl"), "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return dict(stats)


def clean_source(name: str, workers: int, limit: int | None = None) -> dict:
    files = raw_files(name)[:limit]
    out_name = lambda p: p.name[:-len(next(x for x in RAW_SUFFIXES if p.name.lower().endswith(x)))] + ".parquet"
    jobs = [(name, str(p), str(CLEAN / name / out_name(p))) for p in files if not (CLEAN / name / out_name(p)).exists()]
    print(f"{name}：{len(files)} 个原始文件，其中 {len(jobs)} 个待洗，用 {workers} 个进程", flush=True)
    total, start = collections.Counter(), time.time()
    with Pool(workers) as pool:
        for done, stats in enumerate(pool.imap_unordered(process_file, jobs), 1):
            total.update(stats)
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} 个文件，保留 {total['保留'] / max(1, total['读入']):.1%}，"
                      f"{total['保留token'] / 1e9:.2f}B token，{time.time() - start:.0f}s", flush=True)
    summary = dict(total)
    if jobs:
        (CLEAN / name / "_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        reasons = {k[2:]: f"{v / total['读入']:.2%}" for k, v in sorted(total.items(), key=lambda kv: -kv[1]) if k.startswith("丢:")}
        print(f"完成：读入 {total['读入']} 篇，保留 {total['保留']} 篇（{total['保留'] / total['读入']:.1%}），丢弃原因 {reasons}，"
              f"字符减少 {1 - total['保留字符'] / total['读入字符']:.1%}", flush=True)
    return summary


SUSPECTS = {
    "网址": r"https?://|www\.", "来源编辑类": r"(来源|编辑|记者|作者|通讯员)[:：]", "转载声明": r"转载|版权|免责", "引导点击": r"点击(查看|进入|下载|这里|阅读)|扫码|二维码|关注(我们|公众号|微信)|长按识别",
    "联系方式": r"电话|QQ|微信|<手机号>|<邮箱>", "图片残留": r"图\s?\d|\(图\)|图片来源|如图所示|【图", "翻页残留": r"上一[篇页]|下一[篇页]|阅读全文|查看更多|展开全文",
    "省略截断": r"(\.\.\.|…)\s*$", "方括号标记": r"\[[^\]]{1,12}\]", "乱码痕迹": r"&nbsp;|&amp;|&#\d+;|\\u[0-9a-f]{4}|\\n|�", "HTML标签": r"</?[a-zA-Z][^>]{0,40}>", "星号": r"\*{2,}",
}


def audit(name: str, n_docs: int = 30, max_files: int = 40, seed: int = 0) -> dict:
    """抽查洗好的结果。返回 {docs, frequent_lines, suspects, dropped}，同时打印出来。"""
    rnd = random.Random(seed)
    files = sorted((CLEAN / name).glob("*.parquet"))
    files = rnd.sample(files, min(max_files, len(files)))
    texts = [t for f in files for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()]
    line_docs = collections.Counter()
    for t in texts:
        for line in {l.strip() for l in t.split("\n") if 2 <= len(l.strip()) <= 60}:
            line_docs[line] += 1
    frequent = [(line, c) for line, c in line_docs.most_common(60) if c >= max(3, len(texts) // 2000)]
    found = {k: [(t, m) for t in texts for m in [re.search(rx, t)] if m] for k, rx in SUSPECTS.items()}
    suspects = {k: len(v) / len(texts) for k, v in found.items()}
    docs = rnd.sample(texts, min(n_docs, len(texts)))
    dropped = [json.loads(l) for f in files for l in open(f.parent / "_dropped" / (f.stem + ".jsonl"), encoding="utf-8")]
    print(f"=== {name}：抽查 {len(files)} 个文件，{len(texts)} 篇")
    print("--- 可疑痕迹出现在多少比例的文档里：" + "；".join(f"{k} {v:.1%}" for k, v in suspects.items()))
    for k, hits in found.items():  # 每种痕迹看几个真实例子，才知道是该洗的还是正常内容
        for t, m in rnd.sample(hits, min(5, len(hits))):
            print(f"   [{k}] …" + t[max(0, m.start() - 30):m.end() + 30].replace("\n", " ⏎ ") + "…")
    print("--- 重复出现最多的行（行 × 出现的文档数）：")
    for line, c in frequent:
        print(f"   {c:6d}  {line}")
    print("--- 随机抽的文档（开头 ……… 结尾）：")
    for t in docs:
        flat = t.replace("\n", " ⏎ ")
        print("  ·", flat[:230], "………", flat[-170:] if len(flat) > 420 else "")
    by_reason = collections.defaultdict(list)
    for d in dropped:
        by_reason[d["reason"]].append(d["text"])
    print("--- 被丢掉的文档（每种原因抽几篇）：")
    for reason, items in by_reason.items():
        for t in rnd.sample(items, min(4, len(items))):
            print(f"  [{reason}]", t[:200].replace("\n", " ⏎ "))
    return {"docs": docs, "frequent_lines": frequent, "suspects": suspects, "dropped": dropped}


def main(argv=None):
    parser = argparse.ArgumentParser(description="数据集精洗流水线")
    parser.add_argument("command", choices=["clean", "audit"])
    parser.add_argument("source", choices=list(PRETRAIN))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="只处理前几个原始文件")
    parser.add_argument("--docs", type=int, default=30, help="audit 时抽几篇文档")
    parser.add_argument("--seed", type=int, default=0, help="audit 的随机种子，换一个就换一批文档")
    args = parser.parse_args(argv)
    if args.command == "clean":
        clean_source(args.source, args.workers, args.limit)
    else:
        audit(args.source, args.docs, seed=args.seed)


if __name__ == "__main__":
    main()
