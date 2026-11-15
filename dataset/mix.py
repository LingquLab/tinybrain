"""把洗好的各个来源混成最终数据集：去重 → 按配比和质量挑选 → 分档 → 写成四个文件 → 检查。

    python -m dataset.mix index    # 读所有洗好的文件，跨来源去重，记下每篇的质量和长度
    python -m dataset.mix plan     # 按配比算出每个来源每一档取多少，打印成表
    python -m dataset.mix write    # 写出 data/pretrain_nano.parquet、_mini、_base、_full 四个文件
    python -m dataset.mix verify   # 检查这四个文件，报告写到 out/mix/verify.json
    python -m dataset.mix card     # 按检查报告生成数据集说明 out/mix/README.md（发布到 ModelScope / Hugging Face 用）

四档是套娃关系：nano ⊂ mini ⊂ base ⊂ full。每个来源内部按质量从高到低排，最好的进 nano，往下依次进 mini、base；
full 是去重后的全部。
四个文件各自完整：要 mini 就只读 pretrain_mini.parquet。每个文件内部是打乱的。
"""

import argparse
import hashlib
import json
import resource
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset.sources import CLEAN, DATA, PRETRAIN, RL_INPUTS, ROOT, SFT

MIX_DIR = ROOT / "data" / "mix"   # 中间结果：索引、计划、分桶
FINAL_DIR = ROOT / "data"          # 最后只留下 data/pretrain_<档>.parquet 这四个文件

# 每一档的总 token 数（累计，大档包含小档），是用正式分词器（32768）切出来的真实 token 数。
# 前三档按 MIX 的配比挑；full 不设预算，取去重后的全部文档（2026-11-12 定，池子约 70B），比例跟着池子走。
TIERS = {"nano": 1e9, "mini": 5e9, "base": 20e9, "full": None}

# 清洗时记下的是估算的 token 数（汉字 ÷ 1.5 + 其他字符 ÷ 4）。正式分词器切出来的真实数 = 估算 × 这个比例，
# 是在按配比抽出的 793 篇留出文档上量的（tools.tokenizer eval）。法律没有留出文档，按中文文章算。
REAL_PER_EST = {"中文文章": 0.98, "百科": 1.25, "代码": 1.41, "数学": 1.29, "论文": 1.18, "古文和公版书": 1.51,
                "英文网页": 0.96, "英文书": 1.17, "合成": 1.03, "法律": 0.98}

# 类别 → (占总量的比例, {来源: 类别内部的权重})。哪个来源不够，差额先在类别内部补，类别也不够再按比例分给别的类别。
MIX = {
    "中文文章": (0.37, {"fw22_4_5": 34, "fw22_3_4": 3}),
    "百科": (0.155, {"wiki_zh": 7.5, "wiki_en": 5, "baidu_baike": 3}),
    "代码": (0.12, {"udcode_l2": 1, "udstack_v3_l2": 1}),
    "数学": (0.10, {"nemotron_math": 3, "ifm_math_rewrite": 2, "finemath": 2, "udmath_l2": 2, "numinamath": 1}),
    "论文": (0.08, {"arxiv": 3, "pes2o": 1}),
    "古文和公版书": (0.06, {"wikisource_zh": 3, "gujilab": 1, "greathangpt": 2, "shu": 3, "gutenberg_en_zh": 1, "gutenberg_zh": 1, "poetry": 1, "luxun": 1}),
    "英文网页": (0.05, {"ultrafineweb_en": 1}),
    "英文书": (0.04, {"gutenberg_en": 1}),
    "合成": (0.0225, {"cosmopedia_zh": 1}),
    "法律": (0.0025, {"law": 1}),
}

# 两篇重复时留谁：排在前面的来源优先。整理过的来源在前，网页抓取的在后。
PRIORITY = ["law", "poetry", "luxun", "gujilab", "greathangpt", "shu", "wikisource_zh", "gutenberg_zh", "gutenberg_en_zh", "gutenberg_en",
            "wiki_zh", "wiki_en", "baidu_baike", "numinamath", "nemotron_math", "ifm_math_rewrite", "arxiv", "pes2o", "cosmopedia_zh",
            "udcode_l2", "udstack_v3_l2", "finemath", "udmath_l2", "fw22_4_5", "fw22_3_4", "ultrafineweb_en"]

OUT_SCHEMA = pa.schema([("text", pa.string()), ("source", pa.string()), ("category", pa.string()), ("license", pa.string()),
                        ("quality", pa.float32()), ("est_tokens", pa.int32())])


def allocate(budget: float, capacity: dict, weights: dict) -> dict:
    """把 budget 按 weights 的比例分给各项；谁的 capacity 不够就给满，差额再按比例分给还没满的。"""
    result = dict.fromkeys(weights, 0.0)
    active = {k for k, w in weights.items() if w > 0 and capacity.get(k, 0) > 0}
    left = float(budget)
    while active and left > 1e-9:
        total = sum(weights[k] for k in active)
        full = {k for k in active if result[k] + left * weights[k] / total >= capacity[k]}
        if not full:
            for k in active:
                result[k] += left * weights[k] / total
            break
        for k in full:
            left -= capacity[k] - result[k]
            result[k] = float(capacity[k])
        active -= full
    return result


def plan(capacity: dict) -> dict:
    """capacity：每个来源去重后有多少估算 token。返回 {档: {来源: 这一档累计取多少估算 token}}。
    TIERS 里某一档的预算写成 None，表示这一档取去重后的全部文档。

    配比和每档的总量按真实 token 算：类别之间先换算成真实 token 再分，分到类别里以后再换回估算 token 分给各个来源。
    """
    out = {}
    for tier, budget in TIERS.items():
        if budget is None:  # 这一档取去重后的全部文档，不按配比
            out[tier] = {s: float(capacity.get(s, 0)) for _, (_, sources) in MIX.items() for s in sources}
            continue
        ratio = {c: REAL_PER_EST.get(c, 1.0) for c in MIX}
        cat_capacity = {c: sum(capacity.get(s, 0) for s in sources) * ratio[c] for c, (_, sources) in MIX.items()}
        cat_budget = allocate(budget, cat_capacity, {c: share for c, (share, _) in MIX.items()})
        out[tier] = {}
        for cat, (_, sources) in MIX.items():
            out[tier].update(allocate(cat_budget[cat] / ratio[cat], capacity, sources))
    return out


def find_duplicates(doc_ids: np.ndarray, sigs: np.ndarray, n_sigs: np.ndarray) -> np.ndarray:
    """找出重复的文档。doc_ids[i] 是第 i 个指纹属于哪篇（编号越小越优先保留），n_sigs[d] 是第 d 篇有几个指纹。

    一篇文档如果有两个指纹（指纹不足两个的，全部指纹）都在编号更小的文档里出现过，就算重复。
    只撞一个指纹不算：一句常见的话（比如一条法律条文）会出现在很多不同的文章里。
    """
    order = np.lexsort((doc_ids, sigs))
    s, d = sigs[order], doc_ids[order]
    seen_before = np.r_[False, s[1:] == s[:-1]]  # 同一个指纹里排第一的是最优先的那篇，其余都算撞上了
    hits = np.bincount(d[seen_before], minlength=len(n_sigs))
    return (n_sigs > 0) & (hits >= np.minimum(2, n_sigs))


def assign_tiers(quality: np.ndarray, tokens: np.ndarray, keep: np.ndarray, budgets: list, seed: int = 0) -> np.ndarray:
    """给一个来源的每篇文档定档：质量从高到低排，累计 token 落在第几个预算以内就是第几档；超出最大预算或不保留的是 -1。

    budgets 是各档的累计预算（从小到大）。质量相同的随机排，免得总是取到同一批文件。
    """
    tier = np.full(len(quality), -1, dtype=np.int8)
    idx = np.flatnonzero(keep)
    rnd = np.random.default_rng(seed).random(len(idx))
    idx = idx[np.lexsort((rnd, -quality[idx]))]
    cum = np.cumsum(tokens[idx], dtype=np.float64)
    t = np.searchsorted(np.asarray(budgets, dtype=np.float64) + 0.5, cum)
    tier[idx] = np.where(t < len(budgets), t, -1)
    return tier


def _source_files(name: str) -> list[Path]:
    return sorted((CLEAN / name).glob("*.parquet"))


def build_index() -> dict:
    """读所有洗好的文件里的质量、长度和指纹，跨来源去重，结果存到 data/mix/index/<来源>.npz。返回每个来源的统计。"""
    names = [s for s in PRIORITY if _source_files(s)]
    missing = sorted({s for _, sources in MIX.values() for s in sources} - set(names))
    if missing:
        print("还没有洗好的来源（先不算它们）：", "、".join(missing), flush=True)
    per, all_ids, all_sigs, all_n, base = {}, [], [], [], 0
    for name in names:
        file_id, pos, quality, tokens, n_sigs, sig_doc, sig_val = [], [], [], [], [], [], []
        start = 0
        for fi, path in enumerate(_source_files(name)):
            t = pq.read_table(path, columns=["quality", "est_tokens", "sigs"])
            n = t.num_rows
            col = t.column("sigs").combine_chunks()
            counts = np.diff(col.offsets.to_numpy())
            file_id.append(np.full(n, fi, np.int32)); pos.append(np.arange(n, dtype=np.int32))
            quality.append(t.column("quality").to_numpy()); tokens.append(t.column("est_tokens").to_numpy())
            n_sigs.append(counts.astype(np.int8))
            sig_doc.append(np.repeat(np.arange(start, start + n, dtype=np.int64), counts))
            sig_val.append(col.values.to_numpy())
            start += n
        d = dict(file_id=np.concatenate(file_id), pos=np.concatenate(pos), quality=np.concatenate(quality).astype(np.float32),
                 tokens=np.concatenate(tokens).astype(np.int32), n_sigs=np.concatenate(n_sigs))
        rank = np.empty(start, np.int64)
        rank[np.argsort(-d["quality"], kind="stable")] = np.arange(start)  # 来源内部质量高的优先保留
        all_ids.append(base + rank[np.concatenate(sig_doc)]); all_sigs.append(np.concatenate(sig_val))
        n_by_rank = np.empty(start, np.int8); n_by_rank[rank] = d["n_sigs"]
        all_n.append(n_by_rank)
        per[name] = (d, rank, base)
        base += start
        print(f"  读完 {name}：{start} 篇，{d['tokens'].sum() / 1e9:.2f}B token", flush=True)
    dup = find_duplicates(np.concatenate(all_ids), np.concatenate(all_sigs), np.concatenate(all_n))
    (MIX_DIR / "index").mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, (d, rank, start) in per.items():
        d["dup"] = dup[start + rank]
        np.savez(MIX_DIR / "index" / f"{name}.npz", **d)
        stats[name] = {"docs": int(len(rank)), "tokens": int(d["tokens"].sum()), "dup_docs": int(d["dup"].sum()),
                       "unique_tokens": int(d["tokens"][~d["dup"]].sum())}
        print(f"  {name}：重复 {stats[name]['dup_docs'] / max(1, len(rank)):.2%}，去重后 {stats[name]['unique_tokens'] / 1e9:.2f}B token", flush=True)
    (MIX_DIR / "index" / "_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def make_plan(seed: int = 0) -> dict:
    """按配比给每个来源的每篇文档定档，存到 data/mix/index/<来源>.tier.npy，打印各档各来源的 token 数。返回这张表。"""
    stats = json.loads((MIX_DIR / "index" / "_stats.json").read_text(encoding="utf-8"))
    budgets = plan({s: v["unique_tokens"] for s, v in stats.items()})
    table = {}
    in_mix = {src for _, (_, sources) in MIX.items() for src in sources}
    for name in [n for n in stats if n in in_mix]:  # 洗好了但配比里没有的来源不进任何一档
        d = np.load(MIX_DIR / "index" / f"{name}.npz")
        tier = assign_tiers(d["quality"], d["tokens"], ~d["dup"], [budgets[t].get(name, 0.0) for t in TIERS], seed)
        np.save(MIX_DIR / "index" / f"{name}.tier.npy", tier)
        table[name] = [int(d["tokens"][(tier >= 0) & (tier <= i)].sum()) for i in range(len(TIERS))]
    totals = [sum(row[i] for row in table.values()) for i in range(len(TIERS))]
    real = {name: [v * REAL_PER_EST.get(next(c for c, (_, s) in MIX.items() if name in s), 1.0) for v in row] for name, row in table.items()}
    real_totals = [sum(row[i] for row in real.values()) for i in range(len(TIERS))]
    print("估算 token（括号里是占比）；最后一行是换算成正式分词器的真实 token：")
    print(f"{'来源':<18}{'去重后':>9}" + "".join(f"{t:>16}" for t in TIERS))
    for cat, (_, sources) in MIX.items():
        for name in sources:
            if name in table:
                cells = "".join(f"{v / 1e9:>9.3f}B {v / max(1, totals[i]):>5.1%}" for i, v in enumerate(table[name]))
                print(f"{name:<18}{stats[name]['unique_tokens'] / 1e9:>8.2f}B{cells}   {cat}")
    print(f"{'合计':<18}{'':>9}" + "".join(f"{v / 1e9:>9.3f}B {'':>5}" for v in totals))
    print(f"{'真实 token':<16}{'':>9}" + "".join(f"{v / 1e9:>9.3f}B {'':>5}" for v in real_totals))
    (MIX_DIR / "index" / "_plan.json").write_text(json.dumps({"tiers": list(TIERS), "tokens": table}, ensure_ascii=False, indent=1), encoding="utf-8")
    return table


def final_path(tier: str) -> Path:
    return FINAL_DIR / f"pretrain_{tier}.parquet"


def _bucket_path(part: int, bucket: int) -> Path:
    return MIX_DIR / "buckets" / f"part{part}" / f"{bucket:04d}.parquet"


def _selected(only) -> tuple[list[str], int]:
    """要处理哪几档，以及用到的最大档号（第 t 档要用到最早进第 0…t 档的全部文档）。"""
    names = list(TIERS) if not only else [t for t in TIERS if t in only]
    unknown = set(only or ()) - set(TIERS)
    if unknown or not names:
        raise ValueError(f"没有这些档：{sorted(unknown)}；可选 {list(TIERS)}")
    return names, max(list(TIERS).index(n) for n in names)


def _read_rows(path: Path, positions: np.ndarray, columns: list[str], batch_size: int = 10_000):
    """按行号（从小到大）挑出一个 parquet 文件里的行：每次读 batch_size 行，从中挑出要的，给出一个表。
    不整个文件、也不整个分组（row group）地读：有的来源是单个 8 GB 的文件（百科），有的文件一个分组就上百万行，
    整块读出再挑，正文会超过普通字符串列 2 GB 的上限，也太占内存。"""
    start = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        n = batch.num_rows
        lo, hi = np.searchsorted(positions, [start, start + n])
        if hi > lo:
            yield pa.Table.from_batches([batch]).take(pa.array(positions[lo:hi] - start))
        start += n


def write_tiers(seed: int = 0, buckets: int = 256, flush_bytes: int = 4 << 20, only: list[str] | None = None) -> dict:
    """把定好档的文档写成四个独立的文件 data/pretrain_<档>.parquet，每个都是完整的一档（mini 里包含 nano 的全部文档）。

    分两步，内存里不需要放下整个数据集：
      1. 每篇文档抽一个随机数 u，按「最早进哪一档」和 u 落在第几个桶，写进 data/mix/buckets/part<档>/<桶>.parquet；
      2. 逐个读桶：桶里的文档打乱一次，第 t 档取「最早进第 0…t 档」的那些，四个文件同时往后写；读完的桶马上删掉。
    四个文件用同一套打乱，所以小档的文档在大档里是均匀散开的。先写成 .tmp，写完整了才改名。返回每档的文档数。
    only：只写这几档（比如磁盘不够时先写 nano、mini、base），只用到这几档需要的文档。
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard) if hard != resource.RLIM_INFINITY else 8192, hard))  # 同时开着几百上千个桶
    plan_table = json.loads((MIX_DIR / "index" / "_plan.json").read_text(encoding="utf-8"))["tokens"]
    all_tiers = list(TIERS)
    tiers, max_t = _selected(only)
    rng = np.random.default_rng(seed)
    writers, buffers, sizes = {}, {}, {}

    def flush(key):
        if key not in writers:
            path = _bucket_path(*key)
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[key] = pq.ParquetWriter(path, OUT_SCHEMA, compression="zstd")
        writers[key].write_table(pa.Table.from_batches(buffers.pop(key), schema=OUT_SCHEMA))
        sizes[key] = 0

    if (MIX_DIR / "buckets").exists():
        shutil.rmtree(MIX_DIR / "buckets")
    for name in plan_table:
        d = np.load(MIX_DIR / "index" / f"{name}.npz")
        tier = np.load(MIX_DIR / "index" / f"{name}.tier.npy")
        spec = PRETRAIN[name]
        category = next(c for c, (_, sources) in MIX.items() if name in sources)
        for fi, path in enumerate(_source_files(name)):
            rows = np.flatnonzero((d["file_id"] == fi) & (tier >= 0) & (tier <= max_t))
            if not len(rows):
                continue
            parts_of_rows, done = tier[rows], 0
            for chunk in _read_rows(path, d["pos"][rows], ["text", "quality", "est_tokens"]):
                n = chunk.num_rows
                table = pa.table({"text": chunk.column("text"), "source": pa.array([name] * n), "category": pa.array([category] * n),
                                  "license": pa.array([spec["license"]] * n), "quality": chunk.column("quality"),
                                  "est_tokens": chunk.column("est_tokens")}, schema=OUT_SCHEMA)
                bucket = rng.integers(0, buckets, size=n)
                part = parts_of_rows[done:done + n]
                done += n
                for key in set(zip(part.tolist(), bucket.tolist())):
                    piece = table.filter(pa.array((part == key[0]) & (bucket == key[1])))
                    buffers.setdefault(key, []).extend(piece.to_batches())
                    sizes[key] = sizes.get(key, 0) + piece.nbytes
                    if sizes[key] > flush_bytes:
                        flush(key)
        print(f"  分桶写完 {name}", flush=True)
    for key in list(buffers):
        flush(key)
    for w in writers.values():
        w.close()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    # 四个文件同时写：每读一个桶，就把它分给各档，写完立刻删掉这个桶。磁盘峰值只比四个文件本身多一个桶。
    tmp = {name: final_path(name).with_suffix(".tmp") for name in tiers}
    out_writers = {name: pq.ParquetWriter(tmp[name], OUT_SCHEMA, compression="zstd") for name in tiers}
    counts = dict.fromkeys(tiers, 0)
    for b in range(buckets):
        parts = {p: pq.read_table(_bucket_path(p, b)) for p in range(max_t + 1) if _bucket_path(p, b).exists()}
        if parts:
            table = pa.concat_tables(list(parts.values()))
            part_of = np.concatenate([np.full(t.num_rows, p, dtype=np.int8) for p, t in parts.items()])
            order = rng.permutation(table.num_rows)
            table, part_of = table.take(pa.array(order)), part_of[order]
            for name in tiers:  # 第 t 档 = 最早进第 0…t 档的文档，顺序沿用同一个打乱
                rows = np.flatnonzero(part_of <= all_tiers.index(name))
                if len(rows):
                    out_writers[name].write_table(table.take(pa.array(rows)), row_group_size=50_000)
                    counts[name] += len(rows)
            for p in parts:
                _bucket_path(p, b).unlink()
    for name in tiers:
        out_writers[name].close()
        tmp[name].rename(final_path(name))
        print(f"  写完 {final_path(name)}：{counts[name]} 篇", flush=True)
    shutil.rmtree(MIX_DIR / "buckets")
    return counts


def _doc_hashes(path: Path) -> tuple[np.ndarray, dict, dict]:
    """一个文件里每篇文档正文的 64 位指纹，以及按来源统计的篇数和估算 token 数。"""
    hashes, docs, tokens = [], {}, {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=["text", "source", "est_tokens"]):
        texts, sources, est = batch.column("text").to_pylist(), batch.column("source").to_pylist(), batch.column("est_tokens").to_numpy()
        hashes.append(np.fromiter((int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little") for t in texts),
                                  dtype=np.uint64, count=len(texts)))
        for s, e in zip(sources, est):
            docs[s] = docs.get(s, 0) + 1
            tokens[s] = tokens.get(s, 0) + int(e)
    return (np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)), docs, tokens


def verify_tiers(sample_docs: int = 2000, tokenizer: Path | None = None, only: list[str] | None = None, max_dup_frac: float = 1e-5) -> dict:
    """检查四个文件：都能读、列对、每个来源的估算 token 数和计划一模一样、文件内部几乎没有完全相同的文档、小档完整地包含在大档里、
    顺序是打乱的；再随机抽一些文档用正式分词器切，量真实 token 数。全部通过才返回，否则抛出错误说明哪里不对。

    完全相同的文档最多容忍 max_dup_frac（默认十万分之一），数量记在报告里：去重靠每篇最长的几句话当指纹，
    句子太短、取不出指纹的文档（比如满是缺字符号的残诗）不参与去重，会漏下极少数。"""
    plan_table = json.loads((MIX_DIR / "index" / "_plan.json").read_text(encoding="utf-8"))["tokens"]
    report, previous, problems = {}, None, []
    names, _ = _selected(only)
    for name in names:
        t = list(TIERS).index(name)
        path = final_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在")
        pf = pq.ParquetFile(path)
        if pf.schema_arrow != OUT_SCHEMA:
            problems.append(f"{name}：列不对 {pf.schema_arrow}")
        hashes, docs, tokens = _doc_hashes(path)
        want = {s: row[t] for s, row in plan_table.items() if row[t]}
        if tokens != want:
            diff = {s: (tokens.get(s, 0), want.get(s, 0)) for s in set(tokens) | set(want) if tokens.get(s, 0) != want.get(s, 0)}
            problems.append(f"{name}：各来源的 token 数和计划不一致（实际, 计划）{diff}")
        unique = np.unique(hashes)
        dups = len(hashes) - len(unique)
        if dups > max_dup_frac * len(hashes):
            problems.append(f"{name}：有 {dups} 篇完全重复，超过了容忍的 {max_dup_frac:.0e}")
        if previous is not None and not np.isin(previous, unique, assume_unique=True).all():
            problems.append(f"{name}：没有完整包含上一档的全部文档")
        head = pq.read_table(path, columns=["source"]).column("source").to_pylist()
        if len(head) > 100 and len(set(head[: max(10, len(head) // 100)])) < min(len(docs), 3):
            problems.append(f"{name}：开头一段几乎全是同一个来源，像是没打乱")
        report[name] = dict(docs=int(len(hashes)), est_tokens=int(sum(tokens.values())), bytes=path.stat().st_size, sources=docs, source_tokens=tokens,
                            exact_duplicates=int(dups))
        previous = unique
    if problems:
        raise RuntimeError("四个文件的检查没通过：\n  " + "\n  ".join(problems))

    from inference.tokenizer import DEFAULT_MODEL, Tokenizer  # 只在最后量真实 token 时用

    tok = Tokenizer(tokenizer or DEFAULT_MODEL)
    for name in names:  # 文件写的时候已经打乱过（上面也检查了），开头 sample_docs 篇就是均匀的随机样本；不把整个文件读进内存
        batch = next(pq.ParquetFile(final_path(name)).iter_batches(batch_size=sample_docs, columns=["text", "est_tokens"]))
        real = sum(len(ids) for ids in tok.encode(batch.column("text").to_pylist(), num_threads=4))
        est = int(batch.column("est_tokens").to_numpy().sum())
        report[name]["real_tokens_estimate"] = int(report[name]["est_tokens"] * real / max(1, est))
    return report


SOURCE_LICENSE_NOTE = """这是一个混合数据集，没有统一的许可证：每篇文档都带着 `license` 列，写的是它所属来源的许可证（照上游写的）。
使用时请分别遵守各来源的许可：
- 维基百科、维基文库是 CC BY-SA 3.0，要求署名，并且以相同方式共享（再发布时用相同的许可证）；
- 标成「待确认」的来源，上游没有写清楚许可证，可能有版权风险，商用前请自行确认，或者按 `source` 列把它们去掉；
- arxiv 的论文各自有许可证（per-paper），以论文本身为准。"""


# 数据集说明的头部。ModelScope 只认「领域: 任务: language/type」这种嵌套写法（网页上的领域、任务、语言标签从这里来）；
# license、language、task_categories、size_categories、tags 是 Hugging Face 的写法，以后传 HF 用，ModelScope 也认 license。
CARD_HEADER = """---
license: other
language:
- zh
- en
task_categories:
- text-generation
size_categories:
- 10M<n<100M
tags:
- pretraining
- sft
- reinforcement-learning
- chinese
text:
  text-generation:
    language:
    - zh
    - en
    type:
    - pretraining
    - sft
    - rl
---"""


def dataset_card(report: dict) -> str:
    """数据集说明（Markdown，带 ModelScope / Hugging Face 认的头部信息）。report 是 verify_tiers 的结果：
    sources 是每个来源的篇数，source_tokens 是每个来源的估算 token 数。"""
    tiers = [t for t in TIERS if t in report]
    cat_of = {s: c for c, (_, sources) in MIX.items() for s in sources}
    unclear = [s for s in cat_of if "待确认" in PRETRAIN.get(s, {}).get("license", "") and any(report[t]["source_tokens"].get(s) for t in tiers)]
    lines = CARD_HEADER.splitlines() + ["",
             "# tinybrain dataset", "",
             "tinybrain 的中文为主的混合预训练数据集。清洗、去重、分档的代码和模型在 https://github.com/logitcai/tinybrain 。", "",
             "从 26 个公开来源清洗、跨来源去重后，按配比和质量挑选出的四档预训练数据。四档是套娃关系 nano ⊂ mini ⊂ base ⊂ full："
             "每个来源内部按质量从高到低排，最好的进 nano，往下依次进 mini、base；full 是去重后的全部。每个文件各自完整、内部打乱，只读一个就够。", "",
             "## 文件", "",
             "token 数有两种：「真实」是用本项目的分词器（SentencePiece BPE，32768 词）切出来的（抽样换算）；「估算」是清洗时按 汉字 ÷ 1.5 + 其他字符 ÷ 4 记的。", "",
             "| 文件 | 真实 token | 估算 token | 篇数 | 大小 |", "|---|---|---|---|---|"]
    for t in tiers:
        r = report[t]
        lines.append(f"| `{final_path(t).name}` | {r['real_tokens_estimate'] / 1e9:.2f}B | {r['est_tokens'] / 1e9:.2f}B | {r['docs']:,} | {r['bytes'] / 1e9:.2f} GB |")
    lines += ["", "nano、mini、base 按下面的配比挑（按真实 token 算）；full 不设上限，各类别有多少放多少，所以比例和配比不一样。", "",
              "| 类别 | 配比 |", "|---|---|"] + [f"| {c} | {share:.1%} |" for c, (share, _) in MIX.items()]
    lines += ["", "## 列", "",
              "| 列 | 说明 |", "|---|---|",
              "| `text` | 正文 |", "| `source` | 来源（见下表） |", "| `category` | 类别 |", "| `license` | 这篇文档所属来源的许可证 |",
              "| `quality` | 质量分（来源自带的分数或本项目的启发式分数，约 0 到 1；各来源的尺度不同，只在同一来源内部可比） |",
              "| `est_tokens` | 估算 token 数 |", "",
              "## 来源", "",
              "表里是每个来源在各档里的估算 token 数（单位 B，十亿）。", "",
              "| 来源 | 类别 | 出自 | 许可 | " + " | ".join(tiers) + " |", "|---|---|---|---|" + "---|" * len(tiers)]
    for c, (_, sources) in MIX.items():
        for src in sources:
            if not any(report[t]["source_tokens"].get(src) for t in tiers):
                continue
            spec = PRETRAIN.get(src, {})
            cells = " | ".join(f"{report[t]['source_tokens'].get(src, 0) / 1e9:.3f}" for t in tiers)
            lines.append(f"| {src} | {c} | {spec.get('upstream', '')} | {spec.get('license', '')} | {cells} |")
    lines += ["", "## 许可", "", SOURCE_LICENSE_NOTE]
    if unclear:
        lines += ["", f"许可证待确认的来源：{'、'.join(unclear)}。"]
    lines += ["", "## 怎么做的", "",
              "1. 清洗（`dataset/clean.py`、`dataset/pretrain.py`；来源见 `dataset/sources.py`）：修乱码、统一标点和空白，去掉太短、不成句、语言不对、广告、重复太多、有屏蔽字的文档；"
              "手机号、身份证号、邮箱换成占位符；古籍和书按章节切成篇。",
              "2. 去重（`mix.py index`）：每篇取最长的 3 句当指纹，跨来源有 2 句撞上就算重复，按来源的优先级留一篇（整理过的来源优先于网页抓取）。",
              "3. 分档（`mix.py plan`、`write`）：按配比算出每个来源每档取多少，来源内部按质量从高到低取。",
              "4. 检查（`mix.py verify`）：每个来源的 token 数和计划一致、文件内部没有重复、小档完整包含在大档里、顺序是打乱的。", "",
              "## 已知问题", "",
              "- 清洗是「够用就好」：网页来源里还会有少量广告、导航之类的残留。",
              "- 质量分只在同一来源内部可比。"] + [
              f"- {t}：有 {report[t]['exact_duplicates']} 篇和别的文档完全相同（去重靠句子指纹，太短、取不出指纹的文档会漏下，比如满是缺字符号的残诗），"
              f"占 {report[t]['exact_duplicates'] / report[t]['docs']:.1e}。" for t in tiers if report[t].get("exact_duplicates")] + [
              "- 法院裁判文书因为含有当事人的姓名、出生日期和住址，没有放进来。"]
    return "\n".join(lines) + "\n"


# 后训练数据：发布到 posttrain/ 下的文件和它们是什么（dataset/sft.py、dataset/rl.py 写出来的）
POSTTRAIN_FILES = {
    "sft_small": "SFT 小档：去掉推理过程，每条最长 2048 词，外部来源只留质量好的一半",
    "sft_medium": "SFT 中档：推理过程写成 <think>…</think>，每条最长 4096 词",
    "sft_large": "SFT 大档：再加上更长的推理数据，每条最长 8192 词",
    "sft_heldout": "从每个外部来源留出的约 2%，不进 SFT（给 RL 出题和评测用）",
    "sft_expert_math": "数学专家的 SFT", "sft_expert_code": "代码专家的 SFT", "sft_expert_literature": "文学专家的 SFT（文言、诗词）",
    "rl_small": "RL 小档：指令跟随、阅读、诗词、工具调用、评审题",
    "rl_medium": "RL 中档：再加上数学、代码、逻辑、科学", "rl_large": "RL 大档：题型和中档一样，题目另挑",
    "eval_tasks": "评测题：每种题型留出一份，和 RL、SFT 都不重复", "mopd_prompts": "MOPD（多老师在线蒸馏）用的提示，只有问题",
}
# RL 每种题从哪来：RL_INPUTS 里的名字会换成上游数据集和许可，其余是写好的说明
RL_SOURCES = {"数学": ["bigmath", "deepscaler", "gsm8k_zh"], "代码": ["deepcoder"], "科学": ["arc", "openscience"],
              "指令跟随": ["ifeval", "SFT 留出的中文问题，配上中文约束"], "阅读": ["cmrc2018"], "逻辑": ["reasoning-gym 现场生成"],
              "诗词": ["从诗词库出题（Werneror/Poetry，mit）"], "工具调用": ["程序生成；检索资料取自维基百科（cc-by-sa-3.0）"],
              "通用": ["SFT 留出的问题（许可随来源）"], "写作": ["鲁迅作品续写（公有领域）"]}


def _upstream(spec: dict) -> str:
    how = spec.get("download", {})
    return spec.get("upstream") or how.get("hf") or how.get("ms") or ""


def posttrain_card(data_dir: Path = DATA) -> str:
    """数据集说明的后训练部分：每个文件的条数和大小、SFT 每个来源的条数和许可、RL 每种题的来源。没有的文件不列。"""
    files = [(n, data_dir / f"{n}.parquet") for n in POSTTRAIN_FILES if (data_dir / f"{n}.parquet").exists()]
    if not files:
        return ""
    lines = ["## 后训练数据（`posttrain/`）", "",
             "对话都在 `messages` 列：`[{role, content, tool_calls}, …]`，role 是 system、user、assistant、tool；工具调用是 JSON 字符串。"
             "SFT 文件另有 `source`（来源）、`domain`（领域）、`license`（这条所属来源的许可）、`tokens`（真实长度）。"
             "RL 和评测题另有 `reward`（判分方式）、`answer`、`tests` 等判分用的字段。", "",
             "| 文件 | 条数 | 大小 | 内容 |", "|---|---|---|---|"]
    for name, path in files:
        meta = pq.ParquetFile(path).metadata
        lines.append(f"| `posttrain/{name}.parquet` | {meta.num_rows:,} | {path.stat().st_size / 1e6:.0f} MB | {POSTTRAIN_FILES[name]} |")
    biggest = next((p for n, p in reversed(files) if n.startswith("sft_") and "source" in pq.ParquetFile(p).schema_arrow.names
                    and n in ("sft_small", "sft_medium", "sft_large")), None)
    unclear = []
    if biggest is not None:
        t = pq.read_table(biggest, columns=["source", "license"])
        counts, licences = {}, {}
        for s, lic in zip(t.column("source").to_pylist(), t.column("license").to_pylist()):
            counts[s] = counts.get(s, 0) + 1
            licences.setdefault(s, lic)
        lines += ["", f"### SFT 的来源（条数按 `{biggest.stem}`）", "", "| 来源 | 出自 | 许可 | 条数 | 备注 |", "|---|---|---|---|---|"]
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            spec = SFT.get(s, {})
            lines.append(f"| {s} | {_upstream(spec) or '本项目自制'} | {licences[s]} | {n:,} | {spec.get('note', '')} |")
            if "待确认" in licences[s]:
                unclear.append(s)
    rl = next((p for n, p in reversed(files) if n.startswith("rl_")), None)
    if rl is not None:
        domains = {}
        for d in pq.read_table(rl, columns=["domain"]).column("domain").to_pylist():
            domains[d] = domains.get(d, 0) + 1
        lines += ["", f"### RL 题目的来源（条数按 `{rl.stem}`）", "", "| 题型 | 条数 | 来源 |", "|---|---|---|"]
        for d, n in sorted(domains.items(), key=lambda kv: -kv[1]):
            parts = [f"{_upstream(RL_INPUTS[x])}（{RL_INPUTS[x]['license']}）" if x in RL_INPUTS else x for x in RL_SOURCES.get(d, ["本项目自制"])]
            lines.append(f"| {d} | {n:,} | {'、'.join(parts)} |")
    lines += ["", "### 后训练数据的许可", "",
              "和预训练数据一样没有统一的许可：每条 SFT 的 `license` 列写着它所属来源的许可，RL 题目的来源和许可见上表。",
              "- 备注里写着「由 GPT 生成」的来源，回答或解题过程是 OpenAI 的模型写的；它们自己的许可允许再分发，"
              "但 OpenAI 的使用条款对用模型输出开发竞争模型有限制，使用前请自行判断，不需要的话按 `source` 列去掉；",
              "- 评测基准 C-Eval、CMMLU（非商用许可）只在本地评测时下载，不在这个数据集里。"]
    if unclear:
        lines += ["", f"许可证待确认的来源：{'、'.join(unclear)}。上游没有写明许可，可能有版权风险，商用前请自行确认，或者按 `source` 列把它们去掉。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="把洗好的来源混成最终数据集")
    parser.add_argument("command", choices=["index", "plan", "write", "verify", "card"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiers", nargs="*", default=None, help="write、verify 时只处理这几档，比如 nano mini base")
    args = parser.parse_args(argv)
    if args.command == "index":
        build_index()
    elif args.command == "plan":
        make_plan(args.seed)
    elif args.command == "write":
        write_tiers(args.seed, only=args.tiers)
    elif args.command == "card":
        out = ROOT / "out" / "mix"
        report = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        plan_table = json.loads((MIX_DIR / "index" / "_plan.json").read_text(encoding="utf-8"))["tokens"]
        for name, r in report.items():  # 早先的报告没有 source_tokens；verify 已经核对过每个来源的 token 数和计划一模一样，直接用计划的
            t = list(TIERS).index(name)
            r.setdefault("source_tokens", {s: row[t] for s, row in plan_table.items() if row[t]})
        post = posttrain_card()
        (out / "README.md").write_text(dataset_card(report) + ("\n" + post if post else ""), encoding="utf-8")
        print("数据集说明写到了", out / "README.md")
    else:
        report = verify_tiers(only=args.tiers)
        for name, r in report.items():
            print(f"  {final_path(name).name}：{r['docs']} 篇，估算 {r['est_tokens'] / 1e9:.3f}B token，"
                  f"按正式分词器约 {r['real_tokens_estimate'] / 1e9:.3f}B，文件 {r['bytes'] / 1e9:.2f} GB")
        out = ROOT / "out" / "mix"
        out.mkdir(parents=True, exist_ok=True)
        (out / "verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{'、'.join(report)} 全部检查通过，报告在", out / "verify.json")


if __name__ == "__main__":
    main()
