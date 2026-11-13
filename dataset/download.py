"""按 dataset/sources.py 里每个来源的 download 下载原始数据。可以重复运行，下完的文件跳过；断了重跑会接着下。

    python -m dataset.download pretrain          # 全部预训练来源 → data/raw/<来源>/
    python -m dataset.download sft rl extra      # SFT 原料 → data/inputs/sft/；RL 原始数据 → data/inputs/rl/；其他
    python -m dataset.download wiki_zh shu       # 只下这几个来源
    python -m dataset.download pretrain --list   # 只列出会下哪些文件、多大，不下载

ModelScope 优先；ModelScope 上没有的走 hf-mirror。有 aria2c 时用它（多连接、断点续传），没有就用 curl。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from dataset.sources import EXTRA, INPUTS, PRETRAIN, RAW, RL_INPUTS, SFT

MS, HF = "https://www.modelscope.cn", "https://hf-mirror.com"
DATA_SUFFIXES = (".parquet", ".jsonl", ".json", ".gz", ".zst", ".txt", ".csv")
GROUPS = {"pretrain": PRETRAIN, "sft": SFT, "rl": RL_INPUTS, "extra": EXTRA}


def _json(url: str, tries: int = 5):
    """GET 一个 JSON。走海外镜像时常有断线，失败了隔几秒再试。"""
    for attempt in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=60))
        except (OSError, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def ms_tree(repo: str, root: str = "", recursive: bool = False) -> list[tuple[str, int]]:
    out, page = [], 1
    while True:
        u = f"{MS}/api/v1/datasets/{repo}/repo/tree?Revision=master&Root=/{quote(root)}&Recursive={recursive}&PageSize=500&PageNumber={page}"
        files = (_json(u).get("Data") or {}).get("Files") or []
        out += [(f["Path"].lstrip("/"), f.get("Size") or 0) for f in files if f.get("Type") != "tree"]
        if len(files) < 500:
            return out
        page += 1


def hf_tree(repo: str) -> list[tuple[str, int]]:
    return [(f["path"], (f.get("lfs") or {}).get("size", f.get("size", 0)))
            for f in _json(f"{HF}/api/datasets/{repo}/tree/main?recursive=true&limit=1000") if f["type"] == "file"]


def resolve(how: dict, tree_ms=ms_tree, tree_hf=hf_tree) -> list[tuple[str, str, int]]:
    """download 的写法 → [(下载地址, 仓库里的路径, 大小)]。大小不知道时是 0。"""
    if "urls" in how:
        return [(url, name, 0) for url, name in how["urls"]]
    if "ms" in how:
        repo, base = how["ms"], f"{MS}/datasets/{how['ms']}/resolve/master/"
        if "paths" in how:
            files = [(p, 0) for p in how["paths"]]
        elif "pattern" in how:
            files = [(p, s) for p, s in tree_ms(repo, "", True) if re.search(how["pattern"], p)]
        else:
            files = []
            for prefix, limit in (how.get("prefixes") or {how.get("prefix", ""): how.get("limit")}).items():
                listed = sorted((p, s) for p, s in tree_ms(repo, prefix, False) if p.lower().endswith(DATA_SUFFIXES))
                files += listed[:limit]
    else:
        repo, base = how["hf"], f"{HF}/datasets/{how['hf']}/resolve/main/"
        files = [(p, 0) for p in how["paths"]] if "paths" in how else [(p, s) for p, s in tree_hf(repo) if re.search(how["pattern"], p)]
    return [(base + quote(p), p, s) for p, s in sorted(files)]


def fetch(url: str, dest: Path, size: int = 0, max_bytes: int | None = None) -> None:
    """下载到 dest（先写 .part，完整了才改名）。max_bytes：只要开头这么多字节。"""
    if dest.exists() and (not size or dest.stat().st_size == size or max_bytes):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if max_bytes:
        cmd = ["curl", "-sfL", "--http1.1", "--retry", "5", "--retry-all-errors", "-r", f"0-{max_bytes - 1}", "-o", str(part), url]
    elif shutil.which("aria2c") and url.startswith("http"):
        conns = "16" if url.startswith(HF) else "4"
        cmd = ["aria2c", "-q", "-c", "-x", conns, "-s", conns, "-k1M", "--max-tries=0", "--retry-wait=5", "--auto-file-renaming=false",
               "-d", str(part.parent), "-o", part.name, url]
    else:
        cmd = ["curl", "-sfL", "--retry", "5", "-C", "-", "-o", str(part), url]
    subprocess.run(cmd, check=True)
    os.replace(part, dest)


def first_rows(path: Path, n: int) -> list[dict]:
    """parquet 或每行一个 JSON 的文件的前 n 行。"""
    if path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000):
            rows += batch.to_pylist()
            if len(rows) >= n:
                break
        return rows[:n]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # 只下了开头一段时，最后一行可能不完整
                    break
                if len(rows) >= n:
                    break
    return rows


def _stream_jsonl(url: str, dest: Path, n: int) -> None:
    """大的 jsonl 只要前 n 行：边下边数，够了就停。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part, got = dest.with_name(dest.name + ".part"), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=120) as r, open(part, "wb") as f:
        for line in r:
            f.write(line)
            got += bool(line.strip())
            if got >= n:
                break
    os.replace(part, dest)


def download_sft(name: str, spec: dict, out_dir: Path = INPUTS / "sft") -> int:
    """SFT 来源：取前 rows 条，转成统一的对话格式（dataset/sft.py 的 to_messages），写成 <名字>.jsonl。返回条数。"""
    from dataset.sft import to_messages

    out = out_dir / f"{name}.jsonl"
    if out.exists():
        return sum(1 for _ in open(out, encoding="utf-8"))
    how, rows = spec["download"], []
    for url, path, size in resolve(how):
        raw = out_dir / "_raw" / name / Path(path).name
        if path.endswith(".parquet"):
            fetch(url, raw, size)
        else:
            if not raw.exists():
                _stream_jsonl(url, raw, how["rows"] - len(rows))
        rows += first_rows(raw, how["rows"] - len(rows))
    convs = [c for c in map(to_messages, rows) if c and any(m["role"] == "assistant" for m in c)]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".part"), "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps({"messages": c}, ensure_ascii=False, default=str) + "\n")
    os.replace(out.with_suffix(".part"), out)
    shutil.rmtree(out_dir / "_raw" / name, ignore_errors=True)
    return len(convs)


def destination(group: str, name: str, spec: dict) -> Path:
    return {"pretrain": RAW / name, "rl": INPUTS / "rl" / name}.get(group) or spec.get("dest") or RAW / name


def download(group: str, name: str, list_only: bool = False) -> str:
    spec = GROUPS[group][name]
    how = spec.get("download")
    if not how:
        return f"{name}：不用单独下（和别的来源共用原始文件）"
    if group == "sft" and not list_only:
        return f"{name}：{download_sft(name, spec)} 条对话"
    files = resolve(how)
    if list_only:
        return f"{name}：{len(files)} 个文件，{sum(s for *_, s in files) / 1e9:.2f} GB" + ("（只要开头）" if how.get("bytes") else "")
    dest = destination(group, name, spec)
    for url, path, size in files:
        file_name = how.get("save_as") or (path.replace("/", "__") if group == "rl" else Path(path).name)
        fetch(url, dest / file_name, size, how.get("bytes"))
    return f"{name}：{len(files)} 个文件 → {dest}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载原始数据（来源见 dataset/sources.py）")
    parser.add_argument("what", nargs="+", help="组（pretrain、sft、rl、extra）或者来源的名字")
    parser.add_argument("--list", action="store_true", help="只列出会下什么，不下载")
    args = parser.parse_args(argv)
    jobs = []
    for w in args.what:
        if w in GROUPS:
            jobs += [(w, n) for n in GROUPS[w]]
        else:
            group = next((g for g, table in GROUPS.items() if w in table), None)
            if group is None:
                parser.error(f"没有 {w!r} 这个组或来源")
            jobs.append((group, w))
    failed = []
    for group, name in jobs:
        try:
            print(download(group, name, args.list), flush=True)
        except Exception as e:  # 一个来源出错不影响别的，最后一起报
            failed.append(name)
            print(f"{name}：失败，{type(e).__name__}：{str(e)[:200]}", flush=True)
    if failed:
        raise SystemExit(f"这些没成功，可以重跑：{'、'.join(failed)}")


if __name__ == "__main__":
    main()
