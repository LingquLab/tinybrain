"""精洗流水线：把 data/raw/<来源>/ 里的原始文件洗成 data/clean/<来源>/ 里的 parquet。

    python -m data.build.pipeline clean fw22_4_5 --workers 10 [--limit 100]
    python -m data.build.pipeline audit fw22_4_5 [--docs 30]

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

from data.build.clean import clean_with_reason, unwrap_lines
from data.build.books import split_book, strip_gutenberg

ROOT = Path(__file__).resolve().parents[2]
RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"
DROPPED_PER_FILE = 15  # 每个原始文件留几篇被丢掉的文档，供人工检查


def _text(row):
    return row.get("text") or ""


def _field(name):
    return lambda row: row.get(name) or ""


def _titled(title_key, body_key):
    """标题加正文；正文本来就以标题开头的不重复加。"""
    def build(row):
        title, body = (row.get(title_key) or "").strip(), (row.get(body_key) or "").strip()
        return body if not title or body.startswith(title) else f"{title}\n{body}"
    return build


_WIKI_TAIL = re.compile(
    r"^[ \t]*(?:[参參]考(?:文[献獻]|[资資]料|[来來]源|[书書]目)?|[参參][见見看]|外部[链連鏈][接結结]|[注註][释釋脚腳解]|[脚腳][注註]|延伸[阅閱][读讀]"
    r"|相[关關][条條]目|[资資]料[来來]源|References|External links|See also|Further reading|Notes|Footnotes|Bibliography|Sources|Citations)[ \t]*$", re.M)
_WIKI_SKIP = re.compile(r"列表|索引|年表|消歧义|消歧義|^List of |^Lists of |\(disambiguation\)|^Index of |^Timeline of ")
_WIKI_PROSE = re.compile(r"^.{200,}[。.]\s*$", re.M)
_WIKI_MARKUP = re.compile(r"</?(?:noinclude|includeonly|onlyinclude|small|big|center|br|nowiki|ref|div|span|font)\b[^>\n]{0,80}>|^[ \t]*[!|][^\n]*\n?", re.M)
_WIKI_DANGLING = re.compile(r"(?<=[（(])[，,；;、:：\s]+|[，,；;、\s]+(?=[）)])")
_WIKI_HOLLOW = re.compile(r"INSEE市[镇鎮][编編][码碼][为為]。|人口[数數]量[为為]人。")  # 机器人批量建的条目，模板里的数字全是空的
_WIKI_EMPTY_LABEL = re.compile(r"[（(](?:[^（）()\n:：]{0,10}[:：])+[）)]| ?\((?:or|and|also|;|,|\s)*\)|（）| +(?=[,.;:](?:\s|$))")
_WIKI_LOOSE = re.compile(r"[。，；：！？.!?,;:]")


def _wiki(row):
    """维基条目：跳过列表、索引、消歧义页；去掉表格和模板的残留；
    从「参考文献」「外部链接」这类栏目起截掉（后面还有成段正文的不算）；再去掉末尾的分类名和空标题。"""
    title, text = (row.get("title") or "").strip(), row.get("text") or ""
    if _WIKI_SKIP.search(title) or _WIKI_HOLLOW.search(text):
        return ""
    text = _WIKI_EMPTY_LABEL.sub("", _WIKI_DANGLING.sub("", _WIKI_MARKUP.sub("", text)))
    for m in _WIKI_TAIL.finditer(text):
        if not _WIKI_PROSE.search(text, m.end()):
            text = text[:m.start()]
            break
    blocks = [b for b in re.split(r"\n[ \t]*\n", text.strip()) if b.strip()]
    while len(blocks) > 1 and all(len(l.strip()) < 30 and not _WIKI_LOOSE.search(l) for l in blocks[-1].split("\n")):
        blocks.pop()  # 结尾只剩短短几行、没有标点的，是分类名或者空标题
    return f"{title}\n" + "\n\n".join(blocks) if blocks else ""


_INTERWIKI = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,12})?:[^\n]*\n?", re.M)


def _wikisource(row):
    """维基文库：在维基条目的处理之外，再去掉指向其他语言版本的链接行。"""
    return _wiki({**row, "text": _INTERWIKI.sub("", row.get("text") or "")})


def _law(row):
    """法律法规：正文是 Markdown，把转义的书名号尖括号和行尾的反斜杠还原。"""
    return (row.get("content") or "").replace("\\<", "〈").replace("\\>", "〉").replace("\\\n", "\n")


_PAPER_REFS = re.compile(r"^#{1,6} *(?:References?|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature cited)\b[^\n]*$", re.M)
_MD_KEEP_BREAK = re.compile(r"\s*(?:[*+\-] |\d+[.)] |#|>|\||\$\$|\\\[|```)")
PAPER_CHARS = 150_000


def _unwrap_markdown(text: str) -> str:
    """Markdown 里段内的换行只是折行：接回一整段。行尾两个空格的硬换行、列表、标题、公式和表格不动。"""
    blocks = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.split("\n")
        if "$$" in block or "```" in block or "\\[" in block:
            blocks.append(block)
            continue
        out = [lines[0]]
        for line in lines[1:]:
            if out[-1].endswith("  ") or _MD_KEEP_BREAK.match(line) or _MD_KEEP_BREAK.match(out[-1]) and out[-1].lstrip().startswith(("#", "|")):
                out.append(line)
            else:
                out[-1] = out[-1].rstrip() + " " + line.lstrip()
        blocks.append("\n".join(l.rstrip() for l in out))
    return "\n\n".join(blocks)


def _paper(row):
    """论文全文：去掉文末的参考文献表，把折行接回整段，特别长的切成几篇。"""
    text = row.get("text") or ""
    refs = None
    for refs in _PAPER_REFS.finditer(text, int(len(text) * 0.4)):
        pass  # 取后六成里最后一个「References」标题
    if refs:
        text = text[:refs.start()]
    text = _unwrap_markdown(text).strip()
    return [text] if len(text) <= PAPER_CHARS else split_book(text, PAPER_CHARS)


def _guji(row):
    """古籍的一章：标题取章、题、节里第一个有的，前面带上书名；正文本来就以标题开头的不重复加。"""
    head = next((row[k].strip() for k in ("chapter", "title", "section", "subchapter") if (row.get(k) or "").strip()), "")
    book, body = (row.get("source") or "").strip(), (row.get("content") or "").strip()
    if book and book not in head:
        head = f"{book}·{head}" if head else book
    return body if not head or body.startswith(head) else f"{head}\n{body}"


def _merge_shuowen(rows):
    """《说文解字》一个字一行，单看每行只有十几个字：把同一个部首的字并成一篇，每行写成「字：解说」。别的书原样放过。"""
    lines, radical = [], None
    for row in rows:
        if row.get("source") == "说文解字" and row.get("char"):
            if lines and row.get("radical") != radical:
                yield {"chapter": f"说文解字·{radical}", "content": "\n".join(lines)}
                lines = []
            radical = row.get("radical")
            lines.append(f"{row['char']}：{(row.get('content') or '').strip()}")
        else:
            if lines:
                yield {"chapter": f"说文解字·{radical}", "content": "\n".join(lines)}
                lines = []
            yield row
    if lines:
        yield {"chapter": f"说文解字·{radical}", "content": "\n".join(lines)}


def _numina(row):
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return ""
    return f"{row.get('problem') or ''}\n\n{row.get('solution') or ''}"


def _num(name, scale=1.0):
    def get(row):
        try:
            return float(row[name]) / scale
        except (KeyError, TypeError, ValueError):
            return None
    return get


BOOK_CHARS = {"en": 32_000, "classical": 12_000, "zh": 12_000}  # 一本书切成的每篇大约 8000 token


_heldout_books = None


def pg19_heldout() -> frozenset:
    """PG-19 留作验证和测试的古腾堡书号。这些书不进训练数据，以后才能拿它们做评测。"""
    global _heldout_books
    if _heldout_books is None:
        urls = [u for p in sorted((RAW / "pg19_heldout").glob("*.parquet")) for u in pq.read_table(p, columns=["url"]).column("url").to_pylist()]
        _heldout_books = frozenset(m.group(1) for m in (re.search(r"(\d+)\D*$", u or "") for u in urls) if m)
    return _heldout_books


def _gutenberg(lang, kind, unwrap=False):
    """古腾堡的一本书：只要指定语言的（lang 为 None 时不筛，整个文件就是一种语言），跳过 PG-19 留作评测的书，去掉书前书后的说明，切成若干篇。
    unwrap：把按固定宽度折行的中文接回整段。"""
    def build(row):
        language = (row.get("metadata") or {}).get("language") or ""
        if (lang is not None and language != lang) or str(row.get("id")) in pg19_heldout():
            return []
        docs = split_book(strip_gutenberg(row.get("text") or ""), BOOK_CHARS[kind])
        return [unwrap_lines(d) for d in docs] if unwrap else docs
    return build


def _whole_book(kind):
    return lambda row: split_book(row.get("text") or "", BOOK_CHARS[kind])


def _shu(row):
    """shu 里的一本古籍：修掉「·」被存坏成的「?;」，切成若干篇，第一篇前面带上书名。"""
    text = (row.get("text") or "").replace("?;", "·")
    name = (row.get("name") or "").strip()
    docs = split_book(text, BOOK_CHARS["classical"])
    if docs and name and not docs[0].startswith(name):
        docs[0] = f"{name}\n{docs[0]}"
    return docs


def _baike(row):
    """百度百科的一个词条：标题、摘要，再接各个小节（小节标题单独一行）。"""
    parts = [(row.get("title") or "").strip(), (row.get("summary") or "").strip()]
    for sec in row.get("sections") or []:
        if isinstance(sec, dict) and (sec.get("content") or "").strip():
            parts += [(sec.get("title") or "").strip(), sec["content"].strip()]
    return "\n".join(p for p in parts if p)


def _poem(row):
    title, author, era, body = ((row.get(k) or "").strip() for k in ("题目", "作者", "朝代", "内容"))
    return f"{title}\n{era}·{author}\n{body}" if body else ""


# 每个来源：种类、类别、怎么从一行里取正文（一行可以给出多篇）；regroup 可以先把原始的行重新分组、怎么取自带的质量分（换算到 0 到 1 左右）、是否去 Markdown、
# 出自哪个数据集（upstream；下载脚本是 dl_queue.sh、dl_hf.sh、dl_books.sh）、许可（照上游写的；「待确认」是上游没写清楚）
SOURCES = {
    "fw22_4_5": dict(kind="zh", category="文章", text=_text, score=_num("score"), upstream="opencsg/Fineweb-Edu-Chinese-V2.2（4_5 分档）", license="apache-2.0"),
    "fw22_3_4": dict(kind="zh", category="文章", text=_text, score=_num("score"), upstream="opencsg/Fineweb-Edu-Chinese-V2.2（3_4 分档，前 150 个文件）", license="apache-2.0"),
    "ultrafineweb_en": dict(kind="en", category="英文网页", text=_field("content"), score=_num("score"), upstream="openbmb/Ultra-FineWeb（英文，前 10 个文件）", license="apache-2.0"),
    "wiki_zh": dict(kind="zh", category="百科", text=_wiki, upstream="wikimedia/wikipedia（20231101.zh）", license="cc-by-sa-3.0"),
    "wiki_en": dict(kind="en", category="百科", text=_wiki, upstream="wikimedia/wikipedia（20231101.en）", license="cc-by-sa-3.0"),
    "cosmopedia_zh": dict(kind="zh", category="合成", text=_text, score=_num("score"), markdown=True, upstream="opencsg/chinese-cosmopedia（前 3 个文件）", license="apache-2.0"),
    "law": dict(kind="zh", category="法律", text=_law, markdown=True, files="*8329bce6db03c820*", upstream="twang2218/chinese-law-and-regulations", license="apache-2.0"),
    "gujilab": dict(kind="classical", category="古文", text=_guji, regroup=_merge_shuowen, files="corpus.jsonl", upstream="gujilab/chinese-classical-corpus", license="cc0-1.0"),
    "wikisource_zh": dict(kind="classical", category="古文", text=_wikisource, upstream="wikimedia/wikisource（20231201.zh）", license="cc-by-sa-3.0"),
    "greathangpt": dict(kind="classical", category="古文", text=_titled("title", "text"), upstream="kenpusney/greathangpt-classical-chinese", license="apache-2.0"),
    "shu": dict(kind="classical", category="古文", text=_shu, upstream="shjwudp/shu", license="待确认"),
    "baidu_baike": dict(kind="zh", category="百科", text=_baike, upstream="lars1234/baidu-baike-dataset", license="待确认"),
    "poetry": dict(kind="classical", category="古文", text=_poem, skip="当代|近现代", upstream="github.com/Werneror/Poetry", license="mit"),
    "luxun": dict(kind="zh", category="中文书", text=_whole_book("zh"), files="luxun_*.txt", upstream="SkyContact/TheCompleteWorksOfLuXun", license="public-domain"),
    "gutenberg_en": dict(kind="en", category="英文书", text=_gutenberg("en", "en"), upstream="common-pile/project_gutenberg（英文书；排除 PG-19 的验证集和测试集）", license="public-domain"),
    "gutenberg_zh": dict(kind="classical", category="古文", text=_gutenberg(None, "classical", unwrap=True), upstream="manu/project_gutenberg（zh）", license="public-domain"),  # manu/project_gutenberg 的中文部分
    "gutenberg_en_zh": dict(raw="gutenberg_en", kind="classical", category="古文", text=_gutenberg("zh", "classical", unwrap=True), upstream="common-pile/project_gutenberg（里面的中文书）", license="public-domain"),
    "numinamath": dict(kind="math", category="数学", text=_numina, upstream="AI-MO/NuminaMath-1.5", license="apache-2.0"),
    "nemotron_math": dict(kind="math", category="数学", text=_text, upstream="nv-community/Nemotron-Pretraining-Specialized-v1（Math-Textbooks，前 2 个文件）", license="cc-by-4.0"),
    "finemath": dict(kind="math", category="数学", text=_text, score=_num("score", 5.0), upstream="HuggingFaceTB/finemath（finemath-4plus，前 8 个文件）", license="odc-by"),
    "ifm_math_rewrite": dict(kind="math", category="数学", text=_text, upstream="IFM/Math-Reasoning（math-rewrite，前 2 个文件）", license="apache-2.0"),
    "udmath_l2": dict(kind="math", category="数学", text=_field("content"), score=_num("quality_label", 5.0), upstream="openbmb/UltraData-Math（L2-preview，前 6 个文件）", license="apache-2.0"),
    "arxiv": dict(kind="en", category="论文", text=_paper, upstream="common-pile/arxiv_papers（前 6 个文件）", license="per-paper"),
    "pes2o": dict(kind="en", category="论文", text=_text, upstream="allenai/peS2o（v3，前 2 个文件）", license="odc-by"),
    "udcode_l2": dict(kind="code", category="代码", text=_field("content"), score=_num("quality_score", 10.0), upstream="openbmb/UltraData-Code（L2：Python 4、C++ 2、Java/JS/Go/Rust 各 1 个文件）", license="apache-2.0"),
    "udstack_v3_l2": dict(kind="code", category="代码", text=_field("content"), score=_num("quality_score", 10.0), upstream="openbmb/UltraData-Stack-v3-L2（Python 2、C++ 1、Java 1 个文件）", license="odc-by"),
}

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
    pattern, skip = SOURCES[name].get("files", "*"), SOURCES[name].get("skip")
    return sorted(p for p in (RAW / SOURCES[name].get("raw", name)).iterdir()
                  if p.name.lower().endswith(RAW_SUFFIXES) and not p.name.startswith("_")
                  and fnmatch.fnmatch(p.name, pattern) and not (skip and re.search(skip, p.name))
                  and not p.with_name(p.name + ".aria2").exists())


WRITE_EVERY = 50_000  # 攒够这么多篇就写出一组，大文件（比如十几 GB 的百科）也不会全压在内存里


def process_file(job) -> dict:
    """洗一个原始文件，写出一个 parquet 和一份被丢文档的小样。返回统计。"""
    name, in_path, out_path = job
    spec = SOURCES[name]
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
    parser.add_argument("source", choices=list(SOURCES))
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
