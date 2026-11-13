"""所有数据来源一张表：预训练（PRETRAIN）、SFT（SFT）、RL 的原始数据（RL_INPUTS）和其他要下载的东西（EXTRA）。

每个来源写清楚：出自哪个数据集、许可证、怎么下载（download），预训练来源还写怎么从原始的一行里取出正文。
下载见 dataset/download.py，清洗见 dataset/pretrain.py。

download 的写法（都会固定到具体的文件，重跑得到同样的数据）：
    ms=仓库, prefix=目录, limit=最多几个文件          ModelScope 上某个目录下的文件（不往下找子目录），按文件名排序取前 limit 个
    ms=仓库, prefixes={目录: 个数, …}                  同一个仓库的几个目录各取几个
    hf=仓库, pattern=正则                              Hugging Face（走 hf-mirror）上文件名匹配的全部文件
    hf=仓库, paths=[文件, …]                           Hugging Face 上指定的几个文件
    urls=[(地址, 存成的文件名), …]                     直接下载
    rows=N                                             （SFT）只要前 N 条对话，转成统一格式
    bytes=N                                            只下文件开头 N 字节（文件很大、只用开头一段时）
"""
import re
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq

from dataset.clean import split_book, strip_gutenberg, unwrap_lines

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW, CLEAN, INPUTS = DATA / "raw", DATA / "clean", DATA / "inputs"   # 原始下载、洗好的预训练数据、后训练的原料


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


# 预训练的来源：种类、类别、怎么从一行里取正文（一行可以给出多篇）；regroup 可以先把原始的行重新分组、怎么取自带的质量分（换算到 0 到 1 左右）、
# 是否去 Markdown、出自哪个数据集（upstream）、许可（照上游写的；「待确认」是上游没写清楚）、怎么下载（download，见 dataset/download.py）
PRETRAIN = {
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


_POETRY_COMMIT = "https://raw.githubusercontent.com/Werneror/Poetry/4cfe49c06858e00d15f84d192fe5294295f79689/"
_POETRY_FILES = ("元 元末明初 先秦 南北朝 唐 唐末宋初 宋_1 宋_2 宋_3 宋_4 宋末元初 宋末金初 当代 明_1 明_2 明_3 明_4 明末清初 民国末当代初 汉 清_1 清_2 "
                 "清末民国初 清末近现代初 秦 辽 近现代 近现代末当代初 金 金末元初 隋 隋末唐初 魏晋 魏晋末南北朝初").split()
_ANY = r"\.(parquet|jsonl|json)$"

# 预训练来源怎么下载（下到 data/raw/<来源>/）。gutenberg_en_zh 和 gutenberg_en 用同一份原始文件。
PRETRAIN_DOWNLOAD = {
    "fw22_4_5": dict(ms="opencsg/Fineweb-Edu-Chinese-V2.2", prefix="4_5"),
    "fw22_3_4": dict(ms="opencsg/Fineweb-Edu-Chinese-V2.2", prefix="3_4", limit=150),
    "ultrafineweb_en": dict(ms="openbmb/Ultra-FineWeb", prefix="data/ultrafineweb_en", limit=10),
    "wiki_zh": dict(ms="wikimedia/wikipedia", prefix="20231101.zh"),
    "wiki_en": dict(ms="wikimedia/wikipedia", prefix="20231101.en"),
    "cosmopedia_zh": dict(ms="opencsg/chinese-cosmopedia", prefix="data", limit=3),
    "law": dict(hf="twang2218/chinese-law-and-regulations", pattern=r"\.parquet$"),
    "gujilab": dict(hf="gujilab/chinese-classical-corpus", paths=["corpus.jsonl"]),
    "wikisource_zh": dict(ms="wikimedia/wikisource", prefix="20231201.zh"),
    "greathangpt": dict(hf="kenpusney/greathangpt-classical-chinese", pattern=r"all\.parquet$"),
    "shu": dict(hf="shjwudp/shu", paths=["clean_text.jsonl.gz"]),
    "baidu_baike": dict(hf="lars1234/baidu-baike-dataset", pattern=_ANY),
    "poetry": dict(urls=[(_POETRY_COMMIT + quote(f"{n}.csv"), f"{n}.csv") for n in _POETRY_FILES]),
    "luxun": dict(ms="SkyContact/TheCompleteWorksOfLuXun", prefix="data"),
    "gutenberg_en": dict(ms="common-pile/project_gutenberg", prefix="v0/documents"),
    "gutenberg_zh": dict(hf="manu/project_gutenberg", paths=["data/zh-00000-of-00001-bca42330c6f1826c.parquet"]),
    "numinamath": dict(ms="AI-MO/NuminaMath-1.5", prefix="data"),
    "nemotron_math": dict(ms="nv-community/Nemotron-Pretraining-Specialized-v1", prefix="Nemotron-Pretraining-Math-Textbooks", limit=2),
    "finemath": dict(ms="HuggingFaceTB/finemath", prefix="finemath-4plus", limit=8),
    "ifm_math_rewrite": dict(ms="IFM/Math-Reasoning", prefix="math-rewrite", limit=2),
    "udmath_l2": dict(ms="openbmb/UltraData-Math", prefix="data/UltraData-Math-L2-preview", limit=6),
    "arxiv": dict(ms="common-pile/arxiv_papers", prefix="", limit=6),
    "pes2o": dict(ms="allenai/peS2o", prefix="data/v3", limit=2),
    "udcode_l2": dict(ms="openbmb/UltraData-Code", prefixes={f"data/UltraData-Code-L2/{k}": n for k, n in
                                                            (("py", 4), ("cpp", 2), ("java", 1), ("js", 1), ("go", 1), ("rust", 1))}),
    "udstack_v3_l2": dict(ms="openbmb/UltraData-Stack-v3-L2", prefixes={f"data/UltraData-Stack-v3-L2/{k}": n for k, n in (("python", 2), ("cpp", 1), ("java", 1))}),
}
assert set(PRETRAIN_DOWNLOAD) | {"gutenberg_en_zh"} == set(PRETRAIN), "每个预训练来源都要写怎么下载"
for _name, _how in PRETRAIN_DOWNLOAD.items():
    PRETRAIN[_name]["download"] = _how

# SFT 的来源：领域、许可、放进哪几档、出自哪里、怎么下载（固定到上游的具体文件，取前 10 万条，转成统一的对话格式，存到 data/inputs/sft/<名字>.jsonl）
_ALL = ("small", "medium", "large")
SFT = {
    "smoltalk_zh": dict(domain="通用", license="apache-2.0", tiers=_ALL, upstream="opencsg/smoltalk-chinese",
                        download=dict(ms="opencsg/smoltalk-chinese", paths=["advice-seeking.parquet"])),
    "chinese_instruct": dict(domain="通用", license="cc-by-sa-4.0", tiers=_ALL, upstream="Mxode/Chinese-Instruct",
                             download=dict(hf="Mxode/Chinese-Instruct", paths=["Chinese-QA-AFAF/Chinese-QA-AFAF.jsonl"])),
    "coig_cqia": dict(domain="通用", license="见原项目", tiers=_ALL, upstream="m-a-p/COIG-CQIA",
                      download=dict(ms="m-a-p/COIG-CQIA", paths=["COIG-CQIA-full.jsonl"])),
    "deepctrl": dict(domain="通用", license="见原项目", tiers=_ALL, upstream="deepctrl/deepctrl-sft-data（中文部分）",
                     download=dict(ms="deepctrl/deepctrl-sft-data", paths=["sft_data_zh.jsonl"])),
    "ying_k1": dict(domain="通用", license="cc-by-4.0", tiers=_ALL, upstream="Karasumikura/Ying-K1-SFT-Chinese-NonThink-102K",
                    download=dict(hf="Karasumikura/Ying-K1-SFT-Chinese-NonThink-102K", paths=["data/train.jsonl"])),
    "infinity": dict(domain="通用", license="cc-by-sa-4.0", tiers=_ALL, upstream="BAAI/Infinity-Instruct（0625）",
                     download=dict(ms="BAAI/Infinity-Instruct", paths=["0625/train-00000-of-00007.parquet"])),
    "smol_smoltalk": dict(domain="通用", license="apache-2.0", tiers=_ALL, upstream="HuggingFaceTB/smol-smoltalk",
                          download=dict(ms="HuggingFaceTB/smol-smoltalk", paths=["data/test-00000-of-00001.parquet"])),
    "dolci_instruct": dict(domain="通用", license="odc-by", tiers=_ALL, upstream="allenai/Dolci-Instruct-SFT",
                           download=dict(ms="allenai/Dolci-Instruct-SFT", paths=["data/train-00000-of-00015.parquet"])),
    "toolace": dict(domain="工具调用", license="apache-2.0", tiers=_ALL, upstream="Team-ACE/ToolACE",
                    download=dict(ms="mlabonne/ToolACE", paths=["data/train-00000-of-00001.parquet"])),
    "ud_agent_tool": dict(domain="工具调用", license="apache-2.0", tiers=("medium", "large"), upstream="openbmb/UltraData-SFT-Agent-2609（工具部分）",
                          download=dict(ms="openbmb/UltraData-SFT-Agent-2609", paths=["data/Tool_Use/Tool_Use_part-1-of-9.jsonl"])),
    "opc_sft": dict(domain="代码", license="mit", tiers=_ALL, upstream="OpenCoder-LLM/opc-sft-stage2",
                    download=dict(ms="AI-ModelScope/opc-sft-stage2", paths=["educational_instruct/train-00000-of-00001.parquet"])),
    "numina_cot": dict(domain="数学", license="apache-2.0", tiers=_ALL, upstream="AI-MO/NuminaMath-CoT",
                       download=dict(ms="AI-MO/NuminaMath-CoT", paths=["data/train-00000-of-00005.parquet"])),
    "r1_zh_110k": dict(domain="通用", license="apache-2.0", tiers=_ALL, upstream="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
                       download=dict(ms="AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k", paths=["distill_r1_110k.jsonl"])),
    "openscience_zh": dict(domain="通用", license="cc-by-4.0", tiers=("medium", "large"), upstream="zake7749/OpenScience-Chinese-Reasoning-SFT",
                           download=dict(ms="zake7749/OpenScience-Chinese-Reasoning-SFT", paths=["data/train-00000-of-00001.parquet"])),
    "am_qwen3": dict(domain="通用", license="见原项目", tiers=("large",), upstream="a-m-team/AM-Qwen3-Distilled（code 部分）",
                     download=dict(ms="a-m-team/AM-Qwen3-Distilled", paths=["code.jsonl"])),
    "nemotron_if": dict(domain="通用", license="cc-by-4.0 等", tiers=("medium", "large"), upstream="nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
                        download=dict(ms="nv-community/Nemotron-SFT-Instruction-Following-Chat-v3", paths=["data/chat.jsonl"])),
    "nemotron_math_sft": dict(domain="数学", license="cc-by-4.0", tiers=("large",), upstream="nvidia/Nemotron-SFT-Math-v4",
                              download=dict(ms="nv-community/Nemotron-SFT-Math-v4", paths=["data/train-00000-of-00012.parquet"])),
    "nemotron_cp": dict(domain="代码", license="cc-by-4.0 等", tiers=("large",), upstream="nvidia/Nemotron-SFT-Competitive-Programming-v2",
                        download=dict(ms="nv-community/Nemotron-SFT-Competitive-Programming-v2", paths=["data/competitive_programming_cpp_00.jsonl"])),
}
for _spec in SFT.values():
    _spec["download"]["rows"] = 100_000

# RL 题目的原始数据（下到 data/inputs/rl/<名字>/，文件名里的「/」换成「__」）；怎么出题见 dataset/rl.py
RL_INPUTS = {
    "deepscaler": dict(license="mit", download=dict(ms="agentica-org/DeepScaleR-Preview-Dataset", pattern=r"deepscaler\.json$")),
    "bigmath": dict(license="apache-2.0", download=dict(ms="SynthLabsAI/Big-Math-RL-Verified", pattern=r"\.parquet$")),
    "deepcoder": dict(license="mit", download=dict(ms="agentica-org/DeepCoder-Preview-Dataset", pattern=r"(/train-.*|^codeforces/.*)\.parquet$")),
    "ifeval": dict(license="odc-by", download=dict(ms="allenai/RLVR-IFeval", pattern=r"\.parquet$")),
    "cmrc2018": dict(license="cc-by-sa-4.0", download=dict(ms="hfl/cmrc2018", pattern=r"\.parquet$")),
    "arc": dict(license="cc-by-sa-4.0", download=dict(ms="allenai/ai2_arc", pattern=r"\.parquet$")),
    "gsm8k_zh": dict(license="mit", download=dict(hf="meta-math/GSM8K_zh", paths=["GSM8K_zh.json"])),
    "openscience": dict(license="cc-by-4.0", download=dict(hf="nvidia/OpenScience", paths=["OS-Q2.5-32B-4.jsonl"], bytes=50 << 20)),
}

# 其他：PG-19 的验证集和测试集（只为拿到要从古腾堡里排除的书号）；古籍库的文白互译（SFT 文学数据只用开头几千条）
EXTRA = {
    "pg19_heldout": dict(dest=RAW / "pg19_heldout", download=dict(hf="emozilla/pg19", paths=["data/test-00000-of-00001-29a571947c0b5ccc.parquet",
                                                                                         "data/validation-00000-of-00001-0f92e2337f79aeac.parquet"])),
    "gujilab_translate": dict(dest=INPUTS, download=dict(hf="gujilab/chinese-classical-corpus", paths=["translate.jsonl"], bytes=40 << 20,
                                                         save_as="gujilab_translate.jsonl")),
}
