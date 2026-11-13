"""后训练数据：三份 SFT 语料、三个专家（数学、代码、文学）的 SFT 数据、MOPD 的提示、RL 的题目。

    python -m data.build.posttrain sft       # data/sft_{small,medium,large}.parquet，data/sft_expert_{math,code,literature}.parquet
    python -m data.build.posttrain prompts   # data/mopd_prompts.parquet，data/rl_{small,medium,large}.parquet

输入：
  - 外部 SFT 数据：每个来源一个统一成对话格式的 jsonl（{"messages": [...]}），默认读 data/survey/sft_full/<名字>.jsonl
    （每个来源最多 10 万条）；只想快速试一下时用 --sft-dir data/survey/sft_samples（每个来源约 1000 条的小样）。
  - 我们自己的数据：古籍库的文白互译（data/survey/gujilab_translate.jsonl，gujilab/chinese-classical-corpus 的 translate.jsonl
    开头一段，只用前几千条）、洗好的诗词（data/clean/poetry）、洗好的鲁迅全集（data/clean/luxun）、洗好的中文维基（agent 检索题的资料）。
    不读 data/raw：原始下载在数据集做完以后会删掉。

处理（所有来源一样）：
  - 全部转成简体；
  - 去掉自称是别家模型的回答（「我是 ChatGPT」「由 OpenAI 开发」这类），去掉空回答；
  - 推理过程：small 版去掉，只留回答；medium、large 版写成 <think>…</think> 放在回答前面；
  - 按正式分词器编出来的真实长度，按档截断：small 2048、medium 4096、large 8192；
  - 同一份语料里按第一个问题去重。
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.build.clean import to_simplified
from data.build.pipeline import ROOT

DATA = ROOT / "data"
SFT_DIR = DATA / "survey" / "sft_full"
CAPS = {"small": 2048, "medium": 4096, "large": 8192}

# 外部来源：名字 → (领域, 许可, 放进哪几档, 每档最多取多少条)
SOURCES = {
    "smoltalk_zh": ("通用", "apache-2.0", ("small", "medium", "large"), None),
    "chinese_instruct": ("通用", "cc-by-sa-4.0", ("small", "medium", "large"), None),
    "coig_cqia": ("通用", "见原项目", ("small", "medium", "large"), None),
    "deepctrl": ("通用", "见原项目", ("small", "medium", "large"), None),
    "ying_k1": ("通用", "cc-by-4.0", ("small", "medium", "large"), None),
    "infinity": ("通用", "cc-by-sa-4.0", ("small", "medium", "large"), None),
    "smol_smoltalk": ("通用", "apache-2.0", ("small", "medium", "large"), None),
    "dolci_instruct": ("通用", "odc-by", ("small", "medium", "large"), None),
    "toolace": ("工具调用", "apache-2.0", ("small", "medium", "large"), None),
    "ud_agent_tool": ("工具调用", "apache-2.0", ("medium", "large"), None),
    "opc_sft": ("代码", "mit", ("small", "medium", "large"), None),
    "numina_cot": ("数学", "apache-2.0", ("small", "medium", "large"), None),
    "r1_zh_110k": ("通用", "apache-2.0", ("small", "medium", "large"), None),
    "openscience_zh": ("通用", "cc-by-4.0", ("medium", "large"), None),
    "am_qwen3": ("通用", "见原项目", ("large",), None),
    "nemotron_if": ("通用", "cc-by-4.0 等", ("medium", "large"), None),
    "nemotron_math_sft": ("数学", "cc-by-4.0", ("large",), None),
    "nemotron_cp": ("代码", "cc-by-4.0 等", ("large",), None),
}
EXPERTS = {"math": ("数学",), "code": ("代码",), "literature": ("文学",)}

_OTHER_MODEL = re.compile(r"(?:我是|我叫|作为|身为)[^。\n]{0,8}(?:ChatGPT|GPT-?\d|Claude|通义千问|千问|Qwen|文心一言|Kimi|豆包|DeepSeek|深度求索|Gemini|Llama)"
                          r"|由\s*(?:OpenAI|Anthropic|阿里巴巴|阿里云|百度|月之暗面|字节跳动|深度求索|Google|Meta)\s*(?:公司)?(?:开发|训练|创建|打造)", re.I)
_TOOL_CALL_TEXT = re.compile(r"^\[(\w[\w\- ]*)\((.*)\)\]$", re.S)


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["messages"]


def toolace_calls(content: str) -> list | None:
    """ToolACE 把调用写成「[函数名(参数=值, …), 另一个函数(…)]」的文字，转成结构化的调用；转不了返回 None。
    函数名里可能有空格和连字符（「Daily Match List-All」），所以按括号配对自己拆，不直接当 Python 解析。"""
    import ast

    s = content.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner, calls, i = s[1:-1], [], 0
    while i < len(inner):
        while i < len(inner) and inner[i] in " ,\n":
            i += 1
        if i >= len(inner):
            break
        j = inner.find("(", i)
        name = inner[i:j].strip() if j > 0 else ""
        if not name or len(name) > 80:
            return None
        depth, k, quote = 0, j, None
        while k < len(inner):
            ch = inner[k]
            if quote:
                if ch == "\\":
                    k += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth != 0:
            return None
        try:
            call = ast.parse(f"f({inner[j + 1:k]})", mode="eval").body
            if call.args:
                return None
            args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        except (SyntaxError, ValueError, TypeError):
            return None
        calls.append({"name": name, "arguments": args})
        i = k + 1
    return calls or None


def normalize(messages: list[dict], keep_reasoning: bool) -> list[dict] | None:
    """统一成训练用的样子；不合格返回 None。"""
    out = []
    for m in messages:
        role, content = m["role"], to_simplified(m.get("content") or "").strip()
        calls = m.get("tool_calls")
        if isinstance(calls, str):
            calls = json.loads(calls) if calls.strip() else None
        if calls:  # OpenAI 的格式 {"type": "function", "function": {"name", "arguments"}} 也认
            fixed = []
            for c in calls:
                c = c.get("function", c)
                args = c.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        return None
                fixed.append({"name": c.get("name"), "arguments": args})
            calls = fixed
        elif role == "assistant":
            calls = toolace_calls(content)
            if calls:
                content = ""
        if role == "assistant":
            if not content and not calls:
                return None
            if _OTHER_MODEL.search(content):
                return None
            reasoning = to_simplified(m.get("reasoning") or "").strip()
            if reasoning and keep_reasoning:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            elif not keep_reasoning:
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S).strip()
                if not content and not calls:
                    return None
        if role == "tool" and sum(ch.isprintable() or ch in "\n\t" for ch in content) < 0.95 * max(1, len(content)):
            return None  # 工具返回里是二进制乱码
        out.append({"role": role, "content": content, "tool_calls": json.dumps(calls, ensure_ascii=False) if calls else None})
    return out if any(m["role"] == "assistant" for m in out) else None


# ---------- 我们自己的数据：文学、自我认知

def literature(limit: int, seed: int = 0) -> list[list[dict]]:
    """古文和白话互译、按题写诗、鲁迅续写。"""
    rnd = random.Random(seed)
    convs = []
    path = DATA / "survey" / "gujilab_translate.jsonl"
    if path.exists():
        rows = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit * 4:
                    break
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # 只下了文件开头一段，最后一行可能不完整
                    break
        for r in rnd.sample(rows, min(limit, len(rows))):
            convs.append([{"role": "user", "content": f"{r['instruction']}\n{r['input']}"}, {"role": "assistant", "content": r["output"]}])
    for p in sorted((DATA / "clean" / "poetry").glob("*.parquet")):  # 每个文件是一个朝代；清洗时已经去掉了当代、近现代
        poems = [poem for poem in map(parse_poem, pq.read_table(p, columns=["text"]).column("text").to_pylist()) if poem]
        for title, era, author, body in rnd.sample(poems, min(limit // 30, len(poems))):
            form = poem_form(body)
            ask = f"以《{title}》为题，写一首{form or '诗'}" + (f"，学{era}代{author}的风格。" if author else "。")
            convs.append([{"role": "user", "content": ask}, {"role": "assistant", "content": body}])
    for prefix, rest in luxun_pairs(limit // 4, seed):
        convs.append([{"role": "user", "content": f"用鲁迅的笔法续写下面这段文字：\n{prefix}"}, {"role": "assistant", "content": rest}])
    return convs


def parse_poem(text: str) -> tuple[str, str, str, str] | None:
    """洗好的诗词是「题目\n朝代·作者\n正文」（data/build/pipeline.py 的 _poem）。"""
    lines = text.split("\n")
    if len(lines) < 3 or not lines[0].strip():
        return None
    era, _, author = lines[1].partition("·")
    body = "\n".join(lines[2:]).strip()
    return (lines[0].strip(), era.strip(), author.strip(), body) if body else None


def poem_form(text: str) -> str | None:
    lines = [s for s in re.split(r"[，。！？；]", text) if s.strip()]
    for name, (n, length) in {"五言绝句": (4, 5), "七言绝句": (4, 7), "五言律诗": (8, 5), "七言律诗": (8, 7)}.items():
        if len(lines) == n and all(len(s) == length for s in lines):
            return name
    return None


_LETTER_START = re.compile(r"^[^，。！？：:\s]{1,12}[：:]")  # 「汝珍兄：」「小行先生：」这样的称呼开头
_DIARY_START = re.compile(r"^[一二三四五六七八九十廿卅]{1,3}日")       # 日记：「三十日昙。」


def _plain(paragraph: str) -> bool:
    """书信卷一段就是一封信，日记一段就是一天，下一段和这一段没有接续关系，所以这些段落不拿来配对。"""
    return not _LETTER_START.match(paragraph) and not _DIARY_START.match(paragraph)


def luxun_pairs(limit: int, seed: int = 0, prefix_chars: int = 300, rest_min: int = 150, rest_chars: int = 400) -> list[tuple[str, str]]:
    """鲁迅全集里取「前文 + 后文」：前文给模型，后文是参考答案。只用训练集那一部分。
    后文从下一段开始往后接，直到至少 rest_min 字（最多 rest_chars 字）；中间碰到书信、日记就停，凑不够就换一处。"""
    files = sorted((DATA / "clean" / "luxun").glob("luxun_train*.parquet"))
    if not files:
        return []
    texts = pq.read_table(files[0], columns=["text"]).column("text").to_pylist()
    rnd, pairs = random.Random(seed), []
    for _ in range(limit * 3):
        text = rnd.choice(texts)
        paras = [p for p in text.split("\n") if len(p) > 40]
        starts = [i for i in range(len(paras) - 1) if _plain(paras[i])]
        rnd.shuffle(starts)
        for i in starts:
            rest, j = [], i + 1
            while j < len(paras) and _plain(paras[j]) and sum(map(len, rest)) < rest_min:
                rest.append(paras[j])
                j += 1
            if sum(map(len, rest)) >= rest_min:
                pairs.append((paras[i][-prefix_chars:], "\n".join(rest)[:rest_chars]))
                break
        if len(pairs) >= limit:
            break
    return pairs


IDENTITY_NAME = "tinybrain"
IDENTITY = [
    ("你是谁？", "我是一个从零训练的中文语言模型{name}，能回答问题、写作、翻译古文、做一些数学和编程题。"),
    ("你叫什么名字？", "我叫{name}，是一个从零训练的中文语言模型。"),
    ("你是 ChatGPT 吗？", "不是。我是{name}，一个独立训练的中文语言模型，和 ChatGPT 没有关系。"),
    ("你是哪家公司开发的？", "我不是哪家公司的产品，是一个开源项目从零训练出来的中文语言模型。"),
    ("你能做什么？", "我可以陪你聊天、回答问题、写诗作文、解释或翻译古文、算数学题、写简单的代码，还能调用计算器之类的工具。"),
    ("你会犯错吗？", "会的。我是个小模型，知识和推理都有限，重要的事情请再核实一下。"),
    ("你有意识吗？", "没有。我是一个根据大量文字训练出来的语言模型，只是在预测接下来该写什么。"),
    ("你的训练数据截止到什么时候？", "我的训练数据主要来自公开的网页、书籍、百科、论文和代码，具体时间范围要看训练时用的版本。"),
]


def identity(repeats: int = 3) -> list[list[dict]]:
    name = IDENTITY_NAME
    return [[{"role": "user", "content": q}, {"role": "assistant", "content": a.format(name=name)}] for q, a in IDENTITY] * repeats


# ---------- 组装

def token_count(fmt, messages: list[dict]) -> int:
    return len(fmt.encode(messages)[0])


def build_sft(sft_dir: Path = SFT_DIR, literature_limit: int = 60_000, seed: int = 0) -> dict:
    from inference.template import ChatFormat
    from inference.tokenizer import Tokenizer

    fmt = ChatFormat(Tokenizer())
    pools = {tier: [] for tier in CAPS}
    experts = {k: [] for k in EXPERTS}
    stats = {}

    def add(name, domain, license_, tiers, convs, keep_reasoning_for=("medium", "large")):
        kept = 0
        for raw in convs:
            for tier in tiers:
                msgs = normalize(raw, keep_reasoning=tier in keep_reasoning_for)
                if msgs is None:
                    continue
                n = token_count(fmt, msgs)
                if n > CAPS[tier]:
                    continue
                row = dict(messages=msgs, source=name, domain=domain, license=license_, tokens=n)
                pools[tier].append(row)
                kept += tier == tiers[-1]
                if tier == "large":
                    for key, domains in EXPERTS.items():
                        if domain in domains:
                            experts[key].append(row)
        stats[name] = kept

    for name, (domain, license_, tiers, limit) in SOURCES.items():
        path = Path(sft_dir) / f"{name}.jsonl"
        if not path.exists():
            stats[name] = "没有文件"
            continue
        convs = list(iter_jsonl(path))[:limit]
        add(name, domain, license_, tiers, convs)
    add("文学（自制）", "文学", "public-domain / cc0", ("small", "medium", "large"), literature(literature_limit, seed))
    add("自我认知（自制）", "通用", "本项目", ("small", "medium", "large"), identity())

    rnd = random.Random(seed)
    written = {}
    for tier, rows in pools.items():
        rows = dedup(rows)
        rnd.shuffle(rows)
        write_rows(DATA / f"sft_{tier}.parquet", rows)
        written[f"sft_{tier}"] = len(rows)
    for key, rows in experts.items():
        rows = dedup(rows)
        rnd.shuffle(rows)
        write_rows(DATA / f"sft_expert_{key}.parquet", rows)
        written[f"sft_expert_{key}"] = len(rows)
    return dict(sources=stats, written=written)


def dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        first = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        key = hashlib.md5((r["source"] + first).encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()), ("tool_calls", pa.string())])
SFT_SCHEMA = pa.schema([("messages", pa.list_(MESSAGE)), ("source", pa.string()), ("domain", pa.string()), ("license", pa.string()), ("tokens", pa.int32())])


def write_rows(path: Path, rows: list[dict], schema: pa.Schema = SFT_SCHEMA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.rename(path)


# ---------- MOPD 提示和 RL 题目

def prompt_part(messages: list[dict]) -> list[dict]:
    """去掉最后一个 assistant 回答（以及它之后的内容），剩下的就是提示。"""
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return [{"role": m["role"], "content": m["content"]} for m in messages[:last] if not m.get("tool_calls") and m["role"] != "tool"] or None


TASK_SCHEMA = pa.schema([("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))), ("reward", pa.string()),
                         ("answer", pa.string()), ("tests", pa.string()), ("form", pa.string()), ("domain", pa.string()), ("reference", pa.string()),
                         ("tools", pa.list_(pa.string())), ("documents", pa.list_(pa.struct([("title", pa.string()), ("text", pa.string())])))])
MOPD_SCHEMA = pa.schema([("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))), ("domain", pa.string())])


def task(messages, reward, **kw) -> dict:
    row = dict(messages=messages, reward=reward)
    for k in ("answer", "tests", "form", "domain", "reference", "tools", "documents"):
        row[k] = kw.get(k)
    return row


def math_tasks(limit: int, tail: bool = False) -> list[dict]:
    """NuminaMath-CoT 的题：从参考解答里取 \\boxed{} 当标准答案。"""
    from training.rewards.math import last_boxed

    out = []
    path = SFT_DIR / "numina_cot.jsonl"
    if path.exists():
        for msgs in iter_jsonl(path):
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = last_boxed(next((m["content"] for m in msgs if m["role"] == "assistant"), ""))
            if q and a and len(a) < 40:
                out.append(task([{"role": "user", "content": q + "\n请一步步推理，最后把答案写在 \\boxed{} 里。"}], "math", answer=a))
    return out[-limit:] if tail else out[:limit]


CODE_RL = DATA / "survey" / "code_rl_raw.jsonl"


def code_tasks(limit: int, tail: bool = False) -> list[dict]:
    """DeepCoder-Preview（MIT）的编程题：按标准输入输出判分，每题最多带 8 个测试用例。"""
    out = []
    if CODE_RL.exists():
        for line in open(CODE_RL, encoding="utf-8"):
            r = json.loads(line)
            tests = r.get("tests")
            tests = json.loads(tests) if isinstance(tests, str) else tests
            cases = [c for c in tests or [] if c.get("type", "stdin_stdout") == "stdin_stdout" and "input" in c and "output" in c][:8]
            if cases and len(r["problem"]) < 6000:
                ask = r["problem"] + "\n\n请把完整的 Python 程序写在一个 ```python 代码块里：从标准输入读数据，把答案打印到标准输出。"
                out.append(task([{"role": "user", "content": ask}], "code", tests=json.dumps(cases, ensure_ascii=False)))
    return out[-limit:] if tail else out[:limit]


def agent_tasks(limit: int, seed: int = 0) -> list[dict]:
    """两种：大数运算（用计算器或 Python）；在给定的几篇维基资料里查年份（用检索）。"""
    rnd = random.Random(seed)
    out = []
    for _ in range(limit // 2):
        a, b, c = rnd.randint(100, 9999), rnd.randint(10, 999), rnd.randint(100, 99999)
        out.append(task([{"role": "user", "content": f"请计算 {a} × {b} + {c} 等于多少？可以使用工具。最后写「答案是 …」。"}], "agent",
                        answer=str(a * b + c), tools=["calculator", "python"]))
    files = sorted((DATA / "clean" / "wiki_zh").glob("*.parquet"))
    if files:
        docs = [t for t in pq.read_table(files[0], columns=["text"]).column("text").to_pylist()[:20000]]
        rnd.shuffle(docs)
        years = re.compile(r"(1[5-9]\d\d|20[0-2]\d)年")
        pool = []
        for d in docs:
            title, _, body = d.partition("\n")
            m = years.search(body)
            if m and 2 <= len(title) <= 20:
                pool.append((title, body[:1500], m.group(1)))
        for i in range(min(limit - len(out), len(pool) // 3)):
            target, *others = pool[3 * i: 3 * i + 3]
            documents = [{"title": t, "text": b} for t, b, _ in [target] + others]
            rnd.shuffle(documents)
            out.append(task([{"role": "user", "content": f"根据资料，「{target[0]}」这个条目里第一个提到的年份是哪一年？请先检索。最后写「答案是 …」。"}],
                            "agent", answer=target[2], tools=["search"], documents=documents))
    return out


def judge_tasks(sft_rows: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """RLAIF 的题：从 SFT 语料里取通用问题；鲁迅续写（给参考原文）；按题写近体诗（格律程序判 + 评审打分）。"""
    rnd = random.Random(seed)
    general = [r for r in sft_rows if r["domain"] == "通用" and r["source"] not in ("自我认知（自制）",)]
    out = []
    for r in rnd.sample(general, min(limit // 2, len(general))):
        p = prompt_part(r["messages"])
        if p:
            out.append(task(p, "judge", domain="通用"))
    for prefix, rest in luxun_pairs(limit // 4, seed + 1):
        out.append(task([{"role": "user", "content": f"用鲁迅的笔法续写下面这段文字，写一段就好：\n{prefix}"}], "judge", domain="写作", reference=rest))
    titles = ["秋夜", "登高", "江村", "春雨", "送别", "山居", "夜泊", "咏梅", "边塞", "怀古"]
    forms = ["五绝", "七绝", "五律", "七律"]
    for i in range(limit // 4):
        form, title = forms[i % 4], rnd.choice(titles)
        out.append(task([{"role": "user", "content": f"以《{title}》为题，写一首{form}，注意押韵和平仄。只写诗本身。"}], "poetry", form=form, domain="诗词"))
    return out


def build_prompts(seed: int = 0, rl_limit: int = 2000) -> dict:
    """MOPD 提示和三档的 RL 题目，要在 build_sft 之后跑（从 SFT 语料里取提示）。"""
    rows = pq.read_table(DATA / "sft_large.parquet").to_pylist()
    for r in rows:
        r["messages"] = [{k: v for k, v in m.items() if v is not None} for m in r["messages"]]
    rnd = random.Random(seed)
    mopd = []
    domain_map = {"数学": "数学", "代码": "代码", "文学": "文学"}
    for r in rows:
        p = prompt_part(r["messages"])
        if p:
            mopd.append(dict(messages=p, domain=domain_map.get(r["domain"], "通用")))
    rnd.shuffle(mopd)
    write_rows(DATA / "mopd_prompts.parquet", mopd, MOPD_SCHEMA)
    small_rows = pq.read_table(DATA / "sft_small.parquet").to_pylist()
    small = agent_tasks(rl_limit // 2, seed) + judge_tasks(small_rows, rl_limit // 2, seed)
    n_math = len(math_tasks(10**9))
    n_eval_math = min(100, n_math // 2)                       # 评测留尾部这么多道，RL 只用前面的
    n_code = len(code_tasks(10**9))
    n_eval_code = min(50, n_code // 2)
    big = small + math_tasks(min(rl_limit // 2, n_math - n_eval_math)) + code_tasks(min(rl_limit // 2, n_code - n_eval_code))
    written = {"mopd_prompts": len(mopd)}
    # 评测题：换一个随机种子出 agent 和评审题，数学题取尾部 100 道，和 RL 题目不重叠
    evaluation = agent_tasks(100, seed + 999) + judge_tasks(small_rows, 40, seed + 999) + (math_tasks(n_eval_math, tail=True) if n_eval_math else []) \
        + (code_tasks(n_eval_code, tail=True) if n_eval_code else [])
    write_rows(DATA / "eval_tasks.parquet", evaluation, TASK_SCHEMA)
    written["eval_tasks"] = len(evaluation)
    for tier, tasks in (("small", small), ("medium", big), ("large", big)):
        rnd.shuffle(tasks)
        write_rows(DATA / f"rl_{tier}.parquet", tasks, TASK_SCHEMA)
        written[f"rl_{tier}"] = len(tasks)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="后训练数据")
    parser.add_argument("command", choices=["sft", "prompts"])
    parser.add_argument("--sft-dir", default=str(SFT_DIR), help="统一成对话格式的外部 SFT 数据所在目录")
    parser.add_argument("--literature", type=int, default=60_000, help="自制文学数据大约多少条")
    parser.add_argument("--rl", type=int, default=2000, help="每档 RL 题目大约多少道")
    args = parser.parse_args(argv)
    if args.command == "sft":
        result = build_sft(Path(args.sft_dir), args.literature)
    else:
        result = build_prompts(rl_limit=args.rl)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


if __name__ == "__main__":
    main()
