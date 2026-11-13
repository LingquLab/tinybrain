"""后训练的 SFT 数据：三份 SFT 语料、三个专家（数学、代码、文学）的 SFT 数据，以及留给 RL 出题用的一小份提示。

    python -m dataset.sft        # data/sft_{small,medium,large}.parquet、data/sft_expert_{math,code,literature}.parquet、data/sft_heldout.parquet
    python -m dataset.rl         # 之后再跑：RL 题目、评测题、MOPD 提示（见 dataset/rl.py）

输入：
  - 外部 SFT 数据：dataset/sources.py 的 SFT 表，每个来源一个统一成对话格式的 jsonl（{"messages": [...]}），
    在 data/inputs/sft/<名字>.jsonl（python -m dataset.download sft 下载并转换，每个来源最多 10 万条）；
  - 我们自己的数据：古籍库的文白互译（data/inputs/gujilab_translate.jsonl，只用开头几千条）、洗好的诗词（data/clean/poetry）、
    洗好的鲁迅全集（data/clean/luxun）。不读 data/raw：原始下载在数据集做完以后会删掉。

处理（所有来源一样）：
  - 全部转成简体；
  - 去掉自称是别家模型的回答（「我是 ChatGPT」「由 OpenAI 开发」这类），去掉空回答；
  - 推理过程：small 版去掉，只留回答；medium、large 版写成 <think>…</think> 放在回答前面；
  - 按正式分词器编出来的真实长度，按档截断：small 2048、medium 4096、large 8192；
  - 同一份语料里按第一个问题去重；
  - 每个外部来源按第一个问题的哈希留出约 2%，不进 SFT，写进 sft_heldout.parquet，给 RL 的评审题、指令跟随题和评测出题，
    这样 RL 和评测的题模型在 SFT 时没见过；
  - small 档更精：每个外部来源内部按回答的质量分（dataset/clean.py 的 quality）只留好的一半；自制的文学、自我认知数据全留。
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.clean import to_simplified
from dataset.sources import DATA, INPUTS, SFT

SFT_DIR = INPUTS / "sft"
CAPS = {"small": 2048, "medium": 4096, "large": 8192}
HELDOUT_EVERY = 50     # 每 50 条留出 1 条给 RL 和评测
SMALL_KEEP = 0.5       # small 档每个外部来源留质量好的这么多

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
    path = INPUTS / "gujilab_translate.jsonl"
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
    """洗好的诗词是「题目\n朝代·作者\n正文」（dataset/sources.py 的 _poem）。"""
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


def held_out(messages: list[dict]) -> bool:
    """按第一个问题的哈希决定这条对话留不留给 RL 和评测（固定的，重跑结果一样）。"""
    first = next((m["content"] for m in messages if m["role"] == "user"), "") or ""
    return int(hashlib.md5(first.encode("utf-8")).hexdigest(), 16) % HELDOUT_EVERY == 0


def answer_quality(row: dict) -> float:
    """一条对话里助手回答的质量分（按领域和语言选清洗时的种类）。"""
    from dataset.clean import quality

    text = "\n".join(m["content"] or "" for m in row["messages"] if m["role"] == "assistant")
    if row["domain"] == "代码":
        kind = "code"
    elif row["domain"] == "数学":
        kind = "math"
    else:
        kind = "zh" if sum("\u4e00" <= c <= "\u9fff" for c in text) > 0.2 * max(1, len(text)) else "en"
    return quality(text, kind) if text.strip() else 0.0


def keep_better_part(rows: list[dict], share: float, seed: int = 0) -> list[dict]:
    """每个外部来源（SFT 表里的）只留质量分高的 share；其余来源（自制的）全留。"""
    rnd = random.Random(seed)
    by_source, out = {}, []
    for r in rows:
        (by_source.setdefault(r["source"], []) if r["source"] in SFT else out).append(r)
    for rs in by_source.values():
        scored = sorted(rs, key=lambda r: (-answer_quality(r), rnd.random()))
        out += scored[: max(1, round(len(scored) * share))]
    return out


def build_sft(sft_dir: Path = SFT_DIR, literature_limit: int = 60_000, seed: int = 0) -> dict:
    from inference.template import ChatFormat
    from inference.tokenizer import Tokenizer

    fmt = ChatFormat(Tokenizer())
    pools = {tier: [] for tier in CAPS}
    experts = {k: [] for k in EXPERTS}
    heldout = []
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

    for name, spec in SFT.items():
        domain, license_, tiers = spec["domain"], spec["license"], spec["tiers"]
        path = Path(sft_dir) / f"{name}.jsonl"
        if not path.exists():
            stats[name] = "没有文件"
            continue
        convs = list(iter_jsonl(path))
        for raw in convs:
            if held_out(raw):
                p = prompt_part(raw) if any(m["role"] == "assistant" for m in raw) else None
                if p:
                    heldout.append(dict(messages=[{"role": m["role"], "content": to_simplified(m["content"] or "")} for m in p], source=name, domain=domain))
        add(name, domain, license_, tiers, [raw for raw in convs if not held_out(raw)])
    add("文学（自制）", "文学", "public-domain / cc0", ("small", "medium", "large"), literature(literature_limit, seed))
    add("自我认知（自制）", "通用", "本项目", ("small", "medium", "large"), identity())

    rnd = random.Random(seed)
    written = {}
    for tier, rows in pools.items():
        rows = dedup(rows)
        if tier == "small":
            rows = keep_better_part(rows, SMALL_KEEP, seed)
        rnd.shuffle(rows)
        write_rows(DATA / f"sft_{tier}.parquet", rows)
        written[f"sft_{tier}"] = len(rows)
    for key, rows in experts.items():
        rows = dedup(rows)
        rnd.shuffle(rows)
        write_rows(DATA / f"sft_expert_{key}.parquet", rows)
        written[f"sft_expert_{key}"] = len(rows)
    write_rows(DATA / "sft_heldout.parquet", heldout, HELDOUT_SCHEMA)
    written["sft_heldout"] = len(heldout)
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


# ---------- 留给 RL 出题的提示

def prompt_part(messages: list[dict]) -> list[dict]:
    """去掉最后一个 assistant 回答（以及它之后的内容），剩下的就是提示。"""
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return [{"role": m["role"], "content": m["content"]} for m in messages[:last] if not m.get("tool_calls") and m["role"] != "tool"] or None


PROMPT_MESSAGES = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
HELDOUT_SCHEMA = pa.schema([("messages", PROMPT_MESSAGES), ("source", pa.string()), ("domain", pa.string())])


# ---------- 各家 SFT 数据的格式统一成 [{"role", "content", ("reasoning", "tool_calls")}]（dataset/download.py 下载时用）

ROLE = {"human": "user", "user": "user", "prompter": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant", "bot": "assistant",
        "chatgpt": "assistant", "system": "system", "tool": "tool", "function": "tool", "observation": "tool", "ipython": "tool", "function_call": "assistant"}
Q_KEYS = ("instruction", "prompt", "question", "problem", "query", "input")
A_KEYS = ("output", "response", "answer", "solution", "completion", "content", "chosen")


def content_of(m: dict) -> str:
    c = m.get("content", m.get("value", m.get("text")))
    if isinstance(c, list):  # 多模态那种 [{"type": "text", "text": …}]
        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else (json.dumps(c, ensure_ascii=False) if c is not None else "")


def to_messages(row: dict) -> list[dict] | None:
    """一行原始数据 → 对话；认不出来返回 None。认得的写法：messages / conversations 这类对话列表（role 或 from），或者问答两个字段。"""
    for key in ("messages", "conversations", "conversation", "dialog", "dialogue"):
        conv = row.get(key)
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except ValueError:
                conv = None
        if isinstance(conv, list) and conv and isinstance(conv[0], dict):
            out = [{"role": "system", "content": row["system"]}] if isinstance(row.get("system"), str) and row["system"].strip() else []
            for m in conv:
                role = ROLE.get(str(m.get("role", m.get("from", ""))).lower())
                if role is None:
                    return None
                msg = {"role": role, "content": content_of(m)}
                if m.get("reasoning_content") or m.get("reasoning") or m.get("thinking"):
                    msg["reasoning"] = m.get("reasoning_content") or m.get("reasoning") or m.get("thinking")
                if m.get("tool_calls") or m.get("function_call"):
                    msg["tool_calls"] = m.get("tool_calls") or [m.get("function_call")]
                out.append(msg)
            return out
    q = next((row[k] for k in Q_KEYS if isinstance(row.get(k), str) and row[k].strip()), None)
    a_key = next((k for k in A_KEYS if isinstance(row.get(k), str) and row[k].strip() and k != "input"), None)
    if q is None or a_key is None:
        return None
    if isinstance(row.get("instruction"), str) and isinstance(row.get("input"), str) and row["input"].strip():
        q = row["instruction"] + "\n" + row["input"]
    ans = {"role": "assistant", "content": row[a_key]}
    if row.get("reasoning_content"):
        ans["reasoning"] = row["reasoning_content"]
    out = [{"role": "system", "content": row["system"]}] if isinstance(row.get("system"), str) and row["system"].strip() else []
    return out + [{"role": "user", "content": q}, ans]


def main(argv=None):
    parser = argparse.ArgumentParser(description="后训练的 SFT 数据")
    parser.add_argument("--sft-dir", default=str(SFT_DIR), help="统一成对话格式的外部 SFT 数据所在目录")
    parser.add_argument("--literature", type=int, default=60_000, help="自制文学数据大约多少条")
    args = parser.parse_args(argv)
    result = build_sft(Path(args.sft_dir), args.literature)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


if __name__ == "__main__":
    main()
