"""RL 题目、评测题和 MOPD 提示。要在 python -m dataset.sft 之后跑（要用它留出的问题，还要和 SFT 去重）。

    python -m dataset.download rl       # 先下载原始数据到 data/inputs/rl/（来源和许可见 dataset/sources.py 的 RL_INPUTS）
    python -m dataset.rl                # 再出题

写出：
    data/rl_small.parquet、rl_medium.parquet、rl_large.parquet   三档的 RL 题库
    data/eval_tasks.parquet        评测题：每种题先留出一份，和 RL、SFT 都不重复
    data/mopd_prompts.parquet      MOPD 的提示：SFT 的问题加上 RL 的题；domain 决定交给哪位老师（training/train_distill.py 的 ROUTES）

题型（reward 字段，判分见 training/rewards）和来源：
    math      数学，对答案          SynthLabs Big-Math-RL-Verified（挑 Llama 8B 解题率 0.2–0.95 的，小模型才学得动）、
                                    agentica DeepScaleR-Preview（竞赛题，偏难）、meta-math GSM8K_zh（中文应用题）
    code      编程，跑测试用例      agentica DeepCoder-Preview 的训练集；评测用它的 Codeforces 部分
    logic     逻辑、算术、谜题      reasoning-gym 现场生成（GYM 里的题型），用它自己的判分
    choice    科学选择题            allenai ARC、nvidia OpenScience
    instruct  指令跟随              allenai RLVR-IFeval（英文，约束换成我们的格式）；给留出的中文问题加上中文约束
    reading   中文阅读理解          hfl CMRC2018，按字算 F1
    cloze     诗词填空              从诗词里出：给作者、题目和上句，写下句
    agent     工具调用              程序生成：大数运算、百分比、取模、日期、在维基资料里检索
    judge     评审打分              留出的通用问题（domain 通用）；鲁迅续写（domain 写作，带原文参考）
    poetry    诗词创作，格律加评审  真实的诗题
small 只用 SMALL_KINDS 里的题；medium、large 用全部。每种题的数量见 SIZES，评测留多少见 EVAL。
"""
import argparse
import ast
import datetime
import glob
import json
import random
import re
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.sft import DATA, luxun_pairs, parse_poem, poem_form, prompt_part
from dataset.sources import INPUTS
from training.rewards.match import normalize

RL_RAW = INPUTS / "rl"
SIZES = {"math": 30_000, "code": 24_000, "logic": 10_000, "choice": 12_000, "instruct": 10_000, "reading": 10_000, "cloze": 5_000,
         "agent": 5_000, "general": 5_000, "writing": 2_000, "poetry": 2_000}
EVAL = {"math": 100, "code": 50, "logic": 100, "choice": 100, "instruct": 100, "reading": 100, "cloze": 100, "agent": 100, "general": 30,
        "writing": 10, "poetry": 10}
SMALL_KINDS = ("instruct", "reading", "cloze", "agent", "general", "writing", "poetry")

MATH_HINT = "\n请一步步推理，最后把答案写在 \\boxed{} 里。"
CHOICE_HINT = "\n请选出正确答案，最后写「答案：」加上选项字母。"
LOGIC_HINT = "\n\n最后一行写「答案：」加上你的最终答案。"
AGENT_HINT = "可以使用工具。最后写「答案是 …」。"
# 逻辑题的题型：挑小模型有希望做对、答案好判的
GYM = ["basic_arithmetic", "chain_sum", "decimal_arithmetic", "fraction_simplification", "gcd", "lcm", "prime_factorization", "base_conversion",
       "count_bits", "number_sorting", "number_filtering", "number_sequence", "simple_equations", "letter_counting", "caesar_cipher",
       "spell_backward", "word_sorting", "countdown", "puzzle24", "mini_sudoku", "knights_knaves", "propositional_logic", "family_relationships",
       "calendar_arithmetic", "time_intervals", "leg_counting", "maze", "shortest_path", "tower_of_hanoi", "syllogism", "gsm_symbolic", "products",
       "power_function"]
FAMOUS_POETS = {"李白", "杜甫", "白居易", "王维", "孟浩然", "李商隐", "杜牧", "王昌龄", "刘禹锡", "韩愈", "柳宗元", "苏轼", "陆游", "辛弃疾", "王安石",
                "岑参", "高适", "贺知章", "李清照", "欧阳修", "杨万里", "范成大", "黄庭坚", "温庭筠", "韦应物", "刘长卿", "元稹", "贾岛", "孟郊",
                "骆宾王", "王勃", "杨炯", "卢照邻", "陈子昂", "张九龄", "王之涣", "张籍", "王建", "李贺", "罗隐", "崔颢", "朱熹", "文天祥"}

TASK_SCHEMA = pa.schema([("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))), ("reward", pa.string()),
                         ("answer", pa.string()), ("tests", pa.string()), ("form", pa.string()), ("domain", pa.string()), ("reference", pa.string()),
                         ("tools", pa.list_(pa.string())), ("documents", pa.list_(pa.struct([("title", pa.string()), ("text", pa.string())]))),
                         ("constraints", pa.string()), ("gym", pa.string()), ("entry", pa.string())])
MOPD_SCHEMA = pa.schema([("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))), ("domain", pa.string())])
FIELDS = [f.name for f in TASK_SCHEMA if f.name not in ("messages", "reward")]


def task(content: str, reward: str, domain: str, key: str | None = None, **kw) -> dict:
    """一道题。key 是拿来去重的题干（不含我们加的提示语），不写进文件。"""
    row = dict(messages=[{"role": "user", "content": content}], reward=reward, domain=domain, _key=key or content)
    for k in FIELDS:
        row.setdefault(k, kw.get(k))
    row["domain"] = domain
    return row


def _files(name: str, pattern: str = "*") -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(RL_RAW / name / pattern)))


def _rows(name: str, pattern: str = "*.parquet") -> list[dict]:
    return [r for f in _files(name, pattern) for r in pq.read_table(f).to_pylist()]


# ---------- 程序判分的题

def math_pool(rnd: random.Random) -> list[dict]:
    out = []
    big = [r for r in _rows("bigmath") if r.get("answer") and len(str(r["answer"])) <= 30 and len(r["problem"]) <= 1500
           and r.get("llama8b_solve_rate") not in (None, "") and 0.2 <= float(r["llama8b_solve_rate"]) <= 0.95]
    deep = [r for f in _files("deepscaler", "*.json") for r in json.load(open(f, encoding="utf-8")) if r.get("answer") and len(str(r["answer"])) <= 30]
    zh = []
    for f in _files("gsm8k_zh", "*.json"):
        for r in json.load(open(f, encoding="utf-8")):
            q, a = r.get("question_zh") or r.get("question"), r.get("answer_only") or str(r.get("answer", "")).rpartition("####")[2]
            if q and str(a).strip():
                zh.append(dict(problem=q, answer=str(a).strip().replace(",", "")))
    for rows, n in ((big, 20_000), (deep, 5_000), (zh, 7_500)):
        for r in rnd.sample(rows, min(n, len(rows))):
            out.append(task(r["problem"] + MATH_HINT, "math", "数学", key=r["problem"], answer=str(r["answer"])))
    return out


def _stdin_cases(tests) -> list[dict]:
    tests = json.loads(tests) if isinstance(tests, str) else tests
    if isinstance(tests, dict):  # 有的写成 {"inputs": [...], "outputs": [...]}
        tests = [dict(input=i, output=o) for i, o in zip(tests.get("inputs") or [], tests.get("outputs") or [])]
    return [c for c in tests or [] if isinstance(c, dict) and c.get("type", "stdin_stdout") == "stdin_stdout"
            and isinstance(c.get("input"), str) and isinstance(c.get("output"), str)][:8]


def code_pool(pattern: str) -> list[dict]:
    out = []
    for r in _rows("deepcoder", pattern):
        cases = _stdin_cases(r.get("tests"))
        if cases and len(r["problem"]) < 6000:
            ask = r["problem"].strip() + "\n\n请把完整的 Python 程序写在一个 ```python 代码块里：从标准输入读数据，把答案打印到标准输出。"
            out.append(task(ask, "code", "代码", key=r["problem"], tests=json.dumps(cases, ensure_ascii=False)))
    return out


def logic_pool(n: int, seed: int) -> list[dict]:
    try:
        import reasoning_gym as rg
    except ImportError:
        return []
    from training.rewards.logic import dump_entry

    out, per = [], -(-n // len(GYM))
    for i, name in enumerate(GYM):
        try:
            ds = rg.create_dataset(name, size=per, seed=seed * 1000 + i)
            score = rg.get_score_answer_fn(name)
            entries = [ds[j] for j in range(len(ds))]
        except Exception:  # 版本不同可能没有这个题型
            continue
        for e in entries:
            stored = dump_entry(e)
            try:
                ok = score(str(e["answer"]), json.loads(stored)) >= 1.0  # 存进文件再读回来以后，参考答案还要能判对
            except Exception:
                ok = False
            if ok:
                out.append(task(e["question"] + LOGIC_HINT, "logic", "逻辑", key=e["question"], gym=name, entry=stored))
    return out


def choice_pool() -> list[dict]:
    out = []
    for r in _rows("arc"):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        letters = [chr(ord("A") + k) for k in range(len(labels))]
        key = str(r["answerKey"])
        if key not in labels or len(labels) > 6:
            continue
        options = "\n".join(f"{l}. {t}" for l, t in zip(letters, texts))
        out.append(task(f"{r['question']}\n{options}" + CHOICE_HINT, "choice", "科学", key=r["question"], answer=letters[labels.index(key)]))
    for f in _files("openscience", "*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # 只下了文件开头一段，最后一行可能不完整
                break
            q, a = r.get("input") or "", re.search(r"\\boxed\{\s*([A-J])\s*\}", r.get("output") or "")
            if a and re.search(r"(?m)^\s*[A-J][.)]", q) and len(q) <= 2500:
                body = re.sub(r"(?s)^.*?(?=\S)", "", q).strip()
                out.append(task(body + CHOICE_HINT, "choice", "科学", key=body, answer=a.group(1)))
    return out


def map_ifeval(g: dict) -> dict | None:
    """RLVR-IFeval 的校验函数 → training/rewards/instruct.py 的约束；对不上的返回 None。"""
    f, n, q = g["func_name"], g.get("N"), g.get("quantifier") or "at least"
    table = {
        "validate_lowercase": lambda: {"type": "lowercase"}, "validate_uppercase": lambda: {"type": "uppercase"},
        "validate_no_commas": lambda: {"type": "no_commas"}, "validate_quotation": lambda: {"type": "quoted"},
        "validate_title": lambda: {"type": "title"}, "validate_json_format": lambda: {"type": "json"},
        "validate_two_responses": lambda: {"type": "two_responses"},
        "verify_paragraph_count": lambda: {"type": "paragraphs", "n": n, "sep": "* * *"},
        "verify_keyword_frequency": lambda: {"type": "keyword_freq", "word": g["word"], "n": n, "relation": "exactly"},
        "verify_letter_frequency": lambda: {"type": "letter_freq", "letter": g["letter"], "n": n, "relation": "exactly"},
        "validate_highlighted_sections": lambda: {"type": "highlights", "n": n},
        "validate_repeat_prompt": lambda: {"type": "starts_with", "text": g["original_prompt"].strip()},
        "validate_paragraphs": lambda: {"type": "nth_paragraph", "n": n, "i": g["i"], "word": g["first_word"]},
        "validate_frequency_capital_words": lambda: {"type": "capital_words", "n": n, "relation": q},
        "verify_sentence_constraint": lambda: {"type": "sentence_count", "n": n, "relation": q},
        "validate_word_constraint": lambda: {"type": "word_count", "n": n, "relation": q},
        "validate_end": lambda: {"type": "ends_with", "text": g["end_phrase"].strip()},
        "validate_sections": lambda: {"type": "sections", "marker": g["section_splitter"], "n": n},
        "validate_forbidden_words": lambda: {"type": "exclude", "words": list(g["forbidden_words"])},
        "verify_postscript": lambda: {"type": "postscript", "marker": g["postscript_marker"]},
        "validate_placeholders": lambda: {"type": "placeholders", "n": n},
        "verify_bullet_points": lambda: {"type": "bullets", "n": n},
        "verify_keywords": lambda: {"type": "include", "words": list(g["keyword_list"])},
    }
    return table[f]() if f in table else None  # validate_choice 的选项写法太乱，跳过


# 中文约束：(写给模型的话, 约束, 组)；同一道题里每组最多一条，json 只能单独用
ZH_CONSTRAINTS = [(f"回答不超过 {n} 字。", {"type": "max_chars", "n": n}, "长度") for n in (50, 100, 150, 200)] + \
    [(f"回答不少于 {n} 字。", {"type": "min_chars", "n": n}, "长度") for n in (200, 300)] + \
    [(f"分 {n} 点回答，每点用「1.」「2.」这样的编号开头。", {"type": "numbered", "n": n}, "结构") for n in (3, 4, 5)] + \
    [(f"回答分成 {n} 段，段与段之间空一行。", {"type": "paragraphs", "n": n}, "结构") for n in (2, 3)] + \
    [(f"回答里必须用到「{w}」这个词。", {"type": "include", "words": [w]}, "用词") for w in ("总之", "例如", "首先", "因此")] + \
    [(f"回答里不要出现「{w}」。", {"type": "exclude", "words": [w]}, "用词") for w in ("我认为", "非常", "可以")] + \
    [(f"以「{s}」开头。", {"type": "starts_with", "text": s}, "开头") for s in ("好的，", "简单来说，", "首先，")] + \
    [(f"最后一句写「{s}」。", {"type": "ends_with", "text": s}, "结尾") for s in ("以上就是我的回答。", "希望对你有帮助。")] + \
    [("整个回答不要用逗号。", {"type": "no_commas"}, "标点"),
     ("给回答起一个标题，用 <<>> 括起来，比如 <<我的标题>>。", {"type": "title"}, "标题"),
     ("在回答最后另起一行加一句附言，以「附：」开头。", {"type": "postscript", "marker": "附："}, "结尾"),
     ("用 JSON 格式回答，包含「答案」和「理由」两个键，不要输出别的。", {"type": "json", "keys": ["答案", "理由"]}, "json")]


def instruct_pool(rnd: random.Random, heldout: list[dict]) -> list[dict]:
    out = []
    for r in _rows("ifeval"):
        c = map_ifeval(json.loads(r["ground_truth"]))
        msgs = ast.literal_eval(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if c and user and len(user) <= 4000:
            out.append(task(user, "instruct", "指令跟随", constraints=json.dumps([c], ensure_ascii=False)))
    zh = [h for h in heldout if h["domain"] == "通用" and len(h["messages"]) <= 2]
    zh = [h for h in zh if (q := h["messages"][-1]["content"]) and sum("\u4e00" <= ch <= "\u9fff" for ch in q) > 0.3 * len(q) and len(q) <= 1000]
    for h in zh:
        want, picked, groups = rnd.choice((1, 2)), [], set()
        for text, c, group in rnd.sample(ZH_CONSTRAINTS, len(ZH_CONSTRAINTS)):
            if group in groups or (group == "json" and picked):
                continue
            picked.append((text, c))
            groups.add(group)
            if group == "json" or len(picked) == want:
                break
        q = h["messages"][-1]["content"]
        out.append(task(q + "\n\n" + "".join(t for t, _ in picked), "instruct", "指令跟随", key=q,
                        constraints=json.dumps([c for _, c in picked], ensure_ascii=False)))
    return out


def reading_pool() -> list[dict]:
    out = []
    for r in _rows("cmrc2018"):
        answers = list(dict.fromkeys(a for a in r["answers"]["text"] if a))
        if answers and len(r["context"]) <= 1200:
            ask = f"阅读下面的材料，回答问题。答案尽量直接从材料里摘出来，最后写「答案：…」。\n\n材料：{r['context']}\n\n问题：{r['question']}"
            out.append(task(ask, "reading", "阅读", key=r["context"][-100:] + r["question"], answer=json.dumps(answers, ensure_ascii=False)))
    return out


def _poems() -> list[tuple[str, str, str, str]]:
    poems = []
    for f in sorted((DATA / "clean" / "poetry").glob("*.parquet")):
        poems += [p for p in map(parse_poem, pq.read_table(f, columns=["text"]).column("text").to_pylist()) if p]
    return poems


def cloze_pool(rnd: random.Random, poems) -> list[dict]:
    out = []
    for title, era, author, body in poems:
        if author not in FAMOUS_POETS or not poem_form(body):
            continue
        lines = [s for s in re.split(r"[，。！？；]", body) if s.strip()]
        i = 2 * rnd.randrange(len(lines) // 2)
        upper, lower = lines[i].strip(), lines[i + 1].strip()
        out.append(task(f"{era}·{author}《{title}》里，「{upper}」的下一句是什么？只写下一句。", "cloze", "诗词", answer=lower))
    return out


def poetry_pool(rnd: random.Random, poems) -> list[dict]:
    titles = sorted({t for t, _, a, _ in poems if a in FAMOUS_POETS and re.fullmatch(r"[\u4e00-\u9fff]{2,8}", t) and "其" not in t})
    forms = ["五绝", "七绝", "五律", "七律"]
    return [task(f"以《{t}》为题，写一首{f}，注意押韵和平仄。只写诗本身。", "poetry", "诗词", key=t + f, form=f)
            for t in rnd.sample(titles, len(titles)) for f in [rnd.choice(forms)]]


def _num(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{float(x):.4g}"


def agent_pool(rnd: random.Random, n: int) -> list[dict]:
    """程序生成的工具题：一半算术类（计算器、Python），一半检索类（在给定的几篇维基资料里查）。"""
    out = []
    while len(out) < n // 2:
        kind = rnd.randrange(4)
        if kind == 0:
            a, b, c = rnd.randint(100, 9999), rnd.randint(10, 999), rnd.randint(100, 99999)
            out.append(task(f"请计算 {a} × {b} + {c} 等于多少？{AGENT_HINT}", "agent", "工具调用", answer=str(a * b + c), tools=["calculator", "python"]))
        elif kind == 1:
            a, p = rnd.randint(200, 99999), rnd.choice((5, 12, 15, 25, 35, 62, 85))
            out.append(task(f"{a} 的 {p}% 是多少？{AGENT_HINT}", "agent", "工具调用", answer=_num(Fraction(a * p, 100)), tools=["calculator", "python"]))
        elif kind == 2:
            a, b, m = rnd.randint(2, 99), rnd.randint(5, 60), rnd.randint(7, 997)
            out.append(task(f"{a} 的 {b} 次方除以 {m} 的余数是多少？{AGENT_HINT}", "agent", "工具调用", answer=str(pow(a, b, m)), tools=["python"]))
        else:
            d1 = datetime.date(rnd.randint(1900, 2020), rnd.randint(1, 12), rnd.randint(1, 28))
            d2 = d1 + datetime.timedelta(days=rnd.randint(30, 20000))
            out.append(task(f"从 {d1.year}年{d1.month}月{d1.day}日 到 {d2.year}年{d2.month}月{d2.day}日 一共相隔多少天？{AGENT_HINT}", "agent", "工具调用",
                            answer=str((d2 - d1).days), tools=["python"]))
    files = sorted((DATA / "clean" / "wiki_zh").glob("*.parquet"))
    if files:
        docs = pq.read_table(files[0], columns=["text"]).column("text").to_pylist()[:40_000]
        rnd.shuffle(docs)
        years = re.compile(r"(1[5-9]\d\d|20[0-2]\d)年")
        pool = []
        for d in docs:
            title, _, body = d.partition("\n")
            m = years.search(body)
            if m and 2 <= len(title) <= 20:
                pool.append((title, body[:1500], int(m.group(1))))
        i = 0
        while len(out) < n and i + 3 <= len(pool):
            group, i = pool[i:i + 3], i + 3
            documents = [{"title": t, "text": b} for t, b, _ in group]
            rnd.shuffle(documents)
            (t1, _, y1), (t2, _, y2) = group[0], group[1]
            if rnd.random() < 0.5:
                ask, ans, tools = f"根据资料，「{t1}」这个条目里第一个提到的年份是哪一年？请先检索。最后写「答案是 …」。", str(y1), ["search"]
            else:
                ask, ans, tools = (f"根据资料，「{t1}」和「{t2}」两个条目里第一个提到的年份相差多少年？请先检索，再计算。最后写「答案是 …」。",
                                   str(abs(y1 - y2)), ["search", "calculator"])
            out.append(task(ask, "agent", "工具调用", answer=ans, tools=tools, documents=documents))
    return out


# ---------- 评审打分的题

def general_pool(heldout: list[dict]) -> list[dict]:
    return [dict(task("", "judge", "通用", key=h["messages"][-1]["content"]), messages=h["messages"]) for h in heldout if h["domain"] == "通用"]


def writing_pool(n: int, seed: int) -> list[dict]:
    return [task(f"用鲁迅的笔法续写下面这段文字，写一段就好：\n{prefix}", "judge", "写作", key=prefix, reference=rest)
            for prefix, rest in luxun_pairs(n, seed + 1)]


# ---------- 组装

def _sig(text: str) -> str:
    return normalize(text or "")[-200:]


def sft_signatures() -> set[str]:
    """SFT 语料里所有问题的指纹（去掉标点、空白后的最后 200 个字），拿来保证 RL 题和评测题模型在 SFT 时没见过。"""
    sigs = set()
    for f in sorted(DATA.glob("sft_*.parquet")):
        if f.stem == "sft_heldout":
            continue
        for msgs in pq.read_table(f, columns=["messages"]).column("messages").to_pylist():
            first = next((m["content"] for m in msgs if m["role"] == "user"), "")
            sigs.add(_sig(first))
    return sigs


def build(seed: int = 0, scale: float = 1.0) -> dict:
    """scale 把每种题的数量按比例缩小（试跑用）；评测题的数量不变。"""
    rnd = random.Random(seed)
    heldout = pq.read_table(DATA / "sft_heldout.parquet").to_pylist() if (DATA / "sft_heldout.parquet").exists() else []
    rnd.shuffle(heldout)  # 留出的问题一半给评审题、一半给指令跟随题，同一个问题不会两边都用（否则去重时后一种全被去掉）
    for_judge, for_instruct = heldout[0::2], heldout[1::2]
    poems = _poems()
    sizes = {k: max(1, int(v * scale)) for k, v in SIZES.items()}
    pools = {"math": math_pool(rnd), "code": code_pool("*train*.parquet"), "logic": logic_pool(sizes["logic"] + EVAL["logic"] * 2, seed),
             "choice": choice_pool(), "instruct": instruct_pool(rnd, for_instruct), "reading": reading_pool(), "cloze": cloze_pool(rnd, poems),
             "agent": agent_pool(rnd, sizes["agent"] + EVAL["agent"] * 2), "general": general_pool(for_judge),
             "writing": writing_pool(sizes["writing"] + EVAL["writing"] * 2, seed), "poetry": poetry_pool(rnd, poems)}
    code_eval = code_pool("codeforces*.parquet")  # 评测的编程题单独来自 Codeforces，和训练集分开
    seen = sft_signatures()
    rl, evaluation, stats = {}, [], {}
    for kind, pool in pools.items():
        unique = []
        for t in pool:
            s = _sig(t["_key"])
            if s and s not in seen:
                seen.add(s)
                unique.append(t)
        rnd.shuffle(unique)
        if kind == "code" and code_eval:
            eval_part = [t for t in code_eval if _sig(t["_key"]) not in seen][:EVAL[kind]]
            train_part = unique[:sizes[kind]]
        else:
            eval_part, train_part = unique[:EVAL[kind]], unique[EVAL[kind]:EVAL[kind] + sizes[kind]]
        evaluation += eval_part
        rl[kind] = train_part
        stats[kind] = dict(pool=len(pool), unique=len(unique), rl=len(train_part), eval=len(eval_part))
    for rows in [evaluation] + list(rl.values()):
        for t in rows:
            t.pop("_key", None)
    written = {}
    for tier, kinds in (("small", SMALL_KINDS), ("medium", tuple(rl)), ("large", tuple(rl))):
        rows = [t for k in kinds for t in rl[k]]
        rnd.shuffle(rows)
        _write(DATA / f"rl_{tier}.parquet", rows, TASK_SCHEMA)
        written[f"rl_{tier}"] = len(rows)
    _write(DATA / "eval_tasks.parquet", evaluation, TASK_SCHEMA)
    written["eval_tasks"] = len(evaluation)
    written["mopd_prompts"] = build_mopd(rl, rnd)
    return dict(kinds=stats, written=written)


def build_mopd(rl: dict, rnd: random.Random) -> int:
    """MOPD 的提示：large 档 SFT 语料里的问题（领域按 SFT 的标注）加上全部 RL 题（领域按题型）。"""
    domain_map = {"数学": "数学", "代码": "代码", "文学": "文学"}
    prompts = []
    if (DATA / "sft_large.parquet").exists():
        for r in pq.read_table(DATA / "sft_large.parquet", columns=["messages", "domain"]).to_pylist():
            msgs = [{k: v for k, v in m.items() if v is not None} for m in r["messages"]]
            p = prompt_part(msgs) if any(m["role"] == "assistant" for m in msgs) else None
            if p:
                prompts.append(dict(messages=p, domain=domain_map.get(r["domain"], "通用")))
    prompts += [dict(messages=t["messages"], domain=t["domain"]) for rows in rl.values() for t in rows]
    rnd.shuffle(prompts)
    _write(DATA / "mopd_prompts.parquet", prompts, MOPD_SCHEMA)
    return len(prompts)


def _write(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.rename(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL 题目、评测题和 MOPD 提示")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0, help="每种题的数量乘这个数（试跑时用小一点）")
    args = parser.parse_args(argv)
    result = build(args.seed, args.scale)
    print(f"{'题型':<10}{'原始':>8}{'去重后':>8}{'RL':>8}{'评测':>6}")
    for kind, s in result["kinds"].items():
        print(f"{kind:<10}{s['pool']:>8}{s['unique']:>8}{s['rl']:>8}{s['eval']:>6}")
    print(json.dumps(result["written"], ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
