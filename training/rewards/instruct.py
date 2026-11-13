"""指令跟随：题目带一组可以用程序检查的约束（task["constraints"]，JSON 列表），奖励是满足的比例。

每条约束是 {"type": …, 参数…}：
    max_chars / min_chars  n       字数（不算空白）不超过 / 不少于 n
    max_words / min_words  n       英文词数
    numbered               n       恰好 n 条「1. 2. 3.」编号
    bullets                n       恰好 n 条「- 」开头的列表
    paragraphs             n       恰好 n 段（空行分段；sep 可以换成别的分隔，比如 ***）
    nth_paragraph          n, i, word   恰好 n 段（空行分段），第 i 段以 word 开头
    include / exclude      words   这些词都要出现 / 一个都不能出现
    keyword_freq           word, n, relation   某个词出现的次数
    word_count / sentence_count  n, relation   英文词数 / 句子数
    letter_freq            letter, n, relation 某个字母出现的次数
    capital_words          n, relation         全大写单词的个数
    （relation：at least、at most、less than、exactly、around（相差不超过一成），不写就是 at least）
    starts_with / ends_with text   以这句话开头 / 结尾
    json                   keys    整个回答是一个 JSON 对象（可以包在 ```json 里），并且有这些键
    no_commas                      不用逗号（中英文都算）
    lowercase / uppercase          英文字母全小写 / 全大写
    quoted                         整个回答用双引号括起来
    title                          有一个 <<标题>>
    highlights             n       至少 n 处 *强调*
    sections               marker, n   恰好 n 节「marker 编号」，比如 Section: 1、第1部分
    postscript             marker  有附言（比如 P.S.）
    placeholders           n       至少 n 个 [占位符]
    two_responses                  两个不同的回答，用 ****** 隔开
    one_of                 options 回答就是给定的几句之一
    chinese                        回答以中文为主
"""
import json
import re


def _chars(t: str) -> int:
    return len(re.sub(r"\s", "", t))


def _words(t: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", t))


def _count(t: str, word: str) -> int:
    if re.fullmatch(r"[A-Za-z0-9' ]+", word):
        return len(re.findall(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", t, re.I))
    return t.count(word)


def _json(t: str, keys) -> bool:
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    try:
        obj = json.loads((m.group(1) if m else t).strip())
    except ValueError:
        return False
    return isinstance(obj, dict) and all(k in obj for k in keys or [])


def _relation(count: int, c: dict) -> bool:
    n, relation = c["n"], c.get("relation") or "at least"
    if relation == "around":
        return abs(count - n) <= max(1, n // 10)
    ops = {"at least": count >= n, "at most": count <= n, "less than": count < n, "exactly": count == n}
    if relation not in ops:
        raise ValueError(f"不认识的 relation {relation!r}")
    return ops[relation]


def _paragraphs(t: str, sep: str | None = None) -> list[str]:
    pattern = re.escape(sep) if sep else r"\n\s*\n"
    return [p.strip() for p in re.split(pattern, t.strip()) if p.strip()]


def _two(t: str) -> bool:
    parts = [p.strip() for p in t.split("******")]
    return len(parts) == 2 and all(parts) and parts[0] != parts[1]


def _letters(t: str) -> str:
    return "".join(c for c in t if c.isascii() and c.isalpha())


CHECKS = {
    "max_chars": lambda t, c: _chars(t) <= c["n"],
    "min_chars": lambda t, c: _chars(t) >= c["n"],
    "max_words": lambda t, c: _words(t) <= c["n"],
    "min_words": lambda t, c: _words(t) >= c["n"],
    "numbered": lambda t, c: len(re.findall(r"(?m)^\s*\d+[.、)）]\s*\S", t)) == c["n"],
    "bullets": lambda t, c: len(re.findall(r"(?m)^\s*[-*•]\s+\S", t)) == c["n"],
    "paragraphs": lambda t, c: len(_paragraphs(t, c.get("sep"))) == c["n"],
    "nth_paragraph": lambda t, c: len(_paragraphs(t)) == c["n"] and _paragraphs(t)[c["i"] - 1].lower().startswith(c["word"].lower()),
    "include": lambda t, c: all(_count(t, w) > 0 for w in c["words"]),
    "exclude": lambda t, c: not any(_count(t, w) > 0 for w in c["words"]),
    "keyword_freq": lambda t, c: _relation(_count(t, c["word"]), c),
    "word_count": lambda t, c: _relation(_words(t), c),
    "sentence_count": lambda t, c: _relation(len([x for x in re.split(r"[.!?。！？]+", t) if x.strip()]), c),
    "letter_freq": lambda t, c: _relation(t.lower().count(c["letter"].lower()), c),
    "capital_words": lambda t, c: _relation(len(re.findall(r"\b[A-Z]{2,}\b", t)), c),
    "starts_with": lambda t, c: t.strip().startswith(c["text"]),
    "ends_with": lambda t, c: t.strip().endswith(c["text"]),
    "json": lambda t, c: _json(t, c.get("keys")),
    "no_commas": lambda t, c: "," not in t and "，" not in t,
    "lowercase": lambda t, c: bool(_letters(t)) and _letters(t) == _letters(t).lower(),
    "uppercase": lambda t, c: bool(_letters(t)) and _letters(t) == _letters(t).upper(),
    "quoted": lambda t, c: len(t.strip()) >= 2 and t.strip()[0] in "\"“" and t.strip()[-1] in "\"”",
    "title": lambda t, c: re.search(r"<<[^<>\n]+>>", t) is not None,
    "highlights": lambda t, c: len(re.findall(r"\*[^*\n]+\*", t)) >= c["n"],
    "sections": lambda t, c: len(re.findall(re.escape(c["marker"]) + r"\s*\d+", t)) == c["n"],
    "postscript": lambda t, c: c["marker"] in t,
    "placeholders": lambda t, c: len(re.findall(r"\[[^\[\]\n]+\]", t)) >= c["n"],
    "two_responses": lambda t, c: _two(t),
    "one_of": lambda t, c: t.strip() in c["options"],
    "chinese": lambda t, c: sum("一" <= ch <= "鿿" for ch in t) > 0.3 * max(1, _chars(t)),
}


def check(text: str, constraints: list[dict]) -> list[bool]:
    out = []
    for c in constraints:
        if c["type"] not in CHECKS:
            raise ValueError(f"不认识的约束 {c['type']!r}，可选：{'、'.join(CHECKS)}")
        out.append(bool(CHECKS[c["type"]](text, c)))
    return out


def reward(episode, task) -> float:
    results = check(episode.text, json.loads(task["constraints"]))
    return sum(results) / len(results) if results else 0.0
