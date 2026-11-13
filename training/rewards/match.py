"""对照参考答案判分：
    reading  阅读理解：按字算 F1（预测和参考里共有的字），参考答案可以有几个（task["answer"] 是 JSON 列表），取最好的；
    cloze    诗词填空：去掉标点、空白后要和标准答案（task["answer"]）完全一样，回答也不能长出太多（不能把整首诗都写上碰运气）。
回答里有「答案是 / 答案：」时，只看它后面的部分。
"""
import json
import re
import unicodedata
from collections import Counter

_MARK = re.compile(r"(?:最终答案|答案|(?i:final answer|answer))\s*(?:是|为|:|：|is)\s*")


def normalize(text: str) -> str:
    """全角转半角、小写，去掉空白和标点符号。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith(("P", "S")))


def extract(text: str) -> str:
    parts = _MARK.split(text)
    return parts[-1] if len(parts) > 1 else text


def char_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    common = sum((Counter(p) & Counter(r)).values()) if p and r else 0
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def reading_reward(episode, task) -> float:
    pred = extract(episode.text)
    return max((char_f1(pred, ref) for ref in json.loads(task["answer"])), default=0.0)


def cloze_reward(episode, task) -> float:
    target, out = normalize(str(task["answer"])), normalize(extract(episode.text))
    return float(bool(target) and target in out and len(out) <= 2 * len(target) + 4)
