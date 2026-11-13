"""数学题的判分：从回答里取出最终答案，和标准答案比。取答案的顺序：最后一个 \\boxed{…}；「答案是 / 答案：」后面的内容；最后一个数。"""
import re
from fractions import Fraction

_BOXED = re.compile(r"\\boxed\{")
_ANSWER = re.compile(r"(?:答案|最终答案|answer)\s*(?:是|为|:|：|is)\s*([^\n。]+)", re.I)
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def last_boxed(text: str) -> str | None:
    """最后一个 \\boxed{…} 里的内容（花括号可以嵌套）。"""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i, depth = starts[-1], 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    m = list(_ANSWER.finditer(text))
    if m:
        return m[-1].group(1)
    nums = _NUMBER.findall(text)
    return nums[-1] if nums else None


def normalize(ans: str) -> str:
    ans = ans.strip().strip("$。.，, ").replace(" ", "").replace("\\!", "")
    ans = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\[dt]?frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", ans)
    return ans.replace("\\left", "").replace("\\right", "").replace("{", "").replace("}", "")


def as_number(ans: str) -> Fraction | None:
    s = normalize(ans).replace(",", "").rstrip("%")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def same_answer(got: str | None, want: str) -> bool:
    if got is None:
        return False
    a, b = as_number(got), as_number(want)
    if a is not None and b is not None:
        return a == b
    return normalize(got) == normalize(want)


def reward(episode, task) -> float:
    return float(same_answer(extract_answer(episode.text), str(task["answer"])))
