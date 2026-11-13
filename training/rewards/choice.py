"""选择题：从回答里找出最后选定的那个选项字母，和标准答案（task["answer"]，比如 "C"）比。

找的顺序：最后一个 \\boxed{…} 里单独的字母；「答案是 C」「选 C」「the answer is (C)」这类说法里的最后一个；回答最后单独的一个字母。
"""
import re

from .math import last_boxed

_SAID = re.compile(r"(?:正确答案|最终答案|答案|应选|选择|选项|选|(?i:the answer is|answer))\s*(?:是|为|:|：)?\s*[（(]?\s*([A-J])\s*[）)]?(?![A-Za-z])")
_AT_END = re.compile(r"(?<![A-Za-z])[（(]?([A-J])[）)]?[。.]?\s*$")


def extract_choice(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        m = re.fullmatch(r"\s*[（(]?([A-J])[）)]?\s*", boxed)
        if m:
            return m.group(1)
    said = _SAID.findall(text)
    if said:
        return said[-1]
    m = _AT_END.search(text)
    return m.group(1) if m else None


def reward(episode, task) -> float:
    return float(extract_choice(episode.text) == str(task["answer"]).strip().upper())
