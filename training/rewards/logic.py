"""逻辑、算术、谜题这类程序生成的题（reasoning-gym，Apache-2.0）：用生成器自己的判分函数，有的题允许不止一个对的答案。

task["gym"] 是题型名（比如 countdown），task["entry"] 是生成时的那一条（dump_entry 存成的 JSON）。
回答里有「答案：/ Final answer:」时，只看它后面的部分。
"""
import json
import re

_MARK = re.compile(r"(?:最终答案|答案|(?i:final answer|answer))\s*(?:是|为|:|：|is)\s*")


def extract_answer(text: str) -> str:
    parts = _MARK.split(text.strip())
    return (parts[-1] if len(parts) > 1 else text).strip().rstrip("。.").strip()


def dump_entry(entry: dict) -> str:
    return json.dumps(entry, ensure_ascii=False, default=str)


def reward(episode, task) -> float:
    import reasoning_gym  # 只有逻辑题才要它

    score = reasoning_gym.get_score_answer_fn(task["gym"])
    return float(score(extract_answer(episode.text), json.loads(task["entry"])))
