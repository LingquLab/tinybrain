"""编程题的判分：从回答里取出代码，在沙箱里跑测试，全部通过得 1 分。

task["tests"] 两种写法：
  - 用 assert 写的 Python：接在代码后面一起跑，正常结束就算通过；
  - JSON 列表 [{"input": …, "output": …}, …]：把 input 喂给程序的标准输入，标准输出（去掉每行末尾空白）要和 output 一样。最多跑前 max_cases 个。
"""
import json
import re

from .sandbox import run_python

_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    """最后一个 ``` 代码块；没有代码块就当整段都是代码。"""
    blocks = _BLOCK.findall(text)
    if blocks:
        return blocks[-1]
    return text if text.strip() else None


def _same_output(got: str, want: str) -> bool:
    clean = lambda s: "\n".join(line.rstrip() for line in s.strip().splitlines())
    return clean(got) == clean(want)


def reward(episode, task, timeout: float = 10.0, max_cases: int = 8) -> float:
    code = extract_code(episode.text)
    if code is None:
        return 0.0
    tests, limit = task["tests"], task.get("timeout") or timeout
    if tests.lstrip().startswith("["):
        for case in json.loads(tests)[:max_cases]:
            ok, out = run_python(code, timeout=limit, stdin=str(case.get("input", "")))
            if not ok or not _same_output(out, str(case.get("output", ""))):
                return 0.0
        return 1.0
    ok, _ = run_python(code + "\n\n" + tests, timeout=limit)
    return float(ok)
