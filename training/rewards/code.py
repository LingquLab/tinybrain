"""编程题的判分：从回答里取出代码，在沙箱里跑测试，全部通过得 1 分。

task["tests"] 两种写法：
  - 用 assert 写的 Python：接在代码后面一起跑，正常结束就算通过；
  - JSON 列表 [{"input": …, "output": …}, …]：把 input 喂给程序的标准输入，标准输出（去掉每行末尾空白）要和 output 一样。最多跑前 max_cases 个。
"""
import json
import re
import ast

from .sandbox import run_python

_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    """最后一个 ``` 代码块；没有代码块就当整段都是代码。"""
    blocks = _BLOCK.findall(text)
    if blocks:
        return blocks[-1]
    return text if text.strip() else None


_NONUNIQUE=re.compile(r'(multiple (?:valid )?answers|print any|output any|any of (?:them|the)|any valid|any correct (?:answer|solution)|interactive problem|absolute error|relative error|任意一种|任意一个(?:解|答案)|绝对误差|相对误差)',re.I)


def unsupported_reason(task):
    tests=task.get('tests')
    if not isinstance(tests,str) or not tests.strip():return 'missing_tests'
    if _NONUNIQUE.search('\n'.join(m.get('content','') for m in task.get('messages',[]))):
        return 'requires_custom_output_checker'
    if tests.lstrip().startswith('['):
        try:cases=json.loads(tests)
        except ValueError:return 'malformed_tests'
        if not cases:return 'empty_tests'
        if any(not isinstance(c,dict) or not isinstance(c.get('input'),str) or not isinstance(c.get('output'),str) for c in cases):
            return 'unsupported_case_schema'
    else:
        try:tree=ast.parse(tests)
        except SyntaxError:return 'malformed_assert_tests'
        if not any(isinstance(n,ast.Assert) for n in ast.walk(tree)):return 'missing_assertions'
    return None


def _same_output(got: str, want: str, ignore_case=False) -> bool:
    if ignore_case:got,want=got.casefold(),want.casefold()
    return got.split()==want.split()


def reward(episode, task, timeout: float = 10.0, max_cases: int = 8) -> float:
    code = extract_code(episode.text)
    if code is None:
        return 0.0
    reason=unsupported_reason(task)
    if reason:raise ValueError('Code task cannot be checked by exact I/O: '+reason)
    tests, limit = task["tests"], task.get("timeout") or timeout
    prompt='\n'.join(m.get('content','') for m in task.get('messages',[]))
    ignore_case=bool(re.search(r'case.insensitive|any (?:letter )?case|case does not matter|大小写(?:均可|不限)',prompt,re.I))
    if tests.lstrip().startswith("["):
        for case in json.loads(tests)[:max_cases]:
            ok, out = run_python(code, timeout=limit, stdin=str(case.get("input", "")))
            if not ok or not _same_output(out, str(case.get("output", "")),ignore_case):
                return 0.0
        return 1.0
    ok, _ = run_python(code + "\n\n" + tests, timeout=limit)
    return float(ok)
