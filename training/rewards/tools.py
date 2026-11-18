"""基础工具调用的环境：计算器、跑 Python、在任务给的资料里检索。agent RL 和评测共用。

每个任务（RL 数据的一行）可以带：
    tools      这一题能用哪些工具，比如 ["calculator", "python"]；不写就是全部
    documents  search 工具能检索的资料，[{"title": …, "text": …}]
    answer     标准答案，最后的回答和它比（用数学题的比法）
"""
import ast
import json
import math
import operator

from . import math as math_reward
from .sandbox import run_python

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> str:
    """只认数字和加减乘除、乘方、取模，不执行任意代码。"""
    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("指数太大")
            left_bits=abs(left).bit_length() if isinstance(left,int) else 1024
            right_bits=abs(right).bit_length() if isinstance(right,int) else 1024
            if isinstance(node.op,ast.Pow) and left_bits*max(1,abs(right))>4096:
                raise ValueError('计算结果太大')
            if isinstance(node.op,ast.Mult) and left_bits+right_bits>4096:
                raise ValueError('计算结果太大')
            result=_OPS[type(node.op)](left, right)
            if type(result) not in (int,float) or (isinstance(result,int) and result.bit_length()>4096) or (isinstance(result,float) and not math.isfinite(result)):
                raise ValueError('计算结果超出范围')
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("只支持数字和 + - * / ** % //")
    try:
        tree=ast.parse(expression.replace("×", "*").replace("÷", "/"), mode="eval")
        if sum(1 for _ in ast.walk(tree))>128:raise ValueError('表达式太长')
        value = ev(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return f"计算出错：{e}"
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


SPECS = {
    "calculator": {"description": "计算一个算术表达式", "arguments": {"expression": "比如 (3+4)*12"}},
    "python": {"description": "运行一段 Python 代码，返回打印出来的内容", "arguments": {"code": "Python 代码"}},
    "search": {"description": "在给定的资料里按关键词检索，返回最相关的几段", "arguments": {"query": "关键词"}},
}


class ToolEnv:
    def __init__(self, task: dict | None = None, python_timeout: float = 5.0):
        task = task or {}
        self.allowed = set(task.get("tools") or SPECS)
        self.documents = task.get("documents") or []
        self.python_timeout = python_timeout
        self.calls = 0

    def describe(self) -> str:
        """放进 system 提示里的工具说明。"""
        specs = {name: SPECS[name] for name in SPECS if name in self.allowed}
        return ("你可以调用下面的工具。需要工具时只输出调用 JSON：{\"name\": 工具名, \"arguments\": {…}}，也支持训练中的工具调用标记。工具的结果会作为下一轮给你，再根据结果回答。\n"
                + json.dumps(specs, ensure_ascii=False))

    def call(self, name: str, arguments: dict) -> str:
        self.calls += 1
        if name not in SPECS:
            return f"没有这个工具：{name}"
        if name not in self.allowed:
            return f"这一题不能用 {name}"
        if not isinstance(arguments, dict):
            return "arguments 要是一个对象"
        if name == "calculator":
            return calculate(str(arguments.get("expression", "")))
        if name == "python":
            ok, out = run_python(str(arguments.get("code", "")), timeout=self.python_timeout)
            return out if ok else f"运行出错：\n{out}"
        return self._search(str(arguments.get("query", "")))

    def _search(self, query: str, k: int = 3) -> str:
        words = [w for w in query.replace("，", " ").split() if w] or list(query)
        scored = sorted(((sum(d["text"].count(w) + 3 * d.get("title", "").count(w) for w in words), i) for i, d in enumerate(self.documents)), reverse=True)
        hits = [self.documents[i] for s, i in scored[:k] if s > 0]
        return "\n\n".join(f"【{d.get('title', '')}】{d['text'][:1500]}" for d in hits) or "没有找到相关资料"


def reward(episode, task) -> float:
    """最后的回答对不对；格式坏掉的工具调用每次扣 0.1。"""
    correct = math_reward.strict_reward(episode,task)
    if task.get('tool_evidence'):
        evidence = json.loads(task['tool_evidence'])
        passed = 0
        for turn in episode.turns:
            for call, result in zip(turn['calls'], turn.get('results', [])):
                if passed == len(evidence):
                    break
                required = evidence[passed]
                if call.get('name') != required['name']:
                    continue
                if required['name'] == 'search':
                    valid = any(line.startswith('【' + required['title'] + '】') for line in result.splitlines())
                else:
                    expression = str(call.get('arguments', {}).get('expression', '')).replace(' ', '').replace('×','*').replace('÷','/')
                    valid = expression in required['expressions'] and result == required['value']
                passed += int(valid)
        fraction = passed / len(evidence)
        return .3 * fraction + .7 * correct * (passed == len(evidence) and episode.finished)
    prompt='\n'.join(m.get('content','') for m in task.get('messages',[]))
    if '请先检索' in prompt:
        searched=any(call.get('name')=='search' and isinstance(call.get('arguments'),dict)
                     and str(call['arguments'].get('query','')).strip() and index<len(turn.get('results',[]))
                     and turn['results'][index].startswith('【')
                     for turn in episode.turns for index,call in enumerate(turn['calls']))
        if not searched:correct=0.
    broken = sum(len(t["broken"]) for t in episode.turns)
    return correct - 0.1 * broken
