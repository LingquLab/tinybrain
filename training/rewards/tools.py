"""基础工具调用的环境：计算器、跑 Python、在任务给的资料里检索。agent RL 和评测共用。

每个任务（RL 数据的一行）可以带：
    tools      这一题能用哪些工具，比如 ["calculator", "python"]；不写就是全部
    documents  search 工具能检索的资料，[{"title": …, "text": …}]
    answer     标准答案，最后的回答和它比（用数学题的比法）
"""
import ast
import json
import operator

from . import math as math_reward
from .sandbox import run_python

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> str:
    """只认数字和加减乘除、乘方、取模，不执行任意代码。"""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("指数太大")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("只支持数字和 + - * / ** % //")
    try:
        value = ev(ast.parse(expression.replace("×", "*").replace("÷", "/"), mode="eval").body)
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
        return ("你可以调用下面的工具。调用时在回答里写一个 JSON：{\"name\": 工具名, \"arguments\": {…}}，工具的结果会作为下一轮给你。\n"
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
        return "\n\n".join(f"【{d.get('title', '')}】{d['text'][:500]}" for d in hits) or "没有找到相关资料"


def reward(episode, task) -> float:
    """最后的回答对不对；格式坏掉的工具调用每次扣 0.1。"""
    correct = float(math_reward.same_answer(math_reward.extract_answer(episode.text), str(task["answer"])))
    broken = sum(len(t["broken"]) for t in episode.turns)
    return correct - 0.1 * broken
