"""数学题的判分：从回答里取出最终答案，和标准答案比。取答案的顺序：最后一个 \\boxed{…}；「答案是 / 答案：」后面的内容；最后一个数。"""
import re
import json
import operator
import unicodedata
from fractions import Fraction


def exact_decimal(value: Fraction) -> str:
    """Preserve exact terminating decimals; retain a fraction for other rationals."""
    from decimal import Decimal,localcontext
    denominator=value.denominator
    for factor in (2,5):
        while denominator%factor==0:denominator//=factor
    if denominator!=1:return str(value)
    with localcontext() as context:
        context.prec=max(32,len(str(abs(value.numerator)))+len(str(value.denominator))+8)
        return format(Decimal(value.numerator)/Decimal(value.denominator),'f')

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


def strict_answer(text: str) -> str | None:
    """Explicit final answer or a standalone numeric first line; never last-number guessing."""
    tagged = re.findall(r'<answer>\s*([^<>]+?)\s*</answer>', text, re.S)
    if tagged:
        return tagged[-1].strip()
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    marked = list(_ANSWER.finditer(text))
    if marked:
        return marked[-1].group(1).strip()
    first = text.strip().splitlines()[0] if text.strip() else ''
    return first.strip(' 。.') if re.fullmatch(r'\s*[+-]?\d+(?:\.\d+)?(?:/\d+)?\s*[。.]*\s*', first) else None


def strict_reward(episode, task) -> float:
    answer = strict_answer(episode.text)
    if answer is None or '%' in answer:return 0.
    if re.search(r'\d\s+\d',answer):return 0.
    return float(same_answer(answer, str(task['answer'])))


_OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
_TERM = r'[+-]?\d{1,12}(?:\.\d{1,6})?'
_EQUATION = re.compile(rf'\s*({_TERM})\s*([+*/-])\s*({_TERM})\s*=\s*({_TERM})\s*')


def step_credit(text, task):
    """Verify task-specific arithmetic equations, in dependency order, with exact numbers."""
    nodes = json.loads(task['step_checks'])
    equations = []
    normalized = unicodedata.normalize('NFKC', text).translate(str.maketrans('×÷−', '*/-'))
    for part in re.split(r'[\n;。；]', normalized):
        match = _EQUATION.fullmatch(part.rsplit(':', 1)[-1])
        if match:
            a, op, b, result = match.groups()
            equations.append((Fraction(a), op, Fraction(b), Fraction(result)))
    earned = []
    for node in nodes:
        a, b = Fraction(node['a']), Fraction(node['b'])
        op = node['op']
        target = _OPS[op](a, b)
        operands = {(a, b), (b, a)} if op in ('+', '*') else {(a, b)}
        matches = [(i, v) for i, (left, symbol, right, v) in enumerate(equations)
                   if symbol == op and (left, right) in operands]
        deps = [earned[i] for i in node.get('requires', [])]
        after = max((i for i in deps if i is not None), default=-1)
        valid = all(i is not None for i in deps) and all(v == target for _, v in matches)
        earned.append(next((i for i, _ in matches if i > after), None) if valid else None)
    return sum(i is not None for i in earned) / len(nodes)


def process_reward(episode, task):
    return .7 * strict_reward(episode, task) + .3 * step_credit(episode.text, task)
