"""诗词格律的检查：字数、句数、押韵、平仄，给出 0 到 1 的格律分。

规则（近体诗，简化版）：
  - 句数、每句字数要对：五绝 4×5、七绝 4×7、五律 8×5、七律 8×7；
  - 押韵：偶数句的末字同韵，而且是平声；
  - 平仄：「一三五不论，二四六分明」，只看每句的第二、四、六字和非韵脚句的末字。
    全诗由四种基本句式按「仄起」（ABCD ABCD）或「平起」（CDAB CDAB）排列，取对得更好的那种；首句末字不查（首句入韵时它是平声）。

平仄和韵部用 pypinyin 按今天的普通话近似：一、二声算平，三、四声算仄；韵母相同或相近算同韵。
这和平水韵不完全一样，主要差在入声字（比如「白、月、竹」古代是仄声，今天读平声）。
以后换成平水韵的字表时，只要替换 tone_of 和 rhyme_of 两个函数。
"""
import re

FORMS = {"五绝": (4, 5), "七绝": (4, 7), "五律": (8, 5), "七律": (8, 7)}  # 句数、每句字数
# 四种基本句式，0 平 1 仄
_BASE = {5: {"A": "11001", "B": "00110", "C": "00011", "D": "11100"},
         7: {"A": "0011001", "B": "1100110", "C": "1100011", "D": "0011100"}}
_ORDERS = ("ABCD", "CDAB")
_HAN = re.compile(r"[一-鿿]")
_SAME_RHYME = {"ing": "in", "eng": "en", "ong": "eng", "ie": "e", "ve": "e", "ue": "e", "o": "e", "uo": "e", "ui": "ei", "iu": "ou"}


def lines_of(text: str) -> list[str]:
    """按标点和换行切成句，每句只留汉字。"""
    return [s for s in ("".join(_HAN.findall(p)) for p in re.split(r"[，。！？；、,.!?;\n]", text)) if s]


def tone_of(ch: str) -> int | None:
    """0 平、1 仄；读不出来时返回 None。"""
    from pypinyin import Style, pinyin

    py = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)[0][0]
    if not py or not py[-1].isdigit():
        return None
    return 0 if py[-1] in "12" else 1


def rhyme_of(ch: str) -> str:
    """韵母（去掉声调），近似当作韵部。"""
    from pypinyin import Style, pinyin

    final = pinyin(ch, style=Style.FINALS, strict=False)[0][0]
    return _SAME_RHYME.get(final, final)


def _tone_score(lines: list[str], length: int) -> float:
    best = 0.0
    for order in _ORDERS:
        hit = total = 0
        for i, line in enumerate(lines):
            want = _BASE[length][order[i % 4]]
            spots = [p for p in range(1, length, 2)]          # 第二、四、六字
            if i > 0 and i % 2 == 0:
                spots.append(length - 1)                       # 非韵脚句的末字（首句不查）
            for p in spots:
                t = tone_of(line[p])
                total += 1
                hit += t is not None and t == int(want[p])
        best = max(best, hit / max(1, total))
    return best


def check(text: str, form: str) -> dict:
    """返回各项是否通过和总的格律分（score）。字数句数不对时 score 为 0。"""
    if form not in FORMS:
        raise ValueError(f"不认识的体裁 {form!r}，可选：{'、'.join(FORMS)}")
    n_lines, length = FORMS[form]
    lines = lines_of(text)
    shape = len(lines) == n_lines and all(len(s) == length for s in lines)
    result = dict(shape=shape, rhyme=0.0, tones=0.0, score=0.0)
    if not shape:
        return result
    ends = [lines[i][-1] for i in range(1, n_lines, 2)]
    rhymes = [rhyme_of(c) for c in ends]
    common = max(set(rhymes), key=rhymes.count)
    result["rhyme"] = sum(r == common and tone_of(c) == 0 for r, c in zip(rhymes, ends)) / len(ends)
    result["tones"] = _tone_score(lines, length)
    result["score"] = 0.3 + 0.35 * result["rhyme"] + 0.35 * result["tones"]
    return result


def reward(episode, task) -> float:
    return check(episode.text, task["form"])["score"]
