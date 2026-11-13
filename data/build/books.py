"""整本书的处理：去掉古腾堡计划加在书前书后的说明，再把一本书切成一篇篇不太长的文档。"""
import re

_START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n|\*END[* ]*THE SMALL PRINT[^\n]*\n|^[^\n]{0,80}\*\*\*\n")
_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG|^\s*End of (?:the |this )?Project Gutenberg", re.M | re.I)
_CREDIT = re.compile(r"(?:Produced|E-?text prepared|Transcribed|Prepared|Scanned|Digitized|Proofread|This e-?text was (?:produced|prepared))\b[^\n]{0,20}\b(?:by|from)\b", re.I)
_ILLUSTRATION = re.compile(r"\[Illustration[^\]]{0,400}\]\n?")
_ITALIC = re.compile(r"(?<![A-Za-z0-9_])_([^_\n]{1,80})_(?![A-Za-z0-9_])")
_BLOCKS = re.compile(r"\n[ \t　]*\n\s*")
_HEADING = re.compile(r"\s*(?:(?:CHAPTER|Chapter|BOOK|Book|PART|Part|VOLUME|Volume|LETTER|ACT|SCENE)\s+[IVXLCDM\d]+\b|[IVXLC]+\.\s*$|第[〇零一二三四五六七八九十百千两\d]+[章回卷节篇部集折出]|卷[之第]?[〇零一二三四五六七八九十百千\d]+)")
_SENTENCE_END = re.compile(r"[。！？!?；;](?=[^”’」』）)]|$)|\.(?=\s)")


def strip_gutenberg(text: str) -> str:
    """去掉古腾堡的开头声明、结尾声明、制作人署名、插图占位和表示斜体的下划线。"""
    head = None
    for head in _START.finditer(text, 0, len(text) // 3):
        pass  # 取前三分之一里最后一个开头标记
    if head:
        text = text[head.end():]
    tail = _END.search(text, len(text) // 2)
    if tail:
        text = text[:tail.start()]
    blocks = _BLOCKS.split(text.strip())
    while blocks and len(blocks[0]) < 600 and _CREDIT.match(blocks[0].lstrip()):
        blocks.pop(0)
    return _ITALIC.sub(r"\1", _ILLUSTRATION.sub("", "\n\n".join(blocks))).strip()


def _pieces(block: str, limit: int) -> list[str]:
    """一个超过上限的块，先按行切，行还太长就按句切，再不行硬切。"""
    if len(block) <= limit:
        return [block]
    if "\n" in block:
        return [piece for line in block.split("\n") for piece in _pieces(line, limit)]
    out, start = [], 0
    while len(block) - start > limit:
        ends = [m.end() for m in _SENTENCE_END.finditer(block, start, start + limit)]
        cut = ends[-1] if ends and ends[-1] - start >= limit // 4 else start + limit
        out.append(block[start:cut])
        start = cut
    out.append(block[start:])
    return out


def split_book(text: str, max_chars: int) -> list[str]:
    """把一本书切成每篇不超过 max_chars 的文档：只在段落之间切，写够一半以后遇到章节标题就另起一篇。

    太短的结尾并到前一篇里（所以最后一篇可能略超上限）。
    """
    docs, current, size = [], [], 0
    for block in _BLOCKS.split(text.strip()):
        for piece in _pieces(block, max_chars):
            sep = "\n\n" if piece is block else "\n"
            if current and (size + len(piece) > max_chars or (size >= max_chars // 2 and _HEADING.match(piece))):
                docs.append("".join(current).strip())
                current, size = [], 0
            current.append(piece + sep)
            size += len(piece) + len(sep)
    last = "".join(current).strip()
    if last:
        if docs and len(last) < max_chars // 8:
            docs[-1] += "\n\n" + last
        else:
            docs.append(last)
    return docs
