"""数据集精洗：把一篇原始文档变成干净的文本，并给出质量分和去重用的指纹。

一篇文档按「种类」处理，种类决定哪些步骤适用：

    zh         现代中文文章、百科、合成文本
    classical  古文和诗词（短，标点习惯不同）
    en         英文文章、百科、论文
    math       数学文本（英文为主，保留 LaTeX）
    code       源代码（保留缩进，不做隐私打码，数字不能动）

入口是 clean_document()；其余函数各做一步，可以单独测试。
"""

import hashlib
import html
import math
import re
import zlib

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏ -‮⁠-⁤﻿�-]")
_FULLWIDTH = {c: c - 0xFEE0 for c in [*range(0xFF10, 0xFF1A), *range(0xFF21, 0xFF3B), *range(0xFF41, 0xFF5B)]}
_FULLWIDTH[0x3000] = 0x20  # 全角空格
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
HAN = re.compile(r"[㐀-䶿一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")
_ZH_PUNCT = re.compile(r"[，。！？；：、,.;:!?]")
_HALFWIDTH_CJK = str.maketrans("｡､｢｣", "。、「」")
_ASCII_PUNCT_IN_HAN = re.compile(r"(?<=[\u4e00-\u9fff”’）》」])([,;:?!.])(?=[\u4e00-\u9fff“‘（《「])")
_TO_FULLWIDTH = {",": "，", ";": "；", ":": "：", "?": "？", "!": "！", ".": "。"}

# 只出现在网页外壳里的短行，分三种：
#   带标签的（「来源：某某」）——关键词后面必须跟冒号或竖线，否则「作者认为……」这种正文会被误删；
#   导航和声明（「上一篇」「点击查看」）——以这些词开头；
#   单独成行的按钮字样（「登录」「关闭」）——整行只有这个词。
_BOILER_LINE = re.compile(
    r"^(来源|出处|稿源|作者|编辑|责任编辑|责编|原标题|发布时间|发表时间|更新时间|浏览次数|阅读次数|点击数|字体|字号|标签|关键词|分享到?)\s*[:：|｜丨]|"
    r"^(上一篇|下一篇|上一页|下一页|相关阅读|相关推荐|相关文章|延伸阅读|推荐阅读|点击(查看|进入|下载|阅读|这里)|扫一扫|扫码|"
    r"关注(我们|公众号|微信)|微信公众号|版权所有|版权声明|转载请|免责声明|返回(顶部|首页|列表)|您还未登[录陆]|请登录)|"
    r"^(登录|注册|收藏|打印|关闭|更多|分享|目录|导航|首页|去咨询|相关知识推荐|加载中|我的收藏|查看|编辑|文档介绍|扩展资料[:：]?|\d*人看过|导语[】:：]?)$|"
    r"^#[^#\s]{1,14}#$|^该文档贡献者很忙"
)
_BOILER_MAX = 40  # 超过这个长度的行不当作外壳，哪怕以上面的词开头
# 以这些话开头的行是外壳，但只删 _BOILER_ANY_MAX 字以内的：更长的行往往是「本文为某项目成果。正文……」写在了一起
_BOILER_ANY_MAX = 80
_BOILER_ANY = re.compile(
    r"^(爱思想（aisixiang|在方框中输入电子邮件地址|暂无评论|以下评论只代表网友个人观点|将文章分享到朋友圈|他们还推荐了|推荐过这篇日志的人|"
    r"阅读原文$|展开全部$|展开全文$|查看全文$|点击展开|本文(来源|来自|转自|选自|摘自|系|为)|文章来源|图片来源|声明：|特别声明|（?责任编辑|【全文】|"
    r"<? ?上一页|\d+人看过$|微信扫一扫|长按(识别|二维码)|欢迎(关注|订阅|转载)|未经(许可|授权)|如需转载|打开APP|原文(载|刊)于?《|注释从略|参考资料(来源)?[:：]\s*百度)"
)
_PUNCT_ONLY = re.compile(r"^[\W_]*(The End|THE END|END|完|全文完)?[\W_]*$")  # 只有标点，或者只是一个「完」
_INLINE_JUNK = re.compile(r"(?m)\(I_M_G\)|\$P\$|\[page\]|参考资料来源[:：]\s*百度百科[-《][^》\s]{0,30}》?|图片来源[:：][^。，\n]{1,25}摄|[（(]完[）)]\s*$|请点击输入图片描述(（[^）]{0,12}）)?|[（(]本文来源[:：][^）)\n]{1,40}[）)]|点击阅读☞[:：]?|\[\d{1,3}(?:[-–,，]\d{1,3})*\]|（[；;，,、\s]*）|\([;,\s]*\)|[（(]图[）)]")
_ENTITY = re.compile(r"&(#\d{2,6}|#x[0-9a-fA-F]{2,5}|[a-zA-Z]{2,8});")
_TAG = re.compile(r"</?(p|br|div|span|sub|sup|strong|em|b|i|u|font|a|li|ul|ol|td|tr|table|h[1-6])(\s[^<>]{0,80}|=\"\")?/?>", re.I)
_URL = re.compile(r"((原文|论文|文章|参考|相关)?(链接|网址|地址)[:：]?\s*)?"
                  r"(https?://|www\.)[A-Za-z0-9./?&=%_~#:+,;@!$'()*-]+|https?\s*[:：]\s*/\s*/\s*(?:[A-Za-z0-9_~%?&=#+-]+\s*[./]\s*)*[A-Za-z0-9_~%?&=#+-]*")
_BOLD = re.compile(r"(?<!\*)\*\*(?!\*)([^*\n]{1,60})(?<!\*)\*\*(?!\*)")  # 成对的 **加粗**；连串的星号是屏蔽字，不算
_SENTENCES = re.compile(r"[^。！？!?\n]*[。！？!?\n]+[】）)\]」』”’]*|[^。！？!?\n]+$")  # 一句话连同句末标点和紧跟的右括号
_CENSORED = re.compile(r"[\u4e00-\u9fff]\*{2,}|\*{2,}[\u4e00-\u9fff]")
# 整句删掉的：版权转载声明、引导关注扫码。只删 150 字以内的句子
_NOISE_SENTENCE = re.compile(
    r"版权归原?作者|版权声明|如涉(及)?版权|(如有|若有|如涉及?|若涉及?)侵权|侵权.{0,4}请.{0,6}(联系|告知|删除)|转载目的在于|转载(仅供|请|须知|时|自[:：])|如需转载|本文转载|"
    r"未经.{0,12}(授权|许可|允许|同意).{0,12}(转载|摘编|使用本)|未经(授权|许可|允许)\s*禁止|(转载|摘编).{0,12}(须|需|请).{0,6}(授权|许可|注明)|"
    r"仅供(思考|参考|学习|交流).{0,12}(版权|立场|建议)|不代表.{0,10}(观点|立场)|免责声明|本(公众号|平台|网站|站)(所[载发]|转载|不)|后台留言|"
    r"扫描?(下方|上方|以下)?.{0,4}二维码|长按.{0,6}二维码|点击(下方|上方|左下角|文末).{0,6}(链接|名片|蓝字|阅读原文|关注)|关注(本|我们的?)?(微信)?公众号|"
    r"凡(本号)?注明.{0,4}来源|本(服务号|订阅号)|微信(ID|号)[:：]|新浪微博[:：]|点击阅读原文|阅读原文查看|返回搜狐|滑动查看更多|点击此处阅读|"
    r"扫码关注|赶紧扫码|在微信上关注|点击阅读全文|打开CSDN|文章来源[:：]|授权转载|转载授权|联系电话/微信|"
    r"可以关注我们|你们的支持是我们|侵犯您的(版权|权益)|非授权发布|未经.{0,8}(允许|授权|许可|同意).{0,4}不得转载|"
    r"(图片|文字|内容|素材)均?(来源于|来自|从)(网络|互联网)|本公众号|本文(节选|选|摘|转)自|来源[:：]\s*网络|"
    r"谢谢你看完|点亮.{0,4}赞|与新浪网联系|转载于|请继续关注我们|点.{0,3}阅读原文|搜索公众号|本文共\d+页|本文目录|阅读全文\s*>|点击可?查看更多"
)
# 句尾的「来源：某某」。前面紧挨着汉字的不算（「资料来源：」「数据来源：」是正文里的图表说明）
_TAIL_SOURCE = re.compile(r"(?<!资料)(?<!数据)(?<!信息)(?<!主要)(?<!收入)[（(]?(本文|图片|文章|稿件)?(来源|稿源|出处)[:：][^。！？\n]{1,40}[）)]?\s*$", re.M)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

_SPAM = re.compile(
    r"白癜风|癫痫病?医院|牛皮癣医院|男科医院|妇科医院|不孕不育医院|代孕(公司|中介|价格)|六合彩|时时彩|娱乐城|真人娱乐|返水|亚博体育|yabo|bet365|"
    r"代开发票|发票代开|贷款秒批|成人电影|一夜情|在线咨询|免费咨询|咨询热线|加微信|加QQ|QQ[:：]\s?\d{5,}|哪家(医院)?好|多少钱一[次个]",
    re.I,
)

_SENTENCE_END = re.compile(r"[。！？!?；;\n]|(?<=\.)\s")
_NOT_WORD = re.compile(r"[\W_]+")

MIN_CHARS = {"zh": 200, "classical": 30, "en": 300, "math": 200, "code": 100}
MAX_CHARS = 200_000
MIN_QUALITY = 0.3

_converter = None


_MOJIBAKE = re.compile(r"(?:[\u00c2-\u00f4][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc\u2013-\u203a\u20ac\u2122]{1,3})+")


def _demojibake(match: re.Match) -> str:
    try:  # 逐字还原成当初的字节：cp1252 里没有定义的几个位置按 latin-1 算
        raw = b"".join(c.encode("latin-1") if ord(c) < 256 else c.encode("cp1252") for c in match.group())
        return raw.decode("utf-8")
    except UnicodeError:
        return match.group()


def fix_mojibake(text: str) -> str:
    """把 UTF-8 被错当成西欧编码读出来的乱码（â€“、Ã© 这类）还原；还原不了的原样留着。"""
    return _MOJIBAKE.sub(_demojibake, text) if ("â" in text or "Ã" in text or "Â" in text) else text


def normalize(text: str, keep_indent: bool = False) -> str:
    """去掉看不见的字符，统一换行和空白，全角字母数字转半角。keep_indent 用于代码：不动行内空格。"""
    text = fix_mojibake(_CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))).translate(_FULLWIDTH)
    if keep_indent:
        return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def to_simplified(text: str) -> str:
    """繁体转简体（OpenCC）。"""
    global _converter
    if _converter is None:
        import opencc

        _converter = opencc.OpenCC("t2s")
    return _converter.convert(text)


def strip_inline_noise(text: str) -> str:
    """去掉混在正文里面的杂质：HTML 实体和标签、网址、加粗记号、版权转载声明和引导关注的句子、句尾的「来源：某某」。"""
    if "&" in text:
        text = _ENTITY.sub(lambda m: html.unescape(m.group(0)), text)
    text = _BOLD.sub(r"\1", _URL.sub("", _TAG.sub("", text)))
    parts = _SENTENCES.findall(text)
    out, changed = [], False
    for s in parts:
        if len(s) <= 150 and _NOISE_SENTENCE.search(s):
            changed = True
            if s.endswith("\n") and out and not out[-1].endswith("\n"):
                out.append("\n")  # 删掉的句子在行尾：换行留着，免得上下两段粘在一起
        else:
            out.append(s)
    text = "".join(out) if changed else text
    return _TAIL_SOURCE.sub("", text)


# ---- 第五轮：图表指引、开头署名、标题里的站点名、文末参考文献、书目、问答站客套话
_FIGURE_REF = re.compile(
    r"[（(]\s*(如|见|参见|详见)?(下|上|右|左)?[图表]\s?\d*(?:[-.—]\d+)?\s?(所示)?\s*[）)]|"      # （如图1）（见表2-3）（图1）
    r"[，,]?\s*(如|见|参见|详见)(下|上|右|左)?[图表]\s?\d*(?:[-.—]\d+)?\s?(所示)?(?=[，。；,;：:\s]|$)|"  # ，如图20-5所示
    r"(下|上)[图表]所示的?"
)
# 「其基本教学流程如图2所示。」这种整句话就是在指图的，删掉指引后只剩半句，所以整句删
_FIGURE_SENTENCE = re.compile(r"(?:(?<=[。！？；\n])|^)[^。！？；\n，,]{0,40}(如|见)(下|上)?[图表]\s?\d*(?:[-.—]\d+)?\s?所示(?:[。；]|(?=\n)|$)", re.M)
_LEADING_COMMA = re.compile(r"(^|(?<=[。！？；\n]))\s*[，,、]\s*", re.M)
_CITATION_LIKE = re.compile(r"(19|20)\d{2}|\[[A-Z/]{1,5}\]|出版社|学报|et al|\d+\s?[（(]\d+[）)]|[A-Z][a-z]+,\s?[A-Z]\.|《[^》]+》")
_CAPTION_LINE = re.compile(r"^[图表]\s?\d+(?:[-.—]\d+)?[\s:：.、]\s*\S.{0,48}$")
_BYLINE = re.compile(r"^(作者|文|图|文图|来源|编辑|排版|供稿|责编|审核|校对|整理|翻译|编译|出品|记者|通讯员)\s*[|｜丨/:：].{0,60}$")
_TITLE_SITE = re.compile(r"^(.{8,}?)\s*[|｜_]\s*([^|｜_，。！？\n]{2,15})$")
_REF_HEADER = re.compile(r"^(主要)?(参考文献|参考资料|参考书目|引用文献|注释|References|REFERENCES)[:：]?$")
_REF_ENTRY = re.compile(r"\[(J|M|D|C|N|R|P|S|Z|EB/OL|J/OL|M/OL)\]")
_TOC_HEADER = re.compile(r"^(图书|作品|本书|全书)?目\s?录$")
_QA_GREETING = re.compile(
    r"您好[,，！!]?\s*很高兴为[您你](回答|解答)(这个)?(问题)?[,，。！!]?(初步分析如下[,，:：]?)?|希望(我的回答)?(对|能)(您|你)?有所?帮助[。！!]?|"
    r"点击(上方|下方)?蓝字(关注)?(我们)?|^(医生回答|病情分析|指导意见|专家解答)[:：]\s*", re.M
)


def strip_structure_noise(text: str) -> str:
    """去掉依附在文章上的结构性杂质：指向不存在的图表的话、图表标题行、开头的署名行、标题里的站点名、
    文末的参考文献和书目、问答站的客套话。"""
    text = _QA_GREETING.sub("", _FIGURE_REF.sub("", _FIGURE_SENTENCE.sub("", text)))
    text = _LEADING_COMMA.sub("", text)  # 「如图5所示，指令寄存器由……」删掉指引后别留下开头的逗号
    text = re.sub(r"([:：])\s*[，,]\s*", r"\1", text)  # 「范例一：如图4所示，此命令……」→「范例一：此命令……」
    text = re.sub(r"[:：]\s*([。；])", r"\1", text)  # 「证明：如图3。」→「证明。」
    lines = text.split("\n")
    if lines:  # 标题行末尾的「 | 某某网」
        m = _TITLE_SITE.match(lines[0].strip())
        if m and len(lines) > 1:
            lines[0] = m.group(1)
    body_start = next((i for i, l in enumerate(lines) if len(l.strip()) > 80), len(lines))
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        if _CAPTION_LINE.match(s) or (i <= body_start and i < 6 and _BYLINE.match(s)):
            continue
        if _REF_ENTRY.search(s) and len(s) < 200:
            continue  # 「王平. 书画同源浅考[J]. 艺术科技, 2015(6).」这样的文献条目
        out.append(line)
    # 文末的参考文献、书目：从标题行起整段切掉。只在文章后半部分、并且后面基本都是短行时才切
    total = sum(len(l) for l in out) or 1
    seen = 0
    for i, line in enumerate(out):
        s = line.strip()
        if seen / total > 0.4 and (_REF_HEADER.match(s) or _TOC_HEADER.match(s)):
            rest = [l.strip() for l in out[i + 1:] if l.strip()]
            if _REF_HEADER.match(s):  # 后面得真的像文献条目才切；图书页面里「参考文献」可能只是目录里的一项，后面还有正文
                ok = not rest or sum(bool(_CITATION_LIKE.search(l)) for l in rest) >= 0.5 * len(rest)
            else:
                ok = not rest or sum(len(l) < 40 for l in rest) >= 0.7 * len(rest)
            if ok:
                out = out[:i]
                break
        seen += len(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(out)).strip()


def strip_boilerplate(text: str) -> str:
    """去掉网页外壳留下的短行（来源、责任编辑、上一篇……）、残留标记和维基百科的空括号。"""
    lines = []
    for line in _INLINE_JUNK.sub("", text).split("\n"):
        stripped = line.strip()
        if stripped and (_PUNCT_ONLY.match(stripped) or (len(stripped) <= _BOILER_ANY_MAX and _BOILER_ANY.match(stripped))
                         or (len(stripped) <= _BOILER_MAX and _BOILER_LINE.match(stripped))):
            continue
        lines.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def fix_punctuation(text: str) -> str:
    """中文文本里的标点统一成中文标点：半角形式的 ｡､，以及夹在两个汉字之间的英文逗号句号等。数字、英文里的不动。"""
    return _ASCII_PUNCT_IN_HAN.sub(lambda m: _TO_FULLWIDTH[m.group(1)], text.translate(_HALFWIDTH_CJK))


_SENT_END = "。！？；：…”’）》」.!?;:"


def unwrap_lines(text: str) -> str:
    """把 PDF 之类留下的硬换行接回去：一段话被切成许多差不多长的行，行尾不是句末标点。

    只在整篇文档明显是这种情况时才动（8 行以上，六成以上的行这样断开），正常分段的文章原样返回。
    """
    lines = text.split("\n")
    body = [l for l in lines if l.strip()]
    if len(body) < 8:
        return text
    lengths = sorted(len(l) for l in body)
    typical = lengths[int(len(lengths) * 0.75)]  # 满行大概多长
    if not 15 <= typical <= 80:
        return text
    broken = lambda l: len(l) >= typical * 0.7 and l.rstrip()[-1] not in _SENT_END
    if sum(broken(l) for l in body) < 0.6 * len(body):
        return text
    out = []
    for line in lines:
        if out and out[-1].strip() and line.strip() and broken(out[-1]):
            glue = "" if HAN.match(out[-1].rstrip()[-1]) or HAN.match(line.lstrip()[0]) else " "
            out[-1] = out[-1].rstrip() + glue + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)


_WRONG_ZHE = re.compile(r"(显|卓|昭|土|原|名|专|论|巨|遗|拙)着|着(名|称|作|述|书立说)")  # 「编着、合着、译着」可能是动词加着，不碰
_WRONG_MENG = re.compile(r"勐(烈|增|然|涨|跌|进|攻|兽|将|力|地|击|虎|药)|(迅|凶|勇|威|生)勐")


def fix_common_typos(text: str) -> str:
    """改掉上游处理留下的两类成批错字：「显着、着名、专着」里的 着 应为 著；「勐烈、迅勐」里的 勐 应为 猛。"""
    text = _WRONG_ZHE.sub(lambda m: m.group(0).replace("着", "著"), text)
    return _WRONG_MENG.sub(lambda m: m.group(0).replace("勐", "猛"), text)


def mask_pii(text: str) -> str:
    """手机号、身份证号、邮箱换成占位符。"""
    text = _EMAIL.sub("<邮箱>", text)
    text = _ID_CARD.sub("<身份证号>", text)
    return _MOBILE.sub("<手机号>", text)


def strip_markdown(text: str) -> str:
    """去掉 Markdown 记号，留下文字（合成语料用；代码和数学不要调用）。"""
    out = []
    for line in text.split("\n"):
        if re.match(r"^\s*```", line) or re.match(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$", line):
            continue  # 代码围栏、表格分隔线
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        out.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(out)).strip()


def est_tokens(text: str) -> int:
    """估算 token 数：汉字约 1.5 个字一个 token，其余约 4 个字符一个 token。"""
    han = len(HAN.findall(text))
    return round(han / 1.5 + (len(text) - han) / 4)


def signatures(text: str) -> frozenset:
    """去重指纹：最长的 3 个句子各取一个哈希。转载、加了头尾的近似重复会共享其中至少两个。

    长句不足 3 个时（代码、短诗）退回到整篇一个哈希；太短的文本没有指纹。
    """
    def h(s):
        return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")

    sentences = {s for s in (_NOT_WORD.sub("", part) for part in _SENTENCE_END.split(text)) if len(s) >= 20}
    if len(sentences) >= 3:
        return frozenset(h(s) for s in sorted(sentences, key=lambda s: (-len(s), s))[:3])
    whole = _NOT_WORD.sub("", text)
    return frozenset([h(whole)]) if len(whole) >= 20 else frozenset()


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


_LENGTH_RANGE = {"zh": (200, 3000), "classical": (30, 1000), "en": (300, 6000), "math": (200, 4000), "code": (100, 2000)}
_SHORT_LINE = {"zh": 12, "classical": 4, "en": 25, "math": 12, "code": 0}


def quality_parts(text: str, kind: str) -> dict:
    """质量分的各个组成部分，都在 0 到 1 之间。

    前四项是「及格线」：语言对不对、是否大量重复、像不像成句的文字、广告词多不多。
    后四项是「好坏程度」：长度是否充分、用字是否丰富、是不是列表目录体、数字符号是否过多。
    """
    dense = re.sub(r"\s", "", text)
    if not dense:
        return dict.fromkeys(("语言", "重复", "成句", "广告", "长度", "丰富", "结构", "密度"), 0.0)
    han_ratio = len(HAN.findall(dense)) / len(dense)
    latin_ratio = len(_LATIN.findall(dense)) / len(dense)
    raw = text.encode("utf-8")[:200_000]
    compress = len(zlib.compress(raw, 6)) / len(raw)  # 越小越重复
    lines = [line for line in text.split("\n") if line.strip()]
    hits = len(_SPAM.findall(text))
    spam = _clip(1.0 - 0.3 * max(0.0, hits * 1000 / max(len(text), 300) - 1.0)) if hits >= 3 else 1.0  # 每千字超过 1 处才开始扣
    if kind in ("zh", "classical"):
        language = _clip((han_ratio - 0.15) / 0.2)
        repetition = _clip((compress - 0.12) / 0.15)
        sentences = _clip(len(_ZH_PUNCT.findall(text)) / len(dense) / 0.02)  # 古文也要有标点：没断句的白文、目录页不要
        head = HAN.findall(text[:1500])
        diversity = _clip((len(set(head)) / max(1, len(head)) - 0.2) / 0.2) if len(head) >= 200 else 1.0
    elif kind == "code":
        language = _clip((latin_ratio - 0.1) / 0.2)
        repetition = _clip((compress - 0.08) / 0.12)
        sentences = 0.0 if max(map(len, lines)) > 1000 or len(text) / len(lines) > 200 else 1.0
        spam, diversity = 1.0, 1.0
    else:  # en、math
        language = _clip((latin_ratio - (0.2 if kind == "math" else 0.4)) / 0.2)
        repetition = _clip((compress - 0.15) / 0.2)
        sentences = _clip(len(re.findall(r"[.,;:?!]", text)) / len(dense) / 0.01)
        words = re.findall(r"[A-Za-z]+", text[:3000].lower())
        diversity = _clip((len(set(words)) / max(1, len(words)) - 0.3) / 0.25) if len(words) >= 80 else 1.0
    low, high = _LENGTH_RANGE[kind]
    length = _clip(math.log(max(len(text), low) / low) / math.log(high / low))
    short = sum(len(line.strip()) < _SHORT_LINE[kind] for line in lines) / len(lines) if len(lines) >= 8 else 0.0
    structure = _clip(1.0 - max(0.0, short - 0.3) / 0.5)
    symbols = len(re.findall(r"[0-9%|/\\=+*#_~^<>\[\]{}]", dense)) / len(dense)
    density = 1.0 if kind in ("code", "math") else _clip(1.0 - (symbols - 0.15) / 0.3)
    return {"语言": language, "重复": repetition, "成句": sentences, "广告": spam,
            "长度": length, "丰富": diversity, "结构": structure, "密度": density}


def quality(text: str, kind: str) -> float:
    """0 到 1 的质量分。四条及格线里最差的一项封顶，再按好坏程度在 0.5 到 1 之间细分。"""
    p = quality_parts(text, kind)
    gate = min(p["语言"], p["重复"], p["成句"], p["广告"])
    grade = (p["长度"] + p["丰富"] + p["结构"] + p["密度"]) / 4
    return gate * (0.5 + 0.5 * grade)


def clean_with_reason(raw: str, kind: str, native_score: float | None = None, markdown: bool = False) -> tuple[dict | None, str]:
    """精洗一篇文档，返回 (结果, 原因)。合格时原因是空字符串；不合格时结果是 None，原因说明为什么丢。"""
    if kind not in MIN_CHARS:
        raise ValueError(f"未知的种类 {kind!r}，可选：{', '.join(MIN_CHARS)}")
    if not raw or not raw.strip():
        return None, "空"
    if len(raw) > MAX_CHARS * 2:
        return None, "太长"
    text = normalize(raw, keep_indent=(kind == "code"))
    if kind in ("zh", "classical"):
        text = fix_common_typos(strip_structure_noise(strip_boilerplate(strip_inline_noise(fix_punctuation(to_simplified(text))))))
        if len(_CENSORED.findall(text)) >= 2:
            return None, "有屏蔽字"
        if text.endswith("..."):
            return None, "被截断"
    if markdown:
        text = strip_markdown(text)
    if kind in ("zh", "en"):
        text = unwrap_lines(text)
    if kind != "code":
        text = mask_pii(text)
    if len(text) < MIN_CHARS[kind]:
        return None, "太短"
    if len(text) > MAX_CHARS:
        return None, "太长"
    parts = quality_parts(text, kind)
    heuristic = quality(text, kind)
    if heuristic < MIN_QUALITY:
        worst = min(("语言", "重复", "成句", "广告"), key=parts.get)
        return None, {"语言": "语言不对", "重复": "重复太多", "成句": "不成句", "广告": "广告"}[worst]
    return {
        "text": text,
        "n_chars": len(text),
        "est_tokens": est_tokens(text),
        "quality": float(native_score) if native_score is not None else heuristic,
        "heuristic": heuristic,
        "sigs": sorted(signatures(text)),
    }, ""


def clean_document(raw: str, kind: str, native_score: float | None = None, markdown: bool = False) -> dict | None:
    """精洗一篇文档。不合格返回 None；合格返回 {text, n_chars, est_tokens, quality, heuristic, sigs}。

    quality 用来排序挑子集：来源自带分数（native_score）时用它，否则用这里算的 heuristic。
    不管有没有自带分数，heuristic 低于 MIN_QUALITY 的都丢掉。
    """
    return clean_with_reason(raw, kind, native_score, markdown)[0]
