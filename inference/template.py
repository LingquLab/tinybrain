"""对话格式：把一段多轮对话编成词编号，同时给出每个位置算不算损失；生成时拼提示、解析模型写出的回答和工具调用。

格式（<|im_start|>、<|im_end|> 是分词器里预留的控制符，编号固定，正文里写出同样的字符串也不会变成控制符）：

    <|im_start|>system
    你是……<|im_end|>
    <|im_start|>user
    床前明月光的下一句是什么？<|im_end|>
    <|im_start|>assistant
    疑是地上霜。<|im_end|>

工具调用：assistant 的消息里可以带 tool_calls（[{"name": …, "arguments": {…}}]），写成
    <|tool_call|>{"name": …, "arguments": {…}}<|/tool_call|>
接在回答的正文后面（这两个控制符是分词器里预留的 <|reserved_0|>、<|reserved_1|>）。工具返回的结果作为 tool 角色的一轮放回对话，不算损失。

算不算损失（train_on）：
    assistant  只学 assistant 的回答和它后面的 <|im_end|>（SFT 用）
    all        整段都学（退火时把对话混进预训练数据用）
"""
import json

from .tokenizer import Tokenizer

ROLES = ("system", "user", "assistant", "tool")
START, END = "<|im_start|>", "<|im_end|>"
CALL, CALL_END = "<|reserved_0|>", "<|reserved_1|>"   # 工具调用的开头和结尾


class ChatFormat:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.start, self.end = tok.control_id(START), tok.control_id(END)
        self.call, self.call_end = tok.control_id(CALL), tok.control_id(CALL_END)
        self.newline = tok.encode("\n")
        self._heads = {role: [self.start] + tok.encode(f"{role}\n") for role in ROLES}

    def encode(self, messages: list[dict], train_on: str = "assistant") -> tuple[list[int], list[int]]:
        """返回 (词编号, 损失遮罩)，两者一样长；遮罩为 1 的位置是要预测的词。"""
        if train_on not in ("assistant", "all"):
            raise ValueError(f"train_on 只能是 assistant 或 all，收到的是 {train_on!r}")
        if not messages:
            raise ValueError("对话是空的")
        ids, mask = [], []
        everything = train_on == "all"
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role not in ROLES:
                raise ValueError(f"不认识的角色 {role!r}，只能是 {'、'.join(ROLES)}")
            if not isinstance(content, str):
                raise ValueError(f"{role} 的内容不是字符串")
            head, body = self._heads[role], self.tok.encode(content)
            calls = m.get("tool_calls") or []
            if isinstance(calls, str):  # parquet 里存成 JSON 文字
                calls = json.loads(calls) if calls.strip() else []
            for call in calls:
                if role != "assistant":
                    raise ValueError("只有 assistant 能调用工具")
                body += [self.call] + self.tok.encode(json.dumps(call, ensure_ascii=False)) + [self.call_end]
            body += [self.end]
            learn = everything or role == "assistant"
            ids += head + body + self.newline
            mask += [int(everything)] * len(head) + [int(learn)] * len(body) + [int(everything)] * len(self.newline)
        if not everything and not any(mask):
            raise ValueError("对话里没有 assistant 的回答，SFT 学不到东西")
        return ids, mask

    def prompt(self, messages: list[dict]) -> list[int]:
        """生成时的输入：前面的对话，再接上 assistant 的开头，让模型往下写。"""
        ids = self.encode(messages, train_on="all")[0] if messages else []
        return ids + self._heads["assistant"]

    def parse_turn(self, ids: list[int]) -> tuple[str, list[dict], list[str]]:
        """把模型生成的一轮 assistant（到 <|im_end|> 为止）拆成 (正文, 工具调用, 解析不了的调用的原文)。"""
        ids = list(ids)
        if self.end in ids:
            ids = ids[: ids.index(self.end)]
        text, calls, broken, i = [], [], [], 0
        while i < len(ids):
            if ids[i] == self.call:
                j = ids.index(self.call_end, i) if self.call_end in ids[i:] else len(ids)
                raw = self.tok.decode(ids[i + 1: j])
                try:
                    call = json.loads(raw)
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                        raise ValueError
                    calls.append({"name": call["name"], "arguments": call.get("arguments") or {}})
                except ValueError:
                    broken.append(raw)
                i = j + 1
            else:
                text.append(ids[i])
                i += 1
        return self.tok.decode(text), calls, broken

    def tool_turn(self, content: str) -> list[int]:
        """工具返回结果的那一轮，接着是下一轮 assistant 的开头。"""
        return self._heads["tool"] + self.tok.encode(content) + [self.end] + self.newline + self._heads["assistant"]

    def answer(self, ids: list[int]) -> str:
        """把模型生成的词编号变回回答的文字：到 <|im_end|> 为止。"""
        ids = list(ids)
        if self.end in ids:
            ids = ids[: ids.index(self.end)]
        return self.tok.decode(ids)
