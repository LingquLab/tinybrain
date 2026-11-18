"""RL 和在线蒸馏用的采样：一条提示采一组回答；带工具的多轮对话；以及把这些拼成训练用的一批。"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from inference.generate import generate
from inference.template import ChatFormat
from inference.tokenizer import PAD
from lm import LanguageModel


class PlainFormat:
    """Document-style prompts for RL directly on a pretrained base model."""
    def __init__(self, tok):
        self.tok, self.end = tok, tok.eos_id

    def prompt(self, messages):
        if len(messages) != 1 or messages[0]['role'] != 'user':
            raise ValueError('Plain base-model RL requires exactly one user prompt')
        return [self.end] + self.tok.encode(messages[0]['content'])

    def answer(self, ids):
        return self.tok.decode(cut_at(list(ids), self.end)[0]).strip()


@dataclass
class Episode:
    """一次采样的结果。ids 是提示加上后面的全部内容；learn 标出哪些是模型自己写的（要训练的）。"""
    ids: list[int]
    learn: list[int]
    prompt_len: int
    finished: bool                       # 是否在长度上限之前自己写完了
    turns: list[dict] = field(default_factory=list)   # 多轮时每一轮：正文、工具调用、工具返回
    text: str = ""                       # 最后一轮回答的正文


def fits(model: LanguageModel, fmt: ChatFormat, messages: list[dict], max_new: int) -> bool:
    """提示加上要生成的长度放得进模型的上下文。"""
    return len(fmt.prompt(messages)) + max_new <= model.cfg.max_seq_len


def keep_fitting(model, fmt, rows: list[dict], max_new: int, what: str) -> list[dict]:
    """去掉放不进上下文的题，报一下去掉了多少；一道都不剩就报错。"""
    kept = [r for r in rows if fits(model, fmt, r["messages"], max_new)]
    if len(kept) < len(rows):
        print(f"{what}：{len(rows) - len(kept)} / {len(rows)} 道的提示加上 {max_new} 个生成的词超过了上下文 {model.cfg.max_seq_len}，跳过", flush=True)
    if not kept:
        raise ValueError(f"{what}没有一道放得进上下文 {model.cfg.max_seq_len}")
    return kept


def balanced_tasks(rows,limit):
    """Round-robin categories so a prefix-limited evaluation covers every category."""
    pools={}
    for row in rows:pools.setdefault(row.get('category') or row.get('domain') or row['reward'],[]).append(row)
    selected=[]
    for offset in range(max(map(len,pools.values()),default=0)):
        for pool in pools.values():
            if offset<len(pool):
                selected.append(pool[offset])
                if len(selected)==limit:return selected
    return selected


def cut_at(ids: list[int], stop: int) -> tuple[list[int], bool]:
    """截到第一个停止符（包括它）为止；没有停止符就是没写完。"""
    return (ids[: ids.index(stop) + 1], True) if stop in ids else (ids, False)


def sample_group(model: LanguageModel, fmt: ChatFormat, messages: list[dict], n: int, max_new: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None, fast: bool | None = None) -> list[Episode]:
    """同一条提示采 n 个回答（同一条提示长度一样，可以整批一起生成）。"""
    device = next(model.parameters()).device
    prompt = fmt.prompt(messages)
    idx = torch.tensor([prompt] * n, device=device)
    out = generate(model, idx, max_new, temperature=temperature, top_k=top_k, eos_id=fmt.end, generator=generator, fast=fast)
    episodes = []
    for row in out[:, len(prompt):].tolist():
        resp, finished = cut_at(row, fmt.end)
        episodes.append(Episode(prompt + resp, [0] * len(prompt) + [1] * len(resp), len(prompt), finished, text=fmt.answer(resp)))
    return episodes


def run_agent(model: LanguageModel, fmt: ChatFormat, messages: list[dict], env, max_new: int, max_turns: int = 4,
              temperature: float = 1.0, top_k: int | None = None, generator: torch.Generator | None = None) -> Episode:
    """带工具的多轮：模型写一轮；里面有工具调用就交给 env 执行，把结果作为 tool 一轮接回去，再让模型往下写。
    没有工具调用、写满轮数、或者超出长度时结束。env.call(name, arguments) 返回工具结果的文字。"""
    device = next(model.parameters()).device
    ids = fmt.prompt(messages)
    learn = [0] * len(ids)
    prompt_len, turns, finished = len(ids), [], False
    for _ in range(max_turns):
        room = min(max_new, model.cfg.max_seq_len - len(ids))
        if room <= 0:
            break
        out = generate(model, torch.tensor([ids], device=device), room, temperature=temperature, top_k=top_k, eos_id=fmt.end, generator=generator)
        resp, done = cut_at(out[0, len(ids):].tolist(), fmt.end)
        ids, learn = ids + resp, learn + [1] * len(resp)
        text, calls, broken = fmt.parse_turn(resp,allow_plain_calls=True)
        turn = dict(text=text, calls=calls, broken=broken, results=[])
        turns.append(turn)
        if not done:
            break
        if not calls and not broken:
            finished = True
            break
        results = [env.call(c["name"], c["arguments"]) for c in calls] + [f"工具调用格式错误：{b[:200]}" for b in broken]
        turn["results"] = results
        back = fmt.tool_turn("\n".join(results))
        if len(ids) + len(back) >= model.cfg.max_seq_len:
            break
        ids, learn = ids + back, learn + [0] * len(back)
    return Episode(ids, learn, prompt_len, finished, turns, turns[-1]["text"] if turns else "")


def collate(episodes: list[Episode], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """拼成右边补齐的一批：(输入, 目标, 目标位置是否要训练)，形状都是 (B, 最长 - 1)。"""
    longest = max(len(e.ids) for e in episodes)
    ids = torch.full((len(episodes), longest), PAD, dtype=torch.long)
    learn = torch.zeros((len(episodes), longest), dtype=torch.bool)
    for i, e in enumerate(episodes):
        ids[i, : len(e.ids)] = torch.tensor(e.ids)
        learn[i, : len(e.learn)] = torch.tensor(e.learn, dtype=torch.bool)
    return ids[:, :-1].to(device), ids[:, 1:].to(device), learn[:, 1:].to(device)


def token_logprobs(model: LanguageModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """每个位置上，模型给目标词的对数概率，(B, T)。"""
    logits = model(x).logits
    from lm.loss import token_log_probs
    return token_log_probs(logits, y)
