"""检查评审模型和评分细则靠不靠谱：拿语料里的好回答，和故意弄坏的版本一起给评审打分，看它能不能把好的排在前面。

    tinybrain judge-check --judge-url http://127.0.0.1:8080/v1 --judge-model out/models/Qwen3.8-27B-4bit --judge-workers 2 --judge-timeout 300

弄坏的方式：截断（只留前四分之一）、答非所问（换成别的题的回答）、车轱辘话（第一句重复到原来的长度）；
诗词另加一种：每行里的字打乱（体裁、字数不变，意思没了，专门看评审是不是只看格式）。
报告：每个领域、每种版本的平均分，好回答赢过每种坏版本的比例（打平算半场），解析失败的次数。
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq

from training.rewards.judge import Judge

ROOT = Path(__file__).resolve().parents[1]
GENERAL_SOURCES = ("chinese_instruct", "coig_cqia", "smoltalk_zh", "deepctrl")
_SENT = re.compile(r"(?<=[。！？!?\n])")


def _cjk_ratio(text: str) -> float:
    return sum("一" <= c <= "鿿" for c in text) / max(1, len(text))


def literature_domain(question: str) -> str:
    if "写一首" in question or "绝句" in question or "律诗" in question:
        return "诗词"
    if any(k in question for k in ("翻译", "白话", "文言", "古文")):
        return "古文"
    return "写作"


def sample_pairs(per_domain: int, seed: int = 0, sft: Path = ROOT / "data" / "sft_small.parquet",
                 literature: Path = ROOT / "data" / "sft_expert_literature.parquet") -> dict:
    """每个领域取 per_domain 个单轮的（问题、好回答）。通用题只取中文来源、中文为主、长度适中的。"""
    rng = random.Random(seed)
    pools = {"通用": [], "写作": [], "古文": [], "诗词": []}
    for path, general in ((sft, True), (literature, False)):
        for row in pq.read_table(path, columns=["messages", "source"]).to_pylist():
            msgs = [m for m in row["messages"] if m["role"] != "system"]
            if len(msgs) != 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant" or msgs[1].get("tool_calls"):
                continue
            q, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
            if general:
                if row["source"] in GENERAL_SOURCES and 150 <= len(a) <= 1200 and _cjk_ratio(a) > 0.5 and len(q) <= 300:
                    pools["通用"].append((q, a))
            else:
                pools[literature_domain(q)].append((q, a))
    return {d: rng.sample(p, min(per_domain, len(p))) for d, p in pools.items() if p}


def truncate(answer: str) -> str:
    return answer[: max(8, len(answer) // 4)]


def repeat(answer: str) -> str:
    first = next((s for s in _SENT.split(answer) if s.strip()), answer)
    return (first * (len(answer) // max(1, len(first)) + 1))[: len(answer)]


def shuffle_lines(poem: str, rng: random.Random) -> str:
    """每行里的汉字打乱，标点留在原位。"""
    out = []
    for line in poem.split("\n"):
        chars = [c for c in line if "一" <= c <= "鿿"]
        rng.shuffle(chars)
        it = iter(chars)
        out.append("".join(next(it) if "一" <= c <= "鿿" else c for c in line))
    return "\n".join(out)


def variants(pairs: list[tuple[str, str]], domain: str, rng: random.Random) -> list[dict]:
    """每道题一组：好回答加几种坏版本。"""
    rows = []
    for i, (q, a) in enumerate(pairs):
        other = pairs[(i + 1) % len(pairs)][1]
        group = {"好": a, "截断": truncate(a), "答非所问": other, "车轱辘话": repeat(a)}
        if domain == "诗词":
            group = {"好": a, "字序打乱": shuffle_lines(a, rng), "答非所问": other}
        rows += [dict(domain=domain, item=i, variant=v, question=q, answer=text) for v, text in group.items()
                 if v == "好" or text != a]  # 只有一句的回答，车轱辘话就是原文，不算坏版本
    return rows


def summarize(rows: list[dict]) -> dict:
    report = {}
    for domain in dict.fromkeys(r["domain"] for r in rows):
        rs = [r for r in rows if r["domain"] == domain]
        names = list(dict.fromkeys(r["variant"] for r in rs))
        mean = {v: round(sum(r["score"] for r in rs if r["variant"] == v and r["score"] is not None)
                         / max(1, sum(r["variant"] == v and r["score"] is not None for r in rs)), 2) for v in names}
        good = {r["item"]: r["score"] for r in rs if r["variant"] == "好"}
        wins = {}
        for v in names[1:]:
            pairs = [(good[r["item"]], r["score"]) for r in rs if r["variant"] == v and good.get(r["item"]) is not None and r["score"] is not None]
            wins[v] = round(sum(1.0 if g > b else 0.5 if g == b else 0.0 for g, b in pairs) / max(1, len(pairs)), 3)
        report[domain] = dict(n=len(good), mean=mean, good_wins=wins, failures=sum(r["score"] is None for r in rs))
    return report


def run(judge: Judge, per_domain: int = 12, seed: int = 0, out: Path | None = None, **paths) -> dict:
    rng = random.Random(seed)
    rows = [r for d, pairs in sample_pairs(per_domain, seed, **paths).items() for r in variants(pairs, d, rng)]
    t0 = time.perf_counter()
    scores = judge.score([(r["question"], r["answer"], r["domain"], None) for r in rows])
    for r, s in zip(rows, scores):
        r["score"] = s
    report = dict(summary=summarize(rows), judgments=len(rows), seconds=round(time.perf_counter() - t0, 1))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(**report, rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="用好回答和弄坏的回答检查评审模型")
    parser.add_argument("--judge-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--judge-workers", type=int, default=16, help="同时发几个请求；mlx_lm.server 一次只算一个，用 1–2")
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--per-domain", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "out" / "judge_check.json"))
    args = parser.parse_args(argv)
    report = run(Judge(args.judge_url, args.judge_model, timeout=args.judge_timeout, workers=args.judge_workers), args.per_domain, args.seed, Path(args.out))
    for domain, r in report["summary"].items():
        means = "  ".join(f"{v} {m}" for v, m in r["mean"].items())
        wins = "  ".join(f"{v} {w:.0%}" for v, w in r["good_wins"].items())
        print(f"{domain}（{r['n']} 题）平均分：{means}\n    好回答胜过：{wins}  解析失败 {r['failures']}")
    print(f"共 {report['judgments']} 次评审，{report['seconds']}s；明细在 {args.out}")
    return report


if __name__ == "__main__":
    main()
