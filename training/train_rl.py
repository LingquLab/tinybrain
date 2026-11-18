"""RL：GRPO（带 Dr. GRPO 的修正），奖励可插拔——RLVR、agent RL、RLAIF 都用这一个脚本，区别只在题目文件。

    python -m training.train_rl --init out/small/sft/ckpt.pt --tasks data/rl_small.parquet --out out/small/rl --steps 200 \\
        --judge-url http://localhost:8000/v1

每一步：
  1. 抽 prompts_per_step 道题，每道采 group_size 个回答（agent 题按多轮跑工具）；
  2. 按每道题的 reward 字段打分（math / code / poetry / judge / agent，见 training/rewards）；
  3. 组内减去平均奖励得到优势值（Dr. GRPO：不除以标准差）；
  4. 损失 = −Σ 优势 × 模型自己写的词的对数概率 / (回答数 × max_new)，分母是常数（Dr. GRPO：不按回答长度归一化，免得偏爱长回答）。
     完全在线：每批回答只用来更新一次，所以不需要重要性采样比和截断。

题目数据（parquet，一行一道题）：
    messages   提示，[{role, content}]，不含要模型写的回答
    reward     math / code / poetry / judge / agent / choice / reading / cloze / instruct / logic
    answer     math、agent、choice、cloze 用的标准答案；reading 是参考答案的 JSON 列表
    tests      code 用的测试（Python assert）
    form       poetry 的体裁（五绝、七绝、五律、七律）
    domain     judge、poetry 用的评分领域（通用、写作、古文、诗词）
    reference  judge 可选的参考答案（比如鲁迅续写的原文）
    tools、documents   agent 题能用的工具和检索资料
    constraints  instruct 题的约束（JSON 列表，见 training/rewards/instruct.py）
    gym、entry   logic 题的题型名和生成时的那一条（reasoning-gym）
全部参数见文件末尾的 parse_args。
"""
import argparse
from collections import Counter
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from inference.template import ChatFormat
from inference.tokenizer import DEFAULT_MODEL, Tokenizer

from .rewards import choice as choice_reward
from .rewards import code as code_reward
from .rewards import exact as exact_reward
from .rewards import instruct as instruct_reward
from .rewards import judge as judge_mod
from .rewards import logic as logic_reward
from .rewards import match as match_reward
from .rewards import math as math_reward
from .rewards import poetry as poetry_reward
from .rewards import tools as tool_reward
from .rewards.tools import ToolEnv
from .rollout import Episode, PlainFormat, balanced_tasks, collate, keep_fitting, run_agent, sample_group, token_logprobs
from .trainer_utils import (find_stage, add_optimizer_args, autocast, load_checkpoint, load_model, optimizers_from_args, pick_device, resume,
                            restore_rng, rng_state, save_checkpoint, set_lr_scale, setup_device, step_optimizers)

# 程序判分的题：奖励函数 reward(episode, task) -> 0 到 1
PROGRAM_REWARDS = {"math": math_reward.reward, "math_strict": math_reward.strict_reward, "math_steps": math_reward.process_reward, "code": code_reward.reward, "agent": tool_reward.reward, "choice": choice_reward.reward,
                   "reading": match_reward.reading_reward, "cloze": match_reward.cloze_reward, "instruct": instruct_reward.reward,
                   "logic": logic_reward.reward, "exact": exact_reward.reward}
KINDS = tuple(PROGRAM_REWARDS) + ("poetry", "judge")  # 后两种要评审模型


def load_tasks(path: str,allowed=None) -> list[dict]:
    tasks=[row for filename in path.split(',') for row in pq.read_table(filename).to_pylist()]
    if allowed:tasks=[row for row in tasks if row.get('reward') in allowed]
    for i, t in enumerate(tasks):
        if t.get("reward") not in KINDS:
            raise ValueError(f"第 {i} 道题的 reward 是 {t.get('reward')!r}，只能是 {'、'.join(KINDS)}")
        if not t.get("messages"):
            raise ValueError(f"第 {i} 道题没有 messages")
    return tasks


def last_user(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def score(tasks: list[dict], groups: list[list[Episode]], judge=None, meter_weight: float = 0.4) -> list[list[float]]:
    """给每个回答打分。judge 和 poetry 的评审调用攒在一起并发发出去。"""
    rewards = [[0.0] * len(g) for g in groups]
    pending = []  # (第几题, 第几个回答, 问题, 回答, 领域, 参考)
    for ti, (task, group) in enumerate(zip(tasks, groups)):
        kind = task["reward"]
        for ei, ep in enumerate(group):
            if kind in PROGRAM_REWARDS:
                rewards[ti][ei] = PROGRAM_REWARDS[kind](ep, task)
            else:
                domain = task.get("domain") or ("诗词" if kind == "poetry" else "通用")
                question=last_user(task['messages']) if len(task['messages'])==1 else json.dumps(task['messages'],ensure_ascii=False)
                pending.append((ti, ei, question, ep.text, domain, task.get("reference")))
    if pending:
        if judge is None:
            raise ValueError("有 judge 或 poetry 题，但没有给评审模型（--judge-url）")
        scores = judge.score([p[2:] for p in pending])
        if any(s is None for s in scores):
            raise RuntimeError('Judge response missing/invalid; refusing to turn a service failure into a training reward: '
                               +repr(getattr(judge,'last_errors',[])[-3:]))
        for (ti, ei, *_), s in zip(pending, scores):
            task, ep = tasks[ti], groups[ti][ei]
            if task["reward"] == "judge":
                rewards[ti][ei] = judge_mod.to_reward(s)
            else:  # 诗词：格律分（程序）和评审分加权；格律不过关时评审分最多算一半
                meter = poetry_reward.check(ep.text, task["form"])
                judged = 0.0 if s is None else (s - 1) / 9
                if not meter["shape"]:
                    judged *= 0.5
                rewards[ti][ei] = 2 * (meter_weight * meter["score"] + (1 - meter_weight) * judged) - 1
    return rewards


def advantages(rewards: list[float], normalize: bool = False) -> list[float]:
    """Dr. GRPO：组内减平均，不除以标准差。"""
    mean = sum(rewards) / len(rewards)
    scale = (sum((r - mean)**2 for r in rewards) / len(rewards))**.5 if normalize else 1.
    return [(r - mean) / max(scale, 1e-8) for r in rewards]


def policy_loss(model, episodes: list[Episode], advs: list[float], norm: float, micro_batch: int, device,
                reference=None, kl_beta: float = 0., grpo: bool = False, metrics=None) -> float:
    """对一批回答做反向传播（梯度累积在模型参数上），返回损失的数值。norm 是 Dr. GRPO 的常数分母。"""
    total = kl_total = token_count = 0.0
    for start in range(0, len(episodes), micro_batch):
        chunk, a = episodes[start:start + micro_batch], torch.tensor(advs[start:start + micro_batch], device=device)
        x, y, learn = collate(chunk, device)
        with autocast(device):
            logp = token_logprobs(model, x, y)
            if reference is not None:
                with torch.no_grad():ref = token_logprobs(reference, x, y)
        values = -a[:, None] * logp.float()
        if reference is not None:
            delta = torch.where(learn, ref.float() - logp.float(), 0.)
            kl = torch.expm1(delta) - delta
            values = values + kl_beta * kl
            kl_total += float((kl.detach() * learn).sum());token_count += int(learn.sum())
        per_sequence = (values * learn).sum(1)
        if grpo:per_sequence = per_sequence / learn.sum(1).clamp_min(1)
        loss = per_sequence.sum() / norm
        if not bool(torch.isfinite(loss)):raise RuntimeError('Non-finite RL policy loss; optimizer not updated')
        loss.backward()
        total += loss.item()
    if metrics is not None:metrics['sampled_kl'] = kl_total / max(1, token_count)
    return total


def evaluate_policy(model, fmt, tasks, cfg, out, step, judge=None):
    """Fixed held-out questions and sampling seeds, without updating the actor."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1_000_003) if device.type in ('cpu', 'cuda') else None
    rows = []
    selected=balanced_tasks(tasks,cfg.eval_limit)
    for task in selected:
        with autocast(device):
            if task['reward']=='agent':
                environment=ToolEnv(task)
                messages=[dict(role='system',content=environment.describe())]+[m for m in task['messages'] if m['role']!='system']
                group=[run_agent(model,fmt,messages,ToolEnv(task),cfg.max_new,cfg.max_turns,cfg.temperature,cfg.top_k or None,gen)
                       for _ in range(cfg.eval_samples)]
            else:
                group = sample_group(model, fmt, task['messages'], cfg.eval_samples, cfg.max_new,
                                     cfg.temperature, cfg.top_k or None, gen, fast=False if cfg.eager_rollout else None)
        rewards = score([task], [group], judge, cfg.poetry_meter_weight)[0]
        rows.append(dict(prompt=task['messages'][0]['content'], answer=task.get('answer'), category=task.get('category'), kind=task['reward'],
                         rewards=rewards, responses=[e.text for e in group], finished=[e.finished for e in group]))
        if len(rows)%8==0:print('HELDOUT_PROGRESS',step,len(rows),min(len(tasks),cfg.eval_limit),flush=True)
    parsed=[math_reward.strict_answer(text) for row in rows if row['kind'] in ('math','math_strict') for text in row['responses']]
    counts=Counter(p for p in parsed if p is not None)
    expected=Counter(str(row['answer']) for row in rows)
    report = dict(step=step,questions=len(rows),samples_per_question=cfg.eval_samples,seed=cfg.seed+1_000_003,
                  temperature=cfg.temperature,top_k=cfg.top_k,mean_reward=sum(sum(r['rewards']) for r in rows)/(len(rows)*cfg.eval_samples),
                  any_positive_reward_at_n=sum(max(r['rewards']) > 0 for r in rows)/len(rows),
                  finished_fraction=sum(sum(r['finished']) for r in rows)/(len(rows)*cfg.eval_samples),
                  parsed_fraction=sum(p is not None for p in parsed)/len(parsed) if parsed else None,top_answers=counts.most_common(8),
                  best_constant_answer_reward=max(expected.values())/len(rows),rows=rows)
    report['by_kind'] = {}
    for kind in sorted({r['kind'] for r in rows}):
        selected = [r for r in rows if r['kind'] == kind]
        values = [v for r in selected for v in r['rewards']]
        entry = dict(questions=len(selected), mean_reward=sum(values)/len(values))
        if kind in PROGRAM_REWARDS:
            entry['perfect_reward_fraction'] = sum(v >= 1. - 1e-8 for v in values)/len(values)
            entry['perfect_reward_at_n'] = sum(max(r['rewards']) >= 1. - 1e-8 for r in selected)/len(selected)
        report['by_kind'][kind] = entry
        if kind == 'math_steps':
            from types import SimpleNamespace
            pairs = [(text, task) for task, row in zip(balanced_tasks(tasks, cfg.eval_limit), rows)
                     if task['reward'] == kind for text in row['responses']]
            entry['final_answer_accuracy'] = sum(math_reward.strict_reward(SimpleNamespace(text=t), q) for t, q in pairs) / len(pairs)
            entry['verified_step_fraction'] = sum(math_reward.step_credit(t, q) for t, q in pairs) / len(pairs)
    path = out/('baseline.json' if step == 0 else f'eval-step-{step}.json')
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)
    print('HELDOUT',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    return report


def train(cfg, judge=None) -> dict:
    """跑完 RL，返回最后一步的统计。cfg 是 parse_args 的结果；judge 可以直接传一个 Judge（测试时用假的），不传就按 --judge-url 建。"""
    device = pick_device(cfg.device)
    setup_device(device)
    if cfg.gpu_memory_fraction:
        if device.type!='cuda' or not 0<cfg.gpu_memory_fraction<=1:raise ValueError('Invalid CUDA memory fraction')
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction,device.index)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and ((out/'ckpt.pt').exists() or (out/'log.jsonl').exists()):
        raise ValueError('RL output already contains a run; choose another --out to preserve it')
    if cfg.resume and not (out/'ckpt.pt').exists():
        raise FileNotFoundError('--resume requires an RL checkpoint')
    if Path(cfg.init).resolve().is_relative_to(out.resolve()):
        raise ValueError('RL initialization must be outside its training output directory')
    tasks = load_tasks(cfg.tasks,cfg.reward_kinds)
    if judge is None and cfg.judge_url:
        judge = judge_mod.Judge(cfg.judge_url, cfg.judge_model, timeout=cfg.judge_timeout, workers=cfg.judge_workers)
    model = load_model(Path(cfg.init), device)
    model.gradient_checkpointing = device.type == 'cuda'
    model.checkpoint_policy = 'selective'
    fmt = (PlainFormat if cfg.prompt_format == 'plain' else ChatFormat)(Tokenizer(cfg.tokenizer))
    if cfg.prompt_format == 'plain' and any(t['reward']=='agent' for t in tasks):
        raise ValueError('Agent tasks require the chat template')
    tasks = keep_fitting(model, fmt, tasks, cfg.max_new, "RL 题目")
    sampling_weights=None
    if cfg.balanced_kinds or cfg.balanced_categories:
        labels=[(t.get('category') or t.get('domain') or t['reward'])
                if cfg.balanced_categories else t['reward'] for t in tasks]
        counts=Counter(labels)
        sampling_weights=np.array([1/counts[label] for label in labels],dtype=np.float64)
        sampling_weights/=sampling_weights.sum()
        print('Sampling strata:', dict(counts), flush=True)
    reference = load_model(Path(cfg.init), device).eval().requires_grad_(False) if cfg.kl_beta else None
    if cfg.attention_backend=='cudnn':
        from lm.layers.attention import prepare_cudnn_attention
        prepare_cudnn_attention()
    for network in [model,reference]:
        if network is not None:
            for block in network.blocks:
                block.attn.backend=cfg.attention_backend
                # Match rollout/eval and gradient log-probability arithmetic.
                block.attn.fuse_qkv=False
    if cfg.policy_optimizer == 'adamw':
        optimizers = [torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=0.,fused=device.type=='cuda')]
        for group in optimizers[0].param_groups:group['base_lr']=cfg.learning_rate
    else:optimizers = optimizers_from_args(model, cfg)
    start_step = resume(out / 'ckpt.pt', model, optimizers, device) if cfg.resume else 0
    (out / "config.json").write_text(json.dumps({"rl": vars(cfg),"steps":cfg.steps, "model": json.loads(model.cfg.to_json())}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed) if device.type in ('cpu','cuda') else None
    best_reward, best_step = -float('inf'), 0
    total_learned_groups = 0
    if cfg.resume:
        state = load_checkpoint(out / 'ckpt.pt', torch.device('cpu'), mmap=True)
        previous = state.get('rl_config', {})
        for key in ('tasks', 'init', 'seed', 'prompt_format', 'tokenizer', 'policy_optimizer',
                    'prompts_per_step', 'group_size', 'max_new', 'temperature', 'top_k', 'algorithm', 'kl_beta',
                    'reward_kinds', 'balanced_kinds', 'balanced_categories', 'best_trained_only'):
            default = False if key in ('balanced_kinds', 'balanced_categories', 'best_trained_only') else None
            if previous.get(key, default) != getattr(cfg, key):
                raise ValueError(f'RL resume changes the sampling/training contract: {key}')
        progress = state.get('rl_progress', {})
        rng.bit_generator.state = progress['sample_rng']
        if gen is not None:
            gen.set_state(progress['token_rng'])
        best_reward, best_step = progress.get('best_reward', best_reward), progress.get('best_step', 0)
        total_learned_groups = progress.get('total_learned_groups', 0)
        restore_rng(state.get('rng'))
        del state
    top_k = cfg.top_k or None
    evaluation = keep_fitting(model,fmt,load_tasks(cfg.eval_tasks,cfg.reward_kinds),cfg.max_new,'RL heldout') if cfg.eval_tasks else []
    if evaluation:
        train_prompts={json.dumps(t['messages'],sort_keys=True,ensure_ascii=False) for t in tasks}
        if any(json.dumps(t['messages'],sort_keys=True,ensure_ascii=False) in train_prompts for t in evaluation):
            raise ValueError('Training and evaluation prompts overlap')
        train_calculations={t['calculation'] for t in tasks if t.get('calculation')}
        if any(t.get('calculation') in train_calculations for t in evaluation):
            raise ValueError('Training and evaluation calculations overlap')
        train_components = {t['component'] for t in tasks if t.get('component')}
        if any(t.get('component') in train_components for t in evaluation):
            raise ValueError('Training and evaluation components overlap')
    print(f"设备 {device}；策略模型 {cfg.init}；{len(tasks)} 道题；每步 {cfg.prompts_per_step} 道 × {cfg.group_size} 个回答，共 {cfg.steps} 步", flush=True)
    stats = {}
    baseline=evaluate_policy(model,fmt,evaluation,cfg,out,0,judge) if evaluation and not cfg.resume else None
    if baseline and not cfg.best_trained_only:
        best_reward = baseline['mean_reward']
        save_checkpoint(out / 'best.pt', model, [], 0, rl_config=vars(cfg), heldout_reward=best_reward)
    if cfg.steps == 0:return stats
    writer = None
    if cfg.tensorboard or cfg.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cfg.tensorboard_dir or str(out/'tensorboard'),flush_secs=2)
        writer.add_text('run/config',json.dumps(vars(cfg),indent=2),0)
        if baseline:writer.add_scalar('rl/heldout_mean_reward',baseline['mean_reward'],0)
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for step in range(start_step, cfg.steps):
            t0 = time.perf_counter()
            batch = [tasks[i] for i in rng.choice(len(tasks), size=min(cfg.prompts_per_step, len(tasks)), replace=False,p=sampling_weights)]
            groups = []
            for task in batch:
                if task["reward"] == "agent":
                    env = ToolEnv(task)
                    messages = [{"role": "system", "content": env.describe()}] + [m for m in task["messages"] if m["role"] != "system"]
                    groups.append([run_agent(model, fmt, messages, ToolEnv(task), cfg.max_new, cfg.max_turns, cfg.temperature, top_k, gen)
                                   for _ in range(cfg.group_size)])
                else:
                    with autocast(device):
                        groups.append(sample_group(model, fmt, task["messages"], cfg.group_size, cfg.max_new, cfg.temperature, top_k, gen,
                                                   fast=False if cfg.eager_rollout else None))
            rollout_s = time.perf_counter() - t0
            rewards = score(batch, groups, judge, cfg.poetry_meter_weight)
            if cfg.unfinished_penalty:
                rewards = [[value - cfg.unfinished_penalty * (not episode.finished)
                            for value, episode in zip(values, group)] for values, group in zip(rewards, groups)]
            if cfg.record_rollouts:
                record=dict(step=step+1,groups=[dict(prompt=t['messages'][0]['content'],answer=t.get('answer'),
                    rewards=rs,responses=[e.text for e in group],finished=[e.finished for e in group])
                    for t,group,rs in zip(batch,groups,rewards)])
                with (out/'rollouts.jsonl').open('a',encoding='utf-8') as traces:traces.write(json.dumps(record,ensure_ascii=False)+'\n')

            set_lr_scale(optimizers, (1. if cfg.policy_optimizer=='adamw' else cfg.lr_mult) * min(1.0, (step + 1) / max(1, cfg.warmup_steps)))
            episodes, advs = [], []
            for group, rs in zip(groups, rewards):
                for ep, a in zip(group, advantages(rs,normalize=cfg.algorithm=='grpo')):
                    if a != 0 or reference is not None:
                        episodes.append(ep)
                        advs.append(a)
            n_total = sum(len(g) for g in groups)
            loss = 0.0
            metrics = {}
            if episodes:
                model.train()
                loss = policy_loss(model, episodes, advs, norm=n_total if cfg.algorithm=='grpo' else n_total*cfg.max_new,
                                   micro_batch=cfg.micro_batch,device=device,reference=reference,kl_beta=cfg.kl_beta,
                                   grpo=cfg.algorithm=='grpo',metrics=metrics)
                if not math.isfinite(loss):
                    raise RuntimeError(f"第 {step} 步的损失变成了 {loss}")
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm,error_if_nonfinite=True)
                metrics['grad_norm']=float(grad_norm)
                step_optimizers(optimizers)
            flat = [r for rs in rewards for r in rs]
            by_kind = {}
            for task, rs in zip(batch, rewards):
                by_kind.setdefault(task["reward"], []).extend(rs)
            stats = dict(step=step + 1, reward=round(sum(flat) / len(flat), 4), loss=round(loss, 5),
                         by_kind={k: round(sum(v) / len(v), 4) for k, v in by_kind.items()},
                         learned_groups=sum(any(a != 0 for a in advantages(rs)) for rs in rewards),
                         response_tokens=round(sum(sum(e.learn) for g in groups for e in g) / n_total, 1),
                         finished=round(sum(e.finished for g in groups for e in g) / n_total, 3),
                         judge_failures=judge.failures if judge is not None else 0,
                         rollout_s=round(rollout_s, 1), step_s=round(time.perf_counter() - t0, 1),**metrics)
            total_learned_groups += stats['learned_groups']
            if device.type=='cuda':stats['peak_memory_gib']=torch.cuda.max_memory_allocated()/2**30
            log.write(json.dumps(stats, ensure_ascii=False) + "\n")
            log.flush()
            if writer:
                for key in ['reward','learned_groups','sampled_kl','grad_norm','response_tokens','finished','step_s','peak_memory_gib']:
                    if key in stats:writer.add_scalar('rl/'+key,stats[key],step+1)
            if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
                kinds = "，".join(f"{k} {v:+.3f}" for k, v in stats["by_kind"].items())
                print(f"第 {step + 1:>5} 步  平均奖励 {stats['reward']:+.4f}（{kinds}）  回答平均 {stats['response_tokens']} 词  "
                      f"写完的 {stats['finished']:.0%}  采样 {stats['rollout_s']}s / 共 {stats['step_s']}s", flush=True)
            if evaluation and ((step+1)%cfg.eval_every==0 or step+1==cfg.steps):
                report=evaluate_policy(model,fmt,evaluation,cfg,out,step+1,judge)
                if writer:writer.add_scalar('rl/heldout_mean_reward',report['mean_reward'],step+1)
                eligible = not cfg.best_trained_only or total_learned_groups > 0
                if eligible and report['mean_reward'] > best_reward:
                    best_reward, best_step = report['mean_reward'], step + 1
                    save_checkpoint(out / 'best.pt', model, [], step + 1, rl_config=vars(cfg), heldout_reward=best_reward)
            if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
                save_checkpoint(out / 'ckpt.pt', model, optimizers, step + 1, rl_config=vars(cfg), rng=rng_state(),
                                rl_progress=dict(sample_rng=rng.bit_generator.state, token_rng=gen.get_state() if gen is not None else None,
                                                 best_reward=best_reward, best_step=best_step,
                                                 total_learned_groups=total_learned_groups))
    if writer:writer.close()
    (out/'summary.json').write_text(json.dumps(dict(step=cfg.steps,best_step=best_step,
        best_reward=best_reward if math.isfinite(best_reward) else None,stats=stats,
        total_learned_groups=total_learned_groups,
        judge_failures=judge.failures if judge is not None else 0,
        judge_retries=getattr(judge,'retries',0) if judge is not None else 0),ensure_ascii=False,indent=2),encoding='utf-8')
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GRPO：RLVR、agent RL、RLAIF")
    g = p.add_argument_group("模型、题目和输出")
    g.add_argument("--run", default="out/small", help="运行目录，各个阶段都放在这里")
    g.add_argument("--init", default=None, help="策略模型从这个存档出发；默认 <运行目录>/sft/ckpt.pt。纯基座 RL 加 --prompt-format plain")
    g.add_argument("--tasks", default="data/rl_small.parquet", help="题目数据（parquet），格式见文件开头")
    g.add_argument('--reward-kinds',nargs='+',choices=KINDS,help='Filter task kinds, including validation')
    balance=g.add_mutually_exclusive_group()
    balance.add_argument('--balanced-kinds',action='store_true',help='Give each reward kind equal prompt sampling mass')
    balance.add_argument('--balanced-categories',action='store_true',help='Give each task category equal sampling mass (fallback: domain, then reward)')
    g.add_argument("--out", default=None, help="输出目录；默认 <运行目录>/rl")
    g.add_argument('--prompt-format',choices=['chat','plain'],default='chat')
    g.add_argument('--tokenizer',default=str(DEFAULT_MODEL))
    g.add_argument('--eval-tasks',default='')
    g.add_argument('--eval-every',type=int,default=25)
    g.add_argument('--eval-limit',type=int,default=48)
    g.add_argument('--eval-samples',type=int,default=4)
    g.add_argument('--best-trained-only', action='store_true',
                   help='Log the baseline and rank checkpoints only after receiving a non-uniform reward group.')

    g = p.add_argument_group("采样")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--prompts-per-step", type=int, default=8, help="每步抽几道题")
    g.add_argument("--group-size", type=int, default=8, help="每道题采几个回答（组内比较）")
    g.add_argument("--max-new", type=int, default=512, help="每个回答最多多少词；也是 Dr. GRPO 损失的常数分母")
    g.add_argument("--max-turns", type=int, default=4, help="agent 题最多几轮")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0, help="0 表示不截断")
    g.add_argument("--micro-batch", type=int, default=16, help="一次前向放几个回答")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument('--eager-rollout',action='store_true',help='Skip decoding compilation for short pilot runs')

    g = p.add_argument_group("奖励")
    g.add_argument("--judge-url", default="", help="评审模型的 OpenAI 兼容接口；有 judge、poetry 题时要给")
    g.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    g.add_argument("--judge-workers", type=int, default=16, help="同时发几个评审请求；vLLM 能并发，mlx_lm.server 一次只算一个，用 1–2")
    g.add_argument("--judge-timeout", type=float, default=120.0)
    g.add_argument("--poetry-meter-weight", type=float, default=0.4, help="诗词奖励里格律分占多少，其余是评审分")
    g.add_argument('--unfinished-penalty', type=float, default=0., help='Optional penalty for responses that reach the token cap')

    g = p.add_argument_group("训练")
    g.add_argument("--warmup-steps", type=int, default=5)
    g.add_argument('--algorithm',choices=['drgrpo','grpo'],default='drgrpo')
    g.add_argument('--policy-optimizer',choices=['muon','adamw'],default='muon')
    g.add_argument('--learning-rate',type=float,default=2e-6,help='AdamW actor learning rate')
    g.add_argument('--kl-beta',type=float,default=0.)
    g.add_argument('--max-grad-norm',type=float,default=1.)
    add_optimizer_args(p, lr_mult=0.05)

    g = p.add_argument_group("记录、存档和硬件")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--save-every", type=int, default=50)
    g.add_argument("--device", default="auto")
    g.add_argument('--gpu-memory-fraction',type=float,default=0.)
    g.add_argument('--resume', action='store_true')
    g.add_argument('--attention-backend',choices=['sdpa','flex','cudnn'],default='sdpa',help='SDPA supports variable-length RL batches without compilation')
    g.add_argument('--tensorboard',action='store_true')
    g.add_argument('--tensorboard-dir',default='')
    g.add_argument('--record-rollouts',action='store_true',help='Save generated responses and verifier scores for inspection')
    args=p.parse_args(argv)
    args.init = args.init or str(find_stage(args.run, ("sft",)))
    args.out = args.out or str(Path(args.run) / "rl")
    if args.steps<0 or min(args.prompts_per_step,args.group_size,args.max_new,args.micro_batch,args.eval_every,args.eval_limit,args.eval_samples,args.save_every,args.log_every)<1:
        p.error('Invalid nonpositive RL count')
    if args.kl_beta<0 or args.learning_rate<=0 or args.max_grad_norm<=0:p.error('Invalid KL, learning rate or gradient limit')
    if args.unfinished_penalty < 0:p.error('Unfinished penalty must be nonnegative')
    if args.algorithm=='grpo' and (args.temperature!=1. or args.top_k):
        p.error('Single-update on-policy GRPO requires temperature=1 and top-k=0')
    return args


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
