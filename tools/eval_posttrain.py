"""Reproducible free-generation evaluation, separate from candidate ranking."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from inference.artifact import tokenizer_path
from inference.template import ChatFormat
from inference.tokenizer import DEFAULT_MODEL, Tokenizer
from lm import LanguageModel, ModelConfig
from training.pipeline import write_json
from training.rewards import math as math_reward
from training.rewards.code import unsupported_reason
from training.rewards.judge import Judge
from training.rewards.tools import ToolEnv
from training.rollout import PlainFormat, balanced_tasks, fits, run_agent, sample_group
from training.train_rl import KINDS, PROGRAM_REWARDS, load_tasks, score
from training.trainer_utils import autocast, load_checkpoint, pick_device, setup_device


class PlainEvaluationFormat(PlainFormat):
    """Keep dialogue history in plain-text evaluations without chat control tokens."""
    def prompt(self, messages):
        if len(messages) == 1 and messages[0].get('role') == 'user':
            return super().prompt(messages)
        if not messages:
            raise ValueError('Evaluation dialogue is empty')
        for message in messages:
            if message.get('role') not in ('system', 'user', 'assistant', 'tool'):
                raise ValueError('Invalid evaluation dialogue role')
            if not isinstance(message.get('content'), str):
                raise ValueError('Evaluation dialogue content must be text')
        transcript = '\n\n'.join(f"{m['role']}: {m['content']}" for m in messages)
        return super().prompt([dict(role='user', content=transcript + '\n\nassistant:')])


def evaluate(checkpoint, tasks_path, destination, *, device='cpu', limit=100, samples=1,
             max_new=192, temperature=.8, top_k=50, seed=20261117, prompt_format='chat',
             tokenizer=None, threads=2, judge=None, allow_execution=False, gpu_memory_fraction=0.,ungraded=False,reward_kinds=None):
    torch.set_num_threads(threads)
    device = pick_device(device); setup_device(device)
    if gpu_memory_fraction:
        if device.type!='cuda' or not 0<gpu_memory_fraction<=1:raise ValueError('Invalid CUDA memory fraction')
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction,device.index)
    # Read one immutable inode: rolling atomic checkpoint replacement cannot change it.
    state = load_checkpoint(Path(checkpoint), torch.device('cpu'), mmap=True)
    checkpoint_step = int(state.get('step', 0))
    artifact = state.get('artifact_metadata')
    model = LanguageModel(ModelConfig.from_json(state['model_config'])).to(device).eval()
    model.load_state_dict(state['model']); del state
    for block in model.blocks:
        block.attn.backend = 'sdpa'; block.attn.fuse_qkv = False
    tok = Tokenizer(tokenizer or tokenizer_path(checkpoint, DEFAULT_MODEL))
    fmt = (ChatFormat if prompt_format == 'chat' else PlainEvaluationFormat)(tok)
    all_tasks = load_tasks(str(tasks_path)); exclusions = Counter()
    candidates = []
    for task in all_tasks:
        if reward_kinds and task['reward'] not in reward_kinds:
            exclusions['outside_selected_reward_kinds']+=1
        elif task['reward'] in ('code', 'agent') and not allow_execution:
            exclusions['requires_execution_sandbox'] += 1
        elif task['reward']=='code' and unsupported_reason(task):
            exclusions['unsupported_code_checker']+=1
        elif task['reward'] not in PROGRAM_REWARDS and judge is None and not ungraded:
            exclusions['requires_judge'] += 1
        elif not fits(model, fmt, task['messages'], max_new):
            exclusions['too_long'] += 1
        else:
            candidates.append(task)
    tasks = balanced_tasks(candidates, limit)
    if not tasks:
        raise ValueError('No evaluable tasks')
    rows = []; started = time.monotonic()
    for index, task in enumerate(tasks):
        canonical = json.dumps(task['messages'], sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(canonical.encode()).hexdigest()
        generator = torch.Generator(device=device).manual_seed((seed + int(key[:12], 16)) % (2**63-1))
        with torch.inference_mode(), autocast(device):
            if task['reward']=='agent':
                environment=ToolEnv(task)
                messages=[dict(role='system',content=environment.describe())]+[m for m in task['messages'] if m['role']!='system']
                episodes=[run_agent(model,fmt,messages,ToolEnv(task),max_new,4,temperature,top_k or None,generator) for _ in range(samples)]
            else:
                episodes = sample_group(model, fmt, task['messages'], samples, max_new, temperature,
                                        top_k or None, generator, fast=False)
        rewards = ([None]*len(episodes) if ungraded and judge is None and task['reward'] not in PROGRAM_REWARDS
                   else score([task], [episodes], judge)[0])
        kind = task['reward']
        correct = []
        for episode, value in zip(episodes, rewards):
            if kind in ('math', 'math_strict', 'math_steps'):
                # Do not credit a number accidentally mentioned in a long answer.
                match = bool(math_reward.strict_reward(episode, task))
            elif kind in PROGRAM_REWARDS:
                match = value >= 1. - 1e-8
            else:
                match = None
            correct.append(match)
        rows.append(dict(id=key, category=task.get('category') or task.get('domain') or kind,
                         reward_kind=kind, messages=task['messages'], answer=task.get('answer'),
                         responses=[e.text for e in episodes], finished=[e.finished for e in episodes],
                         response_tokens=[sum(e.learn) for e in episodes], rewards=rewards,
                         correct=correct, correct_and_finished=[bool(v and e.finished) if v is not None else None
                                                               for v, e in zip(correct, episodes)]))
        if kind == 'math_steps':
            rows[-1]['verified_step_fraction'] = [math_reward.step_credit(e.text, task) for e in episodes]
        if kind == 'agent':
            rows[-1]['tool_turns'] = [e.turns for e in episodes]
        if (index+1) % 10 == 0:
            write_json(str(destination)+'.partial',dict(complete=False,checkpoint=str(checkpoint),evaluated=len(rows),total=len(tasks),rows=rows))
            print(json.dumps(dict(evaluated=index+1, total=len(tasks), seconds=round(time.monotonic()-started, 1))), flush=True)
    summary = summarize(rows)
    hashes={filename:hashlib.sha256(Path(filename).read_bytes()).hexdigest() for filename in str(tasks_path).split(',')}
    report = dict(checkpoint=str(checkpoint), checkpoint_step=checkpoint_step,
                  tasks=str(tasks_path),task_files=hashes,tasks_sha256=next(iter(hashes.values())) if len(hashes)==1 else hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                  protocol='free generation; explicit answer; EOS completion reported independently',
                  prompt_format=prompt_format, temperature=temperature, top_k=top_k, seed=seed,
                  plain_multiturn_protocol=('role: content; blank-line separators; assistant: suffix; '
                                            'single-user prompts unchanged') if prompt_format == 'plain' else None,
                  samples=samples, max_new=max_new, balanced_limit=limit, exclusions=dict(exclusions),
                  seconds=time.monotonic()-started, summary=summary, rows=rows,complete=True,
                  ungraded_answers_require_semantic_review=ungraded and judge is None)
    lineage=Path(checkpoint).parent/'lineage.json'
    if lineage.exists():report['training_lineage']=json.loads(lineage.read_text(encoding='utf-8'))
    if artifact:
        report['evaluated_artifact'] = dict(format=artifact['format'], stage=artifact['stage'],
                                             model_sha256=artifact['files']['model.safetensors']['sha256'],
                                             tokenizer_sha256=hashlib.sha256(tok.path.read_bytes()).hexdigest())
    write_json(destination, report)
    print(json.dumps(dict(checkpoint_step=checkpoint_step, summary=summary, output=str(destination)), ensure_ascii=False), flush=True)
    return report


def summarize(rows):
    def aggregate(selected):
        values = [v for row in selected for v in row['correct'] if v is not None]
        complete = [v for row in selected for v in row['correct_and_finished'] if v is not None]
        finished = [v for row in selected for v in row['finished']]
        rewards=[v for row in selected for v in row['rewards'] if v is not None]
        result = dict(questions=len(selected), accuracy=sum(values)/len(values) if values else None,
                    complete_accuracy=sum(complete)/len(complete) if complete else None,
                    finished_fraction=sum(finished)/len(finished),
                    mean_reward=sum(rewards)/len(rewards) if rewards else None)
        credits = [v for row in selected for v in row.get('verified_step_fraction', [])]
        if credits:
            result['verified_step_fraction'] = sum(credits)/len(credits)
            result['process_responses'] = len(credits)
        return result
    return dict(overall=aggregate(rows), by_category={category: aggregate([r for r in rows if r['category']==category])
                for category in dict.fromkeys(r['category'] for r in rows)})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--tasks', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--limit', type=int, default=100)
    p.add_argument('--samples', type=int, default=1); p.add_argument('--max-new', type=int, default=192)
    p.add_argument('--temperature', type=float, default=.8); p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261117); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--prompt-format', choices=('plain','chat'), default='chat'); p.add_argument('--tokenizer')
    p.add_argument('--allow-execution',action='store_true',help='Enable sandboxed code and agent verification')
    p.add_argument('--gpu-memory-fraction',type=float,default=0.)
    p.add_argument('--judge-url',default='');p.add_argument('--judge-model',default='Qwen/Qwen3.8-27B-FP8')
    p.add_argument('--ungraded',action='store_true',help='Keep open-ended answers for semantic review without inventing correctness scores')
    p.add_argument('--reward-kinds',nargs='+',choices=KINDS)
    args = p.parse_args(argv)
    if min(args.limit, args.samples, args.max_new, args.threads) <= 0:
        p.error('Counts must be positive')
    values = vars(args); values['destination'] = values.pop('out'); values['tasks_path'] = values.pop('tasks')
    url=values.pop('judge_url');model=values.pop('judge_model')
    values['judge']=Judge(url,model,workers=4,timeout=120) if url else None
    return evaluate(**values)


if __name__ == '__main__':
    main()
