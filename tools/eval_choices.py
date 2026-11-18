"""Compare full candidate-answer likelihoods on a frozen multiple-choice suite."""
import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from inference.tokenizer import Tokenizer
from inference.generate import prompt_ids
from lm import LanguageModel, ModelConfig


@torch.inference_mode()
def score_candidates(model, context_ids, choice_ids):
    """Reuse the context cache; branch it independently for each whole answer."""
    device=next(model.parameters()).device
    context=torch.tensor([context_ids],device=device)
    cache=model.new_state()
    output=model(context,state=cache)
    first_logp=F.log_softmax(output.logits[0,-1],dim=-1)
    del output
    scores=[]
    for answer in choice_ids:
        if not answer:raise ValueError('Candidate answers must contain at least one token')
        total=float(first_logp[answer[0]])
        if len(answer)>1:
            branch=copy.deepcopy(cache)
            for previous,target in zip(answer[:-1],answer[1:]):
                logits=model(torch.tensor([[previous]],device=device),state=branch).logits[0,-1]
                total+=float(F.log_softmax(logits,dim=-1)[target])
            del branch
        scores.append(total)
    return scores


@torch.inference_mode()
def score_candidates_uncached(model,context_ids,choice_ids):
    """Independent teacher-forced reference used to verify cache scoring."""
    device=next(model.parameters()).device
    scores=[]
    for answer in choice_ids:
        inputs=context_ids+answer[:-1]
        logits=model(torch.tensor([inputs],device=device)).logits[0,len(context_ids)-1:]
        targets=torch.tensor(answer,device=device)
        values=F.log_softmax(logits,dim=-1).gather(1,targets[:,None])[:,0]
        scores.append(float(values.double().sum()))
    return scores


def choice_metrics(scores,byte_lengths,gold):
    normalized=[score/length for score,length in zip(scores,byte_lengths)]
    raw_prediction=max(range(len(scores)),key=scores.__getitem__)
    prediction=max(range(len(scores)),key=normalized.__getitem__)
    wrong=[i for i in range(len(scores)) if i!=gold]
    maximum=max(scores)
    weights=[math.exp(value-maximum) for value in scores]
    return dict(prediction=prediction,raw_prediction=raw_prediction,
        correct=prediction==gold,raw_correct=raw_prediction==gold,
        normalized_scores=normalized,
        margin=normalized[gold]-max(normalized[i] for i in wrong),
        raw_margin=scores[gold]-max(scores[i] for i in wrong),
        correct_probability_among_choices=weights[gold]/sum(weights))


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        groups[row['category']].append(row)
        if row['category'].startswith('c3_'):groups['c3_all'].append(row)
        else:groups['synthetic_all'].append(row)
    result={}
    for name,group in groups.items():
        count=len(group)
        result[name]=dict(items=count,correct=sum(row['correct'] for row in group),
            accuracy=sum(row['correct'] for row in group)/count,
            raw_correct=sum(row['raw_correct'] for row in group),
            raw_accuracy=sum(row['raw_correct'] for row in group)/count,
            mean_margin=sum(row['margin'] for row in group)/count,
            mean_raw_margin=sum(row['raw_margin'] for row in group)/count,
            mean_correct_probability_among_choices=sum(row['correct_probability_among_choices'] for row in group)/count)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run',choices=['small','medium'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--verify',type=int,default=3)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--scoring',choices=('cached','teacher-forced'),default=None)
    args=parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    scoring=args.scoring or ('teacher-forced' if device.type=='cuda' else 'cached')
    if hasattr(os,'nice'):os.nice(10)
    args.out.mkdir(parents=True,exist_ok=True)
    source=args.dataset.read_bytes()
    suite=json.loads(source)
    items=suite['items'][:args.limit] if args.limit else suite['items']
    tokenizer_path=Path('data/tokens/base/tokenizer.model')
    tokenizer=Tokenizer(tokenizer_path)
    encoded=[(prompt_ids(tokenizer,item['prompt']),[tokenizer.encode(choice) for choice in item['choices']]) for item in items]
    encoded_hash=hashlib.sha256(json.dumps(encoded).encode()).hexdigest()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    step=int(state['step'])
    progress=state.get('training_progress') or {}
    training=state.get('train_config') or {}
    tokens=progress.get('consumed_rows',step*training.get('batch_size',16)*training.get('grad_accum',1))*progress.get('seq_len',training.get('seq_len',2048))
    lineage=args.checkpoint.parent/'lineage.json'
    if lineage.exists():tokens+=json.loads(lineage.read_text(encoding='utf-8'))['source_tokens']
    model=LanguageModel(ModelConfig.from_json(state['model_config']))
    model.load_state_dict(state['model'])
    model.to(device)
    for block in model.blocks:
        block.attn.backend='sdpa';block.attn.fuse_qkv=False
    model.eval().requires_grad_(False)
    del state
    gc.collect()
    report=dict(model=args.run,step=step,trained_tokens=tokens,token_equivalent_step=tokens//32768,
        parameters=sum(p.numel() for p in model.parameters()),device=str(device),dtype='float32',threads=args.threads,
        dataset_sha256=hashlib.sha256(source).hexdigest(),encoded_inputs_sha256=encoded_hash,
        tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        items_total=len(items),dataset_metadata=suite['metadata'],
        protocol=dict(prefix_eos=True,include_option_labels=False,scoring='Full candidate-answer conditional loglikelihood; no sampling',
            primary='accuracy after dividing full-answer loglikelihood by answer UTF-8 bytes',
            secondary='accuracy using unnormalized full-answer loglikelihood',
            margin_unit='nats per UTF-8 byte',verify_first_items=args.verify,implementation=scoring),
        verification=[],rows=[])
    started=time.monotonic()

    def save():
        report['elapsed_seconds']=round(time.monotonic()-started,2)
        report['summary']=summarize(report['rows'])
        temporary=args.out/'results.json.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(args.out/'results.json')

    print('LOADED',json.dumps(dict(model=args.run,step=step,items=len(items),encoded_hash=encoded_hash)),flush=True)
    with torch.inference_mode():
        for index,(item,(context,answers)) in enumerate(zip(items,encoded)):
            if len(context)+max(map(len,answers))>model.cfg.max_seq_len:raise ValueError(f'Context too long: {item["id"]}')
            scores=(score_candidates if scoring=='cached' else score_candidates_uncached)(model,context,answers)
            if index<args.verify:
                reference=(score_candidates_uncached if scoring=='cached' else score_candidates)(model,context,answers)
                error=max(abs(a-b) for a,b in zip(scores,reference))
                assert error<5e-4,(item['id'],scores,reference,error)
                report['verification'].append(dict(id=item['id'],max_absolute_loglikelihood_error=error))
            lengths=[len(choice.encode('utf-8')) for choice in item['choices']]
            row=dict(id=item['id'],category=item['category'],gold=item['gold'],choices=item['choices'],
                prompt_tokens=len(context),answer_tokens=list(map(len,answers)),answer_bytes=lengths,
                loglikelihood=scores,**choice_metrics(scores,lengths,item['gold']))
            report['rows'].append(row)
            if (index+1)%10==0 or index+1==len(items):
                save()
                print('PROGRESS',index+1,len(items),report['elapsed_seconds'],flush=True)
    report['complete']=True
    save()
    print('COMPLETE',json.dumps(dict(model=args.run,step=step,summary=report['summary'],elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
