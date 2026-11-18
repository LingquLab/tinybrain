"""Measure bits per UTF-8 byte on a fixed validation sample."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from inference.tokenizer import CONTROL, Tokenizer
from lm import LanguageModel, ModelConfig
from training.dataset import TokenData


def token_byte_lengths(tokenizer):
    """Count SentencePiece text bytes; exclude special tokens and handle byte fallback."""
    sp = tokenizer.sp
    lengths = []
    for index in range(tokenizer.vocab_size):
        piece = sp.id_to_piece(index)
        if sp.is_control(index) or sp.is_unknown(index) or sp.is_unused(index) or piece in CONTROL:
            length = 0
        elif sp.is_byte(index):
            length = 1
        else:
            length = len(piece.replace('\u2581', ' ').encode('utf-8'))
        lengths.append(length)
    return torch.tensor(lengths, dtype=torch.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', choices=['small', 'medium'], required=True)
    parser.add_argument('--data', type=Path, default=Path('data/tokens/base'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sequences', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=1000003)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--no-snapshot',action='store_true',help='Avoid duplicating an already immutable source checkpoint')
    args = parser.parse_args(argv)
    if min(args.sequences, args.seq_len, args.threads) <= 0:
        parser.error('sequences, seq-len and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    if hasattr(os, 'nice'):
        os.nice(10)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    step = int(state['step'])
    progress = state.get('training_progress') or {}
    train_config = state.get('train_config') or {}
    trained_tokens = progress.get('consumed_rows', step * train_config.get('batch_size', 16) * train_config.get('grad_accum', 1)) * progress.get('seq_len', train_config.get('seq_len', 2048))
    lineage=args.checkpoint.parent/'lineage.json'
    if lineage.exists():trained_tokens+=json.loads(lineage.read_text(encoding='utf-8'))['source_tokens']
    model = LanguageModel(ModelConfig.from_json(state['model_config']))
    model.load_state_dict(state['model'])
    model.eval().requires_grad_(False)
    snapshot = {key: state[key] for key in ['model_config', 'step', 'train_config', 'training_progress'] if key in state}
    snapshot['model'] = model.state_dict()
    del state
    gc.collect()
    if not args.no_snapshot:torch.save(snapshot, args.out / 'model.pt')
    del snapshot
    model.to(device)
    for block in model.blocks:
        block.attn.backend='sdpa';block.attn.fuse_qkv=False
    tokenizer_path = args.data / 'tokenizer.model'
    tokenizer = Tokenizer(tokenizer_path)
    byte_lengths = token_byte_lengths(tokenizer)
    byte_hash=hashlib.sha256(byte_lengths.numpy().tobytes()).hexdigest()
    byte_lengths=byte_lengths.to(device)
    data = TokenData(args.data, 'val')
    x, y = data.batch(args.sequences, args.seq_len, args.seed, torch.device('cpu'))
    sample_hash = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    report = dict(model=args.run, step=step, trained_tokens=trained_tokens,
                  token_equivalent_step=trained_tokens // 32768,
                  parameters=sum(p.numel() for p in model.parameters()), device=str(device), dtype='float32',
                  seed=args.seed, sequences=args.sequences, seq_len=args.seq_len, threads=args.threads,
                  sample_sha256=sample_hash, tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                  byte_lengths_sha256=byte_hash,
                  metric='sum(non-special target NLL) / (ln(2) * sum(target UTF-8 bytes))', rows=[])

    def save():
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        temporary = args.out / 'results.json.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.out / 'results.json')

    save()
    print('LOADED', json.dumps({key: report[key] for key in ['model', 'step', 'trained_tokens', 'sample_sha256']}), flush=True)
    with torch.inference_mode():
        for index in range(args.sequences):
            targets = y[index].to(device)
            logits = model(x[index:index+1].to(device)).logits[0]
            losses = F.cross_entropy(logits, targets, reduction='none')
            sizes = byte_lengths[targets]
            valid = sizes > 0
            nats = losses[valid].double().sum().item()
            count_bytes = int(sizes.sum())
            report['rows'].append(dict(index=index, nats=nats, bytes=count_bytes,
                text_tokens=int(valid.sum()), special_tokens=int((~valid).sum()),
                all_token_nats=losses.double().sum().item(), bpb=nats/(math.log(2)*count_bytes)))
            del logits, losses
            save()
            if (index + 1) % 8 == 0:
                print('PROGRESS', index + 1, args.sequences, report['elapsed_seconds'], flush=True)
    report['total_nats'] = sum(row['nats'] for row in report['rows'])
    report['total_bytes'] = sum(row['bytes'] for row in report['rows'])
    report['text_tokens'] = sum(row['text_tokens'] for row in report['rows'])
    report['special_tokens'] = sum(row['special_tokens'] for row in report['rows'])
    report['bpb'] = report['total_nats'] / (math.log(2) * report['total_bytes'])
    report['mean_nll_nats_per_text_token'] = report['total_nats'] / report['text_tokens']
    report['all_token_mean_loss'] = sum(row['all_token_nats'] for row in report['rows']) / y.numel()
    report['complete'] = True
    save()
    print('COMPLETE', json.dumps({key: report[key] for key in ['model', 'step', 'bpb', 'total_bytes', 'text_tokens', 'special_tokens', 'all_token_mean_loss', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
