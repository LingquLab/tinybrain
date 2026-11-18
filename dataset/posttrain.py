"""Pinned ModelScope inputs for SFT, domain experts, MOPD and RL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


REPOSITORY = 'caiwangzsnm/tinybrain-dataset'
FILES = ('sft_small.parquet','sft_medium.parquet','sft_expert_math.parquet',
         'sft_expert_code.parquet','sft_expert_literature.parquet','sft_heldout.parquet',
         'rl_small.parquet','rl_medium.parquet','eval_tasks.parquet','mopd_prompts.parquet')


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def digest(path):
    value=hashlib.sha256()
    with open(path,'rb') as source:
        while block:=source.read(1024*1024):value.update(block)
    return value.hexdigest()


def fetch_verified(url,destination,size,expected_sha256=None):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.stat().st_size==size:
        actual=digest(destination)
        if not expected_sha256 or actual==expected_sha256:return actual
    partial=destination.with_name(destination.name+'.partial')
    for attempt in range(3):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>size:partial.unlink();offset=0
            if offset<size:
                request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request,timeout=45) as response:
                    append=offset>0 and response.status==206
                    if not append:offset=0
                    with partial.open('ab' if append else 'wb') as output:
                        last=0
                        while block:=response.read(1024*1024):
                            output.write(block);offset+=len(block)
                            if time.monotonic()-last>=30:
                                print(json.dumps(dict(phase='model_download',file=destination.name,bytes=offset,total=size)),flush=True);last=time.monotonic()
            actual=digest(partial)
            if partial.stat().st_size!=size or expected_sha256 and actual!=expected_sha256:
                partial.unlink();raise ValueError('Downloaded model file failed checksum')
            os.replace(partial,destination);return actual
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))


def download_judge(directory,workers=3):
    """Download the selected judge from ModelScope, pinning every file revision."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor,as_completed
    repository='Qwen/Qwen3.8-27B-FP8';directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    query=urllib.parse.urlencode(dict(Revision='master',Recursive='true'))
    with urllib.request.urlopen(f'https://www.modelscope.cn/api/v1/models/{repository}/repo/files?'+query,timeout=30) as response:index=json.load(response)
    files=[x for x in index['Data']['Files'] if x['Type']=='blob' and not x['Path'].startswith(('.','assets/'))]
    remaining=sum(x['Size'] for x in files if not (directory/x['Path']).exists())
    if shutil.disk_usage(directory).free<remaining+12*2**30:
        raise RuntimeError('Judge download requires its remaining file bytes plus 12 GiB checkpoint reserve')
    write_json(directory/'modelscope-index.json',index)
    manifest=dict(repository=repository,provider='ModelScope',complete=False,files=[])
    def one(record):
        target=directory/record['Path']
        if not target.resolve().is_relative_to(directory.resolve()):raise ValueError('Invalid model path')
        expected=record.get('Sha256') or None
        if target.suffix=='.safetensors' and not expected:raise ValueError('Model weight has no expected checksum')
        url=f"https://www.modelscope.cn/models/{repository}/resolve/{record['Revision']}/{urllib.parse.quote(record['Path'])}"
        actual=fetch_verified(url,target,record['Size'],expected)
        return dict(path=record['Path'],revision=record['Revision'],bytes=record['Size'],sha256=actual,url=url)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,record) for record in files]):
            record=future.result();manifest['files'].append(record)
            write_json(directory/'download-manifest.json',manifest)
            print(json.dumps(dict(phase='model_file_verified',file=record['path'],bytes=record['bytes'])),flush=True)
    manifest['complete']=True;write_json(directory/'download-manifest.json',manifest)
    return manifest


def download(directory, selected=FILES):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    query=urllib.parse.urlencode(dict(Revision='master',Root='posttrain',Recursive='True',PageSize=100))
    with urllib.request.urlopen(f'https://www.modelscope.cn/api/v1/datasets/{REPOSITORY}/repo/tree?'+query,timeout=30) as response:
        index=json.load(response)
    records={r['Name']:r for r in index['Data']['Files']}
    manifest=dict(provider='ModelScope',repository=REPOSITORY,complete=False,files=[])
    write_json(directory/'download-index.json',index)
    for name in selected:
        record=records[name];destination=directory/name
        provenance=dict(provider='ModelScope',repo=REPOSITORY,remote_path=record['Path'],
                        revision=record['Revision'],sha256=record['Sha256'],bytes=record['Size'])
        url=f"https://www.modelscope.cn/datasets/{REPOSITORY}/resolve/{record['Revision']}/{record['Path']}"
        provenance['source_url']=url
        valid=destination.exists() and destination.stat().st_size==record['Size'] and digest(destination)==record['Sha256']
        if not valid:
            partial=destination.with_name(name+'.partial')
            for attempt in range(3):
                try:
                    offset=partial.stat().st_size if partial.exists() else 0
                    if offset>record['Size']:raise ValueError('Partial download longer than expected')
                    if offset<record['Size']:
                        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
                        with urllib.request.urlopen(request,timeout=30) as response:
                            append=offset>0 and response.status==206
                            if not append:offset=0
                            with partial.open('ab' if append else 'wb') as output:
                                last=0
                                while block:=response.read(1024*1024):
                                    output.write(block);offset+=len(block)
                                    if time.monotonic()-last>=20:
                                        print(json.dumps(dict(phase='download',file=name,bytes=offset,total=record['Size'])),flush=True);last=time.monotonic()
                    if partial.stat().st_size!=record['Size'] or digest(partial)!=record['Sha256']:
                        raise ValueError(f'Download checksum mismatch: {name}')
                    os.replace(partial,destination)
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2*(attempt+1))
        write_json(destination.with_name(name+'.source.json'),provenance)
        manifest['files'].append(provenance)
        write_json(directory/'download-manifest.json',manifest)
        print(json.dumps(dict(phase='download_verified',file=name,bytes=record['Size'])),flush=True)
    manifest['complete']=True;write_json(directory/'download-manifest.json',manifest)
    return manifest


def inspect(directory):
    import pyarrow.parquet as pq
    result={}
    for path in sorted(Path(directory).glob('*.parquet')):
        with pq.ParquetFile(path,pre_buffer=False) as source:
            result[path.name]=dict(rows=source.metadata.num_rows,columns=source.schema_arrow.names)
    return result


def prepare_rl(source, destination, protected=None, core=None):
    """Separate verifiers, executable tasks and LLM-judged tasks; exclude heldout prompts."""
    import collections
    import sqlite3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset.anneal import digest as prompt_digest
    source=Path(source);destination=Path(destination)
    if (destination/'manifest.json').exists():raise FileExistsError('Prepared RL destination already exists')
    destination.mkdir(parents=True,exist_ok=True)
    protected_keys=set(json.loads(Path(protected).read_text(encoding='utf-8')).get('keys',[])) if protected else set()
    def key(row):return prompt_digest('\n'.join(m.get('content') or '' for m in row['messages'] if m['role']=='user'))
    evaluation=source.parent/'eval_tasks.parquet'
    eval_rows=pq.read_table(evaluation).to_pylist()
    heldout={key(row) for row in eval_rows}
    blocked=protected_keys|heldout
    db=sqlite3.connect(destination/'dedup.sqlite');db.execute('CREATE TABLE seen (key TEXT PRIMARY KEY)')
    schema=pq.ParquetFile(source).schema_arrow
    writers={};pending=collections.defaultdict(list);counts=collections.Counter();rejected=collections.Counter()
    def group(row):
        kind=row['reward']
        if kind in ('judge','poetry'):return 'judge'
        if kind in ('code','agent'):return 'executable'
        if kind in ('math','math_strict','choice','reading','cloze','instruct','logic','exact'):return 'verifiable'
        raise ValueError(f'Unknown reward kind {kind}')
    def emit(name,row):
        row=dict(row)
        if row['reward']=='math':row['reward']='math_strict'
        pending[name].append(row);counts[name]+=1
        if len(pending[name])>=64:flush(name)
    def flush(name):
        rows=pending[name]
        if not rows:return
        table=pa.Table.from_pylist(rows,schema=schema)
        if name not in writers:writers[name]=pq.ParquetWriter(destination/f'{name}.parquet',table.schema,compression='zstd')
        writers[name].write_table(table);rows.clear()
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                k=key(row)
                if k in blocked:rejected['heldout_or_protected']+=1;continue
                try:db.execute('INSERT INTO seen VALUES (?)',(k,))
                except sqlite3.IntegrityError:rejected['duplicate_prompt']+=1;continue
                emit(group(row)+'-train',row)
        for row in eval_rows:
            if key(row) not in protected_keys:emit(group(row)+'-validation',row)
        for name in list(pending):flush(name)
        db.commit()
    finally:
        db.close()
        for writer in writers.values():writer.close()
    if core:
        for original,target in [('rl','core-train'),('validation','core-validation'),('test','core-test')]:
            path=Path(core)/f'{original}.parquet';table=pq.read_table(path)
            rewards=['math_strict' if v=='math' else v for v in table.column('reward').to_pylist()]
            table=table.set_column(table.schema.get_field_index('reward'),'reward',pa.array(rewards))
            pq.write_table(table,destination/f'{target}.parquet',compression='zstd');counts[target]=len(table)
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in destination.glob('*.parquet')}
    report=dict(complete=True,source=str(source),source_sha256=digest(source),files=files,counts=dict(counts),rejected=dict(rejected),
                protected_prompt_count=len(protected_keys),validation_prompt_count=len(heldout),
                validation_public_source_pretraining_overlap_not_excluded=True,
                executable_tasks_require_verified_sandbox=True,math_verifier='explicit final answer; no last-number fallback')
    write_json(destination/'manifest.json',report)
    return report


def prepare_mopd(source,destination,protected=None,per_domain=4000,seed=20261117):
    """Stream a balanced prompt reservoir, preserving all SFT/RL heldout prompts."""
    import collections
    import random
    import sqlite3
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset.anneal import digest as prompt_digest,rejection
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError('MOPD prompt file already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    blocked=set(json.loads(Path(protected).read_text(encoding='utf-8')).get('keys',[])) if protected else set()
    def key(messages):return prompt_digest('\n'.join(m.get('content') or '' for m in messages if m['role']=='user'))
    for name in ['sft_heldout.parquet','eval_tasks.parquet']:
        file=source.parent/name
        if file.exists():
            for batch in pq.ParquetFile(file).iter_batches(batch_size=64,columns=['messages']):
                blocked.update(key(row['messages']) for row in batch.to_pylist())
    filters=json.loads(Path(__file__).with_name('cos_recipe.json').read_text(encoding='utf-8'))['filters']
    reservoirs={};seen=collections.Counter();rejected=collections.Counter();rng=random.Random(seed)
    with tempfile.TemporaryDirectory() as folder:
        db=sqlite3.connect(Path(folder)/'seen.sqlite');db.execute('CREATE TABLE seen (key TEXT PRIMARY KEY)')
        try:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    messages=row['messages'];k=key(messages)
                    if k in blocked or int(k[:12],16)%50==0:rejected['heldout']+=1;continue
                    text='\n'.join(m.get('content') or '' for m in messages)
                    if len(text)>6000:rejected['too_long']+=1;continue
                    reason=rejection(text,filters)
                    if reason:rejected[reason]+=1;continue
                    try:db.execute('INSERT INTO seen VALUES (?)',(k,))
                    except sqlite3.IntegrityError:rejected['duplicate']+=1;continue
                    domain=row['domain'];seen[domain]+=1;pool=reservoirs.setdefault(domain,[])
                    if len(pool)<per_domain:pool.append(row)
                    else:
                        position=rng.randrange(seen[domain])
                        if position<per_domain:pool[position]=row
        finally:db.close()
    rows=[row for pool in reservoirs.values() for row in pool];rng.shuffle(rows)
    if not rows:raise ValueError('No eligible MOPD prompts')
    pq.write_table(pa.Table.from_pylist(rows),destination,compression='zstd')
    report=dict(complete=True,source=str(source),source_sha256=digest(source),output_sha256=digest(destination),seed=seed,
                counts={k:len(v) for k,v in reservoirs.items()},eligible_counts=dict(seen),rejected=dict(rejected),
                excluded_prompt_hashes=len(blocked),sft_validation_rule='canonical prompt SHA256 modulo 50 is excluded')
    write_json(destination.with_suffix('.manifest.json'),report);return report


def prepare_code(source,destination):
    """Keep tasks whose supplied tests support an exact-output verifier."""
    import collections
    import pyarrow as pa
    import pyarrow.parquet as pq
    from training.rewards.code import unsupported_reason
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError('Verified code task file already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    reader=pq.ParquetFile(source);writer=None;rejected=collections.Counter();count=0
    try:
        for batch in reader.iter_batches(batch_size=64):
            rows=[]
            for row in batch.to_pylist():
                if row['reward']!='code':continue
                reason=unsupported_reason(row)
                if reason:rejected[reason]+=1;continue
                rows.append(row)
            if rows:
                table=pa.Table.from_pylist(rows,schema=reader.schema_arrow)
                if writer is None:writer=pq.ParquetWriter(destination,table.schema,compression='zstd')
                writer.write_table(table);count+=len(rows)
    finally:
        if writer is not None:writer.close()
    if not count:raise ValueError('No tasks supported by the code verifier')
    report=dict(complete=True,source=str(source),source_sha256=digest(source),output_sha256=digest(destination),
                retained=count,rejected=dict(rejected),protocol='exact token output; declared case-insensitivity honored; custom checkers excluded')
    write_json(destination.with_suffix('.manifest.json'),report);return report


def verified_agent_answer(task):
    import datetime
    import re
    from fractions import Fraction
    from training.rewards.math import exact_decimal
    question=next(m['content'] for m in reversed(task['messages']) if m['role']=='user')
    match=re.search(r'请计算\s*(\d+)\s*×\s*(\d+)\s*\+\s*(\d+)',question)
    if match:
        a,b,c=map(int,match.groups());return str(a*b+c)
    match=re.search(r'(\d+)\s*的\s*(\d+)%',question)
    if match:
        a,p=map(int,match.groups());return exact_decimal(Fraction(a*p,100))
    match=re.search(r'(\d+)\s*的\s*(\d+)\s*次方除以\s*(\d+)\s*的余数',question)
    if match:
        a,b,m=map(int,match.groups());return str(pow(a,b,m))
    match=re.search(r'从\s*(\d+)年(\d+)月(\d+)日\s*到\s*(\d+)年(\d+)月(\d+)日',question)
    if match:
        values=list(map(int,match.groups()));return str((datetime.date(*values[3:])-datetime.date(*values[:3])).days)
    if '条目里第一个提到的年份' in question:
        documents={d['title']:d['text'][:1500] for d in task.get('documents') or []}
        titles=[title for title in re.findall(r'「([^」]+)」',question) if title in documents];years=[]
        if len(titles)!=(2 if '两个条目' in question else 1):raise ValueError('missing_retrieval_evidence')
        for title in titles:
            found=re.search(r'(1[5-9]\d\d|20[0-2]\d)年',documents.get(title,''))
            if found is None:raise ValueError('missing_retrieval_evidence')
            years.append(int(found[1]))
        if len(years)==1:return str(years[0])
        if len(years)==2:return str(abs(years[0]-years[1]))
    raise ValueError('unsupported_agent_family')


def prepare_agent(source,destination):
    import collections
    import pyarrow as pa
    import pyarrow.parquet as pq
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError('Verified agent task file already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    reader=pq.ParquetFile(source);writer=None;rejected=collections.Counter();count=corrected=0
    try:
        for batch in reader.iter_batches(batch_size=64):
            rows=[]
            for row in batch.to_pylist():
                if row['reward']!='agent':continue
                try:answer=verified_agent_answer(row)
                except ValueError as error:rejected[str(error)]+=1;continue
                corrected+=answer!=row['answer'];row['answer']=answer;rows.append(row)
            if rows:
                table=pa.Table.from_pylist(rows,schema=reader.schema_arrow)
                if writer is None:writer=pq.ParquetWriter(destination,table.schema,compression='zstd')
                writer.write_table(table);count+=len(rows)
    finally:
        if writer is not None:writer.close()
    if not count:raise ValueError('No verified agent tasks')
    report=dict(complete=True,source=str(source),source_sha256=digest(source),output_sha256=digest(destination),
                retained=count,corrected_answers=corrected,rejected=dict(rejected),
                protocol='exact arithmetic/date answers and retrieval evidence present in supplied documents')
    write_json(destination.with_suffix('.manifest.json'),report);return report


def prepare_sft(source,destination,seq_len,protected=None):
    """Stream vetted conversations into assistant-only train/validation rows."""
    import collections
    import sqlite3
    import shutil
    import numpy as np
    import pyarrow.parquet as pq
    from inference.template import ChatFormat
    from inference.tokenizer import Tokenizer
    from dataset.anneal import digest as prompt_digest, rejection
    from training.dataset import pack,write_meta

    source,destination=Path(source),Path(destination)
    if (destination/'meta.json').exists() or any(destination.glob('*.bin')):
        raise FileExistsError('Refusing to overwrite prepared SFT data')
    destination.mkdir(parents=True,exist_ok=True)
    tokenizer=Path('data/tokens/base/tokenizer.model');tok=Tokenizer(tokenizer);fmt=ChatFormat(tok)
    recipe=json.loads(Path('dataset/cos_recipe.json').read_text(encoding='utf-8'))
    protected_keys=set(json.loads(Path(protected).read_text(encoding='utf-8'))['keys']) if protected else set()
    reserved=source.parent/'sft_heldout.parquet'
    if reserved.exists():
        with pq.ParquetFile(reserved,pre_buffer=False) as table:
            for batch in table.iter_batches(batch_size=64,use_threads=False):
                for row in batch.to_pylist():
                    text='\n'.join(m['content'] for m in row['messages'] if m['role']=='user')
                    protected_keys.add(prompt_digest(text))
    db=sqlite3.connect(destination/'dedup.sqlite');db.execute('CREATE TABLE seen (key TEXT PRIMARY KEY)')
    pending={s:[] for s in ('train','val')}
    handles={s:(open(destination/f'{s}.ids.bin','wb'),open(destination/f'{s}.mask.bin','wb')) for s in pending}
    stats={s:collections.Counter() for s in pending};rejected=collections.Counter();rows=0;last=time.monotonic()

    def flush(split):
        ids,mask,counts=pack(pending[split],seq_len+1)
        handles[split][0].write(ids.astype('uint16').tobytes());handles[split][1].write(mask.tobytes())
        stats[split].update({k:v for k,v in counts.items() if k!='fill'})
        pending[split].clear()

    try:
        with pq.ParquetFile(source,pre_buffer=False) as table:
            for batch in table.iter_batches(batch_size=64,use_threads=False):
                for row in batch.to_pylist():
                    rows+=1;messages=row['messages']
                    if not messages or messages[-1]['role']!='assistant':rejected['unfinished_conversation']+=1;continue
                    prompt='\n'.join(m['content'] or '' for m in messages if m['role']=='user')
                    key=prompt_digest(prompt)
                    if not prompt.strip():rejected['empty_prompt']+=1;continue
                    if key in protected_keys:rejected['protected_evaluation_prompt']+=1;continue
                    text='\n'.join(m['content'] or '' for m in messages)
                    reason=rejection(text,recipe['filters'])
                    if reason:rejected[reason]+=1;continue
                    try:ids,mask=fmt.encode(messages,'assistant')
                    except (TypeError,ValueError,KeyError):rejected['invalid_conversation']+=1;continue
                    if len(ids)>seq_len+1:rejected['too_long']+=1;continue
                    try:db.execute('INSERT INTO seen VALUES (?)',(key,))
                    except sqlite3.IntegrityError:rejected['duplicate_prompt']+=1;continue
                    split=row.get('provided_split') or ('val' if int(key[:12],16)%50==0 else 'train')
                    if split not in ('train','val'):raise ValueError('Invalid explicit SFT split')
                    pending[split].append((ids,mask))
                    if len(pending[split])>=128:flush(split)
                if time.monotonic()-last>=30:
                    db.commit();print(json.dumps(dict(phase='prepare_sft',rows_scanned=rows,train_rows=stats['train']['rows'],rejected=dict(rejected))),flush=True);last=time.monotonic()
        for split in pending:
            if pending[split]:flush(split)
        db.commit()
    finally:
        db.close()
        for values in handles.values():
            for stream in values:stream.close()
    shutil.copyfile(tokenizer,destination/'tokenizer.model')
    write_meta(destination,tok.vocab_size,'uint16',tok.eos_id,'sentencepiece',format='packed',row_len=seq_len+1,
               train_on='assistant',tokenizer_file='tokenizer.model',stats=stats,
               source=str(source),split_rule='explicit provided_split, otherwise canonical user-prompt SHA256 modulo 50',protected_prompts=len(protected_keys),rejected=dict(rejected))
    return dict(rows_scanned=rows,stats=stats,rejected=dict(rejected))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='data/posttrain')
    parser.add_argument('--files',nargs='+',default=list(FILES),choices=FILES)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--prepare-sft')
    parser.add_argument('--prepare-rl')
    parser.add_argument('--prepare-mopd')
    parser.add_argument('--prepare-code')
    parser.add_argument('--prepare-agent')
    parser.add_argument('--per-domain',type=int,default=4000)
    parser.add_argument('--core')
    parser.add_argument('--seq-len',type=int,default=2048)
    parser.add_argument('--protected-prompts')
    parser.add_argument('--download-judge',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args(argv)
    if args.download_judge:
        result=download_judge(args.out,args.workers)
    elif args.prepare_sft:
        result=prepare_sft(args.prepare_sft,args.out,args.seq_len,args.protected_prompts)
    elif args.prepare_rl:
        result=prepare_rl(args.prepare_rl,args.out,args.protected_prompts,args.core)
    elif args.prepare_mopd:
        result=prepare_mopd(args.prepare_mopd,args.out,args.protected_prompts,args.per_domain)
    elif args.prepare_code:
        result=prepare_code(args.prepare_code,args.out)
    elif args.prepare_agent:
        result=prepare_agent(args.prepare_agent,args.out)
    else:result=inspect(args.out) if args.inspect else download(args.out,args.files)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
