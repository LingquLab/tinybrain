"""Persistent checkpoint capture and delivery-stage coordination."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


_service_processes={}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def latest_record(path):
    path=Path(path)
    if not path.exists():return {}
    with path.open('rb') as source:
        source.seek(max(0,path.stat().st_size-32768));lines=source.read().splitlines()
    for line in reversed(lines):
        try:
            record=json.loads(line)
            if 'tokens' in record:return record
        except ValueError:pass
    return {}


def capture_checkpoint(source,destination,target_tokens):
    """Hard-link an atomic checkpoint, then inspect that exact inode on CPU."""
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    checkpoint=destination/'ckpt.pt';report=destination/'archive.json'
    if report.exists():
        previous=json.loads(report.read_text(encoding='utf-8'))
        if previous.get('complete') and checkpoint.exists() and checkpoint.stat().st_size==previous['bytes']:
            return previous
        raise ValueError('Existing archive is incomplete; inspect before replacing it')
    pending=destination/'capture.partial.pt'
    if pending.exists():pending.unlink()
    os.link(source,pending)
    import torch
    state=torch.load(pending,map_location='cpu',mmap=True,weights_only=False)
    progress=state.get('training_progress') or {}
    config=state.get('train_config') or {}
    rows=progress.get('consumed_rows',state['step']*config.get('batch_size',16)*config.get('grad_accum',1))
    tokens=rows*progress.get('seq_len',config.get('seq_len',2048))
    if tokens<target_tokens:
        del state;pending.unlink();return None
    metadata=dict(complete=True,source=str(source),checkpoint=str(checkpoint),step=int(state['step']),
                  trained_tokens=tokens,token_equivalent_step=tokens//32768,target_tokens=target_tokens,
                  bytes=pending.stat().st_size,model_config=json.loads(state['model_config']),
                  captured_at=time.time())
    del state
    os.replace(pending,checkpoint)
    write_json(report,metadata)
    return metadata


def stop_for_scheduled_cos(record, source, snapshot):
    """Stop only the identified pretraining process, after an immutable save."""
    import psutil
    if not snapshot.get('complete') or not Path(snapshot['checkpoint']).is_file():
        raise ValueError('Scheduled Cos requires a verified saved checkpoint')
    if not process_live(record):return dict(stopped=False,reason='original_process_already_terminal',**record)
    process=psutil.Process(record['pid']);command=process.cmdline()
    if 'training.train_pretrain' not in ' '.join(command) or '--out' not in command:
        raise ValueError('Refusing to stop an unrelated process for Cos')
    run=Path(command[command.index('--out')+1])
    if not run.is_absolute():run=Path(process.cwd())/run
    if run.resolve()!=Path(source).resolve().parent:
        raise ValueError('Pretraining output differs from the captured source')
    process.terminate();process.wait(timeout=15)
    result=dict(stopped=True,reason='scheduled_cos_after_saved_checkpoint',**record,stopped_at=time.time(),
                source_step=snapshot['step'],token_equivalent_step=snapshot['token_equivalent_step'])
    paused=run/'paused.json';previous=json.loads(paused.read_text(encoding='utf-8')) if paused.exists() else {}
    previous.update(status='paused_for_scheduled_cos',**result)
    write_json(paused,previous)
    return result


def watch_archive(source,log,destination,target_tokens,poll_seconds=15,stop_process=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    lock=destination/'watcher.lock'
    import psutil
    if lock.exists():
        previous=json.loads(lock.read_text())
        try:
            process=psutil.Process(previous['pid'])
            if abs(process.create_time()-previous['created_at'])<1 and process.is_running():
                raise RuntimeError('An archive watcher is already live')
        except psutil.NoSuchProcess:pass
    lease=dict(pid=os.getpid(),created_at=psutil.Process().create_time())
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL) if not lock.exists() else None
    if descriptor is None:lock.unlink();descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(descriptor,'w') as stream:json.dump(lease,stream)
    print(json.dumps(dict(phase='watching_checkpoint',**lease,target_tokens=target_tokens)),flush=True)
    try:
        while True:
            record=latest_record(log)
            if record.get('tokens',0)>=target_tokens and Path(source).exists():
                result=capture_checkpoint(source,destination,target_tokens)
                if result:
                    if stop_process:
                        stopped=stop_for_scheduled_cos(stop_process,source,result)
                        write_json(destination/'pretrain-stop.json',stopped)
                    write_json(destination/'watcher-status.json',dict(status='complete',**result))
                    print(json.dumps(result),flush=True);return result
            write_json(destination/'watcher-status.json',dict(status='waiting',**lease,target_tokens=target_tokens,
                       observed_tokens=record.get('tokens'),last_checked=time.time()))
            time.sleep(poll_seconds)
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('pid')==os.getpid():lock.unlink()


def process_live(record):
    import psutil
    try:
        process=psutil.Process(record['pid'])
        return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE and abs(process.create_time()-record['created_at'])<1
    except psutil.NoSuchProcess:return False


def exclusive_lock(path):
    handle=open(path,'a+b');handle.seek(0)
    if handle.read(1)==b'':handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('Another pipeline owns the process lock')
    return handle


def service_ready(spec):
    import urllib.request
    import urllib.parse
    url=urllib.parse.urlparse(spec['ready_url'])
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost'):
        raise ValueError('Service health checks must use loopback HTTP')
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(spec['ready_url'],timeout=2) as response:
            value=json.load(response)
        if spec.get('model_id'):
            return any(v.get('id')==spec['model_id'] for v in value.get('data',[]))
        for key in spec.get('ready_field',['ready']):value=value[key]
        return value==spec.get('ready_value',True)
    except (OSError,ValueError,KeyError,TypeError):return False


def start_service(spec_path):
    """Start or observe the same daemon; a startup timeout never implies death."""
    import psutil
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    directory=Path(spec['state_directory']);directory.mkdir(parents=True,exist_ok=True)
    lock=exclusive_lock(directory/'startup.lock')
    lease_path=directory/'process.json';process=None
    signature=hashlib.sha256(json.dumps([spec['command'],spec.get('environment',{})],sort_keys=True).encode()).hexdigest()
    try:
        for pid,owned in list(_service_processes.items()):
            if owned.poll() is not None:owned.wait();_service_processes.pop(pid)
        if lease_path.exists():
            previous=json.loads(lease_path.read_text())
            if process_live(previous):
                if previous['signature']!=signature:raise RuntimeError('A live service uses a different configuration')
                record=previous
            else:
                raise RuntimeError('Prior service is terminal; inspect its logs and archive its attempt before restarting')
        else:
            if service_ready(spec):raise RuntimeError('Service endpoint is already occupied without an owned process')
            env=dict(os.environ,**{str(k):str(v) for k,v in spec.get('environment',{}).items()})
            flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
            with (directory/'stdout.log').open('a') as stdout,(directory/'stderr.log').open('a') as stderr:
                process=subprocess.Popen(spec['command'],cwd=spec['workspace'],env=env,stdout=stdout,stderr=stderr,**flags)
            _service_processes[process.pid]=process
            record=dict(pid=process.pid,created_at=psutil.Process(process.pid).create_time(),command=spec['command'],signature=signature)
            write_json(lease_path,record)
        started=time.monotonic()
        while True:
            if not process_live(record):
                code=process.poll() if process is not None else None
                write_json(directory/'status.json',dict(phase='failed',exit_code=code,checked_at=time.time(),process=record))
                raise RuntimeError('Service process exited before readiness')
            if service_ready(spec):
                result=dict(phase='ready',checked_at=time.time(),process=record,ready_url=spec['ready_url'])
                write_json(directory/'status.json',result);print(json.dumps(result),flush=True);return result
            write_json(directory/'status.json',dict(phase='starting',checked_at=time.time(),process=record))
            if time.monotonic()-started>spec.get('startup_timeout_seconds',1200):
                raise TimeoutError('Service remains live but has not reported ready; inspect this same process')
            time.sleep(2)
    finally:lock.close()


def stop_service(spec_path):
    import psutil
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'));directory=Path(spec['state_directory'])
    record=json.loads((directory/'process.json').read_text())
    if process_live(record):
        process=psutil.Process(record['pid'])
        if process.cmdline()!=record['command']:raise RuntimeError('Service command no longer matches its owned process')
        children=process.children(recursive=True)
        process.terminate()
        for child in children:
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        gone,alive=psutil.wait_procs([process,*children],timeout=10)
        for child in alive:child.kill()
        psutil.wait_procs(alive,timeout=5)
    owned=_service_processes.pop(record['pid'],None)
    if owned is not None:owned.wait(timeout=5)
    result=dict(phase='stopped',checked_at=time.time(),process=record)
    write_json(directory/'status.json',result);return result


def condition_ready(condition):
    kind=condition['kind']
    if kind=='file':return Path(condition['path']).exists()
    if kind=='service':return service_ready(condition)
    if kind=='tokens':return latest_record(condition['path']).get('tokens',0)>=condition['minimum']
    if kind=='process_stopped':return not process_live(condition)
    if kind=='json':
        path=Path(condition['path'])
        if not path.exists():return False
        value=json.loads(path.read_text(encoding='utf-8'))
        for key in condition.get('field',[]):value=value[key]
        if 'minimum' in condition:return value>=condition['minimum']
        return value==condition['equals']
    if kind=='checkpoint':
        path=Path(condition['path'])
        if not path.exists():return False
        import torch
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        return int(state['step'])>=condition['minimum_step'] and bool(state.get('model'))
    if kind=='checksum_manifest':
        path=Path(condition['path']);root=Path(condition['root']).resolve()
        if not path.exists():return False
        records=json.loads(path.read_text(encoding='utf-8'))
        if not records:return False
        for record in records:
            file=(root/record['path']).resolve()
            if not file.is_relative_to(root):raise ValueError('Checksum manifest escapes its data root')
            if not file.exists() or file.stat().st_size!=record['bytes']:return False
        for record in records:
            digest=hashlib.sha256()
            with (root/record['path']).open('rb') as stream:
                while block:=stream.read(1024*1024):digest.update(block)
            if digest.hexdigest()!=record['sha256']:raise ValueError(f"Data checksum mismatch: {record['path']}")
        return True
    raise ValueError(f'Unknown pipeline condition: {kind}')


def anneal_from_snapshot(source, data, out, tokens, batch_size, grad_accum=1, training_args=(),lr_reference=None,round_index=0):
    """Keep source moments and actual scheduled LR; reset the new corpus position."""
    import torch
    from .trainer_utils import load_checkpoint, lr_scale
    from .train_pretrain import parse_args, train, COOLDOWN_OPTIMIZER, schedule_of
    from argparse import Namespace
    state=load_checkpoint(Path(source),torch.device('cpu'),mmap=True)
    config=state['train_config'];progress=state.get('training_progress') or {}
    seq_len=progress.get('seq_len',config['seq_len'])
    rows=progress.get('consumed_rows',state['step']*config['batch_size']*config.get('grad_accum',1))
    reference=progress.get('schedule_reference_rows',config['batch_size']*config.get('grad_accum',1))
    total=progress.get('schedule_total_steps',state.get('total_steps'))
    if not total:raise ValueError('Source checkpoint does not identify the original LR schedule')
    rate_state=load_checkpoint(Path(lr_reference),torch.device('cpu'),mmap=True) if lr_reference else state
    rate_config=rate_state['train_config'];rate_progress=rate_state.get('training_progress') or {}
    rate_rows=rate_progress.get('consumed_rows',rate_state['step']*rate_config['batch_size']*rate_config.get('grad_accum',1))
    rate_reference=rate_progress.get('schedule_reference_rows',rate_config['batch_size']*rate_config.get('grad_accum',1))
    rate_total=rate_progress.get('schedule_total_steps',rate_state.get('total_steps'))
    if not rate_total:raise ValueError('LR reference lacks its schedule horizon')
    warmup=rate_progress.get('schedule_warmup_steps',rate_config.get('warmup_steps',100))
    schedule_config=dict(rate_config,warmup_steps=warmup)
    scale=schedule_of(Namespace(**schedule_config),rate_total)(rate_rows/rate_reference)
    initial_lr_mult=rate_config.get('lr_mult',1.)*scale
    if initial_lr_mult<=0:raise ValueError('Completed cosine schedule has zero LR; supply the immutable pre-Cos --lr-reference for another low-LR segment')
    rows_per_step=batch_size*grad_accum
    if tokens%(rows_per_step*seq_len):raise ValueError('Anneal token budget must be an exact number of updates')
    steps=tokens//(rows_per_step*seq_len)
    argv=['--init',str(source),'--init-optimizer-state','--data',str(data),'--out',str(out),
          '--seq-len',str(seq_len),'--batch-size',str(batch_size),'--grad-accum',str(grad_accum),
          '--steps',str(steps),'--warmup-steps','0','--final-lr-frac','0','--schedule','cosine']
    for key in COOLDOWN_OPTIMIZER:
        if key in config and key!='lr_mult':
            value=config[key];flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:argv.append(flag)
            else:argv.extend([flag,str(value)])
    argv+=['--lr-mult',str(initial_lr_mult),'--seed',str(rate_config.get('seed',0)+round_index),*training_args]
    previous_lineage=Path(source).parent/'lineage.json'
    cumulative_tokens=rows*seq_len
    if previous_lineage.exists():cumulative_tokens+=json.loads(previous_lineage.read_text(encoding='utf-8'))['source_tokens']
    source_step=int(state['step']);del state,rate_state
    args=parse_args(argv)
    # Avoid accidentally interpreting a command-tail override as a fresh restart.
    if args.init!=str(source) or not args.init_optimizer_state or args.steps!=steps:
        raise ValueError('Anneal command tail overrides its lineage or token budget')
    write_json(Path(out)/'lineage.json',dict(source=str(source),source_step=source_step,
               source_tokens=cumulative_tokens,source_stage_tokens=rows*seq_len,source_reference_step=cumulative_tokens/32768,source_lr_scale=scale,
               lr_reference=str(lr_reference or source),initial_lr_mult=initial_lr_mult,round_index=round_index,
               restored_optimizer_state=True,new_corpus_position=0,anneal_tokens=tokens,anneal_updates=steps,
               planned_total_training_tokens=cumulative_tokens+tokens))
    return train(args)


def stage_worker(plan_path,stage_id):
    import psutil
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    stage=next(s for s in plan['stages'] if s['id']==stage_id)
    directory=Path(plan['state_directory'])/stage_id;directory.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update({str(k):str(v) for k,v in plan.get('environment',{}).items()})
    env.update({str(k):str(v) for k,v in stage.get('environment',{}).items()})
    flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
    with (directory/'stdout.log').open('a',encoding='utf-8') as stdout,(directory/'stderr.log').open('a',encoding='utf-8') as stderr:
        process=subprocess.Popen(stage['command'],cwd=plan['workspace'],env=env,stdout=stdout,stderr=stderr,**flags)
        record=dict(pid=process.pid,created_at=psutil.Process(process.pid).create_time(),command=stage['command'])
        write_json(directory/'child.json',record)
        bridge=None
        monitor=None
        monitor_spec=stage.get('checkpoint_evaluation')
        snapshot=directory/'first-checkpoint.pt'
        monitor_stdout=monitor_stderr=None
        while True:
            code=process.poll()
            run_directory=Path(stage.get('run_directory',''))
            if stage.get('tensorboard') and bridge is None and (run_directory/'config.json').exists():
                from tools.tensorboard import MetricBridge
                bridge=MetricBridge(run_directory,stage.get('tensorboard_directory'))
            if bridge is not None:bridge.poll()
            if monitor_spec and monitor is None and Path(monitor_spec['checkpoint']).exists():
                os.link(monitor_spec['checkpoint'],snapshot)
                command=[sys.executable,'-u','-m','tools.eval_posttrain','--checkpoint',str(snapshot),
                         '--out',str(directory/'first-checkpoint-eval.json'),*monitor_spec['arguments']]
                monitor_stdout=(directory/'first-checkpoint-eval.stdout.log').open('a')
                monitor_stderr=(directory/'first-checkpoint-eval.stderr.log').open('a')
                monitor_env=dict(env,CUDA_VISIBLE_DEVICES='')
                monitor=subprocess.Popen(command,cwd=plan['workspace'],env=monitor_env,
                                         stdout=monitor_stdout,stderr=monitor_stderr,**flags)
                write_json(directory/'evaluation-child.json',dict(pid=monitor.pid,created_at=psutil.Process(monitor.pid).create_time(),command=command))
            if code is not None:break
            time.sleep(2)
        if bridge is not None:bridge.close()
        evaluation_code=monitor.wait() if monitor is not None else None
        if monitor_stdout is not None:monitor_stdout.close();monitor_stderr.close()
        if evaluation_code==0:snapshot.unlink()
        if code==0 and monitor_spec and evaluation_code!=0:code=1
    write_json(directory/'result.json',dict(exit_code=code,finished_at=time.time(),child=record,
                                           checkpoint_evaluation_exit_code=evaluation_code))
    return code


def run_pipeline(plan_path,poll_seconds=15):
    """Run sequential stages; recover live workers rather than launching duplicates."""
    import psutil
    import shutil
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text(encoding='utf-8'))
    root=Path(plan['state_directory']);root.mkdir(parents=True,exist_ok=True)
    if not plan.get('armed'):raise ValueError('Pipeline plan is not armed')
    process_lock=exclusive_lock(root/'coordinator.lock')
    lease_path=root/'coordinator.json'
    if lease_path.exists() and process_live(json.loads(lease_path.read_text())):
        raise RuntimeError('A pipeline coordinator is already live')
    lease=dict(pid=os.getpid(),created_at=psutil.Process().create_time())
    write_json(lease_path,lease)
    status_path=root/'status.json'
    owned_workers={}
    try:
        for stage in plan['stages']:
            directory=root/stage['id'];directory.mkdir(exist_ok=True)
            complete=directory/'complete.json';result_path=directory/'result.json';worker_path=directory/'worker.json'
            if complete.exists():
                if not all(condition_ready(c) for c in stage.get('completion',[])):
                    raise RuntimeError(f"Completed stage no longer satisfies its conditions: {stage['id']}")
                continue
            while not all(condition_ready(c) for c in stage.get('requires',[])):
                write_json(status_path,dict(phase='waiting',stage=stage['id'],checked_at=time.time()))
                time.sleep(poll_seconds)
            if not worker_path.exists():
                required_disk=stage.get('minimum_free_disk_gib',plan.get('minimum_free_disk_gib',12))
                while shutil.disk_usage(plan['workspace']).free < required_disk*2**30:
                    write_json(status_path,dict(phase='waiting_for_disk',stage=stage['id'],checked_at=time.time()))
                    time.sleep(poll_seconds)
                command=[sys.executable,'-u','-m','training.pipeline','worker','--plan',str(plan_path),'--stage',stage['id']]
                flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
                with (directory/'worker.stdout.log').open('a') as stdout,(directory/'worker.stderr.log').open('a') as stderr:
                    process=subprocess.Popen(command,cwd=plan['workspace'],stdout=stdout,stderr=stderr,**flags)
                owned_workers[stage['id']]=process
                write_json(worker_path,dict(pid=process.pid,created_at=psutil.Process(process.pid).create_time(),command=command))
            while not result_path.exists():
                worker=json.loads(worker_path.read_text())
                if not process_live(worker):
                    raise RuntimeError(f"Stage worker is terminal without a result; inspect before retry: {stage['id']}")
                write_json(status_path,dict(phase='running',stage=stage['id'],worker=worker,checked_at=time.time()))
                time.sleep(poll_seconds)
            result=json.loads(result_path.read_text())
            if stage['id'] in owned_workers:
                owned_workers[stage['id']].wait(timeout=10)
            if result['exit_code']!=0:raise RuntimeError(f"Stage failed with exit {result['exit_code']}: {stage['id']}")
            if not all(condition_ready(c) for c in stage.get('completion',[])):
                raise RuntimeError(f"Stage exited successfully but validation failed: {stage['id']}")
            write_json(complete,dict(stage=stage['id'],finished_at=time.time(),result=result))
        write_json(status_path,dict(phase='all_configured_stages_finished',finished_at=time.time(),
                                   accepted=False,note='Independent capability acceptance is still required.'))
    except BaseException as error:
        write_json(status_path,dict(phase='failed',error=str(error),failed_at=time.time()))
        raise
    finally:
        if lease_path.exists() and json.loads(lease_path.read_text()).get('pid')==os.getpid():lease_path.unlink()
        process_lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    archive=commands.add_parser('archive')
    archive.add_argument('--source',required=True);archive.add_argument('--log',required=True)
    archive.add_argument('--out',required=True);archive.add_argument('--target-tokens',type=int,required=True)
    archive.add_argument('--poll-seconds',type=float,default=15)
    archive.add_argument('--stop-pid',type=int);archive.add_argument('--stop-created-at',type=float)
    run=commands.add_parser('run');run.add_argument('--plan',required=True);run.add_argument('--poll-seconds',type=float,default=15)
    worker=commands.add_parser('worker');worker.add_argument('--plan',required=True);worker.add_argument('--stage',required=True)
    service=commands.add_parser('service');service.add_argument('--spec',required=True);service.add_argument('--stop',action='store_true')
    anneal=commands.add_parser('anneal')
    anneal.add_argument('--source',required=True);anneal.add_argument('--data',required=True);anneal.add_argument('--out',required=True)
    anneal.add_argument('--tokens',type=int,default=655360000);anneal.add_argument('--batch-size',type=int,required=True)
    anneal.add_argument('--grad-accum',type=int,default=1);anneal.add_argument('training_args',nargs=argparse.REMAINDER)
    anneal.add_argument('--lr-reference',help='Frozen pre-Cos checkpoint supplying the next segment starting LR')
    anneal.add_argument('--round-index',type=int,default=0,help='Use a different deterministic data shuffle in each continuation')
    args=parser.parse_args(argv)
    if args.command=='worker':return stage_worker(args.plan,args.stage)
    if args.command=='service':return stop_service(args.spec) if args.stop else start_service(args.spec)
    if args.command=='anneal':
        tail=args.training_args[1:] if args.training_args[:1]==['--'] else args.training_args
        return anneal_from_snapshot(args.source,args.data,args.out,args.tokens,args.batch_size,args.grad_accum,tail,args.lr_reference,args.round_index)
    if args.poll_seconds<=0:parser.error('Counts must be positive')
    if args.command=='run':return run_pipeline(args.plan,args.poll_seconds)
    if args.target_tokens<=0:parser.error('Counts must be positive')
    if (args.stop_pid is None)!=(args.stop_created_at is None):parser.error('Both stop process identity fields are required')
    identity=dict(pid=args.stop_pid,created_at=args.stop_created_at) if args.stop_pid is not None else None
    return watch_archive(args.source,args.log,args.out,args.target_tokens,args.poll_seconds,identity)


if __name__=='__main__':
    result=main()
    if isinstance(result,int):raise SystemExit(result)
