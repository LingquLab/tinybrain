"""Bounded Python execution with Linux Landlock/seccomp or a loopback remote service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request


def _run_local(code, timeout, memory_mb, stdin, profile_only=False):
    if sys.platform != 'linux':
        raise RuntimeError('Python verification requires Linux Landlock/seccomp or TINYBRAIN_SANDBOX_URL')
    worker = Path(__file__).with_name('sandbox_worker.py')
    payload = dict(code=code, timeout=timeout, memory_mb=memory_mb, profile_only=profile_only)
    with tempfile.TemporaryDirectory(prefix='tinybrain-python-') as tmp:
        source=Path(tmp)/'payload.json'; source.write_text(json.dumps(payload),encoding='utf-8')
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process=subprocess.Popen([sys.executable,'-I','-S','-B',str(worker),str(source)],
                stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,cwd=tmp,start_new_session=True,
                env={'PATH':'/usr/bin:/bin','PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'1'},close_fds=True)
            try: process.communicate(stdin.encode('utf-8'),timeout=timeout+2)
            except subprocess.TimeoutExpired:
                process.kill();process.communicate();return False,'timeout'
            def output(stream):
                stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-4000))
                return stream.read(4000).decode('utf-8',errors='replace')
            result,errors=output(stdout),output(stderr)
    if process.returncode==125 and errors.startswith('SANDBOX_BOOTSTRAP_ERROR '):
        raise RuntimeError('Python isolation bootstrap failed: '+errors)
    if process.returncode!=0: return False,(result+errors)[-4000:]
    return True,result


def run_python(code: str, timeout: float = 10., memory_mb: int = 1024, stdin: str = '') -> tuple[bool,str]:
    if not 0 < timeout <= 30 or not 64 <= memory_mb <= 2048:
        raise ValueError('Invalid Python execution resource limit')
    if len(code.encode('utf-8'))>131072 or len(stdin.encode('utf-8'))>131072:
        return False,'input limit exceeded'
    remote=os.environ.get('TINYBRAIN_SANDBOX_URL','').rstrip('/')
    if remote:
        request=urllib.request.Request(remote+'/run-python',data=json.dumps(dict(code=code,timeout=timeout,memory_mb=memory_mb,stdin=stdin)).encode(),
                                       headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=timeout+10) as response:result=json.load(response)
        if result.get('infrastructure_error'):raise RuntimeError(result['infrastructure_error'])
        return bool(result['ok']),str(result['output'])
    return _run_local(code,timeout,memory_mb,stdin)


def serve(host='127.0.0.1',port=8081):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    if host!='127.0.0.1':raise ValueError('Python verifier service binds only loopback')
    ok,profile=_run_local('',5,512,'',profile_only=True)
    if not ok:raise RuntimeError('Python isolation bootstrap failed: '+profile)
    profile=json.loads(profile);slots=threading.BoundedSemaphore(4)
    class Handler(BaseHTTPRequestHandler):
        def answer(self,status,value):
            body=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.answer(200,dict(ready=True,profile=profile)) if self.path=='/health' else self.answer(404,{})
        def do_POST(self):
            if self.path!='/run-python':self.answer(404,{});return
            length=int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 1048576:self.answer(413,{});return
            try:
                value=json.loads(self.rfile.read(length));code=str(value['code']);stdin=str(value.get('stdin',''))
                timeout=float(value.get('timeout',10));memory=int(value.get('memory_mb',1024))
                if not 0 < timeout <= 30 or not 64 <= memory <= 2048 or len(code.encode())>131072 or len(stdin.encode())>131072:
                    raise ValueError('Invalid resource/input limit')
            except (ValueError,KeyError,TypeError):self.answer(400,{});return
            try:
                with slots:ok,output=_run_local(code,timeout,memory,stdin)
                self.answer(200,dict(ok=ok,output=output))
            except Exception as error:self.answer(503,dict(infrastructure_error=type(error).__name__))
        def log_message(self,*args):pass
    print(json.dumps(dict(ready=True,host=host,port=port,profile=profile)),flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--serve',action='store_true');parser.add_argument('--port',type=int,default=8081)
    args=parser.parse_args(argv)
    if args.serve:return serve(port=args.port)
    ok,result=_run_local('',5,512,'',profile_only=True);print(result)
    if not ok:raise SystemExit(1)


if __name__=='__main__':main()
