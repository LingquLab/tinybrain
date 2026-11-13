"""在一个单独的进程里跑一段 Python：新建的临时目录、限时、限内存、隔离模式（不读当前目录里的模块）。

这只是基本的隔离，不能防住存心捣乱的代码。跑模型写的代码时，整个训练最好放在容器里。
"""
import os
import subprocess
import sys
import tempfile

try:
    import resource
except ImportError:  # Windows 没有
    resource = None


def _limits(memory_mb: int, cpu_seconds: int):
    def apply():
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
        except (ValueError, OSError):  # macOS 不支持限制地址空间
            pass
    return apply


def run_python(code: str, timeout: float = 10.0, memory_mb: int = 1024, stdin: str = "") -> tuple[bool, str]:
    """返回 (是否正常结束, 输出)。输出是标准输出加标准错误，最多保留 4000 字。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run([sys.executable, "-I", path], input=stdin, capture_output=True, text=True, timeout=timeout, cwd=tmp,
                                  preexec_fn=_limits(memory_mb, int(timeout) + 1) if os.name == "posix" else None,
                                  env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return False, "超时"
    if proc.returncode == 0:
        return True, proc.stdout[-4000:]           # 正常结束时只看标准输出（按输入输出判分时要逐字比较）
    return False, (proc.stdout + proc.stderr)[-4000:]
