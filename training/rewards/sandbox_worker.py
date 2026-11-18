"""Trusted Linux bootstrap: restrict the process before executing candidate code."""
import ctypes
import ctypes.util
import errno
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import sysconfig


class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


class CapHeader(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]


class CapData(ctypes.Structure):
    _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]


def checked(result, operation):
    if result < 0: raise OSError(ctypes.get_errno(), operation)
    return result


def restrict(memory_mb, cpu_seconds):
    architecture = platform.machine()
    if architecture not in ('x86_64', 'aarch64'): raise RuntimeError('Unsupported Landlock architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    library = ctypes.util.find_library('seccomp')
    if not library: raise RuntimeError('libseccomp is required')
    seccomp = ctypes.CDLL(library, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]; seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    abi = checked(libc.syscall(444, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1)), 'Landlock unavailable')
    if abi < 1: raise RuntimeError('Landlock ABI 1 or later is required')
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    checked(libc.prctl(38, 1, 0, 0, 0), 'PR_SET_NO_NEW_PRIVS')
    checked(libc.prctl(4, 0, 0, 0, 0), 'PR_SET_DUMPABLE')
    handled = (1 << 13) - 1
    if abi >= 2: handled |= 1 << 13
    if abi >= 3: handled |= 1 << 14
    if abi >= 5: handled |= 1 << 15
    ruleset = Ruleset(handled)
    descriptor = checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0), 'create Landlock ruleset')
    paths = [sysconfig.get_path('stdlib'), sysconfig.get_config_var('LIBDIR'), '/lib', '/lib64',
             '/usr/lib', '/usr/lib64', '/etc/ld.so.cache', '/etc/localtime', '/usr/share/zoneinfo']
    try:
        for path in sorted(set(str(Path(p).resolve()) for p in paths if p and Path(p).exists())):
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule((1 | 4 | 8) if Path(path).is_dir() else 4, fd)
                checked(libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0), 'add Landlock path')
            finally: os.close(fd)
        checked(libc.syscall(446, descriptor, 0), 'restrict Landlock process')
    finally: os.close(descriptor)
    # Keeping UID permits stdlib imports from an interpreter installed under /root.
    # Clear every capability; no_new_privs and seccomp forbid regaining privilege.
    header = CapHeader(0x20080522, 0); capabilities = (CapData * 2)()
    checked(libc.syscall(126 if architecture == 'x86_64' else 91, ctypes.byref(header), ctypes.byref(capabilities)), 'clear capabilities')
    context = seccomp.seccomp_init(0x00050000 | errno.EPERM)
    if not context: raise RuntimeError('Cannot create seccomp filter')
    allowed = '''read write readv writev close close_range fstat newfstatat stat lstat statx lseek
        readlink readlinkat getdents64 access faccessat faccessat2 open openat openat2
        mmap mmap2 mprotect munmap brk mremap madvise rt_sigaction rt_sigprocmask rt_sigreturn
        sigaltstack clock_gettime clock_gettime64 clock_nanosleep nanosleep gettimeofday time
        getpid getppid gettid getuid geteuid getgid getegid getresuid getresgid getgroups getcwd
        fcntl fcntl64 dup dup2 dup3 poll ppoll select pselect6 epoll_create1 epoll_ctl epoll_wait
        futex futex_time64 getrandom uname sysinfo sched_getaffinity sched_yield getrusage
        getrlimit ugetrlimit arch_prctl set_tid_address set_robust_list rseq capget exit exit_group'''.split()
    try:
        for name in allowed:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0:
                result = seccomp.seccomp_rule_add(context, 0x7fff0000, number, 0)
                if result < 0: raise RuntimeError(f'Cannot allow syscall {name}: {result}')
        if seccomp.seccomp_load(context) < 0: raise RuntimeError('Cannot load seccomp filter')
    finally: seccomp.seccomp_release(context)
    return dict(landlock_abi=abi, capabilities_cleared=True, filesystem='runtime-read-only',
                network=False, subprocesses=False, writable_paths=False, output_limit_bytes=65536)


def main():
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        profile = restrict(int(payload['memory_mb']), max(1, math.ceil(payload['timeout'])))
    except Exception as error:
        print('SANDBOX_BOOTSTRAP_ERROR '+repr(error),file=sys.stderr)
        raise SystemExit(125)
    if payload.get('profile_only'): print(json.dumps(profile)); return
    source = compile(payload['code'], '<candidate>', 'exec')
    exec(source, {'__name__': '__main__', '__builtins__': __builtins__})


if __name__ == '__main__': main()
