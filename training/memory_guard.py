"""Monitor physical RAM without treating brief dips as a reason to kill training.

python -m training.memory_guard --pid PID --run out/small/pretrain
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import psutil


@dataclass
class MemoryPolicy:
    reserve_gib: float = 8.0
    critical_gib: float = 2.0
    sustain_seconds: float = 60.0
    low_since: float | None = None

    def update(self, available_gib: float, now: float) -> str:
        if available_gib < self.critical_gib:
            return "critical"
        if available_gib >= self.reserve_gib:
            self.low_since = None
            return "ok"
        if self.low_since is None:
            self.low_since = now
        return "wait_checkpoint" if now - self.low_since >= self.sustain_seconds else "low"


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_stamp(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def stop_decision(action, checkpoint_before_low, checkpoint_now, write_in_progress):
    if action == "critical":
        return "critical_physical_memory"
    if (action == "wait_checkpoint" and checkpoint_now is not None
            and checkpoint_now != checkpoint_before_low and not write_in_progress):
        return "sustained_low_memory_after_checkpoint"
    return None


def monitor(pid, run, policy, poll_seconds=2):
    run = Path(run).resolve()
    process = psutil.Process(pid)
    command = process.cmdline()
    if ("training.train_pretrain" not in " ".join(command) or "--out" not in command
            or Path(command[command.index("--out") + 1]).resolve() != run):
        raise ValueError("Refusing to monitor an unrelated process")
    before_low, last_action, last_report = None, None, 0
    peak_rss, peak_private = 0, 0
    while process.is_running():
        try:
            memory = process.memory_info()
            available = psutil.virtual_memory().available / 2**30
        except psutil.NoSuchProcess:
            break
        now = time.monotonic()
        was_low = policy.low_since is not None
        action = policy.update(available, now)
        current_checkpoint = checkpoint_stamp(run / "ckpt.pt")
        if not was_low and policy.low_since is not None:
            before_low = current_checkpoint
        if action == "wait_checkpoint" and last_action != "wait_checkpoint":
            before_low = current_checkpoint
        if action == "ok":
            before_low = None
        peak_rss = max(peak_rss, memory.rss)
        private = getattr(memory, "private", memory.rss)
        peak_private = max(peak_private, private)
        reason = stop_decision(action, before_low, current_checkpoint, (run / "ckpt.tmp").exists())
        status = dict(pid=pid, timestamp_unix=time.time(), available_gib=round(available, 3),
                      rss_gib=round(memory.rss / 2**30, 3), private_commit_gib=round(private / 2**30, 3),
                      peak_observed_rss_gib=round(peak_rss / 2**30, 3),
                      peak_observed_private_commit_gib=round(peak_private / 2**30, 3),
                      action=action, low_seconds=round(now - policy.low_since, 1) if policy.low_since is not None else 0,
                      reserve_gib=policy.reserve_gib, critical_gib=policy.critical_gib,
                      sustain_seconds=policy.sustain_seconds, checkpoint=current_checkpoint)
        if reason:
            status.update(action="stopped", reason=reason)
            write_json(run / "memory-status.json", status)
            write_json(run / "memory-guard-stop.json", status)
            process.terminate()
            process.wait(15)
            paused_path = run / "paused.json"
            paused = json.loads(paused_path.read_text(encoding="utf-8")) if paused_path.exists() else {}
            paused.update(status="stopped_by_memory_guard", pid=pid, error=reason,
                          stopped_at_unix=time.time(), available_gib=available)
            write_json(paused_path, paused)
            print(json.dumps(status), flush=True)
            return
        if now - last_report >= 30 or action != last_action:
            write_json(run / "memory-status.json", status)
            print(json.dumps(status), flush=True)
            last_report = now
        last_action = action
        time.sleep(poll_seconds)
    print(json.dumps(dict(pid=pid, action="process_exited", timestamp_unix=time.time())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=8)
    parser.add_argument("--critical-gib", type=float, default=2)
    parser.add_argument("--sustain-seconds", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.critical_gib < args.reserve_gib or args.sustain_seconds <= 0:
        parser.error("Require 0 < critical-gib < reserve-gib and positive sustain-seconds")
    monitor(args.pid, args.run, MemoryPolicy(args.reserve_gib, args.critical_gib, args.sustain_seconds))


if __name__ == "__main__":
    main()
