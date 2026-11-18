"""Prepare a fixed token-budget annealing mix with bounded process memory.

    python -m tools.prepare_anneal --plan out/small/anneal/plan.json

The plan describes a packed dialogue directory, a base Parquet/token directory,
and explicit text-selection rules. Existing training and validation splits are
preserved. Output rows contain real tokens only; every target is supervised.
"""
import argparse
import bisect
import collections
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import psutil


def limit_memory(gib):
    """Limit committed private memory on Windows, before importing Arrow/Torch."""
    if os.name != "nt":
        raise RuntimeError("This memory-limit implementation requires Windows")
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", w.DWORD), ("scheduling", w.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                    ("peak_job", ctypes.c_size_t)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.CreateJobObjectW.restype = w.HANDLE
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.SetInformationJobObject.restype = w.BOOL
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    k.AssignProcessToJobObject.restype = w.BOOL
    k.GetCurrentProcess.restype = w.HANDLE
    handle = k.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    info = Extended()
    info.basic.flags = 0x300  # PROCESS_MEMORY | JOB_MEMORY (includes child compilers)
    info.process_memory = int(gib * 2**30)
    info.job_memory = info.process_memory
    if not k.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(handle, k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    # Keep the handle alive for this process. There is no kill-on-close flag.
    return k, handle


def check_memory(min_available_gib):
    available = psutil.virtual_memory().available / 2**30
    if available < min_available_gib:
        raise MemoryError(f"Only {available:.2f} GiB available; stopping preparation")


def wait_memory(min_available_gib):
    """Pause bounded preparation buffers during transient training peaks."""
    last_report = 0
    while psutil.virtual_memory().available < min_available_gib * 2**30:
        if time.monotonic() - last_report >= 20:
            print(json.dumps(dict(phase='waiting_for_memory', minimum_available_gib=min_available_gib)), flush=True)
            last_report = time.monotonic()
        time.sleep(2)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_dialogue(source, target, min_available_gib):
    """Remove SFT padding in 8 MiB reads; keep both prompt and answer tokens."""
    from inference.tokenizer import PAD
    from training.dataset import ShardWriter, write_meta

    source, target = Path(source), Path(target)
    meta = read_json(source / "meta.json")
    if meta.get("format") != "packed":
        raise ValueError("Dialogue input must be an existing packed SFT directory")
    target.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        writer = ShardWriter(target, split, meta["dtype"])
        try:
            with open(source / f"{split}.ids.bin", "rb") as stream:
                while block := stream.read(8 * 2**20):
                    check_memory(min_available_gib)
                    ids = np.frombuffer(block, dtype=meta["dtype"])
                    writer.write(ids[ids != PAD])
        finally:
            writer.close()
        stats[split] = writer.total
        expected = meta.get("stats", {}).get(split, {}).get("tokens")
        if expected is not None and writer.total != expected:
            raise ValueError(f"Unpacked {split} token count {writer.total} != {expected}")
    shutil.copyfile(source / "tokenizer.model", target / "tokenizer.model")
    write_meta(target, meta["vocab_size"], meta["dtype"], meta["eos_id"], meta["tokenizer"],
               tokenizer_file="tokenizer.model", train_on="all", source=str(source), stats=stats)
    return stats


def matches(text, rule):
    if not rule["min_chars"] <= len(text) <= rule["max_chars"]:
        return False
    if rule.get("title_include"):
        lines = text.lstrip().splitlines()
        if not lines or not re.search(rule["title_include"], lines[0], re.I):
            return False
    prefix = text[:rule.get("prefix_chars", 1200)]
    if re.search(rule["exclude"], prefix, re.I):
        return False
    found = re.findall(rule["include"], prefix, re.I)
    if rule.get("distinct_matches"):
        found = set(x.lower() for x in found)
    return len(found) >= rule.get("min_matches", 1)


def select_texts(parquet, target, tokenizer, rules, min_available_gib, val_every=1000):
    """Scan small Arrow batches without parallel decoding or dataset prefetch."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tools.prepare import load_encoder
    from training.dataset import ShardWriter, write_meta

    pa.set_memory_pool(pa.system_memory_pool())
    encode, vocab, eos, kind = load_encoder(str(tokenizer), threads=4)
    dtype = "uint16" if vocab <= 65536 else "uint32"
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    writers = {s: ShardWriter(target, s, dtype) for s in ("train", "val")}
    counts = {r["name"]: {"train": 0, "val": 0, "docs": 0} for r in rules}
    sources = {s: r for r in rules for s in r["sources"]}
    selected_ids = open(target / "selected-documents.jsonl", "w", encoding="utf-8")
    samples = {r["name"]: [] for r in rules}
    n, last_report = 0, time.monotonic()
    try:
        with pq.ParquetFile(parquet, pre_buffer=False) as table:
            for batch in table.iter_batches(batch_size=128, columns=["text", "source", "quality"], use_threads=False):
                check_memory(min_available_gib)
                pending = []
                # Convert at most this batch, never a whole text column/table.
                for text, source, quality in zip(*(batch.column(k).to_pylist() for k in ("text", "source", "quality"))):
                    row = n
                    n += 1
                    rule = sources.get(source)
                    if rule is None or quality < rule["min_quality"]:
                        continue
                    split = "val" if val_every and row % val_every == val_every - 1 else "train"
                    if counts[rule["name"]][split] >= rule[f"max_{split}_tokens"]:
                        continue
                    if matches(text, rule):
                        pending.append((text, source, row, split, rule))
                if pending:
                    for (text, source, row, split, rule), ids in zip(pending, encode([p[0] for p in pending])):
                        st = counts[rule["name"]]
                        if st[split] >= rule[f"max_{split}_tokens"]:
                            continue
                        ids = np.append(np.asarray(ids, dtype=dtype), eos)
                        writers[split].write(ids)
                        st[split] += len(ids)
                        st["docs"] += 1
                        selected_ids.write(json.dumps(dict(row=row, source=source, split=split, tokens=len(ids))) + "\n")
                        if len(samples[rule["name"]]) < 10:
                            samples[rule["name"]].append(dict(source=source, row=row, text=text[:1600]))
                if time.monotonic() - last_report >= 20:
                    m = psutil.Process().memory_info()
                    print(json.dumps(dict(phase="select", scanned=n, counts=counts,
                                          rss_mib=round(m.rss / 2**20), private_mib=round(getattr(m, "private", m.rss) / 2**20))), flush=True)
                    last_report = time.monotonic()
                if all(counts[r["name"]][s] >= r[f"max_{s}_tokens"] for r in rules for s in ("train", "val")):
                    break
    finally:
        selected_ids.close()
        for writer in writers.values():
            writer.close()
    if not writers["train"].total or not writers["val"].total:
        raise ValueError("Selection produced an empty split")
    shutil.copyfile(tokenizer, target / "tokenizer.model")
    stats = dict(scanned=n, counts=counts, train=writers["train"].total, val=writers["val"].total)
    write_meta(target, vocab, dtype, eos, kind, tokenizer_file="tokenizer.model", selection=rules, stats=stats,
               source=str(parquet), original_val_every=val_every)
    write_json(target / "samples.json", samples)
    return stats


class StreamWindows:
    """Bounded file reads of a shuffled token stream; no large mmap working set."""

    def __init__(self, directory, split, seq_len, seed):
        from training.dataset import EpochOrder

        directory = Path(directory)
        self.meta = read_json(directory / "meta.json")
        if self.meta.get("format") == "packed":
            raise ValueError("Mix inputs must be unpadded token streams")
        self.dtype = np.dtype(self.meta["dtype"])
        self.seq_len = seq_len
        paths = sorted(directory.glob(f"{split}-*.bin"))
        self.handles, self.ends = [], []
        count = 0
        for path in paths:
            windows = max(0, (path.stat().st_size // self.dtype.itemsize - 1) // seq_len)
            if windows:
                self.handles.append(open(path, "rb"))
                count += windows
                self.ends.append(count)
        if not count:
            self.close()
            raise ValueError(f"No {split} windows in {directory}")
        self.order, self.position = EpochOrder(count, seed), 0

    def read(self, count):
        out = np.empty((count, self.seq_len + 1), dtype=self.dtype)
        for i, window in enumerate(self.order.take(self.position, count)):
            shard = bisect.bisect_right(self.ends, window)
            offset = int(window) - (self.ends[shard - 1] if shard else 0)
            stream = self.handles[shard]
            stream.seek(offset * self.seq_len * self.dtype.itemsize)
            block = stream.read((self.seq_len + 1) * self.dtype.itemsize)
            if len(block) != (self.seq_len + 1) * self.dtype.itemsize:
                raise IOError("Short token-stream read")
            out[i] = np.frombuffer(block, dtype=self.dtype)
        self.position += count
        return out

    def close(self):
        for handle in self.handles:
            handle.close()


def mix_streams(parts, target, split_rows, seq_len, seed, min_available_gib, resume=False):
    """Write exact 3/4/3 (or other integral) row counts, with no PAD/masked labels."""
    from fractions import Fraction
    from training.dataset import write_meta

    fractions = [Fraction(str(part["weight"])) for part in parts]
    if sum(fractions) != 1:
        raise ValueError("Mix weights must sum to one")
    denominator = math.lcm(*(f.denominator for f in fractions))
    cycle = [i for i, f in enumerate(fractions) for _ in range(int(f * denominator))]
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    stats = {}
    reference = read_json(Path(parts[0]["directory"]) / "meta.json")
    tokenizer_hash = fingerprint(Path(parts[0]["directory"]) / "tokenizer.model")
    for part in parts:
        meta = read_json(Path(part["directory"]) / "meta.json")
        if (meta["vocab_size"], meta["dtype"], meta["eos_id"]) != (reference["vocab_size"], reference["dtype"], reference["eos_id"]):
            raise ValueError("Mixed streams have different token metadata")
        if fingerprint(Path(part["directory"]) / "tokenizer.model") != tokenizer_hash:
            raise ValueError("Mixed streams have different tokenizers")
    for split, rows in split_rows.items():
        if rows % denominator:
            raise ValueError("Split rows must be divisible by the mix denominator")
        streams = []
        counts = collections.Counter()
        last_report = time.monotonic()
        try:
            streams = [StreamWindows(p["directory"], split, seq_len, seed + i + (1000 if split == "val" else 0)) for i, p in enumerate(parts)]
            ids_path, mask_path = target/f'{split}.ids.bin', target/f'{split}.mask.bin'
            start_row = 0
            if resume and ids_path.exists() and mask_path.exists():
                start_row = min(ids_path.stat().st_size//((seq_len+1)*np.dtype(reference['dtype']).itemsize),
                                mask_path.stat().st_size//(seq_len+1), rows)
                start_row = start_row//denominator*denominator
                for i, stream in enumerate(streams):
                    count = int(start_row*fractions[i])
                    stream.position = count
                    counts[parts[i]['name']] = count
            with open(ids_path, 'r+b' if resume and ids_path.exists() else 'w+b') as ids, open(mask_path, 'r+b' if resume and mask_path.exists() else 'w+b') as mask:
                ids.seek(start_row*(seq_len+1)*np.dtype(reference['dtype']).itemsize);ids.truncate()
                mask.seek(start_row*(seq_len+1));mask.truncate()
                for start in range(start_row, rows, denominator * 10):
                    wait_memory(min_available_gib)
                    count = min(denominator * 10, rows - start)
                    which = np.tile(cycle, count // denominator)
                    block = np.empty((count, seq_len + 1), dtype=reference["dtype"])
                    for i, stream in enumerate(streams):
                        positions = np.flatnonzero(which == i)
                        block[positions] = stream.read(len(positions))
                        counts[parts[i]["name"]] += len(positions)
                    ids.write(block.tobytes())
                    mask.write(np.ones(block.shape, dtype=np.uint8).tobytes())
                    # Bound dirty Windows file-cache pages as well as process RAM.
                    ids.flush()
                    mask.flush()
                    os.fsync(ids.fileno())
                    os.fsync(mask.fileno())
                    if time.monotonic() - last_report >= 20:
                        print(json.dumps(dict(phase="mix", split=split, rows=start + count, total=rows)), flush=True)
                        last_report = time.monotonic()
        finally:
            for stream in streams:
                stream.close()
        stats[split] = dict(rows=rows, targets=rows * seq_len, source_rows=dict(counts),
                           source_target_tokens={name: n * seq_len for name, n in counts.items()})
    shutil.copyfile(Path(parts[0]["directory"]) / "tokenizer.model", target / "tokenizer.model")
    write_meta(target, reference["vocab_size"], reference["dtype"], reference["eos_id"], reference["tokenizer"],
               format="packed", row_len=seq_len + 1, train_on="all", tokenizer_file="tokenizer.model",
               tokenizer_sha256=tokenizer_hash, parts=parts, stats=stats)
    return stats


def prepare_plan(path):
    plan = read_json(path)
    limits = plan["preparation_limits"]
    job = limit_memory(limits["max_process_memory_gib"])
    minimum = limits["minimum_available_memory_gib"]
    check_memory(minimum)
    print(json.dumps(dict(phase="start", pid=os.getpid(), hard_private_memory_gib=limits["max_process_memory_gib"])), flush=True)
    prep = plan["preparation"]
    output = Path(prep["output"])
    if any((output / name).exists() for name in ("meta.json", "train.ids.bin")):
        raise FileExistsError(f"Refusing to overwrite {output}")
    plan["status"] = "preparing_data"
    write_json(path, plan)
    try:
        dialogue_meta = Path(prep["dialogue_stream"]) / "meta.json"
        if dialogue_meta.exists():
            previous = read_json(dialogue_meta)
            original = read_json(Path(prep["dialogue_input"]) / "meta.json")
            dialogue = previous["stats"]
            for split in ("train", "val"):
                actual = sum(p.stat().st_size for p in Path(prep["dialogue_stream"]).glob(f"{split}-*.bin")) // np.dtype(previous["dtype"]).itemsize
                if actual != dialogue[split] or actual != original["stats"][split]["tokens"]:
                    raise ValueError("Existing dialogue stream is incomplete")
            if fingerprint(Path(prep["dialogue_stream"]) / "tokenizer.model") != fingerprint(Path(prep["dialogue_input"]) / "tokenizer.model"):
                raise ValueError("Existing dialogue stream has a different tokenizer")
        else:
            dialogue = unpack_dialogue(prep["dialogue_input"], prep["dialogue_stream"], minimum)
        print(json.dumps(dict(phase="dialogue_complete", stats=dialogue)), flush=True)
        textbooks = select_texts(prep["base_parquet"], prep["selected_stream"], prep["tokenizer"], prep["selection"], minimum)
        print(json.dumps(dict(phase="selection_complete", stats=textbooks)), flush=True)
        rows = plan["cooldown_steps"] * plan["batch_size"] * plan["grad_accum"]
        stats = mix_streams(prep["parts"], output, {"train": rows, "val": prep["val_rows"]}, plan["seq_len"], prep["seed"], minimum)
        if stats["train"]["targets"] != plan["planned_tokens"]:
            raise ValueError("Prepared token budget differs from plan")
        plan.update(status="data_ready", prepared_data=prep["output"], preparation_stats=dict(dialogue=dialogue, textbooks=textbooks, mix=stats))
        write_json(path, plan)
        print(json.dumps(dict(phase="ready", data=prep["output"], stats=stats)), flush=True)
    except BaseException as exc:
        plan.update(status="data_preparation_failed", preparation_error=str(exc))
        write_json(path, plan)
        raise
    return job  # retain the process memory limit through completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    prepare_plan(parser.parse_args().plan)
