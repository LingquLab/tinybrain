"""Build auditable, subject-separated Cos candidate pools with bounded RAM.

python -m dataset.anneal --recipe data/cos-v2/recipe.json
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
import unicodedata

import numpy as np


def canonical(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def digest(text):
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def chinese_fraction(text):
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    return chinese / max(1, chinese + latin)


def rejection(text, filters):
    for name in ("clinical", "ephemeral", "advanced", "other_model", "unverifiable_lists"):
        if re.search(filters[name], text, re.I):
            return name
    if re.search(filters["catalog"], text[:700], re.I):
        return "catalog"
    if len(re.findall(filters["boilerplate"], text, re.I)) >= 2:
        return "boilerplate"
    return None


def subject(text, source, buckets, filters):
    if not filters["min_text_chars"] <= len(text) <= filters["max_text_chars"]:
        return None
    if rejection(text, filters) or chinese_fraction(text) < filters["min_chinese_fraction"]:
        return None
    if filters.get('foundation_only') and foundation_rejection(text):
        return None
    prefix = text[:6000]
    if source != "poetry" and not re.search(filters["teaching"], prefix):
        return None
    scored = []
    for bucket in buckets:
        if bucket["kind"] != "text" or source not in bucket["sources"]:
            continue
        terms = set(re.findall(bucket["keywords"], prefix))
        if len(terms) >= 3 or source == "poetry" and bucket["name"] == "knowledge_language":
            scored.append((len(terms), bucket["name"]))
    return max(scored)[1] if scored else None


def foundation_rejection(text):
    """Exclude common non-lesson matches from the education-web candidates."""
    prefix = text[:1800]
    patterns = {
        'catalogue': r'丛书名|本书|本教材|书籍简介|出版信息|图书介绍|书名[：:]|作者[：:]|编著|主编[：:]|书评',
        'pedagogy': r'教学改革|教学策略|教学设计|教学方法|课程改革|课堂教学|教学模式|课程标准|核心素养|教研|教学研究|教学实践|教学过程|教学质量|教学活动|课题研究|教育改革|培养目标|本专业|专业培养|课程体系|就业方向|主要课程',
        'research': r'本文|本研究|摘要|内容提要|中图分类号|文献标识码|文献标志码|文章编号|研究现状|研究进展|研究综述|论文|参考文献|DOI[：:]|收稿日期|发展趋势|研究方法|半结构化访谈|研究成果|此项发现|研究团队|研究人员|最新研究',
        'commercial': r'有限公司|股份公司|供应商|制造商|经销商|厂商|产品型号|产品介绍|产品参数|市场价格|指导价|百公里油耗|马力高达|敞篷车|开蓬车|品牌推荐|营销策略|企业管理|上市企业|独角兽企业|投资回报|Sectigo',
        'specialist': r'生成对抗|神经网络|深度学习|机器学习|大语言模型|大模型|量子场论|代数拓扑|微分流形|泛函分析|张量积|光谱共焦|光纤耦合|超透镜|分子动力学模拟|蛋白质组学|单细胞测序',
        'off_topic': r'物种共产主义|三体文明|决策中心战|战争形态|作战体系|正式施行|修订《|法律法规|政策解读|实施方案|实施细则|工作部署|战略部署',
    }
    for name, pattern in patterns.items():
        if re.search(pattern, prefix, re.I):
            return name
    return None


def dialogue_bucket(messages, source):
    hint = source.lower()
    if any(k in hint for k in ("document-qa", "reading")):
        return "dialogue_reading"
    if any(k in hint for k in ("format-constrain", "instruction")):
        return "dialogue_instruction"
    if any(k in hint for k in ("math23", "reasoning", "numina")):
        return "dialogue_reasoning"
    if any(k in hint for k in ("creative", "summary", "rewrite", "editing", "literature", "文学")):
        return "dialogue_writing"
    if sum(m["role"] == "user" for m in messages) > 1:
        return "dialogue_multiturn"
    return "dialogue_basic"


def normal_messages(row):
    messages = row.get("messages") or row.get("conversations")
    if messages is None and "instruction" in row:
        question = str(row.get("instruction") or "")
        if row.get("input"):
            question += "\n\n" + str(row["input"])
        messages = [dict(role="user", content=question), dict(role="assistant", content=row.get("output") or "")]
    if not isinstance(messages, list):
        return None
    fixed = []
    for message in messages:
        role = message.get("role") or message.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = message.get("content", message.get("value"))
        if role not in ("system", "user", "assistant") or not isinstance(content, str) or not content.strip():
            return None
        if role == "system":
            continue  # Avoid inheriting arbitrary provider personas.
        fixed.append(dict(role=role, content=content.strip()))
    if not fixed or fixed[0]["role"] != "user" or fixed[-1]["role"] != "assistant":
        return None
    if any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(fixed)):
        return None
    return fixed


def prompt_key(messages):
    return digest("\n".join(m["content"] for m in messages if m["role"] == "user"))


def iter_json(path):
    with open(path, encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield index, json.loads(line)


def iter_parquet(path, columns=None):
    import pyarrow.parquet as pq

    index = 0
    with pq.ParquetFile(path, pre_buffer=False) as table:
        for batch in table.iter_batches(batch_size=64, columns=columns, use_threads=False):
            for row in batch.to_pylist():
                yield index, row
                index += 1


def base_validation_hashes(directory, tokenizer):
    """Recover original held-out document hashes from only the small val bins."""
    protected = set()
    pending = np.empty(0, dtype=np.uint16)
    for path in sorted(Path(directory).glob("val-*.bin")):
        with open(path, "rb") as stream:
            while block := stream.read(2**20):
                ids = np.concatenate((pending, np.frombuffer(block, dtype=np.uint16)))
                start = 0
                for end in np.flatnonzero(ids == tokenizer.eos_id):
                    protected.add(digest(tokenizer.decode(ids[start:end])))
                    start = int(end) + 1
                pending = ids[start:].copy()
        if len(pending):
            raise ValueError("Base val shard ended inside a document")
    return protected


class Curator:
    def __init__(self, recipe, resume=False):
        from inference.tokenizer import Tokenizer
        from inference.template import ChatFormat
        from training.dataset import ShardWriter

        self.recipe, self.out = recipe, Path(recipe["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        existing = (self.out / "dedup.sqlite").exists()
        if existing and not resume:
            raise FileExistsError("Refusing to overwrite a previous candidate pool")
        self.tok = Tokenizer(recipe["tokenizer"])
        self.fmt = ChatFormat(self.tok)
        self.db = sqlite3.connect(self.out / "dedup.sqlite")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, split TEXT, bucket TEXT)")
        self.buckets = {b["name"]: b for b in recipe["buckets"] if b["kind"] != "reference"}
        self.stats = {name: {s: dict(documents=0, tokens=0, sources={}, lengths={}) for s in ("train", "val")} for name in self.buckets}
        checkpoint = {}
        if existing:
            checkpoint = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
            self.stats = checkpoint['stats']
            committed = {(b, s): n for b, s, n in self.db.execute("SELECT bucket,split,count(*) FROM seen GROUP BY bucket,split")}
            for name in self.buckets:
                for split in ("train", "val"):
                    stat = self.stats[name][split]
                    if committed.get((name, split), 0) != stat["documents"]:
                        raise ValueError("Dedup ledger and checkpoint disagree; refusing unsafe resume")
                    corpus = self.out / name / f"{split}.jsonl"
                    with open(corpus, "r+b") as stream:
                        tokens = 0
                        for _ in range(stat["documents"]):
                            line = stream.readline()
                            if not line:
                                raise ValueError("Incomplete corpus checkpoint")
                            tokens += json.loads(line)["tokens"]
                        if tokens != stat["tokens"]:
                            raise ValueError("Corpus token total differs from checkpoint")
                        stream.truncate(stream.tell())
                    remaining = tokens * 2
                    for shard in sorted((self.out / name).glob(f"{split}-*.bin")):
                        if remaining == 0:
                            shard.unlink()
                        else:
                            size = shard.stat().st_size
                            if size > remaining:
                                with open(shard, "r+b") as stream:
                                    stream.truncate(remaining)
                            remaining -= min(size, remaining)
                    if remaining:
                        raise ValueError("Token shards are shorter than the committed corpus")
        self.writers = {(name, s): ShardWriter(self.out / name, s, "uint16", 50_000_000) for name in self.buckets for s in ("train", "val")}
        if existing:
            for (name, split), writer in self.writers.items():
                writer.index = len(list((self.out / name).glob(f"{split}-*.bin")))
                writer.total = self.stats[name][split]["tokens"]
        self.files = {(name, s): open(self.out / name / f"{s}.jsonl", "a" if existing else "w", encoding="utf-8") for name in self.buckets for s in ("train", "val")}
        self.samples = {name: [] for name in self.buckets}
        if existing:
            self.samples = json.loads((self.out / "samples.json").read_text(encoding="utf-8"))
        self.rejected = collections.Counter(checkpoint.get('rejected', {}))
        self.protected_prompts, self.protected_texts, self.protected_groups = set(), set(), set()
        self.rows, self.insertions, self.last_report = checkpoint.get('rows_scanned', 0), 0, 0

    def report(self, phase, complete=False):
        from tools.prepare_anneal import write_json
        import psutil

        for writer in self.writers.values():
            if writer.file is not None:
                writer.file.flush()
        for stream in self.files.values():
            if not stream.closed:
                stream.flush()
        self.db.commit()

        result = dict(phase=phase, complete=complete, rows_scanned=self.rows, stats=self.stats,
                      rejected=dict(self.rejected), protected_prompts=len(self.protected_prompts),
                      protected_base_documents=len(self.protected_texts),
                      rss_mib=round(psutil.Process().memory_info().rss / 2**20))
        write_json(self.out / "stats.json", result)
        write_json(self.out / "samples.json", self.samples)
        print(json.dumps(dict(phase=phase, scanned=self.rows, train_tokens=sum(s["train"]["tokens"] for s in self.stats.values()),
                             val_tokens=sum(s["val"]["tokens"] for s in self.stats.values()), rss_mib=result["rss_mib"])), flush=True)

    def tick(self, phase):
        from tools.prepare_anneal import check_memory

        self.rows += 1
        if self.rows % 64 == 0:
            import psutil
            if psutil.virtual_memory().available < self.recipe["minimum_available_gib"] * 2**30:
                self.report("paused_for_memory")
                raise SystemExit(75)
        if time.monotonic() - self.last_report >= 30:
            self.db.commit()
            self.report(phase)
            self.last_report = time.monotonic()

    def full(self, bucket):
        desired = self.recipe["planned_tokens"] * self.buckets[bucket]["weight"] * self.recipe["pool_multiple"]
        return self.stats[bucket]["train"]["tokens"] >= desired

    def emit(self, bucket, key, ids, content, provenance, forced_val=False):
        heldout = forced_val or ("base_row" not in provenance and int(key[:12], 16) % self.recipe["validation_modulus"] == 0)
        split = "val" if heldout else "train"
        if split == "train" and self.full(bucket):
            self.rejected["pool_cap"] += 1
            return
        try:
            self.db.execute("INSERT INTO seen VALUES (?,?,?)", (key, split, bucket))
        except sqlite3.IntegrityError:
            self.rejected["duplicate"] += 1
            return
        self.writers[bucket, split].write(ids)
        record = dict(id=key, split=split, bucket=bucket, tokens=len(ids), language="zh", provenance=provenance, **content)
        self.files[bucket, split].write(json.dumps(record, ensure_ascii=False) + "\n")
        stat = self.stats[bucket][split]
        stat["documents"] += 1
        stat["tokens"] += len(ids)
        source = provenance["source"]
        stat["sources"][source] = stat["sources"].get(source, 0) + len(ids)
        band = "up_to_128" if len(ids) <= 128 else "129_to_512" if len(ids) <= 512 else "513_to_1536" if len(ids) <= 1536 else "over_1536"
        stat["lengths"][band] = stat["lengths"].get(band, 0) + len(ids)
        preview = self.samples[bucket]
        if len(preview) < 12:
            preview.append(record)
        else:
            largest = max(range(len(preview)), key=lambda i: preview[i]["id"])
            if key < preview[largest]["id"]:
                preview[largest] = record

    def chat(self, row, source, index, forced_val=False):
        messages = normal_messages(row)
        if messages is None:
            self.rejected["invalid_conversation"] += 1
            return
        text = "\n".join(m["content"] for m in messages)
        reason = rejection(text, self.recipe["filters"])
        if reason or chinese_fraction(text) < self.recipe["filters"]["min_chinese_fraction"]:
            self.rejected[reason or "non_chinese"] += 1
            return
        if row.get("score") is not None and float(row["score"]) < 4:
            self.rejected["provider_score"] += 1
            return
        if str(row.get("difficulty", "")) in ("困难", "非常困难", "较难", "困难级别"):
            self.rejected["advanced"] += 1
            return
        for question, answer in zip(messages[::2], messages[1::2]):
            if re.search(self.recipe["filters"]["short_request"], question["content"]) and len(answer["content"]) > self.recipe["filters"]["max_answer_chars_when_short_requested"]:
                self.rejected["violates_short_answer"] += 1
                return
        bucket = dialogue_bucket(messages, source)
        key = prompt_key(messages)
        if self.db.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone():
            self.rejected["duplicate"] += 1
            return
        if self.full(bucket) and not forced_val:
            return
        ids = self.fmt.encode(messages, "all")[0] + [self.tok.eos_id]
        if len(ids) > self.recipe["filters"]["max_dialogue_tokens"]:
            self.rejected["dialogue_too_long"] += 1
            return
        group = (row.get("metadata") or {}).get("source_paper") if isinstance(row.get("metadata"), dict) else None
        # Human review does not establish whether an answer was human-written.
        generated = True if source.startswith(("smoltalk-zh/", "fineweb-v23/")) else None
        provenance = dict(source=source, row=index, generated=generated, human_verified=row.get("human_verified"), provider_score=row.get("score"), source_document=group)
        self.emit(bucket, key, np.asarray(ids, dtype=np.uint16), dict(messages=messages), provenance,
                  forced_val or key in self.protected_prompts or group is not None and group in self.protected_groups)

    def text(self, row, index):
        bucket = subject(row["text"], row["source"], list(self.buckets.values()), self.recipe["filters"])
        if bucket is None:
            self.rejected["outside_foundational_topics"] += 1
            return
        key = digest(row["text"])
        if self.db.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone():
            self.rejected["duplicate"] += 1
            return
        heldout = index % 1000 == 999 or key in self.protected_texts
        if self.full(bucket) and not heldout:
            return
        ids = self.tok.encode(row["text"]) + [self.tok.eos_id]
        if len(ids) > 8192:
            self.rejected["text_too_long"] += 1
            return
        self.emit(bucket, key, np.asarray(ids, dtype=np.uint16), dict(text=row["text"]),
                  dict(source=row["source"], base_row=index, provided_quality=row["quality"], verification="educational_topic_filters_only"), heldout)

    def finish(self):
        from training.dataset import write_meta
        from tools.prepare_anneal import write_json

        self.db.commit()
        for writer in self.writers.values():
            writer.close()
        for stream in self.files.values():
            stream.close()
        availability = {}
        for name, bucket in self.buckets.items():
            directory = self.out / name
            shutil.copyfile(self.recipe["tokenizer"], directory / "tokenizer.model")
            write_meta(directory, self.tok.vocab_size, "uint16", self.tok.eos_id, "sentencepiece", tokenizer_file="tokenizer.model", train_on="all", stats=self.stats[name], curation_version=2)
            windows = sum(max(0, (f.stat().st_size // 2 - 1) // self.recipe["seq_len"]) for f in directory.glob("train-*.bin"))
            val_windows = sum(max(0, (f.stat().st_size // 2 - 1) // self.recipe["seq_len"]) for f in directory.glob("val-*.bin"))
            target = round(self.recipe["planned_tokens"] * bucket["weight"])
            availability[name] = dict(label=bucket["label"],requested_weight=bucket["weight"],target_training_tokens=target,
                                     unique_pool_tokens=self.stats[name]["train"]["tokens"],train_windows=windows,
                                     val_windows=val_windows,minimum_one_pass_met=windows*self.recipe["seq_len"]>=target)
        write_json(self.out / "availability.json", availability)
        self.report("candidate_pools_complete", complete=True)
        self.db.close()


def curate(path, resume=False):
    from tools.prepare_anneal import limit_memory, check_memory

    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    job = limit_memory(recipe["memory_limit_gib"])
    check_memory(recipe["minimum_available_gib"])
    import pyarrow as pa

    import psutil
    if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    pa.set_memory_pool(pa.system_memory_pool())
    c = Curator(recipe, resume=resume)
    c.protected_texts = base_validation_hashes("data/tokens/base", c.tok)
    for index, row in iter_parquet(recipe["sft_parquet"], ["messages"]):
        if index % 100 == 99:
            m = normal_messages(row)
            if m:
                c.protected_prompts.add(prompt_key(m))
    for _, row in iter_json("data/cos-v2/raw/fineweb-v23/val_messages_no_sys.jsonl"):
        m = normal_messages(row)
        if m:
            c.protected_prompts.add(prompt_key(m))
        group = (row.get("metadata") or {}).get("source_paper")
        if group:
            c.protected_groups.add(group)
    c.report("heldout_keys_protected")
    # Official validation questions/documents are ingested before training rows.
    paths = [Path("data/cos-v2/raw/fineweb-v23/val_messages_no_sys.jsonl")]
    paths += [Path("data/cos-v2/raw/coig-cqia/COIG-CQIA-full.jsonl")]
    paths += sorted(Path("data/cos-v2/raw/smoltalk-zh").glob("*.parquet"))
    paths += [Path("data/cos-v2/raw/fineweb-v23/train_messages_no_sys.jsonl")]
    for source_path in paths:
        source = source_path.parent.name + "/" + source_path.stem
        iterator = iter_parquet(source_path) if source_path.suffix == ".parquet" else iter_json(source_path)
        for index, row in iterator:
            c.tick(source)
            if source_path.parent.name == "coig-cqia" and row.get("human_verified") is not True:
                c.rejected["not_human_verified"] += 1
                continue
            c.chat(row, source, index, forced_val=source_path.name.startswith("val_"))
        c.report("source_complete:" + source)
    for index, row in iter_parquet(recipe["sft_parquet"]):
        c.tick("existing_sft")
        c.chat(row, "existing_sft/" + str(row.get("source", "unknown")), index, forced_val=index % 100 == 99)
    allowed_sources = {s for b in c.buckets.values() if b["kind"] == "text" for s in b["sources"]}
    for index, row in iter_parquet(recipe["base_parquet"], ["text", "source", "quality"]):
        c.tick("base_educational_subjects")
        if row["source"] in allowed_sources:
            c.text(row, index)
    c.finish()
    return job


def literal_task(text, label, key, variant=0):
    """Derive one task whose answer can be checked against supplied text."""
    sentences = [s.strip(' #*\t') for s in re.split(r'(?<=[。！？])\s*|\n+', text)
                 if 25 <= len(s.strip()) <= 180 and chinese_fraction(s) >= .5]
    if len(sentences) < 3:
        return None
    sentences = sentences[:3]
    number = int(key[:8], 16) ^ variant
    chosen = sentences[number % 3]
    if number % 2:
        bucket = 'dialogue_reading'
        if number % 4 == 1:
            material = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))
            prompt = f'请根据材料提取信息。\n材料：\n{material}\n\n请原样输出第{number % 3 + 1}条内容，不包含编号，不要补充解释。'
            answer, rule = chosen, 'numbered_sentence_extraction'
        else:
            position = max(3, len(chosen) // 3)
            span = chosen[position:position+6]
            blank = chosen[:position] + '____' + chosen[position+6:]
            prompt = '材料：\n'+'\n'.join(sentences)+f'\n\n根据材料补全句子：“{blank}”。只输出空缺的原文。'
            answer, rule = span, 'literal_span_cloze'
    else:
        bucket = 'dialogue_instruction'
        if number % 4 == 0:
            prompt = f'原样复制下面的文字，只输出文字本身，不要加引号、标题或解释：\n{chosen}'
            answer, rule = chosen, 'exact_copy'
        else:
            prompt = f'已知以下两项资料：\n学科：{label}\n原句：{chosen}\n\n把资料转换成一个JSON对象，只保留“学科”和“原句”两个字段，值与资料完全一致，只输出JSON。'
            values = {'学科': label, '原句': chosen}
            answer = json.dumps(values, ensure_ascii=False)
            assert json.loads(answer) == values
            rule = 'json_fields_from_supplied_values'
    assert chosen in text and (answer in text or rule == 'json_fields_from_supplied_values')
    return bucket, [dict(role='user', content=prompt), dict(role='assistant', content=answer)], rule


def refine(path):
    """Remove non-lesson text and regenerate derived tasks from retained parents."""
    from tools.prepare_anneal import limit_memory, check_memory

    recipe = json.loads(Path(path).read_text(encoding='utf-8'))
    job = limit_memory(recipe['memory_limit_gib'])
    check_memory(recipe['minimum_available_gib'])
    source = Path(recipe['candidate_source'])
    if source.resolve() == Path(recipe['out']).resolve():
        raise ValueError('Refinement requires a separate output directory')
    c = Curator(recipe)
    for bucket in recipe['buckets']:
        name = bucket['name']
        if bucket['kind'] == 'reference':
            continue
        for split in ('train','val'):
            paths = iter(sorted((source/name).glob(f'{split}-*.bin')))
            stream = None
            try:
                for _, record in iter_json(source/name/f'{split}.jsonl'):
                    c.tick('refine_foundational_texts')
                    remaining, chunks = record['tokens']*2, []
                    while remaining:
                        if stream is None:
                            stream = open(next(paths), 'rb')
                        chunk = stream.read(remaining)
                        if not chunk:
                            stream.close();stream=None
                            continue
                        chunks.append(chunk);remaining -= len(chunk)
                    if record['provenance']['source'].startswith('literal_grounded/'):
                        continue
                    reason = foundation_rejection(record['text']) if bucket['kind']=='text' else None
                    if reason:
                        c.rejected['foundation_'+reason] += 1
                        continue
                    content = dict(text=record['text']) if bucket['kind']=='text' else dict(messages=record['messages'])
                    c.emit(name, record['id'], np.frombuffer(b''.join(chunks),dtype=np.uint16), content,
                           record['provenance'],forced_val=split=='val')
            finally:
                if stream is not None:
                    stream.close()
    c.finish()
    return job


def supplement(path):
    from tools.prepare_anneal import limit_memory, check_memory
    recipe = json.loads(Path(path).read_text(encoding='utf-8'))
    job = limit_memory(recipe['memory_limit_gib'])
    check_memory(recipe['minimum_available_gib'])
    c = Curator(recipe, resume=True)
    c.db.execute('CREATE TABLE IF NOT EXISTS derived_parents (id TEXT PRIMARY KEY)')
    names = [b['name'] for b in recipe['buckets'] if b['kind'] == 'text']
    desired = {n: round(recipe['planned_tokens'] * c.buckets[n]['weight'] * 1.25)
               for n in ('dialogue_instruction', 'dialogue_reading')}
    handles = collections.deque()
    for name in names:
        for split in ('train', 'val'):
            for variant in (0, 1, 2, 3):
                handles.append((name, split, variant, iter_json(c.out/name/f'{split}.jsonl')))
    while handles:
        name, split, variant, iterator = handles.popleft()
        try:
            _, parent = next(iterator)
        except StopIteration:
            continue
        handles.append((name, split, variant, iterator))
        c.tick('literal_grounded_supplement')
        suffix = '' if variant==0 else ':alternate' if variant==1 else f':variant{variant}'
        ledger_key = parent['id']+suffix
        if c.db.execute('SELECT 1 FROM derived_parents WHERE id=?', (ledger_key,)).fetchone():
            continue
        task = literal_task(parent['text'], c.buckets[name]['label'], parent['id'], variant)
        if task is None:
            continue
        bucket, messages, rule = task
        if split == 'train' and c.stats[bucket]['train']['tokens'] >= desired[bucket]:
            continue
        ids = c.fmt.encode(messages, 'all')[0] + [c.tok.eos_id]
        if len(ids) > recipe['filters']['max_dialogue_tokens']:
            continue
        provenance = dict(parent['provenance'], source='literal_grounded/'+rule, generated=True,
                          verification='literal_span_or_json_roundtrip', parent_id=parent['id'], parent_bucket=name,
                          literal_variant=variant)
        # base_row preserves the parent's original split, without drawing a new split.
        assert 'base_row' in provenance
        c.emit(bucket, prompt_key(messages), np.asarray(ids,dtype=np.uint16), dict(messages=messages),
               provenance, forced_val=split=='val')
        c.db.execute('INSERT INTO derived_parents VALUES (?)', (ledger_key,))
        if all(c.stats[n]['train']['tokens'] >= desired[n] for n in desired):
            break
    c.finish()
    return job


def pack(path, resume=False):
    """Materialize one fixed-budget mix after checking every bucket's supply."""
    from fractions import Fraction
    import math
    from tools.prepare_anneal import limit_memory, wait_memory, mix_streams, write_json

    recipe = json.loads(Path(path).read_text(encoding='utf-8'))
    job = limit_memory(recipe['memory_limit_gib'])
    minimum = recipe['minimum_available_gib']
    wait_memory(minimum)
    import psutil
    if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    seq_len = recipe['seq_len']
    parts = [dict(name=b['name'], label=b['label'], weight=b['weight'],
                  directory=b.get('directory', str(Path(recipe['out'])/b['name'])))
             for b in recipe['buckets']]
    denominator = math.lcm(*(Fraction(str(p['weight'])).denominator for p in parts))
    rows = dict(train=recipe['planned_tokens']//seq_len, val=400)
    if sum(Fraction(str(p['weight'])) for p in parts) != 1:
        raise ValueError('Recipe weights must sum to one')
    if recipe['planned_tokens'] % seq_len or any(n % denominator for n in rows.values()):
        raise ValueError('Token budget and validation rows must fit the recipe exactly')
    supply = {}
    for part in parts:
        supply[part['name']] = {}
        for split, count in rows.items():
            available = sum(max(0, (p.stat().st_size//2-1)//seq_len)
                            for p in Path(part['directory']).glob(f'{split}-*.bin'))
            requested = int(count*Fraction(str(part['weight'])))
            supply[part['name']][split] = dict(available_windows=available, requested_windows=requested)
            if requested > available:
                raise ValueError(f"Insufficient unique windows: {part['name']}/{split}: {available} < {requested}")
    target = Path(recipe.get('packed_out', 'data/tokens/anneal-small-v2'))
    if not resume and (any(target.glob('*.bin')) or (target/'meta.json').exists()):
        raise FileExistsError(f'Refusing to overwrite {target}')
    if resume:
        previous=json.loads((Path(recipe['out'])/'packing-plan.json').read_text(encoding='utf-8'))
        if previous['parts']!=parts or previous['rows']!=rows or Path(previous['output']).resolve()!=target.resolve():
            raise ValueError('Cannot resume a different packing plan')
    candidate_audit = audit_candidates(recipe)
    write_json(Path(recipe['out'])/'candidate-audit.json', candidate_audit)
    write_json(Path(recipe['out'])/'packing-plan.json', dict(parts=parts, rows=rows, supply=supply, output=str(target)))
    stats = mix_streams(parts, target, rows, seq_len, seed=20261117, min_available_gib=minimum, resume=resume)
    if stats['train']['targets'] != recipe['planned_tokens']:
        raise ValueError('Packed token budget differs from recipe')
    audit = audit_packed(target, minimum)
    write_json(target/'audit.json', audit)
    if audit['train_val_overlapping_windows'] or audit['val_duplicate_windows']:
        raise ValueError('Validation contains repeated windows or training overlap')
    write_json(target/'recipe.json', recipe)
    write_json(Path(recipe['out'])/'packing-result.json', dict(status='ready', output=str(target), stats=stats, audit=audit))
    write_inspection(recipe, stats, audit)
    print(json.dumps(dict(phase='packed_and_audited', output=str(target), audit=audit)), flush=True)
    return job


def write_inspection(recipe, packed_stats, audit):
    """Export a readable recipe, deterministic examples and measured counts."""
    import csv
    from tools.prepare_anneal import write_json

    root = Path(recipe['out'])
    destination = root.parent
    stats = json.loads((root/'stats.json').read_text(encoding='utf-8'))
    samples = json.loads((root/'samples.json').read_text(encoding='utf-8'))
    write_json(destination/'samples.json', samples)
    lines = ['Cos 数据配方与抽样预览', '',
             '比例按训练目标 token 计算：对话 30%，学科文本 40%，普通 base 回放 30%。',
             f"训练目标：{recipe['planned_tokens']:,} token，{recipe['cooldown_steps']:,} 步；2048 × 16。",
             '这里只准备数据，不启动训练；原预训练最后 10% 的计划保持不变。',
             '文本通过规则筛选，未逐篇核验事实；机械衍生题只保证答案可从题面校验。',
             '样例按固定哈希抽取，包含原始来源；以下每类展示 3 条，长文截断，完整样例见 samples.json。', '']
    with open(destination/'recipe.csv','w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['类别','名称','训练比例','训练目标token','候选训练token','候选文档数','验证窗口数'])
        for bucket in recipe['buckets']:
            name = bucket['name']
            planned = packed_stats['train']['source_target_tokens'][name]
            pool = stats['stats'].get(name,{}).get('train',{})
            writer.writerow([name,bucket['label'],bucket['weight'],planned,pool.get('tokens','base 原分布回放'),pool.get('documents',''),packed_stats['val']['source_rows'][name]])
            lines.extend([f"[{name}] {bucket['label']}：{bucket['weight']*100:g}%，训练 {planned:,} token", ''])
            for record in sorted(samples.get(name,[]),key=lambda r:r['id'])[:3]:
                lines.extend([f"ID：{record['id']}",f"划分：{record['split']}；来源：{record['provenance']['source']}"])
                if 'text' in record:
                    text=record['text'];lines.append(text[:2200]+('\n[预览截断]' if len(text)>2200 else ''))
                else:
                    for message in record['messages']:
                        text=message['content'];lines.append(message['role']+'：'+text[:2200]+('\n[预览截断]' if len(text)>2200 else ''))
                lines.append('')
    (destination/'samples.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    original = json.loads((Path(recipe['candidate_source'])/'stats.json').read_text(encoding='utf-8'))['stats']
    knowledge = [b['name'] for b in recipe['buckets'] if b['kind']=='text']
    original_tokens = sum(original[n]['train']['tokens'] for n in knowledge)
    retained_tokens = sum(stats['stats'][n]['train']['tokens'] for n in knowledge)
    report = dict(status='prepared_not_training', candidate_directory=str(root),
                  packed_directory=recipe['packed_out'], recipe='dataset/cos_recipe.json',
                  candidate_training_tokens=sum(s['train']['tokens'] for s in stats['stats'].values()),
                  knowledge_original_candidate_tokens=original_tokens,knowledge_retained_tokens=retained_tokens,
                  knowledge_removed_tokens=original_tokens-retained_tokens,
                  packed_stats=packed_stats,exact_window_audit=audit,
                  automatic_training_start=False,factual_verification=False,
                  limitations=['规则筛选不保证事实正确或完全符合基础教材难度。',
                               '近似重复、不同问法的同一题不在精确哈希去重保证范围内。',
                               '衍生任务来自相同原文，不能算成新增的独立知识。',
                               '普通 base 回放可与精选训练文本重合，这是主动加权。'])
    write_json(destination/'build-report.json', report)


def audit_candidates(recipe):
    """Validate corpus accounting and that derived tasks retain parent splits."""
    from tools.prepare_anneal import wait_memory

    root = Path(recipe['out'])
    stats = json.loads((root/'stats.json').read_text(encoding='utf-8'))['stats']
    result = {}
    derived, parent_errors, answer_errors = 0, 0, 0
    db = sqlite3.connect(root/'dedup.sqlite')
    db.execute('PRAGMA cache_size=-8192')
    text_buckets={b['name'] for b in recipe['buckets'] if b['kind']=='text'}
    ledger_counts, parents = collections.Counter(), {}
    for key,split,bucket in db.execute('SELECT key,split,bucket FROM seen'):
        ledger_counts[bucket,split] += 1
        if bucket in text_buckets:
            parents[key]=(split,bucket)
    db.close()
    try:
        for bucket in recipe['buckets']:
            name = bucket['name']
            if bucket['kind'] == 'reference':
                continue
            result[name] = {}
            for split in ('train','val'):
                wait_memory(recipe['minimum_available_gib'])
                documents, tokens = 0, 0
                for _, record in iter_json(root/name/f'{split}.jsonl'):
                    documents += 1
                    tokens += record['tokens']
                    if record['split'] != split or record['bucket'] != name:
                        raise ValueError('Corpus label mismatch')
                    key = prompt_key(record['messages']) if 'messages' in record else digest(record['text'])
                    if key != record['id']:
                        raise ValueError('Corpus identity hash mismatch')
                    if bucket['kind']=='text' and recipe['filters'].get('foundation_only') and foundation_rejection(record['text']):
                        raise ValueError('Excluded non-lesson text remains in final corpus')
                    provenance = record['provenance']
                    if provenance['source'].startswith('literal_grounded/'):
                        derived += 1
                        parent = parents.get(provenance['parent_id'])
                        parent_errors += parent != (split, provenance['parent_bucket'])
                        prompt, answer = (m['content'] for m in record['messages'])
                        if provenance['source'].endswith('json_fields_from_supplied_values'):
                            values = json.loads(answer)
                            answer_errors += set(values) != {'学科','原句'} or any(v not in prompt for v in values.values())
                        else:
                            answer_errors += answer not in prompt
                actual_tokens = sum(p.stat().st_size//2 for p in (root/name).glob(f'{split}-*.bin'))
                if (documents, tokens) != (stats[name][split]['documents'], stats[name][split]['tokens']) or actual_tokens != tokens:
                    raise ValueError(f'Corpus/token accounting mismatch: {name}/{split}')
                ledger_documents = ledger_counts.get((name,split),0)
                if ledger_documents != documents:
                    raise ValueError('Dedup ledger count mismatch')
                result[name][split] = dict(documents=documents, tokens=tokens)
    finally:
        db.close()
    if parent_errors or answer_errors:
        raise ValueError('Derived task has a wrong split or unverifiable answer')
    return dict(buckets=result, derived_tasks=derived, parent_split_errors=parent_errors,
                literal_answer_errors=answer_errors, factual_verification=False,
                deduplication='Canonical exact text/prompt; no near-duplicate guarantee')


def audit_packed(directory, minimum_available_gib=0):
    """Check exact window overlap, labels, vocabulary bounds and file lengths."""
    from tools.prepare_anneal import wait_memory

    root = Path(directory)
    meta = json.loads((root/'meta.json').read_text(encoding='utf-8'))
    width, dtype = meta['row_len'], np.dtype(meta['dtype'])
    hashes = {}
    result = {}
    for split in ('val', 'train'):
        wait_memory(minimum_available_gib)
        seen, count, pad, eos, max_id = set(), 0, 0, 0, 0
        expected = meta['stats'][split]['rows']
        ids_path, mask_path = root/f'{split}.ids.bin', root/f'{split}.mask.bin'
        if ids_path.stat().st_size != expected*width*dtype.itemsize or mask_path.stat().st_size != expected*width:
            raise ValueError('Packed file size differs from metadata')
        with open(ids_path, 'rb') as ids, open(mask_path, 'rb') as masks:
            while block := ids.read(width*dtype.itemsize*256):
                wait_memory(minimum_available_gib)
                data = np.frombuffer(block, dtype=dtype).reshape(-1, width)
                mask = np.frombuffer(masks.read(data.size), dtype=np.uint8)
                if not np.all(mask == 1):
                    raise ValueError('Unexpected masked targets')
                pad += int(np.count_nonzero(data == 3))
                eos += int(np.count_nonzero(data == meta['eos_id']))
                max_id = max(max_id, int(data.max()))
                for row in data:
                    seen.add(hashlib.sha256(row.tobytes()).digest())
                count += len(data)
        if pad or max_id >= meta['vocab_size']:
            raise ValueError('PAD or out-of-vocabulary token in packed corpus')
        hashes[split] = seen
        result[split] = dict(rows=count, unique_windows=len(seen), duplicate_windows=count-len(seen),
                             pad_tokens=pad, eos_tokens=eos, max_token_id=max_id, all_targets_enabled=True)
    result.update(train_val_overlapping_windows=len(hashes['train'] & hashes['val']),
                  val_duplicate_windows=result['val']['duplicate_windows'],
                  scope='Exact 2049-token rows only; not a near-duplicate or factual-quality guarantee.')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    if args.refine:
        refine(args.recipe)
    elif args.pack:
        pack(args.recipe, resume=args.resume)
    elif args.supplement:
        supplement(args.recipe)
    else:
        curate(args.recipe, resume=args.resume)
