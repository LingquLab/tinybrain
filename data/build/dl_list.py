"""生成 aria2c 下载清单（ModelScope 优先）。用法：python dl_list.py 名字 仓库 目录 [最多几个文件]
清单写到 data/raw/<名字>/_list.txt，已经下完整的文件会跳过。"""
import json, sys, time
from pathlib import Path
from urllib.parse import quote
import requests
MS = "https://www.modelscope.cn"
name, repo, prefix = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
dest = Path(__file__).resolve().parents[2] / "data" / "raw" / name
dest.mkdir(parents=True, exist_ok=True)
files, page = [], 1
while True:
    for attempt in range(5):
        try:
            r = requests.get(f"{MS}/api/v1/datasets/{repo}/repo/tree", params={"Revision": "master", "Root": "/" + prefix, "Recursive": "False", "PageNumber": page, "PageSize": 500}, timeout=60)
            r.raise_for_status(); batch = (r.json().get("Data") or {}).get("Files") or []; break
        except Exception as e:
            time.sleep(3 + attempt * 3); batch = None
    if batch is None: sys.exit(f"第 {page} 页列不出来")
    files += [(f["Path"].lstrip("/"), f.get("Size") or 0) for f in batch if f.get("Type") != "tree" and f["Path"].lower().endswith((".parquet", ".jsonl", ".json", ".gz", ".zst"))]
    if len(batch) < 500: break
    page += 1
files = sorted(files)[:limit]
lines, todo = [], 0
for p, size in files:
    out = dest / Path(p).name
    if out.exists() and out.stat().st_size == size and not Path(str(out) + ".aria2").exists(): continue
    todo += size
    lines += [f"{MS}/datasets/{repo}/resolve/master/{quote(p)}", f"  dir={dest}", f"  out={Path(p).name}"]
(dest / "_list.txt").write_text("\n".join(lines) + "\n")
print(f"{name}: 仓库里 {len(files)} 个文件共 {sum(s for _, s in files)/1e9:.1f} GB；还要下 {len(lines)//3} 个，{todo/1e9:.1f} GB")
