#!/bin/bash
# ModelScope 上没有的 4 个来源，走 HF 镜像。单连接很慢，所以每个文件开 16 条连接。可以重复运行，支持断点续传。
cd "$(dirname "$0")/../.."  # 仓库根目录
get() {  # 名字 仓库 [只要名字匹配这个正则的文件]
  python3 - "$@" <<'PY' || return 1
import json, re, sys, urllib.request
from pathlib import Path
from urllib.parse import quote
name, repo = sys.argv[1:3]; pat = sys.argv[3] if len(sys.argv) > 3 else r"\.(parquet|jsonl|json)$"
dest = Path("data/raw") / name; dest.mkdir(parents=True, exist_ok=True)
def tree(path=""):
    out = []
    for f in json.loads(urllib.request.urlopen(urllib.request.Request(f"https://hf-mirror.com/api/datasets/{repo}/tree/main/{quote(path)}?limit=500", headers={"User-Agent": "python-urllib"}), timeout=60).read()):
        out += tree(f["path"]) if f["type"] == "directory" else [(f["path"], (f.get("lfs") or {}).get("size", f.get("size", 0)))]
    return out
files = [(p, s) for p, s in tree() if re.search(pat, p)]
lines = []
for p, s in files:
    out = dest / Path(p).name
    if out.exists() and out.stat().st_size == s and not Path(str(out) + ".aria2").exists(): continue
    lines += [f"https://hf-mirror.com/datasets/{repo}/resolve/main/{quote(p)}", f"  dir={dest}", f"  out={Path(p).name}"]
(dest / "_list.txt").write_text("\n".join(lines) + "\n")
print(f"{name}: {len(files)} 个文件共 {sum(s for _, s in files)/1e9:.2f} GB；还要下 {len(lines)//3} 个", flush=True)
PY
  aria2c -i "data/raw/$1/_list.txt" -j2 -x16 -s16 -k1M -c --max-tries=0 --retry-wait=5 --console-log-level=warn --summary-interval=0 --auto-file-renaming=false > /dev/null
  echo "  $1 现在 $(du -sh data/raw/$1 | cut -f1)"
}
get law          twang2218/chinese-law-and-regulations '\.parquet$'
get gujilab      gujilab/chinese-classical-corpus
get greathangpt  kenpusney/greathangpt-classical-chinese 'all\.parquet$'
get baidu_baike  lars1234/baidu-baike-dataset
echo "HF 镜像这 4 个全部完成"
