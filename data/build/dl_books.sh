#!/bin/bash
# 书籍类：英文古腾堡和鲁迅全集走 ModelScope；中文公版书 shu、古腾堡中文、PG-19 的验证集和测试集（只为拿到要排除的书目）走 HF 镜像。
cd "$(dirname "$0")/../.."  # 仓库根目录
ms() { python3 data/build/dl_list.py "$@" && aria2c -i "data/raw/$1/_list.txt" -j6 -x4 -s4 -c --max-tries=0 --retry-wait=5 --console-log-level=warn --summary-interval=0 --auto-file-renaming=false > /dev/null; echo "  $1 现在 $(du -sh data/raw/$1 | cut -f1)"; }
hf() {  # 名字 仓库 文件路径...
  name=$1; repo=$2; shift 2; mkdir -p data/raw/$name; : > data/raw/$name/_list.txt
  for p in "$@"; do printf '%s\n  dir=data/raw/%s\n  out=%s\n' "https://hf-mirror.com/datasets/$repo/resolve/main/$p" "$name" "$(basename $p)" >> data/raw/$name/_list.txt; done
  aria2c -i data/raw/$name/_list.txt -j2 -x16 -s16 -k1M -c --max-tries=0 --retry-wait=5 --console-log-level=warn --summary-interval=0 --auto-file-renaming=false > /dev/null
  echo "  $name 现在 $(du -sh data/raw/$name | cut -f1)"
}
ms gutenberg_en common-pile/project_gutenberg v0/documents
ms luxun        SkyContact/TheCompleteWorksOfLuXun data
hf pg19_heldout emozilla/pg19 data/test-00000-of-00001-29a571947c0b5ccc.parquet
hf shu          shjwudp/shu clean_text.jsonl.gz
hf gutenberg_zh manu/project_gutenberg data/zh-00000-of-00001-bca42330c6f1826c.parquet
echo "书籍类下载完成"; du -sh data/raw/gutenberg_en data/raw/luxun data/raw/pg19_heldout data/raw/shu data/raw/gutenberg_zh
