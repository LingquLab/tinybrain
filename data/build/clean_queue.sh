#!/bin/zsh
# 把还没洗的来源排队，一个接一个洗（同一时间只跑一个来源），每个洗完做一次抽查，结果写到 data/clean/<来源>/_audit.txt。
# 用法：zsh data/build/clean_queue.sh 进程数 来源1 来源2 ...
# 开始前先等别的清洗任务（除了一直在跑的 fw22_4_5）结束，免得机器同时压太多活。
cd "$(dirname "$0")/../.." || exit 1
workers=$1; shift
while pgrep -f "[p]ipeline clean (wiki_zh|fw22_3_4)" >/dev/null; do sleep 30; done
for src in "$@"; do
  echo "=== $(date '+%H:%M:%S') 开始 $src"
  python3 -m data.build.pipeline clean "$src" --workers "$workers" || { echo "!!! $src 失败"; continue; }
  python3 -m data.build.pipeline audit "$src" --docs 8 --seed 5 > "data/clean/$src/_audit.txt" 2>&1
done
echo "=== $(date '+%H:%M:%S') 队列跑完"
