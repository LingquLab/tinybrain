#!/bin/bash
# 依次下载其余来源（ModelScope）。每行：名字 仓库 目录 [最多几个文件]。可以重复运行，下完的会跳过。
cd "$(dirname "$0")/../.."  # 仓库根目录
get() {
  python3 data/build/dl_list.py "$@" || return 1
  aria2c -i "data/raw/$1/_list.txt" -j8 -x4 -s4 -c --max-tries=0 --retry-wait=5 --console-log-level=warn --summary-interval=0 --auto-file-renaming=false > /dev/null
  echo "  $1 现在 $(du -sh data/raw/$1 | cut -f1)"
}
get wiki_zh          wikimedia/wikipedia                              20231101.zh
get wikisource_zh    wikimedia/wikisource                             20231201.zh
get numinamath       AI-MO/NuminaMath-1.5                             data
get cosmopedia_zh    opencsg/chinese-cosmopedia                       data 3
get arxiv            common-pile/arxiv_papers                         "" 6
get pes2o            allenai/peS2o                                    data/v3 2
get finemath         HuggingFaceTB/finemath                           finemath-4plus 8
get udmath_l2        openbmb/UltraData-Math                           data/UltraData-Math-L2-preview 6
get nemotron_math    nv-community/Nemotron-Pretraining-Specialized-v1 Nemotron-Pretraining-Math-Textbooks 2
get ifm_math_rewrite IFM/Math-Reasoning                               math-rewrite 2
for lang_n in py:4 cpp:2 java:1 js:1 go:1 rust:1; do get udcode_l2 openbmb/UltraData-Code "data/UltraData-Code-L2/${lang_n%:*}" "${lang_n#*:}"; done
for lang_n in python:2 cpp:1 java:1; do get udstack_v3_l2 openbmb/UltraData-Stack-v3-L2 "data/UltraData-Stack-v3-L2/${lang_n%:*}" "${lang_n#*:}"; done
get wiki_en          wikimedia/wikipedia                              20231101.en
get fw22_3_4         opencsg/Fineweb-Edu-Chinese-V2.2                 3_4 150
echo "队列全部完成"; du -sh data/raw/* | grep -v -e climbmix -e poetry -e yuanqu
