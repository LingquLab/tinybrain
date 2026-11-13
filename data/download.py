"""下载原始语料到 data/raw/。已存在的文件会跳过，可以重复运行。

两个来源，都固定到具体版本：
- 诗词：https://github.com/Werneror/Poetry （MIT），85 万余首，简体
- 元曲：https://github.com/chinese-poetry/chinese-poetry （MIT），1.1 万余条

共约 245MB。下载完成后运行 python data/prepare.py 做清洗。
"""

import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

POETRY_BASE = "https://raw.githubusercontent.com/Werneror/Poetry/4cfe49c06858e00d15f84d192fe5294295f79689/"
POETRY_FILES = (
    "元 元末明初 先秦 南北朝 唐 唐末宋初 宋_1 宋_2 宋_3 宋_4 宋末元初 宋末金初 当代 "
    "明_1 明_2 明_3 明_4 明末清初 民国末当代初 汉 清_1 清_2 清末民国初 清末近现代初 秦 辽 "
    "近现代 近现代末当代初 金 金末元初 隋 隋末唐初 魏晋 魏晋末南北朝初"
).split()
POETRY_HEADER = '"题目","朝代","作者","内容"'
YUANQU_URL = (
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/b8594f81a89752241442f2ce267d6f66f96704ee/"
    + urllib.parse.quote("元曲/yuanqu.json")
)


def valid_poetry_csv(content: bytes) -> bool:
    try:
        lines = content.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return False
    return len(lines) > 1 and lines[0].strip() == POETRY_HEADER


def valid_json_list(content: bytes) -> bool:
    try:
        data = json.loads(content)
    except ValueError:
        return False
    return isinstance(data, list) and all(isinstance(item, dict) for item in data)


# (下载地址, 在 data/raw/ 下的相对路径, 内容校验函数)
SOURCES = [
    (POETRY_BASE + urllib.parse.quote(f"{name}.csv"), f"poetry/{name}.csv", valid_poetry_csv)
    for name in POETRY_FILES
] + [
    (YUANQU_URL, "yuanqu.json", valid_json_list),
]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def download(url: str, dest: Path, fetch=fetch, validate=lambda content: True) -> None:
    """下载到 dest。已存在就跳过；内容没通过校验就报错，不留下坏文件。"""
    if dest.exists():
        return
    content = fetch(url)
    if not validate(content):
        raise ValueError(f"{url} 的内容没有通过校验")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(content)
    os.replace(part, dest)


def main():
    def one(source):
        url, path, validate = source
        download(url, RAW_DIR / path, validate=validate)
        print(f"  {path}", flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, SOURCES))  # 任何一个失败都会在这里抛出
    total = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file())
    print(f"{len(SOURCES)} 个文件就绪，共 {total / 1e6:.0f}MB -> {RAW_DIR}")


if __name__ == "__main__":
    main()
