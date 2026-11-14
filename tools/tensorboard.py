"""把训练 JSONL 日志补录并持续同步到 TensorBoard，无需重启正在运行的训练。

tinybrain tensorboard --run out/small/pretrain
python -m tensorboard.main --logdir out/small/pretrain/tensorboard --host 127.0.0.1 --port 6006
"""
import argparse
import json
from pathlib import Path
import time


TAGS = {
    "loss": "loss/train",
    "smooth_loss": "loss/smoothed",
    "val_loss": "loss/validation",
    "lr_scale": "learning_rate/scale",
    "tokens_per_s": "performance/tokens_per_second",
    "tokens": "progress/tokens",
    "epoch": "progress/epoch",
}


class MetricBridge:
    def __init__(self, run: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.run = Path(run)
        self.source = self.run / "log.jsonl"
        self.directory = self.run / "tensorboard"
        self.cursor_file = self.directory / "jsonl-cursor.json"
        config_file = self.run / "config.json"
        self.config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
        state = json.loads(self.cursor_file.read_text(encoding="utf-8")) if self.cursor_file.exists() else {}
        self.offset = state.get("offset", 0)
        self.last_step = state.get("step", 0)
        reset = self.source.exists() and self.source.stat().st_size < self.offset
        if reset:
            self.offset = self.last_step = 0
        self.writer = SummaryWriter(str(self.directory), flush_secs=2, purge_step=0 if reset else None)
        if not state or reset:
            self.writer.add_text("run/config", "```json\n" + json.dumps(self.config, indent=2, ensure_ascii=False) + "\n```", 0)
            self.writer.add_custom_scalars({"Loss": {"Train and validation": ["Multiline", ["loss/train", "loss/smoothed", "loss/validation"]]}})

    def _record(self, record):
        step = int(record["step"])
        for field, tag in TAGS.items():
            if field in record:
                self.writer.add_scalar(tag, record[field], step)
        total = self.config.get("steps", 0)
        if total:
            self.writer.add_scalar("progress/percent", 100 * step / total, step)
        if "lr_scale" in record:
            args = self.config.get("train", self.config.get("sft", {}))
            dim_scale = (self.config.get("model", {}).get("dim", 768) / 768) ** -0.5
            defaults = {"matrix": 0.02, "embed": 0.3, "head": 0.004, "engram": 0.3, "scalar": 0.01}
            for group, default in defaults.items():
                base = args.get(group + "_lr", default) * (1.0 if group == "matrix" else dim_scale)
                lr = base * args.get("lr_mult", 1.0) * record["lr_scale"]
                self.writer.add_scalar("learning_rate/" + group, lr, step)
        self.last_step = step

    def poll(self):
        """只消费完整行；flush 完事件后再保存游标，重启同步进程时不重复补录。"""
        if not self.source.exists():
            return 0
        count = 0
        with self.source.open(encoding="utf-8") as source:
            source.seek(self.offset)
            while True:
                position = source.tell()
                line = source.readline()
                if not line or not line.endswith("\n"):
                    self.offset = position
                    break
                if line.strip():
                    self._record(json.loads(line))
                    count += 1
                self.offset = source.tell()
        if count:
            self.writer.flush()
            temporary = self.cursor_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"offset": self.offset, "step": self.last_step}), encoding="utf-8")
            temporary.replace(self.cursor_file)
        return count

    def close(self):
        self.writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="持续同步训练日志到 TensorBoard")
    parser.add_argument("--run", type=Path, required=True, help="包含 config.json 和 log.jsonl 的训练输出目录")
    parser.add_argument("--once", action="store_true", help="只补录已有日志后退出")
    args = parser.parse_args(argv)
    bridge = MetricBridge(args.run)
    try:
        while True:
            count = bridge.poll()
            if count:
                print(f"同步 {count} 条记录，当前 step={bridge.last_step}，事件目录 {bridge.directory}", flush=True)
            if args.once:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
