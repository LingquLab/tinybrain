"""OpenAI 兼容的服务接口：tinybrain serve out/small/rl/ckpt.pt --port 8001

支持 POST /v1/chat/completions（messages、max_tokens、temperature、top_k、tools）和 GET /v1/models。
模型写出工具调用时，按 OpenAI 的格式放进 message.tool_calls，finish_reason 是 tool_calls；工具的结果用 role=tool 的消息传回来。
这样小米的 mimoagent 这类智能体外壳可以直接驱动我们的模型。只做了单线程、一次一个请求，够评测和收集轨迹用。
"""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from training.trainer_utils import load_model, pick_device

from .generate import generate
from .template import ChatFormat
from .tokenizer import DEFAULT_MODEL, Tokenizer


def to_messages(messages: list[dict], tools: list | None) -> list[dict]:
    """OpenAI 的消息格式转成我们的：工具说明并进 system；assistant 的 tool_calls 转成 {name, arguments}。"""
    out = []
    if tools:
        specs = [t.get("function", t) for t in tools]
        out.append({"role": "system", "content": "你可以调用下面的工具。调用时在回答里写一个 JSON：{\"name\": 工具名, \"arguments\": {…}}。\n"
                                                + json.dumps(specs, ensure_ascii=False)})
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
        msg = {"role": m["role"] if m["role"] in ("system", "user", "assistant", "tool") else "user", "content": content}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", c)
                args = f.get("arguments") or {}
                calls.append({"name": f.get("name"), "arguments": json.loads(args) if isinstance(args, str) else args})
            msg["tool_calls"] = calls
        if out and out[0]["role"] == "system" and msg["role"] == "system" and len(out) == 1 and tools:
            out[0]["content"] = msg["content"] + "\n" + out[0]["content"]
            continue
        out.append(msg)
    return out


class Service:
    def __init__(self, model, fmt: ChatFormat, name: str):
        self.model, self.fmt, self.name = model, fmt, name
        self.lock = threading.Lock()  # 一次只算一个请求

    def complete(self, body: dict) -> dict:
        messages = to_messages(body.get("messages") or [], body.get("tools"))
        prompt = self.fmt.prompt(messages)
        device = next(self.model.parameters()).device
        max_new = min(int(body.get("max_tokens") or 512), self.model.cfg.max_seq_len - len(prompt))
        if max_new <= 0:
            raise ValueError("对话太长，超过了模型的上限")
        temperature = float(body.get("temperature", 0.7))
        with self.lock:
            out = generate(self.model, torch.tensor([prompt], device=device), max_new, temperature=temperature,
                           top_k=body.get("top_k") or None, eos_id=self.fmt.end)
        generated = out[0, len(prompt):].tolist()
        finished = self.fmt.end in generated
        text, calls, _ = self.fmt.parse_turn(generated)
        message = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]
        n_out = generated.index(self.fmt.end) + 1 if finished else len(generated)
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": self.name,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else ("stop" if finished else "length")}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": n_out, "total_tokens": len(prompt) + n_out}}


def make_server(service: Service, host: str = "127.0.0.1", port: int = 8001) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": service.name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "没有这个接口"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "没有这个接口"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self._send(200, service.complete(body))
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": {"message": str(e)}})

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenAI 兼容的服务接口")
    parser.add_argument("ckpt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", default="lm")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    dev = pick_device(args.device)
    service = Service(load_model(args.ckpt, dev).eval(), ChatFormat(Tokenizer(DEFAULT_MODEL)), args.name)
    server = make_server(service, args.host, args.port)
    print(f"在 http://{args.host}:{args.port}/v1 上提供服务（模型 {args.name}）", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
