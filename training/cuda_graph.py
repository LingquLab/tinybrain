"""固定形状优化器图；使用设备端学习率，保留完整的 Muon 和融合 Adam 更新。"""
import torch

from .muon import Muon


class OptimizerGraph:
    """梯度拷进固定缓冲区；学习率留在设备张量中，每一步都可更新。"""
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.graph = None
        self.params = []
        self.grads = []
        for opt in optimizers:
            if not isinstance(opt, (torch.optim.Adam, Muon)):
                raise ValueError("优化器图目前支持融合 Adam 和项目里的 Muon")
            for group in opt.param_groups:
                device = group['params'][0].device
                group['lr'] = torch.as_tensor(group['lr'], dtype=torch.float32, device=device).clone()
                if isinstance(opt, torch.optim.Adam):
                    group['capturable'] = True

    @torch.no_grad()
    def _capture(self):
        self.params = [p for opt in self.optimizers for group in opt.param_groups
                       for p in group['params'] if p.grad is not None]
        self.grads = [p.grad for p in self.params]
        # 预热会执行更新，先备份，再原地恢复，保证捕获只算一次真实更新。
        saved_params = [(p, p.detach().cpu().clone()) for p in self.params]
        saved_state = {id(value): value.detach().cpu().clone()
                       for opt in self.optimizers for state in opt.state.values()
                       for value in state.values() if isinstance(value, torch.Tensor)}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for opt in self.optimizers:
                    opt.step()
        torch.cuda.current_stream().wait_stream(stream)
        for param, saved in saved_params:
            param.copy_(saved)
        for opt in self.optimizers:
            for state in opt.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        if id(value) in saved_state:
                            value.copy_(saved_state[id(value)])
                        else:
                            value.zero_()
        del saved_params, saved_state
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for opt in self.optimizers:
                opt.step()

    def __call__(self):
        if self.graph is None:
            self._capture()
            self.graph.replay()
        else:
            if any(p.grad is None for p in self.params):
                raise RuntimeError("图重放要求每步使用相同的参数和梯度")
            torch._foreach_copy_(self.grads, [p.grad for p in self.params])
            self.graph.replay()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
