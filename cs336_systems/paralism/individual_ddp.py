import torch.distributed as dist
import torch
import torch.cuda.nvtx as nvtx
from torch.autograd.profiler import record_function


class IndividualDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(IndividualDDP, self).__init__()
        self.module = module
        self.handles = []

        # initialize all parameters to be the same
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.transform_grad)

    def transform_grad(self, param):
        with torch.no_grad():
            param.grad.data /= dist.get_world_size()

        with record_function("allreduce_async"):
            self.handles.append(dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True))

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()