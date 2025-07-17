import os
import torch
import torch.distributed as dist
import argparse

import torch.multiprocessing as mp

from multiprocessing import Manager
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from timeit import default_timer as timer

class OptimizerSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.all_params = list(params)

        my_params = [param for i, param in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.my_param_groups = [{"params":my_params}]
        self.optimizer = optimizer_cls(self.my_param_groups, **kwargs)

        self.handles = []
        super().__init__(self.all_params, {})
    
    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        self.synchronize_params()
        self.wait_for_all_params()

    def add_param_group(self, param_group):
        super().add_param_group(param_group)

    def synchronize_params(self):
        for i, param in enumerate(self.all_params):
            rank = i % self.world_size
            self.handles.append(dist.broadcast(param.data, src=rank, async_op=True))
    
    def wait_for_all_params(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) 

def run_model(rank, world_size, shard_optimizer, result_queue, num_trails, num_warmup_trails):
    setup(rank, world_size)
    device = torch.cuda.current_device()
    print ("loading model...")
    model = BasicsTransformerLM(
                vocab_size=10000,
                context_length=1024,
                d_model=1600,
                num_layers=48,
                num_heads=25,
                d_ff=6400,
                rope_theta=1e6
            ).to(device)
    print ("loaded model.")

    x = torch.randint(0, 10000, (1, 1024)).to(device)

    if shard_optimizer:
        optimizer = OptimizerSharding(model.parameters(), AdamW, lr=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=0.01)
    
    torch.cuda.reset_peak_memory_stats()
    mem_init = torch.cuda.memory_allocated()
    peak_init = torch.cuda.max_memory_allocated()
    print (f"[init] mem alloc={mem_init/1e9:.2f} GB, peak={peak_init/1e9:.2f} GB")

    def train_loop():
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad()
        output = model(x)
        loss = output.mean()
        loss.backward()

        mem_pre_optim = torch.cuda.memory_allocated()
        mem_pre_peak = torch.cuda.max_memory_allocated()
        print (f"[before optim] mem alloc={mem_pre_optim/1e9:.2f} GB, peak={mem_pre_peak/1e9:.2f} GB")

        torch.cuda.reset_peak_memory_stats()
        optimizer.step()

        mem_post_optim = torch.cuda.memory_allocated()
        mem_post_peak = torch.cuda.max_memory_allocated()
        print (f"[post optim] mem alloc={mem_post_optim/1e9:.2f} GB, peak={mem_post_peak/1e9:.2f} GB")

    print ("warm up:")
    for _ in range(num_warmup_trails):
        train_loop()
    print ("warm up finish:")

    step_times = []
    print ("benchmark:")
    for _ in range(num_trails):
        start_time = timer()
        train_loop()
        step_times.append((timer() - start_time))
    print ("benchmark finish:")

    step_t = torch.tensor(step_times, device=device)
    gathered_steps = [torch.zeros_like(step_t) for _ in range(world_size)]
    dist.all_gather(gathered_steps, step_t)
    if rank == 0:
        steps = [x for t in gathered_steps for x in t.cpu().tolist()]
        result_queue.put(steps)

if __name__ == "__main__":
    world_size = 2
    shard_optimizer = False
    num_trails = 10
    num_warmup_trails = 5
    mp.set_start_method("spawn", force=True)
    manager = Manager()

    result_queue = manager.Queue()
    mp.spawn(run_model,
             args = (world_size, shard_optimizer, result_queue, num_trails, num_warmup_trails),
             nprocs=world_size, 
             join=True)

    shard_time = result_queue.get()
    print (f"shard one step time: {sum(shard_time) * 1000/len(shard_time):.2f} ms.")

    # shard_optimizer = True
    # result_queue = manager.Queue()
    # mp.spawn(run_model,
    #          args = (1, shard_optimizer, result_queue, num_trails, num_warmup_trails),
    #          nprocs=world_size, 
    #          join=True)

    # no_shard_time = result_queue.get()
    # print (f"no shard one step time: {no_shard_time * 1000:.2f} ms.")