
from cs336_basics.optimizer import AdamW
from multiprocessing import Manager
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd

class ToyModel(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def data_parallel_main(rank, world_size, data, num_steps, result_queue):
    setup(rank, world_size)

    torch.manual_seed(0)
    batch_size, d_model = data.shape
    local_batch_size = batch_size // world_size

    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    device = torch.cuda.current_device()
    toy_model = ToyModel(d_model, d_model).to(device)

    toy_model.train()

    data = data[start_index:end_index].to(device)

    optimizer = AdamW(toy_model.parameters(), lr = 0.001)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = toy_model(data)
        loss = output.mean()
        loss.backward()

        for param in toy_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.data /= world_size
        
        optimizer.step()
    
    if rank == 0:
        cpu_state = {k:v.detach().cpu() for k,v in toy_model.state_dict().items()}
        result_queue.put(cpu_state)
    

def train_single_process(data, num_steps):
    torch.manual_seed(0)
    batch_size, d_model = data.shape
    model = ToyModel(d_model, d_model).to("cuda")
    data = data.to("cuda")
    opt = AdamW(model.parameters(), lr=0.001)
    model.train()

    for _ in range(num_steps):
        opt.zero_grad()
        out = model(data)
        loss = out.mean()
        loss.backward()
        opt.step()
    
    return {k: v.detach().cpu() for k,v in model.state_dict().items()}
    

if __name__ == "__main__":
    full_data = torch.randn(10,5)
    num_steps = 10
    num_procs = 2


    ref_state = train_single_process(full_data, num_steps)

    mp.set_start_method('spawn', force=True)
    manager = Manager()
    result_queue = manager.Queue()

    mp.spawn(data_parallel_main,
             args=(num_procs, full_data, num_steps, result_queue),
             nprocs=num_procs,
             join=True
             )

    state_dict = result_queue.get()

    for key in ref_state.keys():
        assert torch.allclose(ref_state[key], state_dict[key])

    print("All state dicts are equal")