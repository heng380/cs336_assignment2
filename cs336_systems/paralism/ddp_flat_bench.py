import os
import torch
import torch.distributed as dist
import argparse

import torch.multiprocessing as mp

from multiprocessing import Manager
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from timeit import default_timer as timer
from individual_ddp import IndividualDDP
from individual_bucketed_ddp import DDP_Bucketed


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def bucketed_ddp(model, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times, bucket_size_mb):
    print ("warmup:")
    model = DDP_Bucketed(model, bucket_size_mb)
    for _ in range(num_warmup_trails):
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
    
    print ("benchmarking:")
    for _ in range(num_trails):
        torch.cuda.synchronize()
        step_start_time = timer()

        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        torch.cuda.synchronize()
        comms_start_time = timer()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        comms_times.append(timer() - comms_start_time)

        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(timer() - step_start_time)

def individual_ddp(model, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times):
    print ("warmup:")
    model = IndividualDDP(model)
    for _ in range(num_warmup_trails):
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
    
    print ("benchmarking:")
    for _ in range(num_trails):
        torch.cuda.synchronize()
        step_start_time = timer()

        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        torch.cuda.synchronize()
        comms_start_time = timer()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        comms_times.append(timer() - comms_start_time)

        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(timer() - step_start_time)

def naive_ddp(model, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times):
    print ("warmup:")
    for _ in range(num_warmup_trails):
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
        optimizer.step()
    
    print ("benchmarking:")
    for _ in range(num_trails):
        torch.cuda.synchronize()
        step_start_time = timer()

        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()

        torch.cuda.synchronize()
        comms_start_time = timer()

        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)

        torch.cuda.synchronize()
        comms_times.append(timer() - comms_start_time)

        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(timer() - step_start_time)

def flat_ddp(model, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times):
    print ("start warmup:")
    for _ in range(num_warmup_trails):
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()

        flatten_params = torch._utils._flatten_dense_tensors(tensors=[param.grad for param in model.parameters()])

        dist.all_reduce(flatten_params, op=dist.ReduceOp.AVG, async_op=False)

        unflatten_params = torch._utils._unflatten_dense_tensors(flatten_params, tensors=[param.grad for param in model.parameters()])
        for param, unflatten_param in zip(model.parameters(), unflatten_params):
            param.grad = unflatten_param
        optimizer.step()
    
    print ("start benchmark:")
    for _ in range(num_trails):
        torch.cuda.synchronize()
        start = timer()
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()

        torch.cuda.synchronize()
        comms_start = timer()
        flatten_params = torch._utils._flatten_dense_tensors(tensors=[param.grad for param in model.parameters()])

        dist.all_reduce(flatten_params, op=dist.ReduceOp.AVG, async_op=False)

        unflatten_params = torch._utils._unflatten_dense_tensors(flatten_params, tensors=[param.grad for param in model.parameters()])
        for param, unflatten_param in zip(model.parameters(), unflatten_params):
            param.grad = unflatten_param

        comms_times.append(timer() - comms_start)
        optimizer.step()

        torch.cuda.synchronize()
        step_times.append(timer() - start)

def benchmark_driver(rank, world_size, data, num_layers, batch_size,
                     num_trails, num_warmup_trails, vocab_size, context_length,
                     d_model, num_heads, d_ff, ddp_type, bucket_size_mb,result_queue):
    print ("start benchmark")
    setup(rank, world_size)

    device = torch.cuda.current_device()
    transformer = BasicsTransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff,
        rope_theta = 10000
    ).to(device)

    transformer.train()

    local_batch_size = batch_size // world_size
    start_index = local_batch_size * rank
    end_index = min(start_index + local_batch_size, data.shape[0])

    data = data[start_index: end_index].to(device)

    optimizer = AdamW(transformer.parameters(), lr=0.001)
    step_times = []
    comms_times = []

    if ddp_type == "naive":
        naive_ddp(transformer, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times)
    if ddp_type == "flat_ddp":
        flat_ddp(transformer, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times)
    if ddp_type == "individual_ddp":
        individual_ddp(transformer, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times)
    if ddp_type == "bucketed_ddp":
        bucketed_ddp(transformer, data, optimizer, num_trails, num_warmup_trails, step_times, comms_times, bucket_size_mb)

    step_time = torch.tensor(step_times, device=device)
    gathered_steps = [torch.zeros_like(step_time) for _ in range(world_size)]
    dist.all_gather(gathered_steps, step_time)

    comm_time = torch.tensor(comms_times, device=device)
    gathered_comms = [torch.zeros_like(comm_time) for _ in range(world_size)]
    dist.all_gather(gathered_comms, comm_time)

    if rank == 0:
        steps = [x for t in gathered_steps for x in t.cpu().tolist()]
        comms = [x for t in gathered_comms for x in t.cpu().tolist()]
        result_queue.put((steps, comms))
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp_type", type=str, default="naive", choices=["naive", "flat_ddp", "individual_ddp", "bucketed_ddp"])
    parser.add_argument("--bucket_size_mb", type=float, default=100)
    args = parser.parse_args()

    print (f"ddp_type:{args.ddp_type}")
    batch_size = 32
    context_length = 128
    d_model = 1600
    d_ff = 6400
    num_heads = 25
    num_layers = 48
    vocab_size = 10000

    world_size = 2
    num_trails = 10
    num_warmup_trails = 5
    full_data = torch.randint(0, vocab_size, (batch_size, context_length))

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    result_queue = manager.Queue()

    mp.spawn(benchmark_driver, args=(world_size, full_data, num_layers, batch_size,
                     num_trails, num_warmup_trails, vocab_size, context_length,
                     d_model, num_heads, d_ff, args.ddp_type, args.bucket_size_mb,result_queue),
                     nprocs=world_size,
                     join=True)
    
    step_times, comms_times = result_queue.get()

    print (f"Average step time:{sum(step_times) * 1000 / len(step_times):.2f} ms.")
    print (f"Average communication time:{sum(comms_times) * 1000 / len(step_times):.2f} ms.")
