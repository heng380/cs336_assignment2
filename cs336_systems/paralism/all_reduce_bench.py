import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def all_reduce_bench(rank, world_size, data_size_mb, backend, benchmark_trails, warmup_trails, result_queue):
    setup(rank, world_size, backend)

    num_elements = data_size_mb * 1024 * 1024 // 4
    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")

    for _ in range(warmup_trails):
        data = torch.randint(0, 10, (num_elements, ), dtype=torch.float32, device=device)
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        
    times = []
    for _ in range(benchmark_trails):
        data = torch.randint(0, 10, (num_elements, ), dtype=torch.float32, device=device)

        start = timer()
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        times.append((timer() - start) * 1000)
    avg_time = sum(times) / len(times)

    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, avg_time)

    if rank == 0:
        final_avg = sum(gathered_times) / len(gathered_times)
        print (f"average all reduce time is {final_avg:.2f} ms")
        if result_queue is not None:
            result_queue.put(round(final_avg, 2))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    backends = ["nccl"]
    # backends = ["gloo"]
    num_procs = [2]
    data_size_mbs = [1,10,100,1024]
    warmup_trails = 5
    benchmark_trails = 10

    for backend in backends:
        results = []
        for num_proc in num_procs:
            for data_size_mb in data_size_mbs:
                print (f"Running {backend} with {num_proc} processes and {data_size_mb} MB data")
                ctx = mp.get_context('spawn')
                result_queue = ctx.Queue()

                mp.spawn(fn = all_reduce_bench,
                         args=(num_proc, data_size_mb, backend, benchmark_trails, warmup_trails, result_queue),
                         nprocs=num_proc,
                         join=True)

                avg_time = result_queue.get()

                results.append({
                    "Data size (MB)": data_size_mb,
                    "Number of Processes": num_proc,
                    "Average Time (ms)": avg_time
                })
    df = pd.DataFrame(results)
    df.to_markdown("gpu.md", index=False)
