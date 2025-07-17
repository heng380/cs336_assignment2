import torch.distributed as dist
import torch
import copy

class Bucket:
    def __init__(self, num_params: int):
        self.num_params = num_params
        self.params = []

    def add_param(self, param):
        self.params.append(param)
        if len(self.params) == self.num_params:
            output = self.all_reduce_params()
            self.params = []
            return output
        return None

    def all_reduce_params(self):
        flatten_grads = torch._utils._flatten_dense_tensors(tensors=[p.grad for p in self.params])
        flatten_grads /= dist.get_world_size()
        handle = dist.all_reduce(flatten_grads, op=dist.ReduceOp.SUM, async_op=True)
        return handle, self.params, flatten_grads

class DDP_Bucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb):
        super().__init__()
        self.module = module
        self.bucket_results = []
        self.param_to_bucket = {}

        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.all_buckets = []

        self.initialize_parameters()

    def initialize_parameters(self):
        # initialize all parameters to be the same
        curr_bytes = 0
        curr_bucket_params = []
        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)
            # only deal with parameters that require gradients
            if not param.requires_grad:
                continue

            curr_bucket_params.append(param)
            curr_bytes += param.data.nbytes
            if curr_bytes >= self.bucket_size_bytes:
                bucket = Bucket(len(curr_bucket_params))
                self.all_buckets.append(bucket)
                for param in curr_bucket_params:
                    self.param_to_bucket[param] = bucket

                curr_bucket_params = []
                curr_bytes = 0

            # register hook for this param
            param.register_post_accumulate_grad_hook(self.transform_grad)

        if curr_bucket_params:
            bucket = Bucket(len(curr_bucket_params))
            self.all_buckets.append(bucket)
            for param in curr_bucket_params:
                self.param_to_bucket[param] = bucket

    # when this is called, each parameter will be mapped to a bucket
    def transform_grad(self, param):
        info = self.param_to_bucket[param].add_param(param)
        if info is not None:
            self.bucket_results.append(info)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, params, flattened_grads in self.bucket_results:
            handle.wait()
            unflatten = torch._utils._unflatten_dense_tensors(flattened_grads, params)
            for param, unflattened_grad in zip(params, unflatten):
                param.grad = unflattened_grad

        self.bucket_results.clear()