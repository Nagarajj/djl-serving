from collections import defaultdict
import torch
from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
import torch.distributed as dist


def print_rank0(content):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(content)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

properties = {
    "tensor_parallel_degree": 2,
    "dtype": "fp16",
    "max_rolling_batch_size": 8,
    "model_loading_timeout": 7200,
    "max_rolling_batch_prefill_tokens": 10000,
    "paged_attention": "True"
}

model_id = "EleutherAI/gpt-neox-20b"
"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":1000, "do_sample":true, "temperature":0.7}}
"""

input_str = [
    "Memories follow me left and right", "Memories follow me left and right."
]

params = [{
    "max_new_tokens": 50,
    "do_sample": False,
    "temperature": 0.000007
}, {
    "max_new_tokens": 50,
    "do_sample": False,
    "temperature": 0.000007
}]

# ===================== lmi ============================
print("=========== lmi =========")
rolling_batch = SchedulerRollingBatch(model_id, device, properties)
rolling_batch.output_formatter = None
print("reach here")

output_all = defaultdict(list)
result = rolling_batch.inference(input_str, params)
for i, res in enumerate(result):
    output_all[i].append(res['data'])

for _ in range(50):
    result = rolling_batch.inference(input_str, params)
    for i, res in enumerate(result):
        output_all[i].append(res['data'])

for i, out in enumerate(output_all.values()):
    print_rank0(input_str[i] + ''.join(out))
    print_rank0('\n====')
