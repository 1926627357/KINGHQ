import torch.distributed as dist
import torch

dist.init_process_group(backend='gloo')
print("I'am the rank:%d"%dist.get_rank())