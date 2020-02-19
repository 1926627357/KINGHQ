import torch.distributed as dist
import torch

dist.init_process_group(backend='gloo', init_method='tcp://10.150.144.122:30000')
print("I'am the rank:%d"%dist.get_rank())