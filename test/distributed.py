import torch.distributed as dist
import torch
import os


dist.init_process_group(backend='gloo', init_method='tcp://10.150.144.122:30000', 
                        rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
print("I'am the rank:%d"%dist.get_rank())