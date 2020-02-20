import torch.distributed as dist
import torch
import os
# import torchvision.datasets
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '8889'
# os.environ['RANK']='0'
# os.environ['WORLD_SIZE']='1'
dist.init_process_group(backend='mpi')
print("I'am the rank:%d"%dist.get_rank())
t=torch.tensor()