import torch.distributed as dist
import torch
import os
# import torchvision.datasets
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '8889'
# os.environ['RANK']='0'
# os.environ['WORLD_SIZE']='1'
dist.init_process_group(backend='mpi')

t = torch.tensor([float(dist.get_rank())])
dist.broadcast(t,src=0)
if dist.get_rank()==0:
  dist.send(tensor=torch.tensor([1.,2.,3.]),dst=1,tag=0)
else:
  dist.recv(tensor=torch.tensor([1.,2.,3.]),src=0,tag=0)
import time
time.sleep(1)
print("I'am the rank:%d"%dist.get_rank())
print(t.tolist())
