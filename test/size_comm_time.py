import torch

import torch.distributed as dist

dist.init_process_group(backend="mpi")
t=torch.randn(1000,1000,dtype=torch.float)
dist.barrier()

import time

start=time.time()

for _ in range(10):
  if dist.get_rank()==0:
    dist.send(tensor=t, dst=1)
  else:
    dist.recv(tensor=t, src=0)

end=time.time()

print((end-start)/10)    
  
