import torch
import torch.distributed as dist

dist.init_process_group(backend="MPI")

t=torch.tensor([dist.get_rank()],dtype=torch.float)

if dist.get_rank()==0:
    handle=dist.send(t,dst=1)
else:
    handle=dist.irecv(t,src=0)
    # irecv的src不能够为none
    print(handle.wait())
    print(t)
dist.barrier()