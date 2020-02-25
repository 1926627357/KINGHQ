import torch
import torch.distributed as dist

dist.init_process_group(backend="MPI")

t=torch.tensor([dist.get_rank()],dtype=torch.float)
import time
if dist.get_rank()==0:
    time.sleep(0.1)
    dist.send(t,dst=1)
else:
    handle=dist.irecv(t,src=0)
    # irecv的src不能够为none
    # print(handle.wait())
    while not handle.is_completed():
        print(False)
    print("hhhhhh",handle.is_completed())
    print("hhhhhh",handle.is_completed())
    print("hhhhhh",handle.is_completed())
    print(t)
dist.barrier()