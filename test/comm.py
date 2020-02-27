import sys
sys.path.append('/home/haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
import torch
import torch.distributed as dist

dist.init_process_group(backend="mpi")
device="cuda:%d"%dist.get_rank()
print("hello world")
# device="cpu"
model=vgg.vgg19().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.002)
import time

t1=time.time()
for group in optimizer.param_groups:
    for p in group['params']:
        if dist.get_rank()==0:
            p_temp=p.to("cpu")
            dist.send(tensor=torch.tensor([1.,2.,3.]),dst=1,tag=0)
            dist.send(tensor=p_temp,dst=1,tag=1)
            dist.recv(tensor=p_temp,src=1,tag=2)
            p_temp=p_temp.to(device)
            p.detach().add_(p_temp)
        else:
            p_temp=torch.randn(p.size())
            dist.recv(tensor=torch.tensor([1.,2.,3.]),src=0,tag=0)
            dist.recv(tensor=p_temp,src=0,tag=1)
            dist.send(tensor=p_temp,dst=0,tag=2)
t2=time.time()
if dist.get_rank()==0:
    print(t2-t1)