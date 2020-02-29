import sys
sys.path.append('/home/haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
import torch
import torch.distributed as dist

dist.init_process_group(backend="mpi")
device="cuda:%d"%dist.get_rank()

#device="cpu"
model=vgg.vgg19().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.002)
import time

t1=time.time()
n=0
for group in optimizer.param_groups:
  for p in group['params']:
    n+=1
    if p.requires_grad:
      #print("hello!!!%d"%dist.get_rank())
      if dist.get_rank()==0:
        p_temp=p.to("cpu")
        t=torch.tensor([1.,2.,3.])
        #print("I am-%d: I try to send [1,2,3]"%dist.get_rank())
        dist.send(tensor=t,dst=1,tag=1+n*4)
        #print("I am-%d: I try to send param"%dist.get_rank())
        dist.send(tensor=p_temp,dst=1,tag=2+n*4)
        #print("I am-%d: I try to recv param"%dist.get_rank())
        dist.recv(tensor=p_temp,src=1,tag=3+n*4)
        p_temp=p_temp.to(device)
        p.detach().add_(p_temp)
      else:
        p_temp=torch.randn(p.size())
        t=torch.tensor([1.,2.,3.])
        #print("I am-%d: I try to recv [1,2,3]"%dist.get_rank())
        dist.recv(tensor=t,src=0,tag=1+n*4)
        #print("I am-%d: I try to recv param"%dist.get_rank())
        dist.recv(tensor=p_temp,src=0,tag=2+n*4)
        #print("I am-%d: I try to send param"%dist.get_rank())
        dist.send(tensor=p_temp,dst=0,tag=3+n*4)
    
t2=time.time()
if dist.get_rank()==0:
    print(t2-t1)