import sys
sys.path.append('/home/v-haiqwa/Documents/')
from KINGHQ.models import lenet
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn
import queue
model=lenet.LeNet5()

def get_all_children(model):
    q = queue.LifoQueue()
    submodule=[]
    for mod in model.children():
        q.put(mod)
    while not q.empty():
        mod = q.get()
        if len(list(mod.children())) == 0:
            submodule.append(mod)
        else:
            for m in mod.children():
                q.put(m)
    return submodule


submodel = get_all_children(model)# 子模块还包括了激励函数，激励函数没有参数
print(submodel[2])
for p in submodel[1].parameters():
    print(p)