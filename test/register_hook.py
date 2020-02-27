import sys
sys.path.append('/home/haiqwa/Documents/')
from KINGHQ.models import vgg,lenet
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn




dist.init_process_group(backend='gloo')

train_dataset = \
    datasets.MNIST('~/Documents/pytorch_project/dataset/MNIST'+'data-%d'%dist.get_rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler)

model=lenet.LeNet5()
model.train()
optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
loss_function = nn.CrossEntropyLoss()
class a:
    def __init__(self):
        self.n=0
n=a()
def make_hook(p):
    
    def hook(* ignore):
        print(p.grad)
        
    return hook

def printf(p,grad):
    print(grad)

_requires_update = []
_grad_accs = []
for group in optimizer.param_groups:
    for p in group['params']:
        if p.requires_grad:
            p.grad = p.data.new(p.size()).zero_()
            _requires_update.append(p)
            p_tmp = p.expand_as(p)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(make_hook(p))
            _grad_accs.append(grad_acc)


for data, target in train_loader:

    output = model(data)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    break
    
    
    

