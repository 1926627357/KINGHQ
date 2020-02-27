import sys
sys.path.append('/home/haiqwa/Documents/')
from KINGHQ.models import lenet
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from submodel import get_all_children



train_dataset = \
    datasets.MNIST('~/Documents/pytorch_project/dataset/MNIST'+'data-0', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=1, rank=0, shuffle=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler)

model=lenet.LeNet5()
model.train()
optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
loss_function = nn.CrossEntropyLoss()

def pre_forward_hook(mod, input):
    for p in mod.parameters():
        print(p.grad)
        

submodels = get_all_children(model)
for each in submodels:
    each.register_forward_pre_hook(pre_forward_hook)


for group in optimizer.param_groups:
    for p in group['params']:
        if p.requires_grad:
            p.grad = p.data.new(p.size()).zero_()



for data, target in train_loader:

    output = model(data)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    break
