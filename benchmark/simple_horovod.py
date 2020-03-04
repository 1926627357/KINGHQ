import sys
sys.path.append('/home/haiqwa/Documents/')

from KINGHQ.models import vgg,lenet
from KINGHQ.utils.utils import Log,Bar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import horovod.torch as hvd

import time

CUDA=True

hvd.init()
#hvd.size(), hvd.local_rank(), hvd.rank()
rank=hvd.rank()


device=torch.device('cuda:{}'.format(hvd.local_rank()) if CUDA else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
train_dataset = \
    datasets.CIFAR100('~/Documents/.datasets/CIFAR100'+'data-%d' % hvd.rank(), train=True, download=True,
                        transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(15),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                                            )
                                    ])
                   )

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler, **kwargs)

model=vgg.vgg13().to(device)
model.train()
# lr_scaler=hvd.size()
optimizer = optim.SGD(model.parameters(), lr=0.002)
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters())
loss_function = nn.CrossEntropyLoss()
if rank==0:
    print(len(train_loader))
    bar=Bar(total=len(train_loader)*10, description=' worker progress')
    log=Log(title='horovod benchmark',\
            Axis_title=['iterations', 'time', 'accuracy'],\
            path='/home/haiqwa/Documents/KINGHQ/log/hvd_w1.csv',\
            step=21)
iteration=0
for epoch in range(10):
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration+=1
        if CUDA:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if rank==0:
            predict=torch.argmax(output, dim=1)
            accuracy=float(torch.sum(predict == target))/data.size(0)
            log.log([iteration/1, time.time(), accuracy])

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if rank==0:
            bar()
            

if CUDA:
    torch.cuda.synchronize(device)
if rank==0:
    log.data_processing('interval', data=log.get_column_data('time'))
    log.data_processing('rolling_mean',data=log.get_column_data('accuracy'),cycle=12)
    log.write()
print("worker:%d done"%rank)

