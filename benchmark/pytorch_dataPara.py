import sys
sys.path.append('/home/v-haiqwa/Documents/')
from KINGHQ.models import vgg,lenet
from KINGHQ.utils.utils import Log,Bar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
dist.init_process_group(backend='gloo')
rank = dist.get_rank()
size = dist.get_world_size()
CUDA=False
device=torch.device('cuda:{}'.format(hvd.local_rank()) if CUDA else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

train_dataset = \
    datasets.MNIST('~/Documents/pytorch_project/dataset/MNIST'+'data-%d' % rank, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=size, rank=rank, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler, **kwargs)

model=lenet.LeNet5()
model.train()
optimizer=torch.optim.SGD(model.parameters(), lr=0.002)

check_point=torch.load('/home/v-haiqwa/Documents/KINGHQ/model_state/Lenet')
model.load_state_dict(check_point['state_dict'])
optimizer.load_state_dict(check_point['optimizer'])


model=DDP(model)

optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
loss_function = nn.CrossEntropyLoss()

if rank==0:
    bar=Bar(total=len(train_loader)*10, description=' worker progress')
    log=Log(title='horovod benchmark',\
            Axis_title=['iterations', 'time', 'accuracy'],\
            path='/home/v-haiqwa/Documents/KINGHQ/log/torch_w1.csv',\
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