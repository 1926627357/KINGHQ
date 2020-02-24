import sys
sys.path.append('/home/v-haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet
from KINGHQ.utils.utils import Log,Bar,Dice
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# it's just a demo for me to fix some bugs
# 

KINGHQ.init()

# In fact the rank is the worker rank
# the size is the worker size
rank=KINGHQ.rank()
size=KINGHQ.size()

# '~/Documents/pytorch_project/dataset/MNIST'

train_dataset = \
    datasets.MNIST('~/Documents/.datasets/MNIST'+'data-%d' % KINGHQ.rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=KINGHQ.size(), rank=KINGHQ.rank(), shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler)

model=lenet.LeNet5()
model.train()
optimizer=torch.optim.SGD(model.parameters(), lr=0.002)

check_point=torch.load('/home/v-haiqwa/Documents/KINGHQ/config/mod_optim/Lenet')
model.load_state_dict(check_point['state_dict'])
optimizer.load_state_dict(check_point['optimizer'])


loss_function = nn.CrossEntropyLoss()
optimizer=KINGHQ.KINGHQ_Optimizer(optimizer,model)


import time

if rank==0:
    bar=Bar(total=len(train_loader)*10, description=' worker progress')
    log=Log(title='Single machine',\
            Axis_title=['iterations', 'time', 'accuracy'],\
            path='/home/v-haiqwa/Documents/KINGHQ/log/BSP_w3.csv',\
            step=21)
Dice=Dice(6)
iteration=0
for epoch in range(10):
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration+=1

        optimizer.zero_grad()
        # start_time=time.time()
        
        output = model(data)
        loss = loss_function(output, target)
        
        if rank==0:
            predict=torch.argmax(output, dim=1)
            accuracy=float(torch.sum(predict == target))/data.size(0)
            log.log([iteration/1, time.time(), accuracy])
            
        loss.backward()
        # time.sleep(5)
        # if rank==0:
        #     print("computing:%d"%(time.time()-start_time))

        # start_time=time.time()
        
        
            
            # for group in optimizer.param_groups:
            #     for p in group['params']:
            #         p.grad/=4
        
        # time.sleep(0.005*Dice()-0.002)
        optimizer.step()
        
        
        # if rank==0:
        #     print("communication:%d"%(time.time()-start_time))

        
        if rank==0:
            bar()

if rank==0:
    log.data_processing('interval', data=log.get_column_data('time'))
    log.data_processing('rolling_mean',data=log.get_column_data('accuracy'),cycle=12)
    log.write()



print("worker:%d done"%rank)
        




