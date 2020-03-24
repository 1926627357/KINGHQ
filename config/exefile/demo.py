# print("HHHHHHHHHHHHHHHHHHHHHHHHH")
import sys
sys.path.append('/home/haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
from KINGHQ.utils.utils import Log,Bar,Dice,DistSampler
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# # it's just a demo for me to fix some bugs
# # 


KINGHQ.init()
CUDA=True
EPOCH=140
device = torch.device('cuda:{}'.format(KINGHQ.local_rank()) if CUDA else 'cpu')
kwargs = {'pin_memory': True,'num_workers': 2} if CUDA else {}
# # In fact the rank is the worker rank
# # the size is the worker size
rank=KINGHQ.rank()
size=KINGHQ.size()

model=mobilenetv2.MobileNetV2().to(device)
# check_point=torch.load('/home/haiqwa/Documents/KINGHQ/config/mod_optim/mobilenetv2')
# model.load_state_dict(check_point['state_dict'])
# model.train()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)

# optimizer.load_state_dict(check_point['optimizer'])
loss_function = nn.CrossEntropyLoss()
optimizer=KINGHQ.KINGHQ_Optimizer(optimizer,model,{"consistency": "BSP","op":"SUM","staleness":0})
LOG_PATH='/home/haiqwa/Documents/KINGHQ/log/BSP_9W_9S.csv'


# train_dataset = \
#     datasets.MNIST('~/Documents/.datasets/MNIST'+'data-%d' % KINGHQ.rank(), train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))

train_dataset = \
    datasets.CIFAR10('~/Documents/.datasets/CIFAR10'+'data-%d' % KINGHQ.rank(), train=True, download=True,
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
# # Horovod: use DistributedSampler to partition the training data.
# train_sampler = torch.utils.data.distributed.DistributedSampler(
#         train_dataset, num_replicas=KINGHQ.size(), rank=KINGHQ.rank(),shuffle=True)
train_sampler = DistSampler(train_dataset,num_replicas=KINGHQ.size(),rank=KINGHQ.rank(),shuffle=True,total_epoch=EPOCH)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler, drop_last=True, **kwargs)



import time
if rank==0:
    bar=Bar(total=len(train_loader), description=' worker progress')
log=Log(title='Single machine',\
            Axis_title=['iterations', 'time', 'accuracy'],\
            path=LOG_PATH,\
            step=21)
Dice=Dice(6)
iteration=0
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        if CUDA:
            data, target = data.to(device), target.to(device)
        iteration+=1

        optimizer.zero_grad()
        # start_time=time.time()
        
        output = model(data)
        loss = loss_function(output, target)
        
        
        # predict=torch.argmax(output, dim=1)
        # accuracy=float(torch.sum(predict == target))/data.size(0)
        log.log([iteration/1, time.time(), loss.item()])
        
        loss.backward()
        optimizer.step()
        
        if rank==0:
            bar()

        if batch_idx%size==rank:
            time.sleep(0.7)

if rank==0:
    log.data_processing('interval', data=log.get_column_data('time'))
    log.data_processing('rolling_mean',data=log.get_column_data('accuracy'),cycle=12)
    log.write()



print("worker:%d done"%rank)

if rank==0:
    KINGHQ.shut_down()
        




