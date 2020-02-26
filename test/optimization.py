import sys
sys.path.append('/home/v-haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
import torch

model=vgg.vgg19()
optimizer=torch.optim.SGD(model.parameters(),lr=0.002)
param_size=0

for group in optimizer.param_groups:
    for p in group['params']:
        size=1
        for each in p.data.size():
            size*=each
            # print(size)
            param_size+=size
print(param_size)


# model=lenet.LeNet5()
# optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
# print(optimizer.state_dict()['param_groups'][0]['params'][0])