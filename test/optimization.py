import sys
sys.path.append('/home/haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
import torch

model=mobilenetv2.mobilenetv2()

optimizer=torch.optim.SGD(model.parameters(),lr=0.002)
param_size=0

layer=0
size_count={}
for group in optimizer.param_groups:
  for p in group['params']:
    layer+=1
    size=1
    for each in p.data.size():
      size*=each
    if size in size_count:
      size_count[size]+=1
    else:
      size_count[size]=1
    print("layer NO.{}--{}".format(layer,size))
    param_size+=size
print("the whole model size is: ",param_size)
size_count=sorted(size_count.items(),key = lambda item:item[0])
print(size_count)

# model=lenet.LeNet5()
# optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
# print(optimizer.state_dict()['param_groups'][0]['params'][0])