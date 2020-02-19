import sys
sys.path.append('/home/v-haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet
import torch

t = torch.tensor([1.,2.,3.],requires_grad=True)

t.grad = t.data.new(t.size()).zero_()

print(t.is_leaf)


# model=lenet.LeNet5()
# optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
# print(optimizer.state_dict()['param_groups'][0]['params'][0])