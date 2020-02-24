import sys
sys.path.append('/home/v-haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet
import torch

model=lenet.LeNet5()
optimizer=torch.optim.Adadelta(model.parameters(), lr=0.002)

print(model.state_dict())
# for group in optimizer.param_groups:
#     for p in group['params']:
#         del p
#         break
#     break
# n=0
# for group in optimizer.param_groups:
#     for p in group['params']:
#         n+=1
# print(n)

        # p.grad=p.data.new(p.size()).zero_()
    # print(group)
# optimizer.step()

# print(optimizer.state_dict())
print(optimizer.__module__)
print(optimizer.__class__)
print(optimizer.__class__.__name__)