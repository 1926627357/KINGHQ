# # import fcntl
# # import time


# import sys
# sys.path.append('/home/v-haiqwa/Documents/')
# from KINGHQ.utils.utils import Utils

# comm = Utils()
# comm.init()

# print(comm.get_worker_rank)


import sys
sys.path.append('/home/v-haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet
import torch

model=lenet.LeNet5()
optimizer=torch.optim.Adadelta(model.parameters(), lr=0.002)

# # model.to("cuda:1")
# def KINGHQ_Optimizer(Optimizer, model):
#     model.to("cpu")
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             param=p
#     print(param)
# for group in optimizer.param_groups:
#     for p in group['params']:
#         param=p
# print(param)
# KINGHQ_Optimizer(optimizer,model)
for group in optimizer.param_groups:
    for p in group['params']:
        param=p
# print(param)
# import torch

# t=torch.tensor([1],device="cuda:0")
# p=torch.randn(t.size())
# print(p)

value=param.detach().clone()
print(value)

# param.set_(torch.randn(param.size()))
#         p.grad=p.data.new(p.size()).zero_()
# optimizer.step()

# print(optimizer.state_dict())
# print(optimizer.__module__)
# print(optimizer.__class__)
# print(optimizer.__class__.__name__)


