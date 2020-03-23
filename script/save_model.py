
import sys
sys.path.append('/home/haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
import torch

model=mobilenetv2.mobilenetv2()
optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
state={
    'epoch':0,
    'state_dict':model.state_dict(),
    'optimizer':optimizer.state_dict()
}
path='/home/haiqwa/Documents/KINGHQ/config/mod_optim/mobilenetv2'

torch.save(state,path)