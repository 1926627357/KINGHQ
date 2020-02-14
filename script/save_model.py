
import sys
sys.path.append('/home/v-haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet
import torch

model=lenet.LeNet5()
optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
state={
    'epoch':0,
    'state_dict':model.state_dict(),
    'optimizer':optimizer.state_dict()
}
path='/home/v-haiqwa/Documents/KINGHQ/model_state/Lenet'

torch.save(state,path)