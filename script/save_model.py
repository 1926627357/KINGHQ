
import sys
sys.path.append('/home/haiqwa/Documents/')
import KINGHQ
from KINGHQ.models import vgg,lenet,mobilenetv2
import torch

model=mobilenetv2.mobilenetv2()

path='/home/haiqwa/Documents/KINGHQ/config/mod_optim/mobilenetv2'

torch.save(model.state_dict(),path)