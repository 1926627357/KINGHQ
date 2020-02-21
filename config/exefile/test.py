# # import fcntl
# # import time


import sys
sys.path.append('/home/v-haiqwa/Documents/')
from KINGHQ.utils.utils import Utils

comm = Utils()
comm.init()
import os
if comm.get_world_rank()==3:
    print(os.environ)





