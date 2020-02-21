# # import fcntl
# # import time


import sys
sys.path.append('/home/v-haiqwa/Documents/')
from KINGHQ.utils.utils import Utils

comm = Utils()
comm.init()

print(comm.get_worker_rank)



