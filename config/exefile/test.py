# # import fcntl
# # import time


import sys
sys.path.append('/home/v-haiqwa/Documents/')
from KINGHQ.utils.utils import Utils

comm = Utils()
comm.init()

print("local worker size: ",comm.get_local_worker_size)




