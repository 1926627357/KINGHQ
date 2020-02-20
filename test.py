# # import fcntl
# # import time

# with open("/home/v-haiqwa/Documents/KINGHQ/config/recv/10.150.144.122","r+") as f:
#     fcntl.flock(f.fileno(), fcntl.LOCK_EX)
#     l = f.readlines()
#     read=l.pop(0)
#     f.seek(0)
#     f.writelines(l)
#     time.sleep(2)

# # print(read.replace('\n',''))
# import torch.distributed as dist
# dist.init_process_group(backend='mpi')
# print(dist.get_world_size())
import sys
sys.path.append('/home/v-haiqwa/Documents/')
from KINGHQ.utils.utils import Utils

comm = Utils()
comm.init()
