__version__='0.0.1'

from KINGHQ.utils.utils import Utils
from KINGHQ.role.worker import Worker
from KINGHQ.role.server import Server
import torch
import os
util=Utils()

# similar to horovod
init=util.init
# rank=util.get_rank
# size=util.get_size

rank=util.get_worker_rank
size=util.get_worker_size

load_strategy=util.load_strategy
KVStore_=util.get_KVStore
load_strategy=util.load_strategy
strategy=load_strategy('/home/v-haiqwa/Documents/KINGHQ/strategy/SSP_3.json')
KVStore=KVStore_()
# worker_size=util.get_worker_size
# worker_rank=util.get_worker_rank
class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self,params,worker):
        super(self.__class__, self).__init__(params)
        self.worker=worker
    def step(self):
        self.worker.do_()

    def zero_grad(self):
        super(self.__class__, self).zero_grad()


def KINGHQ_Optimizer(Optimizer):
    cls = type(Optimizer.__class__.__name__, (Optimizer.__class__,),
                dict(_DistributedOptimizer.__dict__))

    if os.environ['KINGHQ_Distributed']=='False':
        # single process
        matrix=[[1]]
        worker=Worker(KVStore,matrix,Optimizer,util.get_rank(),strategy)
        worker.register_KVStore()
        return cls(Optimizer.param_groups,worker)

    if strategy['network']['structure']['parameter server']:
        # specify parameter server model
        if util.get_rank()==0:
            # generate worker rank list
            worker_rank=[i for i in range(1,util.get_size())]
            # print(worker_rank)
            server=Server(KVStore,Optimizer,worker_rank,strategy)
            server.register_KVStore()
            while True:
                server.do_()
        else:
            if strategy['network']['matrix']:
                matrix = strategy['network']['matrix']
            else:
                matrix=[
                    [0 for i in range(util.get_size())] for j in range(util.get_size())
                ]
                for i in range(1,util.get_size()):
                    matrix[i][0]=1
                    matrix[0][i]=1
                
            worker=Worker(KVStore,matrix,Optimizer,util.get_rank(),strategy)
            worker.register_KVStore()
            return cls(Optimizer.param_groups,worker)
        