__version__='0.0.2'

from KINGHQ.utils.utils import Utils
from KINGHQ.role.worker import Worker
from KINGHQ.role.server import Server
import torch
import os
util=Utils()

# similar to horovod
init=util.init
rank=util.get_worker_rank
size=util.get_worker_size
broadcast_model=util.broadcast_model

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self,params,worker):
        super(self.__class__, self).__init__(params)
        self.worker=worker
    def step(self):
        self.worker.do_()

    def zero_grad(self):
        super(self.__class__, self).zero_grad()


def KINGHQ_Optimizer(Optimizer, model):
    cls = type(Optimizer.__class__.__name__, (Optimizer.__class__,),
                dict(_DistributedOptimizer.__dict__))
    # return cls(Optimizer.param_groups,worker)

    if util.role=="master":
        pass
    elif util.role=="server":
        model.to("cpu")
        server=Server(util=util,optimizer=Optimizer)
        server.init()
        server.do_()


    elif util.role=="masterworker" or util.role=="worker":
        # partition the model firstly!
        
        worker=Worker(util=util, optimizer=Optimizer, model=model)
        worker.init()
        # worker.do_()
        return cls(Optimizer.param_groups,worker)
        

    
    
        