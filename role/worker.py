import torch.distributed as dist
import torch
from KINGHQ.role import Role
import queue
import threading
from KINGHQ.core.core import Core
from KINGHQ.msg.msg import PushReqMsg,PushResMsg,PullReqMsg,PullResMsg
class Worker(Role):
    def __init__(self, util, optimizer, model):
        self.util=util
        self.model=model
        self.optimizer=optimizer
        super().__init__(util.get_KVStore())
        self.comm_queue=queue.Queue()
        self.mailbox=threading.Thread(target=self.loop_)
        self.key_res={}
        self.core=Core()
    def init(self):

        self.mailbox.start()
        self.param_rank_map=self.util.partition_model(self.optimizer)
        self.register_KVStore()

        self.paramkey_lock={key: threading.Lock() for _,key in self.param_key_map.items()}

        # register the backward and forward hook function
        self.register_bhook()
        self.register_fhook()
        # if self.util.role=="masterworker":
        #     print("PHASE 3 PARTITION THE MODEL")
        #     rank_paramkey_map={i:[] for i in self.util.servers}
        #     for param,rank in self.param_rank_map.items():
        #         rank_paramkey_map[rank].append(self.param_key_map[param])
        #     rank_handles_map={}
        #     for rank,paramkey in rank_paramkey_map.items():
        #         handle0=dist.isend(tensor=torch.tensor([len(paramkey)],dtype=torch.float),dst=rank,tag=0)
        #         handle1=dist.isend(tensor=torch.tensor(paramkey,dtype=torch.float),dst=rank,tag=1)
        #         rank_handles_map[rank]=[handle0,handle1]
        #     for rank,handles in rank_paramkey_map.items():
        #         for each in handles:
        #             each.wait()
        #         print("RANK: {} has recv the solution".format(rank))
            
        #     print("*"*50)
        # self.util.barrier()

    def b_hook(self, p):
        def hook(* ignore):
            # acquire the lock before send it
            self.paramkey_lock[self.param_key_map[p]].acquire()
            msg=PushReqMsg(key=self.param_key_map[p],value=p.grad,src=self.util.world_rank,dst=self.param_rank_map[p])
            self.core.post(msg=msg,ctx=self)
        return hook
    def register_bhook(self):
        # _requires_update = [] and
        # _grad_accs = [] is used to fix the bug
        self._requires_update = []
        self._grad_accs = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.append(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self.b_hook(p))
                    self._grad_accs.append(grad_acc)
    
    def register_fhook(self):
        submodel=self.util.get_submodel(self.model)
        def hook(mod,input):
            for p in mod.parameters():
                self.paramkey_lock[self.param_key_map[p]].acquire()
                if self.param_key_map[p] in self.key_res.keys():
                    value=self.key_res[self.param_key_map[p]].value
                    p.detach().zero_().add_(value)
                self.paramkey_lock[self.param_key_map[p]].release()
        for submod in submodel:
            submod.register_forward_pre_hook(hook)

    def register_KVStore(self):
        # note that: here, we only register the key-value pairs that only worker
        # should know, while the server doesn't need to know that
        super().register_KVStore()
        
        self.param_key_map={}
        # register the parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                key=self.KVStore.register_new_key(value=p,name="params")
                self.param_key_map[p]=key

    def loop_(self):
        while True:
            msg = self.comm_queue.get()
            if msg is not None:
                Res=msg.wait()
                self.key_res[Res.key]=Res
                self.paramkey_lock[Res.key].release()
            else:
                break
    
    def shutdown(self):
        self.comm_queue.put(None)

    def do_(self):
        # do jobs as the flow chart designs
        # push->apply->pull
        self.clock+=1
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.paramkey_lock[self.param_key_map[p]].acquire()
                req=PullReqMsg(key=self.param_key_map[p],version=0,src=self.util.world_rank,dst=self.param_rank_map[p],ctx=self)
                self.core.post(msg=req,ctx=self)

if __name__ == "__main__":
    pass