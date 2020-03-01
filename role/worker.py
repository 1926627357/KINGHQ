import torch.distributed as dist
import torch
from KINGHQ.role import Role
import queue
import threading
from KINGHQ.core.core import Core
from KINGHQ.msg.msg import PushReqMsg,PushResMsg,PullReqMsg,PullResMsg
class Worker(Role):
    def __init__(self, util, optimizer, model, strategy):
        self.strategy= strategy
        self.util=util
        self.model=model
        self.optimizer=optimizer
        super().__init__(util.get_KVStore())
        self.comm_queue=queue.Queue()
        self.mailbox=threading.Thread(target=self.loop_)
        
        self.core=Core()
    def init(self):
        
        self.mailbox.start()
        
        self.param_rank_map=self.util.partition_model(self.optimizer)
        # print(self.param_rank_map)
        self.register_KVStore()
        
        self.paramkey_lock={key: threading.Lock() for _,key in self.param_key_map.items()}

        # register the backward and forward hook function
        self.register_bhook()
        self.register_fhook()


    def b_hook(self, p):
        def hook(* ignore):
            # acquire the lock before send it
            if self.strategy['consistency']=='ASP':
                pass
            else:
                self.paramkey_lock[self.param_key_map[p]].acquire()
            
            msg=PushReqMsg(key=self.param_key_map[p],value=p.grad,src=self.util.world_rank,dst=self.param_rank_map[p],ctx=self)
            # print(self.param_key_map[p])
            # print(self.util.world_rank)
            
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
                # if self.strategy['consistency']=='ASP':
                #     pass
                # # ASP: no need to wait for pull
                # else:
                self.paramkey_lock[self.param_key_map[p]].acquire()
                    # print("worker: I'm in the forward-key:{}".format(self.param_key_map[p]))
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
        
    def handle_res(self,res):
        if res.type=='PullResMsg':
            # pull task completes
            p=self.KVStore(res.key)[res.key]
            p.detach().zero_().add_(res.value)
            
            # release the lock when the task completes
            self.paramkey_lock[res.key].release()
        else:
            # push task completes
            if self.strategy['consistency']=='ASP':
                pass
            else:
                self.paramkey_lock[res.key].release()

    def loop_(self):
        while True:
            msg = self.comm_queue.get()
            if msg is not None:
                if msg.status=="prepare":
                    
                    msg.send()
                    self.comm_queue.put(msg)
                elif msg.status=="send":
                    
                    if msg.is_completed():
                        # this msg has been completed
                        Res=msg.get_response()
                        self.handle_res(Res)
                        
                    else:
                        # if msg.key==12 and msg.type=="PullReqMsg":
                        #     print("Not Completed")
                        self.comm_queue.put(msg)
            else:
                break
    
    def shutdown(self):
        self.comm_queue.put(None)

    def do_(self):
        # do jobs as the flow chart designs
        # push->apply->pull
        self.clock+=1
        # print("begin to pull")
        self.optimizer.step()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                
                self.paramkey_lock[self.param_key_map[p]].acquire()
                # print("send pull req")
                
                req=PullReqMsg(key=self.param_key_map[p],version=0,src=self.util.world_rank,dst=self.param_rank_map[p],ctx=self)
                self.core.post(msg=req,ctx=self)
        

if __name__ == "__main__":
    pass