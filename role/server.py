import torch.distributed as dist
import torch
from KINGHQ.role import Role
import threading
import queue
from KINGHQ.msg.msg import ReqMsg,ResMsg
class Server(Role):
    def __init__(self, util, optimizer):
        # worker_rank: denote all workers' ranks in a format of list
        super().__init__(util.get_KVStore())
        
        self.optimizer=optimizer
        self.util=util
        self.clock_vector=[0 for _ in range(len(util.workers))]
        self.request_queue=queue.Queue()
        self.response_queue=queue.Queue()
        self.Inbox=threading.Thread(target=self.loop_Inbox)
        self.Outbox=threading.Thread(target=self.loop_Outbox)
        
    def init(self):
        self.Inbox.start()
        self.Outbox.start()
        self.param_rank_map=self.util.partition_model(self.optimizer)
        self.register_KVStore()
        self.my_param_keys=[]
        
        for param,rank in self.param_rank_map.items():
            if rank==self.util.world_rank:
                self.my_param_keys.append(self.param_key_map[param])
        for param, _ in self.param_key_map.items():
            param.grad=None
        self.paramkey_lock={key: threading.Lock() for key in self.my_param_keys}


    def register_KVStore(self):
        super().register_KVStore()
        self.param_key_map={}
        # register the parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                key=self.KVStore.register_new_key(value=p,name="params")
                self.param_key_map[p]=key



    def get_request(self):
        # option: 
        #           1. comm: get request from worker
        #           2. queue: get request from waiting queue
        # get a request
        # return the rank and the kind of request
        request=self.request_queue.get()
        return request
    
    def loop_Inbox(self):
        # to recv the request from all workers or masters
        while True:
            msg=ReqMsg()
            msg.recv()
            self.request_queue.put(msg)

    def loop_Outbox(self):
        # to send the tensor to the specific workers
        while True:
            Res=self.response_queue.get()
            Res.send()

    def do_(self):
        while True:
            Req=self.get_request()
            if Req.type=="PushReqMsg":
                Res=ResMsg(msgtype="PushResMsg")
                self.response_queue.put(Res)
                self.KVStore(Req.key)[Req.key].grad=Req.value
                self.optimizer.step()
                self.KVStore(Req.key)[Req.key].grad=None
            elif Req.type=="PullReqMsg":
                value=self.KVStore(Req.key)[Req.key].detach().clone()
                Res=ResMsg(msgtype="PullResMsg",key=Req.key,value=value,src=Req.dst,dst=Req.src)
                self.response_queue.put(Res)
                
            
        


if __name__ == "__main__":
    pass