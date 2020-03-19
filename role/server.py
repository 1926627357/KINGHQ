import torch.distributed as dist
import torch
from KINGHQ.role import Role
import threading
import queue
from KINGHQ.msg.msg import ReqMsg,ResMsg
import time


class Server(Role):
    def __init__(self, util, optimizer, strategy):
        # worker_rank: denote all workers' ranks in a format of list
        super().__init__(util.get_KVStore())
        self.LOG=False
        self.strategy=strategy
        self.optimizer=optimizer
        self.util=util
        
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
        self.clock_vector={key:{rank:0 for rank in self.util.workers} for key in self.my_param_keys}
        self.global_clock={key:0 for key in self.my_param_keys}

        if self.strategy['consistency']=="BSP":
            self.buffer={key:None for key in self.my_param_keys}

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
        ReqMsg_queue=queue.Queue()
        for rank in self.util.workers:
            ReqMsg_queue.put(ReqMsg(src=rank,dst=self.util.world_rank,ctx=self))
        for rank in self.util.master:
            ReqMsg_queue.put(ReqMsg(src=rank,dst=self.util.world_rank,ctx=self))
        
        while True:
            # msg=ReqMsg(ctx=self)
            # msg.recv()
            # self.request_queue.put(msg)
            msg=ReqMsg_queue.get()
            if msg.status=="init":
                msg.recv_head()
            else:
                if msg.is_completed():
                    if msg.status=="recv_head":
                        msg.recv_value()
                        
                    elif msg.status=="recv_value":
                        self.request_queue.put(msg)
                        if self.LOG:
                            print("I recv "+msg.type+"from the worker-%d"%msg.src, time.time())
                        msg=ReqMsg(src=msg.src,dst=self.util.world_rank,ctx=self)
            ReqMsg_queue.put(msg)

    def loop_Outbox(self):
        # to send the tensor to the specific workers
        while True:
            Res=self.response_queue.get()
            if Res.status=="init":
                Res.send()
                self.response_queue.put(Res)
            elif Res.status=="send":
                if Res.is_completed():
                    pass
                else:
                    self.response_queue.put(Res)

    def aggregate(self,req,op=None):
        if self.strategy['consistency']=="ASP":
            pass
        elif self.strategy['consistency']=="BSP":
            if op is None or op=="SUM":
                if self.buffer[req.key] is None:
                    self.buffer[req.key]=req.value
                else:
                    self.buffer[req.key].add_(req.value)
            elif op=="Average":
                if self.buffer[req.key] is None:
                    self.buffer[req.key]=req.value/len(self.util.workers)
                else:
                    self.buffer[req.key].add_(req.value/len(self.util.workers))
        elif self.strategy['consistency']=="SSP":
            pass

    def apply(self,req):
        if self.strategy['consistency']=="ASP":
            self.KVStore(req.key)[req.key].grad=req.value
            self.optimizer.step()
            self.KVStore(req.key)[req.key].grad=None
        elif self.strategy['consistency']=="BSP":
            if max(self.clock_vector[req.key].values())==self.global_clock[req.key]:
                # apply when the server aggregate all the gradients from workers
                self.KVStore(req.key)[req.key].grad=self.buffer[req.key]
                self.optimizer.step()
                self.buffer[req.key]=None
                self.KVStore(req.key)[req.key].grad=None
        elif self.strategy['consistency']=="SSP":
            pass
    def check(self,req):
        if self.strategy['consistency']=="ASP":
            return True
        elif self.strategy['consistency']=="BSP":
            if self.clock_vector[req.key][req.src]-1<=self.global_clock[req.key]:
                # when the requester run no more 0 step than the slowest one
                return True
            else:
                return False
        elif self.strategy['consistency']=="SSP":
            pass
    
    def do_(self):
        
        while True:
            Req=self.get_request()
            if Req.type=="PushReqMsg":
                # Res=ResMsg(msgtype="PushResMsg")
                # self.response_queue.put(Res)
                self.clock_vector[Req.key][Req.src]+=1
                self.global_clock[Req.key]=min(self.clock_vector[Req.key].values())
                if self.LOG:
                    print("I get the Push request from the queue, which is from worker-%d:  "%Req.src,time.time())
                self.aggregate(req=Req)
                self.apply(req=Req)
                # self.KVStore(Req.key)[Req.key].grad=Req.value
                # self.optimizer.step()
                # self.KVStore(Req.key)[Req.key].grad=None
            elif Req.type=="PullReqMsg":
                # value=self.KVStore(Req.key)[Req.key].detach().clone()
                # Res=ResMsg(msgtype="PullResMsg",key=Req.key,value=value,src=Req.dst,dst=Req.src)
                # self.response_queue.put(Res)
                if self.LOG:
                    print("I recv the request from the queue, which comes from worker-%d:  "%Req.src,time.time())
                if self.check(Req):
                    if self.LOG:
                        print("worker-%d can Pull the parameter:  "%Req.src,time.time())
                    value=self.KVStore(Req.key)[Req.key].detach().clone()
                    Res=ResMsg(msgtype="PullResMsg",key=Req.key,value=value,src=Req.dst,dst=Req.src)
                    self.response_queue.put(Res)
                else:
                    self.request_queue.put(Req)

if __name__ == "__main__":
    pass