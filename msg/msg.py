import torch
import torch.distributed as dist
class ReqMsg(object):
    # all request msg class will inherit this class
    def __init__(self, msgtype=None, key=None, value=None, src=None, dst=None, ctx=None):

        self.type=msgtype
        self.key=key
        self.value=value
        self.src=src
        self.dst=dst
        self.ctx=ctx
        self.comm_tag={"reqres":0}
        self.comm_code={"pushreq":0,"pullreq":1}
    def recv(self):
        # server can use this api to get a request!
        msg=torch.randn(3).to(dtype=torch.float)
        src=dist.recv(tensor=msg,tag=self.comm_tag["reqres"])
        dst=ctx.util.world_rank
        self.src=src
        msg=msg.to(torch.int).tolist()
        if msg[0]==self.comm_code["pushreq"]:
            self.type="PushReqMsg"
            self.key=msg[1]
            p=self.ctx.KVStore(self.key)[self.key]
            self.value = p.data.new(p.size())
            dist.recv(tensor=self.value,src=self.src,tag=msg[2])
        elif msg[0]==self.comm_code["pullreq"]:
            self.type="PullReqMsg"
            self.key=msg[1]
            self.value = torch.tensor([0],dtype=torch.float)
            dist.recv(tensor=self.value,src=self.src,tag=msg[2])
            self.value=self.value.to(dtype=torch.int).tolist()[0]

        

class ResMsg(object):
    # all request msg class will inherit this class
    def __init__(self, msgtype=None, key=None, value=None, status=None, src=None, dst=None):

        self.type=msgtype
        self.key=key
        self.value=value
        self.src=src
        self.dst=dst
        self.comm_tag={"reqres":0}
    
    def send(self):
        if self.value is None:
            pass
        else:
            bias=len(self.comm_tag)
            self.send_value=self.value
            dist.send(tensor=self.send_value,dst=self.dst,tag=self.key+bias)




class RoExReqMsg(ReqMsg):
    def __init__(self, key=None, value=None):
        self.msgtype="RoExReqMsg"
        super().__init__(self.msgtype, key, value)
        self.role_map={
            'masterworker':0,
            'worker':1,
            'server':2,
            'master':3
        }
    def encode(self):

        self.send_value = [[i if i==dist.get_rank() else 0,0] for i in range(dist.get_world_size())]
        self.send_value[dist.get_rank()][1]=self.role_map[self.value]
        # [[0,1],[1,0],[2,0]], e.g. rank 0 is the masterworker
        self.send_value = torch.tensor(self.send_value, dtype=torch.float)
        
    
    def send(self):
        self.encode()
        self.handle=dist.all_reduce(self.send_value,async_op=True)
    
    def wait(self):
        self.handle.wait()
        response = RoExResMsg(value=self.send_value)
        response.decode()
        return response

class RoExResMsg(ResMsg):
    def __init__(self, key=None, value=None):
        self.msgtype="RoExResMsg"
        super().__init__(self.msgtype, key, value)
        self.role_map={
            0:'masterworker',
            1:'worker',
            2:'server',
            3:'master'
        }
    def decode(self):
        self.value = self.value.type(torch.int).tolist()
        tmp = dict()
        for each in self.value:
            tmp[each[0]]=self.role_map[each[1]]
        self.value=tmp


class PushReqMsg(ReqMsg):
    def __init__(self,key,value,src,dst,ctx):
        super().__init__("PushReqMsg", key,value,src,dst,ctx)
        self.status="init"
    def encode(self):
        self.status="prepare"
        self.send_value=self.value.detach().clone().to("cpu")
    def send(self):
        self.status="send"
        self.handles=[]
        bias=len(self.comm_tag)
        
        handle2=dist.isend(tensor=self.send_value,dst=self.dst,tag=self.key+bias)
        handle1=dist.isend(tensor=torch.tensor([self.comm_code["pushreq"], self.key, self.key+bias],dtype=torch.float)\
                            ,dst=self.dst,tag=self.comm_tag["reqres"])

        self.handles.append(handle1)
        self.handles.append(handle2)
    def wait(self):
        for each in self.handles:
            each.wait()
        response = PushResMsg(key=self.key,src=self.dst,dst=self.src)
        response.status="ACK"
        return response
class PushResMsg(ResMsg):
    def __init__(self,key,src,dst):
        super().__init__("PushResMsg", key,src,dst)
        self.status=""


class PullReqMsg(ReqMsg):
    def __init__(self,key,version,src,dst,ctx):
        # value is the version
        super().__init__("PullReqMsg", key,src,dst,ctx)
        self.version=version
        self.status="init"
    def encode(self):
        self.status="prepare"
        self.send_value=torch.tensor([self.version],dtype=torch.float)
    def send(self):
        self.status="send"
        self.handles=[]
        bias=len(self.comm_tag)
        # send [1, key]
        handle1=dist.isend(tensor=torch.tensor([self.comm_code["pullreq"], self.key, self.key+bias],dtype=torch.float),\
                            dst=self.dst,tag=self.comm_tag["pullreqres"])
        handle2=dist.isend(tensor=self.send_value,dst=self.dst,tag=self.key+bias)
        self.handles.append(handle1)
        self.handles.append(handle2)
    def wait(self):
        for each in self.handles:
            each.wait()
        buffer=torch.randn(self.ctx.KVStore(self.key)[self.key].size())
        bias=len(self.comm_tag)
        dist.recv(tensor=buffer,src=self.dst,tag=self.key+bias)
        # put the recv value to the GPU
        buffer=buffer.to(self.ctx.KVStore(self.key)[self.key].device)
        response = PullResMsg(key=self.key, value=buffer,src=self.dst,dst=self.src)
        return response

class PullResMsg(ResMsg):
    def __init__(self,key,value,src,dst):
        super().__init__("PullResMsg", key,value,src,dst)
        self.status=""
