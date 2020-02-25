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
        self.status="init"
    def recv_head(self):
        self.status="recv_head"
        self.handles=[]
        self.msg=torch.randn(3).to(dtype=torch.float)
        handle1=dist.irecv(tensor=self.msg,src=self.src,tag=self.comm_tag["reqres"])
        self.handles.append(handle1)
    def recv_value(self):
        self.status="recv_value"
        self.handles=[]
        self.msg=self.msg.to(torch.int).tolist()
        if self.msg[0]==self.comm_code["pushreq"]:
            self.type="PushReqMsg"
            self.key=self.msg[1]
            
            # print("server: I get a push request-key:{}".format(self.key))
            p=self.ctx.KVStore(self.key)[self.key]
            self.value = p.data.new(p.size())
            handle1 = dist.irecv(tensor=self.value,src=self.src,tag=self.msg[2])
        elif self.msg[0]==self.comm_code["pullreq"]:
            self.type="PullReqMsg"
            self.key=self.msg[1]
            # print("server: I get a pull request-key:{}".format(self.key))
            self.value = torch.tensor([0],dtype=torch.float)
            handle1 = dist.irecv(tensor=self.value,src=self.src,tag=self.msg[2])
        self.handles.append(handle1)
        
        # self.value=self.value.to(dtype=torch.int).tolist()[0]
    def is_completed(self):
        result=True
        for each in self.handles:
            result=result and each.is_completed()
        return result

    
        
        

class ResMsg(object):
    # all request msg class will inherit this class
    def __init__(self, msgtype=None, key=None, value=None, status=None, src=None, dst=None):

        self.type=msgtype
        self.key=key
        self.value=value
        self.src=src
        self.dst=dst
        self.comm_tag={"reqres":0}
        self.status="init"
    
    def send(self):
        self.status="send"
        self.handles=[]
        bias=len(self.comm_tag)
        self.send_value=self.value
        # print("server: I send a param-key:{} and size:{}".format(self.key,self.send_value.size()))
        handle1=dist.isend(tensor=self.send_value,dst=self.dst,tag=self.key+bias)
        self.handles.append(handle1)

    def is_completed(self):
        result=True
        for each in self.handles:
            result=result and each.is_completed()
        return result
            

class RoExReqMsg(ReqMsg):
    def __init__(self, key=None, value=None):
        self.msgtype="RoExReqMsg"
        
        super().__init__(msgtype=self.msgtype, key=key, value=value)
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
        
        super().__init__(msgtype=self.msgtype, key=key, value=value)
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
        
        super().__init__(msgtype="PushReqMsg", key=key,value=value,src=src,dst=dst,ctx=ctx)
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
    def get_response(self):
        # print("worker: I have sent a push request-key:{}".format(self.key))
        response = PushResMsg(key=self.key,src=self.dst,dst=self.src)
        response.status="ACK"
        return response
    def is_completed(self):
        result=True
        for each in self.handles:
            result=result and each.is_completed()
        # print(result)
        return result

class PushResMsg(ResMsg):
    def __init__(self,key,src,dst):
        
        super().__init__(msgtype="PushResMsg", key=key,src=src,dst=dst)
        self.status=""

class PullReqMsg(ReqMsg):
    def __init__(self,key,version,src,dst,ctx):
        # value is the version
        
        super().__init__(msgtype="PullReqMsg", key=key,src=src,dst=dst,ctx=ctx)
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
                            dst=self.dst,tag=self.comm_tag["reqres"])
        
        handle2=dist.isend(tensor=self.send_value,dst=self.dst,tag=self.key+bias)
        
        self.buffer=torch.randn(self.ctx.KVStore(self.key)[self.key].size())
        # print("worker: I begin to recv a param-key:{}".format(self.key))
        handle3=dist.irecv(tensor=self.buffer,src=self.dst,tag=self.key+bias)

        self.handles.append(handle1)
        self.handles.append(handle2)
        self.handles.append(handle3)
        
    def get_response(self):
        # put the recv value to the GPU
        # print("worker: I have recv a param-key:{} and size:{}".format(self.key,self.buffer.size()))
        self.buffer=self.buffer.to(self.ctx.KVStore(self.key)[self.key].device)
        response = PullResMsg(key=self.key, value=self.buffer,src=self.dst,dst=self.src)
        return response
    def is_completed(self):
        result=True
        for each in self.handles:
            result=result and each.is_completed()
        return result

class PullResMsg(ResMsg):
    def __init__(self,key,value,src,dst):
        
        super().__init__(msgtype="PullResMsg", key=key,value=value,src=src,dst=dst)
        self.status=""
