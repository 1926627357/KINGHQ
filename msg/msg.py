class ReqMsg(object):
    # all request msg class will inherit this class
    def __init__(self, msgtype=None, key=None, value=None, src=None, dst=None):

        self.type=msgtype
        self.key=key
        self.value=value
        self.src=src
        self.dst=dst

    def encode(self):
        # encode the message to the value that the mpi can handle it
        pass

    def decode(self):
        # decode the message that is recv-ed
        pass

    def send(self):
        # send this msg to dst
        pass

    def wait(self):
        # wait for the apply 
        pass

class ResMsg(object):
    # all request msg class will inherit this class
    def __init__(self, msgtype=None, key=None, value=None, status=None, src=None, dst=None):

        self.type=msgtype
        self.key=key
        self.value=value
        self.src=src
        self.dst=dst

    def encode(self):
        # encode the message to the value that the mpi can handle it
        pass

    def decode(self):
        # decode the message that is recv-ed
        pass

    def send(self):
        # send this msg to dst
        pass

    def wait(self):
        # wait for the apply 
        pass
import torch
import torch.distributed as dist

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

        self.send_value = [[i,0] for i in range(dist.get_world_size())]
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


