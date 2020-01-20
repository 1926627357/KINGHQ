import torch.distributed as dist
import torch
from KINGHQ.role import Role
class Server(Role):
    def __init__(self, KVStore, optimizer, worker_rank,strategy,device='cpu'):
        # worker_rank: denote all workers' ranks in a format of list
        super().__init__(KVStore,optimizer,device)
        
        # map the request with int
        self.request_map={'pull_request':0,'push_request':1}

        self.strategy=strategy['server']
        self.worker_rank=worker_rank
        self.worker_clock={i:torch.tensor([0.]) for i in self.worker_rank}
        self.worker_version={i:torch.tensor([0.]) for i in self.worker_rank}

        # we use the waiting queue to store the rank of blocking worker
        # waiting queue=[(rank,request)]
        self.waiting_queue=[]
        self.is_in_queue=True

        self.iterations=0

    def register_KVStore(self):
        super().register_KVStore()
        # register version and clock for every worker
        for key,value in self.worker_clock.items():
            self.KVStore.register_new_key(value=value,name='worker%d_clock'%key)
        for key,value in self.worker_version.items():
            self.KVStore.register_new_key(value=value,name='worker%d_version'%key)

    def get_request(self, option='comm'):
        # option: 
        #           1. comm: get request from worker
        #           2. queue: get request from waiting queue
        # get a request
        # return the rank and the kind of request
        if option=='comm':
            request=torch.tensor([0.])
            rank=dist.recv(request,tag=self.Communication_Tag['request'])
            self.is_in_queue=False
        elif option=='queue' and self.waiting_queue:
            # choose the first one in the waiting queue
            rank,request=self.waiting_queue[0]
            self.is_in_queue=True
        else:
            # the waiting queue is empty
            return None,None
        return rank,request
    
    def handle_request(self,rank,request):
        if request==self.request_map['pull_request']:
            if self.strategy['check']['staleness']['decision']:
                if self.worker_clock[rank]-self.clock_slow\
                    <=self.strategy['check']['staleness']['staleness']:
                    pull_answer=torch.tensor([self.response_map['send value']],dtype=torch.float32)
                else:
                    
                    pull_answer=torch.tensor([self.response_map['blocking']],dtype=torch.float32)
            elif self.strategy['check']['version']['decision']:
                if self.worker_version[rank]>=self.KVStore('global_version'):
                    pull_answer=torch.tensor([self.response_map['no blocking']],dtype=torch.float32)
                else:
                    pull_answer=torch.tensor([self.response_map['send value']],dtype=torch.float32)
            else:
                pull_answer=torch.tensor([self.response_map['send value']],dtype=torch.float32)

            self.handle_pull(rank, pull_answer)
            if pull_answer==self.response_map['send value']:
                # if send value, we should set the worker's version to global version
                self.replace('global_version','worker%d_version'%rank)

        elif request==self.request_map['push_request']:
            self.iterations+=1

            self.handle_push(rank)

            
            self.replace('version','worker%d_version'%rank)
            # update the clock of the rank worker
            self.worker_clock[rank].add_(1)

            keys=['worker%d_clock'%key for key in self.worker_rank]
            index,_=self.min_max(keys,option='min')
            self.replace(keys[index],'clock_slow')
            

            if self.strategy['apply']['staleness']['decision']:
                stale=self.compute_staleness(rank)
                if stale>0:
                    self.replace('grads','grads',float(1/stale))
                    
            if self.strategy['apply']['accumulate']['decision']:
                self.accumulate(src_keys='grads',dst_keys='Accum_apply')
            if self.strategy['apply']['action']['Interval']['decision']:
                if self.iterations%self.strategy['apply']['action']['Interval']['interval']==0:
                    if self.strategy['apply']['action']['what']['Accum_apply']:
                        if self.strategy['apply']['action']['Interval']['average']:
                            self.replace(src_keys='Accum_apply', dst_keys='grads',\
                                factor=float(1/self.strategy['apply']['action']['Interval']['interval'])
                                    )
                        else:
                            self.replace(src_keys='Accum_apply', dst_keys='grads')
                    elif self.strategy['apply']['action']['what']['grads']:
                        pass
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.strategy['apply']['action']['update']['decision']:
                        self.update(self.strategy['apply']['action']['update']['content'])
            if self.strategy['apply']['clear accumulate']['decision']:
                self.zero_data('Accum_apply')

    def handle_push(self, rank):
        # recv the len of the keys
        keys_len=torch.tensor([0.])
        dist.recv(keys_len,src=rank,tag=self.Communication_Tag['push_num'])

        # change the data structure from tensor to int
        keys_len=keys_len.type(dtype=torch.int32).detach().numpy().tolist()[0]

        # recv the keys and change it to list
        keys=torch.tensor([0. for _ in range(keys_len)])
        dist.recv(keys,src=rank,tag=self.Communication_Tag['push_key'])
        keys=keys.type(dtype=torch.int32).detach().numpy().tolist()

        # classification according to the key
        targets=self.KVStore(keys)
        for key,value in targets.items():
            dist.recv(value,src=rank,tag=key)

    def compute_staleness(self, rank, usr_f=None, **kwargs):
        # usrs can customized their own computing staleness function 
        if usr_f is not None:
            usr_f(**kwargs)
        else:
            # the default staleness = global version - version
            staleness=(self.global_version-self.worker_version[rank])\
                        .type(dtype=torch.int32).detach().numpy().tolist()[0]
            return 0 if staleness<=0 else staleness

    def handle_pull(self, rank, pull_answer):
        if not self.is_in_queue:
            # only send the pull answer one time
            dist.send(pull_answer,dst=rank,tag=self.Communication_Tag['request'])
        if pull_answer==self.response_map['send value']:
            # send value
            # recv keys and its length
            # similar to the procedure of handle push
            keys_len=torch.tensor([0.])
            dist.recv(keys_len,src=rank,tag=self.Communication_Tag['pull_num'])
            keys_len=keys_len.type(dtype=torch.int32).detach().numpy().tolist()[0]
            keys=torch.tensor([0. for _ in range(keys_len)])
            dist.recv(keys,src=rank,tag=self.Communication_Tag['pull_key'])
            keys=keys.type(dtype=torch.int32).detach().numpy().tolist()

            targets=self.KVStore(keys)
            for key,value in targets.items():
                dist.send(value,dst=rank,tag=key)

            if self.is_in_queue:
                self.waiting_queue.pop(0)
            else:
                pass
        elif pull_answer==self.response_map['blocking']:
            # blocking
            if self.is_in_queue:
                pass
            else:
                self.waiting_queue.append((rank,\
                torch.tensor([self.request_map['pull_request']],dtype=torch.float32)))
        elif pull_answer==self.response_map['no blocking']:
            # no blocking
            pass
    
    def do_(self):
        if self.is_in_queue:
            rank,request=self.get_request()
        else:
            rank,request=self.get_request('queue')
            if rank==None:
                rank,request=self.get_request()
        self.handle_request(rank,request)
        


if __name__ == "__main__":
    pass