import torch.distributed as dist
import torch
from KINGHQ.role import Role

class Worker(Role):
    def __init__(self, KVStore, matrix, optimizer, rank, strategy, device='cpu'):
        super().__init__(KVStore,optimizer,device)
        
        self.matrix=matrix
        # describe the synchronization strategy at the format of dictionary
        self.strategy=strategy['worker']


        # record iterations
        self.iterations=torch.tensor([0.])

        self.clock=torch.tensor([0.])
        #information of distribution
        self.rank=rank

        #Accumulation for push
        self.Accum_push={}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.Accum_push[p]=torch.zeros_like(p)
                

    def register_KVStore(self):
        # note that: here, we only register the key-value pairs that only worker
        # should know, while the server doesn't need to know that
        super().register_KVStore()
        self.KVStore.register_new_key(value=self.clock,name='clock')
        self.KVStore.register_new_key(value=self.iterations,name='iterations')
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.KVStore.register_new_key(value=self.Accum_push[p],name='Accum_push')

    def get_sources(self):
        sources=[]
        for index,value in enumerate(self.matrix[self.rank]):
            if value!=0 and index!=self.rank:
                sources.append(index)
        return sources

    def get_destinations(self):
        destinations=[]
        for index,value in enumerate(self.matrix[self.rank]):
            if value!=0 and index!=self.rank:
                destinations.append(index)
        return destinations

    def pull(self,source_keys):
        #source_key: a dictionary in a format of {source:keys}
        #            means that worker would recv the data of 
        #            key from the server
        # I think the key should be seen at the point of server
        # add pull(self,keys)
        buffer={}
        for source, keys in source_keys.items():
            pull_request=torch.tensor([self.request_map['pull_request']],dtype=torch.float32)
            pull_answer=torch.tensor([0.])
            buffer[source]={}
            dist.send(pull_request,dst=source,tag=self.Communication_Tag['request'])
            dist.recv(pull_answer,src=source,tag=self.Communication_Tag['request'])
            # add the answer from the server to the buffer
            buffer[source]['pull_answer']=pull_answer
            if pull_answer==self.response_map['no blocking']:
                #no blocking
                continue
            elif pull_answer==self.response_map['send value'] or \
                pull_answer==self.response_map['blocking']:

                keys_list=list(self.KVStore(keys).keys())
                
                # tell the server with the length and content of keys
                keys_len=torch.tensor([len(keys_list)],dtype=torch.float32)
                keys_tensor=torch.tensor(keys_list,dtype=torch.float32)
                dist.send(keys_len,dst=source,tag=self.Communication_Tag['pull_num'])
                dist.send(keys_tensor,dst=source,tag=self.Communication_Tag['pull_key'])
                for key,value in self.KVStore(keys).items():
                    buffer[source][key] = torch.randn(value.size())
                    dist.recv(buffer[source][key],src=source,tag=key)
        return buffer
    
    def handle_pull_request(self, buffer):
        #the data pulled are all filled in the buffer
        # modify ....
        Hot_key_reverse=self.KVStore.get_Hot_key_reverse()

        param_keys=self.KVStore.get_Hot_key()['params']
        weight={param_key:self.matrix[self.rank][self.rank] for param_key in param_keys}

        for source,recv in buffer.items():
            pull_answer=recv['pull_answer']
            if pull_answer==self.response_map['no blocking']:
                continue
            elif pull_answer==self.response_map['send value'] or \
                pull_answer==self.response_map['blocking']:

                for key,value in recv.items():
                    if not isinstance(key,int):
                        continue
                    
                    category=Hot_key_reverse[key]
                    if category=='global_version':
                        self.KVStore('version').popitem()[1].detach().zero_().add_(value)
                    elif category=='clock_slow':
                        self.KVStore(key).popitem()[1].detach().zero_().add_(value)
                    elif category=='params':
                        # I elaborately a weighted average algorithm here
                        _,p=self.KVStore(key).popitem()
                        base=weight[key]+self.matrix[self.rank][source]
                        
                        p.detach().zero_().add_(weight[key]*p/base\
                                + self.matrix[self.rank][source]*value/base)
                        
                        weight[key] += self.matrix[self.rank][source]
                        
    def push(self, destination_src_keys, destination_dst_keys):
        # destination_src_keys: {destination: keys}, send the values of the keys in the worker
        #                       to the specified destination
        #                       the keys is seen at the point of worker
        #                       it can also be seen as a buffer
        # destination_dst_keys: {destination: keys}, the key is seen at the point of worker
        for destination,src_keys in destination_src_keys.items():
            dst_keys=destination_dst_keys[destination]

            push_request=torch.tensor([self.request_map['push_request']],dtype=torch.float32)
            dist.send(push_request,dst=destination,tag=self.Communication_Tag['request'])

            keys_list=list(self.KVStore(dst_keys).keys())
            keys_len=torch.tensor([len(keys_list)],dtype=torch.float32)
            keys_tensor=torch.tensor(keys_list,dtype=torch.float32)

            dist.send(keys_len,dst=destination,tag=self.Communication_Tag['push_num'])
            dist.send(keys_tensor,dst=destination,tag=self.Communication_Tag['push_key'])

            # unify the src_keys and dst_keys into [int, int, int] format
            src_keys=self.KVStore.handle_list(src_keys)
            dst_keys=self.KVStore.handle_list(dst_keys)

            for src_key,dst_key in zip(src_keys,dst_keys):
                dist.send(self.KVStore(src_key).popitem()[1],dst=destination,tag=dst_key)

    def pull_(self, keys):
        # this api is used to pull the same keys from sources
        # a simple and fast api
        sources=self.get_sources()
        source_keys={source:keys for source in sources}
        self.handle_pull_request(self.pull(source_keys))
    
    def push_(self,src_keys,dst_keys):
        # this api is used to push the same keys to all destinations
        destinations=self.get_destinations()
        destination_src_keys={destination:src_keys for destination in destinations}
        destination_dst_keys={destination:dst_keys for destination in destinations}
        self.push(destination_src_keys,destination_dst_keys)
        



    def do_(self):
        # do jobs as the flow chart designs
        # push->apply->pull
        self.iterations+=1
        if self.strategy['push']['accumulate']['decision']:
            self.accumulate(src_keys='grads',dst_keys='Accum_push')
        if self.strategy['push']['action']['Interval']['decision']:
            if self.iterations%self.strategy['push']['action']['Interval']['interval']==0:
                if self.strategy['push']['action']['what']['solu1']['decision']:
                    self.push_(["version","grads"],["version","grads"])
                elif self.strategy['push']['action']['what']['solu2']['decision']:
                    self.push_(["version","Accum_push"],["version","grads"])
                    if self.strategy['push']['clear accumulate']['decision']:
                        self.zero_data('Accum_push')
                elif self.strategy['push']['action']['what']['solu3']['decision']:
                    self.push_(["Accum_push"],["grads"])
                    if self.strategy['push']['clear accumulate']['decision']:
                        self.zero_data('Accum_push')
                elif self.strategy['push']['action']['what']['solu4']['decision']:
                    self.push_(["grads"],["grads"])
                if self.strategy['push']['action']['update']['decision']:
                    self.clock+=1
        

        if self.strategy['apply']['accumulate']['decision']:
            self.accumulate(src_keys='grads',dst_keys='Accum_apply')
        if self.strategy['apply']['action']['Interval']['decision']:
            if self.iterations%self.strategy['apply']['action']['Interval']['interval']==0:
                if self.strategy['apply']['action']['what']['Accum_apply']:
                    self.replace(src_keys='Accum_apply', dst_keys='grads')
                elif self.strategy['apply']['action']['what']['grads']:
                    pass
                if self.strategy['apply']['clear accumulate']['decision']:
                    self.zero_data('Accum_apply')
                self.optimizer.step()
                if self.strategy['apply']['action']['update']['decision']:
                    self.update(self.strategy['apply']['action']['update']['content'])
        
        
        if self.strategy['pull']['when']['Interval']['decision']:
            if self.iterations%self.strategy['pull']['when']['Interval']['interval']==0:
                if self.strategy['pull']['what']['solu1']['decision']:
                    self.pull_(["params","clock_slow","global_version"])
                elif self.strategy['pull']['what']['solu2']['decision']:
                    self.pull_(["params","clock_slow"])
                    
                elif self.strategy['pull']['what']['solu3']['decision']:
                    self.pull_(["params","global_version"])
                elif self.strategy['pull']['what']['solu4']['decision']:
                    self.pull_(["params"])
        elif self.strategy['pull']['when']['staleness']['decision']:
            if self.clock-self.clock_slow\
                >self.strategy['pull']['when']['staleness']['staleness']:
                if self.strategy['pull']['what']['solu1']['decision']:
                    self.pull_(["params","clock_slow","global_version"])
                elif self.strategy['pull']['what']['solu2']['decision']:
                    self.pull_(["params","clock_slow"])
                    
                elif self.strategy['pull']['what']['solu3']['decision']:
                    self.pull_(["params","global_version"])
                elif self.strategy['pull']['what']['solu4']['decision']:
                    self.pull_(["params"])
        if self.strategy['barrier']['decision']:
            dist.barrier()

if __name__ == "__main__":
    pass