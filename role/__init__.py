# from KINGHQ.role.server import Server
import torch
class Role(object):
    def __init__(self, KVStore, optimizer, device='cpu'):
        self.KVStore=KVStore
        self.optimizer=optimizer
        
        self.device=device
        self.clock_slow=torch.tensor([0.])

        #get the map related to communication
        _,self.Communication_Tag=self.KVStore('Communication_Tag').popitem()
        _,self.response_map=self.KVStore('response_map').popitem()
        _,self.request_map=self.KVStore('request_map').popitem()
        
        #set grad of parameter to zero
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad=p.data.new(p.size()).zero_()

        #Accumulation for apply
        self.Accum_apply={}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.Accum_apply[p]=torch.zeros_like(p)

        # although only the server can handle the global version, we claim it here to make
        # the worker knowing the key of the global version
        self.global_version=torch.tensor([0.])

        # although only the worker would handle version, we claim it here to make
        # the server knowing the key of the version
        self.version=torch.tensor([0.])


    def register_KVStore(self):
        self.KVStore.register_new_key(value=self.version,name='version')
        self.KVStore.register_new_key(value=self.clock_slow,name='clock_slow')
        self.KVStore.register_new_key(value=self.global_version,name='global_version')
        self.KVStore.register_new_key(value=self.request_map,name='request_map')
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.KVStore.register_new_key(value=p,name='params')
                self.KVStore.register_new_key(value=p.grad,name='grads')
                self.KVStore.register_new_key(value=self.Accum_apply[p],name='Accum_apply')


    def min_max(self, keys, option='min'):
        # return the min or max value and its index
        if option=='min':
            value_list=list(self.KVStore(keys).values())
            value=min(value_list)
            index=value_list.index(value)
        elif option=='max':
            value_list=list(self.KVStore(keys).values())
            value=max(value_list)
            index=value_list.index(value)
        return index,value

    def accumulate(self, src_keys, dst_keys, factor=1):
        # accumlate the data according to the keys
        # include the optimizer.step()
        grads=list(self.KVStore(src_keys).values())
        accums=list(self.KVStore(dst_keys).values())
        for grad,accum in zip(grads,accums):
            accum.add_(grad*factor)

    def replace(self, src_keys, dst_keys, factor=1):
        # replace the dst with the src
        src=list(self.KVStore(src_keys).values())
        dst=list(self.KVStore(dst_keys).values())
        for i,j in zip(src,dst):
            j.detach().zero_().add_(i*factor)

    def zero_data(self, keys):
        # zero_accumulate, zero_grad ...
        for _,value in self.KVStore(keys).items():
            value.zero_()

    def update(self, key):
        #update the clock or the version
        _,value=self.KVStore(key).popitem()
        value.add_(1)
