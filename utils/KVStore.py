class KVStore(object):
    # store the communication data as (key,value) format.
    # In my design the push and pull(and many other operations) only need keys and src/dst
    # the worker and server need to have the same key-value pairs, or need to know the allocation
    # of each other
    def __init__(self,min_key):
        self.KVStore_min_key=min_key
        self.KVStore={}
        
        self.Hot_key={}
        #Hot_key_reverse={key:name}
        self.Hot_key_reverse={}

    def register_new_key(self,value,name=None):
        #register in KVStore and VKStore
        #if usr provide the name, he can set a hot key for that key
        self.KVStore[self.KVStore_min_key]=value
        if name is not None:
            self.Hot_key_reverse[self.KVStore_min_key]=name
            if name in self.Hot_key:
                self.Hot_key[name].append(self.KVStore_min_key)
            else:
                self.Hot_key[name]=[self.KVStore_min_key]
        self.KVStore_min_key=self.KVStore_min_key+1
        return self.KVStore_min_key-1

    def get_KVStore(self):
        # return the KVStore dictionary
        return self.KVStore
    
    def get_Hot_key(self):
        # return the Hot key dictionary
        return self.Hot_key
    
    def get_Hot_key_reverse(self):
        # 
        return self.Hot_key_reverse

    def handle_list(self,in_list):
        out_list=[]
        for each in in_list:
            if isinstance(each,int):
                out_list.append(each)
            elif isinstance(each,str):
                out_list+=self.Hot_key[each]
            elif isinstance(each,list):
                out_list+=self.handle_list(each)
        return out_list
    
    def __call__(self,obj):
        if isinstance(obj,int):
            return {obj:self.KVStore[obj]}
        elif isinstance(obj,list):
            return {i:self.KVStore[i] for i in self.handle_list(obj)}
        elif isinstance(obj,str):
            return {i:self.KVStore[i] for i in self.Hot_key[obj]}

if __name__ == "__main__":
    KVStore=KVStore(0)
    for i in range(10):
        KVStore.register_new_key(i,'a')
    for i in range(5):
        KVStore.register_new_key(i,'b')
    print(KVStore.get_KVStore())
    print(KVStore.get_Hot_key())
    print(KVStore(3))
    print(KVStore([0,1,2,3]))
    print(KVStore('a'))
    print(KVStore(['a','b']))