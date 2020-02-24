# from KINGHQ.role.server import Server
import torch
class Role(object):
    def __init__(self, KVStore):
        self.KVStore=KVStore
        self.clock=0
        self.version=0
        self.stable_version=0
    def register_KVStore(self):
        self.KVStore.register_new_key(value=self.clock,name="clock")
        self.KVStore.register_new_key(value=self.version,name="version")
        self.KVStore.register_new_key(value=self.stable_version,name="stable_version")




