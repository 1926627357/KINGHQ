# I realize some practical tools in this file
# TOOL LIST:
#   Utils: init dist and get information related to the distributed environment
#   Log: record and process data for user
#   Data_processing: this object is used by Log for data processing
#   Bar: show the progress in our process
#   CSV: handle .csv file
#   Figure: draw experiment figure
#   Dice: generate a random number

from KINGHQ.utils.KVStore import KVStore
from KINGHQ.msg.msg import RoExReqMsg,RoExResMsg
import torch.distributed as dist
import torch
import json5
import os
import fcntl

# import sys
class Utils:
    def __init__(self):
        self.Communication_Tag={'request':0,'push_num':1,'push_key':2,'pull_num':3,'pull_key':4}
        self.response_map={'send value':0, 'blocking':1, 'no blocking':2}
        self.request_map={'pull_request':0, 'push_request':1}
        self.KVStore_min_key=len(self.Communication_Tag)
        self.KVStore=KVStore(self.KVStore_min_key)
        #register the communication tag firstly
        self.KVStore.register_new_key(self.Communication_Tag,name='Communication_Tag')
        self.KVStore.register_new_key(self.response_map,name='response_map')
        self.KVStore.register_new_key(self.request_map,name='request_map')
        self.strategy=None

        self.role_path="/home/v-haiqwa/Documents/KINGHQ/config/recv/"

    def init(self):
        dist.init_process_group(backend='mpi')
        self.world_size=dist.get_world_size()
        self.world_rank=dist.get_rank()
        if self.world_size == 1:
            # single machine situation
            self.worker_size=1
            self.worker_rank=0
            self.local_size=1
            self.local_rank=0
            self.local_worker_size=1
            self.local_worker_rank=0
        else:
            # multi-machine
            # note that the environment var is in a format of string
            self.local_size=int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
            self.local_rank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            for root_dir,_,filename in os.walk(self.role_path):
                with open(os.path.join(root_dir,filename[0]),"r+") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    lines = f.readlines()
                    line=lines.pop(0)
                    f.seek(0)
                    f.writelines(lines)
            self.role = line.replace('\n','')
            # print("I'm rank-{} and I'am the {}".format(self.rank,self.role))
            if self.world_rank==0:
                print("PHASE 2 GLOBALLY EXCHANGE INFORMATION")
            request=RoExReqMsg(value=self.role)
            request.send()
            response=request.wait()
            # e.g. {0: "master", 1:"server"}
            self.rank_role_map=response.value
            if self.world_rank==3:
                print("END")
                print("*"*30)
                print(self.rank_role_map)
            self.worker_size=0
            self.worker_rank=0
            for rank,role in self.rank_role_map.items():
                if role=="masterworker" or role=="worker":
                    self.worker_size+=1
                    if rank==self.world_rank:
                        self.worker_rank=self.worker_size-1
            
                
        
    

    def load_strategy(self, path):
        # load the strategy through the json file
        with open(path,"r") as load_f:
            self.strategy=json5.load(load_f)
        return self.strategy
        
    def barrier(self):
        dist.barrier()

    def get_KVStore(self):
        return self.KVStore

    def get_role_rank_map(self):
        pass

    def is_masterworker(self):
        # Determine if a node is the masterworker
        if(self.role=="masterworker"):
            return True
        else:
            return False

    # here are normal apis here
    def get_world_rank(self):
        return self.world_rank
    def get_world_size(self):
        return self.world_size
    def get_local_rank(self):
        return self.local_rank
    def get_local_size(self):
        return self.local_size
    def get_worker_size(self):
        return self.worker_size
    def get_worker_rank(self):
        return self.worker_rank
    def get_local_worker_size(self):
        return self.local_worker_size
    def get_local_worker_rank(self):
        return self.local_worker_rank

# record the experiment data
import pandas as pd
class Log(object):
    def __init__(self,title,Axis_title,path,step=1):
        # step: record the data every steps
        self.title=title
        self.columns=Axis_title if isinstance(Axis_title,list) else [Axis_title]
        self.record=[[] for _ in range(len(self.columns))]
        self.path=path
        self.step=step
        self.data_pro=Data_processing()

    def log(self,value):
        # here we record value in different lists according to their columns name
        # value: accuracy or loss, for example
        
        # record the first data point
        if not isinstance(value,list):
            value=[value]
        for i in range(len(value)):
            self.record[i].append(value[i])
        

    def data_processing(self,method,**kwargs):
        # to draw nice curves with data processing method
        getattr(self.data_pro,method)(**kwargs)
        
    def get_column_data(self,column):
        # we can get the data if we provide the corresponding column name or index
        if isinstance(column,str):
            index=self.columns.index(column)
        elif isinstance(column,int):
            index=column
        return self.record[index]
    
    def write(self):
        transpose=[]
        if self.step<=0:
            self.step=1
        for i in range(len(self.record)):
            self.record[i]=self.record[i][::self.step]
        for each in zip(*tuple(self.record)):
            transpose.append(list(each))
        dataframe=pd.DataFrame(columns=self.columns,data=transpose)
        dataframe.to_csv(self.path,index=False,sep=',')


class Data_processing(object):
    def __init__(self):
        pass

    def rolling_mean(self, data, cycle=6):
        # data must be a list
        # it can help us remove the variance in our data record
        for i in range(len(data)):
            data[i]=sum(data[i:i+cycle])/len(data[i:i+cycle])

    def interval(self, data):
        # everyone in the list minus with the first one
        start=data[0]
        for i in range(len(data)):
            data[i]-=start

from tqdm import tqdm
class Bar(object):
    # user can use this object to show progress
    def __init__(self,total,description):
        self.bar=tqdm(total=total)
        self.bar.set_description(description)
    def __call__(self):
        self.bar.update(1)


import matplotlib
import matplotlib.pyplot as plt
class CSV(object):
    # CSV is just a tool. It has no non-function member
    # it's used to handle csv file
    def __init__(self, path):
        self.data=pd.read_csv(path)
    
    def __call__(self, key):
        # key is the column name
        # type of key is str
        return self.data[key].values

class Figure(object):
    def __init__(self, title, xlabel, ylabel, path):
        self.title=title
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.path=path
        fig, ax = plt.subplots()
        self.fig=fig
        self.ax=ax
    def save(self, path=None):
        # set figure configurations
        # save the figure as file
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel,
                title=self.title)
        self.ax.grid()
        self.ax.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=0, mode="expand",
                    borderaxespad=0., ncol=100, fontsize='small')
        if path is None:
            self.fig.savefig(self.path)
        else:
            self.fig.savefig(path)
    def add(self, xdata, ydata,label):
        # add numpy data into the figure
        self.ax.plot(xdata, ydata, label=label)

import random
class Dice(object):
    # it's used to generate a integer randomly like a dice
    def __init__(self, side):
        # the dice will generate the number: 1,2,3,...,side with equal probability
        self.side=side
    def __call__(self):
        return random.randint(1,self.side)

    
    
