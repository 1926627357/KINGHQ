# I realize some practical tools in this file
# TOOL LIST:
#   Utils: init dist and get information related to the distributed environment
#   Log: record and process data for user
#   Data_processing: this object is used by Log for data processing
#   Bar: show the progress in our process
#   CSV: handle .csv file
#   Figure: draw experiment figure

from KINGHQ.utils.KVStore import KVStore
import torch.distributed as dist
import torch
import json5
import os
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

    def init(self):
        if not ('MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ):
            # it illustrates that the user launch programme in single process
            
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '8889'
            os.environ['RANK']='0'
            os.environ['WORLD_SIZE']='1'
            os.environ['KINGHQ_Distributed']='False'
        else:
            os.environ['KINGHQ_Distributed']='True'
        
        
        dist.init_process_group(backend='gloo')
        self.rank=dist.get_rank()
        self.size=dist.get_world_size()
        if self.size<=1:
            os.environ['KINGHQ_Distributed']='False'
        

    def load_strategy(self, path):
        # load the strategy through the json file
        with open(path,"r") as load_f:
            self.strategy=json5.load(load_f)
        return self.strategy
        
    def barrier(self):
        dist.barrier()

    def get_KVStore(self):
        return self.KVStore

    def is_server(self):
        if self.rank==0:
            return True
        else:
            return False

    def get_rank(self):
        return self.rank

    def get_size(self):
        return self.size
    
    def get_worker_size(self):
        if os.environ['KINGHQ_Distributed']=='False':
            # single process
            return 1
        if self.strategy['network']['structure']['parameter server']:
            if self.is_server():
                # server
                return 1
            return self.size-1
        else:
            return self.size
    def get_worker_rank(self):
        if os.environ['KINGHQ_Distributed']=='False':
            # single process
            return 0
        if self.strategy['network']['structure']['parameter server']:
            if self.is_server():
                # server
                return 0
            return self.rank-1
        else:
            return self.rank

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

    


    
    
