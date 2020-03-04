from multiprocessing import Process,Lock

class Test():
    def __init__(self):
        self.item=0
        self.p_list=[]
        self.lock=Lock()
        p1=Process(target=self._loop1,args=(self.lock,))
        p2=Process(target=self._loop2,args=(self.lock,))
        self.p_list.append(p1)
        self.p_list.append(p1)

        p1.start()
        p2.start()
        
    def _loop1(self,lock):
        for _ in range(10):
            lock.acquire()
            self.item+=1
            print("_loop1: ",self.item)
            lock.release()


    def _loop2(self,lock):
        for _ in range(10):
            lock.acquire()
            self.item+=1
            print("_loop2: ",self.item)
            lock.release()

t=Test()

for p in t.p_list:
    p.join()
print("complete!")

