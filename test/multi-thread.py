import queue
import time
import threading
import torch
import torch.distributed as dist
dist.init_process_group("mpi")
if dist.get_rank()==0:
    def recv():
        time.sleep(2)
        dist.recv(torch.tensor([0.]),tag=0)
        print("recv complete")
    def send():
        time.sleep(1)
        dist.isend(torch.tensor([0.]),dst=1,tag=1).wait()
        print("send complete!")
    # q = queue.Queue()
    
    thread1 = threading.Thread(target=recv)   # python识别元组
    thread1.start()
    thread0 = threading.Thread(target=send)
    thread0.start()

    
else:
    time.sleep(3)
    print("rank:{} send value".format(dist.get_rank()))
    dist.irecv(torch.tensor([0.]),src=0,tag=1).wait()
# time.sleep(1)
# q.put(1)
# # thread.join()   #会导致死锁，所以可以通过向queue中塞入None元素来结束该线程
# q.put(2)
# time.sleep(3)
print("main thread complete!")