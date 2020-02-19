import queue
import time
import threading


def _loop(queue):
    while True:
        q = queue.get() # queue一定是线程安全的可以随便放心用
        print("I get the item %d"%q)
q = queue.Queue()
thread = threading.Thread(target=_loop,args=(q,))   # python识别元组
thread.start()

time.sleep(1)
q.put(1)
thread.join()   #会导致死锁，所以可以通过向queue中塞入None元素来结束该线程
q.put(2)
time.sleep(3)
print("main thread complete!")