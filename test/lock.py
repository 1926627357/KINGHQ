import queue
import time
import threading




def _loop(lock):
    time.sleep(3)
    n=0
    # while True:
    #     n+=1
    #     # print("HELLLO!!!!!!!",n)
lock = threading.Lock()
thread = threading.Thread(target=_loop,args=(lock,))   # python识别元组
thread.start()
lock.acquire()
print("wo")
lock.acquire()
print("wooo")

time.sleep(10)