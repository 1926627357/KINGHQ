import queue
import time
import threading




def _loop(lock):
    lock.release()
    lock.acquire()
    print("I get the lock")
lock = threading.Lock()
thread = threading.Thread(target=_loop,args=(lock,))   # python识别元组

lock.acquire()
thread.start()

time.sleep(10)