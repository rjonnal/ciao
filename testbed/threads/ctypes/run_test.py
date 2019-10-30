import ctypes  
import ctypes.util  
import threading  
import time,sys

testlib = ctypes.cdll.LoadLibrary('./libtest.so')  

test = testlib.test  
test.argtypes = [ctypes.c_int, ctypes.c_int]  
  
def t():  
  test(0, 100000000)

if __name__ == '__main__':  
  start_time = time.time()  
  t()  
  t()  
  print "Sequential run time: %.2f seconds" % (time.time() - start_time)  
  
  start_time = time.time()
  t1 = threading.Thread(target=t)
  t2 = threading.Thread(target=t)
  t1.start()
  t2.start()
  print t1.join()
  print t2.join()
  print "Parallel run time: %.2f seconds" % (time.time() - start_time)
