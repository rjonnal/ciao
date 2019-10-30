import ctypes  
import ctypes.util  
import threading  
import time  
import os,sys
import numpy as np


test = np.random.rand(100,100)*1024
test = np.round(test).astype(np.int16)
sy,sx = test.shape
n_lenslets = 36

#testname = ctypes.util.find_library('test')
testlib = ctypes.cdll.LoadLibrary(os.path.join('.','libtest.so'))  
centroid_region = testlib.centroid_region
centroid_region.argtypes = [np.ctypeslib.ndpointer(dtype = np.int16, ndim=2, shape=(sy,sx)),#image
                                ctypes.c_short,#image_width
                                ctypes.c_short,#image_height
                                ctypes.c_short,#x
                                ctypes.c_short,#region_width
                                ctypes.c_short,#y
                                ctypes.c_short,#region_height
                                ctypes.POINTER(ctypes.c_float),#x_center_of_mass
                                ctypes.POINTER(ctypes.c_float),#y_center_of_mass
                                #np.ctypeslib.ndpointer(dtype = np.float32, ndim=1, shape=(n_lenslets)),#x_center_of_mass
                                #np.ctypeslib.ndpointer(dtype = np.float32, ndim=1, shape=(n_lenslets)),#y_center_of_mass
                                ctypes.POINTER(ctypes.c_short),#region_max
                                ctypes.POINTER(ctypes.c_short)]#region_min
  
xcom = 0.0
ycom = 0.0
rmax = -1e10
rmin = 1e10

def t():
    centroid_region(test,100,100,20,10,20,10,xcom,ycom,rmax,rmin)


if __name__ == '__main__':
    N = 100
    start_time = time.time()
    for k in range(N):
        t()
        t()
    print "Sequential run time: %.2f seconds" % (time.time() - start_time)  
  
    start_time = time.time()
    threads = []
    for k in range(N):
        threads.append(threading.Thread(target=t))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
        
    # t1 = threading.Thread(target=t)  
    # t2 = threading.Thread(target=t)  
    # t1.start()  
    # t2.start()  
    # t1.join()  
    # t2.join()  
    print "Parallel run time: %.2f seconds" % (time.time() - start_time)
