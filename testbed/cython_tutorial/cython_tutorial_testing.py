import cython_tutorial
from time import time

N = 100000000

t0 = time()
cython_tutorial.test_1(N)
print 'python: %0.3f'%(time() - t0)
t0 = time()
cython_tutorial.test_2(N)
print 'cython: %0.3f'%(time() - t0)

