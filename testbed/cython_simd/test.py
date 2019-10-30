import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int f(np.ndarray[int, ndim = 1] f):
    cdef int array_length =  f.shape[0]
    cdef int sum = 0
    cdef int k
    for k in range(array_length):
        sum += f[k]
    return sum
