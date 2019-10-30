import numpy as np
cimport numpy as np
from matplotlib import pyplot as plt
import cython

ctypedef np.uint16_t uint16_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_centroids(np.ndarray[np.int16_t,ndim=2] spots_image,
                        np.ndarray[np.int16_t,ndim=1] sb_x1_vec,
                        np.ndarray[np.int16_t,ndim=1] sb_x2_vec,
                        np.ndarray[np.int16_t,ndim=1] sb_y1_vec,
                        np.ndarray[np.int16_t,ndim=1] sb_y2_vec):
    cdef np.int_t n_spots = len(sb_x1_vec)
    cdef np.ndarray x_out = np.zeros([n_spots],dtype=np.float)
    cdef np.ndarray y_out = np.zeros([n_spots],dtype=np.float)
    cdef np.int_t k
    cdef np.float_t intensity
    cdef np.float_t xprod
    cdef np.float_t yprod
    cdef np.int_t x
    cdef np.int_t y
    cdef np.float_t pixel
    for k in range(n_spots):
        intensity = 0.0
        xprod = 0.0
        yprod = 0.0
        for x in range(sb_x1_vec[k],sb_x2_vec[k]+1):
            for y in range(sb_y1_vec[k],sb_y2_vec[k]+1):
                pixel = spots_image[y,x]
                xprod = xprod + pixel*x
                yprod = yprod + pixel*y
                intensity = intensity + pixel
        x_out[k] = xprod/intensity
        y_out[k] = yprod/intensity
    return x_out,y_out
                
cpdef compute_centroids_inplace(np.ndarray[np.int16_t,ndim=2] spots_image,
                                np.ndarray[np.int16_t,ndim=1] sb_x1_vec,
                                np.ndarray[np.int16_t,ndim=1] sb_x2_vec,
                                np.ndarray[np.int16_t,ndim=1] sb_y1_vec,
                                np.ndarray[np.int16_t,ndim=1] sb_y2_vec,
                                np.ndarray[np.float_t,ndim=1] x_out,
                                np.ndarray[np.float_t,ndim=1] y_out):
    cdef np.int_t n_spots = len(sb_x1_vec)
    cdef np.int_t k
    cdef np.float_t intensity
    cdef np.float_t xprod
    cdef np.float_t yprod
    cdef np.int_t x
    cdef np.int_t y
    cdef np.float_t pixel
    for k in range(n_spots):
        intensity = 0.0
        xprod = 0.0
        yprod = 0.0
        for x in range(sb_x1_vec[k],sb_x2_vec[k]+1):
            for y in range(sb_y1_vec[k],sb_y2_vec[k]+1):
                pixel = spots_image[y,x]
                xprod = xprod + pixel*x
                yprod = yprod + pixel*y
                intensity = intensity + pixel
        x_out[k] = xprod/intensity
        y_out[k] = yprod/intensity
                
