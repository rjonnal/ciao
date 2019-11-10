"""A series of tests of centroiding implementations, in order to 
compare their efficiencies."""

from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import time
import cProfile
import sys
import pymp
import ctypes

N_PROCESSORS = 4

# Load necessary data for all implementations; this overhead
# shouldn't be part of any timing
reference_coordinates = np.loadtxt('./reference_coordinates.txt')
spots_image = np.load('./spots.npy').astype(np.int16)
spots_sy,spots_sx = spots_image.shape

print('image size: %d, %d'%(spots_sy,spots_sx))

# split reference_coordinages into x and y; reasons for the data
# layout are obscure and not important; the next five lines can
# be ignored w/o losing any sense of the algorithm
sy,sx = reference_coordinates.shape
x_ref = reference_coordinates[:sy//2,:]
y_ref = reference_coordinates[sy//2:,:]
# the next two lines ravel the 2D arrays
x_ref = x_ref[np.where(x_ref>0)].astype(np.float)
y_ref = y_ref[np.where(y_ref>0)].astype(np.float)

# Now x_ref and y_ref are vectors containing the x and y
# reference coordinates--where the spots fall if the beam
# incident on the sensor has no aberrations.
#
# These coordinates will be used to define the "search boxes",
# regions of given width and height in which the center of mass
# is used to estimate the current spot position. The differences
# in x and y between the current spot position and the reference
# coordinates is a linear function of the x and y components of
# the local wavefront slope, which in turn are the partial
# derivatives of the wavefront height. The latter can be
# reconstructed using an appropriate set of basis vectors
# e.g. Zernike polynomials.
search_box_size = 39 # typical value we use in our AO system
# using the search box size, make vectors of offsets to loop
# through the image relative to the search box center coords
x_offsets = np.arange(search_box_size,dtype=np.int16)-search_box_size//2
y_offsets = np.arange(search_box_size,dtype=np.int16)-search_box_size//2

# count the required search boxes
n_search_boxes = len(x_ref)

# Initialize output vectors, to make sure this is not considered
# when timing implementations.
x_output = np.zeros(len(x_ref))
y_output = np.zeros(len(y_ref))

def plot_box(x,y):
    yr = int(round(y))
    xr = int(round(x))
    y1 = yr-search_box_size//2
    y2 = yr+search_box_size//2+1
    x1 = xr-search_box_size//2
    x2 = xr+search_box_size//2+1
    plt.autoscale(False)
    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-')
    plt.plot(x,y,'r.',markersize=1)
    
clim = np.percentile(spots_image,[50,99.99])
plt.figure(figsize=(10,10))
plt.axes([0,0,1,1])
plt.imshow(spots_image,cmap='gray',interpolation='none',clim=clim)
for x,y in zip(x_ref,y_ref):
    plot_box(x,y)
plt.savefig('spots_illustration.png',dpi=300)    


def naive_implementation():
    """This is a pure Python implementation with nothing fancy.
    Iterate through the search boxes and keep running totals of
    numerator (position*intensity) and denominator (intensity)."""
    for n,(x,y) in enumerate(zip(x_ref,y_ref)):
        x_numerator = 0.0
        y_numerator = 0.0
        denominator = 0.0
        # array indices have to be integers, so use rounded
        # reference coordinates as nominal search box centers:
        xr,yr = int(round(x)),int(round(y))
        
        for xoff in x_offsets:
            for yoff in y_offsets:
                xc,yc = xr+xoff,yr+yoff
                pixel = spots_image[yc,xc]
                x_numerator = x_numerator + xc*pixel
                y_numerator = y_numerator + yc*pixel
                denominator = denominator + pixel
                    
        x_output[n] = x_numerator/denominator - x
        y_output[n] = y_numerator/denominator - y

def vectorized_inner_implementation():
    """In this implementation the two inner loops (looping
    over the x and y cooridnates within each search box)
    have been vectorized using standard numpy 2D slicing,
    followed by calls to ndarray.ravel(). The x and y
    coordinates are represented using meshgrid, resulting
    in matrices XX and YY of the same shape as the spots
    image, but where each value is equal to its own the
    column and row index, respectively. Then, the center
    of mass in each dimension is computed using a single 
    vector computation:
    sum(xcoord * intensity)/sum(intensity):
    The outer loop over the search box indices is not
    parallelized.
    """
    sy,sx = spots_image.shape
    XX,YY = np.meshgrid(range(sx),range(sy))

    for n,(x,y) in enumerate(zip(x_ref,y_ref)):
        xr,yr = int(round(x)),int(round(y))
        y1 = yr-search_box_size//2
        y2 = yr+search_box_size//2+1
        x1 = xr-search_box_size//2
        x2 = xr+search_box_size//2+1

        x_vec = XX[y1:y2,x1:x2].ravel()
        y_vec = YY[y1:y2,x1:x2].ravel()
        spots_vec = spots_image[y1:y2,x1:x2].ravel()
        denominator = np.sum(spots_vec).astype(np.float)
        x_numerator = np.sum(x_vec*spots_vec).astype(np.float)
        y_numerator = np.sum(y_vec*spots_vec).astype(np.float)
        
        x_output[n] = x_numerator/denominator - x
        y_output[n] = y_numerator/denominator - y

def naive_pymp_implementation():
    """This is a pure Python implementation which uses pymp.
    Iterate through the search boxes and keep running totals of
    numerator (position*intensity) and denominator (intensity).
    The search box iteration is parallelized using pymp."""
    with pymp.Parallel(N_PROCESSORS) as p:
        for n in p.range(n_search_boxes):
            x = x_ref[n]
            y = y_ref[n]
            x_numerator = 0.0
            y_numerator = 0.0
            denominator = 0.0
            # array indices have to be integers, so use rounded
            # reference coordinates as nominal search box centers:
            xr,yr = int(round(x)),int(round(y))

            for xoff in x_offsets:
                for yoff in y_offsets:
                    xc,yc = xr+xoff,yr+yoff
                    pixel = spots_image[yc,xc]
                    x_numerator = x_numerator + xc*pixel
                    y_numerator = y_numerator + yc*pixel
                    denominator = denominator + pixel
            x_output[n] = x_numerator/denominator - x
            y_output[n] = y_numerator/denominator - y

def pymp_vectorized_inner_implementation():
    """In this implementation the two inner loops (looping
    over the x and y cooridnates within each search box)
    have been vectorized using standard numpy 2D slicing,
    followed by calls to ndarray.ravel(). The x and y
    coordinates are represented using meshgrid, resulting
    in matrices XX and YY of the same shape as the spots
    image, but where each value is equal to its own the
    column and row index, respectively. Then, the center
    of mass in each dimension is computed using a single 
    vector computation:
    sum(xcoord * intensity)/sum(intensity):
    In this version, the outer loop over the search box indices 
    is parallelized using pymp.
    """
    sy,sx = spots_image.shape
    XX,YY = np.meshgrid(range(sx),range(sy))

    with pymp.Parallel(N_PROCESSORS) as p:
        for n in p.range(n_search_boxes):
            x = x_ref[n]
            y = y_ref[n]
            xr,yr = int(round(x)),int(round(y))
            y1 = yr-search_box_size//2
            y2 = yr+search_box_size//2+1
            x1 = xr-search_box_size//2
            x2 = xr+search_box_size//2+1

            x_vec = XX[y1:y2,x1:x2].ravel()
            y_vec = YY[y1:y2,x1:x2].ravel()
            spots_vec = spots_image[y1:y2,x1:x2].ravel()
            denominator = np.sum(spots_vec).astype(np.float)
            x_numerator = np.sum(x_vec*spots_vec).astype(np.float)
            y_numerator = np.sum(y_vec*spots_vec).astype(np.float)
            x_output[n] = x_numerator/denominator - x
            y_output[n] = y_numerator/denominator - y

centroiding_library = ctypes.cdll.LoadLibrary('./centroiding.so')

ctypes_centroid_spots = centroiding_library.centroid_spots
ctypes_centroid_spots_omp = centroiding_library.centroid_spots_omp

argtypes = [np.ctypeslib.ndpointer(dtype = np.int16, ndim=2, shape=(spots_sy,spots_sx)),#spots_image
                                 ctypes.c_uint,#width
                                 np.ctypeslib.ndpointer(dtype = np.float, ndim=1, shape=(n_search_boxes)),#refx
                                 np.ctypeslib.ndpointer(dtype = np.float, ndim=1, shape=(n_search_boxes)),#refy
                                 ctypes.c_uint,#search_box_size
                                 ctypes.c_uint,#n_search_boxes
                                 np.ctypeslib.ndpointer(dtype = np.float, ndim=1, shape=(n_search_boxes)),#x_output
                                 np.ctypeslib.ndpointer(dtype = np.float, ndim=1, shape=(n_search_boxes))]#y_output

ctypes_centroid_spots.argtypes = argtypes
ctypes_centroid_spots_omp.argtypes = argtypes

def ctypes_implementation():
    ctypes_centroid_spots(spots_image, spots_sx, x_ref, y_ref, search_box_size, n_search_boxes, x_output, y_output)
    
def ctypes_omp_implementation():
    ctypes_centroid_spots_omp(spots_image, spots_sx, x_ref, y_ref, search_box_size, n_search_boxes, x_output, y_output)
            
def time_func(f,reps=30,t_base=None):
    times = []
    for k in range(reps):
        t0 = time.time()
        f()
        times.append(time.time()-t0)
    times_str = ', '.join(['%0.6f s'%t for t in times])
    mean_time = np.mean(times)
    if t_base is not None:
        suffix = ' %0.2fx speedup'%(t_base/mean_time)
    else:
        suffix = ''
    report = '%s times (%d runs): %s (average %0.6f s) %s'%(f.__name__,reps,times_str,mean_time,suffix)
    print(report)
    return mean_time,times

if __name__=='__main__':

    t_naive,times = time_func(naive_implementation)
    print()

    junk, times = time_func(vectorized_inner_implementation,t_base=t_naive)
    print()
    
    junk, times = time_func(naive_pymp_implementation,t_base=t_naive)
    print()
    
    junk, times = time_func(pymp_vectorized_inner_implementation,t_base=t_naive)
    print()

    junk, times = time_func(ctypes_implementation,t_base=t_naive)
    print()

    junk, times = time_func(ctypes_omp_implementation,t_base=t_naive)
    print()

