# Approaches to multi-threading for centroid calculations

These are Python-specific approaches, but probably all generalizable. I'm using the term "shared object (SO)" interchangeably with "dynamically linked library". For our purposes, they're the same. Both of these are approaches for getting around the Python GIL, which makes threading the centroid computations inefficient.

1. One way around the GIL is through ctypes (see http://caswenson.com/2009_06_13_bypassing_the_python_gil_with_ctypes.html). In this approach we use a ctypes interface to an SO function `centroid_region`, say, which computes the center of mass on the SHWS image (array) over specified rows and columns. Python's `threading` module is used to initialize threads with `centroid_region` as a `target` function, and supply the relevant parameters via the `args` parameter.

    The key challenges here are 1) how to share memory (viz. the SHWS image) among the threads, since each one will be reading from different parts of the same array; and 2) how to retrieve the resulting centers of mass.
    
2. 
