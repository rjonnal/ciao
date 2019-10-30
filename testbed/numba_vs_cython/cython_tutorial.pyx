def test_1(x):
    y = 0
    for i in range(x):
        y += i
    return y


cpdef test_2(int x):
    cdef int y = 0
    cdef int i
    for i in range(x):
        y += i
    return y



