from ctypes import *#c_char_p, windll
import numpy as np
import scipy as sp
from time import time,sleep
import sys
from matplotlib import pyplot as plt
from numpy.ctypeslib import ndpointer



kcam = windll.LoadLibrary('./kungpao_camera')
system_name = c_char_p('M_SYSTEM_SOLIOS')
camera_filename = c_char_p('C:\\pyao_etc\\config\\dcf\\acA2040-180km-4tap-12bit_reloaded.dcf')
kcam.setup(system_name,camera_filename)

size_x = kcam.get_size_x()
size_y = kcam.get_size_y()
print size_x,size_y
im = np.zeros((size_y,size_x)).astype(np.uint16)
im_ptr = im.ctypes.data


if False:
    kcam.start()

    plt.figure()
    ph = plt.imshow(im)
    for k in range(10):
        kcam.get_current_image(im_ptr)
        plt.cla()
        plt.imshow(im)
        #ph.set_array(im)
        plt.pause(.001)

    kcam.stop()

class search_box(Structure):
    _fields_ = [("centroid_x", c_float),
                ("centroid_y", c_float),
                ("dc", c_ushort),
                ("box_max", c_ushort),
                ("box_min", c_ushort),
                ("box_total", c_float)]
    def __str__(self):
        out = ''
        for ff in self._fields_:
            out = out + '%s: %f\n'%(ff[0],eval('self.'+ff[0]))
        return out
                

                
if True:
    # test kcam.compute_centroid    
    im_type = ndpointer(dtype = np.uint16,ndim=2,flags='C');
    kcam.compute_centroid.argtypes = [c_void_p,c_void_p,c_float,c_float,c_ushort,
                                      POINTER(search_box),c_ushort,c_ushort,c_ushort]
    s = 21
    noise_max = 64
    im_in = np.round(np.random.rand(s,s)*noise_max).astype(np.uint16)
    im_out = np.round(np.random.rand(s,s)*noise_max).astype(np.uint16)
    f1 = (s-1)/2-2
    f2 = (s-1)/2+1
    fmid = (s-1)/2-1
    for k in range(f1,f2):
        for j in range(f1,f2):
            im_in[k,j] = 100
    im_in[fmid,fmid] = 200
    
    sb = search_box()
    kcam.compute_centroid(im_in.ctypes.data,im_out.ctypes.data,float(fmid+1),float(fmid+1),5,sb,0,s,s)
    plt.subplot(1,2,1)
    plt.imshow(im_in,interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(im_out,interpolation='none')
    plt.pause(1)
    print sb
kcam.release()
                                      
                                      
                                      
                                      