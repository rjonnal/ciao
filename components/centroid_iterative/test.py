import centroid
from matplotlib import pyplot as plt
from kungpao.config import kungpao_config as kcfg
import glob,sys,os
import numpy as np
from time import time

def pp_compute_centroids(spots_image,sb_x1_vec,sb_x2_vec,sb_y1_vec,sb_y2_vec):
    n_spots = len(sb_x1_vec)
    x_out = np.zeros(n_spots)
    y_out = np.zeros(n_spots)
    for k in range(n_spots):
        xprod = 0.0
        yprod = 0.0
        intensity = 0.0
        XX,YY = np.meshgrid(np.arange(sb_x1_vec[k],sb_x2_vec[k]+1),np.arange(sb_y1_vec[k],sb_y2_vec[k]+1))
        subim = spots_image[sb_y1_vec[k]:sb_y2_vec[k]+1,sb_x1_vec[k]:sb_x2_vec[k]+1].astype(np.float)
        x_out[k] = np.sum(XX*subim)/np.sum(subim)
        y_out[k] = np.sum(YY*subim)/np.sum(subim)
    return x_out,y_out
                
def build_sensor_searchbox_mask():
    ref = kcfg.REFERENCE_COORDINATES
    refx_vec = ref[:,0]
    refy_vec = ref[:,1]
    sb_width = kcfg.SEARCH_BOX_WIDTH_PX
    sy,sx = kcfg.SENSOR_HEIGHT_PX,kcfg.SENSOR_WIDTH_PX
    mask = np.ones((sy,sx))*(-1)
    n_ref = len(refx_vec)
    rad = sb_width//2
    
    for ref_index,(refx,refy) in enumerate(zip(refx_vec,refy_vec)):
        x = int(round(refx))
        y = int(round(refy))
        x1 = x-rad
        x2 = x+rad+1
        y1 = y-rad
        y2 = y+rad+1
        mask[y1:y2,x1:x2] = ref_index
    return mask

def build_searchbox_edges():
    ref = kcfg.REFERENCE_COORDINATES
    sb_width = kcfg.SEARCH_BOX_WIDTH_PX
    refx_vec = np.round(ref[:,0]).astype(np.int16)
    refy_vec = np.round(ref[:,1]).astype(np.int16)
    x1_vec = refx_vec - sb_width//2
    x2_vec = refx_vec + sb_width//2
    y1_vec = refy_vec - sb_width//2
    y2_vec = refy_vec + sb_width//2
    return x1_vec,x2_vec,y1_vec,y2_vec

spots_images = sorted(glob.glob('/home/rjonnal/code/kungpao/data/spots/spots*.npy'))
x1v,x2v,y1v,y2v = build_searchbox_edges()
xcentroids = np.zeros((len(x1v)),dtype=np.float)
ycentroids = np.zeros((len(x1v)),dtype=np.float)

times = []
pp_times = []
for fn in spots_images:
    print fn
    im = np.load(fn)
    t0 = time()
    xcentroids,ycentroids = centroid.compute_centroids(im,x1v,x2v,y1v,y2v,xcentroids,ycentroids,True)
    times.append(time()-t0)
    t0 = time()
    pp_xcentroids,pp_ycentroids = pp_compute_centroids(im,x1v,x2v,y1v,y2v)
    pp_times.append(time()-t0)
    print pp_xcentroids[0],xcentroids[0]
    
    #assert (pp_xcentroids==xcentroids).all() and (pp_ycentroids==ycentroids).all()

cython_time = np.mean(times)
pp_time = np.mean(pp_times)

print 'pure python: %0.4f'%pp_time
print 'cython: %0.4f'%cython_time
print 'speedup: %0.1f x'%(pp_time/cython_time)

