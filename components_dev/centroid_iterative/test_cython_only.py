import centroid
from matplotlib import pyplot as plt
import kungpao.config as kcfg
import glob,sys,os
import numpy as np
from time import time

def build_sensor_searchbox_mask():
    ref_fn = kcfg.reference_coordinates_filename
    ref = np.loadtxt(ref_fn)
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
    ref_fn = kcfg.reference_coordinates_filename
    ref = np.loadtxt(ref_fn)
    sb_width = kcfg.search_box_half_width*2+1
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
total_intensity = np.zeros((len(x1v)),dtype=np.float)
maximum_intensity = np.zeros((len(x1v)),dtype=np.float)
minimum_intensity = np.zeros((len(x1v)),dtype=np.float)
background_intensity = np.zeros((len(x1v)),dtype=np.float)

times = []
for fn in spots_images:
    print fn
    im = np.load(fn)
    t0 = time()
    xcentroids,ycentroids = centroid.compute_centroids(im,x1v,x2v,y1v,y2v,
                                                       xcentroids,
                                                       ycentroids,
                                                       total_intensity,
                                                       maximum_intensity,
                                                       minimum_intensity,
                                                       background_intensity)
    times.append(time()-t0)
    
    #assert (pp_xcentroids==xcentroids).all() and (pp_ycentroids==ycentroids).all()

cython_time = np.mean(times)

print 'cython: %0.4f'%cython_time

