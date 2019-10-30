import centroid
import pp_centroid
from matplotlib import pyplot as plt
from kungpao.config import kungpao_config as kcfg
import glob,sys,os
import numpy as np
from time import time

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

ref = kcfg.REFERENCE_COORDINATES
refx_vec = np.round(ref[:,0]).astype(np.int16)
refy_vec = np.round(ref[:,1]).astype(np.int16)

def build_searchbox_edges():
    sb_width = 19#kcfg.SEARCH_BOX_WIDTH_PX
    x1_vec = refx_vec - sb_width//2
    x2_vec = refx_vec + sb_width//2
    y1_vec = refy_vec - sb_width//2
    y2_vec = refy_vec + sb_width//2
    return x1_vec,x2_vec,y1_vec,y2_vec


spots_images = glob.glob('/home/rjonnal/code/kungpao/data/spots/spots*.npy')
x1v,x2v,y1v,y2v = build_searchbox_edges()
xcentroids = np.zeros((len(x1v)),dtype=np.float)
ycentroids = np.zeros((len(x1v)),dtype=np.float)

plot_index = 0
xlim = refx_vec[plot_index]-10,refx_vec[plot_index]+10
ylim = refy_vec[plot_index]-10,refy_vec[plot_index]+10

for k in range(200):
    spots_index = k%len(spots_images)
    im = np.load(spots_images[spots_index])
    xcentroids,ycentroids = centroid.compute_centroids(im,x1v,x2v,y1v,y2v,xcentroids,ycentroids,True)
    plt.cla()
    plt.imshow(im,cmap='gray',clim=(0,1))
    plt.plot(xcentroids,ycentroids,'r+',markersize=12)
    plt.plot(refx_vec,refy_vec,'g+',markersize=8)
    #plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.pause(.00001)
    #assert (pp_xcentroids==xcentroids).all() and (pp_ycentroids==ycentroids).all()

