import centroid
from matplotlib import pyplot as plt
import glob,sys,os
import numpy as np
from time import time
import centroid

dc = 10
spot_intensity = 100
image_width = 100
sb_width = 11
err_std = 1.0
iterations = 3
iteration_step_px = 2

image = np.ones((image_width,image_width),dtype=np.int16)*dc
sb_border = (sb_width-1)//2

sb_x = []
sb_y = []

for y in range(sb_border,image_width,sb_width+1):
    for x in range(sb_border,image_width,sb_width+1):
        sb_x.append(x)
        sb_y.append(y)

        dx = int(np.round(err_std*np.random.randn()))
        dy = int(np.round(err_std*np.random.randn()))

        image[y+dy,x+dx] = np.int16(spot_intensity)



sb_x = np.array(sb_x,dtype=np.int16)
sb_y = np.array(sb_y,dtype=np.int16)

x_out = np.zeros(sb_x.shape)
y_out = np.zeros(sb_x.shape)

mean_intensity = np.zeros(sb_x.shape)
maximum_intensity = np.zeros(sb_x.shape)
minimum_intensity = np.zeros(sb_x.shape)
background0 = np.zeros(sb_x.shape)
background = np.zeros(sb_x.shape)

sb_half_width = (sb_width-1)//2

centroid.estimate_backgrounds(image,sb_x,sb_y,background,sb_half_width)

centroid.compute_centroids(image,sb_x,sb_y,background,sb_half_width,iterations,iteration_step_px,x_out,y_out,mean_intensity,maximum_intensity,minimum_intensity,1)

plt.imshow(image,cmap='gray')
plt.autoscale(False)
plt.plot(sb_x,sb_y,'gs')
plt.plot(x_out,y_out,'r+')

plt.show()
